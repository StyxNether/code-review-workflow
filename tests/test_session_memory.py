"""会话记忆：追加式落盘、载入降级、有效行与阈值压缩。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crw.session.memory import (
    COMPACTION_RATIO,
    MAX_MEMORY_TEXT_CHARS,
    MEMORY_AGENTS,
    AgentMemory,
    memory_file,
)

#: 固定时间戳（断言逐字段一致用）
TS = "2026-10-12T10:00:00+08:00"

#: 压缩摘要超限收尾标记（与实现约定一致）
EARLIER_OMITTED = "…（更早记录略）"


def make_memory(tmp_path: Path, agent: str = "encoder") -> AgentMemory:
    """在临时目录里建一个记忆（路径经 memory_file 生成，与真实会话目录同构）。"""
    return AgentMemory(memory_file(tmp_path / "s-20261012-0001", agent), agent=agent)


def append_rounds(memory: AgentMemory, rounds: range, *, pad: str = "长" * 120) -> None:
    """连写若干轮长文本对话（默认长度足以让折叠产生收益）。"""
    for round_index in rounds:
        memory.append_exchange(
            round_index=round_index,
            task=f"第{round_index}轮任务{pad}",
            outcome=f"第{round_index}轮结论{pad}",
            ts=TS,
        )


# ---- 路径与 agent 白名单 ----


def test_memory_agents_and_thresholds_are_pinned() -> None:
    """记忆归属的三个 Agent 与压缩阈值是固定口径，漂移即失败。"""
    assert MEMORY_AGENTS == ("encoder", "reviewer", "fixer")
    assert COMPACTION_RATIO == 0.7
    assert MAX_MEMORY_TEXT_CHARS == 2000


@pytest.mark.parametrize("agent", MEMORY_AGENTS)
def test_memory_file_maps_agent_to_memory_dir(tmp_path: Path, agent: str) -> None:
    assert memory_file(tmp_path, agent) == tmp_path / "memory" / f"{agent}.jsonl"


@pytest.mark.parametrize("agent", ["../../x", "..\\..\\x", "orchestrator", "", "Encoder", "a/b"])
def test_memory_file_rejects_unknown_agents(tmp_path: Path, agent: str) -> None:
    """白名单挡住路径穿越与未声明 Agent 名（同类白名单见 records.run_dir）。"""
    with pytest.raises(ValueError, match="agent 名非法"):
        memory_file(tmp_path, agent)


# ---- 追加与载入 ----


def test_append_exchange_round_trips_field_by_field(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)

    memory.append_exchange(round_index=1, task="写一个计算器", outcome="已生成 main.py", ts=TS)

    assert memory.lines() == [
        {"kind": "message", "round": 1, "role": "user", "text": "写一个计算器", "ts": TS},
        {"kind": "message", "round": 1, "role": "assistant", "text": "已生成 main.py", "ts": TS},
    ]
    text = memory.path.read_text(encoding="utf-8")
    assert len(text.splitlines()) == 2  # 一行一个 JSON 对象
    assert [json.loads(line) for line in text.splitlines()] == memory.lines()
    assert "写一个计算器" in text  # 中文不转义（前缀缓存友好）


def test_append_keeps_only_declared_fields(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)

    memory.append(
        [
            {
                "kind": "message",
                "round": 2,
                "role": "user",
                "text": "补充要求",
                "ts": TS,
                "extra": "不入库",
            }
        ]
    )

    assert memory.lines() == [
        {"kind": "message", "round": 2, "role": "user", "text": "补充要求", "ts": TS}
    ]


@pytest.mark.parametrize(
    ("record", "error"),
    [
        ({"kind": "note", "round": 1, "ts": TS}, ValueError),  # 未声明的 kind
        ({"kind": "message", "round": 1, "role": "system", "text": "x", "ts": TS}, ValueError),
        ({"kind": "message", "round": 1, "role": "user", "text": "x", "ts": ""}, ValueError),
        ({"kind": "message", "round": "1", "role": "user", "text": "x", "ts": TS}, TypeError),
        ({"kind": "message", "round": True, "role": "user", "text": "x", "ts": TS}, TypeError),
        ({"kind": "message", "round": 1, "role": "user", "text": 7, "ts": TS}, TypeError),
        (
            {"kind": "compaction", "round": 1, "ts": TS, "covers": "3", "summary": "s"},
            TypeError,
        ),
        ("不是映射", TypeError),
    ],
)
def test_append_rejects_undeclared_or_malformed_lines(
    tmp_path: Path, record: object, error: type[Exception]
) -> None:
    """形态非法行不落盘：记忆是所有后续请求的前缀，宁可在写入前报错。"""
    memory = make_memory(tmp_path)

    with pytest.raises(error):
        memory.append([record])
    assert not memory.path.exists()


# ---- 损坏行与缺失文件 ----


def test_lines_skips_broken_blank_and_non_object_lines(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)
    memory.path.parent.mkdir(parents=True)
    original = (
        '{"kind": "message", "round": 1, "role": "user", "text": "第一行", "ts": "t"}\n'
        "不是 JSON 的行\n"
        "\n"
        "[1, 2]\n"
        '"裸字符串"\n'
        "123\n"
        '{"kind": "message", "round": 1, "role": "assistant", "text": "末行", "ts": "t"}'
    )
    memory.path.write_text(original, encoding="utf-8")

    assert [line["text"] for line in memory.lines()] == ["第一行", "末行"]
    assert memory.path.read_text(encoding="utf-8") == original  # 只读，不修改磁盘


def test_missing_file_reads_as_empty_without_creating_it(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)

    assert memory.lines() == []
    assert memory.effective_lines() == []
    assert memory.prompt_history() == []
    assert memory.estimate_tokens() == 0
    assert not memory.path.exists()


# ---- 截断 ----


def test_append_truncates_overlong_text(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)

    memory.append(
        [{"kind": "message", "round": 1, "role": "assistant", "text": "长" * 5000, "ts": TS}]
    )

    (line,) = memory.lines()
    assert len(line["text"]) <= MAX_MEMORY_TEXT_CHARS
    assert line["text"].endswith("（已截断）")
    assert line["text"].startswith("长" * 100)  # 头部内容保留


def test_append_keeps_text_at_the_limit_untouched(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)
    exact = "正" * MAX_MEMORY_TEXT_CHARS

    memory.append([{"kind": "message", "round": 1, "role": "user", "text": exact, "ts": TS}])

    (line,) = memory.lines()
    assert line["text"] == exact


# ---- effective_lines 与 prompt_history ----


def test_effective_lines_and_prompt_history_before_compaction(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)
    append_rounds(memory, range(1, 4))

    assert len(memory.effective_lines()) == 6
    history = memory.prompt_history()
    assert [item["role"] for item in history] == ["user", "assistant"] * 3
    assert history[0]["text"].startswith("第1轮任务")
    assert history[1]["text"].startswith("第1轮结论")


def test_compaction_hides_old_lines_from_prompt_but_not_from_disk(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)
    append_rounds(memory, range(1, 9))
    before_text = memory.path.read_text(encoding="utf-8")

    line = memory.compact(round_index=9, ts=TS, keep_recent=4)

    assert line is not None and line["kind"] == "compaction"
    after_text = memory.path.read_text(encoding="utf-8")
    assert after_text.startswith(before_text)  # 只追加：旧行字节不变
    # 新增行 = compaction 行 + 保留的 4 条 role 行原文续写（"快照 + 尾行"）
    assert len(after_text.splitlines()) == len(before_text.splitlines()) + 1 + 4

    effective = memory.effective_lines()
    assert effective[0] == line
    assert [item["round"] for item in effective[1:]] == [7, 7, 8, 8]  # 保留最近 4 条 role 行
    assert "第1轮任务" not in json.dumps(effective, ensure_ascii=False)  # 旧行退出请求

    history = memory.prompt_history()
    assert history[0]["role"] == "user"
    assert history[0]["text"].startswith("（此前会话的压缩记录）")
    assert "第 1 轮：" in history[0]["text"]  # 折叠行按轮次进摘要
    assert "第1轮结论" in history[0]["text"]
    assert [item["role"] for item in history[1:]] == ["user", "assistant"] * 2  # 其后序列不变


# ---- needs_compaction ----


def test_needs_compaction_uses_ratio_of_budget(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)
    memory.append_exchange(round_index=1, task="甲" * 100, outcome="乙" * 100, ts=TS)

    assert memory.estimate_tokens() == 120  # 200 字符 × 0.6（与 agents/runtime.py 同口径）
    assert memory.needs_compaction(100) is True  # 阈值 70 ≤ 120
    assert memory.needs_compaction(1_000_000) is False
    assert memory.needs_compaction(120, ratio=1.0) is True  # 达到阈值即触发
    assert memory.needs_compaction(200, ratio=0.5) is True  # 阈值 100 ≤ 120
    assert memory.needs_compaction(200, ratio=0.7) is False  # 阈值 140 > 120


@pytest.mark.parametrize("budget", [0, -1, -100])
def test_needs_compaction_rejects_non_positive_budget(tmp_path: Path, budget: int) -> None:
    memory = make_memory(tmp_path)

    with pytest.raises(ValueError, match="预算"):
        memory.needs_compaction(budget)


# ---- compact ----


def test_compact_folds_oldest_rounds_and_reports_metadata(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)
    append_rounds(memory, range(1, 9))

    line = memory.compact(round_index=99, ts=TS, keep_recent=4)

    assert line is not None
    assert line["kind"] == "compaction"
    assert line["round"] == 6  # 被折叠行中的最大轮次（第 1–6 轮）
    assert line["covers"] == 12  # 折叠 12 条 role 行
    assert line["ts"] == TS
    assert line["summary"].startswith("第 1 轮：")
    assert "第 6 轮：" in line["summary"]
    # 16 条 role 行 + compaction 行 + 4 条保留行原文续写；压缩行紧挨保留行之前
    assert len(memory.lines()) == 16 + 1 + 4
    assert memory.lines()[16] == line
    assert memory.effective_lines()[0] == line


def test_compact_returns_none_without_enough_foldable_lines(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)
    append_rounds(memory, range(1, 3))  # 4 条 role 行 = keep_recent 缺省值
    before = memory.path.read_text(encoding="utf-8")

    assert memory.compact(round_index=3, ts=TS) is None
    assert memory.path.read_text(encoding="utf-8") == before


def test_compact_returns_none_when_no_benefit(tmp_path: Path) -> None:
    """折叠后（摘要 + 保留行）不比现状小：不写盘，避免"越压越大"的空转。"""
    memory = make_memory(tmp_path)
    memory.append_exchange(round_index=1, task="甲", outcome="乙", ts=TS)
    memory.append_exchange(round_index=2, task="甲", outcome="乙", ts=TS)
    memory.append(
        [
            {"kind": "message", "round": 2, "role": "assistant", "text": "丙", "ts": TS},
            {"kind": "message", "round": 2, "role": "assistant", "text": "丁", "ts": TS},
        ]
    )
    before = memory.path.read_text(encoding="utf-8")

    assert memory.compact(round_index=3, ts=TS, keep_recent=2) is None
    assert memory.path.read_text(encoding="utf-8") == before  # 行数与字节都不变


def test_compact_rejects_negative_keep_recent(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)

    with pytest.raises(ValueError, match="keep_recent"):
        memory.compact(round_index=1, ts=TS, keep_recent=-1)


def test_second_compact_folds_again_only_with_new_foldable_lines(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)
    append_rounds(memory, range(1, 9))

    first = memory.compact(round_index=9, ts=TS, keep_recent=2)
    assert first is not None and first["covers"] == 14
    count = len(memory.lines())
    assert memory.compact(round_index=9, ts=TS, keep_recent=2) is None  # 保留行不足：无新收益
    assert len(memory.lines()) == count

    append_rounds(memory, range(9, 13))
    second = memory.compact(round_index=13, ts=TS, keep_recent=2)

    assert second is not None
    assert second["covers"] == 8  # 只折叠首次压缩后新增的行（含首轮保留下来的 2 条）
    assert second["round"] == 11
    assert memory.effective_lines()[0] == second  # 最后一次压缩生效
    assert first in memory.lines()  # 旧压缩行留在盘上、不再参与请求


def test_compaction_summary_joins_same_round_points(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)
    pad = "长" * 150
    memory.append_exchange(round_index=1, task=f"任务一{pad}", outcome=f"结论一{pad}", ts=TS)
    memory.append(
        [
            {
                "kind": "message",
                "round": 1,
                "role": "assistant",
                "text": f"补充结论{pad}",
                "ts": TS,
            },
            {"kind": "message", "round": 2, "role": "user", "text": f"任务二{pad}", "ts": TS},
        ]
    )
    memory.append_exchange(round_index=3, task=f"任务三{pad}", outcome=f"结论三{pad}", ts=TS)

    line = memory.compact(round_index=4, ts=TS, keep_recent=2)

    assert line is not None
    assert line["covers"] == 4  # 折叠第 1 轮的 3 条与第 2 轮的 1 条
    assert line["round"] == 2
    summary = line["summary"]
    assert summary.startswith("第 1 轮：")
    assert "；" in summary  # 同轮多条要点连接
    assert "结论一" in summary and "补充结论" in summary
    assert "任务二" in summary  # 只有 user 行的轮次也要留下要点


def test_compaction_summary_caps_and_marks_omitted_earlier_rounds(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)
    for round_index in range(1, 10):
        memory.append_exchange(
            round_index=round_index,
            task=f"第{round_index}轮任务",
            outcome=f"第{round_index}轮结论" + "长" * 400,
            ts=TS,
        )

    line = memory.compact(round_index=10, ts=TS, keep_recent=1)

    assert line is not None and line["covers"] == 17
    summary = line["summary"]
    assert summary.endswith(EARLIER_OMITTED)
    assert len(summary) <= 1500 + len(EARLIER_OMITTED)
    assert "第 9 轮：" in summary  # 较新的折叠轮次保留
    assert "第 1 轮：" not in summary  # 更早的轮次让位


def test_second_compaction_chains_the_previous_summary(tmp_path: Path) -> None:
    """二次压缩把上一次的摘要接在本次之前：被折叠的结论不因"又压了一次"整段消失。"""
    store = AgentMemory(tmp_path / "memory" / "reviewer.jsonl", agent="reviewer")
    for index in range(1, 7):
        store.append_exchange(
            round_index=index,
            task=f"第 {index} 轮：审查当前版本",
            outcome=f"第 {index} 轮结论：发现问题 {index} 项，已如实记录。",
            ts="2026-10-12T10:00:00+08:00",
        )
    first = store.compact(round_index=6, ts="T1", keep_recent=2)
    assert first is not None
    first_summary = str(first["summary"])

    # 再追加两轮 × 让压缩有效益，然后二次压缩
    for index in range(7, 11):
        store.append_exchange(
            round_index=index,
            task=f"第 {index} 轮：审查当前版本",
            outcome=f"第 {index} 轮结论：发现问题 {index} 项，已如实记录。",
            ts="2026-10-12T10:00:00+08:00",
        )
    second = store.compact(round_index=10, ts="T2", keep_recent=2)

    assert second is not None
    second_summary = str(second["summary"])
    # 摘要链：上一次压缩的摘要接在本次之前（第 1 轮来自第一次压缩），不会整段消失
    assert "第 1 轮" in second_summary
    assert first_summary.split("；")[0] in second_summary
    assert "第 5 轮" in second_summary  # 本次折叠的轮次
    history = store.prompt_history()
    assert history[0]["text"].startswith("（此前会话的压缩记录）")
    assert "第 1 轮" in history[0]["text"]
    # keep_recent=2：最近两轮以原文行参与请求，不进摘要
    assert any("第 10 轮" in item["text"] for item in history[1:])
    assert "第 10 轮" not in second_summary


def test_compaction_skips_malformed_lines_when_rewriting(tmp_path: Path) -> None:
    """形态不完整的手改行不参与续写（磁盘留档仍在），压缩不因它抛异常。"""
    path_file = tmp_path / "memory" / "fixer.jsonl"
    path_file.parent.mkdir(parents=True)
    store = AgentMemory(path_file, agent="fixer")
    for index in range(1, 5):
        store.append_exchange(
            round_index=index,
            task=f"第 {index} 轮：修复审查问题",
            outcome=f"第 {index} 轮（code_v{index}/）：修复 1 项",
            ts="2026-10-12T10:00:00+08:00",
        )
    with path_file.open("a", encoding="utf-8") as handle:
        handle.write('{"kind": "message", "role": "assistant", "text": "缺 round 与 ts"}\n')

    line = store.compact(round_index=5, ts="T", keep_recent=1)

    assert line is not None
    texts = [item["text"] for item in store.prompt_history()]
    assert not any("缺 round 与 ts" in text for text in texts)  # 手改行不参与请求
    raw = path_file.read_text(encoding="utf-8")
    assert "缺 round 与 ts" in raw  # 但仍在磁盘上
