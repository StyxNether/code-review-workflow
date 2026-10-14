"""会话模式端到端（``session/chat.py``）用例：假模型驱动真实运行时 + 真实引擎。

覆盖：意图四分支、up-to / exact 语义、轮次夹取、记忆追加与压缩触发、中断落盘 interrupted、
``--resume`` 恢复、会话目录与运行记录同构、跨轮代码版本连续递增、session.md 与退出码。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from fakes import (
    ScriptedChatModel,
    ai_text,
    ai_tool_call,
    clean_review_calls,
    dirty_review_calls,
    file_fence,
    fix_fence,
    orchestrator_reply,
    review_fence,
)

from crw import config as config_mod
from crw.session import chat as chat_mod
from crw.session import memory as memory_mod
from crw.session import state as session_state
from crw.workflow import records as records_mod
from crw.workflow import state as state_mod

FIXED_TIME = "2026-10-12T10:00:00+08:00"

CLEAN_CODE = '''"""计算器（演示用）。"""


def add(left: int, right: int) -> int:
    """相加。"""
    return left + right


if __name__ == "__main__":
    print(add(1, 2))
'''
FIXED_CODE = CLEAN_CODE.replace("return left + right", "return left + right  # 已修复")


class _Session:
    """一次会话测试的驱动器：脚本化输入、收集输出、可选续跑（同一模型实例复用）。"""

    def __init__(
        self,
        model: ScriptedChatModel,
        tmp_path: Path,
        *,
        resolved: config_mod.ResolvedModel | None,
        run_id_factory: Callable[[], str],
    ) -> None:
        self.model = model
        self.tmp_path = tmp_path
        self.resolved = resolved
        self.run_id_factory = run_id_factory
        self.out: list[str] = []
        self.err: list[str] = []

    def io(self, messages: Sequence[str]) -> chat_mod.ChatIO:
        queue: list[str | None] = [*messages, None]
        cursor = {"value": 0}

        def ask(prompt: str) -> str | None:
            if cursor["value"] >= len(queue):
                return None
            value = queue[cursor["value"]]
            cursor["value"] += 1
            return value

        io = chat_mod.ChatIO(
            ask=ask,
            say=self.out.append,
            warn=self.err.append,
            renderer=_SilentRenderer,
        )
        return io

    def run(
        self,
        messages: Sequence[str],
        *,
        file_input: tuple[Path, str] | None = None,
        resume: str | None = None,
        overrides: chat_mod.RoundOverrides | None = None,
        initial: chat_mod.RoundSettings | None = None,
        rounds_limit: int | None = None,
        severe_only: bool = False,
    ) -> tuple[int, str]:
        """驱动一次会话；返回 ``(退出码, stdout 汇总)``。"""
        io = self.io(messages)
        code = chat_mod.run_chat(
            io=io,
            resolved=self.resolved,
            clock=lambda: FIXED_TIME,
            new_session_id_factory=lambda: "s-20261012-0001",
            run_id_factory=self.run_id_factory,
            rounds_limit=rounds_limit,
            initial_rounds=initial
            or chat_mod.RoundSettings(
                max_iterations=config_mod.DEFAULT_ROUNDS_DEFAULT,
                max_iterations_source=state_mod.SOURCE_DEFAULT,
                rounds_mode=state_mod.ROUNDS_UP_TO,
            ),
            overrides=overrides,
            file_input=file_input,
            resume_session_id=resume,
            severe_only=severe_only,
        )
        return code, "\n".join(self.out)


class _SilentRenderer:
    """无输出的观察者（会话用例只断言 stdout / 会话档案，不重复断言渲染细节）。"""

    def on_event(self, event: dict[str, Any]) -> None:
        """忽略事件。"""

    def on_delta(self, node: str, text: str) -> None:
        """忽略增量。"""

    def on_review(self, **kwargs: Any) -> None:
        """忽略审查结论。"""

    def on_fixes(self, **kwargs: Any) -> None:
        """忽略修复处置。"""


@pytest.fixture
def session(tmp_path: Path, fake_chat_model: ScriptedChatModel) -> _Session:
    """会话驱动器：模型连接走真实配置层（占位 key），客户端由假模型顶替。"""
    return _Session(fake_chat_model, tmp_path, resolved=_resolved_model(), run_id_factory=_ids())


def _resolved_model() -> config_mod.ResolvedModel:
    """真实配置层解析出的连接（占位 key；模型客户端被假模型顶替，无网络）。"""
    config_mod.save_raw(
        {
            "default": "DeepSeek/deepseek-flash",
            "providers": [
                {
                    "name": "DeepSeek",
                    "base_url": "https://api.example.invalid/v1",
                    "models": ["deepseek-flash"],
                    "api_key": "placeholder-key-for-tests",
                }
            ],
        }
    )
    resolved = config_mod.resolve()
    assert resolved is not None
    return resolved


def _ids() -> Callable[[], str]:
    counter = {"value": 0}

    def factory() -> str:
        counter["value"] += 1
        return f"r-20261012-{counter['value']:04x}"

    return factory


def _session_dir() -> Path:
    return session_state.sessions_root() / "s-20261012-0001"


def _round_state(index: int) -> dict[str, Any]:
    state = records_mod.read_state(_session_dir() / "rounds" / f"round-{index}")
    assert state is not None
    return state


def _archive() -> dict[str, Any]:
    data = session_state.read_session(_session_dir())
    assert data is not None
    return data


# ---- 意图分支与基本流程 ----


def test_code_intent_runs_encode_then_review_and_reports(
    session: _Session, fake_chat_model: ScriptedChatModel
) -> None:
    """第一轮：新需求 → 编码 → 首审干净（up-to 早退）→ 汇报；退出码 0。"""
    fake_chat_model.load(
        [
            orchestrator_reply("code", reply="好的，我来实现。"),
            ai_text(file_fence(CLEAN_CODE)),
            *clean_review_calls(),
        ]
    )

    code, out = session.run(["写一个加法计算器"])

    assert code == 0
    assert "[编排] 好的，我来实现。" in out
    assert "按「写/改代码」处理" in out
    assert "结论：干净通过（共 1 轮审查 / 0 轮修复）" in out
    assert "建议动作：已完成" in out
    archive = _archive()
    assert archive["status"] == session_state.SESSION_STATUS_ENDED
    assert archive["current_version"] == 1
    assert [item["intent"] for item in archive["rounds"]] == ["code"]
    assert archive["rounds"][0]["status"] == "success"
    state = _round_state(1)
    assert state["session_id"] == "s-20261012-0001"
    assert state["round_index"] == 1
    assert state["version_base"] == 1
    assert state["max_iterations"] == 3
    assert state["max_iterations_source"] == state_mod.SOURCE_DEFAULT


def test_ask_intent_does_not_start_the_workflow(
    session: _Session, fake_chat_model: ScriptedChatModel
) -> None:
    """``ask`` 不启动工作流、不产生运行记录；回复照给。"""
    fake_chat_model.load([orchestrator_reply("ask", reply="你想改哪一部分？")])

    code, out = session.run(["这个项目怎么样？"])

    assert code == 0
    assert "[编排] 你想改哪一部分？" in out
    assert not (_session_dir() / "rounds" / "round-1").exists()
    assert _archive()["rounds"] == []


def test_review_intent_copies_the_current_version_and_continues_numbering(
    session: _Session, fake_chat_model: ScriptedChatModel
) -> None:
    """第二轮"只审查"：把上一版本复制为本轮起始版本（号连续递增），审查后可继续下一轮。"""
    fake_chat_model.load(
        [
            orchestrator_reply("code"),
            ai_text(file_fence(CLEAN_CODE)),
            *clean_review_calls(),
            orchestrator_reply("review", reply="好，我只看不改。"),
            *clean_review_calls(version=2),
        ]
    )

    code, out = session.run(["写一个加法计算器", "再看一遍有没有问题"])

    assert code == 0
    assert "按「只审查」处理" in out
    assert "code_v2/（1 个文件）" in out
    second = _round_state(2)
    assert second["version_base"] == 2
    assert second["input"]["kind"] == state_mod.INPUT_FILE
    assert "code_v1" in str(second["input"]["source_path"])  # 来源是上一轮的版本目录
    review = json.loads(
        (_session_dir() / "rounds" / "round-2" / "workspace" / "review_1.json").read_text(
            encoding="utf-8"
        )
    )
    assert review["code_version"] == 2
    assert _archive()["current_version"] == 2


def test_review_fix_intent_runs_the_loop_and_advances_versions(
    session: _Session, fake_chat_model: ScriptedChatModel
) -> None:
    """第二轮"再检查一轮"：材料化 code_v2 → 首审不干净 → 修复产出 code_v3 → 复审干净。"""
    fake_chat_model.load(
        [
            orchestrator_reply("code"),
            ai_text(file_fence(CLEAN_CODE)),
            *clean_review_calls(),
            orchestrator_reply("review_fix", reply="好，再检查一轮。"),
            *dirty_review_calls(version=2),
            ai_text(
                fix_fence(
                    {"main.py": FIXED_CODE},
                    [{"finding_id": "f-1", "status": "fixed", "note": "已补除数校验。"}],
                )
            ),
            *clean_review_calls(version=3),
        ]
    )

    code, out = session.run(["写一个加法计算器", "再检查一轮看看"])

    assert code == 0
    assert "按「再走一轮审查→修复」处理" in out
    workspace = _session_dir() / "rounds" / "round-2" / "workspace"
    assert sorted(item.name for item in workspace.iterdir()) == [
        "code_v2",
        "code_v3",
        "fixes_1.json",
        "review_1.json",
        "review_2.json",
    ]
    assert (workspace / "code_v3" / "manifest.json").is_file()
    assert "已修复" in (workspace / "code_v3" / "main.py").read_text(encoding="utf-8")
    assert _archive()["current_version"] == 3


def test_file_input_session_modifies_the_existing_code(
    session: _Session, fake_chat_model: ScriptedChatModel, tmp_path: Path
) -> None:
    """``-f`` 起始 + 改需求：既有实现内联进编码提示词，编码从头产出第一个版本。"""
    source = tmp_path / "calc.py"
    source.write_text(CLEAN_CODE, encoding="utf-8")
    fake_chat_model.load(
        [
            orchestrator_reply("code", reply="好的，我在现有实现上改。"),
            ai_text(file_fence(FIXED_CODE, path="calc.py")),
            ai_tool_call("read_file", path="code_v1/calc.py"),
            ai_text(review_fence([], "复审通过。", files=["calc.py"])),
        ]
    )

    code, out = session.run(["把它改成支持减法"], file_input=(source, CLEAN_CODE))

    assert code == 0
    state = _round_state(1)
    assert state["version_base"] == 1
    assert state["version_files"] == ["calc.py"]
    assert state["input"]["kind"] == state_mod.INPUT_REQUIREMENT
    assert "calc.py" in str(state["input"]["source_path"])
    # 既有实现被内联进编码任务（编码节点无工具，读不到工作区），来源如实写成"来自文件 X"
    assert any(
        "## 既有实现（来自文件 calc.py，请在此基础上修改，共 1 个文件）" in str(item.content)
        for request in fake_chat_model.requests
        for item in request
    )
    assert any(
        "def add(left: int, right: int) -> int:" in str(item.content)
        for request in fake_chat_model.requests
        for item in request
    )
    assert "第 1 轮" in out


def test_review_intent_without_any_code_is_refused(
    session: _Session, fake_chat_model: ScriptedChatModel
) -> None:
    """还没有代码时的 review 意图：不静默挑路，给可读提示且不产生运行记录。"""
    fake_chat_model.load([orchestrator_reply("review")])

    code, _out = session.run(["审查一下"])

    assert code == 0
    assert any("还没有代码版本" in line for line in session.err)
    assert not (_session_dir() / "rounds" / "round-1").exists()


# ---- 轮次语义与优先级 ----


def test_exact_mode_runs_the_full_rounds_even_when_clean(
    session: _Session, fake_chat_model: ScriptedChatModel
) -> None:
    """exact：首审干净也继续跑满 n 轮修复 + n+1 次审查；结尾恒为一次审查（Design §4.4）。"""
    fake_chat_model.load(
        [
            orchestrator_reply("code", rounds=1, rounds_mode="exact", reply="好，跑满一轮。"),
            ai_text(file_fence(CLEAN_CODE)),
            *clean_review_calls(),
            ai_text(fix_fence({"main.py": CLEAN_CODE}, [], deleted=[])),
            *clean_review_calls(version=2),
        ]
    )

    code, out = session.run(["写一个加法计算器，跑满一轮"])

    assert code == 0
    state = _round_state(1)
    assert state["rounds_mode"] == state_mod.ROUNDS_EXACT
    assert state["max_iterations"] == 1
    assert state["max_iterations_source"] == state_mod.SOURCE_REQUIREMENT
    assert state["review_round"] == 2  # n+1 次审查
    assert state["fix_rounds"] == 1  # n 轮修复（首审已干净仍继续）
    # 干净轮次的修复是"复核改进"轮：任务提示词说明本轮是跑满轮次下的第几轮修复
    polish_request = "\n".join(
        str(item.content) for request in fake_chat_model.requests for item in request
    )
    assert "本轮要求跑满 1 轮修复（exact 模式），这是第 1 轮" in polish_request
    assert "模式 exact" in out


def test_message_rounds_are_clamped_to_the_configured_limit(
    session: _Session, fake_chat_model: ScriptedChatModel
) -> None:
    """消息里的轮次越界 → 夹到上限并标注（进终端与会话档案，Design §4.4）。"""
    fake_chat_model.load(
        [
            orchestrator_reply("code", rounds=99, reply="好，最多 5 轮。"),
            ai_text(file_fence(CLEAN_CODE)),
            *clean_review_calls(),
        ]
    )

    code, out = session.run(["写一个加法计算器，最多 99 轮"], rounds_limit=5)

    assert code == 0
    state = _round_state(1)
    assert state["max_iterations"] == 5
    assert state["max_iterations_source"] == state_mod.SOURCE_REQUIREMENT
    assert out.count("已按上限 5 夹取") == 1  # 终端只打印一次（不重复标注）
    assert "已按上限 5 夹取" in str(_archive()["rounds"][0]["note"])


def test_command_line_rounds_override_the_message(
    session: _Session, fake_chat_model: ScriptedChatModel
) -> None:
    """优先级：命令行显式 > 消息提取（Design §4.4）。"""
    fake_chat_model.load(
        [
            orchestrator_reply("code", rounds=1, reply="好。"),
            ai_text(file_fence(CLEAN_CODE)),
            *clean_review_calls(),
        ]
    )

    code, _ = session.run(
        ["写一个加法计算器，最多 1 轮"],
        overrides=chat_mod.RoundOverrides(max_iterations=2),
    )

    assert code == 0
    state = _round_state(1)
    assert state["max_iterations"] == 2
    assert state["max_iterations_source"] == state_mod.SOURCE_CLI


def test_residual_round_maps_to_exit_code_one(
    session: _Session, fake_chat_model: ScriptedChatModel
) -> None:
    """达限残留 → 退出码 1 + next_action=fix_again（会话继续可用）。"""
    fake_chat_model.load(
        [
            orchestrator_reply("code", rounds=0, reply="好，只审查。"),
            ai_text(file_fence(CLEAN_CODE)),
            *dirty_review_calls(),
        ]
    )

    code, out = session.run(["写一个加法计算器，先只审一遍"])

    assert code == 1
    assert "达修复轮次上限" in out
    assert "建议动作：可继续修复" in out
    assert _archive()["rounds"][0]["status"] == state_mod.STATUS_RESIDUAL


# ---- 中断与恢复 ----


def test_interrupt_mid_round_is_recorded_and_the_session_survives(
    session: _Session, fake_chat_model: ScriptedChatModel
) -> None:
    """一轮进行中 Ctrl+C：本轮落盘 ``interrupted``、回提示符（会话保留），退出码 2。"""
    fake_chat_model.load(
        [
            orchestrator_reply("code"),
            ai_text(file_fence(CLEAN_CODE)),
            KeyboardInterrupt(),
            orchestrator_reply("ask", reply="好的，我们继续。"),
        ]
    )

    code, out = session.run(["写一个加法计算器", "还在吗？"])

    assert code == 2
    state = _round_state(1)
    assert state["status"] == state_mod.STATUS_INTERRUPTED
    assert state["error"]["node"] == "interrupt"
    assert _archive()["rounds"][0]["status"] == state_mod.STATUS_INTERRUPTED
    assert "[编排] 好的，我们继续。" in out  # 中断后会话仍在，能继续对话
    assert "结束：共 1 轮" in out


def test_resume_restores_memory_versions_and_settings(
    session: _Session, fake_chat_model: ScriptedChatModel
) -> None:
    """``--resume``：恢复记忆（第二轮提示词里带历史）、代码版本（跨轮递增）与轮次设置。"""
    fake_chat_model.load(
        [
            orchestrator_reply("code"),
            ai_text(file_fence(CLEAN_CODE)),
            *clean_review_calls(),
            orchestrator_reply("review_fix", reply="好，接着看。"),
            *dirty_review_calls(version=2),
            ai_text(
                fix_fence(
                    {"main.py": FIXED_CODE},
                    [{"finding_id": "f-1", "status": "fixed", "note": "已修复。"}],
                )
            ),
            *clean_review_calls(version=3),
        ]
    )

    code, _ = session.run(["写一个加法计算器"])
    assert code == 0

    resumed = _Session(
        fake_chat_model,
        session.tmp_path,
        resolved=session.resolved,
        run_id_factory=session.run_id_factory,
    )
    code, out2 = resumed.run(["再检查一轮"], resume="s-20261012-0001")

    assert code == 0
    assert "s-20261012-0001（恢复）" in out2
    assert "已恢复：已完成 1 轮；当前代码 code_v1/" in out2
    assert _archive()["current_version"] == 3
    assert [item["index"] for item in _archive()["rounds"]] == [1, 2]
    # 记忆随会话恢复：第二轮审查节点的历史消息里带第一轮的任务摘要
    assert any(
        "第 1 轮：审查当前版本" in str(item.content)
        for request in fake_chat_model.requests
        for item in request
    )


def test_resume_unknown_session_is_a_readable_error(
    session: _Session, fake_chat_model: ScriptedChatModel
) -> None:
    code, _ = session.run([], resume="s-20261012-9999")

    assert code == 2
    assert any("会话不存在" in line for line in session.err)


# ---- 记忆 ----


def test_memory_appends_structured_summaries_without_code(
    session: _Session, fake_chat_model: ScriptedChatModel
) -> None:
    """记忆按 Agent 追加结构化摘要：无代码围栏、含任务要点与结论。"""
    fake_chat_model.load(
        [
            orchestrator_reply("code"),
            ai_text(file_fence(CLEAN_CODE)),
            *clean_review_calls(),
            orchestrator_reply("review_fix", reply="好，接着看。"),
            *dirty_review_calls(version=2),
            ai_text(
                fix_fence(
                    {"main.py": FIXED_CODE},
                    [{"finding_id": "f-1", "status": "fixed", "note": "已修复。"}],
                )
            ),
            *clean_review_calls(version=3),
        ]
    )

    session.run(["写一个加法计算器", "再检查一轮"])

    encoder_lines = memory_mod.AgentMemory(
        memory_mod.memory_file(_session_dir(), "encoder")
    ).lines()
    reviewer_lines = memory_mod.AgentMemory(
        memory_mod.memory_file(_session_dir(), "reviewer")
    ).lines()
    assert [item["role"] for item in encoder_lines] == ["user", "assistant"]
    assert "写一个加法计算器" in encoder_lines[0]["text"]
    assert "生成 code_v1/" in encoder_lines[1]["text"]
    # 第一轮 1 次审查、第二轮 2 次审查：每次一行任务 + 一行结论
    assert len(reviewer_lines) == 6
    assert any("第 2 轮：审查当前版本" in item["text"] for item in reviewer_lines)
    raw = "\n".join(
        json.dumps(item, ensure_ascii=False) for item in (*encoder_lines, *reviewer_lines)
    )
    assert "```" not in raw  # 代码不进记忆（Design §4.7）


def test_memory_snapshot_compacts_when_the_budget_is_tight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """接近预算时做一次显式压缩：追加 compaction 行，旧行留盘、有效内容变短。"""
    session_path = session_state.create_session_dir("s-20261012-0001")
    store = memory_mod.AgentMemory(
        memory_mod.memory_file(session_path, "reviewer"), agent="reviewer"
    )
    for index in range(1, 5):
        store.append_exchange(
            round_index=index,
            task=f"第 {index} 轮：审查当前版本（本会话第 1 次审查）",
            outcome=f"第 {index} 轮（code_v{index}/，1 个文件）：0 严重 / 0 一般 / 0 建议 —— 干净通过",
            ts=FIXED_TIME,
        )
    before_lines = len(store.lines())
    monkeypatch.setattr(config_mod, "get_max_context_tokens", lambda: 40)

    snapshot = chat_mod._memory_snapshot(session_path, ts=FIXED_TIME)

    after = store.lines()
    assert len(after) > before_lines  # 压缩行是追加的（旧行未改写）
    assert any(item.get("kind") == "compaction" for item in after)
    assert snapshot["reviewer"][0]["text"].startswith("（此前会话的压缩记录）")
    assert len(snapshot["reviewer"]) < before_lines


# ---- 会话目录、报告与退出码 ----


def test_session_directory_is_isomorphic_to_a_run_record(
    session: _Session, fake_chat_model: ScriptedChatModel
) -> None:
    """每轮 = 一条与 ``crw run`` 同构的完整运行记录（report/state/log/workspace）。"""
    fake_chat_model.load(
        [orchestrator_reply("code"), ai_text(file_fence(CLEAN_CODE)), *clean_review_calls()]
    )

    session.run(["写一个加法计算器"])

    rounds_dir = _session_dir() / "rounds" / "round-1"
    assert sorted(item.name for item in rounds_dir.iterdir()) == [
        "log.jsonl",
        "report.md",
        "state.json",
        "workspace",
    ]
    report = (rounds_dir / "report.md").read_text(encoding="utf-8")
    assert "第 1 轮审查（基于 `code_v1/`）" in report
    events = records_mod.read_events(rounds_dir / "log.jsonl")
    assert any(event.get("event") == "start" and event.get("node") == "engine" for event in events)
    assert any(event.get("node") == "encode" for event in events)


def test_session_md_lists_rounds_memory_and_remaining_suggestions(
    session: _Session, fake_chat_model: ScriptedChatModel
) -> None:
    """会话结束写 session.md：逐轮结论演进 + 记忆索引 + 剩余建议（残留时列出）。"""
    fake_chat_model.load(
        [
            orchestrator_reply("code", rounds=0),
            ai_text(file_fence(CLEAN_CODE)),
            *dirty_review_calls(),
        ]
    )

    session.run(["写一个加法计算器"])

    text = (_session_dir() / "session.md").read_text(encoding="utf-8")
    assert "# 会话报告 · s-20261012-0001" in text
    assert "## 逐轮记录" in text
    assert "### 第 1 轮（写/改代码）——达限残留" in text
    assert "缺少除零处理" in text  # 剩余建议来自最后一轮记录
    assert "`reviewer`：2 条记录（`memory/reviewer.jsonl`）" in text


def test_runs_aggregate_session_rounds(
    session: _Session, fake_chat_model: ScriptedChatModel
) -> None:
    """``crw runs`` 跨两处聚合：会话轮次出现在列表里并带会话号。"""
    fake_chat_model.load(
        [orchestrator_reply("code"), ai_text(file_fence(CLEAN_CODE)), *clean_review_calls()]
    )

    session.run(["写一个加法计算器"])

    summaries = records_mod.list_runs()
    assert [item.run_id for item in summaries] == ["r-20261012-0001"]
    assert summaries[0].session_id == "s-20261012-0001"
    assert summaries[0].status == "success"
    assert records_mod.find_run_dir("r-20261012-0001") == _session_dir() / "rounds" / "round-1"
    assert records_mod.load_detail("r-20261012-0001").state["session_id"] == "s-20261012-0001"


def test_agent_names_are_consistent_across_layers() -> None:
    """会话层的节点 → Agent 映射与记忆文件名、各节点读取的记忆键必须一致。

    三处各写一份字面量（记忆文件名、节点映射、节点读取的键）时，任何一处拼错都会让
    历史静默退化为空——不报错、难察觉，因此用一条用例把三者钉在一起。
    """
    import inspect

    from crw.agents import encoder as encoder_mod
    from crw.agents import fixer as fixer_mod
    from crw.agents import reviewer as reviewer_mod
    from crw.session import memory as memory_mod

    node_agents = set(chat_mod._AGENT_OF_NODE.values())
    assert node_agents <= set(memory_mod.MEMORY_AGENTS)

    sources = "\n".join(
        inspect.getsource(module) for module in (encoder_mod, reviewer_mod, fixer_mod)
    )
    for agent in memory_mod.MEMORY_AGENTS:
        assert f'history_from_state(state, "{agent}")' in sources


def test_second_code_round_modifies_the_previous_version(
    session: _Session, fake_chat_model: ScriptedChatModel
) -> None:
    """第二轮"改需求"：既有实现来自上一版本（标题如实写来源），产物版本号跨轮递增。"""
    fake_chat_model.load(
        [
            orchestrator_reply("code", reply="好的，我来实现。"),
            ai_text(file_fence(CLEAN_CODE)),
            *clean_review_calls(),
            orchestrator_reply("code", reply="好，我在现有实现上加功能。"),
            ai_text(file_fence(FIXED_CODE, path="main.py")),
            *clean_review_calls(version=2),
        ]
    )

    code, out = session.run(["写一个加法计算器", "再加一个减法函数"])

    assert code == 0
    first = _round_state(1)
    second = _round_state(2)
    assert first["version_base"] == 1
    assert second["version_base"] == 2
    assert "code_v2" in str(second["current_version_dir"])
    assert _archive()["current_version"] == 2
    # 既有实现来自上一版本目录：标题如实写出来源（不是用户文件、也不是本轮要产出的版本）
    assert any(
        "## 既有实现（来自版本 code_v1/，本会话此前的产出，共 1 个文件）" in str(item.content)
        for request in fake_chat_model.requests
        for item in request
    )
    assert "代码版本 code_v2/（1 个文件）" in out


def test_review_fix_without_explicit_rounds_fixes_even_when_clean(
    session: _Session, fake_chat_model: ScriptedChatModel
) -> None:
    """试用反馈（2026-10-14）：用户要"审查后修复"时，干净的首审不能吞掉修复阶段。

    review_fix 未由消息指定模式、且会话仍是默认 up-to 时，本轮按 exact + 1 轮执行：
    审查 → （复核/修复一轮）→ 复审，轮次用法为 审查 2 轮 / 修复 1 轮。
    """
    fake_chat_model.load(
        [
            orchestrator_reply("code", reply="好的，我来实现。"),
            ai_text(file_fence(CLEAN_CODE)),
            *clean_review_calls(),
            orchestrator_reply("review_fix", reply="好，我先审查，再修一轮。"),
            *clean_review_calls(version=2),
            ai_text(fix_fence({"main.py": CLEAN_CODE}, [])),
            *clean_review_calls(version=3),
        ]
    )

    code, out = session.run(["写一个加法计算器", "审查后修复一轮"])

    assert code == 0
    second = _round_state(2)
    assert second["rounds_mode"] == state_mod.ROUNDS_EXACT
    assert second["max_iterations"] == 1
    assert second["max_iterations_source"] == state_mod.SOURCE_REQUIREMENT
    assert second["review_round"] == 2  # 审查 → 修复 → 复审
    assert second["fix_rounds"] == 1  # 干净的首审也走了一轮修复
    assert "本轮按「审查→修复」默认跑满 1 轮" in out
    assert "模式 exact" in out
    # 干净轮的修复任务明确说"第几轮 / 跑满几轮"（不写错成代码版本号）
    joined = "\n".join(
        str(item.content) for request in fake_chat_model.requests for item in request
    )
    assert "本轮要求跑满 1 轮修复（exact 模式），这是第 1 轮" in joined


def test_review_fix_respects_an_explicit_up_to_mode(
    session: _Session, fake_chat_model: ScriptedChatModel
) -> None:
    """用户明确说"没问题就不用改"（模型给 up-to）时照办：干净的审查不再触发修复。"""
    fake_chat_model.load(
        [
            orchestrator_reply("code", reply="好的，我来实现。"),
            ai_text(file_fence(CLEAN_CODE)),
            *clean_review_calls(),
            orchestrator_reply(
                "review_fix", rounds_mode="up-to", reply="好，我看一眼，有问题再改。"
            ),
            *clean_review_calls(version=2),
        ]
    )

    code, out = session.run(["写一个加法计算器", "再检查一下，没问题就不用改"])

    assert code == 0
    second = _round_state(2)
    assert second["rounds_mode"] == state_mod.ROUNDS_UP_TO
    assert second["fix_rounds"] == 0
    assert "本轮按「审查→修复」默认跑满" not in out


def test_review_fix_with_explicit_rounds_runs_them_all(
    session: _Session, fake_chat_model: ScriptedChatModel
) -> None:
    """消息给了轮次（"修 2 轮"）、未给模式时同样按 exact 跑满：审查 3 轮 / 修复 2 轮。"""
    fake_chat_model.load(
        [
            orchestrator_reply("code", reply="好的，我来实现。"),
            ai_text(file_fence(CLEAN_CODE)),
            *clean_review_calls(),
            orchestrator_reply("review_fix", rounds=2, reply="好，修 2 轮。"),
            *clean_review_calls(version=2),
            ai_text(fix_fence({"main.py": CLEAN_CODE}, [])),
            *clean_review_calls(version=3),
            ai_text(fix_fence({"main.py": CLEAN_CODE}, [])),
            *clean_review_calls(version=4),
        ]
    )

    code, _out = session.run(["写一个加法计算器", "审查后修 2 轮"])

    assert code == 0
    second = _round_state(2)
    assert second["rounds_mode"] == state_mod.ROUNDS_EXACT
    assert second["max_iterations"] == 2
    assert second["review_round"] == 3
    assert second["fix_rounds"] == 2


# ---- 干净判定（默认加严 / --severe-only 放宽，Design §4.4）----


def _moderate_review_calls(*, version: int = 1, name: str = "main.py") -> list[Any]:
    """审查轮次：读文件后只报一条【一般】——默认判据下它同样阻塞。"""
    item = {
        "severity": "一般",
        "location": f"{name}:1",
        "issue": "异常被静默吞掉",
        "suggestion": "记录并给出可读错误",
    }
    return [
        ai_tool_call("read_file", path=f"code_v{version}/{name}"),
        ai_text(review_fence([item], "有一处一般级缺陷。", files=[name])),
    ]


def test_moderate_findings_still_drive_a_fix_round_in_a_session(
    session: _Session, fake_chat_model: ScriptedChatModel
) -> None:
    """会话里的默认判据同样加严：首审只有【一般】也不再早退，修复后仍以残留收尾。"""
    fake_chat_model.load(
        [
            orchestrator_reply("code", reply="好的，我来实现。"),
            ai_text(file_fence(CLEAN_CODE)),
            *_moderate_review_calls(),
            ai_text(
                fix_fence(
                    {"main.py": FIXED_CODE},
                    [{"finding_id": "f-1", "status": "fixed", "note": "已补可读错误"}],
                )
            ),
            *_moderate_review_calls(version=2),
        ]
    )

    code, out = session.run(
        ["写一个加法计算器"], overrides=chat_mod.RoundOverrides(max_iterations=1)
    )

    assert code == 1
    state = _round_state(1)
    assert state["severe_only"] is False
    assert (state["review_round"], state["fix_rounds"]) == (2, 1)
    assert "项阻塞问题（【严重】或【一般】）残留" in out


def test_severe_only_option_reaches_the_round_and_the_archive(
    session: _Session, fake_chat_model: ScriptedChatModel
) -> None:
    """``--severe-only`` 放宽后：只有【一般】的首审即干净早退，并随会话档案留存。"""
    fake_chat_model.load(
        [
            orchestrator_reply("code", reply="好的，我来实现。"),
            ai_text(file_fence(CLEAN_CODE)),
            *_moderate_review_calls(),
        ]
    )

    code, out = session.run(["写一个加法计算器"], severe_only=True)

    assert code == 0
    assert "仅【严重】阻塞（--severe-only 放宽）" in out
    assert _round_state(1)["severe_only"] is True
    assert _archive()["settings"]["severe_only"] is True


def test_session_header_states_the_clean_judgement(
    session: _Session, fake_chat_model: ScriptedChatModel
) -> None:
    """头行写清判据：默认加严后，用户开工前就该知道"有【一般】也要修"。

    放宽开关在恢复会话时给（命令行只能放宽、不能收紧）时，头行仍按**档案现状**显示——
    与轮次设置同口径；本轮实际生效的判据由每轮的引擎头行（``--severe-only 放宽``）给出。
    """
    fake_chat_model.load([orchestrator_reply("ask", reply="你想改哪一部分？")])

    _code, out = session.run(["这个工具有什么用"])

    assert "【严重】与【一般】都算阻塞" in out


def test_resumed_session_keeps_the_relaxed_judgement(
    session: _Session, fake_chat_model: ScriptedChatModel
) -> None:
    """恢复会话不重给开关：档案里的放宽继续生效（与轮次设置同一继承口径）。"""
    fake_chat_model.load(
        [
            orchestrator_reply("code", reply="好的，我来实现。"),
            ai_text(file_fence(CLEAN_CODE)),
            *_moderate_review_calls(),
            orchestrator_reply("review", reply="再看一轮。"),
            *_moderate_review_calls(version=2),
        ]
    )

    assert session.run(["写一个加法计算器"], severe_only=True)[0] == 0
    code, _out = session.run(["再检查一轮"], resume="s-20261012-0001")

    assert code == 0
    assert _round_state(2)["severe_only"] is True
