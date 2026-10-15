"""运行记录 IO：原子写、白名单、列表降级派生与选择。"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from crw import config
from crw.workflow import records
from crw.workflow import state as state_mod


def write_run(
    run_id: str = "r-20261012-0001",
    *,
    state: dict | None = None,
    write_state: bool = True,
    events: list[dict] | None = None,
    workspace: dict[str, str] | None = None,
) -> Path:
    """在重定向后的数据目录里铺一条运行记录（默认字段齐全、状态 success）。

    ``state`` 覆盖默认字段；``write_state=False`` 用于构造"state.json 缺失"的降级场景。
    """
    run_path = records.create_run_dir(run_id)
    payload = {
        "run_id": run_id,
        "input": {"kind": "requirement", "requirement": "写一个计算器"},
        "status": state_mod.STATUS_SUCCESS,
        "created_at": "2026-10-12T10:00:00+08:00",
        "fix_rounds": 1,
        "review_round": 2,
        "findings": [],
    }
    if state:
        payload.update(state)
    if write_state:
        records.atomic_write_text(run_path / "state.json", json.dumps(payload, ensure_ascii=False))
    if events:
        for event in events:
            records.append_log_event(run_path / "log.jsonl", event)
    for name, text in (workspace or {}).items():
        records.atomic_write_text(run_path / "workspace" / name, text)
    return run_path


# ---- 目录与名称白名单 ----


def test_data_dir_follows_crw_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """数据目录经 CRW_HOME 解析：测试不写真实用户目录。"""
    monkeypatch.setenv(config.HOME_ENV, str(tmp_path / "crw-home"))  # 用例自洽：不依赖夹具顺序
    assert config.data_dir().is_relative_to(tmp_path)
    assert records.runs_root() == config.data_dir() / "runs"

    monkeypatch.delenv(config.HOME_ENV)
    assert config.data_dir() == Path.home() / ".crw"


@pytest.mark.parametrize(
    "run_id",
    ["r-20261012-0001", "r-00000000-ffff"],
)
def test_valid_run_ids(run_id: str) -> None:
    assert records.is_valid_run_id(run_id)
    assert records.run_dir(run_id).name == run_id


@pytest.mark.parametrize(
    "run_id",
    [
        "",
        "r-20261012-000",  # 十六进制位数不足
        "r-20261012-00011",
        "r-20261012-000G",  # 非十六进制
        "R-20261012-0001",  # 大写
        "r-2026101-0001",
        "s-20261012-0001",  # 会话号不是运行号
        "../r-20261012-0001",
        "r-20261012-0001/../..",
        "r-20261012-0001 ",
        "r-20261012-0001\n",
    ],
)
def test_invalid_run_ids_are_rejected(run_id: str) -> None:
    """白名单校验挡住路径穿越与格式漂移。"""
    assert not records.is_valid_run_id(run_id)
    with pytest.raises(records.RunRecordError, match="运行号格式非法"):
        records.run_dir(run_id)


def test_new_run_id_retries_on_collision() -> None:
    records.create_run_dir("r-20261012-0001")
    candidates = iter(["0001", "0001", "0002"])

    run_id = records.new_run_id(day="20261012", hex_factory=candidates.__next__)

    assert run_id == "r-20261012-0002"


def test_new_run_id_gives_up_after_repeated_collisions() -> None:
    records.create_run_dir("r-20261012-0001")

    with pytest.raises(records.RunRecordError, match="连续碰撞"):
        records.new_run_id(day="20261012", hex_factory=lambda: "0001")


def test_new_run_id_rejects_generator_output() -> None:
    with pytest.raises(records.RunRecordError, match="非法编号"):
        records.new_run_id(day="20261012", hex_factory=lambda: "zzzz!")


def test_new_run_id_default_generator_matches_the_pattern() -> None:
    assert records.is_valid_run_id(records.new_run_id(day="20261012"))


def test_create_run_dir_refuses_existing_directory() -> None:
    records.create_run_dir("r-20261012-0001")

    with pytest.raises(FileExistsError):
        records.create_run_dir("r-20261012-0001")


# ---- 原子写与事件流 ----


def test_atomic_write_creates_parents_and_replaces_content(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir" / "state.json"

    records.atomic_write_text(target, "第一版")
    records.atomic_write_text(target, "第二版")

    assert target.read_text(encoding="utf-8") == "第二版"
    assert list(target.parent.glob(".*tmp")) == []


def test_atomic_write_leaves_no_temp_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "state.json"

    def broken_replace(source: object, destination: object) -> None:
        raise OSError("替换失败")

    monkeypatch.setattr(os, "replace", broken_replace)

    with pytest.raises(OSError, match="替换失败"):
        records.atomic_write_text(target, "内容")
    assert list(tmp_path.glob(".*tmp")) == []
    assert not target.exists()


def test_append_log_event_keeps_only_schema_keys_and_truncates_detail(tmp_path: Path) -> None:
    log_path = tmp_path / "log.jsonl"

    records.append_log_event(
        log_path,
        {
            "ts": "2026-10-12T10:00:00+08:00",
            "node": "review",
            "event": "end",
            "iteration": 2,
            "detail": "长" * 500,
            "run_id": "r-20261012-0001",  # 渲染用附加键：不落盘
        },
    )
    records.append_log_event(
        log_path, {"ts": "t", "node": "engine", "event": "start", "detail": "短"}
    )

    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert set(lines[0]) == {"ts", "node", "event", "iteration", "detail"}
    assert "run_id" not in lines[0]
    assert len(lines[0]["detail"]) == records.DETAIL_PREVIEW_CHARS + len("…（已截断）")
    assert "已截断" in lines[0]["detail"]
    assert set(lines[1]) == {"ts", "node", "event", "detail"}
    assert lines[1]["detail"] == "短"


def test_read_events_skips_incomplete_and_broken_lines(tmp_path: Path) -> None:
    """跳过写一半的末行与损坏行。"""
    log_path = tmp_path / "log.jsonl"
    log_path.write_text(
        '{"ts": "t", "node": "engine", "event": "start"}\n'
        "不是 JSON 的行\n"
        '{"ts": "t2", "node": "review", "event": "start"}\n'
        '{"ts": "t3", "node": "review", "eve',  # 写一半
        encoding="utf-8",
    )

    events = records.read_events(log_path)

    assert [event["event"] for event in events] == ["start", "start"]


def test_read_events_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert records.read_events(tmp_path / "nonexistent.jsonl") == []


@pytest.mark.parametrize("content", ["{不是 JSON", "[1, 2]", '{"run_id": "x"'])
def test_read_state_rejects_broken_records(tmp_path: Path, content: str) -> None:
    path = tmp_path / "state.json"
    path.write_text(content, encoding="utf-8")

    assert records.read_state(tmp_path) is None


def test_read_rounds_pairs_reviews_with_fixes(tmp_path: Path) -> None:
    run_path = write_run(
        workspace={
            "code_v1/main.py": "print(1)\n",
            "review_1.json": json.dumps({"iteration": 1, "files": ["main.py"], "findings": []}),
            "fixes_1.json": json.dumps({"round": 1, "fixes": []}),
            "review_2.json": json.dumps({"iteration": 2, "findings": []}),
            "review_x.json": "{}",
        }
    )

    rounds = records.read_rounds(run_path)

    assert [item.round_number for item in rounds] == [1, 2]
    assert rounds[0].version_label == "code_v1/"
    assert rounds[0].files == ["main.py"]
    assert rounds[0].fixes is not None
    assert rounds[1].fixes is None
    assert rounds[1].version_label == "code_v2/"  # code_version 缺失时按轮次推出版本目录
    assert rounds[1].files == []


def test_run_artifacts_skip_temporary_and_hidden_files(tmp_path: Path) -> None:
    run_path = write_run(workspace={"code_v1.py": "print(1)\n", ".code_v2.py.tmp": "x"})
    records.atomic_write_text(run_path / ".state.json.tmp", "x")

    names = sorted(path.name for path in records.run_artifacts(run_path))

    assert names == ["code_v1.py", "state.json"]


# ---- 列表与降级派生 ----


def test_list_runs_ignores_foreign_entries(tmp_path: Path) -> None:
    write_run("r-20261012-0001")
    (records.runs_root() / "not-a-run").mkdir(parents=True)
    (records.runs_root() / "r-20261012-0002.tmp").write_text("x", encoding="utf-8")
    (records.runs_root() / "note.txt").write_text("x", encoding="utf-8")

    assert [item.run_id for item in records.list_runs()] == ["r-20261012-0001"]


def test_list_runs_sorts_newest_first() -> None:
    write_run("r-20261012-0001", state={"created_at": "2026-10-12T09:00:00+08:00"})
    write_run("r-20261012-0002", state={"created_at": "2026-10-12T11:00:00+08:00"})
    write_run("r-20261012-0003", state={"created_at": "2026-10-12T10:00:00+08:00"})

    assert [item.run_id for item in records.list_runs()] == [
        "r-20261012-0002",
        "r-20261012-0003",
        "r-20261012-0001",
    ]


def test_list_runs_is_empty_without_records() -> None:
    assert records.list_runs() == []


def test_list_runs_derives_columns_from_workspace_and_reviews() -> None:
    """单条记录状态损坏时按降级派生链回退。"""
    run_path = write_run(
        "r-20261012-0001",
        write_state=False,
        events=[
            {"ts": "t", "node": "engine", "event": "start", "detail": "开始"},
            {"ts": "t", "node": "engine", "event": "end", "detail": "residual"},
        ],
        workspace={
            "code_v1/main.py": "print(1)\n",
            "code_v2/main.py": "print(2)\n",
            "code_v3/main.py": "print(3)\n",
            "review_1.json": json.dumps({"findings": [{"severity": "严重"}]}),
            "review_2.json": json.dumps({"findings": [{"severity": "严重"}, {"severity": "严重"}]}),
        },
    )

    (summary,) = records.list_runs()

    assert summary.path == run_path
    assert summary.status == "residual"
    assert summary.fix_rounds == 2
    assert summary.residual_severe == 2
    assert summary.input_summary == "（记录损坏）"
    assert summary.created_at  # 退回目录修改时间


def test_list_runs_infers_failed_status_from_log_tail() -> None:
    write_run(
        "r-20261012-0001",
        write_state=False,
        events=[
            {"ts": "t", "node": "engine", "event": "start"},
            {"ts": "t", "node": "review", "event": "error", "detail": "模型调用失败"},
        ],
    )

    (summary,) = records.list_runs()

    assert summary.status == state_mod.STATUS_FAILED
    assert summary.fix_rounds is None
    assert summary.residual_severe is None


def test_list_runs_reports_unknown_when_nothing_is_readable() -> None:
    records.create_run_dir("r-20261012-0001")

    (summary,) = records.list_runs()

    assert summary.status == "unknown"


# ---- 单条加载 ----


def test_load_detail_reads_state_events_rounds_and_artifacts() -> None:
    write_run(
        workspace={
            "code_v1.py": "print(1)\n",
            "review_1.json": json.dumps({"findings": [{"severity": "严重"}]}),
        },
        events=[{"ts": "t", "node": "engine", "event": "end", "detail": "success"}],
    )

    detail = records.load_detail("r-20261012-0001")

    assert detail.state["run_id"] == "r-20261012-0001"
    assert len(detail.events) == 1
    assert [item.round_number for item in detail.rounds] == [1]
    assert {path.name for path in detail.artifacts} == {
        "state.json",
        "log.jsonl",
        "code_v1.py",
        "review_1.json",
    }


def test_load_detail_rejects_unknown_or_invalid_ids() -> None:
    with pytest.raises(records.RunRecordError, match="不存在"):
        records.load_detail("r-20261012-0001")
    with pytest.raises(records.RunRecordError, match="格式非法"):
        records.load_detail("../etc/passwd")


def test_load_detail_degrades_when_state_is_corrupt() -> None:
    """state.json 损坏 → 降级展示：给日志与产物，不抛异常、不假装完好。"""
    run_path = records.create_run_dir("r-20261012-0001")
    (run_path / "state.json").write_text("{损坏", encoding="utf-8")
    records.append_log_event(
        run_path / "log.jsonl", {"ts": "t1", "node": "review", "event": "start", "iteration": 1}
    )
    records.append_log_event(
        run_path / "log.jsonl",
        {"ts": "t2", "node": "review", "event": "end", "iteration": 1, "detail": "第 1 轮"},
    )
    (run_path / "workspace" / "review_1.json").write_text(
        json.dumps({"file": "code_v1.py", "findings": []}), encoding="utf-8"
    )

    detail = records.load_detail("r-20261012-0001")

    assert detail.degraded is True
    assert detail.state == {}
    assert "state.json" in detail.degraded_reason
    assert [event["event"] for event in detail.events] == ["start", "end"]
    assert [item.round_number for item in detail.rounds] == [1]


def test_load_detail_is_not_degraded_for_a_healthy_record() -> None:
    write_run(workspace={"code_v1.py": "print(1)\n"})

    detail = records.load_detail("r-20261012-0001")

    assert detail.degraded is False
    assert detail.degraded_reason == ""


# ---- 选择 ----


@pytest.mark.parametrize(
    ("raw", "count", "expected"),
    [("1", 3, 0), ("3", 3, 2), ("2", 2, 1), ("0", 3, None), ("-1", 3, None), ("4", 3, None)],
)
def test_parse_index(raw: str, count: int, expected: int | None) -> None:
    assert records.parse_index(raw, count) == expected


@pytest.mark.parametrize("raw", [None, "", " ", "abc", "1.0", "１", "1 0", "0x1"])
def test_parse_index_rejects_non_ascii_or_out_of_range(raw: str | None) -> None:
    """ASCII 白名单：全角数字、空串与越界一律视为取消。"""
    assert records.parse_index(raw, 3) is None


def test_select_run_returns_the_chosen_record() -> None:
    write_run("r-20261012-0001", state={"created_at": "2026-10-12T09:00:00+08:00"})
    write_run("r-20261012-0002", state={"created_at": "2026-10-12T11:00:00+08:00"})
    printed: list[str] = []

    picked = records.select_run(records.list_runs(), ask=lambda prompt: "1", show=printed.append)

    assert picked is not None and picked.run_id == "r-20261012-0002"  # 列表按时间倒序
    assert len(printed) == 2  # 两行候选
    assert "r-20261012-0002" in printed[0]


@pytest.mark.parametrize("answer", [None, "", "0", "abc"])
def test_select_run_returns_none_when_cancelled(answer: str | None) -> None:
    write_run("r-20261012-0001")

    picked = records.select_run(
        records.list_runs(), ask=lambda prompt: answer, show=lambda text: None
    )

    assert picked is None


# ---- 陈旧 running 标注 ----


def test_stale_running_is_flagged_after_thirty_minutes(tmp_path: Path) -> None:
    """判据 = max(mtime(state.json), mtime(log.jsonl)) 超过 30 分钟；只标注不改状态。"""
    run_path = write_run("r-20261012-0001", state={"status": state_mod.STATUS_RUNNING})
    records.append_log_event(
        run_path / "log.jsonl", {"ts": "t", "node": "engine", "event": "start"}
    )
    old = time.time() - records.STALE_RUNNING_SECONDS - 60
    for name in ("state.json", "log.jsonl"):
        os.utime(run_path / name, (old, old))

    (summary,) = records.list_runs()

    assert summary.stale is True
    assert summary.status == state_mod.STATUS_RUNNING  # 不自动改状态
    state = records.read_state(run_path)
    assert state is not None and state["status"] == state_mod.STATUS_RUNNING


def test_fresh_or_finished_runs_are_not_flagged_stale() -> None:
    write_run("r-20261012-0001", state={"status": state_mod.STATUS_RUNNING})
    write_run("r-20261012-0002")  # success

    summaries = {item.run_id: item for item in records.list_runs()}

    assert summaries["r-20261012-0001"].stale is False
    assert summaries["r-20261012-0002"].stale is False


def test_stale_detection_ignores_non_running_and_missing_state(tmp_path: Path) -> None:
    assert records.is_stale_running(None, tmp_path) is False
    assert records.is_stale_running({"status": state_mod.STATUS_SUCCESS}, tmp_path) is False


def test_log_preview_falls_back_for_unserialisable_details(tmp_path: Path) -> None:
    """detail 含不可 JSON 序列化对象时回退 repr：日志预览不得把整次写日志弄崩。"""
    path = tmp_path / "log.jsonl"

    records.append_log_event(
        path, {"ts": "t", "node": "engine", "event": "start", "detail": {1, 2}}
    )

    line = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert "{" in str(line["detail"]) and "}" in str(line["detail"])  # set 的 repr 形态


# ---- 会话轮次记录：两处查找与降级引用 ----


def _make_session_round(
    session_id: str = "s-20261012-0001", round_name: str = "round-1", *, run_id: str = ""
) -> Path:
    """运行号由 ``round_name`` 确定性派生（不用 ``hash``：进程间随机化会让断言漂移）。"""
    suffix = int(round_name.split("-")[1]) if "-" in round_name else 0
    """手工造一条会话轮次记录（引擎在会话内写出的同构形态：round-<i>/ + state.json）。"""
    path = records.sessions_root() / session_id / "rounds" / round_name
    (path / "workspace").mkdir(parents=True)
    state = {
        "run_id": run_id or f"r-20261012-{suffix:04x}",
        "session_id": session_id,
        "round_index": int(round_name.split("-")[1]),
        "status": state_mod.STATUS_SUCCESS,
        "created_at": "2026-10-12T10:00:00+08:00",
        "input": {"kind": "requirement", "requirement": "写一个计算器"},
        "fix_rounds": 0,
    }
    (path / "state.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    return path


def test_find_run_dir_searches_both_places() -> None:
    """运行号在 runs/ 与会话轮次两处都查得到（会话侧按 state.json 的 run_id 匹配）。"""
    direct = records.create_run_dir("r-20261012-0001")
    round_path = _make_session_round(run_id="r-20261012-0002")

    assert records.find_run_dir("r-20261012-0001") == direct
    assert records.find_run_dir("r-20261012-0002") == round_path
    assert records.find_run_dir("r-20261012-9999") is None
    assert records.find_run_dir("../secret") is None


def test_find_run_dir_supports_the_degraded_reference_form() -> None:
    """state.json 不可读的会话轮次用 ``<会话号>/round-<i>`` 引用（列表的降级标识）。"""
    path = _make_session_round()
    (path / "state.json").unlink()

    assert records.find_run_dir("s-20261012-0001/round-1") == path
    assert records.find_run_dir("s-20261012-0001/round-9") is None
    assert records.parse_session_round_ref("s-20261012-0001/round-2") == (
        "s-20261012-0001",
        "round-2",
    )
    assert records.parse_session_round_ref("r-20261012-0001") is None


def test_list_runs_aggregates_session_rounds_and_degrades_them() -> None:
    """runs 列表跨两处聚合；会话轮次的 state.json 不可读时用降级引用标识。"""
    records.create_run_dir("r-20261012-0001")
    good = _make_session_round(run_id="r-20261012-0002")
    broken = _make_session_round(round_name="round-2", run_id="r-20261012-0003")
    (broken / "state.json").write_text("{坏", encoding="utf-8")

    summaries = {item.run_id: item for item in records.list_runs()}

    assert set(summaries) == {"r-20261012-0001", "r-20261012-0002", "s-20261012-0001/round-2"}
    assert summaries["r-20261012-0002"].session_id == "s-20261012-0001"
    assert summaries["r-20261012-0002"].path == good
    degraded = summaries["s-20261012-0001/round-2"]
    assert degraded.session_id == "s-20261012-0001"
    assert degraded.status == "unknown"  # 状态由日志也推不出来


def test_load_detail_accepts_the_degraded_reference_form() -> None:
    path = _make_session_round(round_name="round-3", run_id="r-20261012-0004")
    (path / "state.json").write_text("{坏", encoding="utf-8")

    detail = records.load_detail("s-20261012-0001/round-3")

    assert detail.degraded is True
    assert detail.summary.session_id == "s-20261012-0001"


def test_create_run_dir_at_refuses_to_overwrite() -> None:
    path = records.sessions_root() / "s-20261012-0001" / "rounds" / "round-1"

    assert records.create_run_dir_at(path) == path
    try:
        records.create_run_dir_at(path)
    except FileExistsError:
        pass
    else:  # pragma: no cover — 上面必须抛
        raise AssertionError("既有轮次目录被覆盖了")
