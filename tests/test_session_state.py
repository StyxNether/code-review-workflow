"""会话态（``session/state.py``）用例：会话号与目录、档案读写、轮次链、列表与降级。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crw.session import state as session_state


def test_session_ids_follow_the_whitelist_and_retry_on_collision(tmp_path: Path) -> None:
    """会话号形如 ``s-YYYYMMDD-xxxx``；与既有目录碰撞时重新抽取。"""
    session_state.create_session_dir("s-20261012-abcd")
    candidates = iter(["abcd", "abcd", "0001"])
    session_id = session_state.new_session_id(day="20261012", hex_factory=lambda: next(candidates))

    assert session_id == "s-20261012-0001"
    assert session_state.is_valid_session_id(session_id)
    assert not session_state.is_valid_session_id("s-20261012-ABCD")  # 大小写：白名单只认小写
    assert not session_state.is_valid_session_id("../../etc/passwd")


def test_session_dir_rejects_invalid_ids() -> None:
    with pytest.raises(ValueError, match="会话号格式非法"):
        session_state.session_dir("../secret")


def test_create_session_dir_builds_memory_and_rounds(tmp_path: Path) -> None:
    """会话目录结构 = session.json 的落点 + memory/ + rounds/。"""
    path = session_state.create_session_dir("s-20261012-0001")

    assert path.is_dir()
    assert (path / "memory").is_dir()
    assert (path / "rounds").is_dir()
    with pytest.raises(FileExistsError):
        session_state.create_session_dir("s-20261012-0001")


def test_session_archive_round_trips_and_whitelists_fields() -> None:
    """档案读写：字段白名单投影（外来键不落盘），轮次条目同样只留声明字段。"""
    state = session_state.new_session_state(
        session_id="s-20261012-0001",
        created_at="2026-10-12T10:00:00+08:00",
        model="DeepSeek/deepseek-flash",
        settings={"max_iterations": 3, "max_iterations_source": "default", "rounds_mode": "up-to"},
    )
    state["unknown_key"] = "不应落盘"  # type: ignore[typeddict-unknown-key]
    session_state.record_round(
        state,
        index=1,
        run_id="r-20261012-0001",
        intent="code",
        message="写一个计算器\n（多行折成一行）",
        created_at="2026-10-12T10:00:01+08:00",
        finished_at="2026-10-12T10:01:00+08:00",
        status="success",
        conclusion="干净通过（共 1 轮审查 / 0 轮修复）",
        rounds_mode="up-to",
        max_iterations=3,
        max_iterations_source="default",
        version=1,
        review_round=1,
        fix_rounds=0,
        note="",
    )
    state["rounds"][-1]["foreign"] = "也不落盘"

    text = session_state.dumps(state)
    data = session_state.loads(text)

    assert "unknown_key" not in data
    assert "foreign" not in data["rounds"][0]
    assert data["rounds"][0]["message"] == "写一个计算器 （多行折成一行）"
    assert data["rounds"][0]["run_id"] == "r-20261012-0001"
    assert data["status"] == session_state.SESSION_STATUS_ACTIVE


def test_write_and_read_session_round_trip(tmp_path: Path) -> None:
    state = session_state.new_session_state(
        session_id="s-20261012-0002",
        created_at="2026-10-12T10:00:00+08:00",
        model="（未解析模型）",
        settings={"max_iterations": 3},
    )
    session_state.write_session(state)

    path = session_state.session_dir("s-20261012-0002")
    assert (path / "session.json").is_file()
    loaded = session_state.read_session(path)
    assert loaded is not None
    assert loaded["session_id"] == "s-20261012-0002"
    assert loaded["rounds"] == []


def test_read_session_tolerates_missing_and_corrupt_files(tmp_path: Path) -> None:
    path = session_state.create_session_dir("s-20261012-0003")
    assert session_state.read_session(path) is None  # 尚未写入

    (path / "session.json").write_text("{不是 JSON", encoding="utf-8")
    assert session_state.read_session(path) is None

    (path / "session.json").write_text("[1, 2]", encoding="utf-8")
    assert session_state.read_session(path) is None


def test_loads_rejects_non_object_payload() -> None:
    with pytest.raises(ValueError, match="顶层必须是对象"):
        session_state.loads("[]")


def test_list_sessions_orders_and_degrades_broken_archives(tmp_path: Path) -> None:
    """列表：按创建时间倒序；``session.json`` 损坏时轮次数数目录、状态标"记录损坏"。"""
    first = "s-20261012-0001"
    second = "s-20261012-0002"
    state = session_state.new_session_state(
        session_id=first,
        created_at="2026-10-12T09:00:00+08:00",
        model="m",
        settings={"max_iterations": 3},
    )
    session_state.write_session(state)
    session_state.record_round(
        state,
        index=1,
        run_id="r-20261012-0001",
        intent="code",
        message="需求",
        created_at="2026-10-12T09:00:01+08:00",
        finished_at="2026-10-12T09:01:00+08:00",
        status="residual",
        conclusion="达限残留",
        rounds_mode="up-to",
        max_iterations=3,
        max_iterations_source="default",
        version=2,
        review_round=1,
        fix_rounds=1,
    )
    session_state.write_session(state)

    broken = session_state.create_session_dir(second)
    (broken / "rounds" / "round-1").mkdir()
    (broken / "rounds" / "round-2").mkdir()
    (broken / "session.json").write_text("{坏", encoding="utf-8")

    summaries = session_state.list_sessions()

    # 降级项没有时间戳，排在有时戳的记录之后（倒序：时戳非空者在前）
    assert [item.session_id for item in summaries] == [first, second]
    healthy = summaries[0]
    assert healthy.round_count == 1
    assert healthy.last_status == "residual"  # 最后状态取最后一轮
    degraded = summaries[1]
    assert degraded.degraded is True
    assert degraded.round_count == 2  # 数轮次目录
    assert degraded.last_status == "（记录损坏）"


def test_next_version_advances_from_current() -> None:
    """起始版本号 = 当前版本 + 1（会话内跨轮连续递增）。"""
    assert session_state.next_version({}) == 1
    assert session_state.next_version({"current_version": 0}) == 1
    assert session_state.next_version({"current_version": 2}) == 3
    assert session_state.next_version({"current_version": -1}) == 1
    assert session_state.next_version({"current_version": "3"}) == 1  # 非法值按 0


def test_summarize_message_flattens_and_truncates() -> None:
    assert session_state.summarize_message("  写 一个\n计算器  ") == "写 一个 计算器"
    long_text = "字" * (session_state.MESSAGE_PREVIEW_CHARS + 10)
    summarized = session_state.summarize_message(long_text)
    assert summarized.endswith("…")
    assert len(summarized) == session_state.MESSAGE_PREVIEW_CHARS


def test_memory_index_uses_posix_style_paths() -> None:
    index = session_state.memory_index({"encoder": "memory/encoder.jsonl", "fixer": "a\\b.jsonl"})
    assert index["encoder"] == "memory/encoder.jsonl"
    assert "\\" not in index["fixer"]


def test_snapshot_keeps_round_entries_serializable() -> None:
    state = session_state.new_session_state(
        session_id="s-20261012-0004",
        created_at="2026-10-12T10:00:00+08:00",
        model="m",
        settings={"max_iterations": 3},
        input_descriptor={"kind": "file", "source_path": Path("a/b.py")},
    )
    data = session_state.snapshot(state)
    assert isinstance(data["input"]["source_path"], str)
    json.dumps(data)  # 可 JSON 序列化（档案的全部字段）


def test_write_session_rejects_an_invalid_session_id(tmp_path: Path) -> None:
    """会话号非法按入参错误抛 ValueError（与 session_dir 同口径，不伪装成 IO 失败）。"""
    with pytest.raises(ValueError, match="会话号非法"):
        session_state.write_session({"session_id": "../etc/passwd"})


def test_snapshot_drops_malformed_round_entries() -> None:
    """轮次条目的坏形态不落盘：空条目会让轮次数与"最后状态"的统计口径失真。"""
    state = session_state.new_session_state(
        session_id="s-20261012-0005",
        created_at="2026-10-12T10:00:00+08:00",
        model="m",
        settings={"max_iterations": 3},
    )
    state["rounds"] = ["坏条目", {}, {"index": 1, "status": "success"}]

    data = session_state.snapshot(state)

    assert data["rounds"] == [{"index": 1, "status": "success"}]
