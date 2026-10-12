"""状态 schema 与序列化：字段齐全、快照投影、JSON 往返（Design §4.3、§9.1）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crw.workflow import state as state_mod


def make_state(**overrides: object) -> dict[str, object]:
    """构造一份完整的初始状态（可覆盖任意字段）。"""
    state: dict[str, object] = dict(
        state_mod.new_state(
            run_id="r-20261012-0001",
            input_descriptor=state_mod.InputDescriptor(
                kind="requirement", requirement="写一个计算器"
            ),
            workspace=Path("runs/r-20261012-0001/workspace"),
            max_iterations=3,
            max_iterations_source=state_mod.SOURCE_DEFAULT,
            model="内置假节点（未接模型）",
            created_at="2026-10-12T10:00:00+08:00",
        )
    )
    state.update(overrides)
    return state


def test_new_state_fills_every_persisted_field() -> None:
    """初始状态字段与 Design §4.3 的持久化列表一一对应（不多不少）。"""
    state = make_state()

    assert set(state) == set(state_mod.PERSISTED_FIELDS)
    assert state["status"] == state_mod.STATUS_RUNNING
    assert state["review_round"] == 0
    assert state["fix_rounds"] == 0
    assert state["findings"] == []
    assert state["clean"] is False
    assert state["next_action"] is None
    assert state["error"] is None
    assert state["finished_at"] is None
    assert state["history"] == []


def test_new_state_records_optional_sources() -> None:
    """续跑与会话字段（parent_run_id / session_id / round_index）按入参写入。"""
    state = make_state(parent_run_id="r-20261011-0001", session_id="s-20261012-0001", round_index=2)

    assert state["parent_run_id"] == "r-20261011-0001"
    assert state["session_id"] == "s-20261012-0001"
    assert state["round_index"] == 2


@pytest.mark.parametrize(
    ("fix_rounds", "expected_version"),
    [(0, 1), (1, 2), (3, 4)],
)
def test_review_round_and_code_version_align_at_each_review_boundary(
    fix_rounds: int, expected_version: int
) -> None:
    """审查边界上（review_round == fix_rounds）本轮审查序号 == 当前代码版本号（Design §4.4）。"""
    state = make_state(fix_rounds=fix_rounds, review_round=fix_rounds)

    assert state_mod.code_version(state) == expected_version
    assert state_mod.round_number_for_review(state) == expected_version
    assert state_mod.round_number_for_fix(state) == expected_version


def test_artifact_names_follow_the_directory_contract() -> None:
    """产物命名与 Design §4.5 的目录契约一致。"""
    assert state_mod.code_file_name(2) == "code_v2.py"
    assert state_mod.review_file_name(3) == "review_3.json"
    assert state_mod.fixes_file_name(3) == "fixes_3.json"


def test_snapshot_projects_only_persisted_fields() -> None:
    """节点交回的暂存键与未来扩展键不进入运行记录（Design §3.1）。"""
    state = make_state(code_text="print('未落盘')", review_text="x", future_field=1)

    snapshot = state_mod.snapshot(state)

    assert set(snapshot) == set(state_mod.PERSISTED_FIELDS)
    assert "code_text" not in snapshot
    assert list(snapshot) == list(state_mod.PERSISTED_FIELDS)


def test_snapshot_fills_defaults_for_missing_fields() -> None:
    """缺席字段补默认值：state.json 恒为完整快照，读取方无需做缺键判断。"""
    snapshot = state_mod.snapshot({"run_id": "r-20261012-0001"})

    assert snapshot["status"] == state_mod.STATUS_RUNNING
    assert snapshot["findings"] == []
    assert snapshot["max_iterations_source"] == state_mod.SOURCE_DEFAULT
    assert snapshot["rounds_mode"] == state_mod.ROUNDS_UP_TO


def test_snapshot_coerces_paths_and_nested_values() -> None:
    """Path 与嵌套结构归一为 JSON 可序列化形态（workspace / current_code 为 str）。"""
    state = make_state(
        workspace=Path("runs/x/workspace"), current_code=Path("runs/x/workspace/code_v1.py")
    )

    snapshot = state_mod.snapshot(state)

    assert snapshot["workspace"] == str(Path("runs/x/workspace"))
    assert str(snapshot["current_code"]).endswith("code_v1.py")
    json.dumps(snapshot)  # 不抛 = 可序列化


def test_dumps_and_loads_round_trip_keeps_chinese_readable() -> None:
    """中文不转义（报告与记录需人读），往返后字段等值。"""
    state = make_state(findings=[{"id": "f-1", "severity": "严重", "location": "code_v1.py:3"}])

    text = state_mod.dumps(state)
    restored = state_mod.loads(text)

    assert "严重" in text
    assert "\\u4e25" not in text
    assert restored == state_mod.snapshot(state)


def test_loads_rejects_non_object_json() -> None:
    with pytest.raises(ValueError, match="顶层必须是对象"):
        state_mod.loads("[1, 2, 3]")


@pytest.mark.parametrize(
    ("descriptor", "expected"),
    [
        (
            {"kind": "file", "source_path": "tests/fixtures/buggy_samples/syntax_error.py"},
            "文件：syntax_error.py",
        ),
        ({"kind": "requirement", "requirement": "写一个计算器"}, "需求：写一个计算器"),
        ({"kind": "requirement", "requirement": " 多  空白  需求 "}, "需求：多 空白 需求"),
        ({"kind": "requirement", "requirement": ""}, "（空需求）"),
    ],
)
def test_summarize_input(descriptor: dict[str, str], expected: str) -> None:
    assert state_mod.summarize_input(descriptor) == expected


def test_summarize_input_truncates_long_requirement() -> None:
    summary = state_mod.summarize_input(
        {"kind": "requirement", "requirement": "长" * 100}, limit=20
    )

    assert len(summary) == 20
    assert summary.endswith("…")
