"""引擎层与 M4 相关的接缝用例：起始版本号、exact 轮次语义、会话轮次的中断口径与运行目录。

节点用脚本化假节点（确定性），引擎的真实包装层（契约、落盘、报告）照走。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fakes import SCRIPTED_NODES, fake_encode

from crw import rounds as rounds_mod
from crw.workflow import engine as engine_mod
from crw.workflow import records as records_mod
from crw.workflow import state as state_mod

FIXED_CLOCK = "2026-10-12T10:00:00+08:00"


def _clock() -> str:
    return FIXED_CLOCK


def _ids() -> Callable[[], str]:
    counter = {"value": 0}

    def factory() -> str:
        counter["value"] += 1
        return f"r-20261012-{counter['value']:04x}"

    return factory


def _engine(**kwargs: Any) -> engine_mod.WorkflowEngine:
    kwargs.setdefault("clock", _clock)
    kwargs.setdefault("run_id_factory", _ids())
    kwargs.setdefault("nodes", SCRIPTED_NODES())
    return engine_mod.WorkflowEngine(**kwargs)


def test_start_version_names_the_first_version_and_advances_fixes(tmp_path: Path) -> None:
    """起始版本号 = 本次运行的第一个版本号（会话跨轮连续递增，Design §4.7）。"""
    result = _engine().execute(
        engine_mod.RunRequest(requirement="写一个计算器", start_version=3, max_iterations=1)
    )

    assert result.status == state_mod.STATUS_SUCCESS
    workspace = result.run_dir / "workspace"
    assert (workspace / "code_v3").is_dir()
    assert (workspace / "code_v1").exists() is False
    state = result.state
    assert state["version_base"] == 3
    assert rounds_mod.code_version(state) == 3
    assert "code_v3" in str(state["current_version_dir"])
    report = (result.run_dir / "report.md").read_text(encoding="utf-8")
    assert "第 1 轮审查（基于 `code_v3/`）" in report


def test_start_version_applies_to_materialized_source_dir(tmp_path: Path) -> None:
    """会话审查轮次：既有版本目录被复制为起始版本目录（编号 = 起始版本号）。"""
    source = tmp_path / "code_v2"
    source.mkdir()
    (source / "main.py").write_text('"""既有代码。"""\n\nprint(1)\n', encoding="utf-8")

    result = _engine().execute(
        engine_mod.RunRequest(source_dir=source, start_version=3, max_iterations=0)
    )

    workspace = result.run_dir / "workspace"
    assert (workspace / "code_v3" / "main.py").is_file()
    assert (workspace / "code_v3" / "manifest.json").is_file()
    assert result.state["version_files"] == ["main.py"]
    assert result.state["version_base"] == 3


def test_prior_files_reach_the_encode_node(tmp_path: Path) -> None:
    """编码上下文（既有实现）经瞬态键 ``prior_files`` 到达编码节点，不进运行记录。"""
    seen: dict[str, Any] = {}

    def capture_encode(state: dict[str, Any]) -> dict[str, Any]:
        seen["prior"] = state.get("prior_files")
        return fake_encode(state)

    result = _engine(nodes={**SCRIPTED_NODES(), "encode": capture_encode}).execute(
        engine_mod.RunRequest(
            requirement="把它加个减法",
            prior_files={"main.py": "print(1)\n"},
            max_iterations=0,
        )
    )

    assert seen["prior"] == {"main.py": "print(1)\n"}
    state = result.state
    assert "prior_files" not in json.loads(state_mod.dumps(state))  # 瞬态键不进快照


def test_exact_mode_runs_every_round_and_ends_with_a_review(tmp_path: Path) -> None:
    """``exact``：跑满 n 轮修复 + n+1 次审查，即使中途干净（Design §4.4）。"""
    result = _engine().execute(
        engine_mod.RunRequest(
            requirement="写一个计算器",
            max_iterations=2,
            rounds_mode=rounds_mod.ROUNDS_EXACT,
        )
    )

    assert result.status == state_mod.STATUS_SUCCESS
    state = result.state
    assert state["review_round"] == 3
    assert state["fix_rounds"] == 2
    assert state["next_action"] == state_mod.NEXT_DONE
    nodes = [entry["node"] for entry in state["history"]]
    assert nodes == ["encode", "review", "fix", "review", "fix", "review"]
    assert "本轮未产生改动" in state["history"][2]["summary"]  # 第 1 轮修复未改内容


def test_exact_mode_with_residual_reports_it(tmp_path: Path) -> None:
    """exact 跑满后仍不干净 → 残留收尾（退出码 1）与 ``fix_again`` 建议。"""
    from fakes import finding

    def always_dirty(state: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "files": ["main.py"],
            "findings": [finding(location="main.py:1", issue="示例严重问题")],
            "summary": "仍有严重问题。",
        }
        return {"review_text": f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```\n"}

    result = _engine(nodes={**SCRIPTED_NODES(), "review": always_dirty}).execute(
        engine_mod.RunRequest(
            requirement="写一个计算器",
            max_iterations=1,
            rounds_mode=rounds_mod.ROUNDS_EXACT,
        )
    )

    assert result.status == state_mod.STATUS_RESIDUAL
    assert result.exit_code == 1
    assert result.fix_rounds == 1
    assert result.review_round == 2
    assert result.state["next_action"] == state_mod.NEXT_FIX_AGAIN


def test_interrupt_in_a_session_round_is_recorded_as_interrupted(tmp_path: Path) -> None:
    """会话轮次被中断 → ``status=interrupted``（一次性运行仍按 §4.6 记 failed）。"""

    def interrupt(state: dict[str, Any]) -> dict[str, Any]:
        raise KeyboardInterrupt

    session_result = _engine(nodes={**SCRIPTED_NODES(), "review": interrupt}).execute(
        engine_mod.RunRequest(
            requirement="写一个计算器", session_id="s-20261012-0001", round_index=1
        )
    )
    plain_result = _engine(nodes={**SCRIPTED_NODES(), "review": interrupt}).execute(
        engine_mod.RunRequest(requirement="写一个计算器")
    )

    assert session_result.status == state_mod.STATUS_INTERRUPTED
    assert session_result.exit_code == 2
    assert session_result.state["error"] == {
        "node": "interrupt",
        "message": "运行被用户中断（Ctrl+C）。",
    }
    assert plain_result.status == state_mod.STATUS_FAILED


def test_explicit_run_dir_is_used_with_the_round_name(tmp_path: Path) -> None:
    """``run_dir`` 指定时按该目录建记录（会话轮次目录名 = ``round-<i>``，Design §4.7）。"""
    rounds_dir = records_mod.sessions_root() / "s-20261012-0001" / "rounds"
    rounds_dir.mkdir(parents=True)

    result = _engine(run_dir=rounds_dir / "round-2").execute(
        engine_mod.RunRequest(
            requirement="写一个计算器", session_id="s-20261012-0001", round_index=2
        )
    )

    assert result.run_dir == rounds_dir / "round-2"
    assert (result.run_dir / "workspace").is_dir()
    state = records_mod.read_state(result.run_dir)
    assert state is not None
    assert state["session_id"] == "s-20261012-0001"
    assert state["round_index"] == 2
    assert state["run_id"] == result.run_id
    assert records_mod.find_run_dir(result.run_id) == result.run_dir


def test_existing_run_dir_is_not_overwritten(tmp_path: Path) -> None:
    """同名轮次目录已存在 → 可读报错（既有轮次记录绝不被覆盖）。"""
    rounds_dir = records_mod.sessions_root() / "s-20261012-0001" / "rounds"
    (rounds_dir / "round-1").mkdir(parents=True)

    result = _engine(run_dir=rounds_dir / "round-1").execute(
        engine_mod.RunRequest(
            requirement="写一个计算器", session_id="s-20261012-0001", round_index=1
        )
    )

    assert result.exit_code == 2
    assert "运行目录创建失败" in (result.error_message or "")


def test_agent_memory_is_passed_through_without_persisting(tmp_path: Path) -> None:
    """三 Agent 记忆经瞬态键到达节点，不落运行记录（Design §4.7）。"""
    seen: dict[str, Any] = {}

    def capture_review(state: dict[str, Any]) -> dict[str, Any]:
        seen["memory"] = state.get("agent_memory")
        payload = {"files": ["main.py"], "findings": [], "summary": "无问题。"}
        return {"review_text": f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```\n"}

    result = _engine(nodes={**SCRIPTED_NODES(), "review": capture_review}).execute(
        engine_mod.RunRequest(
            requirement="写一个计算器",
            agent_memory={"reviewer": [{"role": "user", "text": "第 1 轮：审查当前版本"}]},
        )
    )

    assert seen["memory"] == {"reviewer": [{"role": "user", "text": "第 1 轮：审查当前版本"}]}
    assert "agent_memory" not in json.loads(state_mod.dumps(result.state))


def test_session_round_detail_carries_session_fields(tmp_path: Path) -> None:
    """会话轮次记录在 ``crw show`` 的数据里带会话号 / 轮次序号与版本基础号。"""
    rounds_dir = records_mod.sessions_root() / "s-20261012-0001" / "rounds"
    rounds_dir.mkdir(parents=True)
    result = _engine(run_dir=rounds_dir / "round-1").execute(
        engine_mod.RunRequest(
            requirement="写一个计算器",
            session_id="s-20261012-0001",
            round_index=1,
            start_version=4,
            max_iterations=0,
        )
    )

    detail = records_mod.load_detail(result.run_id)

    assert detail.summary.session_id == "s-20261012-0001"
    assert detail.state["round_index"] == 1
    assert detail.state["version_base"] == 4
    assert detail.rounds[0].version_label == "code_v4/"


def test_requirement_with_source_dir_feeds_the_encoder_without_materializing(
    tmp_path: Path,
) -> None:
    """需求起步 + 既有来源：既有代码只作编码上下文（prior_files），不物化起始版本目录。

    物化会留一份"不在新版本清单里"的旧文件在版本目录里（读取方以 manifest 为准，但记录里
    多一个会误导的产物），因此该组合下引擎把既有代码读进 ``prior_files`` 交给编码节点，
    版本目录由编码输出定义；命令面（会话的"改需求"轮次）按同一语义传参。
    """
    source = tmp_path / "code_v2"
    source.mkdir()
    (source / "old.py").write_text('"""旧实现。"""\n', encoding="utf-8")

    result = _engine().execute(
        engine_mod.RunRequest(
            requirement="改成新实现",
            source_dir=source,
            start_version=3,
            max_iterations=0,
        )
    )

    version_dir = result.run_dir / "workspace" / "code_v3"
    assert result.state["version_files"] == ["main.py"]
    assert (version_dir / "main.py").is_file()
    assert not (version_dir / "old.py").exists()  # 没有物化残留
    manifest = json.loads((version_dir / "manifest.json").read_text(encoding="utf-8"))
    assert [item["path"] for item in manifest["files"]] == ["main.py"]


def test_new_run_id_avoids_ids_used_by_session_rounds() -> None:
    """运行号判重覆盖会话轮次（跨两处全局唯一）：候选撞上时换号。"""
    rounds_dir = records_mod.sessions_root() / "s-20261012-0001" / "rounds"
    (rounds_dir / "round-1" / "workspace").mkdir(parents=True)
    (rounds_dir / "round-1" / "state.json").write_text(
        json.dumps({"run_id": "r-20261012-0001"}, ensure_ascii=False), encoding="utf-8"
    )
    candidates = iter(["0001", "0002"])

    generated = records_mod.new_run_id(day="20261012", hex_factory=lambda: next(candidates))

    assert generated == "r-20261012-0002"
