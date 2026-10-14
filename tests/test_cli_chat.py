"""``crw chat`` / ``crw sessions`` 命令面与两处聚合的用例（Design §8.1 / §4.7）。

会话流程的端到端细节在 ``test_session_chat.py``；本文件盯命令面：参数面、配置默认链、
退出码、sessions/runs/show/resume 的展示与跨两处查找。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

import pytest
from fakes import (
    ScriptedChatModel,
    ai_text,
    clean_review_calls,
    file_fence,
    fix_fence,
    orchestrator_reply,
)

from crw import cli
from crw import config as config_mod
from crw.session import state as session_state
from crw.workflow import records as records_mod
from crw.workflow import state as state_mod

CLEAN_CODE = '''"""计算器（演示用）。"""


def add(left: int, right: int) -> int:
    """相加。"""
    return left + right


if __name__ == "__main__":
    print(add(1, 2))
'''


def _feed_input(monkeypatch: pytest.MonkeyPatch, answers: Sequence[str]) -> None:
    """把交互输入换成脚本（用尽即 EOF，会话按正常退出处理）。"""
    queue = list(answers)

    def fake_input(*args: object) -> str:
        if not queue:
            raise EOFError
        return queue.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)


def _configured(
    configured_provider: Callable[[], dict[str, object]], model: ScriptedChatModel
) -> None:
    configured_provider()
    model.load([])


# ---- 参数面 ----


@pytest.mark.parametrize("mode", ["up-to", "exact"])
def test_run_accepts_rounds_mode(mode: str) -> None:
    parser = cli.build_parser()

    args = parser.parse_args(["run", "--rounds-mode", mode, "写一个计算器"])

    assert args.rounds_mode == mode
    # 轮次语义不并入 max_iter_source：只给模式时轮次数值仍走"配置默认 > 内置"（Design §4.4）
    assert args.max_iter_source is None


def test_invalid_rounds_mode_is_a_parameter_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.build_parser().parse_args(["run", "--rounds-mode", "full", "需求"])

    assert excinfo.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_chat_rejects_file_and_resume_together(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.build_parser().parse_args(["chat", "-f", "a.py", "--resume", "s-20261012-0001"])

    assert excinfo.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


# ---- 配置默认链（--rounds-mode / settings.json）----


def test_run_uses_the_configured_default_rounds(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: ScriptedChatModel,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """配置默认（settings.json 的 rounds.default）进入 run 的取值链，来源记 config。"""
    _configured(configured_provider, fake_chat_model)
    config_mod.save_settings({"rounds": {"default": 5, "limit": 8}})
    fake_chat_model.load([ai_text(file_fence(CLEAN_CODE)), *clean_review_calls()])

    assert cli.main(["run", "写一个计算器"]) == 0

    (summary,) = records_mod.list_runs()
    state = records_mod.read_state(summary.path)
    assert state is not None
    assert state["max_iterations"] == 5
    assert state["max_iterations_source"] == state_mod.SOURCE_CONFIG
    capsys.readouterr()


def test_broken_settings_file_is_a_config_error(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: ScriptedChatModel,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """settings.json 违规按配置错误处理（退出码 2 + 修复指引），不静默夹取。"""
    _configured(configured_provider, fake_chat_model)
    config_mod.settings_path().parent.mkdir(parents=True, exist_ok=True)
    config_mod.settings_path().write_text(
        '{"rounds": {"default": 30, "limit": 30}}', encoding="utf-8"
    )

    code = cli.main(["run", "写一个计算器"])

    assert code == 2
    assert "配置错误" in capsys.readouterr().err
    assert records_mod.list_runs() == []


def test_broken_settings_file_blocks_chat(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: ScriptedChatModel,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configured(configured_provider, fake_chat_model)
    config_mod.settings_path().parent.mkdir(parents=True, exist_ok=True)
    config_mod.settings_path().write_text('{"rounds": {"default": 2}}', encoding="utf-8")
    _feed_input(monkeypatch, [])

    code = cli.main(["chat"])

    assert code == 2
    assert "配置错误" in capsys.readouterr().err


def test_resume_inherits_the_parent_rounds_mode(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: ScriptedChatModel,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """续跑未显式给 --rounds-mode 时继承父运行的模式（与 --severe-only / 审查形态同口径）。"""
    _configured(configured_provider, fake_chat_model)
    fake_chat_model.load(
        [
            ai_text(file_fence(CLEAN_CODE)),
            *clean_review_calls(),
            ai_text(fix_fence({"main.py": CLEAN_CODE}, [])),
            *clean_review_calls(version=2),
        ]
    )
    assert cli.main(["run", "--max-iter", "1", "--rounds-mode", "exact", "写一个计算器"]) == 0
    parent = records_mod.list_runs()[0]
    capsys.readouterr()
    # 子运行继承 exact：首审干净也继续跑满 1 轮修复 + 2 次审查
    fake_chat_model.load(
        [
            *clean_review_calls(version=1),
            ai_text(fix_fence({"main.py": CLEAN_CODE}, [])),
            *clean_review_calls(version=2),
        ]
    )

    child_code = cli.main(["resume", parent.run_id, "--no-exec"])

    assert child_code in (0, 1)
    child = next(item for item in records_mod.list_runs() if item.run_id != parent.run_id)
    state = records_mod.read_state(child.path)
    assert state is not None
    assert state["rounds_mode"] == state_mod.ROUNDS_EXACT
    assert state["max_iterations_source"] == state_mod.SOURCE_PARENT
    capsys.readouterr()


# ---- crw chat：命令面端到端（假模型）----


def test_chat_command_runs_a_full_conversation(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: ScriptedChatModel,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``crw chat`` 两轮对话：第一轮新需求、第二轮"再检查一轮"，退出码按最后一轮。"""
    _configured(configured_provider, fake_chat_model)
    fake_chat_model.load(
        [
            orchestrator_reply("code", reply="好的，我来实现。"),
            ai_text(file_fence(CLEAN_CODE)),
            *clean_review_calls(),
            orchestrator_reply("review_fix", reply="好，先审查再修一轮。"),
            # review_fix 默认 exact + 1 轮：审查（干净）→ 修复 → 复审
            *clean_review_calls(version=2),
            ai_text(fix_fence({"main.py": CLEAN_CODE}, [])),
            *clean_review_calls(version=3),
        ]
    )
    _feed_input(monkeypatch, ["写一个加法计算器", "再检查一轮", "/exit"])

    code = cli.main(["chat", "--no-stream"])

    out = capsys.readouterr().out
    assert code == 0
    assert "── 会话 s-" in out
    assert "[模型] DeepSeek/deepseek-flash" in out
    assert "[编排] 好的，我来实现。" in out
    assert "会话第 1 轮：按「写/改代码」处理" in out
    assert "会话第 2 轮：按「再走一轮审查→修复」处理" in out
    assert "[会话] 结束：共 2 轮；最后状态：干净通过" in out
    archive = _only_session()
    assert [item["intent"] for item in archive["rounds"]] == ["code", "review_fix"]
    assert archive["rounds"][1]["fix_rounds"] == 1  # review_fix 真的修了一轮
    assert archive["current_version"] == 3


def test_chat_exit_code_follows_the_last_round(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: ScriptedChatModel,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """最后一轮残留 → 退出码 1；会话正常结束。"""
    _configured(configured_provider, fake_chat_model)
    fake_chat_model.load(
        [
            orchestrator_reply("code", rounds=0),
            ai_text(file_fence(CLEAN_CODE)),
            *clean_review_calls(),
        ]
    )
    _feed_input(monkeypatch, ["写一个加法计算器，只审一遍", "/exit"])

    code = cli.main(["chat", "--no-stream", "--max-iter", "1"])

    assert code == 0  # 命令行 --max-iter 1 覆盖消息里的 0 → 干净通过
    capsys.readouterr()
    _only_session()


def test_chat_resume_unknown_session_is_readable(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: ScriptedChatModel,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _configured(configured_provider, fake_chat_model)

    code = cli.main(["chat", "--resume", "s-20261012-9999"])

    assert code == 2
    assert "会话不存在" in capsys.readouterr().err


# ---- crw sessions / runs / show / resume ----


def test_sessions_lists_sessions_with_status(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: ScriptedChatModel,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configured(configured_provider, fake_chat_model)
    fake_chat_model.load(
        [
            orchestrator_reply("code"),
            ai_text(file_fence(CLEAN_CODE)),
            *clean_review_calls(),
        ]
    )
    _feed_input(monkeypatch, ["写一个加法计算器", "/exit"])
    assert cli.main(["chat", "--no-stream"]) == 0
    capsys.readouterr()

    assert cli.main(["sessions"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["会话号", "时间", "轮次数", "最后状态"]
    session_id = session_state.list_sessions()[0].session_id
    assert any(session_id in line and "干净通过" in line and "1" in line for line in lines[2:])


def test_sessions_without_records_is_not_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["sessions"]) == 0
    assert "暂无会话记录" in capsys.readouterr().out


def test_runs_lists_session_rounds_with_the_session_column(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: ScriptedChatModel,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configured(configured_provider, fake_chat_model)
    fake_chat_model.load(
        [orchestrator_reply("code"), ai_text(file_fence(CLEAN_CODE)), *clean_review_calls()]
    )
    _feed_input(monkeypatch, ["写一个加法计算器", "/exit"])
    assert cli.main(["chat", "--no-stream"]) == 0
    capsys.readouterr()

    assert cli.main(["runs"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == [
        "运行号",
        "时间",
        "输入摘要",
        "状态",
        "修复轮数",
        "残留严重数",
        "模型",
        "会话",
    ]
    session_id = session_state.list_sessions()[0].session_id
    assert any(session_id in line and "r-" in line for line in lines[2:])


def test_show_finds_a_session_round_by_run_id(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: ScriptedChatModel,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``crw show`` 跨两处查找会话轮次，并标出所属会话与轮次序号。"""
    _configured(configured_provider, fake_chat_model)
    fake_chat_model.load(
        [orchestrator_reply("code"), ai_text(file_fence(CLEAN_CODE)), *clean_review_calls()]
    )
    _feed_input(monkeypatch, ["写一个加法计算器", "/exit"])
    assert cli.main(["chat", "--no-stream"]) == 0
    run_id = records_mod.list_runs()[0].run_id
    capsys.readouterr()

    assert cli.main(["show", run_id]) == 0

    out = capsys.readouterr().out
    assert f"[会话] {session_state.list_sessions()[0].session_id}" in out
    assert "第 1 轮" in out
    assert "第 1 轮（code_v1/，1 个文件）" in out


def test_resume_a_session_round_starts_a_standalone_run(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: ScriptedChatModel,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``crw resume <会话轮次运行号>``：起点取该轮 ``current_version_dir``，新记录落 runs/。"""
    _configured(configured_provider, fake_chat_model)
    fake_chat_model.load(
        [orchestrator_reply("code"), ai_text(file_fence(CLEAN_CODE)), *clean_review_calls()]
    )
    _feed_input(monkeypatch, ["写一个加法计算器", "/exit"])
    assert cli.main(["chat", "--no-stream"]) == 0
    run_id = records_mod.list_runs()[0].run_id
    capsys.readouterr()
    fake_chat_model.load([*clean_review_calls()])

    code = cli.main(["resume", run_id, "--no-exec"])

    assert code == 0
    summaries = records_mod.list_runs()
    assert len(summaries) == 2
    child = next(item for item in summaries if item.run_id != run_id)
    assert child.session_id == ""  # 续跑是独立运行（不在会话内）
    state = records_mod.read_state(child.path)
    assert state is not None
    assert state["parent_run_id"] == run_id
    assert "sessions" in str(state["input"]["source_path"])  # 起点是会话轮次的版本目录
    assert capsys.readouterr().out


def test_session_round_resume_input_summary_uses_version_dir_wording(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: ScriptedChatModel,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M3.5 遗留项 1：续跑记录的输入摘要写"版本目录：<记录名>/code_vN"而不是"文件：code_vN"。"""
    _configured(configured_provider, fake_chat_model)
    fake_chat_model.load(
        [orchestrator_reply("code"), ai_text(file_fence(CLEAN_CODE)), *clean_review_calls()]
    )
    _feed_input(monkeypatch, ["写一个加法计算器", "/exit"])
    assert cli.main(["chat", "--no-stream"]) == 0
    run_id = records_mod.list_runs()[0].run_id
    capsys.readouterr()
    fake_chat_model.load([*clean_review_calls()])
    assert cli.main(["resume", run_id, "--no-exec"]) == 0
    capsys.readouterr()

    assert cli.main(["runs"]) == 0

    out = capsys.readouterr().out
    assert "版本目录：round-1/code_v1" in out
    assert "文件：code_v" not in out


def test_config_menu_sets_rounds_defaults(
    configured_provider: Callable[[], dict[str, object]],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``crw config`` 菜单 7/8 写 settings.json（默认 ≤ 上限 ≤ 20）。"""
    configured_provider()
    answers = iter(["7", "4", "8", "6", "0"])
    monkeypatch.setattr(cli, "_ask", lambda prompt: next(answers, None))

    assert cli.main(["config"]) == 0

    out = capsys.readouterr().out
    assert "7) 默认修复轮次" in out
    assert "已保存：默认修复轮次 4，轮次上限 6" in out
    assert config_mod.rounds_settings() == (4, 6)


def test_config_menu_rejects_an_invalid_rounds_value(
    configured_provider: Callable[[], dict[str, object]],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """违规值不留半状态：default > limit 时报错且 settings.json 保持原状。"""
    configured_provider()
    config_mod.save_settings({"rounds": {"default": 3, "limit": 5}})
    answers = iter(["7", "9", "0"])  # 默认 9 > 上限 5
    monkeypatch.setattr(cli, "_ask", lambda prompt: next(answers, None))

    assert cli.main(["config"]) == 0

    err = capsys.readouterr().err
    assert "不能超过当前上限 5" in err  # 就地在输入项旁报错（不走到写入层）
    assert config_mod.rounds_settings() == (3, 5)  # 违规不留半状态


def _only_session() -> dict[str, object]:
    """读出唯一的会话档案（会话用例只建一个会话）。"""
    summaries = session_state.list_sessions()
    assert len(summaries) == 1
    data = session_state.read_session(summaries[0].path)
    assert data is not None
    return data


def test_session_archive_records_every_round_and_note(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: ScriptedChatModel,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """档案里的轮次条目含运行号 / 结论 / 版本号；夹取标注落 ``note``。"""
    _configured(configured_provider, fake_chat_model)
    config_mod.save_settings({"rounds": {"default": 3, "limit": 4}})
    fake_chat_model.load(
        [
            orchestrator_reply("code", rounds=50),
            ai_text(file_fence(CLEAN_CODE)),
            *clean_review_calls(),
        ]
    )
    _feed_input(monkeypatch, ["写一个加法计算器，最多 50 轮", "/exit"])

    assert cli.main(["chat", "--no-stream"]) == 0

    archive = _only_session()
    entry = archive["rounds"][0]
    assert entry["run_id"].startswith("r-")
    assert entry["conclusion"].startswith("干净通过")
    assert entry["version"] == 1
    assert entry["max_iterations"] == 4
    assert "夹取" in str(entry["note"])
    md = (session_state.sessions_root() / str(archive["session_id"]) / "session.md").read_text(
        encoding="utf-8"
    )
    assert "标注：" in md
    json.dumps(archive)  # 档案可序列化
    capsys.readouterr()


def test_rounds_mode_alone_keeps_the_configured_default(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: ScriptedChatModel,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """只给 ``--rounds-mode`` 时轮次数值仍走配置默认（不并入 max_iter_source，Design §4.4）。"""
    _configured(configured_provider, fake_chat_model)
    config_mod.save_settings({"rounds": {"default": 2, "limit": 8}})
    # exact 跑满 2 轮修复 + 3 次审查（首审干净也不停）
    fake_chat_model.load(
        [
            ai_text(file_fence(CLEAN_CODE)),
            *clean_review_calls(version=1),
            ai_text(fix_fence({"main.py": CLEAN_CODE}, [])),
            *clean_review_calls(version=2),
            ai_text(fix_fence({"main.py": CLEAN_CODE}, [])),
            *clean_review_calls(version=3),
        ]
    )

    assert cli.main(["run", "--rounds-mode", "exact", "写一个计算器"]) == 0

    (summary,) = records_mod.list_runs()
    state = records_mod.read_state(summary.path)
    assert state is not None
    assert state["max_iterations"] == 2  # 配置默认，而不是 argparse 的 3
    assert state["max_iterations_source"] == state_mod.SOURCE_CONFIG
    assert state["rounds_mode"] == state_mod.ROUNDS_EXACT
    assert state["fix_rounds"] == 2 and state["review_round"] == 3
    capsys.readouterr()


def test_configured_default_twenty_is_honoured_and_guarded(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: ScriptedChatModel,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """配置默认可到 20（settings 的天花板）：运行按 20 取值，崩溃护栏按同一数值复算。"""
    _configured(configured_provider, fake_chat_model)
    config_mod.save_settings({"rounds": {"default": 20, "limit": 20}})
    fake_chat_model.load([ai_text(file_fence(CLEAN_CODE)), *clean_review_calls()])

    assert cli.main(["run", "写一个计算器"]) == 0

    (summary,) = records_mod.list_runs()
    state = records_mod.read_state(summary.path)
    assert state is not None
    assert state["max_iterations"] == 20
    assert state["max_iterations_source"] == state_mod.SOURCE_CONFIG
    assert state["review_round"] == 1  # 首审干净即早退，20 是"至多"
    from crw.workflow import graph as graph_mod

    assert graph_mod.recursion_limit(max_iterations=20) == 2 * 20 + 8
    capsys.readouterr()


def test_resume_inherits_a_round_count_above_the_cli_gate(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: ScriptedChatModel,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """父运行的轮次来自配置默认（可达 20）时，续跑按 settings 天花板校验而非命令行的 10。"""
    _configured(configured_provider, fake_chat_model)
    config_mod.save_settings({"rounds": {"default": 15, "limit": 20}})
    fake_chat_model.load([ai_text(file_fence(CLEAN_CODE)), *clean_review_calls()])
    assert cli.main(["run", "写一个计算器"]) == 0
    parent = records_mod.list_runs()[0]
    capsys.readouterr()
    fake_chat_model.load([*clean_review_calls()])

    assert cli.main(["resume", parent.run_id, "--no-exec"]) == 0

    err = capsys.readouterr().err
    assert "不合法" not in err  # 15 是合法记录，不该被判非法
    child = next(item for item in records_mod.list_runs() if item.run_id != parent.run_id)
    state = records_mod.read_state(child.path)
    assert state is not None
    assert state["max_iterations"] == 15
    assert state["max_iterations_source"] == state_mod.SOURCE_PARENT


def test_resume_rejects_an_absurd_inherited_round_count(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: ScriptedChatModel,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """手改成 999 的父运行轮次被拒并如实提示（护栏不因放宽上限而消失）。"""
    _configured(configured_provider, fake_chat_model)
    fake_chat_model.load([ai_text(file_fence(CLEAN_CODE)), *clean_review_calls()])
    assert cli.main(["run", "写一个计算器"]) == 0
    parent = records_mod.list_runs()[0]
    state_path = parent.path / "state.json"
    state = records_mod.read_state(parent.path)
    assert state is not None
    state["max_iterations"] = 999
    state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    capsys.readouterr()
    fake_chat_model.load([*clean_review_calls()])

    assert cli.main(["resume", parent.run_id, "--no-exec"]) == 0

    err = capsys.readouterr().err
    assert "不合法" in err and "999" in err
    child = next(item for item in records_mod.list_runs() if item.run_id != parent.run_id)
    child_state = records_mod.read_state(child.path)
    assert child_state is not None
    assert child_state["max_iterations"] == config_mod.DEFAULT_ROUNDS_DEFAULT


def test_resume_inherits_the_mode_even_when_the_value_comes_from_the_cli(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: ScriptedChatModel,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--max-iter`` 只覆盖数值：未给 ``--rounds-mode`` 时仍继承父运行的 exact 模式。"""
    _configured(configured_provider, fake_chat_model)
    fake_chat_model.load(
        [
            ai_text(file_fence(CLEAN_CODE)),
            *clean_review_calls(version=1),
            ai_text(fix_fence({"main.py": CLEAN_CODE}, [])),
            *clean_review_calls(version=2),
        ]
    )
    assert cli.main(["run", "--max-iter", "1", "--rounds-mode", "exact", "写一个计算器"]) == 0
    parent = records_mod.list_runs()[0]
    capsys.readouterr()
    # 子运行：数值来自命令行（1），模式继承父运行（exact）→ 仍是 1 修复 + 2 审查
    fake_chat_model.load(
        [
            *clean_review_calls(version=1),
            ai_text(fix_fence({"main.py": CLEAN_CODE}, [])),
            *clean_review_calls(version=2),
        ]
    )

    assert cli.main(["resume", parent.run_id, "--max-iter", "1", "--no-exec"]) == 0

    child = next(item for item in records_mod.list_runs() if item.run_id != parent.run_id)
    state = records_mod.read_state(child.path)
    assert state is not None
    assert state["max_iterations"] == 1
    assert state["max_iterations_source"] == state_mod.SOURCE_CLI
    assert state["rounds_mode"] == state_mod.ROUNDS_EXACT
    assert "不合法" not in capsys.readouterr().err


def test_chat_does_not_accept_parallel_review(capsys: pytest.CaptureFixture[str]) -> None:
    """会话内暂不提供 ``--parallel-review``（Design §12.3）：不注册即明确报参数错误。"""
    with pytest.raises(SystemExit) as excinfo:
        cli.build_parser().parse_args(["chat", "--parallel-review"])

    assert excinfo.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
