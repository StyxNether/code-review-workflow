"""命令面测试：版本、帮助、参数解析、输入契约与 run / runs / show 实现（无真实网络调用）。"""

from __future__ import annotations

import argparse
import importlib.metadata
import re
from pathlib import Path

import pytest
from support import sample_path

import crw.cli
import crw.render
from crw.cli import build_parser, main
from crw.workflow import records
from crw.workflow import state as state_mod

COMMANDS = ("run", "runs", "show", "resume", "review", "config", "graph", "uninstall")


def test_version_comes_from_package_metadata(capsys: pytest.CaptureFixture[str]) -> None:
    """--version 输出的版本号取自已安装的发行版元数据。"""
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])

    assert excinfo.value.code == 0
    assert (
        capsys.readouterr().out.strip()
        == f"crw {importlib.metadata.version('code-review-workflow')}"
    )


def test_version_falls_back_when_metadata_is_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """未安装（源码目录直接运行）时给出可辨识占位值，而非误报一个具体版本。"""

    def _missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _missing)

    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])

    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == "crw unknown"


def test_help_lists_every_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    """帮助文本按行列出全部子命令（子串匹配会被 runs 满足 run，故按行锚定）。"""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])

    assert excinfo.value.code == 0
    listed = set(re.findall(r"^ {4}(\S+) {2,}\S", capsys.readouterr().out, flags=re.MULTILINE))
    assert listed >= set(COMMANDS)


def test_missing_subcommand_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])

    assert excinfo.value.code == 2
    assert "usage" in capsys.readouterr().err.lower()


def test_run_defaults() -> None:
    args = build_parser().parse_args(["run"])

    assert args.command == "run"
    assert args.requirement is None
    assert args.file is None
    assert args.max_iter == 3
    assert args.strict is False
    assert args.no_exec is False
    assert args.out is None
    assert args.model is None
    assert args.provider is None
    assert args.no_stream is False


def test_run_accepts_all_documented_options() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "写一个命令行计算器",
            "--max-iter",
            "1",
            "--strict",
            "--no-exec",
            "--out",
            "dist",
            "--model",
            "deepseek-chat",
            "--provider",
            "deepseek",
            "--no-stream",
        ]
    )

    assert args.requirement == "写一个命令行计算器"
    assert args.max_iter == 1
    assert args.strict is True
    assert args.no_exec is True
    assert args.out == "dist"
    assert args.model == "deepseek-chat"
    assert args.provider == "deepseek"
    assert args.no_stream is True


@pytest.mark.parametrize("flag", ["-f", "--file"])
def test_run_accepts_file_input(flag: str) -> None:
    """既有代码文件输入同时支持短名与长名。"""
    args = build_parser().parse_args(["run", flag, "samples/legacy.py"])

    assert args.file == "samples/legacy.py"
    assert args.requirement is None


def test_run_rejects_requirement_and_file_together() -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["run", "需求文本", "-f", "samples/legacy.py"])

    assert excinfo.value.code == 2


@pytest.mark.parametrize("argv", [["run"], ["run", ""], ["run", "   "], ["run", "-f", ""]])
def test_run_requires_non_blank_input(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    """run 的两种输入必须恰好给出一个，且不得为空白（Design §8.1）。"""
    with pytest.raises(SystemExit) as excinfo:
        main(argv)

    assert excinfo.value.code == 2
    assert "二选一" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "1", "3", "10"])
def test_run_max_iter_accepts_values_up_to_the_limit(value: str) -> None:
    """轮次由用户在命令行指定，取值范围 0–10（Design §4.4；0 = 只审不修）。"""
    assert build_parser().parse_args(["run", "--max-iter", value]).max_iter == int(value)


@pytest.mark.parametrize("value", ["11", "100", "999999999"])
def test_run_max_iter_rejects_values_above_the_limit(
    value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """超过硬上限即参数错误（退出码 2），防止无界循环与无界调用费用。"""
    with pytest.raises(SystemExit) as excinfo:
        main(["run", "需求文本", "--max-iter", value])

    assert excinfo.value.code == 2
    assert "最多 10 轮" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["-1", "1.5", "abc", "+3", " 3", "1_0", "\u0663"])
def test_run_max_iter_rejects_non_ascii_or_non_decimal(value: str) -> None:
    """只接受 ASCII 十进制写法：符号前缀、下划线分组、全角/阿拉伯-印度数字均拒绝。"""
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["run", "--max-iter", value])

    assert excinfo.value.code == 2


def test_resume_requires_run_id() -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["resume"])

    assert excinfo.value.code == 2


def test_resume_takes_run_options_without_file_input() -> None:
    args = build_parser().parse_args(
        ["resume", "r-20261011-1a2b", "--max-iter", "2", "--no-stream"]
    )

    assert args.run_id == "r-20261011-1a2b"
    assert args.max_iter == 2
    assert args.no_stream is True
    assert not hasattr(args, "file")


def test_show_run_id_is_optional() -> None:
    assert build_parser().parse_args(["show"]).run_id is None
    assert build_parser().parse_args(["show", "r-20261011-1a2b"]).run_id == "r-20261011-1a2b"


def test_review_accepts_paths_and_options() -> None:
    args = build_parser().parse_args(
        ["review", "src/crw/cli.py", "tests", "-o", "report.md", "--json", "--no-exec"]
    )

    assert args.paths == ["src/crw/cli.py", "tests"]
    assert args.out == "report.md"
    assert args.as_json is True
    assert args.no_exec is True


def test_review_accepts_model_overrides() -> None:
    args = build_parser().parse_args(
        ["review", "src/crw/cli.py", "--model", "some-model", "--provider", "some-provider"]
    )

    assert args.model == "some-model"
    assert args.provider == "some-provider"


def test_review_reads_stdin_marker_as_path() -> None:
    assert build_parser().parse_args(["review", "-"]).paths == ["-"]


def test_review_rejects_blank_paths(capsys: pytest.CaptureFixture[str]) -> None:
    """空白路径不是可审查目标，按参数错误拒绝。"""
    with pytest.raises(SystemExit) as excinfo:
        main(["review", ""])

    assert excinfo.value.code == 2
    assert "空白" in capsys.readouterr().err


def test_review_stdin_marker_must_stand_alone(capsys: pytest.CaptureFixture[str]) -> None:
    """- 读 stdin，与路径互斥（README 命令参考）。"""
    with pytest.raises(SystemExit) as excinfo:
        main(["review", "-", "src/crw/cli.py"])

    assert excinfo.value.code == 2
    assert "stdin" in capsys.readouterr().err


def test_review_requires_at_least_one_path() -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["review"])

    assert excinfo.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["review", "src/crw/cli.py", "--out", "report.md"],
        ["graph", "--out", "graph.mmd"],
    ],
)
def test_short_options_also_accept_long_names(argv: list[str]) -> None:
    """短选项均有等价长名（用户 2026-10-11 决定：全面恢复长名）。"""
    assert build_parser().parse_args(argv).out == argv[-1]


def test_every_short_option_has_a_long_form() -> None:
    """参数面不变量：任何短选项都必须配长名，避免再出现只有 ``-x`` 的选项。"""
    root = build_parser()
    subparser_actions = [a for a in root._actions if isinstance(a, argparse._SubParsersAction)]
    parsers = [root, *subparser_actions[0].choices.values()]

    for parser in parsers:
        for action in parser._actions:
            shorts = [name for name in action.option_strings if not name.startswith("--")]
            longs = [name for name in action.option_strings if name.startswith("--")]
            assert not shorts or longs, f"{parser.prog}：{'/'.join(shorts)} 缺少长名"


def test_config_action_accepts_test_only() -> None:
    assert build_parser().parse_args(["config"]).action is None
    assert build_parser().parse_args(["config", "test"]).action == "test"
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["config", "bogus"])
    assert excinfo.value.code == 2


def test_graph_writes_to_file_only_when_asked() -> None:
    assert build_parser().parse_args(["graph"]).out is None
    assert build_parser().parse_args(["graph", "-o", "graph.mmd"]).out == "graph.mmd"


@pytest.mark.parametrize("command", ["runs", "uninstall"])
def test_commands_without_arguments(command: str) -> None:
    assert build_parser().parse_args([command]).command == command


@pytest.mark.parametrize(
    "argv",
    [
        ["review", "-o"],
        ["run", "-f"],
        ["run", "--max-iter"],
        ["run", "--unknown"],
        ["show", "a", "b"],
        ["runs", "extra"],
        ["config", "test", "extra"],
    ],
)
def test_usage_errors_exit_with_code_2(argv: list[str]) -> None:
    """缺选项值、未知选项与多余位置参数一律按参数错误退出（Design §8.1）。"""
    with pytest.raises(SystemExit) as excinfo:
        main(argv)

    assert excinfo.value.code == 2


@pytest.mark.parametrize(
    ("argv", "command"),
    [
        (["resume", "r-20261011-1a2b"], "resume"),
        (["review", "src/crw/cli.py"], "review"),
        (["config"], "config"),
        (["graph"], "graph"),
        (["uninstall"], "uninstall"),
    ],
)
def test_commands_still_placeholders_report_not_implemented(
    argv: list[str], command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """M1 已实现 run / runs / show；其余命令仍为占位（M2/M3 落地）。"""
    assert main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert command in captured.err


# ---- run / runs / show 实现（Design §8.1）----


def test_run_with_requirement_prints_conclusion_and_persists_record(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """需求文本路径：假节点默认装配即可干净跑通（本轮生效选项 -f / --max-iter）。"""
    assert main(["run", "写一个命令行四则运算计算器"]) == 0

    captured = capsys.readouterr()
    assert "完成：干净通过" in captured.out
    assert "运行记录" in captured.out
    assert "[编码]" in captured.err  # 过程叙述走 stderr（Design §8.4）
    assert "[审查]" in captured.err

    (summary,) = records.list_runs()
    assert summary.status == state_mod.STATUS_SUCCESS
    assert summary.fix_rounds == 0
    assert summary.residual_severe == 0


def test_run_with_syntax_fixture_reports_the_fix_cycle(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "-f", str(sample_path("syntax_error.py"))]) == 0

    captured = capsys.readouterr()
    assert "【严重】" in captured.out  # findings 清单走 stdout
    assert "【修复】f-1 → fixed" in captured.out
    assert "完成：干净通过（共 2 轮审查 / 1 轮修复）" in captured.out
    assert "第 1 轮（code_v1.py）：1 严重" in captured.err

    (summary,) = records.list_runs()
    assert summary.fix_rounds == 1


def test_run_with_divide_by_zero_fixture_residual(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """达限残留：退出码 1 + 残留清单（Design §4.4）。"""
    assert main(["run", "--max-iter", "1", "-f", str(sample_path("divide_by_zero.py"))]) == 1

    captured = capsys.readouterr()
    assert "达修复轮次上限，仍有 1 项【严重】残留" in captured.out

    (summary,) = records.list_runs()
    assert summary.status == state_mod.STATUS_RESIDUAL
    assert summary.residual_severe == 1


@pytest.mark.parametrize(
    ("argv", "expected_source"),
    [
        (["run", "需求", "--max-iter", "2"], state_mod.SOURCE_CLI),
        (["run", "需求"], state_mod.SOURCE_DEFAULT),
    ],
)
def test_run_records_max_iterations_source(
    argv: list[str], expected_source: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """轮次取值来源：命令行显式给出记 cli，否则记内置默认（Design §4.4）。"""
    assert main(argv) == 0
    capsys.readouterr()

    (summary,) = records.list_runs()
    state = records.read_state(summary.path)
    assert state is not None
    assert state["max_iterations_source"] == expected_source


def test_run_reports_unreadable_file_as_run_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "-f", "samples/does-not-exist.py"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "运行失败" in captured.err
    assert "does-not-exist.py" in captured.err or "No such file" in captured.err
    assert records.list_runs() == []  # 输入读不到：不产生运行记录


def test_runs_lists_the_six_columns(capsys: pytest.CaptureFixture[str]) -> None:
    main(["run", "写一个计算器"])
    capsys.readouterr()

    assert main(["runs"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["运行号", "时间", "输入摘要", "状态", "修复轮数", "残留严重数"]
    assert any("写一个计算器" in line for line in lines[1:])  # 数据行（不依赖表头 / 分隔行序）


def test_runs_without_records_is_not_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["runs"]) == 0

    assert "暂无运行记录" in capsys.readouterr().out


def test_show_renders_the_record(capsys: pytest.CaptureFixture[str]) -> None:
    main(["run", "-f", str(sample_path("syntax_error.py"))])
    capsys.readouterr()
    (summary,) = records.list_runs()

    assert main(["show", summary.run_id]) == 0

    text = capsys.readouterr().out
    assert f"── 运行 {summary.run_id}" in text
    assert "[状态] 干净通过" in text
    assert "时间线：" in text and "逐轮审查：" in text
    assert "【严重】code_v1.py:7" in text and "← 处置：fixed" in text
    assert "report.md" in text


def test_show_without_run_id_offers_selection(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["run", "写一个计算器"])
    capsys.readouterr()
    monkeypatch.setattr(crw.render, "ask_user", lambda prompt: "1")

    assert main(["show"]) == 0

    captured = capsys.readouterr()
    assert "干净通过" in captured.out
    assert "1. r-" in captured.err  # 候选清单走 stderr（交互过程）


def test_show_without_run_id_and_cancelled_selection(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["run", "写一个计算器"])
    capsys.readouterr()
    monkeypatch.setattr(crw.render, "ask_user", lambda prompt: None)

    assert main(["show"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "已取消" in captured.err


def test_show_without_records_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["show"]) == 2

    assert "暂无运行记录" in capsys.readouterr().err


def test_show_unknown_run_id_is_readable(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["show", "r-20261012-0001"]) == 2

    assert "运行记录不存在" in capsys.readouterr().err


def test_show_rejects_invalid_run_id_format(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["show", "../secret"]) == 2

    assert "运行号格式非法" in capsys.readouterr().err


def test_run_state_file_records_the_cli_options(capsys: pytest.CaptureFixture[str]) -> None:
    """本轮生效选项（-f / --max-iter）与其余选项的占位口径都写进状态快照。"""
    assert (
        main(["run", "-f", str(sample_path("divide_by_zero.py")), "--max-iter", "0", "--strict"])
        == 1
    )
    capsys.readouterr()

    (summary,) = records.list_runs()
    state = records.read_state(summary.path)
    assert state is not None
    assert state["max_iterations"] == 0
    assert state["strict"] is True
    assert state["input"]["kind"] == state_mod.INPUT_FILE
    assert state["findings"] and state["next_action"] == state_mod.NEXT_FIX_AGAIN


def test_unexpected_failure_reports_readably(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """处理器阶段的未预期异常兜底为可读报错 + 退出码 2，而不是 traceback 退出（Design §8.1）。"""

    def _boom(args: argparse.Namespace) -> int:
        raise RuntimeError("运行目录不可写")

    monkeypatch.setitem(crw.cli._HANDLERS, "runs", _boom)

    assert main(["runs"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "运行失败" in captured.err
    assert "RuntimeError" in captured.err
    assert "运行目录不可写" in captured.err


def test_build_stage_failure_is_covered_too(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """解析 / 校验阶段的未预期异常同样兜底（如发行版元数据损坏）。"""

    def _boom() -> argparse.ArgumentParser:
        raise ValueError("dist-info 损坏")

    monkeypatch.setattr(crw.cli, "build_parser", _boom)

    assert main(["--version"]) == 2
    captured = capsys.readouterr()
    assert "运行失败" in captured.err
    assert "ValueError" in captured.err


def test_failure_message_masks_key_like_text_and_truncates(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """兜底报错遮蔽密钥形态并限长：异常文本可能来自模型 / HTTP 栈且会被重定向保存。"""
    fake_key = "sk-" + "k" * 24

    def _boom(args: argparse.Namespace) -> int:
        raise RuntimeError(f"task-analysis failed: 401 unauthorized, key={fake_key} " + "长" * 500)

    monkeypatch.setitem(crw.cli._HANDLERS, "runs", _boom)

    assert main(["runs"]) == 2
    err = capsys.readouterr().err
    assert fake_key not in err
    assert "sk-***" in err
    assert "task-analysis failed" in err  # 普通连字符词不得被误伤
    assert "已截断" in err


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "写一个计算器"],
        ["run", "-f", "samples/legacy.py"],
        ["review", "src/crw/cli.py", "tests"],
        ["review", "-"],
    ],
)
def test_input_contract_accepts_legal_combinations(argv: list[str]) -> None:
    """合法输入组合不被输入契约校验误伤。"""
    parser = build_parser()

    crw.cli._validate_input_contract(parser, parser.parse_args(argv))


def test_run_reports_directory_input_readably(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``-f`` 目标为目录：给中文可读说明（Design §4.6 可读报错），退出码 2。"""
    assert main(["run", "-f", str(tmp_path)]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "输入是目录" in captured.err
    assert "-f 需要一个代码文件" in captured.err
    assert records.list_runs() == []


def test_run_reports_missing_input_file_readably(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "-f", "no/such/file.py"]) == 2

    assert "输入文件不存在" in capsys.readouterr().err


def test_run_reports_non_utf8_input_readably(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    binary = tmp_path / "sample.py"
    binary.write_bytes(b"\xff\xfe\x00print(1)\n")

    assert main(["run", "-f", str(binary)]) == 2

    assert "不是 UTF-8 文本" in capsys.readouterr().err


def test_run_writes_nothing_to_the_real_user_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """运行记录根目录随 CRW_HOME 走：测试用例不会写真实用户目录（Design §9.1）。"""
    assert main(["run", "写一个计算器"]) == 0
    capsys.readouterr()

    assert records.runs_root().is_relative_to(tmp_path)
