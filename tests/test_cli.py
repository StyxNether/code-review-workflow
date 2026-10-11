"""命令面骨架测试：版本、帮助、参数解析与占位分发（无真实网络调用）。"""

from __future__ import annotations

import argparse
import importlib.metadata
import re

import pytest

import crw.cli
from crw.cli import build_parser, main

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


def test_run_accepts_file_input() -> None:
    args = build_parser().parse_args(["run", "-f", "samples/legacy.py"])

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


def test_run_max_iter_zero_is_allowed() -> None:
    assert build_parser().parse_args(["run", "--max-iter", "0"]).max_iter == 0


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
        (["run", "写一个计算器"], "run"),
        (["runs"], "runs"),
        (["show"], "show"),
        (["resume", "r-20261011-1a2b"], "resume"),
        (["review", "src/crw/cli.py"], "review"),
        (["config"], "config"),
        (["graph"], "graph"),
        (["uninstall"], "uninstall"),
    ],
)
def test_skeleton_commands_report_not_implemented(
    argv: list[str], command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """骨架期八个子命令均已接线：分发到占位处理器并给出可读提示。"""
    assert main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert command in captured.err


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
