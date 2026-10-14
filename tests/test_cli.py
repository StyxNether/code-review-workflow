"""命令面测试：版本、帮助、参数解析、输入契约与 run / runs / show / review / config。

run 路径经假模型驱动真实 Agent 节点——"流程怎么走"由脚本化响应决定，"怎么接起来"走的是
与真实运行完全相同的代码；测试内无任何真实网络调用。
"""

from __future__ import annotations

import argparse
import importlib.metadata
import io
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import fakes
import pytest
from support import read_sample, sample_path

import crw.cli
import crw.render
from crw import config as config_mod
from crw.cli import build_parser, main
from crw.workflow import engine as engine_mod
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
    assert args.max_iter == engine_mod.DEFAULT_MAX_ITERATIONS
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


def test_resume_of_an_unknown_run_reports_readably(capsys: pytest.CaptureFixture[str]) -> None:
    """M3：resume 已落地；记录不存在时给可读原因并退出 2（Design §4.4）。"""
    assert main(["resume", "r-20261011-1a2b"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "运行记录不存在" in captured.err


def test_run_without_configuration_points_to_crw_config(capsys: pytest.CaptureFixture[str]) -> None:
    """未配置供应商：给可读指引并退出 2，不产生运行记录（Design §8.1 的运行错误）。"""
    assert main(["run", "写一个计算器"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "尚未配置模型供应商" in captured.err
    assert "crw config" in captured.err
    assert records.list_runs() == []


# ---- run：真实 Agent 节点（经假模型驱动，Design §8.1）----

#: 假模型给编码节点的产出（可编译的最小模块）
CLEAN_CODE = '"""计算器。"""\n\n\ndef main() -> None:\n    print("ok")\n'


def test_run_with_requirement_prints_conclusion_and_persists_record(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """需求文本路径：编码 → 审查 →（干净）收尾，产物与运行记录落盘。"""
    configured_provider()
    fake_chat_model.load(fakes.clean_requirement_script(code=CLEAN_CODE))

    assert main(["run", "写一个命令行四则运算计算器"]) == 0

    captured = capsys.readouterr()
    assert "完成：干净通过" in captured.out
    assert "运行记录" in captured.out
    assert "[编码]" in captured.err  # 过程叙述走 stderr（Design §8.4）
    assert "[审查]" in captured.err
    assert "· 工具：" in captured.err  # 工具调用过程可见（阶段面板下的打点行）

    (summary,) = records.list_runs()
    assert summary.status == state_mod.STATUS_SUCCESS
    assert summary.fix_rounds == 0
    assert summary.residual_severe == 0
    state = records.read_state(summary.path)
    assert state is not None
    assert state["model"] == "DeepSeek/deepseek-flash"  # 真实模型标识（Design §4.3）


def test_run_records_tool_events_in_the_log(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
) -> None:
    """工具调用过程进 log.jsonl（状态追踪与演示的时间线来源，Design §5.1）。"""
    configured_provider()
    fake_chat_model.load(fakes.clean_requirement_script(code=CLEAN_CODE))

    assert main(["run", "写一个计算器"]) == 0

    (summary,) = records.list_runs()
    events = records.read_events(summary.path / "log.jsonl")
    tools = [event for event in events if event["event"] == "tool"]
    assert tools and all(event["node"] == "review" for event in tools)
    assert "read_file" in tools[0]["detail"]
    assert tools[0].get("iteration") == 1  # 工具事件归属审查第 1 轮


def test_run_with_syntax_fixture_reports_the_fix_cycle(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """修复循环：首审含【严重】→ 修复 → 复审干净（退出码 0）。"""
    configured_provider()
    fake_chat_model.load(
        fakes.fix_cycle_script(
            code=read_sample("syntax_error.py"), fixed_code=CLEAN_CODE, name="syntax_error.py"
        )
    )

    assert main(["run", "-f", str(sample_path("syntax_error.py"))]) == 0

    captured = capsys.readouterr()
    assert "【严重】" in captured.out  # findings 清单走 stdout
    assert "f-1 → 已修复" in captured.out
    assert "完成：干净通过（共 2 轮审查 / 1 轮修复）" in captured.out
    assert "第 1 轮（code_v1/，1 个文件）：1 严重" in captured.err

    (summary,) = records.list_runs()
    assert summary.fix_rounds == 1


def test_run_with_divide_by_zero_fixture_residual(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """达限残留：退出码 1 + 残留清单（Design §4.4）。"""
    configured_provider()
    fake_chat_model.load(
        fakes.residual_script(
            code=read_sample("divide_by_zero.py"),
            name="divide_by_zero.py",
            location="divide_by_zero.py:7",
        )
    )

    assert main(["run", "--max-iter", "1", "-f", str(sample_path("divide_by_zero.py"))]) == 1

    captured = capsys.readouterr()
    assert "达修复轮次上限，仍有 1 项【严重】残留" in captured.out

    (summary,) = records.list_runs()
    assert summary.status == state_mod.STATUS_RESIDUAL
    assert summary.residual_severe == 1


def test_review_without_any_tool_call_is_rejected(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """执行有效性门禁（Design §5.2）：审查一次工具都没调、却报"无问题" → 不许通过。

    首轮与重试都没有工具事件 → 按节点失败处理（退出码 2），而不是把空数组当"干净"。
    """
    configured_provider()
    fake_chat_model.load(
        fakes.requirement_input_with_clean_reviews(code=CLEAN_CODE, reviews=2, with_tools=False)
    )

    assert main(["run", "写一个计算器"]) == 2

    captured = capsys.readouterr()
    assert "运行失败" in captured.err
    assert "执行证据" in captured.err or "审查未执行" in captured.err
    (summary,) = records.list_runs()
    state = records.read_state(summary.path)
    assert state is not None and state["status"] == state_mod.STATUS_FAILED


def test_run_no_exec_does_not_register_the_execution_tool(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
) -> None:
    """``--no-exec``：工具注册层不注册 run_python（模型侧根本看不到该工具，Design §8.1）。"""
    configured_provider()
    fake_chat_model.load(fakes.clean_requirement_script(code=CLEAN_CODE))

    assert main(["run", "写一个计算器", "--no-exec"]) == 0

    bound = [name for tools in fake_chat_model.bound_tools for name in tools]
    assert bound  # 确实发生过工具绑定
    assert "run_python" not in bound
    assert "read_file" in bound


def test_run_streams_model_text_and_no_stream_buffers_it(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--no-stream`` 不丢内容：两种模式都输出模型正文（围栏内容按 §6.2 #6 抑制）。

    流式与缓存的**差别**（立即写出 / 阶段结束整体写出）由 tests/test_render.py 的渲染器用例
    锁定——在整段 stderr 文本上两者的行序相同，只有写入时机不同，这里只保证内容可读。
    """
    configured_provider()
    fake_chat_model.load(fakes.clean_requirement_script(code=CLEAN_CODE))

    assert main(["run", "写一个计算器"]) == 0
    streamed = capsys.readouterr().err
    assert "实现说明" in streamed  # 模型正文（编码节点的说明段）

    fake_chat_model.load(fakes.clean_requirement_script(code=CLEAN_CODE))
    assert main(["run", "写一个计算器", "--no-stream"]) == 0
    buffered = capsys.readouterr().err

    assert "实现说明" in buffered  # 正文内容不丢
    assert "已收到代码块" in buffered  # 代码围栏只留形状摘要（不刷屏）
    assert "已收到 JSON 结论" in buffered  # 审查契约同理：校验后按分级清单呈现
    assert (
        "[编码] 生成 code_v1/（1 个文件" in buffered
    )  # 阶段摘要行在正文之后（阶段收尾时才落这一行）


def test_run_out_copies_final_code_and_report(
    tmp_path: Path,
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--out``：最终版本目录整目录复制为 code_final/，报告一并复制（Design §4.2.2）。"""
    configured_provider()
    fake_chat_model.load(fakes.clean_requirement_script(code=CLEAN_CODE))
    out_dir = tmp_path / "deliver"

    assert main(["run", "写一个计算器", "--out", str(out_dir)]) == 0

    assert (out_dir / "code_final" / "main.py").read_text(encoding="utf-8") == CLEAN_CODE.strip(
        "\n"
    ) + "\n"
    assert (out_dir / "code_final" / "manifest.json").is_file()
    assert "结论" in (out_dir / "report.md").read_text(encoding="utf-8")
    assert "已复制" in capsys.readouterr().err


def test_run_out_avoids_overwriting_existing_outputs(
    tmp_path: Path,
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
) -> None:
    """重名加序号：已有 code_final/ 时写到 code_final_2/，不覆盖用户文件（Design §4.5）。"""
    configured_provider()
    out_dir = tmp_path / "deliver"
    out_dir.mkdir()
    (out_dir / "code_final").mkdir()
    (out_dir / "code_final" / "keep.txt").write_text("# 用户自己的文件\n", encoding="utf-8")

    fake_chat_model.load(fakes.clean_requirement_script(code=CLEAN_CODE))
    assert main(["run", "写一个计算器", "--out", str(out_dir)]) == 0

    assert (out_dir / "code_final" / "keep.txt").read_text(encoding="utf-8") == "# 用户自己的文件\n"
    assert (out_dir / "code_final_2" / "main.py").is_file()
    assert (out_dir / "report_2.md").is_file()


@pytest.mark.parametrize(
    ("argv", "expected_source"),
    [
        (["run", "需求", "--max-iter", "2"], state_mod.SOURCE_CLI),
        (["run", "需求"], state_mod.SOURCE_DEFAULT),
    ],
)
def test_run_records_max_iterations_source(
    argv: list[str],
    expected_source: str,
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """轮次取值来源：命令行显式给出记 cli，否则记内置默认（Design §4.4）。"""
    configured_provider()
    fake_chat_model.load(fakes.clean_requirement_script(code=CLEAN_CODE))

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


def test_runs_lists_the_designed_columns(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configured_provider()
    fake_chat_model.load(fakes.clean_requirement_script(code=CLEAN_CODE))
    main(["run", "写一个计算器"])
    capsys.readouterr()

    assert main(["runs"]) == 0

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
    assert any("写一个计算器" in line for line in lines[1:])  # 数据行（不依赖表头 / 分隔行序）


def test_runs_without_records_is_not_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["runs"]) == 0

    assert "暂无运行记录" in capsys.readouterr().out


def test_show_renders_the_record(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configured_provider()
    fake_chat_model.load(
        fakes.fix_cycle_script(
            code=read_sample("syntax_error.py"),
            fixed_code=CLEAN_CODE,
            name="syntax_error.py",
            location="syntax_error.py:7",
        )
    )
    main(["run", "-f", str(sample_path("syntax_error.py"))])
    capsys.readouterr()
    (summary,) = records.list_runs()

    assert main(["show", summary.run_id]) == 0

    text = capsys.readouterr().out
    assert f"── 运行 {summary.run_id}" in text
    assert "[状态] 干净通过" in text
    assert "时间线：" in text and "逐轮审查：" in text
    assert "【严重】syntax_error.py:7" in text and "← 处置：已修复" in text
    assert "report.md" in text


def test_show_without_run_id_offers_selection(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configured_provider()
    fake_chat_model.load(fakes.clean_requirement_script(code=CLEAN_CODE))
    main(["run", "写一个计算器"])
    capsys.readouterr()
    monkeypatch.setattr(crw.render, "ask_user", lambda prompt: "1")

    assert main(["show"]) == 0

    captured = capsys.readouterr()
    assert "干净通过" in captured.out
    assert "1. r-" in captured.err  # 候选清单走 stderr（交互过程）


def test_show_without_run_id_and_cancelled_selection(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configured_provider()
    fake_chat_model.load(fakes.clean_requirement_script(code=CLEAN_CODE))
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


def test_run_state_file_records_the_cli_options(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """选项落进状态快照：``--max-iter 0``（只审不修）与 ``--strict``。"""
    configured_provider()
    fake_chat_model.load(
        [
            fakes.ai_tool_call("read_file", path="code_v1/divide_by_zero.py"),
            fakes.ai_text(
                fakes.review_fence(
                    [fakes.finding(location="divide_by_zero.py:7")], files=["divide_by_zero.py"]
                )
            ),
        ]
    )

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
    tmp_path: Path,
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """运行记录根目录随 CRW_HOME 走：测试用例不会写真实用户目录（Design §9.1）。"""
    configured_provider()
    fake_chat_model.load(fakes.clean_requirement_script(code=CLEAN_CODE))

    assert main(["run", "写一个计算器"]) == 0
    capsys.readouterr()

    (summary,) = records.list_runs()
    assert summary.path.is_relative_to(tmp_path)


# ---- crw review：单 Agent 审查（Design §8.1 / §6.4）----


def test_review_file_prints_conclusion_and_never_creates_a_run(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """文本模式：结论走 stdout、过程走 stderr，且**不产生运行记录**。"""
    configured_provider()
    fake_chat_model.load(
        [
            fakes.ai_text(
                "【严重】sample.py:1 — 示例问题。\n修改建议：修正它。\n\n整体评价：需要修复。"
            )
        ]
    )

    assert main(["review", str(sample_path("divide_by_zero.py"))]) == 0

    captured = capsys.readouterr()
    assert "【严重】" in captured.out
    assert "正在审查" in captured.err
    assert records.list_runs() == []


def test_review_json_mode_reports_severe_with_exit_code_1(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--json``：结构化输出走 stdout，存在【严重】时退出码 1（Design §8.1）。"""
    configured_provider()
    report = {
        "files": [
            {
                "file": "sample.py",
                "findings": [fakes.finding(location="sample.py:3")],
                "summary": "存在除零风险。",
            }
        ],
        "summary": "存在除零风险。",
    }
    fake_chat_model.load([fakes.ai_text(json.dumps(report, ensure_ascii=False))])

    assert main(["review", "--json", str(sample_path("divide_by_zero.py"))]) == 1

    captured = capsys.readouterr()
    assert json.loads(captured.out)["files"][0]["findings"][0]["severity"] == "严重"


def test_review_json_mode_clean_exits_zero(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--json`` 且无【严重】：退出码 0；顶层 summary 缺省时按文件摘要补全。"""
    configured_provider()
    report = {"files": [{"file": "sample.py", "findings": [], "summary": "未见阻塞问题。"}]}
    fake_chat_model.load([fakes.ai_text(json.dumps(report, ensure_ascii=False))])

    assert main(["review", "--json", str(sample_path("divide_by_zero.py"))]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["files"][0]["findings"] == []
    assert payload["summary"] == "未见阻塞问题。"


def test_review_writes_markdown_report(
    tmp_path: Path,
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``-o``：Markdown 报告落盘（stdout 仍出结论，报告路径提示走 stderr）。"""
    configured_provider()
    fake_chat_model.load([fakes.ai_text("整体评价：可以接受。")])
    target = tmp_path / "report.md"

    assert main(["review", "-o", str(target), str(sample_path("syntax_error.py"))]) == 0

    text = target.read_text(encoding="utf-8")
    assert text.startswith("# crw 审查报告")
    assert "整体评价：可以接受。" in text
    assert "报告已写入" in capsys.readouterr().err


def test_review_directory_expands_first_level_text_files(
    tmp_path: Path,
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """目录入参：展开第一层文本文件，跳过隐藏文件与二进制文件（Design §8.1）。"""
    configured_provider()
    (tmp_path / "a.py").write_text("print(1)\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("print(2)\n", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=1\n", encoding="utf-8")
    (tmp_path / "bin.dat").write_bytes(b"\x00\x01\x02")
    fake_chat_model.load([fakes.ai_text("审查结论一。"), fakes.ai_text("审查结论二。")])

    assert main(["review", str(tmp_path)]) == 0

    captured = capsys.readouterr()
    assert "a.py" in captured.out and "b.py" in captured.out  # 多文件逐个带标题
    assert "审查结论一。" in captured.out and "审查结论二。" in captured.out
    assert "SECRET=1" not in captured.out and "SECRET=1" not in captured.err
    assert "已跳过二进制文件" in captured.err
    assert len(fake_chat_model.requests) == 2


def test_review_reads_stdin(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``-``：读 stdin（Design §8.1）。"""
    configured_provider()
    fake_chat_model.load([fakes.ai_text("stdin 内容的审查结论。")])
    monkeypatch.setattr("sys.stdin", io.StringIO("print('hi')\n"))

    assert main(["review", "-"]) == 0

    assert "stdin 内容的审查结论。" in capsys.readouterr().out


def test_review_no_exec_hides_the_execution_tool(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
) -> None:
    """``--no-exec``：审查侧同样不注册 run_python。"""
    configured_provider()
    fake_chat_model.load([fakes.ai_text("结论。")])

    assert main(["review", "--no-exec", str(sample_path("syntax_error.py"))]) == 0

    bound = [name for tools in fake_chat_model.bound_tools for name in tools]
    assert bound and "run_python" not in bound


def test_review_missing_path_is_a_run_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["review", "no/such/file.py"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "路径不存在" in captured.err


def test_review_reports_model_failure_with_exit_code_2(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """单文件审查失败：stderr 可读报错 + 退出码 2。"""
    configured_provider()
    fake_chat_model.load([RuntimeError("模型连接中断")])

    assert main(["review", str(sample_path("syntax_error.py"))]) == 2

    assert "审查失败" in capsys.readouterr().err


# ---- crw config：供应商配置与连通性自检（Design §8.3）----


def test_config_test_without_providers_is_a_config_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["config", "test"]) == 2

    assert "尚无供应商配置" in capsys.readouterr().err


def test_config_test_reports_each_provider(
    configured_provider: Callable[[], dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``config test``：逐供应商探测，全部 OK 退出 0（探测结果经桩注入，无真实网络）。"""
    configured_provider()
    monkeypatch.setattr(config_mod, "test_provider", lambda entry: ("ok", ""))

    assert main(["config", "test"]) == 0

    assert "DeepSeek（deepseek-flash）：OK" in capsys.readouterr().out


def test_config_test_reports_failures_with_exit_code_1(
    configured_provider: Callable[[], dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configured_provider()
    monkeypatch.setattr(config_mod, "test_provider", lambda entry: ("auth", "401 unauthorized"))

    assert main(["config", "test"]) == 1

    out = capsys.readouterr().out
    assert "401（key 无效）" in out
    assert "401 unauthorized" in out


def test_config_interactive_adds_a_provider_and_writes_models_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """交互式配置：选预设 → 用预填清单 → key 存明文 → 设默认 → 退出，落盘可解析。"""
    answers = iter(
        [
            "1",  # 菜单：添加 / 更新供应商
            "2",  # 供应商类型：DeepSeek（第 1 项是自定义）
            "",  # 接入点：回车用官方
            "",  # 模型清单：用预填
            "placeholder-key",  # API key
            "1",  # key 存储方式：明文写入 models.json
            "4",  # 菜单：设置默认模型
            "1",  # 供应商
            "1",  # 模型
            "0",  # 退出
        ]
    )
    monkeypatch.setattr(crw.cli, "_ask", lambda prompt: next(answers, None))
    monkeypatch.setattr(crw.cli, "_read_key", lambda prompt: next(answers, ""))

    assert main(["config"]) == 0

    out = capsys.readouterr().out
    assert "供应商 'DeepSeek' 已添加。" in out
    data = config_mod.load_raw()
    assert data["default"] == "DeepSeek/deepseek-flash"
    entry = data["providers"][0]
    assert entry["base_url"] == "https://api.deepseek.com"
    assert entry["api_key"] == "placeholder-key"
    assert "placeholder-key" not in out  # key 不回显（菜单只打码展示）


def test_config_interactive_reports_a_broken_file(capsys: pytest.CaptureFixture[str]) -> None:
    """损坏的 models.json：给可读原因 + 修复指引，退出 2（Design §8.1）。"""
    config_mod.config_path().parent.mkdir(parents=True, exist_ok=True)
    config_mod.config_path().write_text("{ 不是 JSON", encoding="utf-8")

    assert main(["config"]) == 2

    captured = capsys.readouterr()
    assert "配置错误" in captured.err  # 错误走 stderr（与 config test 同口径，Design §8.4）
    assert "手工修复" in captured.err
    assert "配置错误" not in captured.out


# ---- config 交互的取消语义与边界（自审问题回归）----


def test_config_add_provider_is_cancelled_when_key_input_is_interrupted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """key 输入被中断（``None``）≠ 空 key：整个添加取消，不留下半成品供应商。"""
    answers = iter(["1", "2", "", ""])
    monkeypatch.setattr(crw.cli, "_ask", lambda prompt: next(answers, None))
    monkeypatch.setattr(crw.cli, "_read_key", lambda prompt: None)

    assert main(["config"]) == 0

    out = capsys.readouterr().out
    assert "未添加该供应商" in out
    assert config_mod.load_raw().get("providers") in (None, [], {})


def test_config_key_storage_choice_reasks_on_invalid_input(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """key 存储方式输错时重新询问，不把笔误静默当成"明文落盘"。"""
    answers = iter(
        [
            "1",  # 添加供应商
            "2",  # DeepSeek 预设
            "",  # 接入点默认
            "",  # 用预填模型清单
            "placeholder-key",  # key 明文（_read_key 走同一迭代器）
            "3",  # 无效选择 → 应重问
            "1",  # 明文写入
            "0",  # 退出
        ]
    )
    monkeypatch.setattr(crw.cli, "_ask", lambda prompt: next(answers, None))
    monkeypatch.setattr(crw.cli, "_read_key", lambda prompt: next(answers, None))

    assert main(["config"]) == 0

    out = capsys.readouterr().out
    assert "无效选择，请重新输入。" in out
    assert config_mod.load_raw()["providers"][0]["api_key"] == "placeholder-key"


def test_config_add_model_reports_an_existing_model_without_a_fake_overwrite_prompt(
    configured_provider: Callable[[], dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configured_provider()
    answers = iter(
        ["3", "1", "1", "deepseek-flash", "0", "0"]
    )  # 菜单 3 → 供应商 1 → 添加模型 → 重名 → 返回 → 退出
    monkeypatch.setattr(crw.cli, "_ask", lambda prompt: next(answers, None))

    assert main(["config"]) == 0

    out = capsys.readouterr().out
    assert "未重复添加" in out
    assert config_mod.load_raw()["providers"][0]["models"] == ["deepseek-flash"]


def test_config_errors_are_masked(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """配置错误的文本同样遮蔽密钥形态（错误通道统一口径，终端输出常被重定向保存）。"""
    fake_key = "sk-" + "k" * 24
    monkeypatch.setattr(
        config_mod,
        "resolve",
        lambda **kwargs: (_ for _ in ()).throw(
            config_mod.ConfigError(f"供应商条目损坏，疑似含 {fake_key}")
        ),
    )

    assert main(["run", "写一个计算器"]) == 2

    err = capsys.readouterr().err
    assert fake_key not in err
    assert "sk-***" in err


def test_review_stdin_is_size_limited(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``crw review -`` 的 stdin 读取同样有上限（与单文件同口径，防内存无界）。"""
    monkeypatch.setattr(crw.tools.fs, "MAX_READ_BYTES", 16)
    monkeypatch.setattr("sys.stdin", io.StringIO("x" * 100))

    assert main(["review", "-"]) == 2

    assert "标准输入超过读取上限" in capsys.readouterr().err


def test_run_out_without_artifacts_reports_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--out`` 但运行没有产物时给出提示，而不是静默跳过。"""
    crw.cli._export_outputs(engine_result_without_dir(), Path("unused"))  # type: ignore[arg-type]

    assert "--out 已跳过" in capsys.readouterr().err


def engine_result_without_dir() -> object:
    """一个 run_dir 为 None 的结果视图（模拟目录都无法创建的情形）。"""
    from crw.workflow import engine as engine_mod

    return engine_mod.RunResult(
        run_id="",
        run_dir=None,
        status="failed",
        exit_code=2,
        review_round=0,
        fix_rounds=0,
        clean=False,
        findings=(),
        error_node="engine",
        error_message="运行目录创建失败",
        interrupted=False,
        state={},
    )


# ---- M3.5：旧单文件记录守卫与思考挡位回显（Design §4.2.2、§8.3） ----


def _write_legacy_record(run_id: str = "r-20261012-0001") -> str:
    """造一条旧单文件格式的运行记录（带 current_code、无 current_version_dir）。"""
    run_dir = records.create_run_dir(run_id)
    state = {
        "run_id": run_id,
        "input": {"kind": "file", "source_path": "legacy.py"},
        "workspace": str(run_dir / "workspace"),
        "review_round": 1,
        "fix_rounds": 0,
        "max_iterations": 3,
        "current_code": str(run_dir / "workspace" / "code_v1.py"),
        "findings": [],
        "status": "success",
        "model": "DeepSeek/deepseek-flash",
        "created_at": "2026-10-12T09:00:00+08:00",
        "finished_at": "2026-10-12T09:01:00+08:00",
    }
    records.atomic_write_text(run_dir / "state.json", json.dumps(state, ensure_ascii=False))
    return run_id


def test_show_rejects_legacy_single_file_records(capsys: pytest.CaptureFixture[str]) -> None:
    """旧格式记录不做兼容：一行可读提示 + 退出码 2，不崩溃（用户裁决 2026-10-13）。"""
    run_id = _write_legacy_record()

    assert main(["show", run_id]) == 2

    captured = capsys.readouterr()
    assert "单文件格式" in captured.err and "无法" not in "".join(captured.out)


def test_resume_rejects_legacy_single_file_records(
    configured_provider: Callable[[], dict[str, object]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    configured_provider()
    run_id = _write_legacy_record()

    assert main(["resume", run_id]) == 2

    captured = capsys.readouterr()
    assert "单文件格式" in captured.err
    assert len(records.list_runs()) == 1  # 未新建运行


def test_run_with_thinking_override_echoes_it_everywhere(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--thinking`` 快捷切换：头行、state.model、报告口径一致（Design §8.3 的回显要求）。"""
    configured_provider()
    fake_chat_model.load(fakes.clean_requirement_script(code=CLEAN_CODE))

    assert main(["run", "写一个计算器", "--thinking", "high"]) == 0

    captured = capsys.readouterr()
    assert "[模型] DeepSeek/deepseek-flash（思考：high）" in captured.err
    (summary,) = records.list_runs()
    state = records.read_state(summary.path)
    assert state is not None
    assert state["model"] == "DeepSeek/deepseek-flash（思考：high）"
    assert summary.model == "DeepSeek/deepseek-flash（思考：high）"
    report = (summary.path / "report.md").read_text(encoding="utf-8")
    assert "| 模型 | DeepSeek/deepseek-flash（思考：high） |" in report


def test_thinking_unknown_level_is_a_parameter_error(
    configured_provider: Callable[[], dict[str, object]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """表外挡位 = 参数错误（退出码 2，列出可用挡位），不静默夹取（Design §8.3）。"""
    configured_provider()

    with pytest.raises(SystemExit) as excinfo:
        main(["run", "写一个计算器", "--thinking", "turbo"])

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "未知思考挡位" in captured.err
    assert "budget:<n>" in captured.err and "high" in captured.err  # 可用挡位清单
    assert records.list_runs() == []


def test_config_menu_sets_model_thinking_levels(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """菜单 6「模型思考挡位」：勾选挡位 + 设默认，写回 models.json 的对象条目形态。"""
    answers = iter(
        [
            "1",  # 菜单：添加供应商
            "2",  # 类型：DeepSeek
            "",  # 接入点：官方
            "",  # 模型清单：用预填
            "placeholder-key",
            "1",  # key 存明文
            "6",  # 菜单：模型思考挡位
            "1",  # 供应商
            "1",  # 模型
            "off,low,high",  # 勾选挡位
            "high",  # 默认挡位
            "0",  # 退出
        ]
    )
    monkeypatch.setattr(crw.cli, "_ask", lambda prompt: next(answers, None))
    monkeypatch.setattr(crw.cli, "_read_key", lambda prompt: next(answers, ""))

    assert main(["config"]) == 0

    data = config_mod.load_raw()
    models = data["providers"][0]["models"]
    assert models[0] == {
        "name": "deepseek-flash",
        "thinking": {"levels": ["off", "low", "high"], "selected": "high"},
    }


def test_config_menu_thinking_clears_when_told_to(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """菜单 6 输入"清空"：条目退回纯字符串形态（不发送任何思考参数）。"""
    data: dict = {
        "providers": [
            {
                "name": "DeepSeek",
                "base_url": "https://api.deepseek.com",
                "api_key": "placeholder-key",
                "models": [
                    {"name": "deepseek-flash", "thinking": {"levels": ["high"], "selected": "high"}}
                ],
            }
        ],
    }
    config_mod.save_raw(data)
    answers = iter(["6", "1", "1", "清空", ""])
    monkeypatch.setattr(crw.cli, "_ask", lambda prompt: next(answers, None))
    monkeypatch.setattr(crw.cli, "_read_key", lambda prompt: next(answers, ""))

    assert main(["config"]) == 0

    reloaded = config_mod.load_raw()
    assert reloaded["providers"][0]["models"] == ["deepseek-flash"]


def test_run_out_is_skipped_when_the_run_produced_no_version(
    tmp_path: Path,
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """失败运行没有版本目录：--out 不得把当前工作目录当产物复制（Path("") 退化回归）。"""
    configured_provider()
    fake_chat_model.load([fakes.ai_text("没有任何围栏的坏输出"), fakes.ai_text("还是坏输出")])
    out_dir = tmp_path / "deliver"

    assert main(["run", "写一个计算器", "--out", str(out_dir)]) == 2

    # 产物目录里只应出现报告这一项：不得把当前工作目录（仓库）当版本复制进去
    assert not (out_dir / "code_final").exists()
    assert sorted(path.name for path in out_dir.iterdir()) == ["report.md"]
    assert "已复制" in capsys.readouterr().err


def test_run_file_input_respects_the_read_limit(
    tmp_path: Path,
    configured_provider: Callable[[], dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """-f 输入与 crw review 同口径的读取上限（超限给可读报错，不整文件进内存）。"""
    import crw.tools.fs

    configured_provider()
    big = tmp_path / "big.py"
    big.write_text("# pad\n" * 200, encoding="utf-8")
    monkeypatch.setattr(crw.tools.fs, "MAX_READ_BYTES", 100)

    assert main(["run", "-f", str(big)]) == 2

    assert "读取上限" in capsys.readouterr().err


def test_version_copy_sources_follow_the_manifest(tmp_path: Path) -> None:
    """``--out`` 复制版本目录时以 manifest 为准：执行副产物（__pycache__ / 数据文件）
    不得进交付产物（真实演示暴露：--out 曾把 __pycache__ 一并复制）。"""
    version_dir = tmp_path / "code_v1"
    version_dir.mkdir()
    (version_dir / "main.py").write_text("print(1)\n", encoding="utf-8")
    (version_dir / "notes.txt").write_text("副产物\n", encoding="utf-8")
    (version_dir / "__pycache__").mkdir()
    (version_dir / "__pycache__" / "x.pyc").write_bytes(b"\x00")
    (version_dir / "manifest.json").write_text(
        json.dumps({"version": 1, "files": [{"path": "main.py", "sha256": "x", "lines": 1}]}),
        encoding="utf-8",
    )

    sources = crw.cli._version_copy_sources(version_dir)
    assert sorted(path.name for path in sources) == ["main.py", "manifest.json"]

    # manifest 缺失（旧目录 / 手工介入）：退回全目录复制，仍排除字节码
    (version_dir / "manifest.json").unlink()
    fallback = crw.cli._version_copy_sources(version_dir)
    assert sorted(path.name for path in fallback) == ["main.py", "notes.txt"]


# ---- 用量采集：state.json / report.md / crw show / crw review ----


def usage_script(*, name: str = "main.py") -> list[Any]:
    """干净通过路径的响应脚本，每次调用都带 usage（用量采集用例的数据源）。

    三个数（800+900+1100 输入、300+30+80 输出、0+512+512 命中）刻意互不相同，
    断言才能证明"按节点归账"与"运行级合计"两件事都真的成立。
    """
    return [
        fakes.ai_text(
            fakes.file_fence(CLEAN_CODE, name),
            usage=fakes.call_usage(
                input_tokens=800, output_tokens=300, cached_tokens=0, reasoning_tokens=120
            ),
        ),
        fakes.ai_tool_call(
            "read_file",
            path=f"code_v1/{name}",
            usage=fakes.call_usage(input_tokens=900, output_tokens=30, cached_tokens=512),
        ),
        fakes.ai_text(
            fakes.review_fence([], "已读取完整文件，未发现阻塞问题。", files=[name]),
            usage=fakes.call_usage(input_tokens=1100, output_tokens=80, cached_tokens=512),
        ),
    ]


def _run_with_usage(fake_chat_model: object, configured_provider: Callable[[], dict]) -> str:
    """跑一次干净通过的需求运行，返回运行号。"""
    configured_provider()
    fake_chat_model.load(usage_script())
    assert main(["run", "写一个计算器"]) == 0
    (summary,) = records.list_runs()
    return summary.run_id


def test_run_writes_token_usage_into_state_json(
    configured_provider: Callable[[], dict[str, object]], fake_chat_model: object
) -> None:
    """按节点归账进 state.json：编码 1 次调用、审查 2 次（工具轮 + 收官轮）。"""
    run_id = _run_with_usage(fake_chat_model, configured_provider)

    state = records.read_state(records.find_run_dir(run_id))
    assert state is not None
    usage = state["usage"]
    assert usage["encode"] == {
        "calls": 1,
        "with_usage": 1,
        "input_tokens": 800,
        "output_tokens": 300,
        "reasoning_tokens": 120,
        "cached_tokens": 0,
        "cache_reported": 1,
    }
    assert usage["review"]["calls"] == 2
    assert usage["review"]["input_tokens"] == 2000
    assert usage["review"]["cached_tokens"] == 1024


def test_run_writes_per_call_usage_events_to_the_log(
    configured_provider: Callable[[], dict[str, object]], fake_chat_model: object
) -> None:
    """每次调用一条 usage 事件（与工具事件同一条时间线，供逐调用归因）。"""
    run_id = _run_with_usage(fake_chat_model, configured_provider)

    events = records.read_events(records.find_run_dir(run_id) / "log.jsonl")
    calls = [event for event in events if event["event"] == "usage"]
    assert [event["node"] for event in calls] == ["encode", "review", "review"]
    assert json.loads(calls[1]["detail"]) == {
        "input_tokens": 900,
        "output_tokens": 30,
        "reasoning_tokens": 0,
        "cached_tokens": 512,
    }
    assert calls[2]["iteration"] == 1  # 引擎按当前节点轮次补 iteration（工具事件同口径）


def test_run_report_has_the_usage_table_with_a_total_row(
    configured_provider: Callable[[], dict[str, object]], fake_chat_model: object
) -> None:
    """report.md 的 Token 用量小节：逐节点一行 + 合计行（命中率 = 命中/输入）。"""
    run_id = _run_with_usage(fake_chat_model, configured_provider)

    report = (records.find_run_dir(run_id) / "report.md").read_text(encoding="utf-8")
    assert "## Token 用量" in report
    assert "| 节点 | 调用 | 输入 | 缓存命中 | 输出 | 其中思考 |" in report
    assert "| **合计** | 3 | 2,800 | 1,024（37%） | 410 | 120 |" in report


def test_show_prints_the_usage_line(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_id = _run_with_usage(fake_chat_model, configured_provider)
    capsys.readouterr()

    assert main(["show", run_id]) == 0

    text = capsys.readouterr().out
    assert "[用量] 输入 2,800 · 输出 410（思考 120） · 缓存命中 1,024（37%） · 3 次调用" in text


def test_show_recovers_usage_from_the_log_when_state_is_broken(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """state.json 损坏（降级展示）时用量仍能回读：数据源退到 log.jsonl。"""
    run_id = _run_with_usage(fake_chat_model, configured_provider)
    run_dir = records.find_run_dir(run_id)
    (run_dir / "state.json").write_text("{ 坏掉的快照", encoding="utf-8")
    capsys.readouterr()

    assert main(["show", run_id]) == 2

    text = capsys.readouterr().out
    assert "[用量] 输入 2,800" in text
    assert "3 次调用" in text


def test_failed_node_still_records_the_usage_it_spent(
    configured_provider: Callable[[], dict[str, object]], fake_chat_model: object
) -> None:
    """节点失败：失败前花掉的用量照记——钱已经花了，账必须留在记录里。"""
    configured_provider()
    fake_chat_model.load(
        [
            fakes.ai_text(
                fakes.file_fence(CLEAN_CODE),
                usage=fakes.call_usage(input_tokens=800, output_tokens=300),
            ),
            # 两次都不是契约要求的 json 围栏 → 审查节点按失败收尾（Design §4.6）
            fakes.ai_text(
                "我要开始审查了。", usage=fakes.call_usage(input_tokens=900, output_tokens=40)
            ),
            fakes.ai_text(
                "我还是没给出 JSON。", usage=fakes.call_usage(input_tokens=950, output_tokens=50)
            ),
        ]
    )

    assert main(["run", "写一个计算器"]) == 2

    (summary,) = records.list_runs()
    state = records.read_state(summary.path)
    assert state is not None
    assert state["status"] == state_mod.STATUS_FAILED
    assert state["usage"]["encode"]["calls"] == 1
    assert state["usage"]["review"] == {
        "calls": 2,
        "with_usage": 2,
        "input_tokens": 1850,
        "output_tokens": 90,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "cache_reported": 0,
    }


def test_review_command_reports_a_usage_line(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``crw review`` 不产生运行记录，用量在命令结束时以一行回显（stderr）。"""
    configured_provider()
    fake_chat_model.load(
        [
            fakes.ai_text(
                "【建议】divide_by_zero.py:1 — 示例",
                usage=fakes.call_usage(input_tokens=1500, output_tokens=200, cached_tokens=1024),
            )
        ]
    )

    assert main(["review", str(sample_path("divide_by_zero.py"))]) == 0

    err = capsys.readouterr().err
    assert "[用量] 输入 1,500 · 输出 200 · 缓存命中 1,024（68%） · 1 次调用" in err


def test_report_and_show_add_the_thinking_cost_hint(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """思考占输出过半时，报告与 crw show 都给一句可操作的降本提示。"""
    configured_provider()
    fake_chat_model.load(
        [
            fakes.ai_text(
                fakes.file_fence(CLEAN_CODE),
                usage=fakes.call_usage(
                    input_tokens=500, output_tokens=400, cached_tokens=256, reasoning_tokens=300
                ),
            ),
            fakes.ai_tool_call(
                "read_file",
                path="code_v1/main.py",
                usage=fakes.call_usage(input_tokens=600, output_tokens=100, reasoning_tokens=60),
            ),
            fakes.ai_text(
                fakes.review_fence([], "已读取完整文件，未发现阻塞问题。"),
                usage=fakes.call_usage(input_tokens=700, output_tokens=300, reasoning_tokens=240),
            ),
        ]
    )

    assert main(["run", "写一个计算器"]) == 0

    (summary,) = records.list_runs()
    report = (summary.path / "report.md").read_text(encoding="utf-8")
    assert "提示：思考 token 占输出 75%" in report
    assert "--thinking off" in report

    capsys.readouterr()
    assert main(["show", summary.run_id]) == 0
    assert "[提示] 思考 token 占输出 75%" in capsys.readouterr().out


def test_review_command_hints_when_thinking_dominates_output(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``crw review`` 没有运行记录，降本提示因此也要出现在回显里。"""
    configured_provider()
    fake_chat_model.load(
        [
            fakes.ai_text(
                "无阻塞问题。",
                usage=fakes.call_usage(input_tokens=1000, output_tokens=500, reasoning_tokens=400),
            )
        ]
    )

    assert main(["review", str(sample_path("divide_by_zero.py"))]) == 0

    err = capsys.readouterr().err
    assert "[提示] 思考 token 占输出 80%" in err


def test_usage_table_escapes_a_poisoned_node_name(
    configured_provider: Callable[[], dict[str, object]], fake_chat_model: object
) -> None:
    """state.json 是"可能被手工改过"的输入：用量表里的节点名不得把表格弄串列。"""
    run_id = _run_with_usage(fake_chat_model, configured_provider)
    run_dir = records.find_run_dir(run_id)
    state = dict(records.read_state(run_dir))
    state["usage"] = {**state["usage"], "evil|node": state["usage"]["encode"]}

    report = engine_mod.build_report(
        state=state,
        rounds=records.read_rounds(run_dir),
        run_dir=run_dir,
        artifacts=records.run_artifacts(run_dir),
    )

    rows = [line for line in report.splitlines() if "evil" in line]
    assert len(rows) == 1
    assert "evil\\|node" in rows[0]
    assert rows[0].count(" | ") == 5  # 仍是 6 列：名字里的竖线被转义，不产生新列
