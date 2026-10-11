"""命令面：参数解析与子命令分发（run / runs / show / resume / review / config / graph / uninstall）。

只有本层接触 argparse 与终端输入；业务逻辑委托 workflow 层，命令面与选项口径见 Design §8.1。
"""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import sys
from collections.abc import Callable, Sequence

_PACKAGE_NAME = "code-review-workflow"
_PROG = "crw"
_EXIT_ERROR = 2


def _package_version() -> str:
    """读取已安装发行版的版本号（Design §9：版本以包元数据为准）。

    源码目录未安装时元数据缺失，返回可辨识的占位值，避免误报一个具体版本。
    """
    try:
        return importlib.metadata.version(_PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


_MAX_ITER_LIMIT = 10


def _max_iter_rounds(text: str) -> int:
    """解析修复轮次上限：0 ~ ``_MAX_ITER_LIMIT`` 的 ASCII 十进制整数（Design §4.4）。

    先按字面收紧再取值——int() 会接受 ``+3`` / ``1_0`` / 全角与阿拉伯-印度数字等写法并静默
    重新解释用户输入；同时设硬上限，避免误填大值造成无界循环与无界调用费用。
    """
    if not text.isascii() or not text.isdecimal():
        raise argparse.ArgumentTypeError(f"需要 0–{_MAX_ITER_LIMIT} 的整数，收到 {text!r}")
    rounds = int(text)
    if rounds > _MAX_ITER_LIMIT:
        raise argparse.ArgumentTypeError(
            f"修复轮次上限最多 {_MAX_ITER_LIMIT} 轮（防止无界循环），收到 {text!r}"
        )
    return rounds


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    """添加 run / resume / review 共用的执行与模型覆盖选项。"""
    parser.add_argument("--no-exec", action="store_true", help="禁止运行代码核实")
    parser.add_argument("--model", metavar="名", help="临时覆盖模型")
    parser.add_argument("--provider", metavar="名", help="临时覆盖供应商")


def _add_workflow_options(parser: argparse.ArgumentParser) -> None:
    """添加 run / resume 共用的工作流选项（resume 不提供 -f）。"""
    parser.add_argument(
        "--max-iter",
        type=_max_iter_rounds,
        default=3,
        metavar="n",
        help=f"修复轮次上限（默认 3，可取值 0–{_MAX_ITER_LIMIT}；0 = 只审不修）",
    )
    parser.add_argument(
        "--strict", action="store_true", help="干净判定加严：无【严重】且无【一般】"
    )
    _add_model_options(parser)
    parser.add_argument("--out", metavar="目录", help="将最终代码与报告复制到指定目录")
    parser.add_argument("--no-stream", action="store_true", help="关闭流式渲染，整体输出")


def _add_review_options(parser: argparse.ArgumentParser) -> None:
    """添加 review 的选项（单 Agent 审查，只查不改）。"""
    parser.add_argument("-o", "--out", dest="out", metavar="文件", help="将 Markdown 报告写入文件")
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="输出结构化结果（存在【严重】时退出码 1）",
    )
    _add_model_options(parser)


def build_parser() -> argparse.ArgumentParser:
    """构造完整的命令行解析器（命令面与选项口径见 Design §8.1）。"""
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description="基于 LLM 的代码审查工作流：编码 → 审查 → 修复（+ 复审循环）。",
    )
    parser.add_argument("--version", action="version", version=f"{_PROG} {_package_version()}")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<命令>")

    run_parser = subparsers.add_parser("run", help="运行工作流（需求文本或既有代码文件）")
    inputs = run_parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "requirement", nargs="?", metavar="需求文本", help="需求文本（与 -f 二选一）"
    )
    inputs.add_argument(
        "-f", "--file", dest="file", metavar="文件", help="既有代码文件（与需求文本二选一）"
    )
    run_parser.set_defaults(_subparser=run_parser)
    _add_workflow_options(run_parser)

    subparsers.add_parser("runs", help="列出运行记录")

    show_parser = subparsers.add_parser("show", help="展示运行详情")
    show_parser.add_argument("run_id", nargs="?", metavar="运行号", help="省略时列出记录供选择")

    resume_parser = subparsers.add_parser("resume", help="以运行的最新代码版本为起点续跑")
    resume_parser.add_argument("run_id", metavar="运行号", help="待续跑的运行号")
    _add_workflow_options(resume_parser)

    review_parser = subparsers.add_parser("review", help="单 Agent 快速审查（只查不改）")
    review_parser.add_argument(
        "paths", nargs="+", metavar="路径", help="审查目标：文件或目录；- 读 stdin"
    )
    review_parser.set_defaults(_subparser=review_parser)
    _add_review_options(review_parser)

    config_parser = subparsers.add_parser("config", help="模型供应商配置 / 连通性自检")
    config_parser.add_argument(
        "action", nargs="?", choices=["test"], help="test = 逐供应商连通性自检；省略则交互式配置"
    )

    graph_parser = subparsers.add_parser("graph", help="导出工作流 mermaid 流程图")
    graph_parser.add_argument(
        "-o",
        "--out",
        dest="out",
        metavar="文件",
        help="写入文件（默认输出到 stdout）",
    )

    subparsers.add_parser("uninstall", help="卸载：分级清理程序、PATH 条目与用户数据")

    return parser


def _validate_input_contract(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """校验跨选项的输入契约；非法组合按参数错误处理（stderr 提示 + 退出码 2）。

    “需求文本与 -f 同给”由解析器的互斥组先行拦截，此处补充其无法表达的另一半（都没给或为空白）。
    报错定位到子命令 usage（子解析器经 ``set_defaults`` 随参数带回），取不到时退回主解析器。
    """
    error_parser = getattr(args, "_subparser", parser)
    if args.command == "run" and not (
        (args.requirement or "").strip() or (args.file or "").strip()
    ):
        error_parser.error("run 需要需求文本或 -f 文件（二选一，且不能为空白）")
    if args.command == "review":
        if len(args.paths) > 1 and "-" in args.paths:
            error_parser.error("review 的 - 表示读取 stdin，只能单独使用")
        if any(not path.strip() for path in args.paths):
            error_parser.error("review 的路径不能为空白")


_MAX_ERROR_CHARS = 400
# 左侧非字母数字断言：避免把 task-analysis 这类普通连字符词误判为密钥形态。
_SECRET_PATTERN = re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9]{8,}")


def _report_failure(command: str | None, exc: Exception) -> int:
    """兜底报错：单行可读提示 + 运行错误退出码。

    异常文本可能来自模型 / HTTP 栈（含请求细节）而终端输出常被重定向保存，故按 Design §8.2
    的截断惯例限长，并遮蔽 ``sk-`` 形态的密钥（覆盖常见供应商，尽力而为而非保证）。
    """
    detail = _SECRET_PATTERN.sub("sk-***", f"{type(exc).__name__}: {exc}")
    if len(detail) > _MAX_ERROR_CHARS:
        detail = f"{detail[:_MAX_ERROR_CHARS]}…（已截断）"
    scope = f" {command}" if command else ""
    print(f"{_PROG}{scope}：运行失败 — {detail}", file=sys.stderr)
    return _EXIT_ERROR


def _not_implemented(command: str) -> int:
    """占位处理：报告命令尚未实现，返回运行错误退出码。"""
    print(f"crw {command}：该命令尚未实现（当前为工程骨架）。", file=sys.stderr)
    return _EXIT_ERROR


def _cmd_run(args: argparse.Namespace) -> int:
    """运行工作流（占位）。"""
    return _not_implemented("run")


def _cmd_runs(args: argparse.Namespace) -> int:
    """列出运行记录（占位）。"""
    return _not_implemented("runs")


def _cmd_show(args: argparse.Namespace) -> int:
    """展示运行详情（占位）。"""
    return _not_implemented("show")


def _cmd_resume(args: argparse.Namespace) -> int:
    """续跑运行（占位）。"""
    return _not_implemented("resume")


def _cmd_review(args: argparse.Namespace) -> int:
    """单 Agent 审查（占位）。"""
    return _not_implemented("review")


def _cmd_config(args: argparse.Namespace) -> int:
    """供应商配置与自检（占位）。"""
    return _not_implemented("config")


def _cmd_graph(args: argparse.Namespace) -> int:
    """导出工作流图（占位）。"""
    return _not_implemented("graph")


def _cmd_uninstall(args: argparse.Namespace) -> int:
    """卸载（占位）。"""
    return _not_implemented("uninstall")


_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "run": _cmd_run,
    "runs": _cmd_runs,
    "show": _cmd_show,
    "resume": _cmd_resume,
    "review": _cmd_review,
    "config": _cmd_config,
    "graph": _cmd_graph,
    "uninstall": _cmd_uninstall,
}


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口：解析参数并分发到子命令处理器。

    参数非法时由 argparse 与 ``_validate_input_contract`` 报出并直接抛 ``SystemExit(2)``（不返回）；
    解析 / 校验与处理器两阶段的未预期异常均兜底为可读报错 + 退出码 2，避免以 traceback 退出
    而混淆"达限残留"语义（Design §8.1）。

    Args:
        argv: 命令行参数（不含程序名）；省略时取 ``sys.argv[1:]``。

    Returns:
        进程退出码：0 完成 / 1 达修复轮次上限仍有【严重】/ 2 运行错误（Design §8.1）。
    """
    command: str | None = None
    try:
        parser = build_parser()
        args = parser.parse_args(argv)
        _validate_input_contract(parser, args)
        command = args.command
    except Exception as exc:  # noqa: BLE001 — 解析/校验阶段的未预期异常同样兜底为可读报错
        return _report_failure(command, exc)
    # 处理器查找置于兜底之外：接线遗漏属编程缺陷，应显式暴露而非伪装成运行错误。
    handler = _HANDLERS[args.command]
    try:
        return handler(args)
    except Exception as exc:  # noqa: BLE001 — 处理器阶段的未预期异常兜底为可读报错
        return _report_failure(args.command, exc)
