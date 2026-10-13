"""命令面：参数解析与子命令分发（run / runs / show / resume / review / config / graph / uninstall）。

只有本层接触 argparse 与终端输入；业务逻辑委托 workflow / agents 层，命令面与选项口径见
Design §8.1。输出通道按 §8.4 分流：结论 / findings 清单 / 报告走 stdout，过程叙述、审查
原文与错误走 stderr（``--json`` 时 stdout 纯 JSON）。
"""

from __future__ import annotations

import argparse
import getpass
import importlib.metadata
import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import agents as agents_mod
from . import config as config_mod
from . import render
from . import tools as tools_mod
from . import uninstall as uninstall_mod
from .workflow import contracts as contracts_mod
from .workflow import engine as engine_mod
from .workflow import graph as graph_mod
from .workflow import records as records_mod
from .workflow import state as state_mod

_PACKAGE_NAME = "code-review-workflow"
_PROG = "crw"
_EXIT_OK = 0
_EXIT_ISSUES = 1
_EXIT_ERROR = 2

_MAX_ITER_LIMIT = 10

_CONFIG_FILE_NOTICE = "提示：{path} 可能含明文 API key——请注意文件权限与备份，勿提交到代码仓库。"
_CONFIG_SCOPE_HINT = "当前无供应商配置。请添加一个供应商（或按 0 退出后设置环境变量）。"


def _package_version() -> str:
    """读取已安装发行版的版本号（Design §9：版本以包元数据为准）。

    源码目录未安装时元数据缺失，返回可辨识的占位值，避免误报一个具体版本。
    """
    try:
        return importlib.metadata.version(_PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


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


class _MaxIterAction(argparse.Action):
    """记录 ``--max-iter`` 是否由命令行显式给出（轮次取值来源，Design §4.4）。"""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        setattr(namespace, self.dest, values)
        namespace.max_iter_source = state_mod.SOURCE_CLI


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
        default=engine_mod.DEFAULT_MAX_ITERATIONS,
        action=_MaxIterAction,
        metavar="n",
        help=f"修复轮次上限（默认 {engine_mod.DEFAULT_MAX_ITERATIONS}，"
        f"可取值 0–{_MAX_ITER_LIMIT}；0 = 只审不修）",
    )
    parser.add_argument(
        "--strict", action="store_true", help="干净判定加严：无【严重】且无【一般】"
    )
    parser.add_argument(
        "--parallel-review",
        dest="parallel_review",
        action="store_true",
        help="并行双视角审查（正确性 + 安全性同轮并行，汇合去重；审查调用量约翻倍）",
    )
    _add_model_options(parser)
    parser.add_argument("--out", metavar="目录", help="将最终代码与报告复制到指定目录")
    parser.add_argument("--no-stream", action="store_true", help="关闭流式渲染，整体输出")
    parser.set_defaults(max_iter_source=None)


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
    graph_parser.add_argument(
        "--parallel-review",
        dest="parallel_review",
        action="store_true",
        help="导出并行双视角审查形态的图（与 crw run --parallel-review 同源）",
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


def _mask_secrets(text: str) -> str:
    """遮蔽 ``sk-`` 形态的密钥（错误通道统一口径：终端输出常被重定向保存）。"""
    return _SECRET_PATTERN.sub("sk-***", text)


def _report_failure(command: str | None, exc: Exception) -> int:
    """兜底报错：单行可读提示 + 运行错误退出码。

    异常文本可能来自模型 / HTTP 栈（含请求细节）而终端输出常被重定向保存，故按 Design §8.2
    的截断惯例限长，并遮蔽 ``sk-`` 形态的密钥（覆盖常见供应商，尽力而为而非保证）。
    """
    detail = _mask_secrets(f"{type(exc).__name__}: {exc}")
    if len(detail) > _MAX_ERROR_CHARS:
        detail = f"{detail[:_MAX_ERROR_CHARS]}…（已截断）"
    scope = f" {command}" if command else ""
    print(f"{_PROG}{scope}：运行失败 — {detail}", file=sys.stderr)
    return _EXIT_ERROR


# ---- 模型解析与运行装配 ----


def _resolve_model(model: str | None, provider: str | None, command: str) -> Any:
    """解析本次运行的模型连接；失败给可读指引并返回 ``None``（调用方退出 2）。"""
    try:
        resolved = config_mod.resolve(model=model, provider=provider)
    except config_mod.ConfigError as exc:
        print(f"crw {command}：配置错误 — {_mask_secrets(str(exc))}", file=sys.stderr)
        return None
    if resolved is None:
        print(
            f"crw {command}：尚未配置模型供应商 — 请运行 crw config 完成配置，"
            "或设置环境变量 CRW_DEEPSEEK_API_KEY（Windows 用 setx，须新开终端生效）。",
            file=sys.stderr,
        )
        legacy_hint = _legacy_config_hint()
        if legacy_hint:
            print(legacy_hint, file=sys.stderr)
        return None
    return resolved


def _legacy_config_hint() -> str:
    """零配置且有前作配置时的导入提示（Design §6.2 #2 / D7：只探测存在性并提示）。"""
    if config_mod.config_path().exists():
        return ""
    legacy = config_mod.legacy_config_path()
    if not legacy.exists():
        return ""
    return f"提示：检测到前作配置 {legacy}——运行 crw config 可选择导入（不会修改前作文件）。"


def _read_input_file(path: Path, *, command: str = "run") -> tuple[str | None, int]:
    """读取输入文件（Design §8.4：CLI 自身读取）；失败给可读原因并返回运行错误码。

    ``command`` 只影响报错前缀：resume 读的是父运行的代码版本，报错里写 "crw run" 会把
    用户引到错误的命令上。
    """
    if not path.exists():
        print(f"crw {command}：运行失败 — 输入文件不存在：{path}", file=sys.stderr)
        return None, _EXIT_ERROR
    if path.is_dir():
        print(
            f"crw {command}：运行失败 — 输入是目录：{path}（-f 需要一个代码文件）",
            file=sys.stderr,
        )
        return None, _EXIT_ERROR
    try:
        return path.read_text(encoding="utf-8"), _EXIT_OK
    except UnicodeDecodeError:
        print(f"crw {command}：运行失败 — 输入文件不是 UTF-8 文本：{path}", file=sys.stderr)
        return None, _EXIT_ERROR
    except OSError as exc:
        print(f"crw {command}：运行失败 — 输入文件读取失败：{path}（{exc}）", file=sys.stderr)
        return None, _EXIT_ERROR


def _cmd_run(args: argparse.Namespace) -> int:
    """运行工作流：解析模型 → 装配请求 → 驱动引擎 → 渲染结论（Design §8.1）。

    ``-f`` 输入由命令面读取（Design §8.4），工作区内落盘交给引擎；``--out`` 在收尾后把
    最终代码与报告复制出去（Design §4.5）。
    """
    requirement = (args.requirement or "").strip() or None
    source_path: Path | None = None
    source_text: str | None = None
    if args.file:
        source_path = Path(args.file)
        source_text, error_code = _read_input_file(source_path)
        if source_text is None:
            return error_code
    resolved = _resolve_model(args.model, args.provider, "run")
    if resolved is None:
        return _EXIT_ERROR
    request = engine_mod.RunRequest(
        requirement=requirement,
        source_path=source_path,
        source_text=source_text,
        max_iterations=args.max_iter,
        max_iterations_source=args.max_iter_source or state_mod.SOURCE_DEFAULT,
        strict=bool(args.strict),
        no_exec=bool(args.no_exec),
        parallel_review=bool(args.parallel_review),
        resolved=resolved,
    )
    engine = engine_mod.WorkflowEngine(observer=render.RunRenderer(stream=not args.no_stream))
    result = engine.execute(request)
    render.render_result(result, command="run")
    if args.out:
        _export_outputs(result, Path(args.out))
    return result.exit_code


def _export_outputs(result: engine_mod.RunResult, out_dir: Path) -> None:
    """``--out``：把最终代码与报告复制到指定目录（Design §4.5；重名加序号）。

    复制失败不影响运行结论（Design §4.6）：只向 stderr 警示，退出码仍按运行状态。
    """
    if result.run_dir is None:
        print("crw run：本次运行没有工作目录，--out 已跳过（无产物可复制）。", file=sys.stderr)
        return
    code_path = Path(str(result.state.get("current_code") or ""))
    report_path = result.run_dir / "report.md"
    kinds: dict[str, Path] = {}
    if code_path.is_file():
        kinds["code_final"] = code_path
    if report_path.is_file():
        kinds["report"] = report_path
    if not kinds:
        print("crw run：--out 无可复制产物（本次运行未产出代码或报告）。", file=sys.stderr)
        return
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"crw run：--out 目录创建失败 — {out_dir}（{exc}）", file=sys.stderr)
        return
    # 代码与报告成对加序号：两者一并落到同一序号，产物目录里不会出现"半新半旧"的一组
    index = 1
    while True:
        suffix = "" if index == 1 else f"_{index}"
        names = {
            "code_final": f"code_final{suffix}.py",
            "report": f"report{suffix}.md",
        }
        targets = {kind: out_dir / names[kind] for kind in kinds}
        if all(not target.exists() for target in targets.values()):
            break
        index += 1
    for kind, target in targets.items():
        source = kinds[kind]
        try:
            records_mod.atomic_write_text(target, source.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as exc:
            print(f"crw run：--out 复制失败 — {source.name} → {target}（{exc}）", file=sys.stderr)
            return
        print(f"已复制：{target}", file=sys.stderr)


def _cmd_runs(args: argparse.Namespace) -> int:
    """列出运行记录（运行号 / 时间 / 输入摘要 / 状态 / 修复轮数 / 残留严重数）。"""
    render.render_runs(records_mod.list_runs())
    return _EXIT_OK


def _inherited_rounds(parent_state: Mapping[str, Any], parent_run_id: str) -> int:
    """续跑继承的轮次上限：与命令行同一套校验（0–10、ASCII 十进制）。

    继承路径不得绕过上限：``state.json`` 可能被手改成 999，而该上限正是"防止误填大值造成
    无界循环与无界调用费用"的硬闸（Design §4.4）。值非法时退回内置默认并如实提示。
    """
    raw = parent_state.get("max_iterations")
    try:
        return _max_iter_rounds(str(raw))
    except argparse.ArgumentTypeError:
        print(
            f"crw resume：父运行 {parent_run_id} 的轮次上限（{raw!r}）不合法，"
            f"已改用内置默认 {engine_mod.DEFAULT_MAX_ITERATIONS} 轮。",
            file=sys.stderr,
        )
        return engine_mod.DEFAULT_MAX_ITERATIONS


def _cmd_show(args: argparse.Namespace) -> int:
    """展示运行详情；省略运行号时列出记录供数字选择（Design §8.1）。

    ``state.json`` 损坏时按 §8.2 降级展示（摘要取自日志与产物清单）并给出退出码 2——
    §8.1 的"记录损坏"口径即此：**展示照给，退出码照实**（脚本化消费能察觉异常，
    人读仍拿得到可用信息）。
    """
    run_id = args.run_id
    if run_id is None:
        summaries = records_mod.list_runs()
        if not summaries:
            print("crw show：暂无运行记录（先执行 crw run）。", file=sys.stderr)
            return _EXIT_ERROR
        # 候选清单属交互过程：走 stderr，stdout 只留被选中记录的详情（Design §8.4）
        picked = records_mod.select_run(
            summaries,
            ask=render.ask_user,
            show=lambda line: print(line, file=sys.stderr),
        )
        if picked is None:
            print("crw show：已取消。", file=sys.stderr)
            return _EXIT_ERROR
        run_id = picked.run_id
    try:
        detail = records_mod.load_detail(run_id)
    except records_mod.RunRecordError as exc:
        print(f"crw show：{exc}", file=sys.stderr)
        return _EXIT_ERROR
    render.render_detail(detail)
    if detail.degraded:
        print(f"crw show：记录损坏，已降级展示 — {detail.degraded_reason}", file=sys.stderr)
        return _EXIT_ERROR
    return _EXIT_OK


def _cmd_resume(args: argparse.Namespace) -> int:
    """续跑：以父运行的最新代码版本为新运行的起点（Design §4.4）。

    ``-f`` 不适用（起点由父运行决定）；父运行须存在 ``current_code`` 且文件仍在，否则给出
    可读原因并退出 2。未显式给出的轮次设置**继承父运行**（来源标 ``parent``），父运行的
    ``--strict`` 与审查形态一并继承——续跑是"接着修"，换判定标准会让结论不可比。
    ``--no-exec`` 不继承（它不进状态快照，命令面无法可靠区分"显式关闭"与"从未设置"）。
    """
    try:
        detail = records_mod.load_detail(args.run_id)
    except records_mod.RunRecordError as exc:
        print(f"crw resume：{exc}", file=sys.stderr)
        return _EXIT_ERROR
    parent_state = detail.state
    if detail.degraded:
        print(
            f"crw resume：父运行 {args.run_id} 的记录已损坏（state.json 不可读），"
            "无法确定续跑起点与轮次设置；可用 crw show 查看降级信息。",
            file=sys.stderr,
        )
        return _EXIT_ERROR
    code_path = Path(str(parent_state.get("current_code") or ""))
    if not code_path.is_file():
        print(
            f"crw resume：父运行 {args.run_id} 没有可续跑的代码版本"
            f"（current_code：{code_path or '（未记录）'}）——该运行可能在建码前就失败了。",
            file=sys.stderr,
        )
        return _EXIT_ERROR
    source_text, error_code = _read_input_file(code_path, command="resume")
    if source_text is None:
        return error_code
    resolved = _resolve_model(args.model, args.provider, "resume")
    if resolved is None:
        return _EXIT_ERROR
    # 轮次取值来源：命令行显式 > 父运行继承（配置默认 ~/.crw/settings.json 随 M4 落地）
    inherited_rounds = _inherited_rounds(parent_state, args.run_id)
    request = engine_mod.RunRequest(
        source_path=code_path,
        source_text=source_text,
        max_iterations=args.max_iter if args.max_iter_source else inherited_rounds,
        max_iterations_source=args.max_iter_source or state_mod.SOURCE_PARENT,
        strict=bool(args.strict) or bool(parent_state.get("strict")),
        no_exec=bool(args.no_exec),
        parallel_review=bool(args.parallel_review) or bool(parent_state.get("parallel_review")),
        resolved=resolved,
        parent_run_id=args.run_id,
    )
    engine = engine_mod.WorkflowEngine(observer=render.RunRenderer(stream=not args.no_stream))
    result = engine.execute(request)
    render.render_result(result, command="resume")
    if args.out:
        _export_outputs(result, Path(args.out))
    return result.exit_code


def _cmd_graph(args: argparse.Namespace) -> int:
    """导出工作流 mermaid 流程图（Design §8.1）：默认 stdout，``-o`` 写文件。

    导出的节点 / 边集合与 ``build_graph`` 同源（同一份结构声明），由快照测试锁定；
    ``--parallel-review`` 导出并行双视角形态的图。
    """
    parallel = bool(getattr(args, "parallel_review", False))
    text = graph_mod.mermaid_flowchart(parallel_review=parallel)
    if not args.out:
        # 展示层故障（下游关闭管道）不改变结论，见 render.guarded 的说明
        render.guarded(lambda: print(text, end=""))
        return _EXIT_OK
    target = Path(args.out)
    try:
        records_mod.atomic_write_text(target, text)
    except OSError as exc:
        print(f"crw graph：写入失败 — {target}（{exc}）", file=sys.stderr)
        return _EXIT_ERROR
    print(f"已写入：{target}", file=sys.stderr)
    return _EXIT_OK


def _cmd_uninstall(args: argparse.Namespace) -> int:
    """卸载：分级确认（程序 / 垫片 → PATH 条目 → 登记的环境变量 → 用户数据），确认前零删除。"""
    return uninstall_mod.run_uninstall()


# ---- crw review：单 Agent 审查（不修复、不产生运行记录）----


class _ReviewInputError(Exception):
    """审查入参的文件级错误（路径不存在 / 目录无文本文件 / 不可读），退出码 2。"""


def _review_roots(paths: Sequence[str]) -> list[Path]:
    """审查上下文本根：文件取其所在目录、目录取自身；stdin 退回进程启动目录（Design §8.4）。"""
    roots: list[Path] = []
    for raw in paths:
        if raw == "-":
            roots.append(Path.cwd())
            continue
        path = Path(raw)
        roots.append(path if path.is_dir() else path.parent)
    if not roots:
        roots.append(Path.cwd())
    return roots


def _collect_review_files(paths: Sequence[str]) -> list[tuple[str, str]]:
    """展开审查入参：目录展开第一层文本文件、``-`` 读 stdin。

    跳过点开头的隐藏文件（含 .env 等敏感文件，其内容不应进入审查）与二进制文件，跳过
    清单打印到 stderr。返回 ``(显示名, 内容)``：显示名保留用户输入形态（location 引用
    与报告标题用它），stdin 固定 ``<stdin>``。
    """
    collected: list[tuple[str, str]] = []
    for raw in paths:
        if raw == "-":
            collected.append(("<stdin>", _read_stdin_text()))
            continue
        path = Path(raw)
        if not path.exists():
            raise _ReviewInputError(f"路径不存在：{path}")
        if path.is_dir():
            selected: list[Path] = []
            skipped: list[str] = []
            for item in sorted(path.iterdir()):
                if not item.is_file() or item.name.startswith("."):
                    continue
                if not tools_mod.fs.looks_textual(item):
                    skipped.append(item.name)
                    continue
                selected.append(item)
            if not selected:
                detail = f"（已跳过：{', '.join(skipped)}）" if skipped else ""
                raise _ReviewInputError(f"目录 {path} 第一层没有可审查的文本文件{detail}。")
            if skipped:
                print(f"已跳过二进制文件：{', '.join(skipped)}", file=sys.stderr)
            collected.extend((str(item), _read_review_text(item)) for item in selected)
        else:
            collected.append((str(path), _read_review_text(path)))
    return collected


def _read_stdin_text() -> str:
    """读 stdin（``crw review -``）：与单文件同口径设读取上限，超限给可读错误。

    无上限的 ``sys.stdin.read()`` 会让 ``crw review - < 巨大文件`` 把内存吃光；审查场景下
    超限本身的可行处置就是"请改用文件路径或分段"。
    """
    limit = tools_mod.fs.MAX_READ_BYTES
    text = sys.stdin.read(limit + 1)
    if len(text) > limit:
        raise _ReviewInputError(
            f"标准输入超过读取上限（{limit // 1048576} MB），请改用文件路径或分段审查。"
        )
    return text


def _read_review_text(path: Path) -> str:
    """读取待审查文件文本；失败或超单文件大小上限抛 :class:`_ReviewInputError`。

    上限与 ``read_file`` 同值同口径：两条读取入口（工具 / 审查 prompt）对"读取体量有界"
    的保证必须一致，否则 ``crw review`` 大文件仍会整文件进内存。
    """
    try:
        if path.stat().st_size > tools_mod.fs.MAX_READ_BYTES:
            raise _ReviewInputError(
                f"文件 {path} 超过单文件读取上限（{tools_mod.fs.MAX_READ_BYTES // 1048576} MB），"
                "请直接指定要审查的具体文件或分段。"
            )
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise _ReviewInputError(f"无法读取文件 {path}：{exc}") from exc


def _answer_or_placeholder(answer: str | None) -> str:
    """多文件文本模式的单个结论：``None`` 才是"审查失败"，空串是成功但空的结果。"""
    return "[该文件审查失败，结论不可用]" if answer is None else answer


def _cmd_review(args: argparse.Namespace) -> int:
    """单 Agent 审查：批量审查 → 结论 / ``--json`` / ``-o`` 报告（Design §8.1）。

    stdout 只承载结论与 ``--json``（过程与错误走 stderr）；退出码 0 完成、1 为 ``--json``
    模式下存在【严重】、2 为运行错误（含单文件失败）。**不产生运行记录**（无 run_id）。
    """
    try:
        files = _collect_review_files(args.paths)
    except _ReviewInputError as exc:
        print(f"crw review：{exc}", file=sys.stderr)
        return _EXIT_ERROR
    resolved = _resolve_model(args.model, args.provider, "review")
    if resolved is None:
        return _EXIT_ERROR
    observer = render.RunRenderer(stream=True)
    runtime, registry = agents_mod.build_review_runtime(
        resolved=resolved,
        roots=_review_roots(args.paths),
        no_exec=bool(args.no_exec),
        emit=observer.on_event,
        on_delta=observer.on_delta,
    )
    json_entries: list[dict[str, Any]] = []
    text_answers: list[str | None] = [None] * len(files)  # 与 files 逐位对齐；None = 该文件审查失败
    failed = False
    severe = False
    for position, (display_name, content) in enumerate(files):
        print(f"正在审查 {display_name} …", file=sys.stderr)
        try:
            answer = agents_mod.review_once(
                runtime,
                registry,
                display_name=display_name,
                content=content,
                as_json=bool(args.as_json),
                max_lines=render.REVIEW_PROMPT_MAX_LINES,
            )
        except Exception as exc:  # noqa: BLE001 — 单文件失败继续批量；最终退出码 2
            print(
                f"[审查失败] {display_name}：{_SECRET_PATTERN.sub('sk-***', str(exc))}",
                file=sys.stderr,
            )
            failed = True
            continue
        if args.as_json:
            try:
                report = contracts_mod.parse_review_report(answer)
            except contracts_mod.ContractViolation as exc:
                print(f"[结构化输出无效] {display_name}：{exc}", file=sys.stderr)
                failed = True
                continue
            entries = report["files"]
            # 模型可能违反提示词约定返回多条 files 条目：全部采纳并全量判定严重性，
            # 静默丢弃会造成退出码误判
            if any(
                finding.get("severity") == contracts_mod.SEVERE
                for entry in entries
                for finding in entry["findings"]
            ):
                severe = True
            json_entries.extend(entries)
        else:
            text_answers[position] = answer

    if args.as_json:
        merged = contracts_mod.merge_review_reports(json_entries)
        print(json.dumps(merged, ensure_ascii=False, indent=2))
        markdown = render.review_json_report(merged)
    else:
        if len(files) > 1:
            for position, ((display_name, _), answer) in enumerate(zip(files, text_answers)):
                if position:
                    print()
                print(f"===== {display_name} =====")
                print(_answer_or_placeholder(answer))
        else:
            for answer in text_answers:
                print(answer if answer is not None else "")
        markdown = render.review_text_report(files, text_answers)
    if args.out:
        target = Path(args.out)
        try:
            records_mod.atomic_write_text(target, markdown)
        except OSError as exc:
            print(f"crw review：报告写入失败 — {target}（{exc}）", file=sys.stderr)
            failed = True
        else:
            print(f"报告已写入：{target}", file=sys.stderr)
    if failed:
        return _EXIT_ERROR
    return _EXIT_ISSUES if (args.as_json and severe) else _EXIT_OK


# ---- crw config：供应商配置 / 连通性自检 ----


def _say(text: str = "") -> None:
    """配置交互的输出（菜单与结论）：stdout（Design §8.4 的结论通道）。"""
    print(text)


def _say_error(text: str) -> None:
    """配置交互的错误（配置损坏 / 写盘失败）：stderr（Design §8.4 的错误通道）。

    与 ``crw config test`` 的错误同口径——同一类"配置错误"不因走交互模式而换通道。
    """
    print(text, file=sys.stderr)


def _ask(prompt: str) -> str | None:
    """配置交互的输入封装：EOF / Ctrl+C / 输入流不可用一律按取消（``None``）处理。

    与 :func:`crw.render.ask_user` 同口径，但提示语写 stdout——配置菜单本身就在 stdout，
    提示与菜单同流才不至于互相错位。
    """
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt, OSError, RuntimeError):
        return None


def _read_key(prompt: str) -> str | None:
    """key 输入不回显（getpass）；``None`` = 取消（中断 / 输入流不可用 / EOF）。

    与 :func:`_ask` 同一取消哨兵：空串是"用户直接回车"，取消必须能被区分出来——否则
    Ctrl+C 会被当成"给了空 key"，把原有 key 覆盖掉。
    """
    try:
        return getpass.getpass(prompt, stream=sys.stderr).strip()
    except (EOFError, KeyboardInterrupt, OSError, RuntimeError):
        return None


def _mask_key(api_key: Any, name: str = "") -> str:
    """key 打码展示：只露前 4 后 4 位，配置界面不整段回显；缺 key 时提示回退链。"""
    if not isinstance(api_key, str) or not api_key:
        chain = (
            f"{config_mod.provider_env_var(name)} → {config_mod.LEGACY_KEY_ENV}"
            if name
            else config_mod.LEGACY_KEY_ENV
        )
        return f"(未设置，回退环境变量 {chain})"
    if len(api_key) <= 8:
        return "****"
    return f"{api_key[:4]}****{api_key[-4:]}"


def _print_providers(data: dict[str, Any]) -> None:
    """当前配置概览（key 打码）。"""
    providers = data.get("providers") or []
    if not providers:
        _say(_CONFIG_SCOPE_HINT)
        return
    _say(f"默认模型：{data.get('default') or '（未设置）'}")
    for entry in providers:
        _say(
            f"  - {entry['name']}　base_url={entry['base_url']}"
            f"　key={_mask_key(entry.get('api_key'), entry['name'])}"
            f"　模型：{', '.join(entry['models'])}"
        )


class _ConfigSession:
    """``crw config`` 交互会话：内存中修改、每个完整操作即写盘（任何时刻退出都一致）。"""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self._notice_shown = False

    def providers(self) -> list[dict[str, Any]]:
        """供应商清单（缺失时按空表就地创建）。"""
        return self.data.setdefault("providers", [])

    def save(self) -> None:
        """校验后原子写盘；失败只报错不中断交互。"""
        try:
            config_mod.save_raw(self.data)
        except (config_mod.ConfigError, OSError) as exc:
            _say_error(f"写入失败：{exc}")
            return
        if not self._notice_shown:
            _say(_CONFIG_FILE_NOTICE.format(path=config_mod.config_path()))
            self._notice_shown = True


def _choose_provider_type() -> tuple[str, config_mod.ProviderPreset | None] | None:
    """供应商类型选择：1 自定义 + 预设清单，编号选择；取消返回 ``None``。"""
    _say("选择供应商类型：")
    _say("  1. 自定义（手动输入名称/接入点/模型清单）")
    for index, preset in enumerate(config_mod.PROVIDER_PRESETS, start=2):
        _say(f"  {index}. {preset.name}")
    index = records_mod.parse_index(
        _ask("选择（回车取消）："), len(config_mod.PROVIDER_PRESETS) + 1
    )
    if index is None:
        return None
    if index == 0:
        return ("custom", None)
    return ("preset", config_mod.PROVIDER_PRESETS[index - 1])


def _input_base_url_and_models(
    preset: config_mod.ProviderPreset | None,
) -> tuple[str, list[str]] | None:
    """接入点与模型清单输入：预设自动预填、预填项均可修改；自定义手填。"""
    default_base = preset.base_url if preset is not None else config_mod.DEFAULT_BASE_URL
    hint = "回车用官方" if preset is not None else "回车默认"
    base_input = _ask(f"API 接入点（{hint} {default_base}）：")
    if base_input is None:
        _say("已取消。")
        return None
    base_url = base_input or default_base
    if preset is not None:
        models = list(preset.models)
        _say(f"预填模型清单：{', '.join(models)}")
        use_prefill = _ask("是否使用预填模型清单？（Y/n）")
        if use_prefill is None:
            _say("已取消。")
            return None
        if use_prefill.lower() not in ("n", "no"):
            return base_url, models
    models_input = _ask("模型清单（逗号分隔，如 deepseek-flash,deepseek-v4-pro）：")
    if models_input is None:
        _say("已取消。")
        return None
    models = [item.strip() for item in models_input.split(",") if item.strip()]
    if not models:
        _say("至少需要一个模型，已取消。")
        return None
    return base_url, models


def _apply_key_storage(session: _ConfigSession, entry: dict[str, Any], key: str) -> bool:
    """key 存储选择：1 明文写入 models.json 或 2 用户环境变量。

    选环境变量时 models.json 不落明文、该供应商已有明文一并移除；变量名统一为
    ``CRW_供应商名大写_API_KEY``（:func:`crw.config.provider_env_var`）。写入失败回退明文
    （有明确提示），避免 key 既不在文件也不在环境而丢失。

    Returns:
        key 是否已按用户选择落盘；中途取消返回 ``False``（调用方据此调整提示文案，
        不把"没写"报成"已更新"）。
    """
    var = config_mod.provider_env_var(entry["name"])
    _say("key 存储方式：")
    _say("  1. 明文写入 models.json（现状；注意文件权限与备份）")
    _say(f"  2. 环境变量 {var}（Windows 经 setx 写入并即时生效；macOS/Linux 打印 export 指令）")
    choice = _ask("选择（回车 = 1）：")
    while choice not in (None, "", "1", "2"):
        _say("无效选择，请重新输入。")
        choice = _ask("选择（回车 = 1）：")
    if choice is None:
        # 中断（EOF/Ctrl+C）不是确认：key 不落任何存储（保留现状，与改 key 的取消语义一致）
        _say("已取消存储选择，key 未写入（可稍后用 crw config 重新设置）。")
        return False
    if choice != "2":
        entry["api_key"] = key
        return True
    entry.pop("api_key", None)
    # 写入前撞名检查：其他未设 api_key 的供应商若回退到同一变量名且接入点主机不同，
    # 写入会把本 key 泄露给那些供应商的接入点（key 串用）
    provider_name = entry["name"]
    probe = {"name": provider_name, "base_url": entry.get("base_url", ""), "api_key": ""}
    others = [item for item in session.providers() if item is not entry and not item.get("api_key")]
    colliding = config_mod.conflicting_keyless_env_vars([*others, probe])
    if any(provider_name in names for names in colliding.values()):
        _say(
            f"环境变量 {var} 同时是其他未设 key 供应商（接入点主机不同）的回退变量，"
            "写入后它们会读到这个 key。"
        )
        answer = _ask("仍写入该环境变量？（y/N，拒绝则 key 改存明文）")
        if answer is None or answer.lower() not in ("y", "yes"):
            entry["api_key"] = key
            _say("key 已改存 models.json（明文）。")
            return True
    if not _write_env_var_key(var, key):
        entry["api_key"] = key
        return True
    return True


def _write_env_var_key(var: str, key: str) -> bool:
    """把 key 写入用户侧环境变量；返回是否写入成功。

    变量名先按 shell 标识符校验（非法回退明文——生成规则的未来漂移不应静默产生不可执行的
    export）。覆盖保护：变量已由本程序设置（登记在案）时直接更新；否则若当前环境已有同名
    变量（可能属用户或其它程序），询问确认后才覆盖，拒绝则回退明文——卸载时只清理登记在案
    的变量，绝不触碰他人变量。
    """
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", var):
        _say(f"环境变量名 {var!r} 非法，无法写入环境变量。")
        _say("key 已改存 models.json（明文）。")
        return False
    if var not in config_mod.registered_env_vars() and var in os.environ:
        answer = _ask(f"环境变量 {var} 已存在（可能由您或其它程序设置）。是否覆盖其值？（y/N）")
        if answer is None or answer.lower() not in ("y", "yes"):
            _say("已保留原值；key 将改存 models.json（明文）。")
            return False
    if sys.platform == "win32":
        if len(key) > 1000:
            _say("key 超长，setx 无法可靠写入（超过 1024 字符会静默截断）。")
            _say("key 已改存 models.json（明文）。")
            return False
        result = subprocess.run(["setx", var, key], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            _say(f"setx 写入失败：{detail}")
            _say("key 已改存 models.json（明文）；可稍后用 crw config 重新设置。")
            return False
        os.environ[var] = key  # setx 不影响已运行进程，当前进程立即生效
        _register_env_var(var)
        _say(f"已写入用户环境变量 {var}（当前进程即时生效，新终端自动可见）。")
        return True
    _say("请执行以下命令使 key 生效（写入 ~/.bashrc 或 ~/.zshrc 可持久化）：")
    # 用 shell 引用形式生成：key 里含 $、反引号或引号时，直接拼 "…" 会破坏命令（甚至被注入）
    _say(f"  export {var}={shlex.quote(key)}")
    os.environ[var] = key  # 当前进程即时生效；新终端须执行上述 export 后才可见
    _register_env_var(var)
    _say("注意：指令含 key 明文，勿粘贴到公共场合；未执行 export 前新终端暂不可见该变量。")
    return True


def _register_env_var(var: str) -> None:
    """登记本程序写入的环境变量名；失败不阻塞 key 设置，但须明示后果（卸载清单缺项）。"""
    try:
        config_mod.register_env_var(var)
    except Exception as exc:  # noqa: BLE001 — 登记是尽力而为的辅助数据，失败不得阻塞 key 设置
        _say(f"环境变量登记失败：{exc}——卸载时将无法自动清理 {var}，请记于手工清理。")


def _choose_entry(entries: list[Any], label: str) -> Any | None:
    """通用序号选择：打印编号清单，回车、取消或非法序号返回 ``None``。"""
    if not entries:
        _say("清单为空，已取消。")
        return None
    for index, item in enumerate(entries, start=1):
        name = item.get("name") if isinstance(item, dict) else item
        _say(f"  {index}. {name}")
    index = records_mod.parse_index(_ask(f"{label}序号（回车取消）："), len(entries))
    if index is None:
        _say("无效序号，已取消。")
        return None
    return entries[index]


def _config_add_provider(session: _ConfigSession) -> None:
    """添加 / 更新供应商（预设 + key 存储选择）；每步取消都不改盘。"""
    chosen = _choose_provider_type()
    if chosen is None:
        _say("已取消。")
        return
    kind, preset = chosen
    if kind == "preset":
        name = preset.name
    else:
        name = (_ask("供应商名称（不含 '/'，如 deepseek）：") or "").strip()
        if not name:
            _say("已取消。")
            return
        if "/" in name:
            _say("供应商名不得含 '/'。")
            return
    providers = session.providers()
    existing = next((item for item in providers if item.get("name") == name), None)
    if existing is not None:
        answer = _ask(f"供应商 '{name}' 已存在，是否更新其配置？（y/N）")
        if answer is None or answer.lower() not in ("y", "yes"):
            _say("已取消。")
            return
        new_base = _ask(f"API 接入点（回车保留 {existing.get('base_url') or '（未设置）'}）：")
        if new_base is None:
            _say("已取消。")
            return
        if new_base:
            existing["base_url"] = new_base
        new_key = _read_key("API key（输入不回显；回车保留现有 key）：")
        if new_key is None:
            _say("已取消 key 输入；接入点等其他修改仍会保存。")
        elif new_key:
            _apply_key_storage(session, existing, new_key)
        _say(f"供应商 '{name}' 已更新。")
        session.save()
        return
    result = _input_base_url_and_models(preset if kind == "preset" else None)
    if result is None:
        return  # 取消 / 无效输入的提示已在输入函数内给出
    base_url, models = result
    entry: dict[str, Any] = {"name": name, "base_url": base_url, "models": models}
    providers.append(entry)
    key = _read_key("API key（输入不回显；回车留空 = 回退环境变量）：")
    if key is None:
        providers.remove(entry)
        _say("已取消：未添加该供应商（key 输入被中断）。")
        return
    if key:
        _apply_key_storage(session, entry, key)
    else:
        entry["api_key"] = ""
        # 留空即依赖环境变量回退：归一化变量名与已有无 key 供应商撞名且接入点主机不同时，
        # 会互相读到对方的凭据（validate_config 同规则，写入时即暴露）
        conflicts = config_mod.conflicting_keyless_env_vars(providers)
        if conflicts:
            providers.remove(entry)
            detail = "；".join(
                f"{var} ← {', '.join(names)}" for var, names in sorted(conflicts.items())
            )
            _say(
                f"无法添加：该供应商将回退读取的环境变量名与已有供应商撞名（{detail}），"
                "会把一家的凭据发往另一家的接入点。请修改供应商名，或为其设置 key。"
            )
            return
    _say(f"供应商 '{name}' 已添加。")
    session.save()


def _config_remove_provider(session: _ConfigSession) -> None:
    """删除供应商；``default`` 引用被删供应商时警告并清除默认。"""
    providers = session.providers()
    entry = _choose_entry(providers, "要删除的供应商")
    if entry is None:
        return
    default = session.data.get("default") or ""
    referenced = default.startswith(f"{entry['name']}/")
    if referenced:
        answer = _ask(f"默认模型 {default} 引用该供应商，删除后将清除默认设置，继续？（y/N）")
        if answer is None or answer.lower() not in ("y", "yes"):
            _say("已取消。")
            return
    providers.remove(entry)
    if referenced:
        session.data.pop("default", None)
    _say(f"供应商 '{entry['name']}' 已删除。")
    session.save()


def _config_manage_models(session: _ConfigSession) -> None:
    """供应商的模型清单管理：增删模型。"""
    entry = _choose_entry(session.providers(), "要管理的供应商")
    if entry is None:
        return
    while True:
        _say(f"供应商 '{entry['name']}' 模型：{', '.join(entry['models'])}")
        choice = _ask("1) 添加模型  2) 删除模型  0) 返回：")
        if choice is None or choice == "0":
            return
        if choice == "1":
            _config_add_model(session, entry)
        elif choice == "2":
            _config_remove_model(session, entry)
        else:
            _say("无效选择。")


def _config_add_model(session: _ConfigSession, entry: dict[str, Any]) -> None:
    """添加模型；同名模型已存在时告知未重复添加（清单是纯名称数组，"覆盖"无意义）。"""
    name = (_ask("模型名称：") or "").strip()
    if not name:
        _say("已取消。")
        return
    if name in entry["models"]:
        _say(f"模型 '{name}' 已在该供应商的清单中，未重复添加。")
        return
    entry["models"].append(name)
    _say(f"模型 '{name}' 已添加到 '{entry['name']}'。")
    session.save()


def _config_remove_model(session: _ConfigSession, entry: dict[str, Any]) -> None:
    """删除模型；``default`` 引用被删组合时警告并清除默认。"""
    if len(entry["models"]) <= 1:
        _say("供应商至少保留一个模型，无法删除。")
        return
    name = _choose_entry(entry["models"], "要删除的模型")
    if name is None:
        return
    default = session.data.get("default") or ""
    if default == f"{entry['name']}/{name}":
        answer = _ask(f"默认模型 {default} 引用该模型，删除后将清除默认设置，继续？（y/N）")
        if answer is None or answer.lower() not in ("y", "yes"):
            _say("已取消。")
            return
        session.data.pop("default", None)
    entry["models"].remove(name)
    _say(f"模型 '{name}' 已从 '{entry['name']}' 删除。")
    session.save()


def _config_set_default(session: _ConfigSession) -> None:
    """设置默认模型（``default`` = ``供应商/模型``）。"""
    entry = _choose_entry(session.providers(), "供应商")
    if entry is None:
        return
    model = _choose_entry(entry["models"], "模型")
    if model is None:
        return
    session.data["default"] = f"{entry['name']}/{model}"
    _say(f"默认模型已设为 {entry['name']}/{model}。")
    session.save()


def _config_update_key(session: _ConfigSession) -> None:
    """修改供应商 key（不回显；存储选择同添加供应商）。"""
    entry = _choose_entry(session.providers(), "供应商")
    if entry is None:
        return
    key = _read_key("新 API key（输入不回显；回车取消）：")
    if not key:
        _say("已取消。")
        return
    saved = _apply_key_storage(session, entry, key)
    _say(
        f"供应商 '{entry['name']}' 的 key 已更新。"
        if saved
        else f"供应商 '{entry['name']}' 的 key 未改动。"
    )
    session.save()


def _maybe_import_legacy_config(session: _ConfigSession) -> None:
    """零配置时提示导入前作 ``~/.cra/models.json``（Design §6.2 #2 / D7）。

    只在"本工具尚无任何供应商"时探测前作配置的存在性，用户显式同意才导入；导入只读取
    前作文件的供应商名称、接入点、模型清单与 key（条目结构两者同构），**不修改前作文件**。
    导入前先按本项目的校验规则整表校验：前作配置与本工具冲突（如同名供应商）时给可读
    原因并保持原状，不产生半导入状态。
    """
    if session.providers():
        return
    data = config_mod.load_legacy_config()
    if data is None:
        return
    entries = config_mod.importable_providers(data)
    if not entries:
        return
    _say(f"检测到前作配置：{config_mod.legacy_config_path()}（{len(entries)} 个供应商）")
    answer = _ask("是否导入到本工具？（y/N；只读取前作文件，不修改它）")
    if answer is None or answer.lower() not in ("y", "yes"):
        _say("已跳过导入。")
        return
    candidate = list(entries)
    default = data.get("default")
    imported_default = ""
    if isinstance(default, str) and "/" in default:
        provider_name, _, model_name = default.partition("/")
        if any(
            item["name"] == provider_name and model_name in item["models"] for item in candidate
        ):
            imported_default = default
    probe = {"providers": [*session.providers(), *candidate]}
    if imported_default:
        probe["default"] = imported_default
    try:
        config_mod.validate_config(probe)
    except config_mod.ConfigError as exc:
        _say_error(f"导入失败：前作配置不符合本工具的配置规则 — {exc}")
        return
    session.data["providers"] = probe["providers"]
    if imported_default:
        session.data["default"] = imported_default
    session.save()
    _say(f"已导入 {len(candidate)} 个供应商。")
    if not imported_default:
        _say(
            "提示：前作的默认模型未一并导入（引用的供应商或模型不在清单内），可用菜单 4) 设置默认模型。"
        )


def _run_config_interactive() -> int:
    """``crw config``：交互式配置（增删供应商 / 模型、设默认、改 key）。"""
    try:
        data = config_mod.load_raw()
    except config_mod.ConfigError as exc:
        _say_error(f"crw config：配置错误 — {exc}")
        _say_error("请手工修复该文件，或删除后重试（将重建空配置）。")
        return _EXIT_ERROR
    session = _ConfigSession(data)
    _say(f"crw 配置（{config_mod.config_path()}）")
    _print_providers(data)
    _maybe_import_legacy_config(session)
    while True:
        _say(
            "\n菜单：1) 添加/更新供应商  2) 删除供应商  3) 管理供应商模型"
            "  4) 设置默认模型  5) 修改供应商 key  0) 退出"
        )
        choice = _ask("选择：")
        if choice is None:
            _say("\n退出配置。")
            return _EXIT_OK
        if choice == "1":
            _config_add_provider(session)
        elif choice == "2":
            _config_remove_provider(session)
        elif choice == "3":
            _config_manage_models(session)
        elif choice == "4":
            _config_set_default(session)
        elif choice == "5":
            _config_update_key(session)
        elif choice == "0":
            _say("退出配置。")
            return _EXIT_OK
        else:
            _say("无效选择。")


def _run_config_test() -> int:
    """``crw config test``：逐供应商发一次最小请求报告连通性（Design §8.3）。

    退出码：全部 OK 为 0，存在失败为 1（检查结论），配置损坏 / 无供应商为 2（运行错误）。
    """
    try:
        data = config_mod.load_raw()
    except config_mod.ConfigError as exc:
        print(f"crw config test：配置错误 — {exc}", file=sys.stderr)
        return _EXIT_ERROR
    providers = data.get("providers") or []
    if not providers:
        print("crw config test：尚无供应商配置，请先运行 crw config 添加供应商。", file=sys.stderr)
        return _EXIT_ERROR
    failures = 0
    for entry in providers:
        name = str(entry.get("name") or "")
        model_name = str(entry["models"][0])
        status, detail = config_mod.test_provider(entry)
        label = {"ok": "OK", "auth": "401（key 无效）", "timeout": "超时（网络不可达）"}.get(
            status, "失败"
        )
        mark = "✓" if status == "ok" else "✗"
        print(f"{mark} {name}（{model_name}）：{label}")
        if status != "ok":
            if detail:
                print(f"  {detail}")
            failures += 1
    return _EXIT_ISSUES if failures else _EXIT_OK


def _cmd_config(args: argparse.Namespace) -> int:
    """供应商配置入口：``test`` 走自检，省略则交互式配置（Design §8.3）。"""
    return _run_config_test() if args.action == "test" else _run_config_interactive()


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
