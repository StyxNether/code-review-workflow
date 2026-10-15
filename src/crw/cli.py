"""命令面：参数解析与子命令分发（run / runs / show / resume / review / config / graph / uninstall）。

只有本层接触 argparse 与终端输入；业务逻辑委托 workflow / agents 层。输出通道分流：
结论 / findings 清单 / 报告走 stdout，过程叙述、审查原文与错误走 stderr
（``--json`` 时 stdout 纯 JSON）。
"""

from __future__ import annotations

import argparse
import dataclasses
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
from . import rounds as rounds_mod
from . import thinking as thinking_mod
from . import tools as tools_mod
from . import uninstall as uninstall_mod
from . import usage as usage_mod
from .session import chat as session_chat_mod
from .session import state as session_state_mod
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
    """读取已安装发行版的版本号（版本以包元数据为准，不写进源码常量）。

    源码目录未安装时元数据缺失，返回可辨识的占位值，避免误报一个具体版本。
    """
    try:
        return importlib.metadata.version(_PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _max_iter_rounds(text: str) -> int:
    """解析修复轮次上限：0 ~ ``_MAX_ITER_LIMIT`` 的 ASCII 十进制整数。

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
    """记录 ``--max-iter`` 是否由命令行显式给出——轮次取值要区分"用户指定"与"默认"。"""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        setattr(namespace, self.dest, values)
        namespace.max_iter_source = state_mod.SOURCE_CLI


def _thinking_level(text: str) -> str:
    """解析 ``--thinking``：只认内置等级表（不要求该模型预配置过该挡位）。

    表外名称按参数错误处理（退出码 2 并列出可用挡位），不静默夹取。
    """
    if thinking_mod.is_builtin_level(text):
        return text.strip()
    raise argparse.ArgumentTypeError(
        f"未知思考挡位 {text!r}；可用挡位：{thinking_mod.available_levels_text()}"
    )


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    """添加 run / resume / review 共用的执行与模型覆盖选项。"""
    parser.add_argument("--no-exec", action="store_true", help="禁止运行代码核实")
    parser.add_argument("--model", metavar="名", help="临时覆盖模型")
    parser.add_argument("--provider", metavar="名", help="临时覆盖供应商")
    parser.add_argument(
        "--thinking",
        type=_thinking_level,
        metavar="挡位",
        help=(
            "快捷切换思考挡位（优先于配置，可与 --model/--provider 同用；"
            f"可用：{thinking_mod.available_levels_text()}）"
        ),
    )


def _add_workflow_options(
    parser: argparse.ArgumentParser,
    *,
    include_parallel_review: bool = True,
    include_out: bool = True,
) -> None:
    """添加 run / resume / chat 共用的工作流选项（resume 不提供 -f）。"""
    parser.add_argument(
        "--max-iter",
        type=_max_iter_rounds,
        default=engine_mod.DEFAULT_MAX_ITERATIONS,
        action=_MaxIterAction,
        metavar="n",
        help=f"修复轮次上限（默认 {engine_mod.DEFAULT_MAX_ITERATIONS}，"
        f"可取值 0–{_MAX_ITER_LIMIT}；0 = 只审不修）",
    )
    # 轮次语义与轮次数值是两件事：本选项**不**并入 max_iter_source（只给模式时数值仍走
    # "消息提取 > 会话设置 > 配置默认"）
    parser.add_argument(
        "--rounds-mode",
        dest="rounds_mode",
        choices=[rounds_mod.ROUNDS_UP_TO, rounds_mod.ROUNDS_EXACT],
        default=None,
        metavar="{up-to,exact}",
        help=("轮次语义：up-to = 干净即早退（默认）；exact = 跑满指定轮次，结尾恒为一次审查"),
    )
    parser.add_argument(
        "--severe-only",
        action="store_true",
        help="放宽干净判定：只把【严重】当阻塞（默认【严重】与【一般】都算阻塞）",
    )
    if include_parallel_review:
        parser.add_argument(
            "--parallel-review",
            dest="parallel_review",
            action="store_true",
            help="并行双视角审查（正确性 + 安全性同轮并行，汇合去重；审查调用量约翻倍）",
        )
    _add_model_options(parser)
    if include_out:
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
    parser.add_argument(
        "--per-file",
        dest="per_file",
        action="store_true",
        help="逐文件各开一次会话的旧快审形态（整项目一次会话的质量对照与安全阀）",
    )
    _add_model_options(parser)


def build_parser() -> argparse.ArgumentParser:
    """构造完整的命令行解析器。"""
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

    subparsers.add_parser("runs", help="列出运行记录（含会话轮次，多一列「会话」）")

    chat_parser = subparsers.add_parser(
        "chat", help="会话模式：与编排节点多轮对话，驱动三 Agent 工作流"
    )
    chat_inputs = chat_parser.add_mutually_exclusive_group()
    chat_inputs.add_argument(
        "-f",
        "--file",
        dest="file",
        metavar="文件",
        help="以既有代码文件开始会话（与 --resume 二选一）",
    )
    chat_inputs.add_argument(
        "--resume", dest="resume", metavar="会话号", help="恢复既有会话（与 -f 二选一）"
    )
    chat_parser.set_defaults(_subparser=chat_parser)
    # 会话内不提供 --parallel-review 与 --out（会话每轮的产物已在会话目录里，导出是一次性
    # 运行的语义）：不注册即明确报参数错误，而不是"解析了却不生效"
    _add_workflow_options(chat_parser, include_parallel_review=False, include_out=False)

    subparsers.add_parser("sessions", help="列出会话（会话号 / 时间 / 轮次数 / 最后状态）")

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


def _report_failure(command: str | None, exc: Exception) -> int:
    """兜底报错：单行可读提示 + 运行错误退出码。

    异常文本可能来自模型 / HTTP 栈（含请求细节）而终端输出常被重定向保存，故限长，
    并遮蔽 ``sk-`` 形态的密钥（覆盖常见供应商，尽力而为而非保证）。
    """
    detail = render.mask_secrets(f"{type(exc).__name__}: {exc}")
    if len(detail) > _MAX_ERROR_CHARS:
        detail = f"{detail[:_MAX_ERROR_CHARS]}…（已截断）"
    scope = f" {command}" if command else ""
    print(f"{_PROG}{scope}：运行失败 — {detail}", file=sys.stderr)
    return _EXIT_ERROR


# ---- 模型解析与运行装配 ----


def _resolve_model(
    model: str | None, provider: str | None, command: str, *, thinking: str | None = None
) -> Any:
    """解析本次运行的模型连接；失败给可读指引并返回 ``None``（调用方退出 2）。

    ``thinking``（``--thinking``）优先于配置里的挡位：切换模型的当次就能指定挡位，
    不要求该模型预配置过该挡位。
    """
    try:
        resolved = config_mod.resolve(model=model, provider=provider)
    except config_mod.ConfigError as exc:
        print(f"crw {command}：配置错误 — {render.mask_secrets(str(exc))}", file=sys.stderr)
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
    if thinking:
        resolved = dataclasses.replace(resolved, thinking=thinking)
    return resolved


def _legacy_config_hint() -> str:
    """零配置且检测到既有配置 ``~/.cra/models.json`` 时的导入提示。

    只探测存在性并提示，不读取内容——配置导入是用户显式发起的动作。
    """
    if config_mod.config_path().exists():
        return ""
    legacy = config_mod.legacy_config_path()
    if not legacy.exists():
        return ""
    return f"提示：检测到可导入的配置 {legacy}——运行 crw config 可选择导入（不会修改那个文件）。"


def _read_input_file(path: Path) -> tuple[str | None, int]:
    """读取输入文件（由命令面自己读，工作区落盘是引擎的事）；失败给可读原因并返回运行错误码。"""
    if not path.exists():
        print(f"crw run：运行失败 — 输入文件不存在：{path}", file=sys.stderr)
        return None, _EXIT_ERROR
    if path.is_dir():
        print(
            f"crw run：运行失败 — 输入是目录：{path}（-f 需要一个代码文件）",
            file=sys.stderr,
        )
        return None, _EXIT_ERROR
    try:
        size = path.stat().st_size
    except OSError as exc:
        print(f"crw run：运行失败 — 输入文件读取失败：{path}（{exc}）", file=sys.stderr)
        return None, _EXIT_ERROR
    if size > tools_mod.fs.MAX_READ_BYTES:
        print(
            f"crw run：运行失败 — 输入文件超过单次读取上限"
            f"（{tools_mod.fs.MAX_READ_BYTES // 1048576} MB）：{path}",
            file=sys.stderr,
        )
        return None, _EXIT_ERROR
    try:
        return path.read_text(encoding="utf-8"), _EXIT_OK
    except UnicodeDecodeError:
        print(f"crw run：运行失败 — 输入文件不是 UTF-8 文本：{path}", file=sys.stderr)
        return None, _EXIT_ERROR
    except OSError as exc:
        print(f"crw run：运行失败 — 输入文件读取失败：{path}（{exc}）", file=sys.stderr)
        return None, _EXIT_ERROR


def _cmd_run(args: argparse.Namespace) -> int:
    """运行工作流：解析模型 → 装配请求 → 驱动引擎 → 渲染结论。

    ``-f`` 输入由命令面读取，工作区内落盘交给引擎；``--out`` 在收尾后把最终代码与报告
    复制出去。轮次取值优先级：命令行 > 配置默认 > 内置。
    """
    requirement = (args.requirement or "").strip() or None
    source_path: Path | None = None
    source_text: str | None = None
    if args.file:
        source_path = Path(args.file)
        source_text, error_code = _read_input_file(source_path)
        if source_text is None:
            return error_code
    resolved = _resolve_model(args.model, args.provider, "run", thinking=args.thinking)
    if resolved is None:
        return _EXIT_ERROR
    try:
        rounds, source = _rounds_for_run(args)
    except config_mod.ConfigError as exc:
        print(f"crw run：配置错误 — {exc}", file=sys.stderr)
        return _EXIT_ERROR
    request = engine_mod.RunRequest(
        requirement=requirement,
        source_path=source_path,
        source_text=source_text,
        max_iterations=rounds,
        max_iterations_source=source,
        rounds_mode=args.rounds_mode or rounds_mod.ROUNDS_UP_TO,
        severe_only=bool(args.severe_only),
        no_exec=bool(args.no_exec),
        parallel_review=bool(args.parallel_review),
        resolved=resolved,
    )
    engine = engine_mod.WorkflowEngine(
        observer=render.RunRenderer(stream=not args.no_stream),
        confirm_deletions=_deletion_confirmer(),
    )
    result = engine.execute(request)
    render.render_result(result, command="run")
    if args.out:
        _export_outputs(result, Path(args.out))
    return result.exit_code


def _rounds_for_run(args: argparse.Namespace) -> tuple[int, str]:
    """``crw run`` 的轮次取值：命令行显式 > 配置默认 > 内置 3。

    配置默认来自 ``~/.crw/settings.json`` 的 ``rounds.default``（未配置时记"内置默认"来源）；
    配置损坏抛 :class:`crw.config.ConfigError`，由调用方按配置错误退出 2。
    """
    if args.max_iter_source:
        return int(args.max_iter), state_mod.SOURCE_CLI
    return _configured_rounds_fallback()


def _configured_rounds_fallback() -> tuple[int, str]:
    """配置默认轮次与其来源标签（一次读取 settings.json）：显式配置记 ``config``，否则记内置 ``default``。"""
    default, _limit, configured = config_mod.rounds_config()
    if configured is not None:
        return configured, state_mod.SOURCE_CONFIG
    return default, state_mod.SOURCE_DEFAULT


def _out_suffix(index: int) -> str:
    """``--out`` 重名序号：首个为空，其后 ``_2`` / ``_3`` …"""
    return "" if index == 1 else f"_{index}"


def _export_target(out_dir: Path, kind: str, suffix: str) -> Path:
    """``--out`` 的一类产物在目标目录里的落点（版本目录与报告共用同一序号）。"""
    if kind == "report":
        return out_dir / f"report{suffix}.md"
    return out_dir / f"code_final{suffix}"


def _export_outputs(result: engine_mod.RunResult, out_dir: Path) -> None:
    """``--out``：把最终版本目录与报告复制到指定目录（重名加序号）。

    版本目录整目录复制为 ``<out>/code_final/``（目录内含 manifest.json 与全部文件）；
    复制失败不影响运行结论：只向 stderr 警示，退出码仍按运行状态。
    """
    if result.run_dir is None:
        print("crw run：本次运行没有工作目录，--out 已跳过（无产物可复制）。", file=sys.stderr)
        return
    version_text = str(result.state.get("current_version_dir") or "")
    version_dir = Path(version_text)
    report_path = result.run_dir / "report.md"
    kinds: dict[str, Path] = {}
    # 空值必须显式挡住：Path("") 等价于 Path(".")，会把整个当前工作目录当产物复制
    if version_text and version_dir.is_dir():
        kinds["code_final"] = version_dir
    if report_path.is_file():
        kinds["report"] = report_path
    if not kinds:
        print("crw run：--out 无可复制产物（本次运行未产出代码版本或报告）。", file=sys.stderr)
        return
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"crw run：--out 目录创建失败 — {out_dir}（{exc}）", file=sys.stderr)
        return
    # 版本目录与报告成对加序号：两者一并落到同一序号，产物目录里不会出现"半新半旧"的一组
    index = 1
    while any(_export_target(out_dir, kind, _out_suffix(index)).exists() for kind in kinds):
        index += 1
    suffix = _out_suffix(index)
    targets = {kind: _export_target(out_dir, kind, suffix) for kind in kinds}
    for kind, target in targets.items():
        source = kinds[kind]
        try:
            if source.is_dir():
                for file_path in _version_copy_sources(source):
                    relative = file_path.relative_to(source)
                    destination = target / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    records_mod.atomic_write_text(
                        destination, file_path.read_text(encoding="utf-8")
                    )
            else:
                records_mod.atomic_write_text(target, source.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as exc:
            print(f"crw run：--out 复制失败 — {source} → {target}（{exc}）", file=sys.stderr)
            return
        print(f"已复制：{target}", file=sys.stderr)


def _version_copy_sources(version_dir: Path) -> list[Path]:
    """``--out`` 复制版本目录时的源文件清单：以 manifest.json 为准（含 manifest 自身）。

    manifest 缺失（旧目录 / 手工介入）时退回全目录复制，并排除字节码与隐藏 / 临时文件——
    执行核实落在版本目录里的副产物不得进交付产物。
    """
    manifest = version_dir / "manifest.json"
    entries: list[str] | None = None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        data = None
    if isinstance(data, dict) and isinstance(data.get("files"), list):
        names = [
            str(item.get("path"))
            for item in data["files"]
            if isinstance(item, dict) and isinstance(item.get("path"), str) and item.get("path")
        ]
        entries = [*names, "manifest.json"] if names else None
    if entries is not None:
        return [version_dir / name for name in entries]
    return [
        path
        for path in sorted(version_dir.rglob("*"))
        if path.is_file()
        and not path.name.startswith(".")
        and not path.name.endswith(".tmp")
        and path.suffix != ".pyc"
        and "__pycache__" not in path.relative_to(version_dir).parts
    ]


def _deletion_confirmer() -> engine_mod.DeletionConfirmer | None:
    """删除确认闸门：非交互（管道 / CI）返回 ``None``（一律保留）。"""
    if not sys.stdin.isatty():
        return None

    def confirm(requests: Sequence[Mapping[str, Any]]) -> bool:
        print("修复提出删除以下文件：", file=sys.stderr)
        for item in requests:
            reasons = "；".join(str(reason) for reason in item.get("reasons") or ())
            detail = f"（依据：{reasons}）" if reasons else "（本轮 findings 未提及该文件）"
            print(f"  - {item.get('path')} {detail}", file=sys.stderr)
        answer = render.ask_user("确认删除这些文件？（y/N，直接回车 = 保留）：")
        return answer is not None and answer.lower() in ("y", "yes")

    return confirm


# ---- crw chat / crw sessions：会话模式 ----


def _cmd_chat(args: argparse.Namespace) -> int:
    """会话模式：装配会话 IO 与模型，交给 session 层的主循环。

    命令面的职责只有三件：解析 ``-f`` 起始文件、按优先级链算好**初始轮次设置**、
    把真实终端 IO 与渲染器装进 :class:`crw.session.chat.ChatIO`；会话的时序全在 session 层。
    """
    resolved = _resolve_model(args.model, args.provider, "chat", thinking=args.thinking)
    if resolved is None:
        return _EXIT_ERROR
    file_input: tuple[Path, str] | None = None
    if args.file:
        path = Path(args.file)
        text, error_code = _read_input_file(path)
        if text is None:
            return error_code
        file_input = (path, text)
    try:
        seed_rounds, limit, configured = config_mod.rounds_config()
    except config_mod.ConfigError as exc:
        print(f"crw chat：配置错误 — {exc}", file=sys.stderr)
        print("请手工修复该文件，或删除后重试（将重建默认设置）。", file=sys.stderr)
        return _EXIT_ERROR
    # 命令行覆盖按字段生效：只给 --rounds-mode 时不钉住轮次数值
    overrides = session_chat_mod.RoundOverrides(
        max_iterations=int(args.max_iter) if args.max_iter_source else None,
        rounds_mode=args.rounds_mode,
    )
    initial = session_chat_mod.RoundSettings(
        max_iterations=seed_rounds,
        max_iterations_source=(
            state_mod.SOURCE_CONFIG if configured is not None else state_mod.SOURCE_DEFAULT
        ),
        rounds_mode=args.rounds_mode or rounds_mod.ROUNDS_UP_TO,
    )
    io = session_chat_mod.ChatIO(
        ask=render.ask_user,
        say=lambda line="": print(line),
        warn=lambda line: print(line, file=sys.stderr),
        renderer=lambda: render.RunRenderer(stream=not args.no_stream),
    )
    return session_chat_mod.run_chat(
        io=io,
        resolved=resolved,
        clock=engine_mod.local_now,
        rounds_limit=limit,
        initial_rounds=initial,
        overrides=overrides,
        file_input=file_input,
        resume_session_id=args.resume,
        severe_only=bool(args.severe_only),
        no_exec=bool(args.no_exec),
        confirm_deletions=_deletion_confirmer(),
    )


def _cmd_sessions(args: argparse.Namespace) -> int:
    """列出会话（会话号 / 时间 / 轮次数 / 最后状态）。"""
    render.render_sessions(session_state_mod.list_sessions())
    return _EXIT_OK


def _cmd_runs(args: argparse.Namespace) -> int:
    """列出运行记录（运行号 / 时间 / 输入摘要 / 状态 / 修复轮数 / 残留严重数）。"""
    render.render_runs(records_mod.list_runs())
    return _EXIT_OK


def _inherited_rounds(parent_state: Mapping[str, Any], parent_run_id: str) -> int | None:
    """续跑继承的轮次上限；父运行未记录返回 ``None``。

    校验上限用**代码内绝对天花板**（``config.ROUNDS_LIMIT_CEILING`` = 20）而不是
    命令行的 0–10 硬闸：父运行的轮次可能来自配置默认或会话内消息提取（那些路径的上限是
    settings 的 ``rounds.limit``，可达 20），用 10 校验会把合法记录判为非法。护栏仍在：
    ``state.json`` 被手改成 999 会被拒并如实提示，退回配置默认。
    """
    raw = parent_state.get("max_iterations")
    if raw is None:
        return None
    text = str(raw)
    if text.isascii() and text.isdecimal() and 0 <= int(text) <= config_mod.ROUNDS_LIMIT_CEILING:
        return int(text)
    print(
        f"crw resume：父运行 {parent_run_id} 的轮次上限（{raw!r}）不合法"
        f"（应为 0–{config_mod.ROUNDS_LIMIT_CEILING} 的整数），已改用配置默认。",
        file=sys.stderr,
    )
    return None


def _rounds_for_resume(
    args: argparse.Namespace, parent_state: Mapping[str, Any], run_id: str
) -> tuple[int, str, str]:
    """``crw resume`` 的轮次取值：命令行显式 > 继承父运行 > 配置默认 > 内置。

    ``--severe-only`` 与审查形态（``--parallel-review``）一并继承父运行——续跑是"接着修"，
    换判定标准或换审查方式会让结论不可比；``--no-exec`` 例外（它不进状态快照）。
    父运行的轮次数值只在未被命令行显式覆盖时才解析：否则那句"不合法 / 已回退"的提示
    描述的是一个并未发生的回退。
    """
    inherited: int | None = None
    if args.max_iter_source:
        rounds, source = int(args.max_iter), state_mod.SOURCE_CLI
    else:
        inherited = _inherited_rounds(parent_state, run_id)
        if inherited is not None:
            rounds, source = inherited, state_mod.SOURCE_PARENT
        else:
            rounds, source = _configured_rounds_fallback()
    # 轮次语义独立继承：即使命令行显式给了轮次数值，未给 --rounds-mode 时仍继承父运行的
    # 模式与审查形态——续跑是"接着修"，换判定标准会让结论不可比
    inherited_mode = str(parent_state.get("rounds_mode") or "")
    if args.rounds_mode:
        mode = args.rounds_mode
    elif inherited_mode in (rounds_mod.ROUNDS_UP_TO, rounds_mod.ROUNDS_EXACT):
        mode = inherited_mode
    else:
        mode = rounds_mod.ROUNDS_UP_TO
    return rounds, source, mode


def _cmd_show(args: argparse.Namespace) -> int:
    """展示运行详情；省略运行号时列出记录供数字选择。

    ``state.json`` 损坏时降级展示（摘要取自日志与产物清单）并给出退出码 2——口径是
    **展示照给，退出码照实**：脚本化消费能察觉异常，人读仍拿得到可用信息。
    """
    run_id = args.run_id
    if run_id is None:
        summaries = records_mod.list_runs()
        if not summaries:
            print("crw show：暂无运行记录（先执行 crw run）。", file=sys.stderr)
            return _EXIT_ERROR
        # 候选清单属交互过程：走 stderr，stdout 只留被选中记录的详情
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
    if records_mod.is_legacy_record(detail.state):
        print(
            "crw show：该记录来自单文件格式，已被多文件版本格式取代，不再支持展示。",
            file=sys.stderr,
        )
        return _EXIT_ERROR
    return _EXIT_OK


def _cmd_resume(args: argparse.Namespace) -> int:
    """续跑：以父运行的最新版本目录为新运行的起点。

    ``-f`` 不适用（起点由父运行决定）；父运行须存在 ``current_version_dir`` 且目录仍在，
    否则给出可读原因并退出 2。旧单文件格式记录不做兼容：给一行可读提示并退出 2。
    未显式给出的轮次设置**继承父运行**（来源标 ``parent``），父运行的
    ``--severe-only`` 与审查形态一并继承——续跑是"接着修"，换判定标准会让结论不可比。
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
    if records_mod.is_legacy_record(parent_state):
        print(
            "crw resume：该记录来自单文件格式，已被多文件版本格式取代，无法续跑。",
            file=sys.stderr,
        )
        return _EXIT_ERROR
    version_text = str(parent_state.get("current_version_dir") or "")
    version_dir = Path(version_text)
    if not version_text or not version_dir.is_dir():
        print(
            f"crw resume：父运行 {args.run_id} 没有可续跑的代码版本"
            f"（current_version_dir：{version_text or '（未记录）'}）——"
            "该运行可能在建码前就失败了。",
            file=sys.stderr,
        )
        return _EXIT_ERROR
    resolved = _resolve_model(args.model, args.provider, "resume", thinking=args.thinking)
    if resolved is None:
        return _EXIT_ERROR
    # 轮次取值来源：命令行显式 > 父运行继承 > 配置默认 > 内置
    try:
        rounds, rounds_source, rounds_mode = _rounds_for_resume(args, parent_state, args.run_id)
    except config_mod.ConfigError as exc:
        print(f"crw resume：配置错误 — {exc}", file=sys.stderr)
        return _EXIT_ERROR
    request = engine_mod.RunRequest(
        source_dir=version_dir,
        max_iterations=rounds,
        max_iterations_source=rounds_source,
        rounds_mode=rounds_mode,
        severe_only=bool(args.severe_only) or bool(parent_state.get("severe_only")),
        no_exec=bool(args.no_exec),
        parallel_review=bool(args.parallel_review) or bool(parent_state.get("parallel_review")),
        resolved=resolved,
        parent_run_id=args.run_id,
    )
    engine = engine_mod.WorkflowEngine(
        observer=render.RunRenderer(stream=not args.no_stream),
        confirm_deletions=_deletion_confirmer(),
    )
    result = engine.execute(request)
    render.render_result(result, command="resume")
    if args.out:
        _export_outputs(result, Path(args.out))
    return result.exit_code


def _cmd_graph(args: argparse.Namespace) -> int:
    """导出工作流 mermaid 流程图：默认 stdout，``-o`` 写文件。

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


#: 整项目清单的文件数闸门。清单每行（路径 + 行数）约 40–60 字节，200 行约 12 KB——
#: 清单本身按输入计费，且审查者按轮次预算最多读几十个文件，超过这个规模该由用户切分
#: 目标（分批审查）而不是把提示词撑爆。
_REVIEW_MAX_FILES = 200
#: 整项目清单的总字节闸门：低于单文件读取上限（``tools.fs.MAX_READ_BYTES`` = 10 MB），
#: 保证"扫描 + 行数统计"这一趟全量读入内存有界；8 MB 约合两万行代码，已远超一次会话
#: 读得完的量（``_REVIEW_PROJECT_TOOL_ROUNDS`` × ``MAX_READ_LINES`` 量级）。
_REVIEW_MAX_BYTES = 8 * 1024 * 1024
#: 整项目形态的工具轮次预算：一轮约读一个文件，缺省 10 轮只够小工程；给 30 轮才能
#: 顺 import / 调用链读到依赖末端。单文件与 stdin 的内联快审不改（仍用缺省 10）。
_REVIEW_PROJECT_TOOL_ROUNDS = 30
#: 清单排序里视为"程序入口"的文件名（审查者的读取顺序：入口与主干先看）
_REVIEW_ENTRY_NAMES = ("main.py", "__main__.py", "cli.py", "app.py", "index.py")
#: 跳过清单的打印上限（超大目录不得把 stderr 刷成日志）
_REVIEW_SKIP_NAMES = 20


def _review_roots(paths: Sequence[str]) -> list[Path]:
    """审查上下文本根：文件取其所在目录、目录取自身；stdin 退回进程启动目录。"""
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


def _as_absolute(path: Path) -> Path:
    """目标的绝对形态（只作路径算术用，不改用户输入的显示形态）。"""
    return Path(os.path.abspath(str(path)))


def _review_common_root(paths: Sequence[str]) -> Path | None:
    """整项目形态的**单一**上下文根 = 各目标的公共父目录；无法确定时返回 ``None``。

    唯一化是清单可读的前提：多根下 ``read_file("pkg/util.py")`` 会按根的顺序命中第一个
    同名候选，同一个名字对应两个文件时审查者读到哪个都可能错。公共父目录
    无法确定（跨盘符等）时不猜、不改语义，由调用方退回逐文件形态。
    """
    candidates: list[str] = []
    for raw in paths:
        path = Path(raw)
        base = path if path.is_dir() else path.parent
        candidates.append(str(_as_absolute(base)))
    if not candidates:
        return None
    try:
        return Path(os.path.commonpath(candidates))
    except ValueError:
        return None


def _review_skip_line(label: str, names: Sequence[str]) -> None:
    """跳过项的一行 stderr 说明（超限只打印前若干名并给出总数，不静默）。"""
    shown = "、".join(names[:_REVIEW_SKIP_NAMES])
    rest = f"…（共 {len(names)} 项）" if len(names) > _REVIEW_SKIP_NAMES else ""
    print(f"已跳过{label}：{shown}{rest}", file=sys.stderr)


def _review_data_dir_names() -> frozenset[str]:
    """数据目录名：运行记录与配置不是待审代码，整目录跳过。"""
    return frozenset({config_mod.data_dir().name, ".crw", ".cra"})


def _scan_review_dir(directory: Path) -> list[Path]:
    """递归展开目录目标为文本文件清单。

    跳过**隐藏文件与目录**（含 ``.env`` 等敏感文件，其内容不得进入审查）、**忽略名单目录**
    与数据目录（依赖 / 构建 / 缓存产物对审查没有价值且会撑爆清单）、**二进制文件**；
    三类跳过各打一行 stderr 说明（用户看得见"哪些没被审"）。
    """
    data_names = _review_data_dir_names()
    selected: list[Path] = []
    ignored: list[str] = []
    hidden: list[str] = []
    binary: list[str] = []
    for current, dir_names, file_names in os.walk(directory):
        base = Path(current)
        kept: list[str] = []
        for name in sorted(dir_names):
            if name.startswith("."):
                hidden.append(name)
                continue
            if name in tools_mod.fs.IGNORED_DIRS or name in data_names:
                ignored.append(name)
                continue
            kept.append(name)
        # 原地改写 dir_names：os.walk 只递归留下来的子目录（剪枝，不是扫完再筛）
        dir_names[:] = kept
        for name in sorted(file_names):
            item = base / name
            if name.startswith("."):
                hidden.append(name)
                continue
            if not tools_mod.fs.looks_textual(item):
                binary.append(item.relative_to(directory).as_posix())
                continue
            selected.append(item)
    if ignored:
        _review_skip_line("忽略名单与数据目录", sorted(set(ignored)))
    if hidden:
        _review_skip_line("隐藏文件与目录", sorted(set(hidden)))
    if binary:
        _review_skip_line("二进制文件", sorted(set(binary)))
    return selected


def _apply_review_gate(files: Sequence[Path]) -> tuple[list[Path], list[Path]]:
    """规模闸门：按目标顺序装入清单，超出文件数或总字节的部分退回"未列入"并如实打印。

    宁可少覆盖也要让规模有界，但**未列入清单的文件必须逐个可见**——静默丢文件会让
    "审查通过"被读成"全都看过"。
    """
    kept: list[Path] = []
    omitted: list[Path] = []
    total = 0
    for item in files:
        try:
            size = item.stat().st_size
        except OSError as exc:
            raise _ReviewInputError(f"无法读取文件 {item}：{exc}") from exc
        if len(kept) >= _REVIEW_MAX_FILES or total + size > _REVIEW_MAX_BYTES:
            omitted.append(item)
            continue
        total += size
        kept.append(item)
    if omitted:
        _review_skip_line(
            f"超出清单规模上限的文件（{_REVIEW_MAX_FILES} 个 / "
            f"{_REVIEW_MAX_BYTES // 1048576} MB，本次不覆盖）",
            [str(item) for item in omitted],
        )
    return kept, omitted


def _gather_review_files(paths: Sequence[str]) -> tuple[list[Path], list[Path]]:
    """展开审查目标为待审文件（目录递归、跨目标去重）；返回（列入清单, 因闸门未列入）。"""
    found: list[Path] = []
    seen: set[str] = set()
    for raw in paths:
        if raw == "-":
            continue
        path = Path(raw)
        if not path.exists():
            raise _ReviewInputError(f"路径不存在：{path}")
        for item in _scan_review_dir(path) if path.is_dir() else [path]:
            key = str(_as_absolute(item))
            if key in seen:
                continue
            seen.add(key)
            found.append(item)
    if not found and "-" not in paths:
        raise _ReviewInputError("目标里没有可审查的文本文件（隐藏文件、二进制与忽略目录不入库）。")
    return _apply_review_gate(found)


def _relative_to_root(path: Path, root: Path) -> str:
    """相对上下文根的 posix 路径（清单与 ``location`` 的口径）；不在根内时退回绝对路径。"""
    absolute = _as_absolute(path)
    if not absolute.is_relative_to(root):
        return absolute.as_posix()
    return absolute.relative_to(root).as_posix()


def _review_listing_priority(entry: tuple[str, int]) -> tuple[int, int, int, str]:
    """清单排序键：入口文件 > 非测试文件 > 行数多的在前 > 路径字典序。

    读取预算有限时清单顺序就是实际读取顺序，入口与主干文件被依赖最多，先看它们才能
    在后续轮次里顺链验证跨文件一致性。
    """
    path, lines = entry
    segments = path.split("/")
    name = segments[-1]
    is_entry = 0 if name in _REVIEW_ENTRY_NAMES else 1
    is_test = 1 if (name.startswith("test_") or "tests" in segments[:-1]) else 0
    return (is_entry, is_test, -lines, path)


@dataclasses.dataclass(frozen=True)
class _ReviewScope:
    """整项目形态的审查范围：单一上下文根、任务清单、未列入清单者与规模读数。"""

    root: Path
    listing: tuple[tuple[str, int], ...]
    omitted: tuple[str, ...]
    total_lines: int

    @property
    def file_count(self) -> int:
        """清单里的文件数（提示词、报告标题与进度行共用）。"""
        return len(self.listing)


def _build_review_scope(root: Path, files: Sequence[Path], omitted: Sequence[Path]) -> _ReviewScope:
    """整项目形态的任务清单：``(相对根的路径, 行数)``，按建议读取优先级排序。"""
    entries: list[tuple[str, int]] = []
    for item in files:
        lines = len(tools_mod.fs.split_lines(_read_review_text(item)))
        entries.append((_relative_to_root(item, root), lines))
    entries.sort(key=_review_listing_priority)
    return _ReviewScope(
        root=root,
        listing=tuple(entries),
        omitted=tuple(_relative_to_root(item, root) for item in omitted),
        total_lines=sum(count for _, count in entries),
    )


def _review_contents(paths: Sequence[str], files: Sequence[Path]) -> list[tuple[str, str]]:
    """内联形态（单文件 / stdin / ``--per-file``）的入参：``(显示名, 内容)``。

    显示名保留用户输入形态（location 引用与报告标题用它），stdin 固定 ``<stdin>``。
    """
    collected: list[tuple[str, str]] = []
    if "-" in paths:
        collected.append(("<stdin>", _read_stdin_text()))
    collected.extend((str(item), _read_review_text(item)) for item in files)
    return collected


def _wants_project_review(args: argparse.Namespace) -> bool:
    """整项目一次会话的选形：目标是目录**或**多于一个路径，且未 ``--per-file``、不含 stdin。

    单文件与 stdin 保持内联快审（内容已在手，无需再读）；``--per-file``
    是显式回退开关，用于与整项目形态做质量对照。
    """
    if args.per_file or "-" in args.paths:
        return False
    if len(args.paths) > 1:
        return True
    return Path(args.paths[0]).is_dir()


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
    """单 Agent 审查：整项目一次会话，或单文件 / stdin / ``--per-file`` 的内联快审。

    stdout 只承载结论与 ``--json``（过程与错误走 stderr）；退出码 0 完成、1 为 ``--json``
    模式下存在【严重】、2 为运行错误（整项目形态下这一次调用失败即整批失败；内联形态下
    逐文件隔离，任一失败退出 2）。**不产生运行记录**（无 run_id）。
    """
    project = _wants_project_review(args)
    scope: _ReviewScope | None = None
    contents: list[tuple[str, str]] = []
    try:
        files, omitted = _gather_review_files(args.paths)
        if project:
            root = _review_common_root(args.paths)
            if root is None:
                print(
                    "crw review：各目标没有公共父目录，已退回逐文件审查形态。",
                    file=sys.stderr,
                )
            else:
                scope = _build_review_scope(root, files, omitted)
        if scope is None:
            contents = _review_contents(args.paths, files)
    except _ReviewInputError as exc:
        print(f"crw review：{exc}", file=sys.stderr)
        return _EXIT_ERROR
    resolved = _resolve_model(args.model, args.provider, "review", thinking=args.thinking)
    if resolved is None:
        return _EXIT_ERROR
    observer = render.RunRenderer(stream=True)
    if scope is not None:
        runtime, registry = agents_mod.build_review_runtime(
            resolved=resolved,
            roots=[scope.root],
            no_exec=bool(args.no_exec),
            emit=observer.on_event,
            on_delta=observer.on_delta,
            max_tool_rounds=_REVIEW_PROJECT_TOOL_ROUNDS,
        )
        return _review_project_session(args, runtime, registry, scope)
    runtime, registry = agents_mod.build_review_runtime(
        resolved=resolved,
        roots=_review_roots(args.paths),
        no_exec=bool(args.no_exec),
        emit=observer.on_event,
        on_delta=observer.on_delta,
    )
    return _review_per_file_session(args, runtime, registry, contents)


def _write_review_report(args: argparse.Namespace, markdown: str) -> bool:
    """``-o`` 报告落盘：成功返回 ``True``；失败给一行可读原因并返回 ``False``（退出码 2）。"""
    target = Path(args.out)
    try:
        records_mod.atomic_write_text(target, markdown)
    except OSError as exc:
        print(f"crw review：报告写入失败 — {target}（{exc}）", file=sys.stderr)
        return False
    print(f"报告已写入：{target}", file=sys.stderr)
    return True


def _review_project_session(
    args: argparse.Namespace,
    runtime: agents_mod.AgentRuntime,
    registry: agents_mod.ToolRegistry,
    scope: _ReviewScope,
) -> int:
    """整项目形态：一次会话产出全工程结论（失败隔离由"逐文件"变"整批"）。

    只调用一次，因此没有"部分成功"：调用失败或 ``--json`` 输出不合契约都给 stderr 一行
    可读原因 + 退出码 2，stdout 不留半截结论。
    """
    print(
        f"正在审查 {scope.root.as_posix()}（整项目一次会话，{scope.file_count} 个文件）…",
        file=sys.stderr,
    )
    try:
        answer = agents_mod.review_project_once(
            runtime,
            registry,
            listing=scope.listing,
            as_json=bool(args.as_json),
            allow_exec=not bool(args.no_exec),
            omitted=scope.omitted,
            total_lines=scope.total_lines,
        )
    except KeyboardInterrupt:
        # 中断：已花掉的用量照样回显——钱花了就该看得见
        print("crw review：已中断（整项目一次会话），未产出结论。", file=sys.stderr)
        _report_review_usage(runtime)
        return _EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 — 一次调用即整批：如实报错，不产出部分结论
        print(f"crw review：整项目审查失败 — {render.mask_secrets(str(exc))}", file=sys.stderr)
        _report_review_usage(runtime)
        return _EXIT_ERROR
    severe = False
    if args.as_json:
        try:
            report = contracts_mod.parse_review_report(answer)
        except contracts_mod.ContractViolation as exc:
            print(f"crw review：结构化输出无效 — {exc}", file=sys.stderr)
            _report_review_usage(runtime)
            return _EXIT_ERROR
        # 一位审查者一次结论：报告里的多条 files 条目全部采纳，不做逐文件拼接
        entries = report["files"]
        merged = contracts_mod.merge_review_reports(entries)
        summary = str(report.get("summary") or "") or str(merged.get("summary") or "")
        payload = {"files": entries, "summary": summary}
        severe = any(
            finding.get("severity") == contracts_mod.SEVERE
            for entry in entries
            for finding in entry["findings"]
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        body = render.review_json_report(payload)
    else:
        print(answer)
        body = answer
    if args.out:
        markdown = render.review_project_report(
            targets=args.paths, file_count=scope.file_count, body=body
        )
        if not _write_review_report(args, markdown):
            _report_review_usage(runtime)
            return _EXIT_ERROR
    _report_review_usage(runtime)
    return _EXIT_ISSUES if (args.as_json and severe) else _EXIT_OK


def _review_per_file_session(
    args: argparse.Namespace,
    runtime: agents_mod.AgentRuntime,
    registry: agents_mod.ToolRegistry,
    files: list[tuple[str, str]],
) -> int:
    """内联形态（单文件 / stdin / ``--per-file``）：逐文件各开一次会话，逐文件失败隔离。"""
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
        except KeyboardInterrupt:
            # 中断整批审查：已花掉的用量照样回显——钱花了就该看得见
            print(
                f"crw review：已中断（第 {position + 1} / {len(files)} 个文件），未产出结论。",
                file=sys.stderr,
            )
            _report_review_usage(runtime)
            return _EXIT_ERROR
        except Exception as exc:  # noqa: BLE001 — 单文件失败继续批量；最终退出码 2
            print(
                f"[审查失败] {display_name}：{render.mask_secrets(str(exc))}",
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
            for position, ((display_name, _), answer) in enumerate(
                zip(files, text_answers, strict=True)
            ):
                if position:
                    print()
                print(f"===== {display_name} =====")
                print(_answer_or_placeholder(answer))
        else:
            for answer in text_answers:
                print(_answer_or_placeholder(answer))
        markdown = render.review_text_report(files, text_answers)
    if args.out and not _write_review_report(args, markdown):
        failed = True
    _report_review_usage(runtime)
    if failed:
        return _EXIT_ERROR
    return _EXIT_ISSUES if (args.as_json and severe) else _EXIT_OK


def _report_review_usage(runtime: Any) -> None:
    """``crw review`` 的用量一行（stderr）：本命令不产生运行记录，用量只能在结束时回显。

    ``crw run`` / ``crw chat`` 的用量进 state.json 与 report.md，这里给的是
    同一套数的即时读数——批量审查花掉多少输入 / 输出 token、缓存命中多少，不看记录也要能知道。
    """
    record = usage_mod.totals(runtime.drain_usage())
    if usage_mod.is_empty(record):
        return
    print(f"[用量] {usage_mod.format_line(record)}", file=sys.stderr)
    hint = usage_mod.hint(record)
    if hint:
        print(f"[提示] {hint}", file=sys.stderr)


# ---- crw config：供应商配置 / 连通性自检 ----


def _say(text: str = "") -> None:
    """配置交互的输出（菜单与结论）：stdout（结论通道）。"""
    print(text)


def _say_error(text: str) -> None:
    """配置交互的错误（配置损坏 / 写盘失败）：stderr（错误通道）。

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
        thinking_notes = []
        for item in entry.get("models") or []:
            if isinstance(item, dict) and item.get("thinking"):
                selected = item["thinking"].get("selected")
                thinking_notes.append(f"{item.get('name')}（思考：{selected or '未设'}）")
        note = f"　挡位：{'、'.join(thinking_notes)}" if thinking_notes else ""
        _say(
            f"  - {entry['name']}　base_url={entry['base_url']}"
            f"　key={_mask_key(entry.get('api_key'), entry['name'])}"
            f"　模型：{', '.join(config_mod.model_names(entry.get('models') or []))}"
            f"{note}"
        )
        thinking_note = config_mod.thinking_note_for(str(entry.get("name") or ""))
        if thinking_note:
            _say(f"    思考说明：{thinking_note}")


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
        _say(f"预填模型清单：{', '.join(config_mod.model_names(models))}")
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
        if not _apply_key_storage(session, entry, key):
            _say(
                "提示：key 未写入任何存储（存储选择被取消）——该供应商将走环境变量回退，"
                "可稍后用菜单 5 补设。"
            )
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
        _say(f"供应商 '{entry['name']}' 模型：{', '.join(config_mod.model_names(entry['models']))}")
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
    if name in config_mod.model_names(entry["models"]):
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
    chosen = _choose_entry(entry["models"], "要删除的模型")
    if chosen is None:
        return
    name = config_mod.model_entry_name(chosen)
    if name is None:
        _say("该模型条目缺少名称，无法删除。")
        return
    default = session.data.get("default") or ""
    if default == f"{entry['name']}/{name}":
        answer = _ask(f"默认模型 {default} 引用该模型，删除后将清除默认设置，继续？（y/N）")
        if answer is None or answer.lower() not in ("y", "yes"):
            _say("已取消。")
            return
        session.data.pop("default", None)
    entry["models"].remove(chosen)
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
    model_name = config_mod.model_entry_name(model)
    if model_name is None:
        _say("该模型条目缺少名称，无法设为默认。")
        return
    session.data["default"] = f"{entry['name']}/{model_name}"
    _say(f"默认模型已设为 {entry['name']}/{model_name}。")
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


def _config_manage_thinking(session: _ConfigSession) -> None:
    """模型思考挡位配置：选供应商 → 模型 → 勾选挡位 → 选默认。

    挡位名以内置等级表为主、允许自定义名（发送侧经 extra_body 原样透传）；
    未能确认挡位通道的供应商不做预填——查不到的交给用户自己填。
    未配置挡位的模型不发送任何思考参数。
    """
    entry = _choose_entry(session.providers(), "供应商")
    if entry is None:
        return
    models = entry["models"]
    chosen = _choose_entry(models, "模型")
    if chosen is None:
        return
    model_name = config_mod.model_entry_name(chosen)
    if model_name is None:
        _say("该模型条目缺少名称，无法配置挡位。")
        return
    current = config_mod.model_thinking_config(models, model_name) or {}
    levels = list(current.get("levels") or [])
    selected = current.get("selected")
    _say(
        f"模型 '{model_name}' 当前思考挡位：{selected or '（未配置）'}"
        f"（可选挡位：{', '.join(levels) or '（无）'}）"
    )
    _say(f"内置挡位表：{thinking_mod.available_levels_text()}")
    _say(
        "说明：off/none/disabled = 关闭思考；minimal…high = reasoning_effort；"
        "max = 最高档；enabled = 布尔开关；budget:<n> = 数值预算；其他名称按供应商参数原样透传。"
    )
    raw = _ask(
        "要启用的挡位（逗号分隔挡位名，如 off,low,high；回车保留现状；输入 清空 移除全部）："
    )
    if raw is None:
        _say("已取消。")
        return
    text = raw.strip()
    if text == "":
        new_levels = levels
    elif text in ("清空", "clear"):
        # "清空"= 一次性清除该模型的全部挡位配置（含默认挡位）：只清清单却留着
        # 默认值会得到自相矛盾的配置（默认不在可选清单内），这里不制造这种半状态。
        # 清空后不再发送任何思考参数——对"默认就是思考开"的供应商，这等于把挡位交回供应商
        # 默认（比关闭更贵），所以要提醒一句，并给出"跟随默认"这条更省的路。
        _set_model_thinking(session, models, model_name, [], None)
        _say(
            "注意：清空后不发送任何思考参数，部分供应商（如 DeepSeek）的出厂默认是"
            "「思考开、high」。"
        )
        _say("      想按本产品的默认关闭思考，用「跟随默认」（或把默认挡位填成 off）。")
        return
    else:
        new_levels = []
        for token in (item.strip() for item in text.split(",")):
            if not token:
                continue
            if token.startswith("budget:") and not thinking_mod.is_budget_level(token):
                _say(f"budget 挡位形如 budget:<正整数>，收到 {token!r}，已取消。")
                return
            if token not in new_levels:
                new_levels.append(token)
    selected_raw = _ask(
        f"默认挡位（须在上面清单内或 off；输入 跟随默认 用预设默认值；"
        f"回车保留 {selected or '（无）'}）："
    )
    if selected_raw is None:
        _say("已取消。")
        return
    if selected_raw.strip() in ("跟随默认", "default"):
        preset_default = config_mod.preset_default_thinking(
            str(entry.get("name") or ""), model_name
        )
        if preset_default is None:
            _say("该模型在预设里没有可靠的默认挡位（查不到就不猜），请手选挡位或输入 清空。")
            return
        new_selected = preset_default
        _say(f"将按预设默认挡位设为 {new_selected}。")
    else:
        new_selected = selected_raw.strip() or selected
    if new_selected and new_selected != "off" and new_selected not in new_levels:
        _say(f"默认挡位 {new_selected!r} 不在挡位清单内，已取消（请先启用它再设默认）。")
        return
    _set_model_thinking(session, models, model_name, new_levels, new_selected)


def _set_model_thinking(
    session: _ConfigSession,
    models: list[Any],
    model_name: str,
    levels: list[str],
    selected: str | None,
) -> None:
    """把挡位配置写回模型条目：无挡位退回纯字符串形态，有挡位写对象形态。"""
    item = {"name": model_name}
    if levels or selected:
        payload: dict[str, Any] = {}
        if levels:
            payload["levels"] = levels
        if selected:
            payload["selected"] = selected
        item["thinking"] = payload
    index = next(
        (
            position
            for position, value in enumerate(models)
            if config_mod.model_entry_name(value) == model_name
        ),
        None,
    )
    if index is None:
        models.append(item if len(item) > 1 else model_name)
    else:
        models[index] = item if len(item) > 1 else model_name
    if levels or selected:
        _say(f"模型 '{model_name}' 的思考挡位已保存（默认：{selected or '（未设）'}）。")
    else:
        _say(f"模型 '{model_name}' 的思考挡位已清空（不发送任何思考参数）。")
    session.save()


def _maybe_import_legacy_config(session: _ConfigSession) -> None:
    """零配置时提示导入既有配置 ``~/.cra/models.json``。

    只在"本工具尚无任何供应商"时探测该文件的存在性，用户显式同意才导入；导入只读取
    它的供应商名称、接入点、模型清单与 key（条目结构两者同构），**不修改那个文件**。
    导入前先按本工具的校验规则整表校验：与现有配置冲突（如同名供应商）时给可读
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
    _say(f"检测到可导入的配置：{config_mod.legacy_config_path()}（{len(entries)} 个供应商）")
    answer = _ask("是否导入到本工具？（y/N；只读取该文件，不修改它）")
    if answer is None or answer.lower() not in ("y", "yes"):
        _say("已跳过导入。")
        return
    candidate = list(entries)
    default = data.get("default")
    imported_default = ""
    if isinstance(default, str) and "/" in default:
        provider_name, _, model_name = default.partition("/")
        if any(
            item["name"] == provider_name and model_name in config_mod.model_names(item["models"])
            for item in candidate
        ):
            imported_default = default
    probe = {"providers": [*session.providers(), *candidate]}
    if imported_default:
        probe["default"] = imported_default
    try:
        config_mod.validate_config(probe)
    except config_mod.ConfigError as exc:
        _say_error(f"导入失败：该配置不符合本工具的配置规则 — {exc}")
        return
    session.data["providers"] = probe["providers"]
    if imported_default:
        session.data["default"] = imported_default
    session.save()
    _say(f"已导入 {len(candidate)} 个供应商。")
    if not imported_default:
        _say(
            "提示：该配置的默认模型未一并导入（引用的供应商或模型不在清单内），可用菜单 4) 设置默认模型。"
        )


def _print_rounds_settings() -> None:
    """当前运行默认值（轮次）概览；读取失败按配置错误提示（不中断菜单）。"""
    try:
        default, limit = config_mod.rounds_settings()
    except config_mod.ConfigError as exc:
        _say_error(f"运行默认值读取失败：{exc}")
        return
    _say(
        f"轮次默认：{default}（命令行与会话消息都未指定时使用）；"
        f"轮次上限：{limit}（会话内消息提取值的夹取上限，天花板 {config_mod.ROUNDS_LIMIT_CEILING}）"
    )


def _parse_settings_int(raw: str) -> int | None:
    """解析设置项的整数输入（ASCII 十进制，0–天花板）；非法给可读提示并返回 ``None``。

    范围在此就地拦下（而不是等写入时由校验层报错）：提示贴着用户刚输入的那一项，
    菜单里不会出现"走到最后一步才被拒"的落差。
    """
    text = raw.strip()
    if not text.isascii() or not text.isdecimal() or int(text) > config_mod.ROUNDS_LIMIT_CEILING:
        _say_error(f"需要 0–{config_mod.ROUNDS_LIMIT_CEILING} 的整数，收到 {raw!r}")
        return None
    return int(text)


def _save_rounds(default: int, limit: int) -> None:
    """保存轮次设置（写 settings.json，与 models.json 分开）；违规由校验层给出原因。"""
    try:
        config_mod.save_settings({"rounds": {"default": default, "limit": limit}})
    except (config_mod.ConfigError, OSError) as exc:
        _say_error(f"写入失败：{exc}")
        return
    _say(f"已保存：默认修复轮次 {default}，轮次上限 {limit}（{config_mod.settings_path()}）")


def _config_set_rounds_default() -> None:
    """菜单 7：默认修复轮次（命令行与会话消息都未指定时使用）。"""
    try:
        default, limit = config_mod.rounds_settings()
    except config_mod.ConfigError as exc:
        _say_error(f"运行默认值读取失败：{exc}")
        return
    raw = _ask(f"默认修复轮次（当前 {default}；须满足 0 ≤ 值 ≤ 上限 {limit}；回车保留）：")
    if raw is None:
        _say("已取消。")
        return
    if not raw.strip():
        _say("未改动。")
        return
    value = _parse_settings_int(raw)
    if value is None:
        return
    if value > limit:
        _say_error(f"默认修复轮次不能超过当前上限 {limit}（可先用菜单 8) 抬高上限）。")
        return
    _save_rounds(value, limit)


def _config_set_rounds_limit() -> None:
    """菜单 8：轮次上限（会话内消息提取值的夹取上限，天花板 20）。"""
    try:
        default, limit = config_mod.rounds_settings()
    except config_mod.ConfigError as exc:
        _say_error(f"运行默认值读取失败：{exc}")
        return
    raw = _ask(
        f"轮次上限（当前 {limit}；须满足默认 {default} ≤ 值 ≤ "
        f"{config_mod.ROUNDS_LIMIT_CEILING}；回车保留）："
    )
    if raw is None:
        _say("已取消。")
        return
    if not raw.strip():
        _say("未改动。")
        return
    value = _parse_settings_int(raw)
    if value is None:
        return
    if value < default:
        _say_error(f"轮次上限不能低于当前默认 {default}（可先用菜单 7) 调低默认）。")
        return
    _save_rounds(default, value)


def _run_config_interactive() -> int:
    """``crw config``：交互式配置（增删供应商 / 模型、设默认、改 key、思考挡位、轮次）。"""
    try:
        data = config_mod.load_raw()
    except config_mod.ConfigError as exc:
        _say_error(f"crw config：配置错误 — {exc}")
        _say_error("请手工修复该文件，或删除后重试（将重建空配置）。")
        return _EXIT_ERROR
    session = _ConfigSession(data)
    _say(f"crw 配置（{config_mod.config_path()}）")
    _print_providers(data)
    _print_rounds_settings()
    _maybe_import_legacy_config(session)
    while True:
        _say(
            "\n菜单：1) 添加/更新供应商  2) 删除供应商  3) 管理供应商模型"
            "  4) 设置默认模型  5) 修改供应商 key  6) 模型思考挡位"
            "  7) 默认修复轮次  8) 轮次上限  0) 退出"
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
        elif choice == "6":
            _config_manage_thinking(session)
        elif choice == "7":
            _config_set_rounds_default()
        elif choice == "8":
            _config_set_rounds_limit()
        elif choice == "0":
            _say("退出配置。")
            return _EXIT_OK
        else:
            _say("无效选择。")


def _run_config_test() -> int:
    """``crw config test``：逐供应商发一次最小请求报告连通性。

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
        models = entry["models"]
        model_name = config_mod.model_entry_name(models[0]) or "（未命名模型）"
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
    """供应商配置入口：``test`` 走自检，省略则交互式配置。"""
    return _run_config_test() if args.action == "test" else _run_config_interactive()


_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "run": _cmd_run,
    "chat": _cmd_chat,
    "sessions": _cmd_sessions,
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
    而混淆"达限残留"语义。

    Args:
        argv: 命令行参数（不含程序名）；省略时取 ``sys.argv[1:]``。

    Returns:
        进程退出码：0 完成 / 1 达修复轮次上限仍有阻塞问题（默认含【一般】，
        ``--severe-only`` 时只算【严重】）/ 2 运行错误。
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
