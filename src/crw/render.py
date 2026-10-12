"""终端渲染管线：行级流式、分级着色与节点阶段展示。

渲染输入由工作流事件驱动（节点开始 / 工具边界 / 重试 / 清单分区 / 节点结束），
不依赖框架的流式元数据；仅命令面使用本模块（Design §3.1、§8.4）。

M1 为简版渲染：阶段标题 + 事件行 + 清单分区（结论与清单走 stdout，过程叙述走 stderr，
Design §8.4）；行级流式与分级着色管线（复用矩阵 #6）随 M2/M3 接入，接口不变。
"""

from __future__ import annotations

import sys
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any, TextIO

from .workflow import contracts
from .workflow import engine as engine_mod
from .workflow import records as records_mod
from .workflow import state as state_mod

_RULE_WIDTH = 52


def display_width(text: str) -> int:
    """终端显示宽度：东亚宽/全角字符计 2，组合字符计 0（列表对齐用）。"""
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
    return width


def pad(text: str, width: int) -> str:
    """按显示宽度右侧补空格（列对齐）。"""
    return text + " " * max(0, width - display_width(text))


def ask_user(prompt: str) -> str | None:
    """交互输入统一封装：回车返回空串；Ctrl+C / EOF / 输入流不可用按取消处理（返回 ``None``）。

    提示语写 stderr，stdout 只留结论与详情（Design §8.4）；stdin 被捕获（pytest）或被整体
    关闭时 CPython 抛 ``OSError`` / ``RuntimeError``（lost sys.stdin），与 EOF 同义——按
    "取消"处理而不是让命令以 traceback 退出。
    """
    try:
        print(prompt, end="", file=sys.stderr, flush=True)
        return input().strip()
    except (EOFError, KeyboardInterrupt, OSError, RuntimeError):
        return None


class RunRenderer:
    """运行过程渲染器：实现引擎的观察者接口（Design §6.2 #6 的阶段序列）。"""

    def __init__(self, *, out: TextIO | None = None, err: TextIO | None = None) -> None:
        self._out = out if out is not None else sys.stdout
        self._err = err if err is not None else sys.stderr

    def on_event(self, event: Mapping[str, Any]) -> None:
        """按事件类型输出：开始头行与阶段行 / 重试 / 工具 / 错误 / 中断。"""
        node = str(event.get("node") or "")
        kind = str(event.get("event") or "")
        detail = event.get("detail")
        text = "" if detail is None else str(detail)
        if node == "engine" and kind == "start":
            run_id = str(event.get("run_id") or "?")
            _write(self._err, f"── 运行 {run_id} " + "─" * _RULE_WIDTH)
            _write(self._err, f"[输入] {text}")
        elif node == "engine":
            return  # 收尾结论由命令面经 render_result 渲染
        elif kind == "start":
            return  # 阶段标题由对应的结束事件给出
        elif kind == "end":
            _write(self._err, f"[{engine_mod.node_label(node)}] {text}")
        elif kind == "retry":
            _write(self._err, f"[{engine_mod.node_label(node)}] 重试：{text}")
        elif kind == "tool":
            _write(self._err, f"[{engine_mod.node_label(node)}] 工具：{text}")
        elif kind == "error":
            _write(self._err, f"[{engine_mod.node_label(node)}] 错误：{text}")
        elif kind == "interrupt":
            _write(self._err, f"[{engine_mod.node_label(node)}] {text}")

    def on_review(
        self,
        *,
        round_number: int,
        file_name: str,
        findings: Sequence[Mapping[str, Any]],
        summary: str,
    ) -> None:
        """审查清单：逐条分级问题（【严重】/【一般】/【建议】）走 stdout。"""
        if not findings:
            _write(self._out, "  （本轮无问题）")
            return
        for finding in findings:
            location = str(finding.get("location") or "")
            issue = str(finding.get("issue") or "")
            _write(self._out, f"  【{finding.get('severity')}】{location} — {issue}")
            suggestion = str(finding.get("suggestion") or "").strip()
            if suggestion:
                _write(self._out, f"    建议：{suggestion}")
        if summary.strip():
            _write(self._out, f"  评述：{_brief(summary)}")

    def on_fixes(
        self,
        *,
        round_number: int,
        fixes: Sequence[Mapping[str, Any]],
        findings: Sequence[Mapping[str, Any]],
    ) -> None:
        """修复处置：逐条"问题 ← 处置"对照走 stdout。"""
        for fix in fixes:
            finding_id = str(fix.get("finding_id") or "")
            note = str(fix.get("note") or "").strip()
            line = f"  【修复】{finding_id} → {fix.get('status')}"
            if note:
                line = f"{line}：{note}"
            _write(self._out, line)


def render_result(
    result: engine_mod.RunResult,
    *,
    out: TextIO | None = None,
    err: TextIO | None = None,
    command: str = "run",
) -> None:
    """收尾结论：干净 / 残留 → stdout；失败与中断 → stderr 可读报错 + 已保留产物路径。"""
    stream_out = out if out is not None else sys.stdout
    stream_err = err if err is not None else sys.stderr
    reached_limit = result.status == state_mod.STATUS_RESIDUAL
    if result.succeeded or reached_limit:
        # 结论句与报告同源（engine.conclusion_text）：判定口径变化时两处一起变
        conclusion = engine_mod.conclusion_text(
            status=result.status,
            review_round=result.review_round,
            fix_rounds=result.fix_rounds,
            findings=result.findings,
            strict=bool(result.state.get("strict")),
        )
        _write(stream_out, f"── 完成：{conclusion}──")
    else:
        # 中断结论有两个来源：本次进程的中断标记，或记录里的 status=interrupted（会话轮次口径）
        interrupted = result.interrupted or result.status == state_mod.STATUS_INTERRUPTED
        label = "已中断" if interrupted else "运行失败"
        node = result.error_node or "engine"
        node_text = (
            f"[{engine_mod.node_label(node)}] " if node in ("encode", "review", "fix") else ""
        )
        _write(
            stream_err,
            f"crw {command}：{label} — {node_text}{result.error_message or '（无详细信息）'}",
        )
    if result.run_dir is not None:
        # 有结论可给人看（干净 / 残留）时运行记录路径随结论走 stdout，否则随报错走 stderr
        target = stream_out if (result.succeeded or reached_limit) else stream_err
        tail = "（报告 report.md）" if result.report_written else "（已中断，未生成报告）"
        _write(target, f"运行记录：{result.run_dir}{tail}")
        if not result.report_written and result.interrupted:
            _write(target, "说明：报告未生成；状态与事件已落盘（state.json / log.jsonl）。")


def render_runs(summaries: Sequence[records_mod.RunSummary], *, out: TextIO | None = None) -> None:
    """运行列表：运行号 | 时间 | 输入摘要 | 状态 | 修复轮数 | 残留严重数（Design §8.1）。"""
    stream = out if out is not None else sys.stdout
    if not summaries:
        _write(stream, "（暂无运行记录；先执行 crw run）")
        return
    header = ("运行号", "时间", "输入摘要", "状态", "修复轮数", "残留严重数")
    rows = [
        (
            item.run_id,
            item.created_at,
            item.input_summary,
            engine_mod.status_label(item.status) + ("（可能已中断）" if item.stale else ""),
            _number(item.fix_rounds),
            _number(item.residual_severe),
        )
        for item in summaries
    ]
    widths = [
        max(display_width(row[column]) for row in (header, *rows)) for column in range(len(header))
    ]
    _write(
        stream, "  ".join(pad(text, widths[index]) for index, text in enumerate(header)).rstrip()
    )
    _write(stream, "  ".join("-" * width for width in widths).rstrip())
    for row in rows:
        _write(
            stream, "  ".join(pad(text, widths[index]) for index, text in enumerate(row)).rstrip()
        )


def render_detail(detail: records_mod.RunDetail, *, out: TextIO | None = None) -> None:
    """运行详情：摘要 → 时间线 → 逐轮 findings 表 → 产物路径（Design §8.1 的 crw show）。"""
    stream = out if out is not None else sys.stdout
    state = detail.state
    _write(stream, f"── 运行 {state.get('run_id') or detail.summary.run_id} " + "─" * _RULE_WIDTH)
    _write(stream, f"[输入] {_input_line(state)}")
    _write(stream, f"[模型] {state.get('model') or '（未记录）'}")
    stale = "（可能已中断）" if detail.summary.stale else ""
    _write(stream, f"[状态] {engine_mod.status_label(str(state.get('status') or ''))}{stale}")
    _write(
        stream,
        f"[轮次] 审查 {state.get('review_round', 0)} 轮 / 修复 {state.get('fix_rounds', 0)} 轮；"
        f"上限 {state.get('max_iterations', 0)}（来源：{engine_mod.source_label(state.get('max_iterations_source'))}）",
    )
    _write(stream, f"[时间] {state.get('created_at')} → {state.get('finished_at')}")
    if state.get("parent_run_id"):
        _write(stream, f"[父运行] {state.get('parent_run_id')}")
    error = _mapping(state.get("error"))
    if error:
        label = "已中断" if error.get("node") == "interrupt" else "运行失败"
        _write(
            stream, f"[{label}] {engine_mod.node_label(error.get('node'))}：{error.get('message')}"
        )

    history = state.get("history") if isinstance(state.get("history"), list) else []
    _write(stream, "")
    _write(stream, "时间线：")
    if history:
        header = ("#", "节点", "轮次", "开始时间", "耗时(ms)", "摘要")
        rows = [
            (
                str(index),
                engine_mod.node_label(entry.get("node")),
                str(entry.get("round") or ""),
                str(entry.get("started_at") or ""),
                str(entry.get("duration_ms") or ""),
                _brief(str(entry.get("summary") or ""), limit=60),
            )
            for index, entry in enumerate(history, start=1)
            if isinstance(entry, Mapping)
        ]
        widths = [
            max(display_width(row[column]) for row in (header, *rows))
            for column in range(len(header))
        ]
        for row in (header, *rows):
            _write(
                stream,
                "  ".join(pad(text, widths[index]) for index, text in enumerate(row)).rstrip(),
            )
    else:
        _write(stream, "（无节点事件）")

    _write(stream, "")
    _write(stream, "逐轮审查：")
    if detail.rounds:
        for item in detail.rounds:
            findings = item.review.get("findings")
            findings = findings if isinstance(findings, list) else []
            counts = _counts(findings)
            _write(
                stream,
                f"  第 {item.round_number} 轮（{item.file_name}）："
                f"{counts['严重']} 严重 / {counts['一般']} 一般 / {counts['建议']} 建议",
            )
            for finding in findings:
                if not isinstance(finding, Mapping):
                    continue
                fix_line = _fix_line(item, str(finding.get("id")))
                _write(
                    stream,
                    f"    【{finding.get('severity')}】{finding.get('location')} — "
                    f"{_brief(str(finding.get('issue') or ''), limit=60)}{fix_line}",
                )
    else:
        _write(stream, "（无审查结论）")

    _write(stream, "")
    _write(stream, "产物路径：")
    for path in detail.artifacts:
        _write(stream, f"  {path}")
    _write(stream, f"  （运行目录：{detail.summary.path}）")


def _fix_line(item: records_mod.RoundReport, finding_id: str) -> str:
    """逐条 findings 的处置后缀（数据来源 fixes_N.json）。"""
    if not isinstance(item.fixes, Mapping):
        return ""
    fixes = item.fixes.get("fixes")
    if not isinstance(fixes, list):
        return ""
    for fix in fixes:
        if isinstance(fix, Mapping) and str(fix.get("finding_id")) == finding_id:
            return f"　← 处置：{fix.get('status')}"
    return ""


def _counts(findings: Sequence[Any]) -> dict[str, int]:
    """逐级计数（详情页的逐轮摘要行）：复用契约层的计数口径。"""
    return contracts.severity_counts([item for item in findings if isinstance(item, Mapping)])


def _number(value: int | None) -> str:
    return "?" if value is None else str(value)


def _brief(text: str, *, limit: int = 80) -> str:
    """单行摘要：折叠空白并限长（终端行宽友好）。"""
    flattened = " ".join(text.split())
    return flattened if len(flattened) <= limit else f"{flattened[: limit - 1]}…"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _input_line(state: Mapping[str, Any]) -> str:
    """详情页的输入行：``-f`` 输入标注完整来源路径与工作区映射（Design §4.5）。"""
    descriptor = _mapping(state.get("input"))
    if descriptor.get("kind") == state_mod.INPUT_FILE:
        return (
            f"文件：{descriptor.get('source_path') or '（未记录路径）'}"
            f" → {state_mod.code_file_name(1)}"
        )
    return state_mod.summarize_input(descriptor, limit=200)


def _write(stream: TextIO, text: str) -> None:
    print(text, file=stream)
    stream.flush()
