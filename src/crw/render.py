"""终端渲染管线：阶段面板、行级流式、分级着色与围栏抑制。

渲染输入由工作流事件驱动（节点开始 / 工具边界 / 重试 / 清单分区 / 节点结束），不依赖框架
的流式元数据；仅命令面使用本模块（Design §3.1、§8.4）。

三条呈现约定（Design §6.2 #6 的完整管线）：

- **阶段面板**：每个节点开始时打一条彩色横幅（``── [审查] 第 1 轮 ──``），Agent 切换一眼
  可辨；阶段结束给一行摘要，清单分区随后渲染，各块之间留空行。
- **围栏抑制**：模型正文里的 ```代码块 / ```json 结论不进终端（生产代码整份回显会刷屏，
  且围栏内容是交给系统的结构化载荷，不是给人读的正文），只留一行形状摘要；验收结论以
  分级清单与处置对照的形式呈现（本模块 ``on_review`` / ``on_fixes``）。
- **分级着色**：【严重】红 /【一般】黄 /【建议】暗，处置 fixed 绿 / partial 黄 / skipped 暗；
  不支持颜色时（非 TTY、``NO_COLOR``、``color=False``）自动退化为纯文本。

输出通道（§8.4）：结论 / findings 清单 / 报告走 stdout；模型正文与过程叙述、错误走 stderr。
``--no-stream`` 时模型正文按阶段缓存、于阶段结束时整体输出（Design §8.1）。
"""

from __future__ import annotations

import datetime
import re
import sys
import threading
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from rich.console import Console

from . import usage as usage_mod
from .workflow import contracts
from .workflow import engine as engine_mod
from .workflow import records as records_mod
from .workflow import state as state_mod

#: 规则线宽度（结论横幅与阶段标题共用）
RULE_WIDTH = 52
#: 兼容别名（既有实现按私有名引用）
_RULE_WIDTH = RULE_WIDTH

#: 密钥形态（与命令面同一构造：左侧非字母数字断言，避免误伤普通连字符词）
_SECRET_PATTERN = re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9]{8,}")

#: 换行常量（正文拼接与裁剪用；写成常量以免源码里的转义在编辑中被误读）
NEWLINE = "\n"

#: ``crw review`` 内联审查的单文件行数上限（超出部分不进提示词，并在提示词中明示）
REVIEW_PROMPT_MAX_LINES = 2000

#: 阶段横幅样式（按 Agent 着色，切换 Agent 一眼可辨）
_NODE_STYLE: dict[str, str] = {
    "encode": "bold cyan",
    "review": "bold blue",
    "review_correctness": "bold blue",
    "review_security": "bold blue",
    "merge_reviews": "bold white",
    "fix": "bold magenta",
    "orchestrator": "bold green",
}
#: 分级着色（Design §6.2 #6）
_SEVERITY_STYLE: dict[str, str] = {
    contracts.SEVERE: "bold red",
    contracts.MODERATE: "yellow",
    contracts.ADVICE: "dim",
}
#: 处置着色与中文标签（fixes 清单）
_STATUS_STYLE: dict[str, str] = {"fixed": "green", "partial": "yellow", "skipped": "dim"}
_STATUS_LABEL: dict[str, str] = {"fixed": "已修复", "partial": "部分修复", "skipped": "未采纳"}

_STAGE_HINTS: dict[str, str] = {"encode": "生成初始代码"}

#: 会话层节点（不在工作流图内）的显示名：编排节点（Design §5.4）
_EXTRA_NODE_LABELS: dict[str, str] = {"orchestrator": "编排"}

#: 并行双视角的分支节点：正文一律按节点缓存后整块输出（见 :meth:`RunRenderer.on_delta`）
_PARALLEL_BRANCH_NODES = frozenset({"review_correctness", "review_security"})

#: 交互提示（"你>" 等）的样式：天蓝色，便于在流式输出里一眼找到输入位置；
#: 非终端 / NO_COLOR 时 rich 自动退化纯文本（管道、重定向与测试里不会出现转义码）
PROMPT_STYLE = "sky_blue1"


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


def ask_user(prompt: str, *, color: bool | None = None) -> str | None:
    """交互输入统一封装：回车返回空串；Ctrl+C / EOF / 输入流不可用按取消处理（返回 ``None``）。

    提示语写 stderr（Design §8.4），并按 :data:`PROMPT_STYLE`（天蓝）着色——终端里一眼能
    找到输入位置；``color=None`` 时按输出流是否终端自动判定（管道 / ``NO_COLOR`` 下为纯文本，
    重定向保存的日志里不会混入转义码）。stdin 被捕获（pytest）或被整体关闭时 CPython 抛
    ``OSError`` / ``RuntimeError``（lost sys.stdin），与 EOF 同义——按"取消"处理而不是让
    命令以 traceback 退出。
    """
    try:
        console_for(sys.stderr, color).print(prompt, end="", style=PROMPT_STYLE)
        sys.stderr.flush()
        return input().strip()
    except (EOFError, KeyboardInterrupt, OSError, RuntimeError):
        return None


def guarded(callback: Callable[[], None]) -> None:
    """执行展示动作并隔离输出流故障（下游关闭管道 / 终端断开，Design §4.6 的同口径）。

    渲染是呈现层：管道被下游关掉（``crw show | head``）时无处分说，也**不得改变命令的结论
    与退出码**（运行记录仍是真相来源）。渲染器自身的编码缺陷（非 ``OSError``）照常抛出，
    不在此处掩盖。
    """
    try:
        callback()
    except OSError:
        pass


def console_for(stream: TextIO, color: bool | None = None) -> Console:
    """按输出流构造着色控制台（``color`` = ``None`` 自动判定，``True``/``False`` 强制）。

    ``rich`` 自带 TTY 探测、``NO_COLOR`` 支持与 Windows 控制台的 ANSI 启用，故不自行处理
    转义序列；``soft_wrap`` 关掉自动换行——模型正文自带换行，终端宽度重排会打乱缩进。
    """
    if color is None:
        return Console(file=stream, markup=False, highlight=False, soft_wrap=True, emoji=False)
    return Console(
        file=stream,
        markup=False,
        highlight=False,
        soft_wrap=True,
        emoji=False,
        force_terminal=color,
        no_color=not color,
    )


# ---- 围栏抑制（流式正文的代码块 / JSON 结论不进终端）----

#: 围栏标记行（CommonMark 语义，允许 0–3 空格缩进）
_MARKER_RE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>[^\s`~]*)[ \t]*$")
#: 行首即围栏 + 已知语言标签：模型把整块代码挤在一行时也要能识别（宁多抑制、不刷屏）
_INLINE_MARKER_RE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>python|py|json)\b")


@dataclass(frozen=True)
class _Emission:
    """过滤器产出：``text`` = 可直接输出的正文片段，``notice`` = 围栏摘要行。"""

    kind: str
    text: str


class _ProseFilter:
    """流式正文过滤器：抑制围栏内容，围栏闭合时给出形状摘要。

    增量回调可能把围栏标记切在任意位置，因此按行装配：行首仍可能长成围栏标记的片段先
    暂扣（只有 0–3 空格 + 反引号/波浪号前缀会被扣住），一旦确定为普通正文立即放行——
    正文的流式感不受影响。
    """

    def __init__(self) -> None:
        self._hold = ""
        self._fence: tuple[str, int, str] | None = None
        self._fence_body: list[str] = []
        self._fence_partial = ""

    def feed(self, text: str) -> list[_Emission]:
        """喂入一段正文增量，返回本次可输出的内容（顺序与原文一致）。"""
        if self._fence is not None:
            return self._feed_fenced(text)
        return self._feed_prose(text)

    def flush(self) -> list[_Emission]:
        """阶段收尾：围栏内容给形状摘要，暂扣片段放行（内容是否完整由契约层判定）。

        末行没有换行是常态（模型正文的最后一片常不带换行）：暂存片段若本身是闭合标记
        （纯反引号、长度不短于开启行），就并入计数而不是算作围栏内容。
        """
        out: list[_Emission] = []
        if self._fence is not None:
            if self._is_closer(self._fence_partial):
                self._fence_partial = ""
            out.append(self._notice())
            self._fence = None
            self._fence_body = []
        # 收尾即整体归零：同一个过滤器被复用（阶段重入 / 重试）时不得残留上一轮的行内片段
        self._fence_partial = ""
        out.extend(self.drain_hold())
        return out

    def drain_hold(self) -> list[_Emission]:
        """放行暂扣的片段（工具行出现 / 阶段收尾时不必再等换行）。"""
        if not self._hold:
            return []
        text, self._hold = self._hold, ""
        return [_Emission("text", text)]

    def _is_closer(self, line: str) -> bool:
        """该行（可无换行）是否为当前围栏的闭合标记。"""
        if self._fence is None:
            return False
        char, length, _ = self._fence
        stripped = line.strip()
        return stripped.startswith(char * length) and set(stripped) <= {char}

    # ---- 内部 ----

    def _feed_prose(self, text: str) -> list[_Emission]:
        out: list[_Emission] = []
        pending = self._hold + text
        self._hold = ""
        while pending:
            newline = pending.find("\n")
            if newline == -1:
                if _marker_candidate(pending):
                    self._hold = pending
                else:
                    out.append(_Emission("text", pending))
                break
            line = pending[: newline + 1]
            pending = pending[newline + 1 :]
            marker = _marker_of(line)
            if marker is None:
                out.append(_Emission("text", line))
                continue
            fence, info = marker
            self._fence = (fence[0], len(fence), info)
            self._fence_body = []
            self._fence_partial = ""
            # 围栏开启后，**同一增量里的剩余内容**已经在围栏内（模型把整块挤在一片里时
            # 最容易出现），必须交给围栏分支处理，否则正文会原样漏出去
            out.extend(self._feed_fenced(pending))
            return out
        return out

    def _feed_fenced(self, text: str) -> list[_Emission]:
        assert self._fence is not None
        out: list[_Emission] = []
        # 未完的行单独暂存：它可能是被切开的闭合标记（`` ` `` / ` `` ` / ` ``` `` 分片），
        # 直接并入围栏体会让"```\n" 永远凑不齐、围栏再也不会闭合
        pending = self._fence_partial + text
        self._fence_partial = ""
        while pending:
            newline = pending.find("\n")
            if newline == -1:
                self._fence_partial = pending
                return out
            line = pending[: newline + 1]
            pending = pending[newline + 1 :]
            if self._is_closer(line):
                out.append(self._notice())
                self._fence = None
                self._fence_body = []
                self._fence_partial = ""
                out.extend(self._feed_prose(pending))
                return out
            self._fence_body.append(line)
        return out

    def _notice(self) -> _Emission:
        """围栏形状摘要。

        不标注"未闭合"：流式增量里无法权威区分"模型真的漏了闭合标记"与"正文里提了一句
        反引号"；这次输出是否完整由契约层判定（缺围栏即重试或可读失败）——终端这条摘要
        只负责"内容不进屏幕"，不冒充结论（真模型演示中该后缀确实出现过误报）。
        """
        _, _, info = self._fence or ("`", 3, "")
        # 围栏体逐段累积、每段自带换行：用 "" 连接，用 "\n" 连接会把行数按双倍空行算
        body = "".join(self._fence_body) + self._fence_partial
        lines = len(body.strip("\n").splitlines())
        if info in contracts.CODE_LABELS:
            head = "已收到代码块"
        elif info in contracts.JSON_LABELS:
            head = "已收到 JSON 结论"
        elif info:
            head = f"已收到 {info} 围栏"
        else:
            head = "已收到围栏块"
        return _Emission("notice", f"{head}（{lines} 行，内容不在此显示）")


def _could_be_marker(partial: str) -> bool:
    """暂扣判定：片段是否仍可能长成围栏标记行（0–3 空格 + 纯反引号/波浪号前缀）。"""
    stripped = partial.lstrip(" ")
    if len(partial) - len(stripped) > 3:
        return False
    return stripped == "" or set(stripped) <= {"`", "~"}


def _marker_candidate(partial: str) -> bool:
    """行首片段是否需要继续等待：可能是标记开头，或本身已经是标记（只差换行）。"""
    return _could_be_marker(partial) or _marker_of(partial) is not None


def _marker_of(line: str) -> tuple[str, str] | None:
    """整行是否为围栏标记；返回 ``(围栏串, 语言标签)``（普通正文返回 ``None``）。

    行尾同时剥 ``\r``：模型正文若按 CRLF 落行，标记正则的 ``[ \\t]*$`` 匹配不上 ``\r``，
    非 python/json 标签的围栏会被当成正文整块漏出去。
    """
    body = line.rstrip("\r\n")
    match = _MARKER_RE.match(body)
    if match is not None:
        return match.group("fence"), match.group("info").lower()
    inline = _INLINE_MARKER_RE.match(body)
    if inline is not None:
        return inline.group("fence"), inline.group("info").lower()
    return None


class RunRenderer:
    """运行过程渲染器：实现引擎的观察者接口（Design §6.2 #6 的阶段序列）。

    Args:
        stream: 是否流式输出模型正文。``False``（``crw run --no-stream``）时按节点缓存
            正文，于该阶段结束时整体输出。
        out / err: 输出流；缺省 stdout / stderr。
        color: 着色开关；``None``（缺省）按输出流是否终端自动判定，``True`` / ``False``
            强制开 / 关（测试与重定向场景）。
    """

    def __init__(
        self,
        *,
        stream: bool = True,
        out: TextIO | None = None,
        err: TextIO | None = None,
        color: bool | None = None,
    ) -> None:
        self._out = console_for(out if out is not None else sys.stdout, color)
        self._err = console_for(err if err is not None else sys.stderr, color)
        self.stream = stream
        self._filters: dict[str, _ProseFilter] = {}
        self._buffers: dict[str, list[str]] = {}
        self._open_stages: list[str] = []
        self._lock = threading.Lock()
        self._suppress: set[str] = set()
        self._line_open = False

    # ---- 引擎观察者接口 ----

    def on_event(self, event: Mapping[str, Any]) -> None:
        """按事件类型输出：阶段面板 / 工具 / 重试 / 错误 / 中断。"""
        node = str(event.get("node") or "")
        kind = str(event.get("event") or "")
        detail = event.get("detail")
        text = "" if detail is None else str(detail)
        if node == "engine":
            if kind == "start":
                run_id = str(event.get("run_id") or "?")
                self._line(f"── 运行 {run_id} " + "─" * _RULE_WIDTH, style="bold")
                self._line(f"[输入] {text}", style="dim")
                model = str(event.get("model") or "")
                if model:
                    self._line(f"[模型] {model}", style="dim")
                return  # 收尾结论由命令面经 render_result 渲染
            if kind not in ("error", "interrupt"):
                return  # 引擎级 end 由 render_result 渲染；其余未知事件不在此臆造输出
        if kind == "start":
            with self._lock:
                self._open_stages.append(node)
            self._line("", style="")
            self._line(_stage_banner(node, event.get("iteration")), style=_style_of(node))
        elif kind == "end":
            self._end_stage(node)
            self._line(f"[{node_label(node)}] {text}", style=_style_of(node))
        elif kind == "retry":
            # Design §4.6：重试的第二次模型输出不重复渲染正文，只给一行提示
            self.discard_text(node)
            with self._lock:
                self._suppress.add(node)
            self._line(f"  ⚠ 重试：{text}", style="yellow")
        elif kind == "tool":
            self._flush_pending_line(node)
            self._line(f"  · 工具：{text}", style="dim")
        elif kind == "error":
            self._end_stage(node)
            self._line(f"[{engine_mod.node_label(node)}] 错误：{text}", style="bold red")
        elif kind == "interrupt":
            self._end_stage(node)
            self._line(f"[{node_label(node)}] {text}", style="bold yellow")

    def on_delta(self, node: str, text: str) -> None:
        """模型正文增量：单阶段时流式输出，多阶段（并行分支）时按节点缓存到该阶段结束。

        围栏内容（代码 / JSON 结论）不进终端（见 :class:`_ProseFilter`）。并行分支会**同时**
        产生增量，若各自即时写出，两位审查者的叙述会在终端交错成一段读不成句的文本
        （实测并发执行确实同时到达）；此时按分支缓存，各分支在自己的阶段结束时整块输出，
        顺序与阶段结束顺序一致。
        """
        with self._lock:
            if node in self._suppress:
                return
            # 并行分支无条件按节点缓存：两个分支的 start 事件与正文增量来自不同线程，
            # "已开启阶段数 ≤ 1 就流式"会随到达顺序偶发地把一个分支的叙述直接写出去
            streaming = (
                self.stream and len(self._open_stages) <= 1 and node not in _PARALLEL_BRANCH_NODES
            )
            if streaming:
                filtr = self._filters.setdefault(node, _ProseFilter())
        if streaming:
            self._emit_prose(filtr.feed(text))
            return
        with self._lock:
            self._buffers.setdefault(node, []).append(text)

    def flush_text(self) -> None:
        """把所有缓存的正文整体输出（阶段结束 / 出错前的收尾）。

        遍历"缓冲"与"过滤器"两类暂存：流式写出的节点没有缓冲，但其过滤器里可能仍
        暂扣着行首片段或待闭合的围栏——只遍历缓冲会把它们丢掉。
        """
        with self._lock:
            nodes = list(self._buffers.keys() | self._filters.keys())
        for node in nodes:
            self._end_stage(node, keep_open=True)

    def discard_text(self, node: str | None = None) -> None:
        """丢弃缓存的正文与过滤器状态（契约重试时前次输出不再渲染）。"""
        with self._lock:
            targets = [node] if node is not None else list(self._buffers)
            for name in targets:
                self._buffers.pop(name, None)
                self._filters.pop(name, None)

    def on_review(
        self,
        *,
        round_number: int,
        version_label: str,
        findings: Sequence[Mapping[str, Any]],
        summary: str,
        view: str | None = None,
    ) -> None:
        """审查清单：逐条分级问题（【严重】/【一般】/【建议】）走 stdout。

        并行双视角时每个视角先给一行计数（``view`` 非空），合并结果仍按完整清单渲染——
        用户先看到两个视角各自的规模，再看到去重合并后的判定依据。
        """
        counts = contracts.severity_counts(findings)
        if view:
            # 分支结论属"结论"通道（stdout，§8.4）：合并结果随后按完整清单展开
            self._out.print(
                f"  · {view}视角：{counts[contracts.SEVERE]} 严重 / "
                f"{counts[contracts.MODERATE]} 一般 / {counts[contracts.ADVICE]} 建议",
                style="dim",
            )
            return
        if not findings:
            self._out.print("  （本轮无问题）", style="green")
            return
        for index, finding in enumerate(findings, start=1):
            location = str(finding.get("location") or "")
            issue = str(finding.get("issue") or "")
            severity = str(finding.get("severity") or "未分级")
            self._out.print(
                f"  {index}. 【{severity}】{location} — {issue}",
                style=_SEVERITY_STYLE.get(severity, "dim"),
            )
            suggestion = str(finding.get("suggestion") or "").strip()
            if suggestion:
                self._out.print(f"     建议：{suggestion}", style="dim")
        if summary.strip():
            self._out.print(f"  评述：{_brief(summary)}", style="dim")

    def on_fixes(
        self,
        *,
        round_number: int,
        fixes: Sequence[Mapping[str, Any]],
        findings: Sequence[Mapping[str, Any]],
    ) -> None:
        """修复处置：逐条"问题 ← 处置"对照走 stdout（着色按 status）。"""
        for fix in fixes:
            finding_id = str(fix.get("finding_id") or "")
            status = str(fix.get("status") or "")
            note = str(fix.get("note") or "").strip()
            label = _STATUS_LABEL.get(status, status)
            line = f"  {finding_id} → {label}"
            if note:
                line = f"{line}：{note}"
            self._out.print(line, style=_STATUS_STYLE.get(status, "dim"))

    # ---- 内部输出 ----

    def _emit_prose(self, emissions: Sequence[_Emission]) -> None:
        for item in emissions:
            if item.kind == "notice":
                # 系统摘要行加前缀：与模型正文一眼分开（正文是原样流式写出的）
                self._write_line(f"  ⋯ {item.text}", style="dim")
            else:
                self._write_raw(item.text)

    def _end_stage(self, node: str, *, keep_open: bool = False) -> None:
        """阶段收尾：把该节点缓存的正文整块输出（围栏内容转成形状摘要）。

        ``keep_open`` 供 :meth:`flush_text` 复用——那是"把缓存吐出去"而不是"阶段结束"，
        阶段是否仍在进行由调用方（end / error / interrupt 事件）决定。
        """
        with self._lock:
            if not keep_open and node in self._open_stages:
                self._open_stages.remove(node)
            buffered = self._buffers.pop(node, None)
            filtr = self._filters.pop(node, None)
            self._suppress.discard(node)
        if buffered is None and filtr is None:
            return
        text = "".join(buffered or []).strip(NEWLINE)
        filtr = filtr or _ProseFilter()
        emissions: list[_Emission] = []
        if text:
            emissions.extend(filtr.feed(text + NEWLINE))
        emissions.extend(filtr.flush())
        self._emit_prose(emissions)

    def _flush_pending_line(self, node: str) -> None:
        """把暂扣的行首片段放行（工具行出现时不必再等换行）。"""
        with self._lock:
            filtr = self._filters.get(node)
        if filtr is not None:
            self._emit_prose(filtr.drain_hold())

    def _line(self, text: str, *, style: str) -> None:
        """输出一整行：上一次若是未以换行收尾的流式正文，先补一个换行。"""
        self._write_line(text, style=style)

    def _write_line(self, text: str, *, style: str) -> None:
        if self._line_open:
            self._err.file.write("\n")
            self._line_open = False
        self._err.print(text, style=style or None)

    def _write_raw(self, text: str) -> None:
        """原样写出模型正文片段（保留缩进与换行，不重排）。"""
        self._err.print(text, end="")
        self._err.file.flush()
        self._line_open = not text.endswith("\n")


def node_label(node: Any) -> str:
    """节点显示名：工作流节点（编码 / 审查 / 修复…）与会话层节点（编排）统一口径。"""
    return _EXTRA_NODE_LABELS.get(str(node), engine_mod.node_label(node))


def mask_secrets(text: str) -> str:
    """遮蔽 ``sk-`` 形态的密钥：会话层与命令面的报错通道统一口径（Design §8.2 的截断惯例）。"""
    return _SECRET_PATTERN.sub("sk-***", text)


def _style_of(node: str) -> str:
    return _NODE_STYLE.get(node, "bold")


def _stage_banner(node: str, iteration: Any) -> str:
    """阶段横幅文本：``── [审查] 第 1 轮 ──`` / ``── [编码] 生成初始代码 ──``。"""
    label = node_label(node)
    round_number = iteration if isinstance(iteration, int) and iteration > 0 else None
    if node == "encode":
        title = f"[{label}] {_STAGE_HINTS['encode']}"
    elif node == "merge_reviews":
        title = f"[{label}] 第 {round_number} 轮：合并双视角结论" if round_number else f"[{label}]"
    elif round_number:
        title = f"[{label}] 第 {round_number} 轮"
    else:
        title = f"[{label}]"
    head = f"── {title} "
    return head + "─" * max(2, _RULE_WIDTH - display_width(head))


def render_result(
    result: engine_mod.RunResult,
    *,
    out: TextIO | None = None,
    err: TextIO | None = None,
    command: str = "run",
    color: bool | None = None,
) -> None:
    """收尾结论：干净 / 残留 → stdout；失败与中断 → stderr 可读报错 + 已保留产物路径。"""
    guarded(lambda: _render_result(result, out=out, err=err, command=command, color=color))


def _render_result(
    result: engine_mod.RunResult,
    *,
    out: TextIO | None = None,
    err: TextIO | None = None,
    command: str = "run",
    color: bool | None = None,
) -> None:
    stdout = out if out is not None else sys.stdout
    stderr = err if err is not None else sys.stderr
    out_console = console_for(stdout, color)
    err_console = console_for(stderr, color)
    reached_limit = result.status == state_mod.STATUS_RESIDUAL
    if result.succeeded or reached_limit:
        out_console.print("")  # 结论前留空行：与阶段输出分区（Design §6.2 #6 的排版约定）
        # 结论句与报告同源（engine.conclusion_text）：判定口径变化时两处一起变
        conclusion = engine_mod.conclusion_text(
            status=result.status,
            review_round=result.review_round,
            fix_rounds=result.fix_rounds,
            findings=result.findings,
            strict=bool(result.state.get("strict")),
        )
        head = f"── 完成：{conclusion} "
        style = "bold green" if result.succeeded else "bold yellow"
        out_console.print(head + "─" * max(2, _RULE_WIDTH - display_width(head)), style=style)
    else:
        # 中断结论有两个来源：本次进程的中断标记，或记录里的 status=interrupted（会话轮次口径）
        interrupted = result.interrupted or result.status == state_mod.STATUS_INTERRUPTED
        label = "已中断" if interrupted else "运行失败"
        node = result.error_node or "engine"
        node_text = (
            f"[{node_label(node)}] "
            if node in {**engine_mod.NODE_LABELS, **_EXTRA_NODE_LABELS}
            else ""
        )
        err_console.print(
            f"crw {command}：{label} — {node_text}{result.error_message or '（无详细信息）'}",
            style="bold red",
        )
    if result.run_dir is not None:
        # 有结论可给人看（干净 / 残留）时运行记录路径随结论走 stdout，否则随报错走 stderr
        target = out_console if (result.succeeded or reached_limit) else err_console
        tail = "（报告 report.md）" if result.report_written else "（已中断，未生成报告）"
        target.print(f"运行记录：{result.run_dir}{tail}", style="dim")
        if not result.report_written and result.interrupted:
            target.print(
                "说明：报告未生成；状态与事件已落盘（state.json / log.jsonl）。", style="dim"
            )


def render_sessions(
    summaries: Sequence[Any],
    *,
    out: TextIO | None = None,
    color: bool | None = None,
) -> None:
    """会话列表：会话号 | 时间 | 轮次数 | 最后状态（Design §8.1）。"""
    guarded(lambda: _render_sessions(summaries, out=out, color=color))


def _render_sessions(
    summaries: Sequence[Any],
    *,
    out: TextIO | None = None,
    color: bool | None = None,
) -> None:
    console = console_for(out if out is not None else sys.stdout, color)
    if not summaries:
        console.print("（暂无会话记录；先执行 crw chat 或 crw chat --resume <会话号>）")
        return
    header = ("会话号", "时间", "轮次数", "最后状态")
    rows = [
        (
            item.session_id,
            item.created_at or "（未记录）",
            str(item.round_count),
            item.last_status if item.degraded else engine_mod.status_label(item.last_status),
        )
        for item in summaries
    ]
    widths = [
        max(display_width(row[column]) for row in (header, *rows)) for column in range(len(header))
    ]
    console.print(
        "  ".join(pad(text, widths[index]) for index, text in enumerate(header)).rstrip(),
        style="bold",
    )
    console.print("  ".join("-" * width for width in widths).rstrip())
    for item, row in zip(summaries, rows, strict=True):
        console.print(
            "  ".join(pad(text, widths[index]) for index, text in enumerate(row)).rstrip(),
            style=_status_style(item.last_status) if not item.degraded else "",
        )


def render_runs(
    summaries: Sequence[records_mod.RunSummary],
    *,
    out: TextIO | None = None,
    color: bool | None = None,
) -> None:
    """运行列表：运行号 | 时间 | 输入摘要 | 状态 | 修复轮数 | 残留严重数 | 模型（Design §8.1）。"""
    guarded(lambda: _render_runs(summaries, out=out, color=color))


def _render_runs(
    summaries: Sequence[records_mod.RunSummary],
    *,
    out: TextIO | None = None,
    color: bool | None = None,
) -> None:
    console = console_for(out if out is not None else sys.stdout, color)
    if not summaries:
        console.print("（暂无运行记录；先执行 crw run）")
        return
    header = ("运行号", "时间", "输入摘要", "状态", "修复轮数", "残留严重数", "模型", "会话")
    rows = [
        (
            item.run_id,
            item.created_at,
            item.input_summary,
            engine_mod.status_label(item.status) + ("（可能已中断）" if item.stale else ""),
            _number(item.fix_rounds),
            _number(item.residual_severe),
            item.model or "（未记录）",
            item.session_id or "—",
        )
        for item in summaries
    ]
    widths = [
        max(display_width(row[column]) for row in (header, *rows)) for column in range(len(header))
    ]
    console.print(
        "  ".join(pad(text, widths[index]) for index, text in enumerate(header)).rstrip(),
        style="bold",
    )
    console.print(
        "  ".join("-" * width for width in widths).rstrip(),
    )
    for item, row in zip(summaries, rows, strict=True):
        console.print(
            "  ".join(pad(text, widths[index]) for index, text in enumerate(row)).rstrip(),
            style=_status_style(item.status),
        )


def render_detail(
    detail: records_mod.RunDetail,
    *,
    out: TextIO | None = None,
    color: bool | None = None,
) -> None:
    """运行详情：摘要 → 时间线（含工具调用）→ 逐轮 findings 表 → 产物路径（Design §8.1）。

    ``state.json`` 损坏时降级展示（Design §8.2）：摘要行取自日志与产物清单，逐轮审查表仍
    来自 ``review_*.json``（文件独立于状态快照），并显式说明当前展示的数据来源。
    """
    guarded(lambda: _render_detail(detail, out=out, color=color))


def _render_detail(
    detail: records_mod.RunDetail,
    *,
    out: TextIO | None = None,
    color: bool | None = None,
) -> None:
    console = console_for(out if out is not None else sys.stdout, color)
    state = detail.state
    run_id = state.get("run_id") or detail.summary.run_id
    if detail.degraded:
        head = f"── 运行 {run_id}（记录降级展示）"
        console.print(
            head + " " + "─" * max(2, _RULE_WIDTH - display_width(head)), style="bold yellow"
        )
        console.print(f"[警告] {detail.degraded_reason}", style="yellow")
        console.print("[来源] 以下摘要取自 log.jsonl 与产物清单（state.json 不可读）", style="dim")
    else:
        head = f"── 运行 {run_id}"
        console.print(head + " " + "─" * max(2, _RULE_WIDTH - display_width(head)), style="bold")
    console.print(f"[输入] {_input_line(state, detail.summary)}")
    console.print(f"[模型] {state.get('model') or '（未记录）'}")
    stale = "（可能已中断）" if detail.summary.stale else ""
    status = state.get("status") or detail.summary.status
    console.print(
        f"[状态] {engine_mod.status_label(str(status))}{stale}",
        style=_status_style(str(status)),
    )
    # 降级态（state.json 不可读）：修复轮次可由产物数派生，审查轮次没有可靠来源——
    # 用"修复轮次 + 1"会与同页口径打架，故如实写"（未记录）"
    review_round = state.get("review_round")
    console.print(
        f"[轮次] 审查 {review_round if review_round is not None else '（未记录）'} 轮 / "
        f"修复 {state.get('fix_rounds', detail.summary.fix_rounds or 0)} 轮；"
        f"上限 {state.get('max_iterations', '（未记录）')}"
        f"（来源：{engine_mod.source_label(state.get('max_iterations_source'))}）"
    )
    console.print(f"[时间] {_time_line(state, detail)}")
    usage_record = _usage_totals(detail)
    if not usage_mod.is_empty(usage_record):
        console.print(f"[用量] {usage_mod.format_line(usage_record)}")
        hint = usage_mod.hint(usage_record)
        if hint:
            console.print(f"[提示] {hint}", style="dim")
    if state.get("session_id"):
        console.print(
            f"[会话] {state.get('session_id')} / 第 {state.get('round_index', '（未记录）')} 轮"
        )
    if state.get("parent_run_id"):
        console.print(f"[父运行] {state.get('parent_run_id')}（续跑来源）")
    error = _mapping(state.get("error"))
    if error:
        label = "已中断" if error.get("node") == "interrupt" else "运行失败"
        console.print(
            f"[{label}] {engine_mod.node_label(error.get('node'))}：{error.get('message')}",
            style="red",
        )

    console.print("")
    console.print("时间线：", style="bold")
    rows = _timeline_rows(detail)
    if rows:
        header = ("#", "节点", "轮次", "开始时间", "耗时(ms)", "工具", "摘要")
        widths = [
            max(display_width(row[column]) for row in (header, *rows))
            for column in range(len(header))
        ]
        console.print(
            "  ".join(pad(text, widths[index]) for index, text in enumerate(header)).rstrip(),
            style="bold",
        )
        for row in rows:
            console.print("  ".join(pad(text, widths[i]) for i, text in enumerate(row)).rstrip())
    else:
        console.print("（无节点事件）")

    tools = _tool_lines(detail)
    if tools:
        console.print("")
        console.print("工具调用：", style="bold")
        for line in tools:
            console.print(f"  {line}", style="dim")

    console.print("")
    console.print("逐轮审查：", style="bold")
    if detail.rounds:
        for item in detail.rounds:
            findings = item.review.get("findings")
            findings = findings if isinstance(findings, list) else []
            counts = contracts.severity_counts(findings)
            console.print(
                f"  第 {item.round_number} 轮（{item.version_label}{len(item.files)} 个文件）："
                f"{counts[contracts.SEVERE]} 严重 / {counts[contracts.MODERATE]} 一般 / "
                f"{counts[contracts.ADVICE]} 建议",
                style="bold",
            )
            if _not_converged(detail, item.round_number):
                console.print(
                    "    ⚠ 未收敛：本轮问题集合与上一轮完全相同（已如实标注，未提前中止）",
                    style="yellow",
                )
            for index, finding in enumerate(findings, start=1):
                if not isinstance(finding, Mapping):
                    continue
                severity = str(finding.get("severity") or "未分级")
                console.print(
                    f"    {index}. 【{severity}】{finding.get('location')} — "
                    f"{_brief(str(finding.get('issue') or ''), limit=60)}"
                    f"{_fix_line(item, str(finding.get('id')))}",
                    style=_SEVERITY_STYLE.get(severity, "dim"),
                )
    else:
        console.print("（无审查结论）")

    console.print("")
    console.print("产物路径：", style="bold")
    for path in detail.artifacts:
        console.print(f"  {path}")
    console.print(f"  （运行目录：{detail.summary.path}）", style="dim")


def _timeline_rows(detail: records_mod.RunDetail) -> list[tuple[str, ...]]:
    """时间线表：优先取 state.history；损坏记录改由 log.jsonl 的事件配对派生。"""
    history = detail.state.get("history") if isinstance(detail.state.get("history"), list) else []
    if history:
        rows: list[tuple[str, ...]] = []
        for index, entry in enumerate(history, start=1):
            if not isinstance(entry, Mapping):
                continue
            node = str(entry.get("node") or "")
            rows.append(
                (
                    str(index),
                    engine_mod.node_label(node),
                    str(entry.get("round") or ""),
                    str(entry.get("started_at") or ""),
                    str(entry.get("duration_ms") or ""),
                    str(_tool_count(detail, node, entry.get("round"))),
                    _brief(str(entry.get("summary") or ""), limit=60),
                )
            )
        return rows
    return _timeline_rows_from_log(detail)


def _timeline_rows_from_log(detail: records_mod.RunDetail) -> list[tuple[str, ...]]:
    """降级时间线（state.json 不可读）：用 start / end 事件的时间戳配对算耗时。"""
    rows: list[tuple[str, ...]] = []
    starts: dict[tuple[str, Any], str] = {}
    order: list[tuple[str, Any, str]] = []
    for event in detail.events:
        node = str(event.get("node") or "")
        kind = str(event.get("event") or "")
        if node == "engine":
            continue
        round_number = event.get("iteration")
        key = (node, round_number)
        if kind == "start":
            starts[key] = str(event.get("ts") or "")
            order.append((node, round_number, ""))
        elif kind == "end":
            detail_text = str(event.get("detail") or "")
            for index, (entry_node, entry_round, _) in enumerate(order):
                if (entry_node, entry_round) == key:
                    order[index] = (entry_node, entry_round, detail_text)
                    break
    if not order:
        return rows
    for index, (node, round_number, summary) in enumerate(order, start=1):
        started = starts.get((node, round_number), "")
        ended = next(
            (
                str(event.get("ts") or "")
                for event in detail.events
                if str(event.get("node") or "") == node
                and event.get("iteration") == round_number
                and str(event.get("event") or "") == "end"
            ),
            "",
        )
        rows.append(
            (
                str(index),
                engine_mod.node_label(node),
                "" if round_number is None else str(round_number),
                started,
                _duration_ms(started, ended),
                str(_tool_count(detail, node, round_number)),
                _brief(summary, limit=60),
            )
        )
    return rows


def _tool_lines(detail: records_mod.RunDetail) -> list[str]:
    """工具调用明细（来自 log.jsonl）：``[审查] 第 1 轮 · read_file(…)``。"""
    lines: list[str] = []
    for event in detail.events:
        if str(event.get("event") or "") != "tool":
            continue
        node = str(event.get("node") or "")
        round_number = event.get("iteration")
        where = f"[{node_label(node)}]"
        if round_number is not None:
            where = f"{where} 第 {round_number} 轮"
        lines.append(f"{where} · {_brief(str(event.get('detail') or ''), limit=100)}")
    return lines


def _tool_count(detail: records_mod.RunDetail, node: str, round_number: Any) -> int:
    return sum(
        1
        for event in detail.events
        if str(event.get("event") or "") == "tool"
        and str(event.get("node") or "") == node
        and event.get("iteration") == round_number
    )


def _not_converged(detail: records_mod.RunDetail, round_number: int) -> bool:
    """未收敛判定（Design §4.4）：与上一轮的 ``(location, issue)`` 集合完全相同。"""
    if round_number <= 1:
        return False
    previous = next((item for item in detail.rounds if item.round_number == round_number - 1), None)
    current = next((item for item in detail.rounds if item.round_number == round_number), None)
    if previous is None or current is None:
        return False
    return engine_mod.not_converged(previous.review, current.review)


def version_base_of(state: Mapping[str, Any]) -> int:
    """状态的起始版本号（展示用）：坏值按 1 处理，口径见 ``state.normalize_version_base``。"""
    return state_mod.normalize_version_base(state.get("version_base"))


def _duration_ms(started: str, ended: str) -> str:
    """两个 ISO 时间戳的耗时（毫秒，降级时间线用）；解析失败留空。"""
    try:
        start = datetime.datetime.fromisoformat(started)
        end = datetime.datetime.fromisoformat(ended)
    except (TypeError, ValueError):
        return ""
    return str(max(0, int((end - start).total_seconds() * 1000)))


def _time_line(state: Mapping[str, Any], detail: records_mod.RunDetail) -> str:
    """开始 → 结束时间；state.json 不可读时取日志首末事件时间。"""
    if state.get("created_at") or state.get("finished_at"):
        return f"{state.get('created_at') or '（未记录）'} → {state.get('finished_at') or '（未记录）'}"
    stamps = [str(event.get("ts") or "") for event in detail.events if event.get("ts")]
    if not stamps:
        return "（未记录）"
    return f"{stamps[0]} → {stamps[-1]}（取自日志）"


def _usage_totals(detail: records_mod.RunDetail) -> dict[str, int]:
    """本次运行的用量合计：``state.json`` 的按节点账优先，读不出时回退 log.jsonl 的 usage 事件。

    旧记录（本功能之前留下的）两处都没有 usage 行 → 空账，展示侧据此整行略过。
    """
    record = usage_mod.totals(_mapping(detail.state.get("usage")))
    if usage_mod.is_empty(record):
        record = usage_mod.from_log_events(detail.events)
    return record


def _status_style(status: str) -> str:
    return {
        state_mod.STATUS_SUCCESS: "green",
        state_mod.STATUS_RESIDUAL: "yellow",
        state_mod.STATUS_FAILED: "red",
        state_mod.STATUS_INTERRUPTED: "red",
        state_mod.STATUS_RUNNING: "dim",
    }.get(status, "")


def _fix_line(item: records_mod.RoundReport, finding_id: str) -> str:
    """逐条 findings 的处置后缀（数据来源 fixes_N.json）。"""
    if not isinstance(item.fixes, Mapping):
        return ""
    fixes = item.fixes.get("fixes")
    if not isinstance(fixes, list):
        return ""
    for fix in fixes:
        if isinstance(fix, Mapping) and str(fix.get("finding_id")) == finding_id:
            status = str(fix.get("status") or "")
            return f"　← 处置：{_STATUS_LABEL.get(status, status)}"
    return ""


def _number(value: int | None) -> str:
    return "?" if value is None else str(value)


def _brief(text: str, *, limit: int = 80) -> str:
    """单行摘要：折叠空白并限长（终端行宽友好）。"""
    flattened = " ".join(text.split())
    return flattened if len(flattened) <= limit else f"{flattened[: limit - 1]}…"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _input_line(state: Mapping[str, Any], summary: records_mod.RunSummary) -> str:
    """详情页的输入行：``-f`` / resume 输入标注来源与工作区映射（Design §4.5）。

    resume 的来源是父运行的版本目录（名称形如 ``code_vN``），映射口径为
    "父版本目录 → ``code_v1/``"；``-f`` 为"原文件 → ``code_v1/<原文件名>``"。
    """
    descriptor = _mapping(state.get("input"))
    start_dir = state_mod.code_version_dir_name(version_base_of(state))
    if descriptor.get("kind") == state_mod.INPUT_FILE:
        source = str(descriptor.get("source_path") or "")
        if re.fullmatch(r"code_v\d+", Path(source).name or ""):
            return f"父运行版本目录：{source or '（未记录）'} → {start_dir}/"
        return (
            f"文件：{source or '（未记录路径）'}"
            f" → {start_dir}/{Path(source).name if source else '（未命名）'}"
        )
    if not descriptor:
        return summary.input_summary
    text = state_mod.summarize_input(descriptor, limit=200)
    source = str(descriptor.get("source_path") or "")
    if source and re.fullmatch(r"code_v\d+", Path(source).name or ""):
        # 会话轮次：需求驱动编码起步、既有代码来自上一版本（Design §4.7）
        return f"{text}（基于 {Path(source).name}/ 继续开发 → {start_dir}/）"
    return text


def review_json_report(merged: Mapping[str, Any]) -> str:
    """``crw review --json`` 的 Markdown 报告（``-o`` 与 ``--json`` 可同时使用）。"""
    stamp = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    lines = ["# crw 审查报告", "", f"> 生成时间：{stamp}（--json 模式）", ""]
    for entry in merged.get("files") or []:
        # 展示层不做契约假设：条目缺字段时给占位，不让报告端直接 KeyError
        lines += [
            f"## {entry.get('file') or '（未命名文件）'}",
            "",
            f"**整体评价**：{entry.get('summary') or ''}",
            "",
        ]
        for finding in entry.get("findings") or []:
            uncertain = "（不确定）" if finding.get("uncertain") else ""
            lines += [
                (
                    f"- **【{finding.get('severity') or '未分级'}】** "
                    f"`{finding.get('location') or '（未定位）'}` — "
                    f"{finding.get('issue') or ''}{uncertain}"
                ),
                f"  修改建议：{finding.get('suggestion') or '（无）'}",
            ]
        lines.append("")
    # 单文件时顶层 summary 与 files[0].summary 同文，跳过"总体评价"段避免重复；
    # 无可用结果（files 为空）时该段是唯一承载结论的地方，必须保留
    if len(merged.get("files") or []) != 1:
        lines += ["## 总体评价", "", str(merged.get("summary") or ""), ""]
    return "\n".join(lines)


def review_text_report(files: Sequence[tuple[str, str]], answers: Sequence[str | None]) -> str:
    """非 ``--json`` 模式的 Markdown 报告：结论原文嵌入（本身即 Markdown）。

    ``answers`` 与 ``files`` 逐位对齐，``None`` 表示该文件审查失败（保留标题并标注缺章）。
    按位置对齐而非按显示名查表：同名的两个路径不会互相覆盖结论，标题清单与正文恒一致。
    """
    stamp = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# crw 审查报告",
        "",
        f"> 生成时间：{stamp}",
        f"> 审查对象：{', '.join(name for name, _ in files)}",
        "",
    ]
    for (display_name, _), answer in zip(files, answers, strict=True):
        lines += [f"## {display_name}", "", answer or "[该文件审查失败，结论不可用]", ""]
    return "\n".join(lines)
