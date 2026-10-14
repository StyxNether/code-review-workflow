"""编排 / 路由节点：用户消息 → 意图与轮次参数（结构化）→ 选择下一节点（Design §5.4 / D12）。

会话里只有编排节点面向用户（单口约束）：它把用户消息解释为 ``code | review |
review_fix | ask`` 并给出 ``rounds`` / ``rounds_mode`` / ``reply``，由本模块校验后交给
chat 驱动工作流。**分不清意图一律按 ask 反问**（不静默挑一条路）；模型输出无法解析时
也走同一条反问兜底——猜错方向的代价是多跑一整轮工作流。

系统提示词是**常量**（不注入时间戳 / 会话号 / 轮次上限等易变值，Design §4.7）：会话现状
与本次消息都在用户消息里，前缀稳定才能吃到供应商的上下文缓存。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..agents import AgentRuntime, ToolRegistry
from ..workflow import engine as engine_mod
from ..workflow import state as state_mod
from ..workflow.contracts import scan_fences

#: 意图枚举（Design §5.4）
INTENT_CODE = "code"
INTENT_REVIEW = "review"
INTENT_REVIEW_FIX = "review_fix"
INTENT_ASK = "ask"
INTENTS: tuple[str, ...] = (INTENT_CODE, INTENT_REVIEW, INTENT_REVIEW_FIX, INTENT_ASK)

#: 意图的显示名（汇报与日志共用）
INTENT_LABELS: Mapping[str, str] = {
    INTENT_CODE: "写/改代码",
    INTENT_REVIEW: "只审查",
    INTENT_REVIEW_FIX: "再走一轮审查→修复",
    INTENT_ASK: "提问",
}

#: 收尾建议的显示名（state.next_action → 人话）
NEXT_ACTION_LABELS: Mapping[str, str] = {
    state_mod.NEXT_DONE: "已完成",
    state_mod.NEXT_REVIEW_AGAIN: "可再检查一轮",
    state_mod.NEXT_FIX_AGAIN: "可继续修复（crw resume，或在会话里说「继续修」）",
}

#: 模型输出无法解析时的兜底反问（不静默挑路，Design §5.4 的路由兜底）
FALLBACK_ASK = (
    "我没能确定你的意图。请说明一下：是要（1）写/改代码、（2）只审查当前实现、"
    "还是（3）对当前实现再走一轮「审查→修复」？"
)

#: 编排节点的系统提示词（**常量**：会话现状与本次消息都在用户消息里）
ORCHESTRATOR_SYSTEM_PROMPT = """你是代码审查工作流（crw）会话里的编排助手：\
用户只与你对话，你负责判断用户的意图，把工作交给下游的三个 Agent（编码 / 审查 / 修复）。\
你不写代码、不做审查、不给修复结论。

# 意图判定
- code：要写新代码，或按新需求修改 / 扩展会话里已有的实现（"再加个功能""把它改成支持参数"）。
- review：只检查当前实现、不修改（"看看有没有问题""审查一下"）。
- review_fix：不改需求，对当前实现再走一轮"审查→修复"循环（"再检查一轮""再修一遍看看"）。
  这个意图表示**用户要动手修**：rounds_mode 填 "exact"（未提到轮次时 rounds 填 1）——
  "审查干净就不修"会违背用户的明确指示。
- ask：提问、闲聊、意图不明确、或需要用户补充信息。
- **分不清就用 ask 反问**：宁可多问一句，也不要猜错方向——猜错会白跑一整轮工作流。
- 会话里还没有代码时，review / review_fix 没有对象：这种消息用 ask 说明"当前还没有代码版本"，
  或按 code 处理明显的开发需求。

# 轮次与"要不要修"
- rounds / rounds_mode **只在用户明确表达时才填**；没提到就省略这两个字段（系统有默认值）。
  不要臆造用户没说的轮次。
- 用户明确要求跑满固定轮次（"跑满 3 轮""固定修 2 轮，不要提前停"）→ rounds_mode 填 "exact"。
- **用户明确要求"审查之后要修复 / 审完修一轮 / 不要跳过修复"→ rounds_mode 必须填 "exact"**
  （并给出 rounds，缺省 1）：这是硬要求，"审查干净就不修"会违背用户的明确指示。
- 用户明确表示"没问题就不用改 / 只看不改 / 有问题再修"→ 照用户说的填（"有问题再修"填 "up-to"）。
- rounds 必须是整数；超出上限由系统按上限夹取并在汇报里标注。

# reply
一句话、中文、面向用户：ask 时它就是你要问的问题（或回答）；其余意图里说明你会怎么做\
（如"好的，我按你说的改代码，最多 3 轮"）；review_fix 时说明"先审查、再修一轮"。

# 输出格式（必须严格遵守）
只输出**一个 ```json 围栏**，围栏内是下面这个对象；不要输出其他文字或围栏：

{"intent": "code|review|review_fix|ask", "rounds": 3, "rounds_mode": "up-to|exact", "reply": "…"}"""


@dataclass(frozen=True)
class Decision:
    """编排节点对一条用户消息的结构化解释（已过校验 / 夹取，Design §5.4）。"""

    intent: str
    reply: str
    rounds: int | None = None
    """消息里提取到的轮次（None = 未指定，由调用方按优先级链取值）。"""
    rounds_mode: str | None = None
    """消息里提取到的轮次模式（None = 未指定）。"""
    note: str = ""
    """给用户/日志的附加说明（如"轮次已按上限夹取""输出无法解析，已反问"）。"""
    raw: Mapping[str, Any] = field(default_factory=dict)
    """模型输出的原始对象（未知字段原样保留，供 log 审计；不改写不丢字段）。"""

    @property
    def starts_workflow(self) -> bool:
        """是否要启动工作流（``ask`` 不启动，Design §5.4）。"""
        return self.intent != INTENT_ASK


def interpret(
    runtime: AgentRuntime,
    registry: ToolRegistry,
    *,
    message: str,
    context: Mapping[str, Any],
    rounds_limit: int,
    rounds_default: int,
) -> Decision:
    """把一条用户消息解释为结构化决策；解析失败重试一次，仍失败按 ask 反问。

    Args:
        runtime: 编排节点的运行时（与三 Agent 同一装配：同一模型、同一事件与增量通道）。
        registry: 编排节点的工具注册表（空工具集——编排只做判断，不需要读写文件）。
        message: 用户消息原文。
        context: 会话现状（轮次数、当前版本与文件、上一轮结论），渲染进用户消息。
        rounds_limit: 轮次夹取上限（``~/.crw/settings.json`` 的 ``rounds.limit``）。
        rounds_default: 轮次默认值（消息未指定时的回退，仅用于提示语）。

    Returns:
        :class:`Decision`；``ask`` 且 ``note`` 非空表示这是解析失败后的反问兜底。
    """
    prompt = _orchestrator_prompt(
        message=message, context=context, rounds_limit=rounds_limit, rounds_default=rounds_default
    )
    feedback: str | None = None
    problem = ""
    for _ in range(2):
        result = runtime.run(
            node="orchestrator",
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
            user_prompt=prompt
            if feedback is None
            else f"{prompt}\n\n上一次输出未被接受，原因如下（请修正后重新输出）：\n{feedback}",
            registry=registry,
        )
        payload, error = _parse_payload(result.text)
        if not error:
            maybe = _validate(payload, rounds_limit=rounds_limit)
            if isinstance(maybe, Decision):
                return maybe
            error = maybe
        feedback = error
        problem = error
    return Decision(
        intent=INTENT_ASK,
        reply=FALLBACK_ASK,
        note=f"编排节点输出无法解析（{problem}），已按 ask 反问。",
    )


def _orchestrator_prompt(
    *, message: str, context: Mapping[str, Any], rounds_limit: int, rounds_default: int
) -> str:
    """用户消息 + 会话现状 + 输出契约（全部在用户侧，系统提示词保持常量）。"""
    lines = ["## 会话现状", *_context_lines(context), ""]
    lines.append(
        f"轮次上限（夹取用）：{rounds_limit}；轮次默认值（未指定时的回退）：{rounds_default}。"
    )
    lines.extend(["", "## 用户消息", str(message).strip() or "（空消息）"])
    lines.append("")
    lines.append("请按系统提示词的格式输出这个 JSON 对象（一个 ```json 围栏）。")
    return "\n".join(lines)


def _context_lines(context: Mapping[str, Any]) -> list[str]:
    """会话现状的自然语言描述（模型据此理解"继续""再检查一轮"指什么）。"""
    rounds = context.get("rounds")
    lines: list[str] = []
    count = len(rounds) if isinstance(rounds, (list, tuple)) else 0
    lines.append(f"- 本会话已进行 {count} 轮。")
    version = context.get("current_version")
    files = context.get("version_files")
    if isinstance(version, int) and version > 0:
        names = ", ".join(str(item) for item in files or []) or "（未记录）"
        lines.append(f"- 当前代码版本：code_v{version}/（{names}）。")
    else:
        lines.append("- 当前还没有任何代码版本。")
    conclusion = context.get("last_conclusion")
    if isinstance(conclusion, str) and conclusion:
        lines.append(f"- 上一轮结论：{conclusion}")
    return lines


def _parse_payload(text: str) -> tuple[Any, str]:
    """从模型正文提取 JSON 对象（优先 ```json 围栏，退化到首尾花括号切片）。

    Returns:
        ``(payload, error)``：解析成功时 ``error`` 为空串（调用方按"空串即成功"判定）。
    """
    candidates = [content for label, content in scan_fences(text) if label == "json"]
    candidates.append(text)
    for candidate in candidates:
        body = candidate.strip()
        if not body:
            continue
        start, end = body.find("{"), body.rfind("}")
        if start == -1 or end <= start:
            continue
        try:
            data = json.loads(body[start : end + 1])
        except json.JSONDecodeError:
            continue
        return data, ""
    return None, "回答里没有可解析的 JSON 对象"


def _validate(payload: Any, *, rounds_limit: int) -> Decision | str:
    """校验编排节点的输出（Design §5.4 的校验规则）；违规时返回可读原因（字符串）。

    - ``intent`` 必须是枚举值；``ask`` 必须给出非空 ``reply``；
    - ``rounds`` 缺省 / 非整数按"未指定"处理，越界**夹取并标注**（意图理解 ≠ 用户显式参数）；
    - ``rounds_mode`` 缺省 / 非枚举按"未指定"处理；
    - 未知字段原样保留在 ``raw`` 里（忽略但留档）。
    """
    if not isinstance(payload, Mapping):
        return "顶层必须是 JSON 对象"
    intent = payload.get("intent")
    if not isinstance(intent, str) or intent not in INTENTS:
        return f"intent 必须是 {' / '.join(INTENTS)} 之一（收到 {intent!r}）"
    reply = payload.get("reply")
    reply_text = reply.strip() if isinstance(reply, str) else ""
    if intent == INTENT_ASK and not reply_text:
        return "ask 必须给出非空的 reply（要问用户什么 / 答什么）"
    rounds, note = _clamp_rounds(payload.get("rounds"), rounds_limit)
    mode = payload.get("rounds_mode")
    if mode not in (state_mod.ROUNDS_UP_TO, state_mod.ROUNDS_EXACT):
        mode = None
    return Decision(
        intent=intent,
        reply=reply_text,
        rounds=rounds,
        rounds_mode=mode,
        note=note,
        raw=dict(payload),
    )


def _clamp_rounds(value: Any, limit: int) -> tuple[int | None, str]:
    """``rounds`` 取值：非整数按未指定；越界夹到 ``[0, limit]`` 并给出标注文本。"""
    if value is None:
        return None, ""
    if isinstance(value, bool):
        return None, ""
    if isinstance(value, int):
        rounds = value
    elif isinstance(value, str) and _is_ascii_decimal(value.strip()):
        rounds = int(value.strip())
    else:
        return None, ""
    if rounds < 0:
        return 0, f"消息里的轮次 {rounds} 已被按 0 处理（下限）。"
    if rounds > limit:
        return limit, f"消息里的轮次 {rounds} 已按上限 {limit} 夹取。"
    return rounds, ""


def _is_ascii_decimal(text: str) -> bool:
    """ASCII 十进制整数判定（与命令面 `--max-iter` 同一口径）。

    不用 ``str.isdigit()``：它对上标等非十进制数字字符也为真，而 ``int()`` 会抛错——
    模型的 JSON 里出现这类字符时应按"未指定"处理，而不是让整轮会话崩掉。
    """
    return bool(text) and text.isascii() and text.isdecimal()


def _as_int(value: object, *, default: int = 0) -> int:
    """汇报取整容错：坏值按默认值处理（汇报是展示层，不该因坏记录整体失败）。

    ``bool`` 视为坏值（与全仓"布尔不算整数"的口径一致）：``True/False`` 在轮次字段上
    属于形态异常，按默认值展示比按 1/0 猜更诚实。
    """
    if isinstance(value, bool):
        return default
    if not isinstance(value, int):
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default
    return value


def intent_label(intent: str) -> str:
    """意图显示名（汇报与日志共用）。"""
    return INTENT_LABELS.get(intent, intent)


def decision_notice(decision: Decision, *, rounds: int, source: str, rounds_mode: str) -> str:
    """本轮开始前的一句话口径说明（"本条消息按 X 处理"+轮次取值），用户可当场纠正。

    标注（``decision.note`` 与夹取事实）由调用方统一附加一次：本函数只管口径三件套，
    避免同一条标注在"会话第 N 轮"这一行里出现两遍。
    """
    return "；".join(
        [
            f"按「{intent_label(decision.intent)}」处理",
            f"轮次上限 {rounds}（来源：{engine_mod.source_label(source)}）",
            f"模式 {rounds_mode}",
        ]
    )


def round_report(
    state: Mapping[str, Any],
    *,
    round_index: int,
    intent: str,
    run_dir: str,
    report_written: bool,
) -> str:
    """每轮汇报（**从结构化状态渲染**，不由模型复述，Design §4.7 / §8.1）。

    内容 = 轮次使用情况 + 结论 + ``next_action`` + 代码版本与产物路径。
    """
    status = str(state.get("status") or "")
    conclusion = engine_mod.conclusion_text(
        status=status,
        review_round=_as_int(state.get("review_round")),
        fix_rounds=_as_int(state.get("fix_rounds")),
        findings=state.get("findings") if isinstance(state.get("findings"), list) else [],
        severe_only=bool(state.get("severe_only")),
    )
    next_action = str(state.get("next_action") or "")
    version_dir = str(state.get("current_version_dir") or "")
    version_name = (
        version_dir.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] if version_dir else ""
    )
    files = state.get("version_files")
    file_count = (
        len([item for item in files if isinstance(item, str)]) if isinstance(files, list) else 0
    )
    lines = [
        (
            f"[会话] 第 {round_index} 轮（{intent_label(intent)}）："
            f"审查 {_as_int(state.get('review_round'))} 轮 / "
            f"修复 {_as_int(state.get('fix_rounds'))} 轮"
            f"（上限 {_as_int(state.get('max_iterations'))}"
            f"，来源：{engine_mod.source_label(state.get('max_iterations_source'))}）"
        ),
        (
            f"[会话] 结论：{conclusion}；"
            + (f"代码版本 {version_name}/（{file_count} 个文件）；" if version_name else "")
            + f"建议动作：{NEXT_ACTION_LABELS.get(next_action, next_action or '（无）')}"
        ),
    ]
    tail = "（报告 report.md）" if report_written else "（未生成报告）"
    lines.append(f"[会话] 运行记录：{run_dir}{tail}")
    return "\n".join(lines)


def session_context(
    *,
    rounds: Sequence[Mapping[str, Any]],
    current_version: int,
    version_files: Sequence[str],
) -> dict[str, Any]:
    """编排节点需要的会话现状（渲染进用户消息，见 :func:`_orchestrator_prompt`）。"""
    last = rounds[-1] if rounds else {}
    last_conclusion = str(last.get("conclusion") or "") if isinstance(last, Mapping) else ""
    return {
        "rounds": list(rounds),
        "current_version": current_version,
        "version_files": list(version_files),
        "last_conclusion": last_conclusion,
    }


__all__ = [
    "FALLBACK_ASK",
    "INTENTS",
    "INTENT_ASK",
    "INTENT_CODE",
    "INTENT_LABELS",
    "INTENT_REVIEW",
    "INTENT_REVIEW_FIX",
    "NEXT_ACTION_LABELS",
    "ORCHESTRATOR_SYSTEM_PROMPT",
    "Decision",
    "decision_notice",
    "intent_label",
    "interpret",
    "round_report",
    "session_context",
]
