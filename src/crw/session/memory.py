"""会话记忆：三 Agent 各自的追加式记忆文件（落盘 / 载入 / 阈值压缩，Design §4.7）。

只追加、绝不改写——记忆是模型请求的前缀，原地改写会让供应商的前缀缓存整体失效；
接近上下文预算时做一次显式压缩（:meth:`AgentMemory.compact`）：追加一条
``{"kind": "compaction", …}`` 行让旧行退出后续请求，旧行本身仍完整留在磁盘上；
需要继续参与请求的最近若干行以"快照 + 尾行"方式原文续写在压缩行之后。

本模块不接触代码与思考内容：调用方只传文本摘要，截断与折叠都在文本层完成；
token 估算与 ``agents/runtime.py`` 的 ``_estimate_tokens`` 同口径（字符数 × 0.6）。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

#: 三 Agent 的记忆文件名（不含扩展名），与 workflow 节点名映射：encode→encoder / review→reviewer / fix→fixer
MEMORY_AGENTS: tuple[str, ...] = ("encoder", "reviewer", "fixer")

#: 压缩阈值比例（预算的 70%）
COMPACTION_RATIO = 0.7

#: 单条记忆行的文本上限（超出折行尾加 "…（已截断）"；上限含该后缀）
MAX_MEMORY_TEXT_CHARS = 2000

#: 压缩摘要的总长上限：被折叠记录多到装不下时，保留较新的轮次并省略更早的
_SUMMARY_CHARS = 1500
#: 摘要里单条要点的限长（超长以 "…" 收尾，不再叠加"已截断"长尾）
_SUMMARY_ITEM_CHARS = 240
#: 摘要因超限省略更早记录时的收尾标记
_EARLIER_OMITTED = "…（更早记录略）"

#: 单条文本的截断后缀（与 MAX_MEMORY_TEXT_CHARS 配套）
_TRUNCATED_SUFFIX = "…（已截断）"

#: compaction 行在 prompt_history 里的固定前缀（旧结论以 user 角色回放给模型）
_COMPACTION_PREFIX = "（此前会话的压缩记录）"

#: 摘要内多要点 / 多轮次的分隔符
_SUMMARY_JOIN = "；"

#: 记忆行的声明字段白名单：append 只落盘这些键
_LINE_KINDS = ("message", "compaction")
_MESSAGE_ROLES = ("user", "assistant")

#: 字符估算系数（与 runtime._estimate_tokens 的折中系数一致）
_TOKEN_FACTOR = 0.6


def memory_file(session_dir: Path, agent: str) -> Path:
    """该会话下某 Agent 的记忆文件路径（``memory/<agent>.jsonl``）。

    agent 走白名单而非"清掉分隔符"式校验：名字直接拼进路径，白名单保证任何合法名
    都只能是单个路径成分；非法名抛 ``ValueError``（与运行号白名单同一防穿越口径）。
    """
    if agent not in MEMORY_AGENTS:
        raise ValueError(
            f"记忆文件 agent 名非法：{agent!r}（应为 {' / '.join(MEMORY_AGENTS)} 之一）"
        )
    return session_dir / "memory" / f"{agent}.jsonl"


class AgentMemory:
    """单个 Agent 的追加式记忆：只追加、可载入、按阈值压缩（Design §4.7）。

    为什么追加式：记忆以"前缀稳定"的方式喂给模型来摊薄供应商缓存成本；压缩不改写
    旧行，而是追加一条 compaction 行，由 :meth:`effective_lines` 判定旧行不再参与
    后续请求（磁盘上仍可审计全部历史）。

    Args:
        path: 记忆文件路径（通常由 :func:`memory_file` 生成）。
        agent: 所属 Agent 名（仅用于错误文案，缺省不标注）。
    """

    def __init__(self, path: Path, *, agent: str = "") -> None:
        self._path = path
        self._agent = agent

    @property
    def path(self) -> Path:
        """记忆文件路径。"""
        return self._path

    def lines(self) -> list[dict[str, Any]]:
        """已落盘的全部行（含 compaction 行与 role 行）；文件缺失 / 损坏行 / 非对象行跳过。

        只读，不修改磁盘；单行损坏不影响其余行（与 ``log.jsonl`` 的降级口径一致）。
        """
        try:
            text = self._path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []
        items: list[dict[str, Any]] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue  # 写一半的行 / 损坏行：不影响其余记忆
            if isinstance(parsed, dict):
                items.append(parsed)
        return items

    def effective_lines(self) -> list[dict[str, Any]]:
        """当前参与后续请求的行：最后一条 compaction 行 + 其后的全部 role 行。

        无 compaction 时为全部 role 行；被压缩掉的旧行只有磁盘留档、不再出现。
        """
        all_lines = self.lines()
        last = -1
        for index, line in enumerate(all_lines):
            if line.get("kind") == "compaction":
                last = index
        if last < 0:
            return [line for line in all_lines if _is_role_line(line)]
        return [all_lines[last], *(line for line in all_lines[last + 1 :] if _is_role_line(line))]

    def append(self, records: Sequence[Mapping[str, Any]]) -> None:
        """追加若干行：逐行 append + flush；每行一个 JSON 对象，中文不转义。

        行内容只保留声明字段（message 行 kind/round/role/text/ts；compaction 行
        kind/round/ts/covers/summary），文本按 :data:`MAX_MEMORY_TEXT_CHARS` 截断。
        全部行先整体校验再写盘；写入失败原样上抛 ``OSError``（由调用方决定如何报错）。
        """
        if not records:
            return
        normalized = [_normalize_line(record, agent=self._agent) for record in records]
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8", newline="\n") as handle:
            for line in normalized:
                handle.write(json.dumps(line, ensure_ascii=False) + "\n")
                handle.flush()

    def append_exchange(self, *, round_index: int, task: str, outcome: str, ts: str) -> None:
        """便捷追加一轮对话：user 行（``task``）与 assistant 行（``outcome``）各一条。"""
        self.append(
            [
                {"kind": "message", "round": round_index, "role": "user", "text": task, "ts": ts},
                {
                    "kind": "message",
                    "round": round_index,
                    "role": "assistant",
                    "text": outcome,
                    "ts": ts,
                },
            ]
        )

    def prompt_history(self) -> list[dict[str, str]]:
        """供 Agent 运行时使用的历史消息：``[{"role": …, "text": …}]``。

        compaction 行回放为一条 ``role="user"`` 的行（文本以"（此前会话的压缩记录）"
        开头）：被折叠的结论必须仍能抵达模型，但不得再作为 assistant 轮次参与对话。
        """
        history: list[dict[str, str]] = []
        for line in self.effective_lines():
            if line.get("kind") == "compaction":
                summary = _text_of(line, key="summary")
                history.append(
                    {"role": "user", "text": f"{_COMPACTION_PREFIX}\n{summary}".rstrip()}
                )
            else:
                history.append({"role": str(line.get("role")), "text": _text_of(line)})
        return history

    def estimate_tokens(self) -> int:
        """有效内容（:meth:`effective_lines`）的字符估算：总字符 × 0.6。

        与 ``agents/runtime.py`` 同口径（中英混合折中系数的止损护栏，不是精确计量）；
        只计正文（text / summary），不计 JSON 键等结构性开销。
        """
        return _estimate_texts(_content_texts(self.effective_lines()))

    def needs_compaction(self, max_context_tokens: int, *, ratio: float = COMPACTION_RATIO) -> bool:
        """估算是否已接近上下文预算（阈值 = ``ratio`` × ``max_context_tokens``）。

        ``max_context_tokens`` 非正抛 ``ValueError``：预算为 0 会让判定恒真，
        属调用方配置错误（与 ``config.get_max_context_tokens`` 的校验口径一致）。
        """
        if max_context_tokens <= 0:
            raise ValueError(f"上下文预算必须为正整数：{max_context_tokens!r}")
        return self.estimate_tokens() >= ratio * max_context_tokens

    def _previous_summary(self) -> str:
        """上一条 compaction 行的摘要文本（无压缩记录时为空串）。"""
        for line in reversed(self.effective_lines()):
            if line.get("kind") == "compaction":
                return _text_of(line, key="summary")
        return ""

    def compact(self, *, round_index: int, ts: str, keep_recent: int = 4) -> dict[str, Any] | None:
        """把较早已不起作用的 role 行折叠成一条 compaction 行，**追加**到文件末尾。

        追加顺序 = compaction 行 + 最近 ``keep_recent`` 条 role 行的原文续写：旧行留在
        磁盘上、绝不改写，而保留行复制到压缩行之后才能继续落在
        :meth:`effective_lines` 的"compaction 之后"一侧、参与后续请求（追加式"快照 +
        尾行"）。新摘要会**并入上一条压缩摘要**（摘要链），被折叠的结论因此只以摘要形态
        退出原文层、不会整段丢失；形态不完整的手改行不参与续写（磁盘留档仍在）。
        可折叠的行不足，或折叠后（摘要 + 保留行）的估算不比现状小（"越压越
        大"的空转）时返回 ``None`` 且不写盘；否则返回写入的 compaction 行本身。
        ``keep_recent`` 为负抛 ``ValueError``（0 = 全部 role 行都可折叠）。
        """
        if keep_recent < 0:
            raise ValueError(f"keep_recent 不能为负：{keep_recent!r}")
        roles = [line for line in self.effective_lines() if _is_role_line(line)]
        foldable = roles if keep_recent == 0 else roles[:-keep_recent]
        if not foldable:
            return None
        summary = _summarize(foldable)
        if not summary:
            return None
        # 摘要链：上一次压缩的摘要接在本次之前——被折叠的结论只以摘要形态留在请求里，
        # 不该因为"又压了一次"而整段消失（旧行仍在磁盘上可审计）
        summary = _chain_summary(self._previous_summary(), summary)
        kept = [line for line in roles[len(foldable) :] if _reappendable(line)]
        current = self.estimate_tokens()
        projected = _estimate_texts([summary, *_content_texts(kept)])
        if projected >= current:
            return None  # 无收益：摘要 + 保留行不比现状小，写盘只会让后续请求更长
        rounds = [value for line in foldable if (value := _round_of(line)) is not None]
        line: dict[str, Any] = {
            "kind": "compaction",
            "round": max(rounds) if rounds else round_index,
            "ts": ts,
            "covers": len(foldable),
            "summary": summary,
        }
        self.append([line, *kept])
        return line


def _reappendable(line: Mapping[str, Any]) -> bool:
    """该行能否原文续写进下一次请求（形态不完整的手改行跳过：磁盘留档仍在，不进请求）。"""
    try:
        _normalize_line(line, agent="")
    except (TypeError, ValueError):
        return False
    return True


def _chain_summary(previous: str, current: str) -> str:
    """摘要链：把上一次压缩的摘要接在本次摘要之前。

    两段合计超上限时保留较新的本次摘要（更早的一段整体略去并以标记收尾）——旧行始终
    留在磁盘上，信息不丢，只是不再进入请求。
    """
    if not previous:
        return current
    if previous == current:
        return current
    return _assemble_summary([previous, current])


def _is_role_line(line: Mapping[str, Any]) -> bool:
    """是否为参与对话的 role 行（compaction 行与形态不明的行都不算）。"""
    return line.get("role") in _MESSAGE_ROLES


def _text_of(line: Mapping[str, Any], *, key: str = "text") -> str:
    """行内文本字段；形态不符退化为空串（载入面向任意历史文件，不因单行形状崩溃）。"""
    value = line.get(key)
    return value if isinstance(value, str) else ""


def _round_of(line: Mapping[str, Any]) -> int | None:
    """行的来源轮次；缺失 / 非整数（含 bool）返回 ``None``。"""
    value = line.get("round")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _clip_text(text: str) -> str:
    """按 :data:`MAX_MEMORY_TEXT_CHARS` 截断单条行文本（长度含后缀不超上限）。"""
    if len(text) <= MAX_MEMORY_TEXT_CHARS:
        return text
    keep = max(0, MAX_MEMORY_TEXT_CHARS - len(_TRUNCATED_SUFFIX))
    return text[:keep] + _TRUNCATED_SUFFIX


def _content_texts(lines: Iterable[Mapping[str, Any]]) -> list[str]:
    """行的正文序列（message 取 text、compaction 取 summary）：估算与收益判定的共同口径。"""
    texts: list[str] = []
    for line in lines:
        texts.append(
            _text_of(line, key="summary") if line.get("kind") == "compaction" else _text_of(line)
        )
    return texts


def _estimate_texts(texts: Iterable[str]) -> int:
    """文本总字符 × 0.6（与 ``agents/runtime.py`` 的 ``_estimate_tokens`` 同口径）。"""
    return int(sum(len(text) for text in texts) * _TOKEN_FACTOR)


def _normalize_line(record: Mapping[str, Any], *, agent: str) -> dict[str, Any]:
    """按声明字段白名单整理一行（行形态是对外契约：外来键与超长文本不得落盘）。

    类型错用抛 ``TypeError``、取值非法抛 ``ValueError``，消息都为可读中文：记忆是所有
    后续请求的前缀，宁可让调用方立刻看到错误，也不能把形态不明的行写进记忆。
    """
    who = f"「{agent}」记忆" if agent else "记忆"
    if not isinstance(record, Mapping):
        raise TypeError(f"{who}行必须是映射对象：{record!r}")
    kind = record.get("kind") or "message"
    if kind not in _LINE_KINDS:
        raise ValueError(f"{who}行的 kind 非法：{kind!r}（应为 message / compaction）")
    number = _require_int(record.get("round"), who, "round")
    timestamp = record.get("ts")
    if not isinstance(timestamp, str) or not timestamp:
        raise ValueError(f"{who}行的 ts 非法：{timestamp!r}（应为非空字符串）")
    if kind == "message":
        role = record.get("role")
        if role not in _MESSAGE_ROLES:
            raise ValueError(f"{who}行的 role 非法：{role!r}（应为 user / assistant）")
        text = record.get("text")
        if not isinstance(text, str):
            raise TypeError(f"{who}行的 text 应为字符串，得到 {type(text).__name__}")
        return {
            "kind": "message",
            "round": number,
            "role": role,
            "text": _clip_text(text),
            "ts": timestamp,
        }
    summary = record.get("summary")
    if not isinstance(summary, str):
        raise TypeError(f"{who}行的 summary 应为字符串，得到 {type(summary).__name__}")
    return {
        "kind": "compaction",
        "round": number,
        "ts": timestamp,
        "covers": _require_int(record.get("covers"), who, "covers"),
        "summary": _clip_text(summary),
    }


def _require_int(value: Any, who: str, field: str) -> int:
    """整数校验（bool 不算：``True`` 当轮次或计数会静默错位）；类型不符抛 ``TypeError``。"""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{who}行的 {field} 应为整数，得到 {value!r}")
    return value


def _round_points(lines: Sequence[Mapping[str, Any]]) -> list[str]:
    """该轮折叠行的要点文本：优先 assistant 结论，无 assistant 时退回 user 文本。

    记忆是 Agent 自己的历史，摘要以"我上一轮给出了什么"为主；只有 user 行的轮次
    （例如该轮未收尾）也不能在摘要里消失。
    """
    points = [
        text
        for text in (_text_of(line) for line in lines if line.get("role") == "assistant")
        if text
    ]
    if points:
        return points
    return [
        text for text in (_text_of(line) for line in lines if line.get("role") == "user") if text
    ]


def _clip_item(text: str) -> str:
    """摘要里单条要点的限长（短标记收尾，避免摘要被单条长文本挤满）。"""
    if len(text) <= _SUMMARY_ITEM_CHARS:
        return text
    return text[:_SUMMARY_ITEM_CHARS] + "…"


def _sorted_rounds(grouped: Mapping[int | None, Sequence[Mapping[str, Any]]]) -> list[int | None]:
    """轮次升序；轮次不详（``None``）排在最后。"""
    numbered = sorted(key for key in grouped if key is not None)
    return [*numbered, *([None] if None in grouped else [])]


def _summarize(folded: Sequence[Mapping[str, Any]]) -> str:
    """折叠行 → 摘要文本：按来源轮次归组（``第 1 轮：<要点>``），超上限保留较新轮次。

    同一轮的多条要点用"；"连接、逐条限长；没有任何要点时返回空串（调用方据此
    判"无可摘要"，避免写出空摘要的 compaction 行）。
    """
    grouped: dict[int | None, list[Mapping[str, Any]]] = {}
    for line in folded:
        grouped.setdefault(_round_of(line), []).append(line)
    segments: list[str] = []
    for key in _sorted_rounds(grouped):
        points = _round_points(grouped[key])
        if not points:
            continue
        label = f"第 {key} 轮：" if key is not None else "（轮次不详）："
        segments.append(f"{label}{_SUMMARY_JOIN.join(_clip_item(point) for point in points)}")
    if not segments:
        return ""
    return _assemble_summary(segments)


def _assemble_summary(segments: Sequence[str]) -> str:
    """拼接摘要段：超上限时保留较新的折叠轮次，并以"…（更早记录略）"收尾。

    摘要服务的是后续请求的连续性，最近折进去的结论价值最高；被略去的更早记录
    仍完整留在磁盘上可审计。
    """
    kept: list[str] = []
    total = 0
    omitted = False
    for segment in reversed(segments):
        cost = len(segment) + (len(_SUMMARY_JOIN) if kept else 0)
        if total + cost > _SUMMARY_CHARS:
            omitted = True
            break
        kept.append(segment)
        total += cost
    if not kept:
        # 最新一条摘要段本身就超上限（同一轮要点极多）：截断保留，避免摘要只剩省略标记
        kept = [segments[-1][:_SUMMARY_CHARS]]
        omitted = True
    text = _SUMMARY_JOIN.join(reversed(kept))
    if omitted:
        text += _EARLIER_OMITTED
    return text


__all__ = [
    "COMPACTION_RATIO",
    "MAX_MEMORY_TEXT_CHARS",
    "MEMORY_AGENTS",
    "AgentMemory",
    "memory_file",
]
