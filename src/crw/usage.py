"""模型调用用量：单次调用的 token 记账、按节点累计、命中率与展示文本。

**为什么单独一层**：M4.5 起本产品在本地记账每次模型调用的输入 / 缓存命中 / 输出 token
（Design §4.5 的运行目录与 §8.2 的记录格式相应新增 ``usage`` 与 ``event=usage`` 行）。
记账数据要从框架的消息对象里取，而 LangChain 类型不得越出 :mod:`crw.agents.runtime`
（§3.1）——本模块因此只收**纯映射**（``usage_metadata`` 的形态）并只产出整数字典，
runtime、engine、render、cli 四层共用同一套口径。

字段（``FIELDS``，全部为非负整数；键名即 state.json 与 log.jsonl 里的键名）：

- ``calls``：观测到的模型调用次数（含未回报 usage 的调用）；
- ``with_usage``：回报了 token 数的调用次数——与 ``calls`` 不等即"有调用没记账"；
- ``input_tokens`` / ``output_tokens``：输入 / 输出 token 累计（输出含思考部分）；
- ``reasoning_tokens``：思考（reasoning）token 累计；供应商不回报时为 0；
- ``cached_tokens``：前缀缓存命中的输入 token 累计；
- ``cache_reported``：回报了缓存字段的调用次数；命中率只有它等于 ``with_usage`` 才可信。

"缺字段按未知处理"是本模块的硬约束：模型不带 usage（假模型、部分兼容端点）不得让
采集失败影响运行结论，因此未回报的调用只进 ``calls``，绝不虚增 token 数。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

#: token 数字段（"未回报用量"的调用这些字段恒为 0，靠 ``calls`` 与 ``with_usage`` 区分）
TOKEN_FIELDS: tuple[str, ...] = (
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cached_tokens",
)

#: 记账字段（顺序即报告与日志里的呈现顺序）
FIELDS: tuple[str, ...] = ("calls", "with_usage", *TOKEN_FIELDS, "cache_reported")

#: log.jsonl 里 usage 事件 detail 的键（计数键不入 detail：一行本就是一次调用）
_DETAIL_FIELDS: tuple[str, ...] = TOKEN_FIELDS


def empty() -> dict[str, int]:
    """空账（全 0）：合并的单位元。"""
    return dict.fromkeys(FIELDS, 0)


def is_empty(record: Mapping[str, Any]) -> bool:
    """是否一条未记到任何调用的空账。"""
    return not _as_int(record.get("calls"))


def from_usage_metadata(usage_metadata: Mapping[str, Any] | None) -> dict[str, int]:
    """一次模型调用的回报 → 记账条目（缺字段按未知处理）。

    Args:
        usage_metadata: LangChain ``AIMessage.usage_metadata`` 的形态
            （``input_tokens`` / ``output_tokens`` 与 ``input_token_details.cache_read``、
            ``output_token_details.reasoning``）。``None`` 或非映射 = 该调用未回报。

    Returns:
        ``FIELDS`` 全集的整数字典；未回报用量的调用只记 ``calls``。
    """
    record = empty()
    record["calls"] = 1
    if not isinstance(usage_metadata, Mapping):
        return record
    input_tokens = _as_int(usage_metadata.get("input_tokens"))
    output_tokens = _as_int(usage_metadata.get("output_tokens"))
    if input_tokens is None or output_tokens is None:
        return record  # 缺任一主字段都不构成一次可累计的回报（宁少记不虚增）
    record["with_usage"] = 1
    record["input_tokens"] = input_tokens
    record["output_tokens"] = output_tokens
    details = usage_metadata.get("input_token_details")
    cached = _as_int(details.get("cache_read")) if isinstance(details, Mapping) else None
    if cached is not None:
        record["cached_tokens"] = cached
        record["cache_reported"] = 1
    reasoning = _as_int(_nested(usage_metadata, "output_token_details", "reasoning"))
    if reasoning is not None:
        record["reasoning_tokens"] = reasoning
    return record


def merge(left: Mapping[str, Any] | None, right: Mapping[str, Any] | None) -> dict[str, int]:
    """两条账逐字段相加（非映射入参按空账处理）。"""
    result = empty()
    for side in (left, right):
        if not isinstance(side, Mapping):
            continue
        for field in FIELDS:
            result[field] += _as_int(side.get(field), default=0) or 0
    return result


def merge_by_node(
    left: Mapping[str, Any] | None, right: Mapping[str, Any] | None
) -> dict[str, dict[str, int]]:
    """按节点合并两张账表（``节点名 → 账``）——LangGraph 状态里 ``usage`` 键的 reducer。

    并行双视角的两个审查分支在同一个超级步里写同一个 ``usage`` 键（§12.2：同超级步多写者
    必须有 reducer），分支各自写自己的节点名，故这里是"按键并入、同键相加"；
    既有值形态不符（手改过的 state.json）按空账替换，不让坏数据把运行弄崩。
    """
    merged: dict[str, dict[str, int]] = {}
    for side in (left, right):
        if not isinstance(side, Mapping):
            continue
        for node, record in side.items():
            name = str(node)
            if not isinstance(record, Mapping):
                merged[name] = empty()
                continue
            merged[name] = merge(merged.get(name), record)
    return merged


def totals(by_node: Mapping[str, Any] | None) -> dict[str, int]:
    """一次运行的合计账（各节点相加）。"""
    result = empty()
    if isinstance(by_node, Mapping):
        for record in by_node.values():
            result = merge(result, record)
    return result


def from_log_events(events: Iterable[Any]) -> dict[str, int]:
    """从 log.jsonl 的事件回读用量合计（state.json 不可读时的降级来源）。

    detail 既接受内存里的映射，也接受落盘后的 JSON 文本（records 的 ``_preview`` 形态）；
    解析失败的行跳过——降级展示本就取自律完整的日志尾部。
    """
    result = empty()
    for event in events:
        if not isinstance(event, Mapping) or event.get("event") != "usage":
            continue
        result = merge(result, _call_from(event.get("detail")))
    return result


def _call_from(detail: Any) -> dict[str, int]:
    """一条 usage 事件的 detail → 单次调用账。"""
    payload = detail
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return empty()
    if not isinstance(payload, Mapping):
        return empty()
    record = empty()
    record["calls"] = 1
    reported = [
        value
        for value in (_as_int(payload.get(field)) for field in TOKEN_FIELDS)
        if value is not None
    ]
    if reported:
        record["with_usage"] = 1
    for field in TOKEN_FIELDS:
        value = _as_int(payload.get(field))
        if value is not None:
            record[field] = value
    if _as_int(payload.get("cached_tokens")) is not None:
        record["cache_reported"] = 1
    return record


def cache_hit_rate(record: Mapping[str, Any]) -> float | None:
    """前缀缓存命中率（命中输入 token / 输入 token）；回报不全或未回报时给 ``None``。

    只在**全部**回报了缓存字段的调用上给数：部分回报时按比例算出的命中率会把
    未回报的输入当成"未命中"，读数比不读更误导。
    """
    input_tokens = _as_int(record.get("input_tokens")) or 0
    if input_tokens <= 0:
        return None
    with_usage = _as_int(record.get("with_usage")) or 0
    if _as_int(record.get("cache_reported")) != with_usage or with_usage <= 0:
        return None
    return (_as_int(record.get("cached_tokens")) or 0) / input_tokens


#: 思考 token 占输出达到该比例时，报告与回显附一句降本提示（实测 deepseek 思考挡位下常占七八成）
THINKING_HINT_SHARE = 0.5


def thinking_share(record: Mapping[str, Any]) -> float | None:
    """思考 token 占输出的比例；输出未回报或为 0 时给 ``None``（不猜占比）。"""
    if not _as_int(record.get("with_usage")):
        return None
    output = _as_int(record.get("output_tokens"), default=0) or 0
    if output <= 0:
        return None
    return (_as_int(record.get("reasoning_tokens"), default=0) or 0) / output


def hint(record: Mapping[str, Any]) -> str:
    """降本提示：思考占比过半时给一句可操作的说明，否则空串。

    推理文本按**输出**计费（输出单价高于输入），所以"输出占比高"这件事在数据上多半
    就是思考挡位造成的——读数要顺带把这条因果讲清，用户才知道该调什么。
    """
    share = thinking_share(record)
    if share is None or share < THINKING_HINT_SHARE:
        return ""
    return (
        f"思考 token 占输出 {share:.0%}（推理文本按输出计费，输出单价高于输入）："
        "如需降低输出用量，可用 --thinking off 或在 crw config 里调整该模型的思考挡位。"
    )


def log_detail(record: Mapping[str, Any]) -> dict[str, int]:
    """单次调用进 log.jsonl 的 detail（只带 token 数；未回报用量的调用为空对象）。

    缓存字段只在供应商**真的回报了**命中数时才写：把"未回报"落成 ``cached_tokens: 0``
    会让从日志回读的降级展示报出 0% 命中率（把未知当成确认未命中），与 state.json
    口径的"缓存命中未回报"自相矛盾。
    """
    if not _as_int(record.get("with_usage")):
        return {}
    detail = {
        field: _as_int(record.get(field), default=0) or 0
        for field in _DETAIL_FIELDS
        if field != "cached_tokens"
    }
    if _as_int(record.get("cache_reported")):
        detail["cached_tokens"] = _as_int(record.get("cached_tokens"), default=0) or 0
    return detail


def cells(record: Mapping[str, Any]) -> list[str]:
    """报告用量表的一行单元：``调用 / 输入 / 缓存命中 / 输出 / 其中思考``。

    "未回报"与"0"是两回事：未回报用量的调用不把 token 栏写成 0（读数的人会当成
    真的没花钱）；思考栏为 0 时写全角破折号——多数不支持思考的供应商根本没有这一项。
    """
    reported = bool(_as_int(record.get("with_usage")))
    rate = cache_hit_rate(record)
    reasoning = _as_int(record.get("reasoning_tokens"), default=0) or 0
    return [
        _num(record.get("calls")),
        _num(record.get("input_tokens")) if reported else "未回报",
        f"{_num(record.get('cached_tokens'))}（{rate:.0%}）" if rate is not None else "未回报",
        _num(record.get("output_tokens")) if reported else "未回报",
        _num(reasoning) if reasoning else "—",
    ]


def note(record: Mapping[str, Any]) -> str:
    """账目本身的缺陷说明（数据不全时给人读的交代）；数据完整返回空串。

    只交代"有多少调用没回报"，不复述表里已有的 token 数。
    """
    calls = _as_int(record.get("calls"), default=0) or 0
    with_usage = _as_int(record.get("with_usage"), default=0) or 0
    if calls and with_usage < calls:
        return f"其中 {calls - with_usage} 次调用未回报 usage（按未知处理，未计入 token 数）"
    if with_usage and not _as_int(record.get("cache_reported")):
        return "该供应商未回报前缀缓存命中数"
    return ""


def format_line(record: Mapping[str, Any]) -> str:
    """一行展示文本（``crw show`` / ``crw review`` 的用量行、报告的合计行）。

    三种如实形态：无任何调用（未记录）/ 有调用但未回报 usage（用量未知，不给 0 数）/
    正常账（命中率仅在全部回报了缓存字段时给出）。
    """
    calls = _as_int(record.get("calls"), default=0) or 0
    if calls == 0:
        return "（未记录用量）"
    with_usage = _as_int(record.get("with_usage"), default=0) or 0
    if with_usage == 0:
        return f"用量未知（{_num(calls)} 次调用未回报 usage）"
    output = f"输出 {_num(record.get('output_tokens'))}"
    reasoning = _as_int(record.get("reasoning_tokens"), default=0) or 0
    if reasoning:
        output += f"（思考 {_num(reasoning)}）"
    parts = [f"输入 {_num(record.get('input_tokens'))}", output]
    rate = cache_hit_rate(record)
    parts.append(
        f"缓存命中 {_num(record.get('cached_tokens'))}（{rate:.0%}）"
        if rate is not None
        else "缓存命中未回报"
    )
    silent = calls - with_usage
    calls_note = f"{_num(calls)} 次调用"
    if silent:
        calls_note += f"，其中 {_num(silent)} 次未回报用量"
    parts.append(calls_note)
    return " · ".join(parts)


def _nested(data: Mapping[str, Any], outer: str, inner: str) -> Any:
    """两层映射取值（``input_token_details`` / ``output_token_details`` 形态）。"""
    layer = data.get(outer)
    if not isinstance(layer, Mapping):
        return None
    return layer.get(inner)


def _num(value: Any) -> str:
    """千分位数字（读数时位数不易看错）。"""
    return f"{_as_int(value, default=0) or 0:,}"


def _as_int(value: Any, *, default: int | None = None) -> int | None:
    """非负整数取值：布尔 / 负数 / 非整数一律按未回报（``default``）。"""
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    if value < 0:
        return default
    return value


__all__ = [
    "FIELDS",
    "TOKEN_FIELDS",
    "cache_hit_rate",
    "cells",
    "empty",
    "format_line",
    "from_log_events",
    "from_usage_metadata",
    "is_empty",
    "log_detail",
    "merge",
    "merge_by_node",
    "note",
    "totals",
]
