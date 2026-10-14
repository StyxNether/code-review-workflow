"""usage 模块：单次调用的 token 记账、按节点累计、命中率与展示文本。

全部为纯数据用例（本模块不含框架类型）：覆盖"缺字段按未知处理"、并行分支的按节点合并、
降级态从 log.jsonl 的 usage 事件回读。
"""

from __future__ import annotations

from crw import usage

#: 一条 DeepSeek 风格的完整回报（缓存命中与思考 token 都在）
DEEPSEEY_STYLE = {
    "input_tokens": 5231,
    "output_tokens": 412,
    "total_tokens": 5643,
    "input_token_details": {"cache_read": 4096},
    "output_token_details": {"reasoning": 300},
}


def test_missing_usage_is_counted_as_a_call_only() -> None:
    """假模型不带 usage：记一次调用、token 全按未知（0），with_usage=0。"""
    record = usage.from_usage_metadata(None)
    assert record == {
        "calls": 1,
        "with_usage": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "cache_reported": 0,
    }


def test_full_usage_reads_cache_and_reasoning() -> None:
    """完整回报：输入 / 输出 / 缓存命中 / 思考 token 各归其位。"""
    record = usage.from_usage_metadata(DEEPSEEY_STYLE)
    assert record["input_tokens"] == 5231
    assert record["output_tokens"] == 412
    assert record["cached_tokens"] == 4096
    assert record["reasoning_tokens"] == 300
    assert record["with_usage"] == 1
    assert record["cache_reported"] == 1


def test_cache_field_absent_is_not_reported() -> None:
    """供应商未回报缓存字段：cache_reported 不计数（命中率因此不可信）。"""
    record = usage.from_usage_metadata({"input_tokens": 10, "output_tokens": 5})
    assert record["with_usage"] == 1
    assert record["cached_tokens"] == 0
    assert record["cache_reported"] == 0


def test_junk_values_are_treated_as_unknown() -> None:
    """字段值形态不符（字符串 / 负数 / 布尔 / 嵌套非映射）一律按未知，不崩也不虚增。"""
    record = usage.from_usage_metadata(
        {
            "input_tokens": "5231",
            "output_tokens": -3,
            "input_token_details": "not-a-mapping",
            "output_token_details": {"reasoning": True},
        }
    )
    assert record["with_usage"] == 0
    assert record["calls"] == 1
    assert all(record[field] == 0 for field in usage.TOKEN_FIELDS)


def test_non_mapping_usage_metadata_is_unknown() -> None:
    """usage_metadata 不是映射（框架升级换了形态）：按未回报处理。"""
    assert usage.from_usage_metadata("weird")["with_usage"] == 0  # type: ignore[arg-type]


def test_merge_sums_every_field() -> None:
    """合并两次调用：各字段逐项相加，不丢缓存回报计数。"""
    left = usage.from_usage_metadata(DEEPSEEY_STYLE)
    right = usage.from_usage_metadata({"input_tokens": 100, "output_tokens": 20})
    merged = usage.merge(left, right)
    assert merged["calls"] == 2
    assert merged["input_tokens"] == 5331
    assert merged["output_tokens"] == 432
    assert merged["cached_tokens"] == 4096
    assert merged["cache_reported"] == 1


def test_merge_with_empty_is_identity() -> None:
    """与空账合并不得改动任何字段（None / 非映射入参同样退回另一侧）。"""
    record = usage.from_usage_metadata(DEEPSEEY_STYLE)
    assert usage.merge(usage.empty(), record) == record
    assert usage.merge(None, record) == record  # type: ignore[arg-type]


def test_merge_by_node_keeps_branches_apart() -> None:
    """并行双视角：两个分支写同一状态键但节点名不同 → 按节点并入。"""
    left = {
        "review_correctness": usage.from_usage_metadata({"input_tokens": 1, "output_tokens": 10})
    }
    right = {"review_security": usage.from_usage_metadata({"input_tokens": 2, "output_tokens": 20})}
    merged = usage.merge_by_node(left, right)
    assert set(merged) == {"review_correctness", "review_security"}
    assert usage.totals(merged)["output_tokens"] == 30


def test_merge_by_node_sums_same_node() -> None:
    """同一节点跨轮累加（审查跑多轮时该节点的用量是各轮之和）。"""
    one = usage.from_usage_metadata({"input_tokens": 50, "output_tokens": 5})
    merged = usage.merge_by_node({"review": one}, {"review": one})
    assert merged["review"]["input_tokens"] == 100
    assert merged["review"]["calls"] == 2


def test_merge_by_node_tolerates_broken_state() -> None:
    """状态里已有坏形态（手工改过的 state.json）：非映射值按空账替换，不抛异常。"""
    one = usage.from_usage_metadata({"input_tokens": 50, "output_tokens": 5})
    merged = usage.merge_by_node({"review": "损坏"}, {"review": one})
    assert merged["review"]["calls"] == 1


def test_totals_across_nodes() -> None:
    """运行级合计 = 各节点之和；空表给出全 0 账。"""
    by_node = {
        "encode": usage.from_usage_metadata({"input_tokens": 300, "output_tokens": 900}),
        "review": usage.from_usage_metadata(DEEPSEEY_STYLE),
    }
    totals = usage.totals(by_node)
    assert totals["calls"] == 2
    assert totals["input_tokens"] == 300 + 5231
    assert totals["reasoning_tokens"] == 300
    assert usage.totals({}) == dict.fromkeys(usage.FIELDS, 0)


def test_cache_hit_rate_only_when_fully_reported() -> None:
    """命中率：全部调用都回报缓存字段才给数；部分回报或零回报给 None。"""
    all_reported = usage.totals({"review": usage.from_usage_metadata(DEEPSEEY_STYLE)})
    assert usage.cache_hit_rate(all_reported) == 4096 / 5231
    partial = usage.merge(
        usage.from_usage_metadata(DEEPSEEY_STYLE),
        usage.from_usage_metadata({"input_tokens": 100, "output_tokens": 1}),
    )
    assert usage.cache_hit_rate(partial) is None
    assert usage.cache_hit_rate(usage.empty()) is None


def test_cache_hit_rate_zero_input_is_none() -> None:
    """输入为 0（未回报用量的调用）：不做除零，命中率给 None。"""
    assert usage.cache_hit_rate({"calls": 1, "input_tokens": 0}) is None


def test_format_line_covers_known_partial_and_unknown() -> None:
    """展示文本三种形态：完整（含命中率）/ 未回报用量 / 无用量。"""
    known = usage.totals({"review": usage.from_usage_metadata(DEEPSEEY_STYLE)})
    line = usage.format_line(known)
    assert "输入 5,231" in line
    assert "输出 412" in line
    assert "思考 300" in line
    assert "缓存命中 4,096（78%）" in line
    assert "1 次调用" in line

    unknown = usage.merge(usage.empty(), usage.from_usage_metadata(None))
    assert usage.format_line(unknown) == "用量未知（1 次调用未回报 usage）"
    assert usage.format_line(usage.empty()) == "（未记录用量）"


def test_format_line_notes_calls_without_usage() -> None:
    """部分调用未回报：如实标注次数，不把未知混进已知数里。"""
    mixed = usage.merge(usage.from_usage_metadata(DEEPSEEY_STYLE), usage.from_usage_metadata(None))
    assert "1 次未回报用量" in usage.format_line(mixed)


def test_log_detail_and_events_round_trip() -> None:
    """每次调用的 detail 形态可被回读汇总（dict 与落盘后的 JSON 文本都接受）。"""
    call = usage.from_usage_metadata(DEEPSEEY_STYLE)
    events = [
        {"node": "review", "event": "usage", "detail": usage.log_detail(call)},
        {"node": "review", "event": "tool", "detail": "read_file(path=main.py)"},
        {"node": "review", "event": "usage", "detail": '{"input_tokens": 100}'},
        {"node": "review", "event": "usage", "detail": "不是 JSON"},
    ]
    totals = usage.from_log_events(events)
    assert totals["input_tokens"] == 5231 + 100
    assert totals["cached_tokens"] == 4096


def test_log_detail_omits_unknown_call() -> None:
    """未回报用量的调用：detail 为空对象（日志里如实留一条"这次没有回报"）。"""
    assert usage.log_detail(usage.from_usage_metadata(None)) == {}


def test_log_detail_omits_unreported_cache_field() -> None:
    """供应商回报了 token 但没有缓存字段：detail 不写 cached_tokens=0。

    写了就分不清"确实未命中"与"没回报"，从 log 回读会报出 0% 命中率；
    state.json 口径同样情形写"缓存命中未回报"——两个数据源必须给同一个结论。
    """
    record = usage.from_usage_metadata({"input_tokens": 100, "output_tokens": 20})
    detail = usage.log_detail(record)
    assert "cached_tokens" not in detail
    assert detail["input_tokens"] == 100
    recovered = usage.from_log_events([{"event": "usage", "detail": detail}])
    assert usage.cache_hit_rate(recovered) is None
    assert "缓存命中未回报" in usage.format_line(recovered)


def test_log_detail_keeps_a_real_zero_hit() -> None:
    """回报了缓存字段且真是 0：detail 保留 0，回读得到 0% 而不是"未回报"。"""
    record = usage.from_usage_metadata(
        {"input_tokens": 100, "output_tokens": 20, "input_token_details": {"cache_read": 0}}
    )
    assert usage.log_detail(record)["cached_tokens"] == 0
    recovered = usage.from_log_events([{"event": "usage", "detail": usage.log_detail(record)}])
    assert usage.cache_hit_rate(recovered) == 0.0


def test_field_names_are_stable() -> None:
    """字段名即 state.json / log.jsonl 里的键名：改动属记录格式变更，须同步 Design。"""
    assert usage.FIELDS == (
        "calls",
        "with_usage",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cached_tokens",
        "cache_reported",
    )


def test_thinking_share_and_hint_fire_only_when_dominant() -> None:
    """思考占比过半才提示降本：低挡位/不思考的模型不该被唠叨。"""
    heavy = usage.merge(
        usage.from_usage_metadata(
            {"input_tokens": 100, "output_tokens": 100, "output_token_details": {"reasoning": 71}}
        ),
        usage.empty(),
    )
    assert usage.thinking_share(heavy) == 0.71
    assert "思考 token 占输出 71%" in usage.hint(heavy)
    assert "--thinking off" in usage.hint(heavy)

    light = usage.from_usage_metadata(
        {"input_tokens": 100, "output_tokens": 100, "output_token_details": {"reasoning": 20}}
    )
    assert usage.hint(light) == ""


def test_thinking_hint_needs_reported_output() -> None:
    """输出未回报（或为 0）：不算占比、也不提示。"""
    assert usage.thinking_share(usage.empty()) is None
    assert usage.hint(usage.empty()) == ""
    unknown = usage.from_usage_metadata(None)
    assert usage.thinking_share(unknown) is None
