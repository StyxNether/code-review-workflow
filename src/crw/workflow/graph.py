"""工作流图组装：StateGraph 的节点、边、条件路由与复审循环。

节点以“节点名 → 节点函数”的覆盖表注入，未覆盖的键使用调用方给出的缺省装配（生产路径
为 agents 层的真实 Agent 节点，测试为脚本化假节点）；LangGraph 类型不越出本层
（Design §3.1、§4.1、§12.2）。

节点输出协议（Design §3.1）：节点为纯函数 ``node(state) -> dict``，只返回增量字段；原始
文本经暂存键交回 workflow 层消费——``encode`` 用 ``code_text``、``review`` 用
``review_text``、``fix`` 用 ``fix_text``。暂存键不在状态 schema 内，不进入运行记录；
阶段摘要是引擎按产物与契约结论自算的（节点不产出摘要文本）。

审查节点额外可返回 ``review_evidence: bool``（本轮是否至少一次成功工具事件，Design §5.2
的执行有效性门禁）；不返回该键的节点（脚本化假节点 / 注入桩）按"内联送达"视为满足门禁。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from . import state as state_mod
from .state import RunState

#: 图节点名（Design §4.1）
NODE_NAMES: tuple[str, ...] = ("encode", "review", "fix")

#: 节点函数类型：``(state[, retry_feedback=...]) -> Mapping[str, Any]``（Design §3.1）
NodeFn = Callable[..., Mapping[str, Any]]


def resolve_nodes(
    nodes: Mapping[str, NodeFn] | None = None, *, defaults: Mapping[str, NodeFn] | None = None
) -> dict[str, NodeFn]:
    """合并覆盖表与缺省装配：逐键替换、未给出的键用缺省（Design §12.2 第 3 步）。

    Args:
        nodes: 节点覆盖表（键为 ``encode`` / ``review`` / ``fix``）；逐键替换缺省节点。
        defaults: 缺省装配（生产路径 = agents 的真实 Agent 节点；测试 = 脚本化假节点）。
            省略时要求 ``nodes`` 覆盖全部节点名——宁可显式报错，也不要静默走到一个
            没接模型的实现上。

    Raises:
        ValueError: 覆盖表含未知节点名，或缺省装配与覆盖表未能补齐全部节点。
    """
    base: dict[str, NodeFn] = dict(defaults) if defaults else {}
    if nodes:
        unknown = sorted(set(nodes) - set(NODE_NAMES))
        if unknown:
            raise ValueError(f"未知节点名：{unknown}（可用：{list(NODE_NAMES)}）")
        base.update(nodes)
    missing = [name for name in NODE_NAMES if name not in base]
    if missing:
        raise ValueError(
            f"缺少节点装配：{missing}（可用：{list(NODE_NAMES)}）——请给出完整覆盖表或提供缺省装配。"
        )
    return base


def build_graph(
    *,
    nodes: Mapping[str, NodeFn] | None = None,
    defaults: Mapping[str, NodeFn] | None = None,
) -> CompiledStateGraph:
    """组装并编译工作流图（Design §4.1 / §4.4）。

    Args:
        nodes: 节点覆盖表；逐键替换缺省装配（测试与真实装配共用该接缝）。
        defaults: 缺省装配（见 :func:`resolve_nodes`）。

    Returns:
        编译后的 StateGraph；调用方负责 ``stream`` 驱动与 ``recursion_limit``。
    """
    resolved = resolve_nodes(nodes, defaults=defaults)
    builder = StateGraph(RunState)
    for name in NODE_NAMES:
        builder.add_node(name, resolved[name])
    builder.add_conditional_edges(START, route_entry, {"encode": "encode", "review": "review"})
    builder.add_edge("encode", "review")
    builder.add_conditional_edges("review", route_after_review, {"fix": "fix", END: END})
    builder.add_edge("fix", "review")
    return builder.compile()


def route_entry(state: Mapping[str, Any]) -> str:
    """入口路由（只读纯函数）：需求文本从编码起步；代码文件直接进入审查（Design §4.4）。"""
    descriptor = state.get("input") or {}
    return "encode" if descriptor.get("kind") == state_mod.INPUT_REQUIREMENT else "review"


def route_after_review(state: Mapping[str, Any]) -> str:
    """复审条件边（只读纯函数）：干净 → 收尾；达修复轮次上限 → 收尾（残留）；否则进入修复。

    干净判定与轮次口径见 Design §4.4；``rounds-mode=exact``（跑满 n 轮）随 M4 的会话轮次
    语义落地，届时本函数按 ``rounds_mode`` 分支。
    """
    if state.get("clean"):
        return END
    if int(state.get("fix_rounds", 0)) >= int(state.get("max_iterations", 0)):
        return END
    return "fix"
