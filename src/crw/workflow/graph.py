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

两种图形态：

- ``single``（缺省）：``encode → review → fix``（+ 复审循环），节点名见 :data:`SINGLE_NODE_NAMES`；
- ``parallel``（扩展指南的落地示例，Design §12.2）：``encode → {review_correctness,
  review_security}``（同一超级步并行，各写**独立 key**）``→ merge_reviews``（按
  ``(location, issue)`` 去重取较严重者、统一重编号 id、推进审查轮次）``→`` 沿用同一条
  复审条件边与修复循环。汇合节点无条件覆盖 ``findings``（未运行分支的旧值不得残留）。

``mermaid_flowchart`` 与 :func:`build_graph` 描述同一组节点 / 边（``crw graph`` 由此导出，
快照测试锁定）；``recursion_limit`` 按形态复算——并行形态每轮多一个汇合超级步。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from . import contracts
from . import state as state_mod
from .state import RunState

#: 默认（单视角）图的节点名（Design §4.1）
SINGLE_NODE_NAMES: tuple[str, ...] = ("encode", "review", "fix")

#: 并行双视角图的节点名（Design §12.2 的扩展示例）
PARALLEL_NODE_NAMES: tuple[str, ...] = (
    "encode",
    "review_correctness",
    "review_security",
    "merge_reviews",
    "fix",
)

#: 兼容别名：默认图的节点名（既有调用方按此引用）
NODE_NAMES: tuple[str, ...] = SINGLE_NODE_NAMES

#: 并行审查的视角与节点名映射（引擎按视角分派独立 key 与产物名）
REVIEW_VIEWS: tuple[str, ...] = ("correctness", "security")
VIEW_LABELS: Mapping[str, str] = {"correctness": "正确性", "security": "安全性"}
REVIEW_NODE_FOR_VIEW: Mapping[str, str] = {
    "correctness": "review_correctness",
    "security": "review_security",
}
#: 汇合节点名（引擎在缺省装配里补上本层的纯函数实现）
MERGE_NODE = "merge_reviews"

#: 节点函数类型：``(state[, retry_feedback=...]) -> Mapping[str, Any]``（Design §3.1）
NodeFn = Callable[..., Mapping[str, Any]]


def node_names(*, parallel_review: bool = False) -> tuple[str, ...]:
    """图形态的节点名清单（覆盖表校验与缺省装配共用）。"""
    return PARALLEL_NODE_NAMES if parallel_review else SINGLE_NODE_NAMES


def view_of_review_node(name: str) -> str | None:
    """审查节点名 → 视角；非审查节点返回 ``None``。"""
    for view, node in REVIEW_NODE_FOR_VIEW.items():
        if node == name:
            return view
    return None


def findings_key_for_view(view: str) -> str:
    """某视角的 findings 状态键（独立 key，避免同超级步多写者冲突）。"""
    return f"findings_{view}"


def evidence_key_for_view(view: str) -> str:
    """某视角的执行证据状态键（节点交回、包装层消费）。"""
    return f"review_evidence_{view}"


def resolve_nodes(
    nodes: Mapping[str, NodeFn] | None = None,
    *,
    defaults: Mapping[str, NodeFn] | None = None,
    parallel_review: bool = False,
) -> dict[str, NodeFn]:
    """合并覆盖表与缺省装配：逐键替换、未给出的键用缺省（Design §12.2 第 3 步）。

    Args:
        nodes: 节点覆盖表（键为当前形态的节点名）；逐键替换缺省节点。
        defaults: 缺省装配（生产路径 = agents 的真实 Agent 节点 + 本层的汇合节点；测试 =
            脚本化假节点）。省略时要求 ``nodes`` 覆盖全部节点名——宁可显式报错，也不要
            静默走到一个没接模型的实现上。
        parallel_review: 是否按并行双视角形态校验节点集合。

    Raises:
        ValueError: 覆盖表含未知节点名，或缺省装配与覆盖表未能补齐全部节点。
    """
    names = node_names(parallel_review=parallel_review)
    base: dict[str, NodeFn] = dict(defaults) if defaults else {}
    if nodes:
        unknown = sorted(set(nodes) - set(names))
        if unknown:
            raise ValueError(f"未知节点名：{unknown}（可用：{list(names)}）")
        base.update(nodes)
    missing = [name for name in names if name not in base]
    if missing:
        raise ValueError(
            f"缺少节点装配：{missing}（可用：{list(names)}）——请给出完整覆盖表或提供缺省装配。"
        )
    return base


def build_graph(
    *,
    nodes: Mapping[str, NodeFn] | None = None,
    defaults: Mapping[str, NodeFn] | None = None,
    parallel_review: bool = False,
) -> CompiledStateGraph:
    """组装并编译工作流图（Design §4.1 / §12.2）。

    Args:
        nodes: 节点覆盖表；逐键替换缺省装配（测试与真实装配共用该接缝）。
        defaults: 缺省装配（见 :func:`resolve_nodes`）。
        parallel_review: 并行双视角审查形态（缺省单视角）。

    Returns:
        编译后的 StateGraph；调用方负责 ``stream`` 驱动与 ``recursion_limit``。
    """
    resolved = resolve_nodes(nodes, defaults=defaults, parallel_review=parallel_review)
    names = node_names(parallel_review=parallel_review)
    builder = StateGraph(RunState)
    for name in names:
        builder.add_node(name, resolved[name])
    if parallel_review:
        builder.add_conditional_edges(
            START,
            route_entry_parallel,
            ["encode", *REVIEW_NODE_FOR_VIEW.values()],
        )
        builder.add_edge("encode", "review_correctness")
        builder.add_edge("encode", "review_security")
        builder.add_edge("review_correctness", MERGE_NODE)
        builder.add_edge("review_security", MERGE_NODE)
        builder.add_conditional_edges(MERGE_NODE, route_after_review, {"fix": "fix", END: END})
        builder.add_edge("fix", "review_correctness")
        builder.add_edge("fix", "review_security")
    else:
        builder.add_conditional_edges(START, route_entry, {"encode": "encode", "review": "review"})
        builder.add_edge("encode", "review")
        builder.add_conditional_edges("review", route_after_review, {"fix": "fix", END: END})
        builder.add_edge("fix", "review")
    return builder.compile()


def route_entry(state: Mapping[str, Any]) -> str:
    """入口路由（只读纯函数）：需求文本从编码起步；代码文件直接进入审查（Design §4.4）。"""
    descriptor = state.get("input") or {}
    return "encode" if descriptor.get("kind") == state_mod.INPUT_REQUIREMENT else "review"


def route_entry_parallel(state: Mapping[str, Any]) -> list[str]:
    """并行形态的入口路由：需求文本先编码，代码文件直接扇出到两个视角。

    LangGraph 的条件边路由函数返回**目标清单**即触发同一超级步的并行执行（本形态的唯一
    扇出点之一，另一处是修复后的复审）。
    """
    descriptor = state.get("input") or {}
    if descriptor.get("kind") == state_mod.INPUT_REQUIREMENT:
        return ["encode"]
    return list(REVIEW_NODE_FOR_VIEW.values())


def route_after_review(state: Mapping[str, Any]) -> str:
    """复审条件边（只读纯函数）：干净 → 收尾；达修复轮次上限 → 收尾（残留）；否则进入修复。

    干净判定与轮次口径见 Design §4.4；并行形态下判据作用于汇合后的 ``findings``（与单视角
    同一口径）。``rounds-mode=exact``（跑满 n 轮）随 M4 的会话轮次语义落地，届时本函数按
    ``rounds_mode`` 分支。
    """
    if state.get("clean"):
        return END
    if int(state.get("fix_rounds", 0)) >= int(state.get("max_iterations", 0)):
        return END
    return "fix"


def merge_review_node(state: Mapping[str, Any]) -> Mapping[str, Any]:
    """汇合节点（本层纯函数，无模型、无 IO）：合并两个视角的 findings。

    多写者规则见 Design §12.2：按轮替换语义的字段（``findings``）由汇合节点**显式覆盖、
    无条件写**——两个分支写的是各自的独立 key，此处统一重编号 id（``f-1``…）并把来源
    视角并进每条 finding 的 ``views``（可审计：这条结论是谁提的）。

    **装配约定**：本节点只交回 ``findings``；``clean`` 判定与 ``review_round`` 推进由
    engine 的包装层按权威值写回（节点不掌握轮次口径）。条件是挂在汇合节点上的——直接用
    "裸节点"装配并行图（绕过 :meth:`WorkflowEngine._produce_merge`）会让 ``clean`` 读到
    上一轮的旧值，条件路由随之失真；生产与测试装配都走 engine/包装层，此处显式说明。
    """
    groups = [
        (VIEW_LABELS.get(view, view), state.get(findings_key_for_view(view)) or [])
        for view in REVIEW_VIEWS
    ]
    merged = contracts.merge_findings(
        [(view, [item for item in items if isinstance(item, Mapping)]) for view, items in groups]
    )
    return {"findings": merged}


def recursion_limit(*, max_iterations: int, parallel_review: bool = False) -> int:
    """崩溃护栏上限（Design §4.4）：按图形态复算，护栏必须留足正常收尾的余量。

    串行形态每轮 = 审查 + 修复两个超级步（``2n + 8``）；并行形态每轮 = 两视角（同一超级步）
    + 汇合 + 修复三个超级步（``3n + 8``）。数值是护栏而非预算：偏大只是更晚截停，偏小会把
    正常运行掐断。
    """
    per_round = 3 if parallel_review else 2
    return per_round * int(max_iterations) + 8


def mermaid_flowchart(*, parallel_review: bool = False) -> str:
    """规范化 mermaid 流程图（``crw graph`` 的导出内容，Design §8.1）。

    ``__start__`` / END 映射为可读的"开始 / 结束"标签，条件边给出中文分支标签；节点 / 边
    集合与 :func:`build_graph` 同源，由快照测试锁定。
    """
    if parallel_review:
        return _PARALLEL_MERMAID
    return _SINGLE_MERMAID


_SINGLE_MERMAID = """flowchart TD
    start(["开始"])
    encode["编码 Agent：生成初始代码 code_v1.py"]
    review["审查 Agent：读文件 / 运行核实 → findings"]
    fix["修复 Agent：按 findings 逐条处置 → code_vN+1.py"]
    finish(["收尾：report.md（干净通过退出码 0 / 达限残留退出码 1）"])

    start -- "需求文本" --> encode
    start -- "代码文件（-f）/ 续跑（resume）" --> review
    encode --> review
    review -- "干净通过 或 已达修复轮次上限" --> finish
    review -- "未干净，且未达修复轮次上限" --> fix
    fix --> review
"""

_PARALLEL_MERMAID = """flowchart TD
    start(["开始"])
    encode["编码 Agent：生成初始代码 code_v1.py"]
    review_correctness["审查 Agent·正确性视角：并行分支，写 findings_correctness"]
    review_security["审查 Agent·安全性视角：并行分支，写 findings_security"]
    merge_reviews["汇合：按 (location, issue) 去重取较严重者、统一重编号 id、审查轮次 +1"]
    fix["修复 Agent：按合并 findings 逐条处置 → code_vN+1.py"]
    finish(["收尾：report.md（干净通过退出码 0 / 达限残留退出码 1）"])

    start -- "需求文本" --> encode
    start -- "代码文件（-f）/ 续跑（resume）" --> review_correctness
    start -- "代码文件（-f）/ 续跑（resume）" --> review_security
    encode -- "扇出：同一超级步并行" --> review_correctness
    encode -- "扇出：同一超级步并行" --> review_security
    review_correctness --> merge_reviews
    review_security --> merge_reviews
    merge_reviews -- "干净通过 或 已达修复轮次上限" --> finish
    merge_reviews -- "未干净，且未达修复轮次上限" --> fix
    fix -- "扇出：同一超级步并行" --> review_correctness
    fix -- "扇出：同一超级步并行" --> review_security
"""
