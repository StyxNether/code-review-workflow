"""Agent 层：编码 / 审查 / 修复三个节点的装配与运行时。

装配分两步，便于引擎把过程事件与正文增量接进来（Design §3.1、§6.2 #5）::

    runtime = build_runtime(resolved=..., emit=engine_emit, on_delta=engine_delta)
    nodes = build_nodes(runtime, roots=(workspace,), no_exec=False)

每个节点的工具集按 Design §4.2 的表固定：编码无工具（纯生成）；审查可列目录 / 读文件 /
运行核实；修复只用读文件与运行核实（无需再列目录）。``--no-exec`` 在工具注册表组装层
生效（不注册 ``run_python``），不做提示词层假设。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from .. import config
from ..tools import (
    ALL_TOOL_NAMES,
    EXEC_TOOL_NAME,
    LIST_DIR_TOOL,
    READ_FILE_TOOL,
    ToolRegistry,
    build_registry,
)
from .encoder import build_encode_node
from .fixer import build_fix_node
from .reviewer import EVIDENCE_KEY, build_review_node, review_once
from .runtime import (
    MAX_TOOL_ROUNDS,
    AgentRuntime,
    DeltaSink,
    EventSink,
    NodeFn,
    TaskResult,
    history_from_state,
)

#: 各节点的工具集（Design §4.2 的"工具"列）
NODE_TOOL_NAMES: Mapping[str, tuple[str, ...]] = {
    "encode": (),
    "review": (LIST_DIR_TOOL, READ_FILE_TOOL, EXEC_TOOL_NAME),
    "review_correctness": (LIST_DIR_TOOL, READ_FILE_TOOL, EXEC_TOOL_NAME),
    "review_security": (LIST_DIR_TOOL, READ_FILE_TOOL, EXEC_TOOL_NAME),
    "fix": (READ_FILE_TOOL, EXEC_TOOL_NAME),
}

#: 并行双视角审查的节点名 → 视角（与 workflow/graph.py 的节点名映射由测试锁定一致性）
PARALLEL_REVIEW_NODES: Mapping[str, str] = {
    "review_correctness": "correctness",
    "review_security": "security",
}


def build_runtime(
    *,
    resolved: config.ResolvedModel | None = None,
    model: object | None = None,
    emit: EventSink | None = None,
    on_delta: DeltaSink | None = None,
    max_tool_rounds: int | None = None,
    max_context_tokens: int | None = None,
) -> AgentRuntime:
    """构建单任务运行时（模型装配 + 事件 / 增量回调）。

    Args:
        resolved: 配置层解析出的连接参数（生产路径）。
        model: 注入的聊天模型（仅测试用假模型），优先于 ``resolved``。
        emit: 过程事件回调（工具调用等）→ 引擎落 log.jsonl 并转渲染。
        on_delta: 正文增量回调 → 渲染层（``--no-stream`` 由渲染层缓存整段输出）。
        max_tool_rounds: 工具轮次上限；缺省取运行时默认值。
        max_context_tokens: 上下文预算；缺省取 ``CRW_MAX_CONTEXT_TOKENS``。
    """
    return AgentRuntime(
        model=model,  # type: ignore[arg-type]
        resolved=resolved,
        emit=emit,
        on_delta=on_delta,
        max_tool_rounds=max_tool_rounds if max_tool_rounds is not None else MAX_TOOL_ROUNDS,
        max_context_tokens=max_context_tokens,
    )


def build_nodes(
    runtime: AgentRuntime,
    *,
    roots: Sequence[Path],
    no_exec: bool = False,
    parallel_review: bool = False,
) -> dict[str, NodeFn]:
    """按节点装配（提示词 + 运行时 + 工具），返回 ``节点名 → 节点函数`` 的完整装配表。

    Args:
        runtime: 单任务运行时（由引擎装配，带事件与增量回调）。
        roots: 上下文根（工作流运行 = 运行工作区）。
        no_exec: 关闭执行（工具注册表不注册 ``run_python``）。
        parallel_review: 并行双视角形态：两个审查节点共用同一 runtime 与工具集，只有
            系统提示词与任务说明按视角分派；汇合节点由 workflow 层补齐（本层不 import 上层）。

    Returns:
        该形态下除汇合节点以外的全部节点；既有的单视角键（``review``）在并行形态下不出现。
    """
    names = ["encode", "review", "fix"]
    if parallel_review:
        names = ["encode", *PARALLEL_REVIEW_NODES, "fix"]
    registries = {
        name: build_registry(roots, names=NODE_TOOL_NAMES[name], allow_exec=not no_exec)
        for name in names
    }
    nodes: dict[str, NodeFn] = {
        "encode": build_encode_node(runtime, registries["encode"]),
        "fix": build_fix_node(runtime, registries["fix"]),
    }
    if parallel_review:
        for name, view in PARALLEL_REVIEW_NODES.items():
            nodes[name] = build_review_node(runtime, registries[name], name=name, view=view)
    else:
        nodes["review"] = build_review_node(runtime, registries["review"])
    return nodes


def build_review_runtime(
    *,
    resolved: config.ResolvedModel | None = None,
    model: object | None = None,
    roots: Sequence[Path],
    no_exec: bool = False,
    emit: EventSink | None = None,
    on_delta: DeltaSink | None = None,
) -> tuple[AgentRuntime, ToolRegistry]:
    """``crw review`` 的装配：独立审查运行时 + 上下文根为审查目标目录的工具集。"""
    runtime = build_runtime(resolved=resolved, model=model, emit=emit, on_delta=on_delta)
    registry = build_registry(roots, names=NODE_TOOL_NAMES["review"], allow_exec=not no_exec)
    return runtime, registry


#: 审查节点交回引擎的执行证据键（引擎侧经此常量引用，避免两处字面量漂移）
REVIEW_EVIDENCE_KEY = EVIDENCE_KEY

__all__ = [
    "ALL_TOOL_NAMES",
    "NODE_TOOL_NAMES",
    "PARALLEL_REVIEW_NODES",
    "REVIEW_EVIDENCE_KEY",
    "AgentRuntime",
    "DeltaSink",
    "EventSink",
    "NodeFn",
    "TaskResult",
    "build_nodes",
    "build_review_runtime",
    "build_runtime",
    "history_from_state",
    "review_once",
]
