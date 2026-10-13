"""审查 Agent 节点：读取代码文件、运行核实，产出分级 findings（Design §4.2、§5.2）。

**执行有效性门禁的接线点**：审查契约要求"本轮至少有一次成功工具事件（或内联送达
记录）"，否则空 ``findings`` 就成了"没审就通过"的逃逸。工作流内代码不在提示词里
（模型必须自己 read_file），故本节点把运行时统计到的**成功工具事件数**经
``review_evidence`` 交回引擎，由引擎传给 :func:`crw.workflow.contracts.parse_review_output`
的 ``has_tool_evidence``。

并行双视角审查（Design §12.2）复用本节点：同一 runtime、同一工具集，只有系统提示词
与任务说明按视角分派（``name`` 取分支节点名，工具事件与增量按分支归轮，便于终端与
log 区分是哪位审查者在动作）。

``crw review``（代码内联送达）不走本节点，由 :func:`review_once` 直接审查。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..tools import EXEC_TOOL_NAME, ToolRegistry
from ..workflow import state as state_mod
from . import prompts
from .runtime import REVIEW_EVIDENCE_KEY, AgentRuntime

#: 节点函数类型（与 workflow 层同构；agents 不向上 import，故此处自带别名）
NodeFn = Callable[..., Mapping[str, Any]]

#: 交回引擎的执行证据键（引擎读它作为 ``has_tool_evidence``）
EVIDENCE_KEY = REVIEW_EVIDENCE_KEY


def build_review_node(
    runtime: AgentRuntime,
    registry: ToolRegistry,
    *,
    name: str = "review",
    view: str | None = None,
) -> NodeFn:
    """构造审查节点：``node(state[, retry_feedback]) -> {"review_text": …}``。

    Args:
        runtime: 单任务运行时。
        registry: 审查工具集（``--no-exec`` 时不含 ``run_python``）。
        name: 节点名（单视角 ``review``；并行双视角取分支节点名，用于事件与增量归类）。
        view: 视角标识（``None`` = 单视角审查）。

    重试（第二次调用）沿用本轮已积累的工具事件——"本轮"指这一次审查，重试属同一轮。
    """

    evidence = {"value": False}

    def review_node(
        state: Mapping[str, Any], retry_feedback: str | None = None
    ) -> Mapping[str, Any]:
        if retry_feedback is None:
            evidence["value"] = False  # 新一轮审查：重新判定执行证据
        version = state_mod.code_version(state)
        file_name = state_mod.code_file_name(version)
        result = runtime.run(
            node=name,
            system_prompt=prompts.reviewer_system_prompt(view),
            user_prompt=prompts.review_task_prompt(
                file_name=file_name,
                review_round=state_mod.round_number_for_review(state),
                code_version=version,
                allow_exec=EXEC_TOOL_NAME in registry.names(),
                retry_feedback=retry_feedback,
                view=view,
            ),
            registry=registry,
        )
        evidence["value"] = evidence["value"] or result.tool_events > 0
        return {"review_text": result.text, EVIDENCE_KEY: evidence["value"]}

    return review_node


def review_once(
    runtime: AgentRuntime,
    registry: ToolRegistry,
    *,
    display_name: str,
    content: str,
    as_json: bool,
    max_lines: int,
) -> str:
    """单次独立审查（``crw review``）：文件内容内联送达，返回模型正文。

    与工作流节点的两点差异（Design §6.4 / §8.1）：内容直接附在提示词里（用户显式传入的
    审查目标，无需工具读取）；``--json`` 走前作的 ``files[]`` 形态，由命令面校验与消费。
    """
    result = runtime.run(
        node="review",
        system_prompt=prompts.REVIEWER_SYSTEM_PROMPT,
        user_prompt=prompts.inline_review_prompt(
            display_name=display_name, content=content, as_json=as_json, max_lines=max_lines
        ),
        registry=registry,
    )
    return result.text
