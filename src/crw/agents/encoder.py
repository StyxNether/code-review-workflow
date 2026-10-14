"""编码 Agent 节点：按需求生成初始代码版本（1..n 个文件，无工具，Design §4.2）。

节点自身不落盘、不打印、不写日志：它只把模型正文经暂存键 ``code_text`` 交回 workflow
层的包装（抽取、落盘、契约重试都在包装层完成，Design §3.1）。

会话轮次（Design §4.7）：状态的瞬态键 ``prior_files``（本会话当前版本的既有实现）会
内联进任务提示词，"按新需求修改"因此有真实依据；``agent_memory``（该 Agent 自己的
历史）作为历史消息进入本次请求。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .. import rounds as rounds_mod
from ..tools import ToolRegistry
from . import prompts
from .runtime import AgentRuntime, history_from_state

#: 节点函数类型（与 workflow 层同构；agents 不向上 import，故此处自带别名）
NodeFn = Callable[..., Mapping[str, Any]]


def build_encode_node(runtime: AgentRuntime, registry: ToolRegistry) -> NodeFn:
    """构造编码节点：``node(state[, retry_feedback]) -> {"code_text": …}``。

    编码是纯生成任务：产出的 ``python file=…`` 围栏由引擎抽取并落盘为起始版本目录。
    """

    def encode_node(
        state: Mapping[str, Any], retry_feedback: str | None = None
    ) -> Mapping[str, Any]:
        descriptor = state.get("input") if isinstance(state.get("input"), Mapping) else {}
        requirement = str(descriptor.get("requirement") or "")
        prior = state.get("prior_files")
        result = runtime.run(
            node="encode",
            system_prompt=prompts.ENCODER_SYSTEM_PROMPT,
            user_prompt=prompts.encode_task_prompt(
                requirement=requirement,
                version_label=rounds_mod.code_version_dir_name(rounds_mod.code_version(state)),
                retry_feedback=retry_feedback,
                prior_files=prior if isinstance(prior, Mapping) else None,
                prior_source=str(descriptor.get("source_path") or ""),
            ),
            registry=registry,
            history=history_from_state(state, "encoder"),
        )
        return {"code_text": result.text}

    return encode_node
