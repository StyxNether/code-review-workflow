"""编码 Agent 节点：按需求生成初始代码（单文件，无工具，Design §4.2）。

节点自身不落盘、不打印、不写日志：它只把模型正文经暂存键 ``code_text`` 交回 workflow
层的包装（抽取、落盘、契约重试都在包装层完成，Design §3.1）。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..tools import ToolRegistry
from ..workflow import state as state_mod
from . import prompts
from .runtime import AgentRuntime

#: 节点函数类型（与 workflow 层同构；agents 不向上 import，故此处自带别名）
NodeFn = Callable[..., Mapping[str, Any]]


def build_encode_node(runtime: AgentRuntime, registry: ToolRegistry) -> NodeFn:
    """构造编码节点：``node(state[, retry_feedback]) -> {"code_text": …}``。

    编码是纯生成任务：输出的代码围栏由引擎抽取落盘为 ``code_v1.py``。
    """

    def encode_node(
        state: Mapping[str, Any], retry_feedback: str | None = None
    ) -> Mapping[str, Any]:
        requirement = str((state.get("input") or {}).get("requirement") or "")
        result = runtime.run(
            node="encode",
            system_prompt=prompts.ENCODER_SYSTEM_PROMPT,
            user_prompt=prompts.encode_task_prompt(
                requirement=requirement,
                file_name=state_mod.code_file_name(1),
                retry_feedback=retry_feedback,
            ),
            registry=registry,
        )
        return {"code_text": result.text}

    return encode_node
