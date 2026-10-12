"""修复 Agent 节点：按 findings 逐条修复或说明未采纳，产出修复后代码与 fixes 清单。

输出契约（Design §4.2.1 / §5.2）：恰好一个 ```python 围栏（修复后的**完整文件**）+
恰好一个 ```json 围栏（fixes 清单，必须覆盖上一轮 findings 的 id 全集）。抽取、校验、
落盘与"逐字节相同"判定都在 workflow 包装层完成，节点只交回原始文本。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..tools import EXEC_TOOL_NAME, ToolRegistry
from ..workflow import state as state_mod
from . import prompts
from .runtime import AgentRuntime

#: 节点函数类型（与 workflow 层同构；agents 不向上 import，故此处自带别名）
NodeFn = Callable[..., Mapping[str, Any]]


def build_fix_node(runtime: AgentRuntime, registry: ToolRegistry) -> NodeFn:
    """构造修复节点：``node(state[, retry_feedback]) -> {"fix_text": …}``。"""

    def fix_node(state: Mapping[str, Any], retry_feedback: str | None = None) -> Mapping[str, Any]:
        version = state_mod.code_version(state)
        findings = [item for item in (state.get("findings") or []) if isinstance(item, Mapping)]
        result = runtime.run(
            node="fix",
            system_prompt=prompts.FIXER_SYSTEM_PROMPT,
            user_prompt=prompts.fix_task_prompt(
                file_name=state_mod.code_file_name(version),
                version=version,
                findings=findings,
                allow_exec=EXEC_TOOL_NAME in registry.names(),
                retry_feedback=retry_feedback,
            ),
            registry=registry,
        )
        return {"fix_text": result.text}

    return fix_node
