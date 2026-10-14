"""修复 Agent 节点：按 findings 逐条修复，产出完整版本（全部文件）与 fixes 清单。

输出契约（Design §4.2.2 / §5.2）：版本内每个文件一个 ```python file=<相对路径> 围栏
（未改动的文件也原样给出）+ 恰好一个 ```json 围栏（fixes 清单，必须覆盖上一轮 findings
的 id 全集；可附 ``deleted`` 删除申请）。抽取、校验、落盘、删除确认与"整版本逐字节
相同"判定都在 workflow 包装层完成，节点只交回原始文本。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..tools import EXEC_TOOL_NAME, ToolRegistry
from ..workflow import state as state_mod
from . import prompts
from .runtime import AgentRuntime, history_from_state

#: 节点函数类型（与 workflow 层同构；agents 不向上 import，故此处自带别名）
NodeFn = Callable[..., Mapping[str, Any]]


def _as_int(value: Any) -> int:
    """状态里的整数字段（缺失 / 坏值按 0 处理）：只用于提示词里的轮次描述。"""
    if isinstance(value, bool) or not isinstance(value, int):
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0
    return value


def build_fix_node(runtime: AgentRuntime, registry: ToolRegistry) -> NodeFn:
    """构造修复节点：``node(state[, retry_feedback]) -> {"fix_text": …}``。"""

    def fix_node(state: Mapping[str, Any], retry_feedback: str | None = None) -> Mapping[str, Any]:
        version = state_mod.code_version(state)
        findings = [item for item in (state.get("findings") or []) if isinstance(item, Mapping)]
        files = state.get("version_files")
        result = runtime.run(
            node="fix",
            system_prompt=prompts.FIXER_SYSTEM_PROMPT,
            user_prompt=prompts.fix_task_prompt(
                version_label=state_mod.code_version_dir_name(version),
                files=[str(item) for item in files] if isinstance(files, list) else [],
                version=version,
                findings=findings,
                allow_exec=EXEC_TOOL_NAME in registry.names(),
                retry_feedback=retry_feedback,
                polish=state.get("rounds_mode") == state_mod.ROUNDS_EXACT,
                round_number=state_mod.round_number_for_fix(state),
                rounds_total=_as_int(state.get("max_iterations")),
            ),
            registry=registry,
            history=history_from_state(state, "fixer"),
        )
        return {"fix_text": result.text}

    return fix_node
