"""修复 Agent 节点：按 findings 逐条修复，用 ``write_file`` **只写改动的文件**。

写入根 = 本轮产出的版本目录（引擎已把上一版本整目录复制过来）：路径因此
天然限定在该目录内，未改动的文件不必"整份重贴"，也不会因重写而顺手改坏。manifest、
版本完整性校验与"本轮未产生改动"的逐字节比较都由引擎按**目录实际内容**重算——节点只
交回 fixes 清单文本（恰好一个 ```json 围栏，覆盖上一轮 findings 的全部 id，可附
``deleted`` 删除申请）。抽取、校验、落盘、删除确认都在 workflow 包装层完成。

工具集**按轮装配**：上下文根只有在引擎复制出新版本目录之后才确定，所以根取自状态的
``current_version_dir``，而不是运行开始时固定的工作区。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .. import rounds as rounds_mod
from ..tools import EXEC_TOOL_NAME, READ_FILE_TOOL, WRITE_FILE_TOOL, build_registry
from . import prompts
from .runtime import AgentRuntime, history_from_state

#: 节点函数类型（与 workflow 层同构；agents 不向上 import，故此处自带别名）
NodeFn = Callable[..., Mapping[str, Any]]

#: 修复节点的工具集：读文件、只写改动文件、运行核实
FIX_TOOL_NAMES: tuple[str, ...] = (READ_FILE_TOOL, WRITE_FILE_TOOL, EXEC_TOOL_NAME)


def _as_int(value: Any) -> int:
    """状态里的整数字段（缺失 / 坏值按 0 处理）：只用于提示词里的轮次描述。"""
    if isinstance(value, bool) or not isinstance(value, int):
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0
    return value


def build_fix_node(runtime: AgentRuntime, *, no_exec: bool = False) -> NodeFn:
    """构造修复节点：``node(state[, retry_feedback]) -> {"fix_text": …}``。

    Args:
        runtime: 单任务运行时。
        no_exec: 关闭执行（``--no-exec``：注册表不注册 ``run_python``，只做静态修复）。

    目标目录缺失属装配缺陷（引擎应先复制上一版本）：显式报错按节点失败收尾，
    不让节点在一个不存在的根上"假装修复"。
    """

    def fix_node(state: Mapping[str, Any], retry_feedback: str | None = None) -> Mapping[str, Any]:
        target = str(state.get("current_version_dir") or "")
        if not target:
            raise ValueError("修复节点缺少目标版本目录（引擎应已把上一版本复制为新版本）")
        registry = build_registry([Path(target)], names=FIX_TOOL_NAMES, allow_exec=not no_exec)
        version = rounds_mod.code_version(state)
        findings = [item for item in (state.get("findings") or []) if isinstance(item, Mapping)]
        files = state.get("version_files")
        result = runtime.run(
            node="fix",
            system_prompt=prompts.FIXER_SYSTEM_PROMPT,
            user_prompt=prompts.fix_task_prompt(
                version_label=rounds_mod.code_version_dir_name(version),
                out_label=Path(target).name,
                files=[str(item) for item in files] if isinstance(files, list) else [],
                findings=findings,
                allow_exec=EXEC_TOOL_NAME in registry.names(),
                retry_feedback=retry_feedback,
                polish=state.get("rounds_mode") == rounds_mod.ROUNDS_EXACT,
                round_number=rounds_mod.round_number_for_fix(state),
                rounds_total=_as_int(state.get("max_iterations")),
            ),
            registry=registry,
            history=history_from_state(state, "fixer"),
        )
        return {"fix_text": result.text}

    return fix_node
