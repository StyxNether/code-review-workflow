"""工具注册表：按上下文根装配文件与执行工具，统一返回结构化结果。

注册表屏蔽 handler 细节（数据封装）：对运行时只暴露 schema 列表与
``dispatch(name, arguments_json) -> str``；工具异常一律结构化为 ``错误：`` 文本回传
模型，不向上抛出——工具失败不应让整轮运行以 traceback 结束。

``--no-exec`` 的实施闸口唯一在本层：**不注册** ``run_python``，不做
提示词层假设——模型看不到执行工具，也就无从调用。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from . import exec_py, fs
from .fs import ERROR_PREFIX

Handler = Callable[..., str]

#: 工具名常量（提示词、节点装配与测试按名引用）
LIST_DIR_TOOL = "list_dir"
READ_FILE_TOOL = "read_file"
WRITE_FILE_TOOL = "write_file"
EXEC_TOOL_NAME = "run_python"
#: 全部工具名（注册表默认装配集）
ALL_TOOL_NAMES: tuple[str, ...] = (LIST_DIR_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL, EXEC_TOOL_NAME)

_MAX_ARGS_PREVIEW = 160


@dataclass(frozen=True)
class Tool:
    """单个工具的定义：LLM 可见的 schema 三元组 + 本地执行入口。"""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler


class ToolRegistry:
    """工具集合：schema 暴露、参数解析、分发与异常兜底。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self, name: str, description: str, parameters: dict[str, Any], handler: Handler
    ) -> None:
        """注册一个工具；同名重复注册以后者为准（便于测试替换）。"""
        self._tools[name] = Tool(
            name=name, description=description, parameters=parameters, handler=handler
        )

    def names(self) -> list[str]:
        """已注册工具名（保持注册顺序）。"""
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """OpenAI function calling 的 tools 参数格式。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def definitions(self) -> list[Tool]:
        """全部工具定义（供 agents 层桥接为框架原生工具）；注册顺序保持稳定。"""
        return list(self._tools.values())

    def dispatch(self, name: str, arguments_json: str) -> str:
        """分发工具调用并返回字符串结果；任何失败都以 ``错误：…`` 文本回传。

        未注册工具名、非法 JSON、参数不符 schema（如缺必填参数）都不抛出，而是作为错误
        文本交给模型自行调整策略，不触发 API 重试。
        """
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self._tools) or "（无）"
            return f"{ERROR_PREFIX}未注册的工具 '{name}'，可用工具：{available}"
        if not isinstance(arguments_json, str):
            # 契约是"任何失败都以 错误：文本回传、不向上抛出"：非字符串入参也不例外
            return (
                f"{ERROR_PREFIX}工具 '{name}' 的参数必须是 JSON 字符串，收到："
                f"{type(arguments_json).__name__}"
            )
        try:
            arguments = json.loads(arguments_json) if arguments_json.strip() else {}
        except json.JSONDecodeError as exc:
            return f"{ERROR_PREFIX}工具 '{name}' 的参数不是合法 JSON：{exc}"
        if not isinstance(arguments, dict):
            return (
                f"{ERROR_PREFIX}工具 '{name}' 的参数必须是 JSON 对象，收到："
                f"{type(arguments).__name__}"
            )
        try:
            return str(tool.handler(**arguments))
        except Exception as exc:  # noqa: BLE001 — 设计要求：工具异常必须全部在此兜底回传模型
            return f"{ERROR_PREFIX}工具 '{name}' 执行失败：{exc}"


def build_registry(
    roots: Sequence[Path],
    *,
    names: Sequence[str] | None = None,
    allow_exec: bool = True,
) -> ToolRegistry:
    """按上下文根装配工具集。

    Args:
        roots: 上下文根（运行工作区；``crw review`` 为审查目标所在目录的并集）。
            所有文件 / 执行工具的相对路径都相对它解析，越界一律结构化拒绝。
        names: 要注册的工具名（按各节点的工具集裁剪）；缺省注册全部。
        allow_exec: 是否注册 ``run_python``；``--no-exec`` 传 ``False``（不注册即无法调用）。
    """
    registry = ToolRegistry()
    if names is None or LIST_DIR_TOOL in names:
        registry.register(
            name=LIST_DIR_TOOL,
            description=(
                "列出上下文根下的两层目录树（本层与每个子目录的一层），忽略 .venv、"
                "__pycache__、.git、.pytest_cache、.ruff_cache 及点开头的隐藏文件与目录。"
                "相对路径相对上下文根解析；绝对路径、~ 与越界路径（.. 回溯出根）会被拒绝。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对上下文根的目录路径，默认根目录"}
                },
                "required": [],
            },
            handler=partial(fs.list_dir, roots),
        )
    if names is None or READ_FILE_TOOL in names:
        registry.register(
            name=READ_FILE_TOOL,
            description=(
                "带行号读取上下文根内的文本文件（1-based 闭区间 start/end），单次最多 400 行，"
                "超出请用 start/end 分段读取。相对路径相对上下文根解析；绝对路径、~、"
                "隐藏文件与越界路径会被拒绝；二进制文件返回结构化错误。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对上下文根的文件路径"},
                    "start": {"type": "integer", "description": "起始行（1-based，含），默认 1"},
                    "end": {
                        "type": "integer",
                        "description": "结束行（1-based，含），默认到文件尾",
                    },
                },
                "required": ["path"],
            },
            handler=partial(fs.read_file, roots),
        )
    if names is None or WRITE_FILE_TOOL in names:
        registry.register(
            name=WRITE_FILE_TOOL,
            description=(
                "把**整个文件的完整内容**写入上下文根内的指定文件（已存在则整份覆盖，"
                "不存在则新建，父目录自动创建）。只写你实际改动的文件；未改动的文件不要重写。"
                "单次写入上限 10 MB。相对路径相对上下文根解析；绝对路径、~、隐藏文件、"
                "越界路径与数据目录内的路径一律被拒绝；内容是纯文本，不要放 Markdown 围栏。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对上下文根的文件路径"},
                    "content": {"type": "string", "description": "文件的完整文本内容"},
                },
                "required": ["path", "content"],
            },
            handler=partial(fs.write_file, roots),
        )
    if allow_exec and (names is None or EXEC_TOOL_NAME in names):
        registry.register(
            name=EXEC_TOOL_NAME,
            description=(
                "运行 Python 代码并返回退出码与 stdout/stderr（各截断至 2000 字符）。"
                "code（内联代码）与 path（相对上下文根路径的 .py 文件）二选一。"
                "**本机直接运行，非沙箱**：代码拥有当前用户全部权限，只用于验证本轮"
                "需要核实的可信代码片段；限时执行，默认 10 秒（可用 timeout 参数调低，"
                "上限 60 秒），超时将强制终止整棵进程树。子进程工作目录 = 上下文根。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "要执行的 Python 代码（与 path 二选一）",
                    },
                    "path": {
                        "type": "string",
                        "description": "要执行的 .py 文件路径（相对上下文根；与 code 二选一）",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "超时秒数，默认 10（CRW_EXEC_TIMEOUT），上限 60",
                    },
                },
                "required": [],
            },
            handler=partial(exec_py.run_python, roots),
        )
    return registry


def arguments_preview(arguments_json: str, *, limit: int = _MAX_ARGS_PREVIEW) -> str:
    """工具参数的展示预览（事件 detail 与终端打点共用，超长截断）。"""
    text = " ".join(str(arguments_json or "").split())
    return text if len(text) <= limit else f"{text[: limit - 1]}…"
