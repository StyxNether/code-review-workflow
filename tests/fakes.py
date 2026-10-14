"""测试用替身：脚本化节点（不接模型）与脚本化聊天模型（不连网络）。

生产路径的缺省装配是 agents 层的真实 Agent 节点；这组假节点只用于测试注入——
节点级契约用例、引擎三路径回归等都靠它构造确定性的输入。用例经
``SCRIPTED_NODES`` 整表注入（``build_graph(nodes=...)`` / ``WorkflowEngine(nodes=...)``）。

``ScriptedChatModel`` 是给真实 Agent 运行时用的假模型：它按脚本回放消息，用于
验证"提示词 → 工具调用 → 契约产出"的完整链路，而无需任何真实网络调用。
"""

from __future__ import annotations

import ast
import itertools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

from crw.workflow import graph as graph_mod

#: 脚本化生成失败时的保底需求摘要（保证假编码节点恒产出可编译代码）
FALLBACK_REQUIREMENT = "脚本化占位实现"

#: 脚本化需求摘要的行长上限
_MAX_SUMMARY_CHARS = 80

#: 节点名与真实装配一致（``graph.NODE_NAMES``）
NODE_NAMES = graph_mod.NODE_NAMES


def SCRIPTED_NODES() -> dict[str, Any]:
    """整表假节点装配（每次调用返回新表，便于按用例逐键覆盖）。"""
    return {"encode": fake_encode, "review": fake_review, "fix": fake_fix}


def PARALLEL_SCRIPTED_NODES() -> dict[str, Any]:
    """并行双视角形态的整表假节点装配（含汇合节点，与生产装配同形）。

    安全性视角故意与正确性视角报**同一批位置但级别更低**：合并时取较严重者（正确性的
    判定胜出），另外多报一条安全性独有的问题——一条用例即可覆盖去重、取严重、并集三件事。
    """
    return {
        "encode": fake_encode,
        "review_correctness": fake_review_view("correctness"),
        "review_security": fake_review_view("security"),
        "merge_reviews": graph_mod.merge_review_node,
        "fix": fake_fix,
    }


def fake_review_view(view: str) -> Any:
    """假审查节点（并行视角）：脚本化检查结果按视角改写。

    未知视角显式报错：拼错时静默按"正确性"回放会让用例"能跑但错位"，
    与生产侧 ``prompts.reviewer_system_prompt`` 的显式失败口径一致。
    """
    if view not in graph_mod.REVIEW_VIEWS:
        raise ValueError(f"未知的审查视角：{view!r}（可选：{list(graph_mod.REVIEW_VIEWS)}）")

    def node(state: Mapping[str, Any]) -> Mapping[str, Any]:
        files = read_version_files(state)
        findings = [
            item for name, code in files.items() for item in inspect_code(code, file_name=name)
        ]
        if view == "security":
            findings = [
                {**item, "severity": "建议", "suggestion": "安全性视角：收紧输入校验。"}
                for item in findings
            ]
            first = next(iter(files), "main.py")
            findings.append(
                {
                    "severity": "严重",
                    "location": f"{first}:1",
                    "issue": "安全性视角独有：模块级执行外部命令。",
                    "suggestion": "改为受控调用。",
                    "uncertain": False,
                }
            )
        payload = {
            "files": sorted(files),
            "findings": findings,
            "summary": f"{view} 视角的脚本化结论。",
        }
        return {
            "review_text": (f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n")
        }

    return node


def fake_encode(state: Mapping[str, Any]) -> Mapping[str, Any]:
    """假编码节点：按需求文本确定性地生成单文件实现（多文件契约的"目录里一个文件"特例）。"""
    requirement = str((state.get("input") or {}).get("requirement") or "")
    code = scripted_module_for(requirement)
    return {"code_text": f"```python file=main.py\n{code}```\n"}


def fake_review(state: Mapping[str, Any]) -> Mapping[str, Any]:
    """假审查节点：对工作区当前版本做脚本化分级检查，输出审查契约（JSON 围栏）。"""
    files = read_version_files(state)
    findings = [item for name, code in files.items() for item in inspect_code(code, file_name=name)]
    payload = {
        "files": sorted(files),
        "findings": findings,
        "summary": summarize_findings(findings),
    }
    return {
        "review_text": f"脚本化审查结论如下。\n\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n"
    }


def fake_fix(state: Mapping[str, Any]) -> Mapping[str, Any]:
    """假修复节点：只做可脚本化且可复核的修复（语法缺陷），其余逐条标记 skipped。

    与真实修复节点同形态（Design §4.2.2 的写盘形态）：把**改动过的文件**写回本轮的版本
    目录（引擎已把上一版本复制进来），只交回 fixes 清单文本，未改动的文件不动。
    """
    files = read_version_files(state)
    findings = list(state.get("findings") or [])
    repaired_files, fixes = scripted_repair(files, findings)
    target = Path(str(state.get("current_version_dir") or ""))
    for name in sorted(repaired_files):
        if _normalized(repaired_files[name]) == _normalized(files.get(name, "")):
            continue
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(repaired_files[name], encoding="utf-8")
    return {"fix_text": fix_contract(fixes)}


def _normalized(code: str) -> str:
    """与引擎落盘同口径的归一化（首尾换行剥离 + 单个结尾换行）：比较"有没有真的改动"。"""
    return code.strip("\n") + "\n"


def read_version_files(state: Mapping[str, Any]) -> dict[str, str]:
    """读出状态的版本目录内容（``路径 → 内容``，跳过 manifest.json）。

    空目录值按"没有版本"处理（``Path("")`` 会解析成当前目录，必须显式挡住）。
    """
    raw = str(state.get("current_version_dir") or "")
    if not raw.strip():
        return {}
    directory = Path(raw)
    files: dict[str, str] = {}
    if not directory.is_dir():
        return files
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        if path.suffix == ".pyc" or "__pycache__" in path.relative_to(directory).parts:
            continue  # 与引擎读取同口径：字节码是执行副产物，不是版本内容
        files[path.relative_to(directory).as_posix()] = path.read_text(encoding="utf-8")
    return files


def scripted_module_for(requirement: str) -> str:
    """脚本化生成单文件实现：确定性输出，恒为可编译、带模块 docstring 的 Python。

    需求文本内联进 docstring 与输出语句；内联后不可编译（极端字符组合）时退回保底文案，
    保证假编码节点不会产出坏语法。
    """
    code = _module_template(_inline_text(requirement) or FALLBACK_REQUIREMENT)
    if _compiles(code):
        return code
    return _module_template(FALLBACK_REQUIREMENT)


def inspect_code(code: str, *, file_name: str) -> list[dict[str, Any]]:
    """脚本化分级检查（假审查节点的判定规则，确定性、可测）。

    规则：① 解析失败 → 【严重】语法错误；② 除数为字面量 0 或 ``len(...)`` → 【严重】
    除零风险；③ 缺模块 docstring → 【建议】。语法不通时不再继续检查。
    """
    findings: list[dict[str, Any]] = []
    try:
        tree = ast.parse(code, filename=file_name)
    except SyntaxError as exc:
        findings.append(
            {
                "severity": "严重",
                "location": f"{file_name}:{exc.lineno or 1}",
                "issue": f"语法错误：{exc.msg}，该文件无法解析。",
                "suggestion": "修正该行语法（例如补全语句末尾的冒号）后重新提交审查。",
                "uncertain": False,
            }
        )
        return findings
    if not ast.get_docstring(tree):
        findings.append(
            {
                "severity": "建议",
                "location": file_name,
                "issue": "模块缺少说明文档（模块级 docstring）。",
                "suggestion": "在文件开头补一句模块用途说明。",
                "uncertain": False,
            }
        )
    findings.extend(_division_risks(tree, file_name))
    return findings


def summarize_findings(findings: list[dict[str, Any]]) -> str:
    """本轮整体评价（审查契约的 summary，保证空 findings 时非空串）。"""
    if not findings:
        return "本轮未发现问题。"
    severe = sum(1 for item in findings if item.get("severity") == "严重")
    detail = "含【严重】问题，需修复后复审。" if severe else "无【严重】问题，可继续判定。"
    return f"发现 {len(findings)} 项问题：{detail}"


def scripted_repair(
    files: dict[str, str], findings: list[dict[str, Any]]
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """脚本化修复：逐条给出处置，语法缺陷尝试可复核的修补，其余标记 skipped。

    按 finding 的 location 定位文件；修补结果必须整文件可编译才被采用（否则视为无法
    安全处理）。返回（修复后的全部文件，fixes 清单）——未涉及文件原样保留。
    """
    repaired = dict(files)
    fixes: list[dict[str, Any]] = []
    for finding in findings:
        finding_id = str(finding.get("id") or "")
        location = str(finding.get("location") or "")
        name = location.rsplit(":", 1)[0] if ":" in location else location
        code = repaired.get(name)
        if code is not None and not _compiles(code):
            fixed = _repair_syntax(code)
            if fixed is not None:
                repaired[name] = fixed
                fixes.append(
                    {
                        "finding_id": finding_id,
                        "status": "fixed",
                        "note": "已修正语法缺陷并就地复核（整文件可编译）通过。",
                    }
                )
                continue
            fixes.append(
                {
                    "finding_id": finding_id,
                    "status": "skipped",
                    "note": "脚本化修复无法安全处理该语法缺陷，代码保持原样。",
                }
            )
            continue
        if finding.get("severity") == "严重":
            fixes.append(
                {
                    "finding_id": finding_id,
                    "status": "skipped",
                    "note": "该缺陷需结合业务语义处置，脚本化修复不改动（建议人工确认）。",
                }
            )
        else:
            fixes.append(
                {
                    "finding_id": finding_id,
                    "status": "skipped",
                    "note": "建议类问题不阻塞判定，脚本化修复不改动。",
                }
            )
    return repaired, fixes


def _module_template(summary: str) -> str:
    """假编码节点的输出模板：模块 docstring + 入口函数 + main 守卫（无除零、无语法问题）。"""
    message = json.dumps(f"已接收需求：{summary}", ensure_ascii=False)
    return (
        f'"""{summary}。\n\n'
        "脚本化占位实现：由内置假节点生成，未接入真实模型。\n"
        '"""\n'
        "\n"
        "\n"
        "def main() -> None:\n"
        '    """入口：打印需求接收说明。"""\n'
        f"    print({message})\n"
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )


def _inline_text(requirement: str) -> str:
    """需求文本 → 安全的内联摘要：折叠空白、去除三引号、限长。"""
    text = " ".join(requirement.split()).replace('"""', "'''")
    if len(text) > _MAX_SUMMARY_CHARS:
        text = f"{text[: _MAX_SUMMARY_CHARS - 1]}…"
    return text


def _compiles(code: str) -> bool:
    """整文件可解析判定；源码含不可编码字符（如 NUL）时 ast 抛 ``ValueError``，一并视为不可编译。"""
    try:
        ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    return True


def _division_risks(tree: ast.AST, file_name: str) -> list[dict[str, Any]]:
    """除零风险：除数为字面量 0，或为 ``len(...)``（空序列时为 0）。"""
    findings: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(
            node.op, (ast.Div, ast.FloorDiv, ast.Mod)
        ):
            continue
        divisor = node.right
        risky = False
        issue = ""
        if isinstance(divisor, ast.Constant) and divisor.value in (0, 0.0):
            risky = True
            issue = "除数是字面量 0，运行时将抛出 ZeroDivisionError。"
        elif isinstance(divisor, ast.Call) and getattr(divisor.func, "id", "") == "len":
            risky = True
            issue = "除数是 len(...)，序列为空时结果为 0，将抛出 ZeroDivisionError。"
        if risky:
            findings.append(
                {
                    "severity": "严重",
                    "location": f"{file_name}:{node.lineno}",
                    "issue": issue,
                    "suggestion": "运算前校验除数，为 0 时给出可读错误或返回默认值。",
                    "uncertain": False,
                }
            )
    return findings


def _repair_syntax(code: str) -> str | None:
    """语法缺陷的脚本化修补：按解析报错行补语句结尾冒号；整文件可编译才返回结果。"""
    try:
        ast.parse(code)
    except SyntaxError as exc:
        lines = code.splitlines()
        lineno = exc.lineno or 0
        if not 1 <= lineno <= len(lines):
            return None
        original = lines[lineno - 1]
        stripped = original.rstrip()
        if not stripped or stripped.endswith(":"):
            return None
        lines[lineno - 1] = f"{stripped}:"
        repaired = "\n".join(lines) + ("\n" if code.endswith("\n") else "")
        return repaired if _compiles(repaired) else None
    return None


# ---- 脚本化聊天模型（假模型驱动真实 Agent 运行时）----


class _ScriptExhausted(RuntimeError):
    """脚本用尽：用例给的响应数与实际请求数不符（多数情况下说明流程走向与预期不同）。"""


class ScriptedChatModel(BaseChatModel):
    """按脚本回放消息的假模型：记录收到的请求与每次 ``bind_tools`` 的工具集。

    脚本项为 ``AIMessage``（正常回放）或异常实例（到达时抛出，模拟请求失败 / 中断）；
    脚本用尽即抛错——静默重复最后一条会把"流程多跑了一轮"变成隐性通过。
    """

    responses: list[Any] = Field(default_factory=list)
    routes: dict[str, list[Any]] = Field(default_factory=dict)
    cursor: int = 0
    cursors: dict[str, int] = Field(default_factory=dict)
    requests: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[list[str]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def load(self, responses: Sequence[Any]) -> ScriptedChatModel:
        """装载新一轮脚本（同时重置游标与请求记录）：同一用例内可跑多次流程。"""
        self.responses = list(responses)
        self.routes = {}
        self.cursor = 0
        self.cursors = {}
        self.requests = []
        self.bound_tools = []
        return self

    def route(self, routes: Mapping[str, Sequence[Any]]) -> ScriptedChatModel:
        """装载"按提示词路由"的脚本：键出现在系统提示词里即选中该队列。

        并行分支会**同时**发起模型请求，按调用顺序回放会把响应错配给另一个分支（实测：
        安全性分支拿到了正确性分支的"无问题"结论）；按内容路由使用例与执行顺序无关。
        """
        self.routes = {key: list(items) for key, items in routes.items()}
        self.responses = []
        self.cursor = 0
        self.cursors = {}
        self.requests = []
        self.bound_tools = []
        return self

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        self.bound_tools.append([_tool_name(item) for item in tools])
        return self

    def bind(self, **kwargs: Any) -> ScriptedChatModel:
        # 框架在"请求无工具可绑"时的路径（护栏去工具后的模型调用）：记录为空绑定
        self.bound_tools.append([])
        return self

    def _next_response(self, messages: Sequence[BaseMessage]) -> Any:
        self.requests.append(list(messages))
        if self.routes:
            return self._next_routed(messages)
        if self.cursor >= len(self.responses):
            raise _ScriptExhausted(
                f"假模型脚本已用尽（第 {self.cursor + 1} 次请求）：请为本次流程补足响应。"
            )
        response = self.responses[self.cursor]
        self.cursor += 1
        if isinstance(response, BaseException):
            raise response
        return response

    def _next_routed(self, messages: Sequence[BaseMessage]) -> Any:
        """按提示词里的分支标记选队列（标记出现在该分支每次请求的拼接文本里）。"""
        joined = "\n".join(str(message.content) for message in messages)
        # 命中必须唯一：任务提示词里有"另一位审查者并行负责【X】"这类互指文案，多命中
        # 说明路由键选得不够特异——静默取一个会让另一分支的响应被顶替而不报错
        matched = sorted(key for key in self.routes if key in joined)
        if len(matched) > 1:
            raise _ScriptExhausted(
                f"分支脚本路由命中多个键 {matched}：请把路由键写得互斥（例如带上「你只负责」前缀）。"
            )
        if not matched:
            raise _ScriptExhausted(
                f"没有匹配的分支脚本（请求内容开头：{joined[:60]!r}）；"
                f"可用路由：{sorted(self.routes)}。"
            )
        key = matched[0]
        queue = self.routes[key]
        index = self.cursors.get(key, 0)
        if index >= len(queue):
            raise _ScriptExhausted(f"分支「{key}」的脚本已用尽（第 {index + 1} 次请求）。")
        self.cursors[key] = index + 1
        response = queue[index]
        if isinstance(response, BaseException):
            raise response
        return response

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        response = self._next_response(list(messages))
        return ChatResult(generations=[ChatGeneration(message=response)])

    def _stream(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> Any:
        """流式回放：工具调用按 tool_call_chunks 还原，文本按 3 段切片（含 usage 末块）。"""
        response = self._next_response(list(messages))
        if response.tool_calls:
            for index, call in enumerate(response.tool_calls):
                # index 必须逐调用各异：langchain 聚合 chunk 时按 index 归并，
                # 全部写 0 会把一条消息里的多个并行调用合并成一个（args 直接相连）
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            {
                                "name": call["name"],
                                "args": json.dumps(call["args"], ensure_ascii=False),
                                "id": call["id"],
                                "index": index,
                                "type": "tool_call_chunk",
                            }
                        ],
                    )
                )
        else:
            text = response.text
            step = max(1, len(text) // 3)
            pieces = [text[i : i + step] for i in range(0, len(text), step)] or [""]
            for index, piece in enumerate(pieces):
                extra: dict[str, Any] = {}
                if index == len(pieces) - 1 and response.response_metadata.get("finish_reason"):
                    extra["response_metadata"] = response.response_metadata
                yield ChatGenerationChunk(message=AIMessageChunk(content=piece, **extra))
        if response.usage_metadata:
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="", usage_metadata=response.usage_metadata)
            )


def _tool_name(item: Any) -> str:
    """绑定工具项的展示名（``StructuredTool`` / dict schema 两种形态）。"""
    return str(
        getattr(item, "name", None) or (item.get("name") if isinstance(item, dict) else item)
    )


def call_usage(
    *,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int | None = None,
    reasoning_tokens: int | None = None,
) -> dict[str, Any]:
    """LangChain ``usage_metadata`` 形态的构造（供应商回报什么就给什么）。

    ``cached_tokens`` / ``reasoning_tokens`` 省略即"该调用未回报这个字段"——
    与真实兼容端点的形态一致（不同供应商回报的 detail 字段不一样多）。
    """
    metadata: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if cached_tokens is not None:
        metadata["input_token_details"] = {"cache_read": cached_tokens}
    if reasoning_tokens is not None:
        metadata["output_token_details"] = {"reasoning": reasoning_tokens}
    return metadata


def ai_text(
    text: str, *, finish_reason: str | None = None, usage: dict[str, Any] | None = None
) -> AIMessage:
    """文本响应（可带 finish_reason 与 usage_metadata，分别用于截断提示与用量采集用例）。"""
    metadata = {"finish_reason": finish_reason} if finish_reason else {}
    return AIMessage(content=text, response_metadata=metadata, usage_metadata=usage)


_tool_call_counter = itertools.count(1)


def ai_tool_call(name: str, *, usage: dict[str, Any] | None = None, **arguments: Any) -> AIMessage:
    """工具调用响应（``call_id`` 按构造顺序递增：可复现、且重复调用不会撞 id）。"""
    call_id = f"call-{next(_tool_call_counter)}-{name}"
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": arguments, "id": call_id}],
        usage_metadata=usage,
    )


def file_fence(code: str, path: str = "main.py") -> str:
    """编码 / 修复产出的代码围栏文本（``python file=<相对路径>``，Design §4.2.2）。"""
    return f"实现说明。\n\n```python file={path}\n{code}\n```\n"


def review_fence(
    findings: Sequence[Mapping[str, Any]],
    summary: str = "本轮未发现问题。",
    *,
    files: Sequence[str] = ("main.py",),
) -> str:
    """审查契约的 JSON 围栏文本（files 清单 + findings + summary）。"""
    payload = {"files": list(files), "findings": list(findings), "summary": summary}
    return f"检查完毕。\n\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n"


def fix_contract(fixes: Sequence[Mapping[str, Any]], *, deleted: Sequence[str] = ()) -> str:
    """修复轮的输出契约：恰好一个 ```json 围栏。

    产物内容不在契约里——修复节点用 ``write_file`` 只写改动的文件（Design §4.2.2），
    版本目录的实际内容由引擎按磁盘校验并重算 manifest。
    """
    payload: dict[str, Any] = {"fixes": list(fixes)}
    if deleted:
        payload["deleted"] = list(deleted)
    return f"已修复。\n\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n"


def fix_turn(
    files: Mapping[str, str],
    fixes: Sequence[Mapping[str, Any]],
    *,
    deleted: Sequence[str] = (),
) -> list[Any]:
    """修复轮（假模型驱动真实节点）：用 ``write_file`` 写改动文件，再交回 fixes 围栏。

    ``write_file`` 的 ``path`` 相对上下文根，而修复节点的根就是本轮产出的版本目录，
    所以路径**不带** ``code_vN/`` 前缀。
    """
    calls = [
        ai_tool_call("write_file", path=path, content=content)
        for path, content in sorted(files.items())
    ]
    return [*calls, ai_text(fix_contract(fixes, deleted=deleted))]


def finding(
    severity: str = "严重",
    location: str = "main.py:1",
    issue: str = "示例问题",
    suggestion: str = "示例建议",
    *,
    uncertain: bool = False,
) -> dict[str, Any]:
    """一条 finding（id 由契约校验层生成，用例不写）。"""
    return {
        "severity": severity,
        "location": location,
        "issue": issue,
        "suggestion": suggestion,
        "uncertain": uncertain,
    }


# ---- 流程脚本（按图执行顺序排列的假模型响应）----


def clean_requirement_script(*, code: str) -> list[Any]:
    """需求 → 编码 → 审查（读文件后报无问题）：干净早退路径。"""
    return [
        ai_text(file_fence(code)),
        ai_tool_call("read_file", path="code_v1/main.py"),
        ai_text(review_fence([], "已读取完整文件，未发现阻塞问题。")),
    ]


def fix_cycle_script(
    *,
    code: str,
    fixed_code: str,
    name: str = "main.py",
    location: str | None = None,
    issue: str = "示例严重问题",
) -> list[Any]:
    """代码文件 → 首审含【严重】→ 修复 → 复审干净：修复循环路径。

    ``name`` 是版本内文件名（``-f`` 输入保留原文件名，须与运行输入一致）。
    """
    where = location or f"{name}:1"
    return [
        ai_tool_call("read_file", path=f"code_v1/{name}"),
        ai_text(review_fence([finding(location=where, issue=issue)], files=[name])),
        *fix_turn(
            {name: fixed_code},
            [{"finding_id": "f-1", "status": "fixed", "note": "已按建议修复并自验通过。"}],
        ),
        ai_tool_call("read_file", path=f"code_v2/{name}"),
        ai_text(review_fence([], "复审通过。", files=[name])),
    ]


def residual_script(*, code: str, name: str = "main.py", location: str | None = None) -> list[Any]:
    """每轮都报同类【严重】：`--max-iter 1` 下以残留收尾（退出码 1）。"""
    where = location or f"{name}:1"
    return [
        ai_tool_call("read_file", path=f"code_v1/{name}"),
        ai_text(review_fence([finding(location=where, issue="除零风险未处理")], files=[name])),
        ai_text(
            fix_contract(
                [{"finding_id": "f-1", "status": "skipped", "note": "脚本化样例：本轮不改动。"}]
            )
        ),
        ai_tool_call("read_file", path=f"code_v2/{name}"),
        ai_text(review_fence([finding(location=where, issue="除零风险未处理")], files=[name])),
    ]


def requirement_input_with_clean_reviews(
    *, code: str, reviews: int, with_tools: bool = True
) -> list[Any]:
    """需求 → 编码 → `reviews` 次"无问题"审查；`with_tools=False` 时一次工具都不调。

    后者用于验证执行有效性门禁：没有工具事件却报"无问题"的审查必须被拒绝。
    """
    script: list[Any] = [ai_text(file_fence(code))]
    for _ in range(reviews):
        if with_tools:
            script.append(ai_tool_call("read_file", path="code_v1/main.py"))
        script.append(ai_text(review_fence([], "未发现问题。")))
    return script


# ---- 会话模式的脚本片段（编排节点 + 审查轮次；Design §5.4 / §4.7）----


def orchestrator_reply(
    intent: str,
    *,
    rounds: int | None = None,
    rounds_mode: str | None = None,
    reply: str = "好的。",
) -> AIMessage:
    """编排节点的响应：一个合法的 JSON 围栏（意图 + 可选的轮次参数）。"""
    payload: dict[str, Any] = {"intent": intent, "reply": reply}
    if rounds is not None:
        payload["rounds"] = rounds
    if rounds_mode is not None:
        payload["rounds_mode"] = rounds_mode
    return ai_text("```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```\n")


def clean_review_calls(name: str = "main.py", *, version: int = 1) -> list[Any]:
    """审查轮次：读文件后报无问题（干净）。"""
    return [
        ai_tool_call("read_file", path=f"code_v{version}/{name}"),
        ai_text(review_fence([], "已读取完整文件，未发现阻塞问题。", files=[name])),
    ]


def dirty_review_calls(
    name: str = "main.py", *, version: int = 1, issue: str = "缺少除零处理"
) -> list[Any]:
    """审查轮次：读文件后报一条【严重】。"""
    return [
        ai_tool_call("read_file", path=f"code_v{version}/{name}"),
        ai_text(review_fence([finding(location=f"{name}:1", issue=issue)], files=[name])),
    ]
