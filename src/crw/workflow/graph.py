"""工作流图组装：StateGraph 的节点、边、条件路由与复审循环。

节点以“节点名 → 节点函数”的覆盖表注入，未覆盖的键使用内置装配，测试与真实装配共用同一入口；
LangGraph 类型不越出本层（Design §3.1、§4.1、§12.2）。

节点输出协议（Design §3.1）：节点为纯函数 ``node(state) -> dict``，只返回增量字段；原始文本
经暂存键交回 workflow 层消费——``encode`` 用 ``code_text``、``review`` 用 ``review_text``、
``fix`` 用 ``fix_text``。暂存键不在状态 schema 内，不进入运行记录；阶段摘要是引擎按产物与
契约结论自算的（节点不产出摘要文本）。
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from . import state as state_mod
from .state import RunState

#: 图节点名（Design §4.1）
NODE_NAMES: tuple[str, ...] = ("encode", "review", "fix")

#: 节点函数类型：``(state[, retry_feedback=...]) -> Mapping[str, Any]``（Design §3.1）
NodeFn = Callable[..., Mapping[str, Any]]

#: 脚本化生成失败时的保底需求摘要（保证假编码节点恒产出可编译代码）
FALLBACK_REQUIREMENT = "脚本化占位实现"

_MAX_SUMMARY_CHARS = 80


def resolve_nodes(nodes: Mapping[str, NodeFn] | None = None) -> dict[str, NodeFn]:
    """把覆盖表与缺省装配合并：逐键替换，未给出的键用缺省假节点（Design §12.2 第 3 步）。"""
    resolved: dict[str, NodeFn] = dict(default_nodes())
    if nodes:
        unknown = sorted(set(nodes) - set(resolved))
        if unknown:
            raise ValueError(f"未知节点名：{unknown}（可用：{list(NODE_NAMES)}）")
        resolved.update(nodes)
    return resolved


def build_graph(*, nodes: Mapping[str, NodeFn] | None = None) -> CompiledStateGraph:
    """组装并编译工作流图（Design §4.1 / §4.4）。

    Args:
        nodes: 节点覆盖表（键为 ``encode`` / ``review`` / ``fix``）；逐键替换缺省假节点。
            M1 起缺省为脚本化假节点，M2 起缺省替换为真实 Agent 节点，接缝不变。

    Returns:
        编译后的 StateGraph；调用方负责 ``stream`` 驱动与 ``recursion_limit``。
    """
    resolved = resolve_nodes(nodes)
    builder = StateGraph(RunState)
    for name in NODE_NAMES:
        builder.add_node(name, resolved[name])
    builder.add_conditional_edges(START, route_entry, {"encode": "encode", "review": "review"})
    builder.add_edge("encode", "review")
    builder.add_conditional_edges("review", route_after_review, {"fix": "fix", END: END})
    builder.add_edge("fix", "review")
    return builder.compile()


def route_entry(state: Mapping[str, Any]) -> str:
    """入口路由（只读纯函数）：需求文本从编码起步；代码文件直接进入审查（Design §4.4）。"""
    descriptor = state.get("input") or {}
    return "encode" if descriptor.get("kind") == state_mod.INPUT_REQUIREMENT else "review"


def route_after_review(state: Mapping[str, Any]) -> str:
    """复审条件边（只读纯函数）：干净 → 收尾；达修复轮次上限 → 收尾（残留）；否则进入修复。

    干净判定与轮次口径见 Design §4.4；``rounds-mode=exact``（跑满 n 轮）随 M4 的会话轮次
    语义落地，届时本函数按 ``rounds_mode`` 分支。
    """
    if state.get("clean"):
        return END
    if int(state.get("fix_rounds", 0)) >= int(state.get("max_iterations", 0)):
        return END
    return "fix"


# ---- 缺省假节点（M1 脚本化装配；M2 起由 agents/ 的真实节点取代）----


def default_nodes() -> dict[str, NodeFn]:
    """缺省节点装配：脚本化假实现，不接真实模型、行为按输入确定（Design §12.2 第 3 步）。"""
    return {"encode": fake_encode, "review": fake_review, "fix": fake_fix}


def fake_encode(state: Mapping[str, Any]) -> Mapping[str, Any]:
    """假编码节点：按需求文本确定性地生成单文件实现。"""
    requirement = str((state.get("input") or {}).get("requirement") or "")
    code = scripted_module_for(requirement)
    return {"code_text": f"```python\n{code}```\n"}


def fake_review(state: Mapping[str, Any]) -> Mapping[str, Any]:
    """假审查节点：对工作区当前代码做脚本化分级检查，输出审查契约（JSON 围栏）。"""
    code_path = Path(str(state.get("current_code") or ""))
    code = code_path.read_text(encoding="utf-8")
    findings = inspect_code(code, file_name=code_path.name)
    payload = {"findings": findings, "summary": summarize_findings(findings)}
    return {
        "review_text": f"脚本化审查结论如下。\n\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n"
    }


def fake_fix(state: Mapping[str, Any]) -> Mapping[str, Any]:
    """假修复节点：只做可脚本化且可复核的修复（语法缺陷），其余逐条标记 skipped。"""
    code_path = Path(str(state.get("current_code") or ""))
    code = code_path.read_text(encoding="utf-8")
    findings = list(state.get("findings") or [])
    repaired, fixes = scripted_repair(code, findings)
    payload = {"fixes": fixes}
    text = (
        f"```python\n{repaired}```\n\n"
        f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n"
    )
    return {"fix_text": text}


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


def scripted_repair(code: str, findings: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """脚本化修复：逐条给出处置，语法缺陷尝试可复核的修补，其余标记 skipped。

    修补结果必须整文件可编译才被采用（否则视为无法安全处理）；因此假修复节点不会
    把坏语法写回工作区。返回（修复后的代码，fixes 清单）。
    """
    current = code
    syntax_open = not _compiles(current)
    fixes: list[dict[str, Any]] = []
    for finding in findings:
        finding_id = str(finding.get("id") or "")
        if syntax_open:
            repaired = _repair_syntax(current)
            syntax_open = False
            if repaired is not None:
                current = repaired
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
    return current, fixes


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
