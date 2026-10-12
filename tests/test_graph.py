"""图组装与缺省假节点：节点 / 边集合、条件路由与脚本化行为（Design §4.1、§4.4、§12.2）。"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from support import sample_path

from crw.workflow import graph as graph_mod


def state_for(*, kind: str = "file", max_iterations: int = 3, **overrides: object) -> dict:
    """构造节点可用的最小状态（只含节点读取的字段）。"""
    base: dict = {
        "input": {"kind": kind, "requirement": "写一个计算器"},
        "current_code": str(sample_path("syntax_error.py")),
        "findings": [],
        "clean": False,
        "fix_rounds": 0,
        "review_round": 0,
        "max_iterations": max_iterations,
        "strict": False,
    }
    base.update(overrides)
    return base


def requirement_state(requirement: str) -> dict:
    """需求文本输入的节点状态。"""
    return state_for(kind="requirement", input={"kind": "requirement", "requirement": requirement})


def write_code(tmp_path: Path, text: str, name: str = "code_v1.py") -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def fenced(text: str, label: str) -> str:
    """取文本中第一个指定标签围栏的内容（假节点输出格式即契约格式）。"""
    return text.split(f"```{label}", 1)[1].split("```", 1)[0]


def review_payload(review_text: str) -> dict:
    return json.loads(fenced(review_text, "json"))


def fix_payload(fix_text: str) -> dict:
    return json.loads(fenced(fix_text, "json"))


# ---- 组装与路由 ----


def test_default_nodes_are_the_three_expected_keys() -> None:
    assert set(graph_mod.default_nodes()) == set(graph_mod.NODE_NAMES)


def test_resolve_nodes_replaces_key_by_key() -> None:
    """覆盖表逐键替换：未给出的键仍用缺省假节点（Design §12.2 第 3 步）。"""

    def my_review(state: dict) -> dict:
        return {"review_text": "```json\n{}\n```"}

    resolved = graph_mod.resolve_nodes({"review": my_review})

    assert resolved["review"] is my_review
    assert resolved["encode"] is graph_mod.fake_encode
    assert resolved["fix"] is graph_mod.fake_fix


def test_resolve_nodes_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="未知节点名"):
        graph_mod.resolve_nodes({"deploy": graph_mod.fake_encode})


def test_build_graph_exposes_the_designed_nodes_and_edges() -> None:
    """节点 / 边集合与 Design §4.1 一致（M3 的 crw graph 快照测试以此为基线）。"""
    compiled = graph_mod.build_graph()
    structure = compiled.get_graph()

    assert set(structure.nodes) == {"__start__", "__end__", "encode", "review", "fix"}
    edges = {(edge.source, edge.target) for edge in structure.edges}
    assert edges == {
        ("__start__", "encode"),
        ("__start__", "review"),
        ("encode", "review"),
        ("review", "fix"),
        ("review", "__end__"),
        ("fix", "review"),
    }


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("requirement", "encode"), ("file", "review")],
)
def test_route_entry(kind: str, expected: str) -> None:
    """需求文本从编码起步；代码文件直接进入审查（Design §4.4）。"""
    assert graph_mod.route_entry(state_for(kind=kind)) == expected


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"clean": True}, "__end__"),
        ({"clean": True, "fix_rounds": 3, "max_iterations": 3}, "__end__"),
        ({"clean": False, "fix_rounds": 3, "max_iterations": 3}, "__end__"),
        ({"clean": False, "fix_rounds": 0, "max_iterations": 0}, "__end__"),
        ({"clean": False, "fix_rounds": 1, "max_iterations": 3}, "fix"),
    ],
)
def test_route_after_review(overrides: dict, expected: str) -> None:
    """干净 / 达限收尾，否则进入修复（Design §4.4 的循环语义）。"""
    assert graph_mod.route_after_review(state_for(**overrides)) == expected


# ---- 假节点：脚本化且确定性 ----


def test_fake_encode_produces_compilable_module_with_docstring() -> None:
    output = graph_mod.fake_encode(requirement_state("写一个计算器"))

    code = fenced(output["code_text"], "python")
    ast.parse(code)
    assert ast.get_docstring(ast.parse(code))
    assert "写一个计算器" in code


def test_fake_encode_is_deterministic_and_handles_hostile_requirement() -> None:
    hostile = '需求含"""三引号"""与反斜杠\\与 "引号"'

    first = graph_mod.fake_encode(requirement_state(hostile))
    second = graph_mod.fake_encode(requirement_state(hostile))

    assert first == second
    ast.parse(fenced(first["code_text"], "python"))  # 内联恶意字符后仍须是可编译代码


def test_fake_encode_falls_back_when_requirement_breaks_inlining() -> None:
    """内联后无法编译时退回保底文案，而不是产出坏语法（NUL 等不可进源码的字符）。"""
    code = graph_mod.scripted_module_for("正常需求\x00注入")

    ast.parse(code)
    assert graph_mod.FALLBACK_REQUIREMENT in code


def test_fake_review_reports_clean_module_as_no_findings(tmp_path: Path) -> None:
    code = '"""示例模块。"""\n\n\ndef main() -> None:\n    print("hi")\n'
    state = state_for(current_code=write_code(tmp_path, code))

    payload = review_payload(graph_mod.fake_review(state)["review_text"])

    assert payload["findings"] == []
    assert payload["summary"].strip()


def test_fake_review_detects_syntax_error_with_location(tmp_path: Path) -> None:
    code = '"""示例模块。"""\n\n\ndef greet(name)\n    print(name)\n'
    state = state_for(current_code=write_code(tmp_path, code))

    payload = review_payload(graph_mod.fake_review(state)["review_text"])

    assert [item["severity"] for item in payload["findings"]] == ["严重"]
    assert payload["findings"][0]["location"] == "code_v1.py:4"
    assert "语法错误" in payload["findings"][0]["issue"]


@pytest.mark.parametrize(
    "expression",
    ["sum(values) / len(values)", "total / 0", "total % len(items)"],
)
def test_fake_review_detects_division_risks(tmp_path: Path, expression: str) -> None:
    code = f'"""示例模块。"""\n\n\ndef compute(values):\n    return {expression}\n'
    state = state_for(current_code=write_code(tmp_path, code))

    payload = review_payload(graph_mod.fake_review(state)["review_text"])

    assert [item["severity"] for item in payload["findings"]] == ["严重"]
    assert payload["findings"][0]["location"] == "code_v1.py:5"


def test_fake_review_suggests_docstring_when_missing(tmp_path: Path) -> None:
    state = state_for(current_code=write_code(tmp_path, "print(1)\n"))

    payload = review_payload(graph_mod.fake_review(state)["review_text"])

    assert [item["severity"] for item in payload["findings"]] == ["建议"]
    assert "docstring" in payload["findings"][0]["issue"]


def test_fake_review_on_repository_fixtures() -> None:
    """仓库内两份缺陷样例的行为即演示口径：语法错误 → 严重；除零 → 严重。"""
    syntax_state = state_for(current_code=str(sample_path("syntax_error.py")))
    division_state = state_for(current_code=str(sample_path("divide_by_zero.py")))

    syntax_findings = review_payload(graph_mod.fake_review(syntax_state)["review_text"])["findings"]
    division_findings = review_payload(graph_mod.fake_review(division_state)["review_text"])[
        "findings"
    ]

    assert [item["severity"] for item in syntax_findings] == ["严重"]
    assert "语法错误" in syntax_findings[0]["issue"]
    assert [item["severity"] for item in division_findings] == ["严重"]
    assert "len(...)" in division_findings[0]["issue"]


def test_fake_fix_repairs_syntax_and_verifies_by_compiling() -> None:
    code = '"""示例模块。"""\n\n\ndef greet(name)\n    print(name)\n'
    findings = [
        {
            "id": "f-1",
            "severity": "严重",
            "location": "code_v1.py:4",
            "issue": "语法错误：expected ':'",
        }
    ]

    repaired, fixes = graph_mod.scripted_repair(code, findings)

    ast.parse(repaired)
    assert "def greet(name):" in repaired
    assert fixes == [
        {
            "finding_id": "f-1",
            "status": "fixed",
            "note": "已修正语法缺陷并就地复核（整文件可编译）通过。",
        }
    ]


def test_fake_fix_skips_semantic_defects_without_touching_code() -> None:
    code = '"""示例模块。"""\n\n\ndef average(values):\n    return sum(values) / len(values)\n'
    findings = [
        {"id": "f-1", "severity": "严重", "location": "code_v1.py:5", "issue": "除数是 len(...)"}
    ]

    repaired, fixes = graph_mod.scripted_repair(code, findings)

    assert repaired == code
    assert fixes[0]["status"] == "skipped"
    assert fixes[0]["note"].strip()


def test_fake_fix_covers_every_finding_id() -> None:
    code = '"""示例模块。"""\n\n\ndef greet(name)\n    print(name)\n'
    findings = [
        {
            "id": "f-1",
            "severity": "严重",
            "location": "code_v1.py:4",
            "issue": "语法错误：expected ':'",
        },
        {"id": "f-2", "severity": "建议", "location": "code_v1.py", "issue": "模块缺少说明文档"},
    ]

    _, fixes = graph_mod.scripted_repair(code, findings)

    assert [item["finding_id"] for item in fixes] == ["f-1", "f-2"]


def test_fake_fix_output_carries_one_python_and_one_json_fence(tmp_path: Path) -> None:
    code = '"""示例模块。"""\n\n\ndef greet(name)\n    print(name)\n'
    state = state_for(
        current_code=write_code(tmp_path, code),
        findings=[
            {
                "id": "f-1",
                "severity": "严重",
                "location": "code_v1.py:4",
                "issue": "语法错误：expected ':'",
            }
        ],
    )

    text = graph_mod.fake_fix(state)["fix_text"]

    assert text.count("```python") == 1
    assert text.count("```json") == 1
    assert fix_payload(text)["fixes"][0]["status"] == "fixed"


def test_fake_fix_cannot_repair_unrecognized_syntax() -> None:
    """无法安全修补（整文件仍不可编译）时标记 skipped 且不改动代码。"""
    code = "def broken(:\n    pass\n"
    findings = [{"id": "f-1", "severity": "严重", "location": "code_v1.py:1", "issue": "语法错误"}]

    repaired, fixes = graph_mod.scripted_repair(code, findings)

    assert repaired == code
    assert fixes[0]["status"] == "skipped"
