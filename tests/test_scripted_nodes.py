"""脚本化假节点：确定性行为用例（假节点只作测试注入，不进生产装配）。

用例内容与 M1 一致，仅把节点实现从 ``crw.workflow.graph`` 迁到测试支持模块 ``tests/fakes.py``
——生产路径的缺省装配已是 agents 层的真实 Agent 节点（M2）。
"""

from __future__ import annotations

import ast
from pathlib import Path

import fakes
import pytest
from support import (
    fenced,
    fix_payload,
    read_sample,
    requirement_state,
    review_payload,
    write_version,
)
from support import (
    node_state as state_for,
)


def test_fake_encode_produces_compilable_module_with_docstring() -> None:
    output = fakes.fake_encode(requirement_state("写一个计算器"))

    code = fenced(output["code_text"], "python")
    ast.parse(code)
    assert ast.get_docstring(ast.parse(code))
    assert "写一个计算器" in code
    assert "file=main.py" in output["code_text"]  # 多文件契约的归属声明


def test_fake_encode_is_deterministic_and_handles_hostile_requirement() -> None:
    hostile = '需求含"""三引号"""与反斜杠\\与 "引号"'

    first = fakes.fake_encode(requirement_state(hostile))
    second = fakes.fake_encode(requirement_state(hostile))

    assert first == second
    ast.parse(fenced(first["code_text"], "python"))  # 内联恶意字符后仍须是可编译代码


def test_fake_encode_falls_back_when_requirement_breaks_inlining() -> None:
    """内联后无法编译时退回保底文案，而不是产出坏语法（NUL 等不可进源码的字符）。"""
    code = fakes.scripted_module_for("正常需求\x00注入")

    ast.parse(code)
    assert fakes.FALLBACK_REQUIREMENT in code


def test_fake_review_reports_clean_module_as_no_findings(tmp_path: Path) -> None:
    code = '"""示例模块。"""\n\n\ndef main() -> None:\n    print("hi")\n'
    state = state_for(
        version_dir=write_version(tmp_path, {"main.py": code}), files={"main.py": code}
    )

    payload = review_payload(fakes.fake_review(state)["review_text"])

    assert payload["findings"] == []
    assert payload["files"] == ["main.py"]
    assert payload["summary"].strip()


def test_fake_review_detects_syntax_error_with_location(tmp_path: Path) -> None:
    code = '"""示例模块。"""\n\n\ndef greet(name)\n    print(name)\n'
    state = state_for(
        version_dir=write_version(tmp_path, {"main.py": code}), files={"main.py": code}
    )

    payload = review_payload(fakes.fake_review(state)["review_text"])

    assert [item["severity"] for item in payload["findings"]] == ["严重"]
    assert payload["findings"][0]["location"] == "main.py:4"
    assert "语法错误" in payload["findings"][0]["issue"]


@pytest.mark.parametrize(
    "expression",
    ["sum(values) / len(values)", "total / 0", "total % len(items)"],
)
def test_fake_review_detects_division_risks(tmp_path: Path, expression: str) -> None:
    code = f'"""示例模块。"""\n\n\ndef compute(values):\n    return {expression}\n'
    state = state_for(
        version_dir=write_version(tmp_path, {"main.py": code}), files={"main.py": code}
    )

    payload = review_payload(fakes.fake_review(state)["review_text"])

    assert [item["severity"] for item in payload["findings"]] == ["严重"]
    assert payload["findings"][0]["location"] == "main.py:5"


def test_fake_review_suggests_docstring_when_missing(tmp_path: Path) -> None:
    code = "print(1)\n"
    state = state_for(
        version_dir=write_version(tmp_path, {"main.py": code}), files={"main.py": code}
    )

    payload = review_payload(fakes.fake_review(state)["review_text"])

    assert [item["severity"] for item in payload["findings"]] == ["建议"]
    assert "docstring" in payload["findings"][0]["issue"]


def test_fake_review_on_repository_fixtures(tmp_path: Path) -> None:
    """仓库内两份缺陷样例的行为即演示口径：语法错误 → 严重；除零 → 严重。"""
    syntax_state = state_for(
        version_dir=write_version(tmp_path / "syntax", {"main.py": read_sample("syntax_error.py")}),
        files={"main.py": ""},
    )
    division_state = state_for(
        version_dir=write_version(
            tmp_path / "division", {"main.py": read_sample("divide_by_zero.py")}
        ),
        files={"main.py": ""},
    )

    syntax_findings = review_payload(fakes.fake_review(syntax_state)["review_text"])["findings"]
    division_findings = review_payload(fakes.fake_review(division_state)["review_text"])["findings"]

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
            "location": "main.py:4",
            "issue": "语法错误：expected ':'",
        }
    ]

    repaired, fixes = fakes.scripted_repair({"main.py": code}, findings)

    ast.parse(repaired["main.py"])
    assert "def greet(name):" in repaired["main.py"]
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
        {"id": "f-1", "severity": "严重", "location": "main.py:5", "issue": "除数是 len(...)"}
    ]

    repaired, fixes = fakes.scripted_repair({"main.py": code}, findings)

    assert repaired == {"main.py": code}
    assert fixes[0]["status"] == "skipped"
    assert fixes[0]["note"].strip()


def test_fake_fix_covers_every_finding_id() -> None:
    code = '"""示例模块。"""\n\n\ndef greet(name)\n    print(name)\n'
    findings = [
        {
            "id": "f-1",
            "severity": "严重",
            "location": "main.py:4",
            "issue": "语法错误：expected ':'",
        },
        {"id": "f-2", "severity": "建议", "location": "main.py", "issue": "模块缺少说明文档"},
    ]

    _, fixes = fakes.scripted_repair({"main.py": code}, findings)

    assert [item["finding_id"] for item in fixes] == ["f-1", "f-2"]


def test_fake_fix_writes_only_the_repaired_file(tmp_path: Path) -> None:
    """假修复节点与真实节点同形态：改动写回版本目录，只交回一个 fixes 围栏（§4.2.2）。"""
    code = '"""示例模块。"""\n\n\ndef greet(name)\n    print(name)\n'
    other = '"""另一个模块。"""\n'
    version_dir = write_version(tmp_path, {"main.py": code, "util.py": other})
    state = state_for(
        version_dir=version_dir,
        files={"main.py": code, "util.py": other},
        findings=[
            {
                "id": "f-1",
                "severity": "严重",
                "location": "main.py:4",
                "issue": "语法错误：expected ':'",
            }
        ],
    )

    text = fakes.fake_fix(state)["fix_text"]

    assert text.count("```json") == 1
    assert "```python" not in text  # 产物内容不进契约文本
    directory = Path(version_dir)
    assert (directory / "main.py").read_text(encoding="utf-8") != code
    assert (directory / "util.py").read_text(encoding="utf-8") == other  # 未改动文件不动
    assert fix_payload(text)["fixes"][0]["status"] == "fixed"


def test_fake_fix_cannot_repair_unrecognized_syntax() -> None:
    """无法安全修补（整文件仍不可编译）时标记 skipped 且不改动代码。"""
    code = "def broken(:\n    pass\n"
    findings = [{"id": "f-1", "severity": "严重", "location": "main.py:1", "issue": "语法错误"}]

    repaired, fixes = fakes.scripted_repair({"main.py": code}, findings)

    assert repaired == {"main.py": code}
    assert fixes[0]["status"] == "skipped"
