"""结构化契约：围栏抽取、审查 / 修复输出校验与干净判定（Design §4.2.1、§5.2）。"""

from __future__ import annotations

import json

import pytest

from crw.workflow import contracts


def review_text(payload: object, *, fence: str = "json") -> str:
    """构造审查节点的原始输出（带围栏的正文 + 围栏外叙述）。"""
    body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"审查结论如下。\n\n```{fence}\n{body}\n```\n"


def finding(
    severity: str = "严重", location: str = "code_v1.py:3", **extra: object
) -> dict[str, object]:
    return {"severity": severity, "location": location, "issue": "示例问题", **extra}


def parse_review(payload: object, **kwargs: object):
    defaults = {"review_round": 1, "code_version": 1, "file_name": "code_v1.py"}
    defaults.update(kwargs)
    return contracts.parse_review_output(review_text(payload), **defaults)  # type: ignore[arg-type]


# ---- 围栏抽取（Design §4.2.1）----


def test_extract_fence_accepts_python_and_py_case_insensitively() -> None:
    text = "```python\nprint(1)\n```\n\n```PY\nprint(2)\n```\n\n```Py\nprint(3)\n```"

    assert contracts.extract_fence(text, contracts.CODE_LABELS) == [
        "print(1)",
        "print(2)",
        "print(3)",
    ]


def test_extract_fence_requires_closing_fence_at_least_as_long() -> None:
    """闭合围栏需同种字符且不短于开启行（Markdown 语义）。"""
    text = "````python\n```\nprint(1)\n````\n"

    assert contracts.extract_fence(text, contracts.CODE_LABELS) == ["```\nprint(1)"]


def test_extract_fence_ignores_other_languages_and_unclosed_blocks() -> None:
    text = "```text\npython 不是围栏标签\n```\n\n```python\nprint(1)\n"  # 末块未闭合

    assert contracts.extract_fence(text, contracts.CODE_LABELS) == []


def test_extract_fence_skips_nested_foreign_blocks() -> None:
    """非目标语言的围栏整体跳过：其内容中的标签行不得被当作新的围栏开启。"""
    text = '```markdown\n```json\n{}\n```\n\n```json\n{"findings": []}\n```\n'

    assert contracts.extract_fence(text, contracts.JSON_LABELS) == ['{"findings": []}']


# ---- 审查契约（Design §5.2）----


def test_review_contract_generates_ids_in_order_and_fills_defaults() -> None:
    result = parse_review({"findings": [finding(), finding("建议")], "summary": "还行"})

    assert [item["id"] for item in result.findings] == ["f-1", "f-2"]
    assert result.findings[0]["uncertain"] is False
    assert result.findings[0]["suggestion"] == ""
    assert result.contract["iteration"] == 1
    assert result.contract["code_version"] == 1
    assert result.contract["file"] == "code_v1.py"
    assert result.contract["summary"] == "还行"
    assert result.summary == "还行"


def test_review_contract_keeps_unknown_fields() -> None:
    """未知字段忽略并原样保留（Design §5.2）。"""
    result = parse_review(
        {
            "findings": [finding(confidence=0.9)],
            "summary": "s",
            "extra_top": {"a": 1},
        }
    )

    assert result.contract["extra_top"] == {"a": 1}
    assert result.findings[0]["confidence"] == 0.9  # type: ignore[typeddict-item]


def test_review_contract_accepts_matching_iteration_and_file() -> None:
    result = parse_review(
        {
            "iteration": 2,
            "code_version": 2,
            "file": "code_v2.py",
            "findings": [],
            "summary": "干净",
        },
        review_round=2,
        code_version=2,
        file_name="code_v2.py",
    )

    assert result.contract["iteration"] == 2


@pytest.mark.parametrize(
    ("payload", "kwargs", "expected"),
    [
        ({"iteration": 9, "findings": [], "summary": "s"}, {}, "iteration 应为 1"),
        ({"code_version": 9, "findings": [], "summary": "s"}, {}, "code_version 应为 1"),
        ({"file": "other.py", "findings": [], "summary": "s"}, {}, "file 应为 'code_v1.py'"),
        ({"findings": []}, {}, "视为审查未执行"),
        ({"findings": [], "summary": "   "}, {}, "视为审查未执行"),
        ({"findings": {}, "summary": "s"}, {}, "findings 必须为数组"),
        ({"findings": [finding("致命")], "summary": "s"}, {}, "severity 必须为"),
        ({"findings": [finding(location="")], "summary": "s"}, {}, "location 必须为非空字符串"),
        ({"findings": [finding(issue=" ")], "summary": "s"}, {}, "issue 必须为非空字符串"),
        ({"findings": [finding(suggestion=3)], "summary": "s"}, {}, "suggestion 必须为字符串"),
        ({"findings": [finding(uncertain="是")], "summary": "s"}, {}, "uncertain 必须为布尔值"),
        ({"findings": [], "summary": 1}, {}, "summary 必须为字符串"),
        ({"summary": "s"}, {}, "findings 必须为数组"),
    ],
)
def test_review_contract_violations_report_readable_errors(
    payload: dict[str, object], kwargs: dict[str, object], expected: str
) -> None:
    with pytest.raises(contracts.ContractViolation) as excinfo:
        parse_review(payload, **kwargs)

    assert any(expected in error for error in excinfo.value.errors)


def test_review_contract_requires_exactly_one_json_fence() -> None:
    with pytest.raises(contracts.ContractViolation, match="恰好一个 ```json 围栏"):
        contracts.parse_review_output(
            "没有围栏的结论", review_round=1, code_version=1, file_name="code_v1.py"
        )


def test_review_contract_rejects_invalid_json_body() -> None:
    with pytest.raises(contracts.ContractViolation, match="不是合法 JSON"):
        contracts.parse_review_output(
            review_text("{不是 JSON}"), review_round=1, code_version=1, file_name="code_v1.py"
        )


def test_review_contract_rejects_non_object_top_level() -> None:
    with pytest.raises(contracts.ContractViolation, match="顶层必须是对象"):
        parse_review([1, 2])


def test_review_contract_requires_execution_evidence_for_empty_findings() -> None:
    """空 findings 且无成功工具事件时视为"审查未执行"（Design §5.2 门禁）。"""
    with pytest.raises(contracts.ContractViolation, match="视为审查未执行"):
        contracts.parse_review_output(
            review_text({"findings": [], "summary": "没有发现问题"}),
            review_round=1,
            code_version=1,
            file_name="code_v1.py",
            has_tool_evidence=False,
        )


def test_review_contract_accepts_empty_findings_with_summary() -> None:
    result = parse_review({"findings": [], "summary": "本轮未发现问题。"})

    assert result.findings == []


# ---- 干净判定 ----


@pytest.mark.parametrize(
    ("severities", "strict", "expected"),
    [
        ([], False, True),
        (["建议"], False, True),
        (["一般"], False, True),
        (["严重"], False, False),
        (["建议", "一般"], True, False),
        (["建议"], True, True),
        ([], True, True),
    ],
)
def test_judge_clean(severities: list[str], strict: bool, expected: bool) -> None:
    findings = [finding(severity) for severity in severities]

    assert contracts.judge_clean(findings, strict=strict) is expected


def test_severity_counts_and_residual_severe() -> None:
    findings = [finding("严重"), finding("一般"), finding("一般"), finding("建议")]

    assert contracts.severity_counts(findings) == {"严重": 1, "一般": 2, "建议": 1}
    assert contracts.residual_severe(findings) == 1
    assert list(contracts.severity_counts(findings)) == list(contracts.SEVERITIES)


# ---- 修复契约（Design §5.2）----


def fix_text(code: str, fixes: object) -> str:
    return (
        f"修复完成。\n\n```python\n{code}\n```\n\n"
        f"```json\n{json.dumps({'fixes': fixes}, ensure_ascii=False)}\n```\n"
    )


def test_fix_contract_covers_all_findings_in_order() -> None:
    findings = [finding(), finding("建议")]
    findings[0]["id"] = "f-1"
    findings[1]["id"] = "f-2"
    text = fix_text(
        "print(1)",
        [
            {"finding_id": "f-2", "status": "skipped", "note": "不采纳"},
            {"finding_id": "f-1", "status": "fixed", "note": "已修"},
        ],
    )

    result = contracts.parse_fix_output(text, findings=findings)

    assert result.code == "print(1)"
    assert [item["finding_id"] for item in result.fixes] == ["f-1", "f-2"]
    assert result.unknown_ids == ()


def test_fix_contract_last_duplicate_wins() -> None:
    findings = [{"id": "f-1"}]
    text = fix_text(
        "print(1)",
        [
            {"finding_id": "f-1", "status": "skipped", "note": "先跳过"},
            {"finding_id": "f-1", "status": "fixed", "note": "最终修复"},
        ],
    )

    result = contracts.parse_fix_output(text, findings=findings)

    assert result.fixes == [{"finding_id": "f-1", "status": "fixed", "note": "最终修复"}]


def test_fix_contract_ignores_unknown_ids_but_reports_them() -> None:
    """未知 id 忽略并在报告标注（Design §5.2 边界）。"""
    result = contracts.parse_fix_output(
        fix_text("print(1)", [{"finding_id": "f-9", "status": "fixed", "note": "无中生有"}]),
        findings=[],
    )

    assert result.fixes == []
    assert result.unknown_ids == ("f-9",)


@pytest.mark.parametrize(
    ("fixes", "expected"),
    [
        ([], "缺少 f-1"),
        ([{"finding_id": "f-1", "status": "skip", "note": "x"}], "status 必须为"),
        ([{"finding_id": "f-1", "status": "skipped", "note": " "}], "未给出理由"),
        ([{"finding_id": "f-1", "status": "skipped", "note": 3}], "note 必须为字符串"),
        ([{"finding_id": "", "status": "fixed", "note": ""}], "finding_id 必须为非空字符串"),
        (["不是对象"], "必须是对象"),
    ],
)
def test_fix_contract_violations(fixes: object, expected: str) -> None:
    with pytest.raises(contracts.ContractViolation) as excinfo:
        contracts.parse_fix_output(fix_text("print(1)", fixes), findings=[{"id": "f-1"}])

    assert any(expected in error for error in excinfo.value.errors)


def test_fix_contract_requires_a_code_fence() -> None:
    with pytest.raises(contracts.ContractViolation, match="缺少 ```python 围栏"):
        contracts.parse_fix_output("没有围栏", findings=[])


def test_fix_contract_rejects_empty_code_fence() -> None:
    with pytest.raises(contracts.ContractViolation, match="代码围栏为空"):
        contracts.parse_fix_output(fix_text("\n\n", []), findings=[])


def test_fix_contract_requires_a_json_fence() -> None:
    with pytest.raises(contracts.ContractViolation, match="缺少 ```json 围栏"):
        contracts.parse_fix_output("```python\nprint(1)\n```\n", findings=[])


def test_fix_contract_takes_the_first_fences_and_counts_extras() -> None:
    """Design §4.2.1：修复场景取第一个 python / json 围栏，其余围栏忽略并计数供记 log。"""
    fixes_body = json.dumps({"fixes": []}, ensure_ascii=False)
    text = (
        "```python\nprint(1)\n```\n\n"
        "```text\n说明块\n```\n\n"
        "```python\nprint(2)\n```\n\n"
        f"```json\n{fixes_body}\n```\n\n"
        "```json\n{}\n```\n"
    )

    result = contracts.parse_fix_output(text, findings=[])

    assert result.code == "print(1)"
    assert result.extra_fences == 3  # 共 5 个围栏：取用 1 python + 1 json，其余 3 个忽略


def test_scan_fences_reports_labels_and_content_in_order() -> None:
    text = "```Python\nprint(1)\n```\n\n```text\n旁白\n```\n\n```json\n{}\n```\n"

    assert contracts.scan_fences(text) == [
        ("python", "print(1)"),
        ("text", "旁白"),
        ("json", "{}"),
    ]


# ---- crw review 报告契约（前作 files[] 形态，Design §6.4）----


def _report(**overrides: object) -> dict:
    entry = {
        "file": "a.py",
        "findings": [
            {
                "severity": "严重",
                "location": "a.py:1",
                "issue": "问题",
                "suggestion": "建议",
                "uncertain": False,
            }
        ],
        "summary": "整体评价",
    }
    entry.update(overrides)
    base = {"files": [entry], "summary": "整体评价"}
    return base


def test_review_report_accepts_the_documented_shape() -> None:
    assert contracts.review_report_errors(_report()) == []


def test_review_report_fills_uncertain_default() -> None:
    data = _report(
        findings=[{"severity": "一般", "location": "a.py:2", "issue": "问题", "suggestion": ""}]
    )

    assert contracts.review_report_errors(data) == []
    assert data["files"][0]["findings"][0]["uncertain"] is False


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ([], "顶层必须是 JSON 对象"),
        ({"files": []}, "files 必须是非空数组"),
        ({"files": ["x"]}, "必须是对象"),
        ({"files": [{"file": "", "summary": "", "findings": []}]}, "file 必须是非空字符串"),
        ({"files": [{"file": "a.py", "findings": []}]}, "summary 必须是字符串"),
        ({"files": [{"file": "a.py", "summary": "", "findings": {}}]}, "findings 必须是数组"),
        (
            {"files": [{"file": "a.py", "summary": "", "findings": [{"severity": "致命"}]}]},
            "severity 必须是",
        ),
        (
            {
                "files": [
                    {
                        "file": "a.py",
                        "summary": "",
                        "findings": [
                            {"severity": "严重", "location": "", "issue": "x", "suggestion": ""}
                        ],
                    }
                ]
            },
            "location 必须是非空字符串",
        ),
        (
            {
                "files": [
                    {
                        "file": "a.py",
                        "summary": "",
                        "findings": [
                            {
                                "severity": "严重",
                                "location": "a.py:1",
                                "issue": "x",
                                "suggestion": "",
                                "uncertain": "yes",
                            }
                        ],
                    }
                ]
            },
            "uncertain 必须是布尔值",
        ),
    ],
)
def test_review_report_reports_schema_errors(data: object, match: str) -> None:
    assert any(match in error for error in contracts.review_report_errors(data))


def test_parse_review_report_strips_fences_and_fills_summary() -> None:
    payload = json.dumps(
        {"files": [{"file": "a.py", "findings": [], "summary": "干净。"}]}, ensure_ascii=False
    )

    parsed = contracts.parse_review_report(f"```json\n{payload}\n```")

    assert parsed["summary"] == ""  # 缺省的顶层 summary 补为空串（单文件合并时按文件摘要回填）


def test_parse_review_report_rejects_text_without_json() -> None:
    with pytest.raises(contracts.ContractViolation, match="未找到 JSON 对象"):
        contracts.parse_review_report("这次回答全是散文。")


def test_parse_review_report_rejects_broken_json() -> None:
    with pytest.raises(contracts.ContractViolation, match="JSON 解析失败"):
        contracts.parse_review_report('{"files": [}')


def test_merge_review_reports_single_and_multiple() -> None:
    single = {"file": "a.py", "findings": [], "summary": "干净。"}
    assert contracts.merge_review_reports([single]) == {"files": [single], "summary": "干净。"}

    other = {"file": "b.py", "findings": [], "summary": "也可接受。"}
    merged = contracts.merge_review_reports([single, other])
    assert merged["summary"] == "a.py：干净。；b.py：也可接受。"

    assert contracts.merge_review_reports([])["files"] == []
