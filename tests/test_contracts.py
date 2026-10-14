"""结构化契约：围栏抽取、审查 / 修复输出校验与干净判定（Design §4.2.2、§5.2）。"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from crw.workflow import contracts


def review_text(payload: object, *, fence: str = "json") -> str:
    """构造审查节点的原始输出（带围栏的正文 + 围栏外叙述）。"""
    body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"审查结论如下。\n\n```{fence}\n{body}\n```\n"


def finding(
    severity: str = "严重", location: str = "main.py:3", **extra: object
) -> dict[str, object]:
    return {"severity": severity, "location": location, "issue": "示例问题", **extra}


def parse_review(payload: object, **kwargs: object):
    defaults = {"review_round": 1, "code_version": 1, "files": ["main.py"]}
    defaults.update(kwargs)
    return contracts.parse_review_output(review_text(payload), **defaults)  # type: ignore[arg-type]


# ---- 围栏抽取（Design §4.2.1 / §4.2.2）----


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


def test_scan_file_fences_parses_the_file_declaration() -> None:
    """`` ```python file=<相对路径> `` 的声明被解析为版本内相对路径（Design §4.2.2）。"""
    text = (
        "```python file=main.py\nprint(1)\n```\n\n"
        "```PY file=pkg/util.py\nprint(2)\n```\n\n"
        '```python file="带 空格.py"\nprint(3)\n```\n'
    )

    fences = contracts.scan_file_fences(text)

    assert [(fence.file, fence.content) for fence in fences] == [
        ("main.py", "print(1)"),
        ("pkg/util.py", "print(2)"),
        ("带 空格.py", "print(3)"),
    ]
    assert {fence.label for fence in fences} == {"python", "py"}


def test_scan_file_fences_normalizes_windows_separators() -> None:
    """反斜杠按 Windows 心智归一为 posix 风格（模型常按本机习惯写路径）。"""
    text = "```python file=pkg\\util.py\nprint(1)\n```\n"

    assert contracts.scan_file_fences(text)[0].file == "pkg/util.py"


def test_scan_file_fences_keeps_undeclared_fences_detectable() -> None:
    """未声明 file= 的围栏保留 file=None，由校验层按违规处理。"""
    text = "```python\nprint(1)\n```\n"

    assert contracts.scan_file_fences(text)[0].file is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("main.py", "main.py"),
        (" pkg/util.py ", "pkg/util.py"),
        ("pkg/./util.py", "pkg/util.py"),
    ],
)
def test_validate_version_path_accepts_relative_paths(raw: str, expected: str) -> None:
    assert contracts.validate_version_path(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["/abs.py", "C:\\abs.py", "~/home.py", "../outside.py", "a/../../b.py", "", "   "],
)
def test_validate_version_path_rejects_paths_leaving_the_version(raw: str) -> None:
    with pytest.raises(contracts.ContractViolation):
        contracts.validate_version_path(raw)


# ---- 审查契约（Design §5.2）----


def test_review_contract_generates_ids_in_order_and_fills_defaults() -> None:
    result = parse_review({"findings": [finding(), finding("建议")], "summary": "还行"})

    assert [item["id"] for item in result.findings] == ["f-1", "f-2"]
    assert result.findings[0]["uncertain"] is False
    assert result.findings[0]["suggestion"] == ""
    assert result.contract["iteration"] == 1
    assert result.contract["code_version"] == 1
    assert result.contract["files"] == ["main.py"]
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


def test_review_contract_accepts_matching_files_and_version() -> None:
    result = parse_review(
        {
            "iteration": 2,
            "code_version": 2,
            "files": ["b.py", "a.py"],
            "findings": [],
            "summary": "干净",
        },
        review_round=2,
        code_version=2,
        files=["a.py", "b.py"],
    )

    assert result.contract["iteration"] == 2
    assert result.contract["files"] == ["a.py", "b.py"]  # 权威清单按字典序回填


def test_review_contract_drops_the_legacy_file_key() -> None:
    """旧单文件契约的顶层 ``file`` 被 files 取代：残留键不进落盘对象（Design §4.2.2）。"""
    result = parse_review({"file": "code_v1.py", "findings": [], "summary": "干净"})

    assert "file" not in result.contract
    assert result.contract["files"] == ["main.py"]


@pytest.mark.parametrize(
    ("payload", "kwargs", "expected"),
    [
        ({"iteration": 9, "findings": [], "summary": "s"}, {}, "iteration 应为 1"),
        ({"code_version": 9, "findings": [], "summary": "s"}, {}, "code_version 应为 1"),
        (
            {"files": ["other.py"], "findings": [], "summary": "s"},
            {},
            "files 应为版本内清单",
        ),
        (
            {"files": "main.py", "findings": [], "summary": "s"},
            {},
            "files 必须是字符串数组",
        ),
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
            "没有围栏的结论", review_round=1, code_version=1, files=["main.py"]
        )


def test_review_contract_rejects_invalid_json_body() -> None:
    with pytest.raises(contracts.ContractViolation, match="不是合法 JSON"):
        contracts.parse_review_output(
            review_text("{不是 JSON}"), review_round=1, code_version=1, files=["main.py"]
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
            files=["main.py"],
            has_tool_evidence=False,
        )


def test_review_contract_accepts_empty_findings_with_summary() -> None:
    result = parse_review({"findings": [], "summary": "本轮未发现问题。"})

    assert result.findings == []


# ---- 干净判定 ----


@pytest.mark.parametrize(
    ("severities", "severe_only", "expected"),
    [
        ([], False, True),
        (["建议"], False, True),
        (["一般"], False, False),
        (["严重"], False, False),
        (["建议", "一般"], False, False),
        ([], True, True),
        (["建议"], True, True),
        (["一般"], True, True),
        (["严重"], True, False),
    ],
)
def test_judge_clean(severities: list[str], severe_only: bool, expected: bool) -> None:
    """默认判据把【一般】也当阻塞（Design §4.4）；``severe_only`` 才是旧的"只数【严重】"。"""
    findings = [finding(severity) for severity in severities]

    assert contracts.judge_clean(findings, severe_only=severe_only) is expected


def test_severity_counts_and_residual_severe() -> None:
    findings = [finding("严重"), finding("一般"), finding("一般"), finding("建议")]

    assert contracts.severity_counts(findings) == {"严重": 1, "一般": 2, "建议": 1}
    assert contracts.residual_severe(findings) == 1
    assert list(contracts.severity_counts(findings)) == list(contracts.SEVERITIES)


# ---- 修复契约：写盘形态（Design §4.2.2 / §5.2）----

#: 修复契约的输入基准：上一轮审查的两条 findings（fixes 必须逐条对应它们的 id）
_FIX_FINDINGS: list[dict[str, object]] = [{"id": "f-1"}, {"id": "f-2"}]


def fix_text(fixes: object, *, deleted: object = None, ignored_fences: str = "") -> str:
    """构造修复节点的原始输出：恰好一个 ```json 围栏（fixes 清单 + 可选删除申请）。

    产物内容不经本契约（修复节点用 ``write_file`` 只写改动的文件，版本目录由引擎按磁盘
    内容校验），因此这里不构造代码围栏；``ignored_fences`` 只服务"多余围栏计数"的用例。
    """
    payload: dict[str, object] = {"fixes": fixes}
    if deleted is not None:
        payload["deleted"] = deleted
    return (
        f"修复完成。\n\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```\n{ignored_fences}"
    )


def parse_fix(
    text: str,
    *,
    findings: list[dict[str, object]] | None = None,
    current_files: list[str] | None = None,
) -> contracts.FixResult:
    """按修复契约解析（findings 缺省为 f-1 / f-2，current_files 缺省为空版本）。"""
    return contracts.parse_fix_output(
        text,
        findings=_FIX_FINDINGS if findings is None else findings,
        current_files=[] if current_files is None else current_files,
    )


def deleted_paths(result: contracts.FixResult) -> list[str]:
    """删除申请的路径清单（断言写法统一成 list，比较更直观）。"""
    return list(result.deleted)


def unknown_ids(result: contracts.FixResult) -> list[str]:
    """未识别 finding 的 id 序列（按提交顺序）。"""
    return list(result.unknown_ids)


def test_fix_result_carries_only_the_disposal_lists() -> None:
    """契约层的修复结果只有处置清单、未知 id、删除申请与忽略围栏计数。

    产物文件表不在这里：写盘形态下文件内容经 ``write_file`` 直接落版本目录，
    契约再持有一份就会与磁盘上的权威内容分叉。
    """
    assert [field.name for field in dataclasses.fields(contracts.FixResult)] == [
        "fixes",
        "unknown_ids",
        "deleted",
        "extra_fences",
    ]


def test_fix_contract_covers_all_findings_in_order() -> None:
    """fixes 覆盖全部 id，输出按 findings 顺序而非提交顺序（报告按"问题 ← 处置"渲染）。"""
    text = fix_text(
        [
            {"finding_id": "f-2", "status": "skipped", "note": "不采纳"},
            {"finding_id": "f-1", "status": "fixed", "note": "已修"},
        ]
    )

    result = parse_fix(text)

    assert [item["finding_id"] for item in result.fixes] == ["f-1", "f-2"]
    assert [item["status"] for item in result.fixes] == ["fixed", "skipped"]
    assert result.fixes[1]["note"] == "不采纳"
    assert unknown_ids(result) == []
    assert deleted_paths(result) == []
    assert result.extra_fences == 0  # 恰好一个 json 围栏：没有忽略项


def test_fix_contract_takes_the_last_entry_for_a_duplicate_id() -> None:
    """重复 finding_id 取最后一条（Design §5.2 边界：后面的是修正过的结论）。"""
    result = parse_fix(
        fix_text(
            [
                {"finding_id": "f-1", "status": "fixed", "note": "先说修好了"},
                {"finding_id": "f-1", "status": "partial", "note": "复核后只改了一半"},
            ]
        ),
        findings=[{"id": "f-1"}],
    )

    assert result.fixes == [{"finding_id": "f-1", "status": "partial", "note": "复核后只改了一半"}]


def test_fix_contract_ignores_unknown_ids_but_reports_them() -> None:
    """未知 id 忽略并在报告标注（Design §5.2 边界）。"""
    result = parse_fix(
        fix_text([{"finding_id": "f-9", "status": "fixed", "note": "无中生有"}]), findings=[]
    )

    assert result.fixes == []
    assert unknown_ids(result) == ["f-9"]


def test_fix_contract_keeps_known_items_and_reports_unknown_ids_in_submission_order() -> None:
    """未知 id 不进 fixes、也不影响已知项；报告里按提交顺序列出它们。"""
    result = parse_fix(
        fix_text(
            [
                {"finding_id": "f-9", "status": "fixed", "note": "无中生有"},
                {"finding_id": "f-2", "status": "fixed", "note": "已修"},
                {"finding_id": "f-8", "status": "partial", "note": "同样无中生有"},
            ]
        ),
        findings=[{"id": "f-2"}],
    )

    assert result.fixes == [{"finding_id": "f-2", "status": "fixed", "note": "已修"}]
    assert unknown_ids(result) == ["f-9", "f-8"]


def test_fix_contract_lists_every_missing_finding_id() -> None:
    """缺项即违规，且缺失清单要逐条列全——重试反馈得让模型知道该补哪几条。"""
    with pytest.raises(contracts.ContractViolation) as excinfo:
        parse_fix(
            fix_text([{"finding_id": "f-2", "status": "fixed", "note": "已修"}]),
            findings=[{"id": "f-1"}, {"id": "f-2"}, {"id": "f-3"}],
        )

    assert any("缺少 f-1, f-3" in error for error in excinfo.value.errors)


@pytest.mark.parametrize(
    ("fixes", "expected"),
    [
        ([], "缺少 f-1"),
        ("不是数组", "fixes 必须为数组"),
        ({"f-1": "fixed"}, "fixes 必须为数组"),
        (None, "fixes 必须为数组"),
        ([{"finding_id": "f-1", "status": "skip", "note": "x"}], "status 必须为"),
        ([{"finding_id": "f-1", "status": "skipped", "note": " "}], "未给出理由"),
        ([{"finding_id": "f-1", "status": "skipped", "note": 3}], "note 必须为字符串"),
        ([{"finding_id": "f-1", "status": "skipped"}], "未给出理由"),
        ([{"finding_id": "", "status": "fixed", "note": ""}], "finding_id 必须为非空字符串"),
        (["不是对象"], "必须是对象"),
        ([{"status": "fixed", "note": "缺 id"}], "finding_id 必须为非空字符串"),
    ],
)
def test_fix_contract_violations(fixes: object, expected: str) -> None:
    with pytest.raises(contracts.ContractViolation) as excinfo:
        parse_fix(fix_text(fixes), findings=[{"id": "f-1"}])

    assert any(expected in error for error in excinfo.value.errors)


def test_fix_contract_requires_usable_finding_ids() -> None:
    """上一轮 findings 缺少可用 id（记录可能被手改）时给可读违规，而不是报"缺少 None"。"""
    with pytest.raises(contracts.ContractViolation, match="缺少可用 id"):
        parse_fix(fix_text([]), findings=[{"severity": "严重"}])


@pytest.mark.parametrize(
    "text",
    [
        "没有围栏的修复说明。",
        "```python file=main.py\nprint(1)\n```\n",  # 只贴代码、不交处置清单
    ],
)
def test_fix_contract_requires_a_json_fence(text: str) -> None:
    with pytest.raises(contracts.ContractViolation, match="缺少 ```json 围栏"):
        parse_fix(text, findings=[])


def test_fix_contract_rejects_invalid_json_body() -> None:
    with pytest.raises(contracts.ContractViolation, match="不是合法 JSON"):
        parse_fix("```json\n{不是 JSON}\n```\n", findings=[])


def test_fix_contract_rejects_non_object_top_level() -> None:
    with pytest.raises(contracts.ContractViolation, match="fixes 契约顶层必须是对象"):
        parse_fix("```json\n[1, 2]\n```\n", findings=[])


def test_fix_contract_does_not_gate_version_contents() -> None:
    """文件覆盖齐全 / 空文件属引擎的写盘校验（``_verify_fix_writes``），契约不重复把关。"""
    result = parse_fix(fix_text([]), findings=[], current_files=["main.py", "util.py"])

    assert result.fixes == []
    assert deleted_paths(result) == []
    assert result.extra_fences == 0


def test_fix_contract_parses_the_deleted_declaration() -> None:
    """``{"deleted": […]}`` 是删除申请：解析出来交给引擎确认（Design §4.2.2）。"""
    result = parse_fix(
        fix_text([], deleted=["legacy.py", "legacy.py", "extra.py"]),
        findings=[],
        current_files=["main.py", "legacy.py", "extra.py"],
    )

    # 重复路径去重且保持首次出现顺序
    assert deleted_paths(result) == ["legacy.py", "extra.py"]


def test_fix_contract_rejects_deleting_paths_outside_the_version() -> None:
    with pytest.raises(contracts.ContractViolation, match="版本外的路径：nope.py"):
        parse_fix(fix_text([], deleted=["nope.py"]), findings=[], current_files=["main.py"])


def test_fix_contract_keeps_a_deleted_proposal_without_a_matching_finding() -> None:
    """findings 里没有对应条目的删除申请照样成立：依据为空也要过用户的确认闸门。"""
    result = parse_fix(
        fix_text(
            [{"finding_id": "f-1", "status": "fixed", "note": "已按建议处理。"}],
            deleted=["unused.py"],
        ),
        findings=[{"id": "f-1", "severity": "严重", "location": "main.py:1", "issue": "除零"}],
        current_files=["main.py", "unused.py"],
    )

    assert deleted_paths(result) == ["unused.py"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("/abs.py", "必须是版本内相对路径"),
        ("C:\\abs.py", "必须是版本内相对路径"),
        ("~/home.py", "必须是版本内相对路径"),
        ("../outside.py", "不得越出版本目录"),
        ("a/../../b.py", "不得越出版本目录"),
        ("", "为空路径"),
    ],
)
def test_fix_contract_rejects_malformed_deleted_paths(raw: str, expected: str) -> None:
    """删除声明的路径与围栏的 file= 声明同一把尺子（拒绝绝对路径 / ``~`` / ``..``）。"""
    with pytest.raises(contracts.ContractViolation) as excinfo:
        parse_fix(fix_text([], deleted=[raw]), findings=[], current_files=[raw or "main.py"])

    errors = excinfo.value.errors
    assert any(error.startswith("deleted 声明") and expected in error for error in errors), errors


@pytest.mark.parametrize("declared", ["legacy.py", 1, [1], {"path": "legacy.py"}, True])
def test_fix_contract_requires_deleted_to_be_a_string_array(declared: object) -> None:
    with pytest.raises(contracts.ContractViolation, match="deleted 必须是字符串数组"):
        parse_fix(
            f"```json\n{json.dumps({'fixes': [], 'deleted': declared}, ensure_ascii=False)}\n```\n",
            findings=[],
            current_files=["legacy.py"],
        )


def test_fix_contract_reports_deleted_and_fixes_violations_together() -> None:
    """两类问题并入同一份错误清单：重试一次就能拿全部问题，少一轮往返。"""
    with pytest.raises(contracts.ContractViolation) as excinfo:
        parse_fix(fix_text([], deleted=["nope.py"]), current_files=["main.py"])

    errors = excinfo.value.errors
    assert any("版本外的路径：nope.py" in error for error in errors)
    assert any("缺少 f-1, f-2" in error for error in errors)


def test_fix_contract_counts_every_fence_beyond_the_json_one() -> None:
    """修复产出只应有那一个 json 围栏：其余围栏（含模型沿用旧形态贴的代码围栏）忽略但计数。"""
    result = parse_fix(
        fix_text(
            [],
            ignored_fences=(
                "```python file=main.py\nprint(1)\n```\n\n```text\n旁白\n```\n\n```json\n{}\n```\n"
            ),
        ),
        findings=[],
    )

    assert result.fixes == []
    assert result.extra_fences == 3


def test_fix_contract_uses_the_first_json_fence_only() -> None:
    """第二个 json 围栏不取用（只数不计入清单），否则处置清单会被模型自己顶掉。"""
    result = parse_fix(
        fix_text(
            [{"finding_id": "f-1", "status": "fixed", "note": "已修"}],
            ignored_fences='```json\n{"fixes": []}\n```\n',
        ),
        findings=[{"id": "f-1"}],
    )

    assert result.fixes == [{"finding_id": "f-1", "status": "fixed", "note": "已修"}]
    assert result.extra_fences == 1


# ---- 版本量级上限（Design §4.2.2：编码围栏抽取与引擎写盘校验共用）----


def test_version_size_errors_accepts_a_version_within_the_limits() -> None:
    """上限是"超过才违规"：边界值本身必须放行，否则重试判据比 Design 更严。"""
    assert contracts.version_size_errors({}) == []
    assert contracts.version_size_errors({"main.py": "print(1)\n"}) == []
    assert contracts.version_size_errors({"main.py": "x" * contracts.MAX_FILE_BYTES}) == []
    assert (
        contracts.version_size_errors(
            {f"f{index}.py": "x" for index in range(contracts.MAX_VERSION_FILES)}
        )
        == []
    )


def test_version_size_errors_reports_an_oversized_single_file() -> None:
    """超限文件要逐条列全：只报第一个会让重试漏掉其余文件。"""
    big = "x" * (contracts.MAX_FILE_BYTES + 1)

    errors = contracts.version_size_errors(
        {"main.py": "print(1)", "pkg/util.py": big, "legacy.py": big}
    )

    reported = [error for error in errors if "单文件上限" in error]
    assert len(reported) == 1
    assert "pkg/util.py" in reported[0] and "legacy.py" in reported[0]
    assert "main.py" not in reported[0]


def test_version_size_errors_measures_bytes_not_characters() -> None:
    """中文源码一个字符三个字节：按字符数判上限会放行超限版本。"""
    content = "中" * (contracts.MAX_FILE_BYTES // 3 + 1)

    assert len(content) < contracts.MAX_FILE_BYTES  # 按字符看"没超限"
    assert any("单文件上限" in error for error in contracts.version_size_errors({"a.py": content}))


def test_version_size_errors_reports_too_many_files() -> None:
    files = {f"f{index}.py": "print(1)" for index in range(contracts.MAX_VERSION_FILES + 1)}

    errors = contracts.version_size_errors(files)

    assert any("文件数超过上限" in error and f"实际 {len(files)} 个" in error for error in errors)


def test_version_size_errors_reports_the_total_size() -> None:
    """总量上限独立成立：每个文件都不超限、合起来超限。"""
    files = {f"f{index}.py": "x" * (contracts.MAX_FILE_BYTES - 1) for index in range(5)}

    errors = contracts.version_size_errors(files)

    assert any("总大小超过上限" in error for error in errors)
    assert not any("单文件上限" in error for error in errors)
    assert not any("文件数" in error for error in errors)


def test_collect_version_files_enforces_the_size_limits() -> None:
    """编码围栏抽取用同一把尺子（Design §4.2.2）：超限按契约违规重试。"""
    big = "x" * (contracts.MAX_FILE_BYTES + 1)
    with pytest.raises(contracts.ContractViolation, match="单文件上限"):
        contracts.collect_version_files(f"```python file=main.py\n{big}\n```\n", require_path=True)

    many = "\n\n".join(
        f"```python file=f{index}.py\nprint(1)\n```"
        for index in range(contracts.MAX_VERSION_FILES + 1)
    )
    with pytest.raises(contracts.ContractViolation, match="文件数超过上限"):
        contracts.collect_version_files(f"{many}\n", require_path=True)


def test_collect_version_files_keeps_the_last_duplicate_and_counts_extras() -> None:
    """重复路径取最后一个并计数；非代码围栏只计数不取用（阶段摘要据此记"已忽略"）。"""
    text = (
        "```python file=main.py\nprint(1)\n```\n\n"
        "```python file=main.py\nprint(2)\n```\n\n"
        "```text\n旁白\n```\n"
    )

    files, duplicates, extra_fences = contracts.collect_version_files(text, require_path=True)

    assert files == {"main.py": "print(2)"}
    assert duplicates == 1
    assert extra_fences == 1


def test_collect_version_files_requires_a_code_fence() -> None:
    with pytest.raises(contracts.ContractViolation, match="缺少 ```python 围栏"):
        contracts.collect_version_files("只有一句说明。", require_path=True)


def test_collect_version_files_requires_the_file_declaration() -> None:
    """没有 ``file=`` 声明的代码围栏属于契约违规（多文件产出的归属必须显式）。"""
    with pytest.raises(contracts.ContractViolation, match="file= 声明"):
        contracts.collect_version_files("```python\nprint(1)\n```\n", require_path=True)


def test_collect_version_files_rejects_an_empty_fence() -> None:
    with pytest.raises(contracts.ContractViolation, match="内容为空"):
        contracts.collect_version_files("```python file=main.py\n\n```\n", require_path=True)


def test_scan_fences_reports_labels_and_content_in_order() -> None:
    text = "```Python\nprint(1)\n```\n\n```text\n旁白\n```\n\n```json\n{}\n```\n"

    assert contracts.scan_fences(text) == [
        ("python", "print(1)"),
        ("text", "旁白"),
        ("json", "{}"),
    ]


def test_manifest_payload_lists_files_with_hashes_in_order() -> None:
    """manifest 是"路径 → sha256"的权威清单（整版本比较的基准，Design §4.2.2）。"""
    payload = contracts.manifest_payload(version=2, files={"b.py": "print(2)", "a.py": "print(1)"})

    assert payload["version"] == 2
    assert [entry["path"] for entry in payload["files"]] == ["a.py", "b.py"]
    assert payload["files"][0]["lines"] == 1
    assert payload["total_bytes"] == len("print(1)") + len("print(2)")
    assert payload["files"][0]["sha256"] == hashlib.sha256(b"print(1)").hexdigest()


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


# ---- M3：并行双视角的汇合规则（Design §12.2）----


def test_merge_findings_dedupes_and_keeps_the_more_severe_one() -> None:
    """同一 (location, issue) 只留一条，取较严重者并记下来源视角。"""
    correctness = [
        finding("严重", issue="除零"),
        finding("建议", location="code_v1.py:9", issue="命名"),
    ]
    security = [
        finding("建议", issue="除零"),
        finding("严重", location="code_v1.py:12", issue="注入"),
    ]

    merged = contracts.merge_findings([("正确性", correctness), ("安全性", security)])

    assert [item["id"] for item in merged] == ["f-1", "f-2", "f-3"]
    shared = merged[0]
    assert shared["issue"] == "除零" and shared["severity"] == "严重"
    assert shared["views"] == ["正确性", "安全性"]
    assert merged[1]["views"] == ["正确性"] and merged[1]["severity"] == "建议"
    assert merged[2]["views"] == ["安全性"] and merged[2]["severity"] == "严重"


def test_merge_findings_prefers_the_more_severe_suggestion_and_flag() -> None:
    """取较严重者时连同它的建议与 uncertain 标记一起采纳（结论与依据保持一致）。"""
    merged = contracts.merge_findings(
        [
            ("正确性", [finding("建议", suggestion="补个注释", uncertain=False)]),
            ("安全性", [finding("严重", suggestion="加长度上限", uncertain=True)]),
        ]
    )

    assert merged[0]["severity"] == "严重"
    assert merged[0]["suggestion"] == "加长度上限"
    assert merged[0]["uncertain"] is True


def test_merge_findings_keeps_first_seen_order_and_is_empty_safe() -> None:
    assert contracts.merge_findings([]) == []
    assert contracts.merge_findings([("正确性", []), ("安全性", [])]) == []
    merged = contracts.merge_findings([("正确性", [finding()]), ("安全性", [])])
    assert len(merged) == 1 and merged[0]["id"] == "f-1"


def test_finding_keys_ignores_non_mapping_entries() -> None:
    """未收敛判定读的是展示数据：残缺条目跳过，不因一条坏数据整体失败。"""
    keys = contracts.finding_keys([finding(), "坏数据", None])  # type: ignore[list-item]

    assert keys == frozenset({("main.py:3", "示例问题")})


def test_merge_findings_normalizes_unknown_severity() -> None:
    """枚举外的 severity（手改过的记录）归一到【建议】：不得留在结果里"看不见地"绕过干净判定。"""
    merged = contracts.merge_findings([("正确性", [finding("critical")])])

    assert merged[0]["severity"] == contracts.ADVICE
    assert contracts.judge_clean(merged) is True  # 归一后判定看得见它（【建议】不阻塞）
    assert contracts.severity_counts(merged)[contracts.ADVICE] == 1


def test_quoted_file_argument_last_occurrence_wins() -> None:
    """info 里出现多个 file= 时两条分支口径一致：最后一个生效。"""
    text = '```python file="a.py" file="b.py"\nprint(1)\n```\n'

    assert contracts.scan_file_fences(text)[0].file == "b.py"


def test_severity_counts_skips_non_mapping_entries() -> None:
    """计数读展示数据：残缺条目（手改过的记录）跳过，不把列表页 / 详情页打崩。"""
    findings = [finding("严重"), "坏数据", None]  # type: ignore[list-item]

    counts = contracts.severity_counts(findings)

    assert counts[contracts.SEVERE] == 1
    assert contracts.blocking_count(findings) == 1
