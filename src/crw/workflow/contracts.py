"""结构化交接契约：审查 findings 与修复 fixes 的定义、校验与 id 追溯。

契约对象为普通数据结构，校验规则见 Design §5.2；校验失败的反馈经节点重试接缝回传。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .state import Finding

#: severity 枚举（Design §5.2）
SEVERE = "严重"
MODERATE = "一般"
ADVICE = "建议"
SEVERITIES: tuple[str, ...] = (SEVERE, MODERATE, ADVICE)

#: fixes.status 枚举（Design §5.2）
FIX_STATUSES: tuple[str, ...] = ("fixed", "partial", "skipped")

#: 代码围栏可接受的语言标签（Design §4.2.1：python / py 不区分大小写）
CODE_LABELS: tuple[str, ...] = ("python", "py")
JSON_LABELS: tuple[str, ...] = ("json",)

_FENCE_RE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>[^\s`~]*)[ \t]*$")


class ContractViolation(ValueError):
    """契约校验失败：携带逐条可读错误清单（Design §5.2 的重试接缝输入）。"""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors: tuple[str, ...] = tuple(errors)
        super().__init__("；".join(self.errors))


@dataclass(frozen=True)
class ReviewResult:
    """一轮审查的校验后结果。"""

    contract: dict[str, Any]
    """落盘对象（= review_N.json 内容，未知字段原样保留）。"""

    findings: list[Finding]
    summary: str


@dataclass(frozen=True)
class FixResult:
    """一轮修复的校验后结果。"""

    code: str
    fixes: list[dict[str, Any]]
    unknown_ids: tuple[str, ...]
    """未知 finding_id（Design §5.2：忽略并在报告标注）。"""

    extra_fences: int = 0
    """被忽略的多余围栏数（Design §4.2.1：其余围栏忽略并记 log）。"""


def extract_fence(text: str, labels: Sequence[str]) -> list[str]:
    """按 Markdown 语义抽取指定语言标签的围栏内容（Design §4.2.1）。

    闭合行需同种字符且长度不短于开启行（可写更长的反引号包住内容）；语言标签大小写
    不敏感；未闭合的围栏整块丢弃（内容不完整，抽取失败交给重试处理）。
    """
    wanted = {label.lower() for label in labels}
    return [content for label, content in scan_fences(text) if label in wanted]


def scan_fences(text: str) -> list[tuple[str, str]]:
    """按出现顺序扫描全部围栏，返回 ``(语言标签, 内容)``（标签已小写化）。

    供"其余围栏忽略并记 log"计数用：抽取只认目标标签，但被忽略的围栏也要能被数出来。
    """
    blocks: list[tuple[str, str]] = []
    fence: str | None = None
    info = ""
    buffer: list[str] = []
    for line in text.splitlines():
        match = _FENCE_RE.match(line)
        if fence is None:
            if match:
                fence = match.group("fence")
                info = match.group("info").lower()
                buffer = []
            continue
        if (
            match
            and match.group("fence")[0] == fence[0]
            and len(match.group("fence")) >= len(fence)
            and not match.group("info")
        ):
            blocks.append((info, "\n".join(buffer).strip("\n")))
            fence = None
        else:
            buffer.append(line)
    return blocks


def parse_review_output(
    text: str,
    *,
    review_round: int,
    code_version: int,
    file_name: str,
    has_tool_evidence: bool = True,
) -> ReviewResult:
    """抽取并校验审查契约对象（Design §5.2）。

    Args:
        text: 审查节点交回的原始文本，应含恰好一个 ```json 围栏。
        review_round: 权威审查序号（节点若给出 ``iteration`` 必须与之相等）。
        code_version: 权威代码版本号（节点若给出 ``code_version`` 必须与之相等）。
        file_name: 权威工作区实名（节点若给出 ``file`` 必须与之相等）。
        has_tool_evidence: 本轮是否已有成功工具事件或内联送达记录（执行有效性门禁）。
            M1 的假节点没有工具调用，调用方传 ``True``——契约文本本身就是"内联送达记录"；
            工具事件的真实判定随 M2 的 Agent 运行时接入（Design §5.2 的同一门禁）。

    Returns:
        :class:`ReviewResult`；契约非法时抛 :class:`ContractViolation`（附错误清单）。
    """
    blocks = extract_fence(text, JSON_LABELS)
    if len(blocks) != 1:
        raise ContractViolation([f"审查输出需要恰好一个 ```json 围栏，实际 {len(blocks)} 个"])
    try:
        raw = json.loads(blocks[0])
    except json.JSONDecodeError as exc:
        raise ContractViolation([f"审查输出的 ```json 围栏不是合法 JSON：{exc}"]) from exc
    if not isinstance(raw, dict):
        raise ContractViolation(["审查契约顶层必须是对象"])

    errors: list[str] = []
    for key, expected in (
        ("iteration", review_round),
        ("code_version", code_version),
        ("file", file_name),
    ):
        provided = raw.get(key)
        if provided is not None and provided != expected:
            errors.append(f"{key} 应为 {expected!r}，实际 {provided!r}")

    findings: list[Finding] = []
    raw_findings = raw.get("findings")
    if not isinstance(raw_findings, list):
        errors.append("findings 必须为数组（可为空数组）")
    else:
        findings = [
            _normalize_finding(item, index, errors) for index, item in enumerate(raw_findings, 1)
        ]

    summary = raw.get("summary", "")
    if not isinstance(summary, str):
        errors.append("summary 必须为字符串")
        summary = ""

    # 执行有效性门禁（Design §5.2）：空 findings 必须是"确实审查过"的结论，而非未执行的逃逸。
    if not findings and (not has_tool_evidence or not summary.strip()):
        errors.append("findings 为空且缺少执行证据（工具事件 / summary）：视为审查未执行")

    if errors:
        raise ContractViolation(errors)

    contract: dict[str, Any] = dict(raw)
    contract.update(
        {
            "iteration": review_round,
            "code_version": code_version,
            "file": file_name,
            "findings": findings,
            "summary": summary,
        }
    )
    return ReviewResult(contract=contract, findings=findings, summary=summary)


def parse_fix_output(
    text: str,
    *,
    findings: Sequence[Mapping[str, Any]],
) -> FixResult:
    """抽取并校验修复产物：代码围栏 + fixes 清单（Design §5.2）。

    fixes 必须覆盖上一轮 findings 的 id 全集（缺项即违规，交重试处理）；重复 id 取最后
    一条；未知 id 忽略并在报告标注；``skipped`` 必须给出理由。

    Design §4.2.1：修复场景取**第一个** python 围栏与**第一个** json 围栏，其余围栏忽略
    并记 log（由调用方把 ``extra_fences`` 写进事件摘要）；缺少任一围栏才按重试处理。

    Args:
        text: 修复节点交回的原始文本，应含 ```python 与 ```json 围栏各至少一个。
        findings: 上一轮审查的 findings（契约 id 全集的来源）。

    Returns:
        :class:`FixResult`；契约非法时抛 :class:`ContractViolation`。
    """
    code_blocks = extract_fence(text, CODE_LABELS)
    if not code_blocks:
        raise ContractViolation(["修复输出缺少 ```python 围栏（需要给出修复后的代码）"])
    code = code_blocks[0]
    if not code.strip():
        raise ContractViolation(["修复输出的代码围栏为空"])

    json_blocks = extract_fence(text, JSON_LABELS)
    if not json_blocks:
        raise ContractViolation(["修复输出缺少 ```json 围栏（需要给出 fixes 清单）"])
    try:
        raw = json.loads(json_blocks[0])
    except json.JSONDecodeError as exc:
        raise ContractViolation([f"修复输出的 ```json 围栏不是合法 JSON：{exc}"]) from exc
    if not isinstance(raw, dict):
        raise ContractViolation(["fixes 契约顶层必须是对象"])
    # Design §4.2.1：取第一个 python 围栏与第一个 json 围栏，其余围栏（含其他语言）忽略并记 log
    extra_fences = max(0, len(scan_fences(text)) - 2)

    raw_fixes = raw.get("fixes")
    if not isinstance(raw_fixes, list):
        raise ContractViolation(["fixes 必须为数组"])

    known = [str(item.get("id")) for item in findings]
    errors: list[str] = []
    latest: dict[str, dict[str, Any]] = {}
    unknown: list[str] = []
    for index, item in enumerate(raw_fixes, 1):
        if not isinstance(item, dict):
            errors.append(f"fixes[{index}] 必须是对象")
            continue
        finding_id = item.get("finding_id")
        if not isinstance(finding_id, str) or not finding_id.strip():
            errors.append(f"fixes[{index}].finding_id 必须为非空字符串")
            continue
        status_value = item.get("status")
        if status_value not in FIX_STATUSES:
            errors.append(
                f"fixes[{index}].status 必须为 fixed / partial / skipped 之一，实际 {status_value!r}"
            )
            continue
        note = item.get("note", "")
        if note is None:
            note = ""
        if not isinstance(note, str):
            errors.append(f"fixes[{index}].note 必须为字符串")
            continue
        if status_value == "skipped" and not note.strip():
            errors.append(f"fixes[{index}] 标记 skipped 但未给出理由（note）")
            continue
        if finding_id not in known:
            unknown.append(finding_id)
            continue
        latest[finding_id] = {"finding_id": finding_id, "status": status_value, "note": note}

    missing = [finding_id for finding_id in known if finding_id not in latest]
    if missing:
        errors.append(f"fixes 未覆盖全部 findings：缺少 {', '.join(missing)}")
    if errors:
        raise ContractViolation(errors)

    fixes = [latest[finding_id] for finding_id in known if finding_id in latest]
    return FixResult(code=code, fixes=fixes, unknown_ids=tuple(unknown), extra_fences=extra_fences)


def judge_clean(findings: Sequence[Mapping[str, Any]], *, strict: bool = False) -> bool:
    """干净判定（Design §4.4）：默认无【严重】；``strict`` 加严为无严重且无一般。"""
    return blocking_count(findings, strict=strict) == 0


# ---- crw review 的报告契约（前作 files[] 形态，Design §6.4）----


def review_report_errors(data: Any) -> list[str]:
    """``crw review --json`` 报告的 schema 校验；返回错误清单（空 = 通过）。

    沿用前作的 ``files[]`` 形态（Design §6.4：保持既有消费方式）：顶层
    ``{"files": [{file, findings, summary}], "summary"}``。**会就地补全** ``uncertain``
    缺省值（`False`）——本函数既校验也归一，调用方拿到的是补全后的同一对象；``files``
    为空是唯一允许的空数组例外（表示"没有可用的审查结果"，由
    :func:`merge_review_reports` 产出，不是模型输出）。
    """
    if not isinstance(data, dict):
        return ["顶层必须是 JSON 对象"]
    files = data.get("files")
    if not isinstance(files, list) or not files:
        return ["files 必须是非空数组"]
    errors: list[str] = []
    for index, entry in enumerate(files):
        where = f"files[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{where} 必须是对象")
            continue
        if not isinstance(entry.get("file"), str) or not entry.get("file"):
            errors.append(f"{where}.file 必须是非空字符串")
        if not isinstance(entry.get("summary"), str):
            errors.append(f"{where}.summary 必须是字符串")
        findings = entry.get("findings")
        if not isinstance(findings, list):
            errors.append(f"{where}.findings 必须是数组")
            continue
        for item_index, finding in enumerate(findings):
            item_where = f"{where}.findings[{item_index}]"
            if not isinstance(finding, dict):
                errors.append(f"{item_where} 必须是对象")
                continue
            if finding.get("severity") not in SEVERITIES:
                errors.append(f"{item_where}.severity 必须是 严重/一般/建议 之一")
            for key in ("location", "issue"):
                if not isinstance(finding.get(key), str) or not finding.get(key):
                    errors.append(f"{item_where}.{key} 必须是非空字符串")
            if not isinstance(finding.get("suggestion"), str):
                errors.append(f"{item_where}.suggestion 必须是字符串")
            uncertain = finding.get("uncertain", False)
            if not isinstance(uncertain, bool):
                errors.append(f"{item_where}.uncertain 必须是布尔值")
            else:
                finding["uncertain"] = uncertain  # 补全缺省，输出满足完整 schema
    return errors


def parse_review_report(answer: str) -> dict[str, Any]:
    """从模型回答中提取并校验 ``files[]`` 报告；失败抛 :class:`ContractViolation`。

    围栏可选：``crw review --json`` 的提示词要求纯 JSON，但模型常加围栏，此处宽容剥离。
    """
    text = answer.strip()
    if text.startswith("```"):
        newline = text.find("\n")
        text = text[newline + 1 :] if newline != -1 else text[3:]
        text = text.rstrip()
        if text.endswith("```"):
            text = text[:-3].rstrip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ContractViolation(["回答中未找到 JSON 对象"])
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ContractViolation([f"JSON 解析失败：{exc}"]) from exc
    errors = review_report_errors(data)
    if errors:
        raise ContractViolation(errors)
    if not isinstance(data.get("summary"), str):
        data["summary"] = ""  # 模型省略顶层 summary 时补全，下游消费口径稳定
    return data


def merge_review_reports(entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """多文件的 ``files[]`` 片段合并为统一形态；单文件直接透传其 summary。

    入参须是 :func:`parse_review_report` 校验过的条目（``file`` / ``summary`` 齐备）；
    字段缺失时按可读占位兜底，不抛 ``KeyError``——这是展示层，不该因一条残缺条目整体炸掉。
    ``entries`` 为空（全部文件审查失败）时返回 ``files: []`` + 说明性 summary，是
    ``files`` 允许为空的唯一场景。
    """
    if not entries:
        return {"files": [], "summary": "没有可用的审查结果。"}
    files = list(entries)
    if len(files) == 1:
        return {"files": files, "summary": str(files[0].get("summary") or "")}
    summary = "；".join(
        f"{entry.get('file') or '（未命名文件）'}：{entry.get('summary') or ''}" for entry in files
    )
    return {"files": files, "summary": summary}


def blocking_count(findings: Sequence[Mapping[str, Any]], *, strict: bool = False) -> int:
    """阻塞性结论条数（默认只数【严重】，``strict`` 下并入【一般】）。

    与 :func:`judge_clean` 同源：达限残留的文案与判定口径必须一致——``--strict`` 下
    残留可能全部由【一般】构成，报告与渲染不得再写死"【严重】"。
    """
    blocking = (SEVERE, MODERATE) if strict else (SEVERE,)
    return sum(1 for item in findings if item.get("severity") in blocking)


def severity_counts(findings: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """逐级计数（报告与阶段标题用），恒定键序：严重 / 一般 / 建议。"""
    return {
        level: sum(1 for item in findings if item.get("severity") == level) for level in SEVERITIES
    }


def residual_severe(findings: Sequence[Mapping[str, Any]]) -> int:
    """【严重】问题数（runs 列表的"残留严重数"列）。"""
    return severity_counts(findings)[SEVERE]


def _normalize_finding(item: Any, index: int, errors: list[str]) -> Finding:
    """单条 finding 校验与归一：id 按序生成、uncertain 缺省补 false、未知字段保留。"""
    if not isinstance(item, dict):
        errors.append(f"findings[{index}] 必须是对象")
        return Finding(id=f"f-{index}")
    severity = item.get("severity")
    if severity not in SEVERITIES:
        errors.append(
            f"findings[{index}].severity 必须为 严重 / 一般 / 建议 之一，实际 {severity!r}"
        )
    for key in ("location", "issue"):
        value = item.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"findings[{index}].{key} 必须为非空字符串")
    suggestion = item.get("suggestion", "")
    if suggestion is None:
        suggestion = ""
    if not isinstance(suggestion, str):
        errors.append(f"findings[{index}].suggestion 必须为字符串（可为空）")
        suggestion = ""
    uncertain = item.get("uncertain", False)
    if not isinstance(uncertain, bool):
        errors.append(f"findings[{index}].uncertain 必须为布尔值")
        uncertain = bool(uncertain)

    finding = Finding(**{str(key): value for key, value in item.items()})
    finding["id"] = f"f-{index}"
    finding["severity"] = str(severity)
    finding["location"] = str(item.get("location") or "")
    finding["issue"] = str(item.get("issue") or "")
    finding["suggestion"] = suggestion
    finding["uncertain"] = uncertain
    return finding
