"""结构化交接契约：审查 findings 与修复 fixes 的定义、校验与 id 追溯。

契约对象为普通数据结构：每个入口只做"抽取 + 逐条校验 + 归一"，校验失败时把可读错误清单
经节点重试接缝回传给模型重试一次。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .state import Finding

#: severity 枚举：阻塞判定与报告只认这三个取值
SEVERE = "严重"
MODERATE = "一般"
ADVICE = "建议"
SEVERITIES: tuple[str, ...] = (SEVERE, MODERATE, ADVICE)

#: fixes.status 枚举：标记 skipped 时必须附理由
FIX_STATUSES: tuple[str, ...] = ("fixed", "partial", "skipped")

#: 代码围栏可接受的语言标签（python / py 不区分大小写）
CODE_LABELS: tuple[str, ...] = ("python", "py")
JSON_LABELS: tuple[str, ...] = ("json",)

#: 版本量级上限：文件数 ≤ 20、单文件 ≤ 256 KB、版本总计 ≤ 1 MB
MAX_VERSION_FILES = 20
MAX_FILE_BYTES = 256 * 1024
MAX_VERSION_BYTES = 1024 * 1024

# 围栏 info 允许空格分隔的多个词（`` ```python file=<相对路径> ``），
# 但不得含反引号 / 波浪号：闭合标记行靠"info 为空"识别，info 里出现围栏字符会破坏该判据
_FENCE_RE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})(?:[ \t]*(?P<info>[^`~]*?))?[ \t]*$")


def normalize_code(code: str) -> str:
    """代码文本归一：去掉首尾换行、统一以单个换行结尾（行内空白原样保留）。

    围栏抽取会剥掉内容首尾的换行，磁盘版本统一为"单个结尾换行"——两处表示一致，
    整版本逐字节比较才有意义。``-f`` 输入、引擎物化与写入口径都走这一把尺子。
    """
    return code.strip("\n") + "\n"


class ContractViolation(ValueError):
    """契约校验失败：携带逐条可读错误清单，清单即节点重试的反馈输入。"""

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
    """一轮修复的校验后结果（写盘形态）。

    修复节点用 ``write_file`` 只写改动的文件，产物内容因此**不经过本契约**——这里只
    承载处置清单与删除申请；版本目录的实际文件集合由引擎按磁盘内容校验与重算。
    """

    fixes: list[dict[str, Any]]
    unknown_ids: tuple[str, ...]
    """未知 finding_id：忽略该条，仅在报告中标注。"""

    deleted: tuple[str, ...] = ()
    """修复 Agent **提出**删除的路径：执行前须用户确认。"""

    extra_fences: int = 0
    """被忽略的多余围栏数：其余围栏忽略但记入日志。"""


@dataclass(frozen=True)
class CodeFence:
    """一个声明了产出路径的代码围栏（围栏 info 形如 `` ```python file=<相对路径> ``）。"""

    label: str
    file: str | None
    """``file=`` 声明的相对路径；未声明为 ``None``（多文件契约下属违规）。"""

    content: str


def extract_fence(text: str, labels: Sequence[str]) -> list[str]:
    """按 Markdown 语义抽取指定语言标签的围栏内容。

    闭合行需同种字符且长度不短于开启行（可写更长的反引号包住内容）；语言标签大小写
    不敏感；未闭合的围栏整块丢弃（内容不完整，抽取失败交给重试处理）。
    """
    wanted = {label.lower() for label in labels}
    return [content for label, content in scan_fences(text) if label in wanted]


def scan_fences(text: str) -> list[tuple[str, str]]:
    """按出现顺序扫描全部围栏，返回 ``(语言标签, 内容)``（标签已小写化）。

    语言标签取 info 的第一个词（`` ```python file=a.py `` 的标签是 ``python``）。
    供"其余围栏忽略并记 log"计数用：抽取只认目标标签，但被忽略的围栏也要能被数出来。
    """
    return [(label, content) for label, _, content in _scan_fence_blocks(text)]


def scan_file_fences(text: str, labels: Sequence[str] = CODE_LABELS) -> list[CodeFence]:
    """抽取带语言标签的代码围栏及其 ``file=`` 声明（版本产出契约的入口）。

    语言标签不区分大小写；``file=`` 参数取 info 中最后一个 ``file=`` 词，可带成对引号
    （路径含空格时的声明写法）；未声明 ``file=`` 时 :attr:`CodeFence.file` 为 ``None``。
    """
    wanted = {label.lower() for label in labels}
    result: list[CodeFence] = []
    for label, info, content in _scan_fence_blocks(text):
        if label in wanted:
            result.append(CodeFence(label=label, file=_file_argument(info), content=content))
    return result


def _scan_fence_blocks(text: str) -> list[tuple[str, str, str]]:
    """围栏扫描的单一实现：按出现顺序返回 ``(标签, info 原文, 内容)``。

    闭合行需同种字符且长度不短于开启行、info 为空；未闭合的围栏整块
    丢弃（内容不完整，抽取失败交给重试处理）。
    """
    blocks: list[tuple[str, str, str]] = []
    fence: str | None = None
    info = ""
    label = ""
    buffer: list[str] = []
    for line in text.splitlines():
        match = _FENCE_RE.match(line)
        if fence is None:
            if match:
                fence = match.group("fence")
                info = (match.group("info") or "").strip()
                label = info.split()[0].lower() if info else ""
                buffer = []
            continue
        if (
            match
            and match.group("fence")[0] == fence[0]
            and len(match.group("fence")) >= len(fence)
            and not (match.group("info") or "").strip()
        ):
            blocks.append((label, info, "\n".join(buffer).strip("\n")))
            fence = None
        else:
            buffer.append(line)
    return blocks


def _file_argument(info: str) -> str | None:
    """从围栏 info 解析 ``file=`` 参数；未声明返回 ``None``。

    带成对引号的形态优先（``py file="带 空格.py"``，引号内允许空格）；否则按空白分词取
    最后一个 ``file=`` 词。反斜杠归一为 posix 风格（Windows 心智的模型输出）。
    """
    quoted = re.findall(r'file="([^"]+)"', info) or re.findall(r"file='([^']+)'", info)
    if quoted:
        return quoted[-1].replace("\\", "/")
    found: str | None = None
    for token in info.split():
        if token.startswith("file=") and len(token) > len("file="):
            found = token[len("file=") :]
    if found is None:
        return None
    if len(found) >= 2 and found[0] == found[-1] and found[0] in ("'", '"'):
        found = found[1:-1]
    return found.replace("\\", "/")


def validate_version_path(raw: str) -> str:
    """校验"版本内相对路径"并归一为 posix 风格；违规抛 :class:`ContractViolation`。

    拒绝绝对路径、盘符、``~`` 展开、越出版本的 ``..`` 段与空路径——版本目录是产物
    边界，声明路径必须落在它内部。反斜杠按 Windows 心智归一为 ``/``。
    """
    text = raw.replace("\\", "/").strip()
    if not text:
        raise ContractViolation(["围栏的 file= 声明为空路径"])
    if text.startswith(("/", "~")) or (len(text) > 1 and text[1] == ":"):
        raise ContractViolation([f"围栏的 file= 声明必须是版本内相对路径，收到 {raw!r}"])
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise ContractViolation([f"围栏的 file= 声明不得越出版本目录，收到 {raw!r}"])
    return "/".join(parts)


def parse_review_output(
    text: str,
    *,
    review_round: int,
    code_version: int,
    files: Sequence[str],
    has_tool_evidence: bool = True,
) -> ReviewResult:
    """抽取并校验审查契约对象（多文件形态）。

    Args:
        text: 审查节点交回的原始文本，应含恰好一个 ```json 围栏。
        review_round: 权威审查序号（节点若给出 ``iteration`` 必须与之相等）。
        code_version: 权威代码版本号（节点若给出 ``code_version`` 必须与之相等）。
        files: 当前版本的权威文件清单（版本内相对路径；节点若给出 ``files`` 必须一致）。
        has_tool_evidence: 本轮是否已有成功工具事件或内联送达记录（执行有效性门禁）。
            假节点 / 注入桩没有工具调用，调用方传 ``True``——契约文本本身就是"内联送达
            记录"；工具事件的真实判定由 Agent 运行时统计，两处是同一门禁。

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
    for key, expected in (("iteration", review_round), ("code_version", code_version)):
        provided = raw.get(key)
        if provided is not None and provided != expected:
            errors.append(f"{key} 应为 {expected!r}，实际 {provided!r}")
    provided_files = raw.get("files")
    if provided_files is not None:
        if not isinstance(provided_files, list) or not all(
            isinstance(item, str) for item in provided_files
        ):
            errors.append("files 必须是字符串数组")
        elif sorted(provided_files) != sorted(files):
            errors.append(f"files 应为版本内清单 {sorted(files)}，实际 {sorted(provided_files)}")

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

    # 执行有效性门禁：空 findings 必须是"确实审查过"的结论，而非未执行的逃逸。
    if not findings and (not has_tool_evidence or not summary.strip()):
        errors.append("findings 为空且缺少执行证据（工具事件 / summary）：视为审查未执行")

    if errors:
        raise ContractViolation(errors)

    contract: dict[str, Any] = dict(raw)
    # 旧单文件契约的顶层键已被 files 取代：残留的 "file" 不进落盘对象，
    # 避免 show / 报告两类消费方读到两种口径
    contract.pop("file", None)
    contract.update(
        {
            "iteration": review_round,
            "code_version": code_version,
            "files": sorted(files),
            "findings": findings,
            "summary": summary,
        }
    )
    return ReviewResult(contract=contract, findings=findings, summary=summary)


def collect_version_files(text: str, *, require_path: bool) -> tuple[dict[str, str], int, int]:
    """抽取产出契约的版本文件表：路径 → 内容。

    每个文件一个 python 围栏，围栏 info 必须声明 ``file=<版本内相对路径>``；重复路径取
    最后一个并计数（阶段摘要记"已忽略重复声明"）。同时执行量级上限：文件数、单文件与
    版本总字节数超限按契约违规重试。

    Args:
        text: 节点交回的原始文本。
        require_path: ``file=`` 声明是否必需（编码 / 修复产出恒为真；保留参数是为
            把"围栏必须有归属路径"的判定收敛在一处）。

    Returns:
        ``(路径 → 内容, 被忽略的重复声明数, 被忽略的多余围栏数)``。

    Raises:
        ContractViolation: 缺围栏 / 空内容 / 缺声明 / 路径越界 / 超量级上限。
    """
    fences = scan_file_fences(text)
    if not fences:
        raise ContractViolation(["输出缺少 ```python 围栏（需要给出代码文件内容）"])
    errors: list[str] = []
    files: dict[str, str] = {}
    duplicates = 0
    for index, fence in enumerate(fences, 1):
        if fence.file is None:
            if require_path:
                errors.append(
                    f"第 {index} 个代码围栏缺少 file= 声明（形如 ```python file=main.py）"
                )
            continue
        try:
            path = validate_version_path(fence.file)
        except ContractViolation as exc:
            errors.extend(exc.errors)
            continue
        if not fence.content.strip():
            errors.append(f"围栏 {path} 的内容为空")
            continue
        if path in files:
            duplicates += 1
        files[path] = fence.content
    if not files and not errors:
        errors.append("未抽取出任何代码文件（每个文件需要一个 ```python file=… 围栏）")
    errors.extend(version_size_errors(files))
    if errors:
        raise ContractViolation(errors)
    extra_fences = max(0, len(scan_fences(text)) - len(fences))
    return files, duplicates, extra_fences


def parse_fix_output(
    text: str,
    *,
    findings: Sequence[Mapping[str, Any]],
    current_files: Sequence[str],
) -> FixResult:
    """校验修复产出：fixes 处置清单与删除申请（写盘形态）。

    fixes 必须覆盖上一轮 findings 的 id 全集（缺项即违规，交重试处理）；重复 id 取最后
    一条；未知 id 忽略并在报告标注；``skipped`` 必须给出理由。删除在同一围栏里以
    ``{"deleted": […]}`` **提出**，声明路径必须在当前版本内，执行前由引擎征得用户确认。

    **产物内容不在本契约里**：修复节点用 ``write_file`` 只写它改动的文件，版本目录的
    实际文件集合由引擎按磁盘内容校验（缺文件 / 空文件 / 超量级 → 契约违规重试一次）。

    Args:
        text: 修复节点交回的原始文本。
        findings: 上一轮审查的 findings（契约 id 全集的来源）。
        current_files: 当前版本的文件清单（deleted 声明的校验基准）。

    Returns:
        :class:`FixResult`；契约非法时抛 :class:`ContractViolation`。
    """
    json_blocks = extract_fence(text, JSON_LABELS)
    if not json_blocks:
        raise ContractViolation(["修复输出缺少 ```json 围栏（需要给出 fixes 清单）"])
    try:
        raw = json.loads(json_blocks[0])
    except json.JSONDecodeError as exc:
        raise ContractViolation([f"修复输出的 ```json 围栏不是合法 JSON：{exc}"]) from exc
    if not isinstance(raw, dict):
        raise ContractViolation(["fixes 契约顶层必须是对象"])
    # 修复产出只应有一个 json 围栏：其余围栏（含模型沿用旧形态贴出的代码围栏）忽略但留痕
    extra_fences = max(0, len(scan_fences(text)) - 1)

    errors: list[str] = []
    deleted = _deleted_paths(raw, current_files, errors)

    known: list[str] = []
    for index, item in enumerate(findings, 1):
        finding_id = item.get("id")
        if not isinstance(finding_id, str) or not finding_id.strip():
            errors.append(f"上一轮 findings[{index}] 缺少可用 id（记录可能被手改），无法做逐条对应")
            continue
        known.append(finding_id)

    raw_fixes = raw.get("fixes")
    if not isinstance(raw_fixes, list):
        errors.append("fixes 必须为数组")
        if errors:
            raise ContractViolation(errors)
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
            if finding_id not in unknown:  # 同一未知 id 重复出现只记一条
                unknown.append(finding_id)
            continue
        latest[finding_id] = {"finding_id": finding_id, "status": status_value, "note": note}

    missing = [finding_id for finding_id in known if finding_id not in latest]
    if missing:
        errors.append(f"fixes 未覆盖全部 findings：缺少 {', '.join(missing)}")
    if errors:
        raise ContractViolation(errors)

    fixes = [latest[finding_id] for finding_id in known if finding_id in latest]
    return FixResult(
        fixes=fixes,
        unknown_ids=tuple(unknown),
        deleted=tuple(deleted),
        extra_fences=extra_fences,
    )


def _deleted_paths(
    raw: Mapping[str, Any], current_files: Sequence[str], errors: list[str]
) -> list[str]:
    """解析 fixes 围栏里的 ``{"deleted": […]}`` 删除声明。

    声明路径必须都在当前版本内；重复路径去重保序。错误并入调用方的同一份清单，
    重试一次就能拿到全部问题，少一轮往返。
    """
    declared = raw.get("deleted")
    if declared is None:
        return []
    if not isinstance(declared, list) or not all(isinstance(item, str) for item in declared):
        errors.append("deleted 必须是字符串数组（提出删除的版本内相对路径）")
        return []
    known = set(current_files)
    result: list[str] = []
    for item in declared:
        try:
            path = validate_version_path(item)
        except ContractViolation as exc:
            errors.extend(f"deleted 声明{error}" for error in exc.errors)
            continue
        if path not in known:
            errors.append(f"deleted 引用了版本外的路径：{path}")
            continue
        if path not in result:
            result.append(path)
    return result


def version_size_errors(files: Mapping[str, str]) -> list[str]:
    """版本量级上限：文件数 ≤ 20、单文件 ≤ 256 KB、版本总计 ≤ 1 MB。

    编码围栏产出与修复节点的写盘结果共用这一把尺子：两条产出路径的超限语义必须一致。
    """
    errors: list[str] = []
    oversize = [
        path for path, content in files.items() if len(content.encode("utf-8")) > MAX_FILE_BYTES
    ]
    if oversize:
        errors.append(
            f"以下文件超过单文件上限（{MAX_FILE_BYTES // 1024} KB）：{', '.join(oversize)}"
        )
    if len(files) > MAX_VERSION_FILES:
        errors.append(f"版本文件数超过上限（{MAX_VERSION_FILES} 个），实际 {len(files)} 个")
    total_bytes = sum(len(content.encode("utf-8")) for content in files.values())
    if total_bytes > MAX_VERSION_BYTES:
        errors.append(
            f"版本总大小超过上限（{MAX_VERSION_BYTES // 1024} KB），实际 {total_bytes // 1024} KB"
        )
    return errors


def judge_clean(findings: Sequence[Mapping[str, Any]], *, severe_only: bool = False) -> bool:
    """干净判定：默认【严重】与【一般】都阻塞；``severe_only`` 放宽为只阻塞【严重】。"""
    return blocking_count(findings, severe_only=severe_only) == 0


#: severity 的严重度次序（越小越严重）
_SEVERITY_RANK: dict[str, int] = {SEVERE: 0, MODERATE: 1, ADVICE: 2}


def _severity_name(value: Any) -> str:
    """归一 severity 取值：枚举外的取值（含缺失 / 拼写错）一律按【建议】处理。

    ``review_N.json`` 可能被手改，合并结果又直接进干净判定——把未知取值留在结果里会让
    :func:`blocking_count` 视而不见（既不阻塞也不提示），等于给出一条"看不见的结论"。
    """
    text = str(value)
    return text if text in SEVERITIES else ADVICE


def _severity_rank(value: Any) -> int:
    """严重度次序（归一后查表，保证入表与比较两处用同一把尺子）。"""
    return _SEVERITY_RANK[_severity_name(value)]


def merge_findings(
    groups: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
) -> list[Finding]:
    """并行双视角审查的 findings 汇合规则。

    - 去重键 = ``(location, issue)``：同一问题被两个视角都报出时只留一条；
    - 取**较严重**者：``severity`` 按 严重 < 一般 < 建议 的次序取更严重的一个，随它一起
      采纳该项的 ``suggestion`` 与 ``uncertain``（更严重的结论通常基于更强的证据）；
    - ``id`` 统一重编号为 ``f-1``…（保持首次出现顺序），使下游"逐条处置"仍可追溯；
    - 来源视角并入该条的 ``views``（附加字段，供报告与本产品自身审计"这条是谁提的"）。

    Args:
        groups: ``(视角显示名, findings)`` 序列，按给定顺序参与去重与统计。

    Returns:
        合并后的 findings（新对象，不改动入参）。
    """
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for view, findings in groups:
        for item in findings:
            key = (str(item.get("location") or ""), str(item.get("issue") or ""))
            entry = merged.get(key)
            if entry is None:
                merged[key] = {
                    # 首次入表也走同一套归一：枚举外的 severity 一律按【建议】处理，
                    # 否则它会原样落进结果、在 blocking_count 里"既不计严重也不计一般"
                    # 而静默绕过干净判定——判据只能看枚举内取值
                    "severity": _severity_name(item.get("severity")),
                    "location": key[0],
                    "issue": key[1],
                    "suggestion": str(item.get("suggestion") or ""),
                    "uncertain": bool(item.get("uncertain")),
                    "views": [view],
                }
                continue
            if _severity_rank(item.get("severity")) < _severity_rank(entry.get("severity")):
                entry["severity"] = _severity_name(item.get("severity"))
                entry["suggestion"] = str(item.get("suggestion") or "") or entry["suggestion"]
                entry["uncertain"] = bool(item.get("uncertain"))
            if view not in entry["views"]:
                entry["views"].append(view)
    result: list[Finding] = []
    for index, entry in enumerate(merged.values(), start=1):
        finding = Finding(**{str(key): value for key, value in entry.items()})
        finding["id"] = f"f-{index}"
        result.append(finding)
    return result


def finding_keys(findings: Sequence[Mapping[str, Any]]) -> frozenset[tuple[str, str]]:
    """findings 的 ``(location, issue)`` 集合（未收敛判定用）。

    非 Mapping 的条目（损坏记录 / 手改过的 review_N.json）跳过——判定读的是展示数据，
    不该因一条残缺条目整体失败。
    """
    return frozenset(
        (str(item.get("location") or ""), str(item.get("issue") or ""))
        for item in findings
        if isinstance(item, Mapping)
    )


# ---- crw review 的报告契约（files[] 形态）----


def review_report_errors(data: Any) -> list[str]:
    """``crw review --json`` 报告的 schema 校验；返回错误清单（空 = 通过）。

    报告形态固定为顶层 ``{"files": [{file, findings, summary}], "summary"}``，
    消费方（终端展示与脚本）只按这一种结构读。**会就地补全** ``uncertain``
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


def blocking_count(findings: Sequence[Mapping[str, Any]], *, severe_only: bool = False) -> int:
    """阻塞性结论条数（默认【严重】+【一般】；``severe_only`` 放宽为只数【严重】）。

    与 :func:`judge_clean` 同源：达限残留的文案与判定口径必须一致——默认判据下
    残留可能全部由【一般】构成，报告与渲染不得再写死"【严重】"。
    """
    blocking = (SEVERE,) if severe_only else (SEVERE, MODERATE)
    return sum(
        1 for item in findings if isinstance(item, Mapping) and item.get("severity") in blocking
    )


def severity_counts(findings: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """逐级计数（报告与阶段标题用），恒定键序：严重 / 一般 / 建议。

    非 Mapping 条目（手改过的 review_N.json）跳过——计数读的是展示数据，
    不得因为一条残缺条目把列表页 / 详情页整体打崩（与 finding_keys 同口径）。
    """
    return {
        level: sum(
            1 for item in findings if isinstance(item, Mapping) and item.get("severity") == level
        )
        for level in SEVERITIES
    }


def residual_severe(findings: Sequence[Mapping[str, Any]]) -> int:
    """【严重】问题数（runs 列表的"残留严重数"列）。"""
    return severity_counts(findings)[SEVERE]


def manifest_payload(*, version: int, files: Mapping[str, str]) -> dict[str, Any]:
    """版本 manifest 的载荷：文件清单 + sha256 + 行数。

    "本轮未产生改动"的整版本比较即对比两份 manifest 的 ``路径 → sha256`` 映射；
    行数供报告与展示层直接取用。路径按字典序排列，保证同内容同清单。
    """
    entries = [
        {
            "path": path,
            "sha256": hashlib.sha256(files[path].encode("utf-8")).hexdigest(),
            "lines": len(files[path].splitlines()),
        }
        for path in sorted(files)
    ]
    return {
        "version": version,
        "files": entries,
        "total_bytes": sum(len(files[path].encode("utf-8")) for path in sorted(files)),
    }


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
