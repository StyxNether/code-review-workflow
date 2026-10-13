"""运行状态 schema 与快照的纯序列化 / 反序列化。

字段定义见 Design §4.3；本模块只做数据与序列化，文件 IO 归 records、时序归 engine（Design §3.1）。
"""

from __future__ import annotations

import json
import operator
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, TypedDict

# ---- 取值枚举（Design §4.3 / §8.1）----

STATUS_RUNNING = "running"
STATUS_SUCCESS = "success"
STATUS_RESIDUAL = "residual"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"

NEXT_DONE = "done"
NEXT_REVIEW_AGAIN = "review_again"
NEXT_FIX_AGAIN = "fix_again"

INPUT_REQUIREMENT = "requirement"
INPUT_FILE = "file"

ROUNDS_UP_TO = "up-to"
ROUNDS_EXACT = "exact"

#: 轮次取值来源（Design §4.3 / §4.4）
SOURCE_CLI = "cli"
SOURCE_REQUIREMENT = "requirement"
SOURCE_CONFIG = "config"
SOURCE_PARENT = "parent"
SOURCE_DEFAULT = "default"


class ErrorInfo(TypedDict, total=False):
    """运行失败信息（Design §4.3 的 ``error`` 字段）。"""

    node: str
    message: str


class InputDescriptor(TypedDict, total=False):
    """输入描述：需求文本或既有代码文件（Design §4.3 的 ``input`` 字段）。"""

    kind: str
    requirement: str
    source_path: str


class Finding(TypedDict, total=False):
    """单条审查问题（Design §4.3 / §5.2）。

    ``views`` 是并行双视角审查的附加字段（来源视角，Design §12.2）：契约校验层不认识它
    （模型输出不含该键），汇合节点写入、报告与 `crw show` 读它。
    """

    id: str
    severity: str
    location: str
    issue: str
    suggestion: str
    uncertain: bool
    views: list[str]


class NodeEvent(TypedDict, total=False):
    """节点执行事件（Design §4.3 的 ``history`` 元素）。"""

    node: str
    round: int
    started_at: str
    duration_ms: int
    summary: str
    artifacts: list[str]


class RunState(TypedDict, total=False):
    """单轮运行状态（Design §4.3）；``history`` 追加累积，其余字段按轮替换。

    以 ``total=False`` 声明全部可选：状态在运行过程中逐字段填充，缺席字段按
    ``_DEFAULTS`` 归一（见 :func:`snapshot`）。
    """

    run_id: str
    parent_run_id: str | None
    input: InputDescriptor
    workspace: str
    review_round: int
    fix_rounds: int
    max_iterations: int
    max_iterations_source: str
    rounds_mode: str
    parallel_review: bool
    session_id: str | None
    round_index: int | None
    current_version_dir: str
    #: 当前版本内的文件清单（版本内相对路径，posix 风格，与 manifest 一致）：
    # 审查 / 修复节点的任务提示词需要它（节点不接触文件，Design §3.1）
    version_files: list[str]
    findings: list[Finding]
    #: 并行双视角审查的分支结论（独立 key：同超级步多写者不得冲突，Design §12.2）。
    #: 汇合节点消费后由 ``findings`` 承载权威结论，故不进入运行记录（见 PERSISTED_FIELDS）。
    findings_correctness: list[Finding]
    findings_security: list[Finding]
    clean: bool
    strict: bool
    history: Annotated[list[NodeEvent], operator.add]
    status: str
    next_action: str | None
    error: ErrorInfo | None
    model: str
    created_at: str
    finished_at: str | None


#: 持久化字段与顺序 = Design §4.3；新增状态字段时同步本表与 Design（§12.2 第 1 步）。
PERSISTED_FIELDS: tuple[str, ...] = (
    "run_id",
    "parent_run_id",
    "input",
    "workspace",
    "review_round",
    "fix_rounds",
    "max_iterations",
    "max_iterations_source",
    "rounds_mode",
    "parallel_review",
    "session_id",
    "round_index",
    "current_version_dir",
    "version_files",
    "findings",
    "clean",
    "strict",
    "history",
    "status",
    "next_action",
    "error",
    "model",
    "created_at",
    "finished_at",
)

# 说明：并行双视角的分支 key（``findings_correctness`` / ``findings_security``）刻意**不进**
# 本表与 ``PERSISTED_FIELDS``——它们是图执行期的中间态，汇合后由 ``findings`` 承载权威结论
# （Design §12.2），归一化重建的状态不含这两键，读取方按"缺省即空"处理（``state.get(k) or []``）。
_DEFAULTS: dict[str, Any] = {
    "run_id": "",
    "parent_run_id": None,
    "input": {},
    "workspace": "",
    "review_round": 0,
    "fix_rounds": 0,
    "max_iterations": 0,
    "max_iterations_source": SOURCE_DEFAULT,
    "rounds_mode": ROUNDS_UP_TO,
    "parallel_review": False,
    "session_id": None,
    "round_index": None,
    "current_version_dir": "",
    "version_files": [],
    "findings": [],
    "clean": False,
    "strict": False,
    "history": [],
    "status": STATUS_RUNNING,
    "next_action": None,
    "error": None,
    "model": "",
    "created_at": "",
    "finished_at": None,
}


def code_version_dir_name(version: int) -> str:
    """代码版本目录名：``code_v<version>``（版本 = 目录 + manifest.json，Design §4.2.2）。"""
    return f"code_v{version}"


def review_file_name(round_number: int, *, view: str | None = None) -> str:
    """审查结论文件名：``review_<轮次>.json``（并行双视角的分支加视角后缀，Design §12.2）。"""
    if view:
        return f"review_{round_number}_{view}.json"
    return f"review_{round_number}.json"


def fixes_file_name(round_number: int) -> str:
    """修复处置文件名：``fixes_<轮次>.json``（与 ``review_<轮次>.json`` 配对）。"""
    return f"fixes_{round_number}.json"


def round_number_for_review(state: Mapping[str, Any]) -> int:
    """本轮审查序号（= 已完成审查数 + 1，1-based，Design §4.3）。"""
    return int(state.get("review_round", 0)) + 1


def round_number_for_fix(state: Mapping[str, Any]) -> int:
    """本轮修复序号（= 已完成修复数 + 1，1-based，Design §4.3）。"""
    return int(state.get("fix_rounds", 0)) + 1


def code_version(state: Mapping[str, Any]) -> int:
    """当前代码版本号 = 已完成修复轮数 + 1（Design §4.4：审查恒有 review_round == fix_rounds + 1）。"""
    return int(state.get("fix_rounds", 0)) + 1


def new_state(
    *,
    run_id: str,
    input_descriptor: InputDescriptor,
    workspace: Path,
    max_iterations: int,
    max_iterations_source: str,
    model: str,
    created_at: str,
    rounds_mode: str = ROUNDS_UP_TO,
    strict: bool = False,
    parallel_review: bool = False,
    parent_run_id: str | None = None,
    session_id: str | None = None,
    round_index: int | None = None,
) -> RunState:
    """构造运行初始状态（字段与默认值见 Design §4.3）。

    Args:
        run_id: 运行号（全局唯一，会话轮次亦复用同一编号空间）。
        input_descriptor: 输入描述（requirement / file）。
        workspace: 运行工作区绝对路径（产物落盘根）。
        max_iterations: 修复轮次上限；``max_iterations_source`` 记录其来源。
        model: 本次运行的模型标识（``供应商/模型名``，Design §4.3）。
        created_at: 开始时间戳（ISO 8601，由注入时钟给出）。
        rounds_mode: 轮次语义（M1 恒为 up-to，exact 随 M4 落地）。
        strict: 是否启用加严干净判定（Design §4.4）。
        parallel_review: 是否启用并行双视角审查（Design §12.2 的扩展示例）。
        parent_run_id: 续跑来源运行号（resume 时写入，新运行为空）。
        session_id: 所属会话号（会话轮次写入；一次性运行为空）。
        round_index: 会话内轮次序号（从 1 起；一次性运行为空）。

    Returns:
        完整的运行初始状态字典。
    """
    return RunState(
        run_id=run_id,
        parent_run_id=parent_run_id,
        input=dict(input_descriptor),
        workspace=str(workspace),
        review_round=0,
        fix_rounds=0,
        max_iterations=max_iterations,
        max_iterations_source=max_iterations_source,
        rounds_mode=rounds_mode,
        parallel_review=parallel_review,
        session_id=session_id,
        round_index=round_index,
        current_version_dir="",
        version_files=[],
        findings=[],
        findings_correctness=[],
        findings_security=[],
        clean=False,
        strict=strict,
        history=[],
        status=STATUS_RUNNING,
        next_action=None,
        error=None,
        model=model,
        created_at=created_at,
        finished_at=None,
    )


def snapshot(state: Mapping[str, Any]) -> dict[str, Any]:
    """把图执行上下文投影为可持久化的状态快照（Design §4.3）。

    只保留 ``PERSISTED_FIELDS`` 声明的字段：节点交回的暂存键（如 ``code_text``）与
    未来扩展键不进入运行记录；缺席字段补默认值，保证 state.json 恒为完整快照。
    """
    return {name: _jsonable(state.get(name, _DEFAULTS[name])) for name in PERSISTED_FIELDS}


def dumps(state: Mapping[str, Any]) -> str:
    """状态快照 → JSON 文本（中文不转义，缩进 2）。"""
    return json.dumps(snapshot(state), ensure_ascii=False, indent=2)


def loads(text: str) -> dict[str, Any]:
    """JSON 文本 → 状态快照；顶层非对象抛 ``ValueError``。

    与 ``json.loads`` 的失败口径同族（``JSONDecodeError`` 也是 ``ValueError``）：调用方
    按"内容/格式不合法"统一处理，不需区分类型错误与解析错误。
    """
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("state.json 顶层必须是对象。")  # noqa: TRY004 — 内容格式错误，非参数类型错误
    return raw


def summarize_input(input_descriptor: Mapping[str, Any], *, limit: int = 40) -> str:
    """输入摘要：供运行列表、报告与阶段标题共用（超长截断）。"""
    kind = input_descriptor.get("kind")
    if kind == INPUT_FILE:
        source = str(input_descriptor.get("source_path") or "")
        name = Path(source).name or source or "（未命名文件）"
        return _ellipsis(f"文件：{name}", limit)
    requirement = " ".join(str(input_descriptor.get("requirement") or "").split())
    return _ellipsis(f"需求：{requirement}" if requirement else "（空需求）", limit)


def _ellipsis(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def _jsonable(value: Any) -> Any:
    """Path → str 归一：状态字段必须可 JSON 序列化（Design §8.2）。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value
