"""运行记录文件 IO：原子写、列表、加载、选择与结构自检。

原子写采用临时文件 + 替换；列表 / 加载忽略临时文件与非目录项，并对损坏记录降级——
坏记录必须仍能列出、仍能给出可读标识，而不是让列表页整体报错。
"""

from __future__ import annotations

import contextlib
import datetime
import json
import os
import re
import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import rounds as rounds_mod
from ..config import data_dir, sessions_root
from . import contracts
from . import state as state_mod

#: 运行号白名单：``r-YYYYMMDD-`` + 4 位小写十六进制。
#: 运行号直接拼进路径，白名单（而非仅剥离分隔符）使任何合法编号都只能是单个路径成分。
RUN_ID_PATTERN = re.compile(r"^r-[0-9]{8}-[0-9a-f]{4}$")

#: 会话号与轮次目录名白名单：本层只扫这两级目录，同样走白名单防穿越。
#: 会话号格式与 ``session.state`` 同源，此处独立声明——扫描方必须自己校验看到的名字，
#: 不依赖上游生成器"应该只产出合法名"。
SESSION_ID_PATTERN = re.compile(r"^s-[0-9]{8}-[0-9a-f]{4}$")
ROUND_DIR_PATTERN = re.compile(r"^round-[0-9]+$")

#: 降级记录的展示引用形式：``<会话号>/round-<i>``（state.json 不可读时没有运行号可用）
_SESSION_ROUND_REF = re.compile(r"^(s-[0-9]{8}-[0-9a-f]{4})/(round-[0-9]+)$")

#: 序号选择输入：仅 ASCII 十进制正整数（``0`` / 负数 / ``＋1`` 等一律视为取消）。
_INDEX_PATTERN = re.compile(r"^[1-9][0-9]*$")

_REVIEW_NAME_PATTERN = re.compile(r"^review_([0-9]+)\.json$")

#: 旧单文件格式的产物名（版本形态已是目录；该形态只用于识别并提示"记录来自旧格式"）
_LEGACY_CODE_NAME_PATTERN = re.compile(r"^code_v([0-9]+)\.py$")

#: 事件 detail 只存截断预览（日志是可读线索，不是产物副本）
DETAIL_PREVIEW_CHARS = 400

#: 陈旧 running 判定阈值：判据 = max(mtime(state.json), mtime(log.jsonl)) 超过 30 分钟
STALE_RUNNING_SECONDS = 30 * 60

_RUN_ID_ATTEMPTS = 8


class RunRecordError(RuntimeError):
    """运行记录不可用（名称非法 / 不存在 / 结构损坏），消息为可读中文。"""


@dataclass(frozen=True)
class RunSummary:
    """运行列表的一行（含降级派生结果）。"""

    run_id: str
    created_at: str
    input_summary: str
    status: str
    fix_rounds: int | None
    residual_severe: int | None
    path: Path
    stale: bool = False
    """``status=running`` 且长时间无更新：可能已中断（只标注，绝不自动改状态）。"""
    model: str = ""
    """本次运行的模型标识（``供应商/模型名（思考：…）``口径）。"""
    session_id: str = ""
    """所属会话号（会话轮次非空，供 runs 列表的「会话」列）。"""


@dataclass(frozen=True)
class RoundReport:
    """一轮审查与紧随其后的修复处置（报告与 `crw show` 的逐轮数据）。"""

    round_number: int
    version_label: str
    """本轮审查基于的版本目录显示名（如 ``code_v1/``）。"""
    files: list[str]
    """本轮审查覆盖的版本内文件清单（来自 review 契约顶层 ``files``）。"""
    review: Mapping[str, Any]
    fixes: Mapping[str, Any] | None = field(default=None)


@dataclass(frozen=True)
class RunDetail:
    """单条运行记录的完整视图。

    ``degraded`` 为真表示 ``state.json`` 不可读，摘要字段取自日志与产物清单；
    此时 ``state`` 为空映射，逐轮审查表仍来自 ``workspace/review_*.json``。
    """

    summary: RunSummary
    state: Mapping[str, Any]
    events: tuple[Mapping[str, Any], ...]
    rounds: tuple[RoundReport, ...]
    artifacts: tuple[Path, ...]
    degraded: bool = False
    degraded_reason: str = ""


def runs_root() -> Path:
    """运行记录根目录：``data_dir()/runs``（数据目录经 ``CRW_HOME`` 解析）。"""
    return data_dir() / "runs"


def is_valid_run_id(run_id: str) -> bool:
    """运行号是否合法（白名单：格式 + 仅单一路径成分，防路径穿越）。"""
    return bool(RUN_ID_PATTERN.fullmatch(run_id))


def parse_session_round_ref(ref: str) -> tuple[str, str] | None:
    """``<会话号>/round-<i>`` → ``(会话号, 轮次目录名)``；不是该形态返回 ``None``。

    这是**降级记录**的引用形式：会话轮次目录按 ``round-<i>`` 命名，运行号存在
    ``state.json`` 里——状态快照不可读时没有运行号可用，展示层据此给一个仍可定位的标识。
    """
    match = _SESSION_ROUND_REF.fullmatch(ref)
    if match is None:
        return None
    return match.group(1), match.group(2)


def run_dir(run_id: str, *, parent: Path | None = None) -> Path:
    """运行号 → 运行目录路径；非法运行号抛 :class:`RunRecordError`。

    ``parent`` 用于会话轮次（记录落在 ``sessions/<会话号>/rounds/`` 下的独立目录里）；
    省略时取 ``runs/``。查找既有记录用 :func:`find_run_dir`（会跨两处扫描）。
    """
    if not is_valid_run_id(run_id):
        raise RunRecordError(
            f"运行号格式非法：{run_id!r}（应为 r-YYYYMMDD-xxxx，xxxx 为 4 位小写十六进制）"
        )
    return (parent if parent is not None else runs_root()) / run_id


def iter_session_round_dirs() -> list[Path]:
    """枚举会话轮次记录目录（``sessions/<会话号>/rounds/round-<i>/``）。

    会话目录与轮次目录都走白名单：本函数是"扫描方"，不依赖上游只产出合法名。
    不可读的目录跳过——坏记录不拖垮列表页。
    """
    found: list[Path] = []
    for session_path in _iter_dir(sessions_root()):
        if not session_path.is_dir() or not SESSION_ID_PATTERN.fullmatch(session_path.name):
            continue
        for round_path in _iter_dir(session_path / "rounds"):
            if round_path.is_dir() and ROUND_DIR_PATTERN.fullmatch(round_path.name):
                found.append(round_path)
    return found


def find_run_dir(run_id: str) -> Path | None:
    """按运行号在两处查找记录目录：``runs/<运行号>/`` 与 ``sessions/*/rounds/*/``。

    运行号全局唯一，但目录名不一定等于运行号（会话轮次目录按 ``round-<i>`` 命名），
    故会话侧按 ``state.json`` 里的 ``run_id`` 匹配。也接受降级引用
    ``<会话号>/round-<i>``（state.json 不可读的轮次记录）。找不到返回 ``None``。
    """
    ref = parse_session_round_ref(run_id)
    if ref is not None:
        candidate = sessions_root() / ref[0] / "rounds" / ref[1]
        return candidate if candidate.is_dir() else None
    if not is_valid_run_id(run_id):
        return None
    direct = runs_root() / run_id
    if direct.is_dir():
        return direct
    for path in iter_session_round_dirs():
        state = read_state(path)
        if isinstance(state, Mapping) and str(state.get("run_id") or "") == run_id:
            return path
    return None


def new_run_id(*, day: str, hex_factory: Callable[[], str] | None = None) -> str:
    """生成运行号 ``r-YYYYMMDD-xxxx``；与既有记录碰撞时重新抽取。

    碰撞检查覆盖两处记录（``runs/`` 与会话轮次）：运行号在全局唯一，跨两处重名会让
    ``crw show`` 无从分辨。

    Args:
        day: 日期部分（8 位数字，由调用方从注入时钟截取）。
        hex_factory: 4 位小写十六进制生成器；省略时用 ``secrets.token_hex(2)``（可注入以复现碰撞）。
    """
    generator = hex_factory or (lambda: secrets.token_hex(2))
    existing = _existing_run_ids()
    for _ in range(_RUN_ID_ATTEMPTS):
        candidate = f"r-{day}-{generator().lower()}"
        if not is_valid_run_id(candidate):
            raise RunRecordError(f"运行号生成器给出非法编号：{candidate!r}")
        if candidate not in existing:
            return candidate
    raise RunRecordError("运行号生成失败：连续碰撞次数过多，请检查系统随机源。")


def _existing_run_ids() -> set[str]:
    """两处记录里已占用的运行号集合（**一次**扫描，供候选判重）。

    ``state.json`` 不可读的降级轮次拿不到运行号，无法参与判重——它的运行号本就无从得知，
    只能靠"降级记录用 <会话号>/round-<i> 引用"的展示口径区分。
    """
    ids = {
        path.name for path in _iter_dir(runs_root()) if path.is_dir() and is_valid_run_id(path.name)
    }
    for path in iter_session_round_dirs():
        state = read_state(path)
        if isinstance(state, Mapping):
            value = str(state.get("run_id") or "")
            if value:
                ids.add(value)
    return ids


def create_run_dir(run_id: str, *, parent: Path | None = None) -> Path:
    """创建运行目录与工作区；目录已存在抛 ``FileExistsError``（编号碰撞由调用方重试）。"""
    return create_run_dir_at(run_dir(run_id, parent=parent))


def create_run_dir_at(path: Path) -> Path:
    """在指定目录创建一条运行记录（会话轮次目录按 ``round-<i>`` 命名）。

    目录已存在即抛 ``FileExistsError``：既有轮次记录不得被覆盖（运行号进不去目录名，
    重试换号解决不了这个问题，故由调用方按可读错误处理）。
    """
    path.mkdir(parents=True, exist_ok=False)
    (path / "workspace").mkdir()
    return path


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """原子写文本：临时文件 ``.<名称>.tmp`` + 替换（POSIX 尽力 0600）。

    写入失败（含中断）不留半写目标文件，临时文件尽力清理。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        descriptor = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding=encoding, newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def append_log_event(path: Path, event: Mapping[str, Any]) -> None:
    """追加一条事件到 ``log.jsonl``（逐事件 append + flush）。

    只落盘 schema 声明的键（``ts`` / ``node`` / ``event`` / ``iteration`` / ``detail``）：
    渲染层需要的附加信息（如运行号）不进入日志；``detail`` 只存截断预览。
    """
    record: dict[str, Any] = {
        "ts": event.get("ts"),
        "node": event.get("node"),
        "event": event.get("event"),
    }
    if event.get("iteration") is not None:
        record["iteration"] = event["iteration"]
    if event.get("detail") is not None:
        record["detail"] = _preview(event["detail"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def read_events(log_path: Path) -> list[dict[str, Any]]:
    """读取事件流；空行与损坏行（含最后一条写一半的行）跳过。"""
    try:
        text = log_path.read_text(encoding="utf-8")
    except OSError:
        return []
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue  # 写一半的行 / 损坏行：不影响其余事件
        if isinstance(item, dict):
            events.append(item)
    return events


def read_state(run_path: Path) -> dict[str, Any] | None:
    """读取 ``state.json``；缺失 / 损坏 / 顶层非对象返回 ``None``（调用方决定降级口径）。"""
    return _read_json_object(run_path / "state.json")


def read_rounds(run_path: Path) -> list[RoundReport]:
    """逐轮读取 ``review_N.json`` 与配对的 ``fixes_N.json``（报告与 `crw show` 的数据源）。"""
    workspace = run_path / "workspace"
    rounds: list[RoundReport] = []
    for path in _iter_dir(workspace):
        match = _REVIEW_NAME_PATTERN.fullmatch(path.name)
        if not match:
            continue
        review = _read_json_object(path)
        if review is None:
            continue
        round_number = int(match.group(1))
        fixes = _read_json_object(workspace / state_mod.fixes_file_name(round_number))
        version = review.get("code_version")
        version_number = int(version) if isinstance(version, int) else round_number
        files = review.get("files")
        rounds.append(
            RoundReport(
                round_number=round_number,
                version_label=f"{rounds_mod.code_version_dir_name(version_number)}/",
                files=[str(item) for item in files] if isinstance(files, list) else [],
                review=review,
                fixes=fixes,
            )
        )
    return sorted(rounds, key=lambda item: item.round_number)


def run_artifacts(run_path: Path) -> list[Path]:
    """运行目录内可见产物：顶层文件、工作区文件与版本目录内的全部文件。

    版本目录（``workspace/code_v*/``）逐文件展开——产物清单要能直接
    指到具体文件；其余子目录不展开。临时文件与隐藏文件一律忽略。
    """
    artifacts: list[Path] = []
    workspace = run_path / "workspace"
    for directory in (run_path, workspace):
        for path in _iter_dir(directory):
            if path.is_dir():
                if directory is workspace and rounds_mod.version_of_dir_name(path.name) is not None:
                    artifacts.extend(_version_dir_files(path))
                continue
            if not path.name.startswith(".") and not path.name.endswith(".tmp"):
                artifacts.append(path)
    return artifacts


def _version_dir_files(version_dir: Path) -> list[Path]:
    """版本目录内的文件（含子目录，posix 相对路径的层级原样保留）。

    ``__pycache__`` / ``*.pyc`` 是执行核实（run_python）落下的运行时副产物，
    不属于版本内容，不进产物清单。
    """
    files: list[Path] = []
    for path in sorted(version_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(version_dir)
        if path.name.startswith(".") or path.name.endswith(".tmp"):
            continue
        if path.suffix == ".pyc" or "__pycache__" in relative.parts:
            continue
        files.append(path)
    return files


def is_legacy_record(state: Mapping[str, Any]) -> bool:
    """是否旧单文件格式的记录（代码存于 ``current_code`` 字段、没有版本目录）。

    旧记录不做迁移：show / resume 遇到时给一行可读提示并按
    "记录不可用"退出，而不是崩溃或静默出错。
    """
    return "current_code" in state and "current_version_dir" not in state


def list_runs(*, root: Path | None = None) -> list[RunSummary]:
    """列出运行记录（按开始时间倒序）；非目录项 / 非法名称 / 临时文件一律忽略。

    默认跨两处聚合：``runs/`` 与 ``sessions/*/rounds/round-*/``；给出
    ``root`` 时只扫该目录（测试与"只看一次性运行"的调用方用）。
    单条记录损坏不拖垮列表页：缺失字段按降级派生链回退
    （修复轮数 → 工作区版本文件数；残留严重数 → 最新可解析审查结论；状态 → 日志末条事件）。
    """
    if root is not None:
        paths = [path for path in _iter_dir(root) if path.is_dir() and is_valid_run_id(path.name)]
    else:
        paths = [
            path for path in _iter_dir(runs_root()) if path.is_dir() and is_valid_run_id(path.name)
        ]
        paths.extend(iter_session_round_dirs())
    summaries = [_summarize(path) for path in paths]
    summaries.sort(key=lambda item: (item.created_at, item.run_id), reverse=True)
    return summaries


def is_valid_run_ref(run_id: str) -> bool:
    """是否是可定位的记录引用：运行号本身，或降级形态的 ``<会话号>/round-<i>``。"""
    return is_valid_run_id(run_id) or parse_session_round_ref(run_id) is not None


def load_detail(run_id: str) -> RunDetail:
    """按运行号加载完整记录；引用非法 / 记录不存在抛 :class:`RunRecordError`。

    查找跨两处记录（``runs/`` 与会话轮次）。``state.json`` 损坏时**降级
    返回**：摘要与时间线改由 ``log.jsonl`` 与产物清单派生，``degraded``
    置真并附可读原因——记录坏了不代表这一条运行不可查，命令面据此提示并按"记录损坏"
    给非零退出码。
    """
    if not is_valid_run_ref(run_id):
        raise RunRecordError(
            f"运行号格式非法：{run_id!r}（应为 r-YYYYMMDD-xxxx，xxxx 为 4 位小写十六进制；"
            "会话轮次记录可用 <会话号>/round-<i> 引用）"
        )
    path = find_run_dir(run_id)
    if path is None:
        raise RunRecordError(f"运行记录不存在：{run_id}（可用 crw runs 查看现有记录）")
    state = read_state(path)
    summary = _summarize(path)
    if state is None:
        return RunDetail(
            summary=summary,
            state={},
            events=tuple(read_events(path / "log.jsonl")),
            rounds=tuple(read_rounds(path)),
            artifacts=tuple(run_artifacts(path)),
            degraded=True,
            degraded_reason=(
                f"{path / 'state.json'} 不可读或不是 JSON 对象（摘要取自 log.jsonl 与产物清单）"
            ),
        )
    return RunDetail(
        summary=summary,
        state=state,
        events=tuple(read_events(path / "log.jsonl")),
        rounds=tuple(read_rounds(path)),
        artifacts=tuple(run_artifacts(path)),
    )


def parse_index(raw: str | None, count: int) -> int | None:
    """序号输入 → 0-based 下标；空 / 取消 / 越界（含 0 与负数）返回 ``None``。

    显式拒绝 0 与负数（``"0"`` 经 ``int(raw) - 1`` 会命中 Python 负索引取到最后一
    项）；匹配用 ASCII 白名单而非 ``isdigit()``——上标数字等 ``isdigit()`` 为真而
    ``int()`` 抛错，两者接受集不一致。
    """
    if raw is None or not _INDEX_PATTERN.fullmatch(raw) or not 1 <= int(raw) <= count:
        return None
    return int(raw) - 1


def select_run(
    runs: Sequence[RunSummary],
    *,
    ask: Callable[[str], str | None],
    show: Callable[[str], None],
) -> RunSummary | None:
    """列表 + 数字选择；回车 / 取消 / 非法序号返回 ``None``。

    Args:
        runs: 候选运行记录（调用方已排序）。
        ask: 读取用户输入（返回 ``None`` 表示取消 / EOF / 中断）；由命令面提供。
        show: 输出一行候选或提示；由命令面提供。
    """
    for index, run in enumerate(runs, start=1):
        show(f"  {index}. {run.run_id}  {run.created_at}  [{run.status}]  {run.input_summary}")
    raw = ask("选择运行号序号（回车取消）：")
    index = parse_index(raw, len(runs))
    return None if index is None else runs[index]


def is_stale_running(
    state: Mapping[str, Any] | None, run_path: Path, *, now: float | None = None
) -> bool:
    """陈旧 ``running`` 判定：判据 = ``max(mtime(state.json), mtime(log.jsonl))``
    超过 30 分钟。只用于展示标注，绝不自动改状态。"""
    if not state or str(state.get("status") or "") != state_mod.STATUS_RUNNING:
        return False
    stamps: list[float] = []
    for name in ("state.json", "log.jsonl"):
        with contextlib.suppress(OSError):
            stamps.append((run_path / name).stat().st_mtime)
    if not stamps:
        return False
    reference = time.time() if now is None else now
    return reference - max(stamps) > STALE_RUNNING_SECONDS


def _summarize(path: Path) -> RunSummary:
    """单条运行的摘要（含降级派生）；不抛异常——坏记录也要能在列表页出现。

    会话轮次的运行号取自 ``state.json``（目录名是 ``round-<i>``）；状态快照不可读时
    用 ``<会话号>/round-<i>`` 作展示标识（仍可被 ``crw show`` 定位）。
    """
    state = read_state(path)
    session_id = _state_str(state, "session_id") or _session_of(path)
    created_at = _state_str(state, "created_at") or _mtime_text(path)
    input_descriptor = state.get("input") if state else None
    input_summary = (
        state_mod.summarize_input(input_descriptor)
        if isinstance(input_descriptor, Mapping)
        else "（记录损坏）"
    )
    status = _state_str(state, "status") or _infer_status(path)
    fix_rounds = _state_int(state, "fix_rounds")
    if fix_rounds is None:
        fix_rounds = _count_code_versions(path)
    residual: int | None = None
    findings = state.get("findings") if state else None
    if isinstance(findings, list):
        residual = contracts.residual_severe(findings)
    else:
        residual = _latest_review_severe(path)
    run_id = _display_run_id(path, state, session_id)
    return RunSummary(
        run_id=run_id,
        created_at=created_at,
        input_summary=input_summary,
        status=status,
        fix_rounds=fix_rounds,
        residual_severe=residual,
        path=path,
        stale=is_stale_running(state, path),
        model=_state_str(state, "model"),
        session_id=session_id,
    )


def _display_run_id(path: Path, state: Mapping[str, Any] | None, session_id: str) -> str:
    """列表 / 详情的展示标识（降级时仍给出可定位的引用）。

    正常记录取 ``state.json`` 里的运行号；会话轮次目录名是 ``round-<i>``，状态快照不可读时
    用 ``<会话号>/round-<i>`` 作标识（``crw show`` 也能按它定位）；一次性运行的目录名就是
    运行号。
    """
    run_id = _state_str(state, "run_id")
    if run_id:
        return run_id
    if is_valid_run_id(path.name):
        return path.name
    if session_id:
        return f"{session_id}/{path.name}"
    return path.name


def _session_of(path: Path) -> str:
    """记录目录所属的会话号（``.../sessions/<会话号>/rounds/round-<i>``）；非会话记录返回空串。"""
    if path.parent.name == "rounds" and SESSION_ID_PATTERN.fullmatch(path.parent.parent.name):
        return path.parent.parent.name
    return ""


def _infer_status(path: Path) -> str:
    """状态派生（state.json 不可读时）：日志末条事件推断，推断不出记 ``unknown``。"""
    known = (
        state_mod.STATUS_RUNNING,
        state_mod.STATUS_SUCCESS,
        state_mod.STATUS_RESIDUAL,
        state_mod.STATUS_FAILED,
        state_mod.STATUS_INTERRUPTED,
    )
    for event in reversed(read_events(path / "log.jsonl")):
        kind = event.get("event")
        if kind == "end" and event.get("node") == "engine":
            detail = event.get("detail")
            return detail if isinstance(detail, str) and detail in known else "unknown"
        if kind in ("interrupt", "error"):
            return state_mod.STATUS_FAILED
    return "unknown"


def _count_code_versions(path: Path) -> int | None:
    """最大版本号 − 1（降级派生的修复轮数；版本 = ``workspace/code_v*/`` 目录）。

    版本号从 1 起，故结果 ≥ 0；``code_v0/`` 之类手工造的异常目录按 0 兜底，不产出负数。
    """
    versions = [
        version
        for entry in _iter_dir(path / "workspace")
        if (version := rounds_mod.version_of_dir_name(entry.name)) is not None
    ]
    return max(0, max(versions) - 1) if versions else None


def _latest_review_severe(path: Path) -> int | None:
    rounds = read_rounds(path)
    if not rounds:
        return None
    latest = max(rounds, key=lambda item: item.round_number)
    findings = latest.review.get("findings")
    return contracts.residual_severe(findings) if isinstance(findings, list) else None


def _iter_dir(path: Path) -> list[Path]:
    """目录内容（排序）；不可读 / 不存在时返回空表——列表页不因坏目录整体报错。"""
    try:
        return sorted(path.iterdir())
    except OSError:
        return []


def _state_str(state: Mapping[str, Any] | None, key: str) -> str:
    value = state.get(key) if state else None
    return value if isinstance(value, str) and value else ""


def _state_int(state: Mapping[str, Any] | None, key: str) -> int | None:
    value = state.get(key) if state else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _mtime_text(path: Path) -> str:
    """目录修改时间（created_at 缺失时的兜底，供列表排序与展示）。"""
    with contextlib.suppress(OSError):
        return (
            datetime.datetime.fromtimestamp(path.stat().st_mtime)
            .astimezone()
            .isoformat(timespec="seconds")
        )
    return ""


def _read_json_object(path: Path) -> dict[str, Any] | None:
    """读一个 JSON 对象；文件缺失 / 非 UTF-8 / 损坏 / 顶层非对象一律返回 ``None``。

    解析复用 ``state.loads``（解析归 state、文件 IO 归本模块），失败口径统一为
    ``ValueError``（``JSONDecodeError`` 亦属其子类）。
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        return state_mod.loads(text)
    except ValueError:
        return None


def _preview(value: Any) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            # detail 含不可 JSON 序列化对象（集合 / 自定义类）时回退 repr：
            # 日志预览不得因为内容形态把整次运行变成"log 写入失败"
            text = repr(value)
    if len(text) <= DETAIL_PREVIEW_CHARS:
        return text
    return f"{text[:DETAIL_PREVIEW_CHARS]}…（已截断）"
