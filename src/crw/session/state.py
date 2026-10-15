"""会话态：会话号与目录、跨轮的会话状态（轮次链 / 当前设置 / 记忆索引）与落盘。

会话态是**跨轮存活**的那一层：单轮运行状态仍由 ``StateGraph`` 托管、
engine 不感知会话态；本模块只做 schema、纯序列化/反序列化与 ``session.json`` 的文件 IO
（原子写复用 :mod:`crw.workflow.records` 的既有原语），时序（何时创建、何时落盘、何时收尾）
归 :mod:`crw.session.chat`。

会话目录::

    ~/.crw/sessions/<会话号>/
    ├── session.json         # 会话档案（本模块的读写对象）
    ├── session.md           # 会话总报告（chat 收尾时生成）
    ├── memory/<agent>.jsonl # 三 Agent 各自的追加式记忆（session.memory）
    └── rounds/round-<i>/    # 每轮 = 一条完整运行记录（与 crw run 同构）
"""

from __future__ import annotations

import json
import re
import secrets
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from ..config import data_dir
from ..workflow import records

#: 会话状态枚举：进行中 / 已结束（结束由会话收尾写入，用于 sessions 列表展示）
SESSION_STATUS_ACTIVE = "active"
SESSION_STATUS_ENDED = "ended"

#: 会话号白名单：``s-YYYYMMDD-`` + 4 位小写十六进制。
#: 与 ``records.SESSION_ID_PATTERN`` 同源但独立声明：本模块是生成 / 落盘方，
#: records 是扫描方，两侧各自校验自己看到的名字（防穿越靠白名单而不是上游自觉）。
SESSION_ID_PATTERN = re.compile(r"^s-[0-9]{8}-[0-9a-f]{4}$")

#: 用户消息在会话档案里的保存上限（档案是人读摘要，不存全文）
MESSAGE_PREVIEW_CHARS = 200

#: 会话档案的字段白名单：新增字段时同步本表（含下表的 rounds 字段）
_PERSISTED_FIELDS: tuple[str, ...] = (
    "session_id",
    "created_at",
    "updated_at",
    "status",
    "input",
    "settings",
    "model",
    "rounds",
    "memory",
    "current_version_dir",
    "current_version",
    "version_files",
)

#: 轮次记录条目的字段白名单
_ROUND_FIELDS: tuple[str, ...] = (
    "index",
    "run_id",
    "intent",
    "message",
    "created_at",
    "finished_at",
    "status",
    "conclusion",
    "rounds_mode",
    "max_iterations",
    "max_iterations_source",
    "note",
    "version",
    "review_round",
    "fix_rounds",
)


class SessionState(TypedDict, total=False):
    """会话状态（``session.json`` 的内容，即会话档案）。

    轮次链（``rounds``）是逐轮追加的记录摘要；``settings`` 是**当前轮次设置**（下一轮
    的默认值，取值优先级为命令行显式 > 消息提取 > 配置默认）；``memory`` 是记忆索引
    （agent → 相对路径），内容本体在 ``memory/<agent>.jsonl``（session.memory 负责）。
    """

    session_id: str
    created_at: str
    updated_at: str
    status: str
    input: dict[str, Any]
    settings: dict[str, Any]
    model: str
    rounds: list[dict[str, Any]]
    memory: dict[str, str]
    current_version_dir: str
    current_version: int
    version_files: list[str]


@dataclass(frozen=True)
class SessionSummary:
    """``crw sessions`` 的一行（含降级派生）。"""

    session_id: str
    created_at: str
    round_count: int
    last_status: str
    path: Path
    model: str = ""
    degraded: bool = False
    """``session.json`` 不可读（轮次数与状态改由轮次目录派生，列表页据此标"记录损坏"）。"""


def sessions_root() -> Path:
    """会话根目录：``data_dir()/sessions``（经 ``CRW_HOME`` 解析）。"""
    return data_dir() / "sessions"


def is_valid_session_id(session_id: str) -> bool:
    """会话号是否合法（白名单：格式 + 仅单一路径成分，防路径穿越）。"""
    return bool(SESSION_ID_PATTERN.fullmatch(session_id))


def session_dir(session_id: str) -> Path:
    """会话号 → 会话目录；非法会话号抛 ``ValueError``。"""
    if not is_valid_session_id(session_id):
        raise ValueError(f"会话号格式非法：{session_id!r}（应为 s-YYYYMMDD-xxxx）")
    return sessions_root() / session_id


def new_session_id(*, day: str, hex_factory: Callable[[], str] | None = None) -> str:
    """生成会话号 ``s-YYYYMMDD-xxxx``；与既有会话目录碰撞时重新抽取。

    Args:
        day: 日期部分（8 位数字，由调用方从注入时钟截取）。
        hex_factory: 4 位小写十六进制生成器；省略时用 ``secrets.token_hex(2)``（可注入）。
    """
    generator = hex_factory or (lambda: secrets.token_hex(2))
    for _ in range(8):
        candidate = f"s-{day}-{generator().lower()}"
        if not is_valid_session_id(candidate):
            raise ValueError(f"会话号生成器给出非法编号：{candidate!r}")
        if not (sessions_root() / candidate).exists():
            return candidate
    raise RuntimeError("会话号生成失败：连续碰撞次数过多，请检查系统随机源。")


def create_session_dir(session_id: str) -> Path:
    """创建会话目录与 ``memory/``、``rounds/`` 子目录；已存在抛 ``FileExistsError``。"""
    path = session_dir(session_id)
    path.mkdir(parents=True, exist_ok=False)
    (path / "memory").mkdir()
    (path / "rounds").mkdir()
    return path


def new_session_state(
    *,
    session_id: str,
    created_at: str,
    model: str,
    settings: Mapping[str, Any],
    input_descriptor: Mapping[str, Any] | None = None,
) -> SessionState:
    """构造会话初始状态（字段见 :class:`SessionState`）。

    ``settings`` 的取值来源由调用方（chat）按优先级链算好后传入；``input_descriptor``
    记会话的起始输入（``-f`` 起始时保留来源路径，便于 "原文件 → 工作区" 的追溯）。
    """
    return SessionState(
        session_id=session_id,
        created_at=created_at,
        updated_at=created_at,
        status=SESSION_STATUS_ACTIVE,
        input=dict(input_descriptor or {}),
        settings={str(key): value for key, value in settings.items()},
        model=model,
        rounds=[],
        memory={},
        current_version_dir="",
        current_version=0,
        version_files=[],
    )


def snapshot(state: Mapping[str, Any]) -> dict[str, Any]:
    """会话状态 → 可持久化的档案（字段白名单 + 轮次条目字段白名单两层过滤）。"""
    data: dict[str, Any] = {}
    for name in _PERSISTED_FIELDS:
        if name == "rounds":
            # 形态不符的条目直接丢弃：空条目落盘会让轮次数与"最后状态"的统计口径失真
            data[name] = [
                item
                for item in (_round_snapshot(entry) for entry in state.get("rounds") or [])
                if item
            ]
        elif name in state:
            data[name] = _jsonable(state[name])
    return data


def dumps(state: Mapping[str, Any]) -> str:
    """会话档案 → JSON 文本（中文不转义，缩进 2）。"""
    return json.dumps(snapshot(state), ensure_ascii=False, indent=2) + "\n"


def loads(text: str) -> dict[str, Any]:
    """JSON 文本 → 会话档案；顶层非对象抛 ``ValueError``（与 state.loads 同口径）。"""
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("session.json 顶层必须是对象。")  # noqa: TRY004 — 内容格式错误
    return raw


def session_file(session_id: str) -> Path:
    """会话档案路径（``sessions/<会话号>/session.json``）。"""
    return session_dir(session_id) / "session.json"


def read_session(session_dir_path: Path) -> dict[str, Any] | None:
    """读取**会话目录**下的 ``session.json``；缺失 / 损坏 / 顶层非对象返回 ``None``。

    入参是会话目录（不是文件路径）：档案文件名是目录布局的一部分，由本模块掌握
    （需要文件路径本身时用 :func:`session_file`）。失败返回 ``None`` 由调用方决定降级。
    """
    try:
        text = (session_dir_path / "session.json").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        return loads(text)
    except ValueError:
        return None


def write_session(state: Mapping[str, Any]) -> None:
    """原子写会话档案。

    异常契约：会话号非法抛 ``ValueError``（入参校验，与 :func:`session_dir` 同口径）；
    磁盘写入失败抛 ``OSError``（调用方按可读报错处理）。时间戳由调用方给出（注入时钟，可测），
    本函数只负责"把给到的状态原样落盘"。
    """
    session_id = str(state.get("session_id") or "")
    if not is_valid_session_id(session_id):
        raise ValueError(f"会话号非法，无法写入会话档案：{session_id!r}")
    records.atomic_write_text(session_file(session_id), dumps(state))


def list_sessions() -> list[SessionSummary]:
    """列出会话（按创建时间倒序）；非目录项与非法名称忽略。

    单条会话损坏不拖垮列表页：``session.json`` 不可读时轮次数改数轮次目录、
    状态标"（记录损坏）"。
    """
    root = sessions_root()
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return []
    summaries = [
        _summarize(path) for path in entries if path.is_dir() and is_valid_session_id(path.name)
    ]
    summaries.sort(key=lambda item: (item.created_at, item.session_id), reverse=True)
    return summaries


def _summarize(path: Path) -> SessionSummary:
    """单个会话的摘要（降级容忍）；不抛异常——坏会话也要能在列表页出现。

    "最后状态"取**最后一轮的运行状态**（会话自身只有进行中 / 已结束两态，对使用者没有
    信息量）；一轮都没有时退回会话状态。
    """
    state = read_session(path)
    if state is None:
        return SessionSummary(
            session_id=path.name,
            created_at="",
            round_count=len(_round_dirs(path)),
            last_status="（记录损坏）",
            path=path,
            degraded=True,
        )
    rounds = state.get("rounds")
    # 只统计非空对象条目：坏条目不该在列表页虚增轮次数或顶替"最后状态"
    round_list = (
        [item for item in rounds if isinstance(item, Mapping) and item]
        if isinstance(rounds, list)
        else []
    )
    last_round_status = str(round_list[-1].get("status") or "") if round_list else ""
    return SessionSummary(
        session_id=path.name,
        created_at=str(state.get("created_at") or ""),
        round_count=len(round_list),
        last_status=last_round_status or str(state.get("status") or ""),
        path=path,
        model=str(state.get("model") or ""),
    )


def _round_dirs(path: Path) -> list[Path]:
    """会话目录下的轮次记录目录（``rounds/round-<i>/``）；不可读返回空表。"""
    try:
        entries = sorted((path / "rounds").iterdir())
    except OSError:
        return []
    return [
        item for item in entries if item.is_dir() and records.ROUND_DIR_PATTERN.fullmatch(item.name)
    ]


def next_version(state: Mapping[str, Any]) -> int:
    """下一轮运行的起始版本号 = 当前版本 + 1（会话内跨轮连续递增）。"""
    value = state.get("current_version")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 1
    return value + 1


def summarize_message(text: str, *, limit: int = MESSAGE_PREVIEW_CHARS) -> str:
    """用户消息 → 档案里的单行摘要（折叠空白并限长）。"""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else f"{flat[: limit - 1]}…"


def record_round(
    state: SessionState,
    *,
    index: int,
    run_id: str,
    intent: str,
    message: str,
    created_at: str,
    finished_at: str,
    status: str,
    conclusion: str,
    rounds_mode: str,
    max_iterations: int,
    max_iterations_source: str,
    version: int | None,
    review_round: int,
    fix_rounds: int,
    note: str = "",
) -> dict[str, Any]:
    """把一轮的结果追加进轮次链并返回该条目（调用方随后落盘档案）。

    只记**结构化摘要**：用户消息预览、运行号、结论句与轮次计数——完整的运行详情在
    ``rounds/round-<i>/`` 的运行记录里（每轮一条，与 ``crw run`` 同构）；
    ``note`` 记本轮的标注（如消息轮次被按上限夹取）。
    """
    entry: dict[str, Any] = {
        "index": index,
        "run_id": run_id,
        "intent": intent,
        "message": summarize_message(message),
        "created_at": created_at,
        "finished_at": finished_at,
        "status": status,
        "conclusion": conclusion,
        "rounds_mode": rounds_mode,
        "max_iterations": max_iterations,
        "max_iterations_source": max_iterations_source,
        "version": version,
        "review_round": review_round,
        "fix_rounds": fix_rounds,
    }
    if note:
        entry["note"] = note
    rounds = state.setdefault("rounds", [])
    rounds.append(entry)
    return entry


def memory_index(paths: Mapping[str, str]) -> dict[str, str]:
    """记忆索引（agent → 相对会话目录的路径，统一 posix 分隔符）。"""
    return {str(agent): str(value).replace("\\", "/") for agent, value in paths.items()}


def _round_snapshot(entry: Any) -> dict[str, Any]:
    """轮次条目 → 白名单投影（外来键不落盘）。"""
    if not isinstance(entry, Mapping):
        return {}
    return {name: _jsonable(entry[name]) for name in _ROUND_FIELDS if name in entry}


def _jsonable(value: Any) -> Any:
    """Path → str 归一（与 workflow.state 同口径：档案必须可 JSON 序列化）。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


__all__ = [
    "MESSAGE_PREVIEW_CHARS",
    "SESSION_ID_PATTERN",
    "SESSION_STATUS_ACTIVE",
    "SESSION_STATUS_ENDED",
    "SessionState",
    "SessionSummary",
    "create_session_dir",
    "dumps",
    "is_valid_session_id",
    "list_sessions",
    "loads",
    "memory_index",
    "new_session_id",
    "new_session_state",
    "next_version",
    "read_session",
    "record_round",
    "session_dir",
    "session_file",
    "sessions_root",
    "snapshot",
    "summarize_message",
    "write_session",
]
