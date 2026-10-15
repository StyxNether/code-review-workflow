"""会话循环：读入用户消息、编排路由、驱动工作流、中断与退出语义、会话档案与会话总报告。

职责边界：本模块是会话层的**时序组织者**——它决定何时新建 / 恢复
会话、何时该跑一轮工作流、何时落盘会话档案与 ``session.md``；单轮运行的执行、事件与
记录仍全部交给引擎（每轮 = 一条与 ``crw run`` 同构的完整运行记录）。

关键口径：

- **每轮一条完整运行记录**：``sessions/<会话号>/rounds/round-<i>/``，``state.json`` 带
  ``session_id`` / ``round_index``；代码版本号跨轮连续递增。
- **中断**：一轮进行中 Ctrl+C = 引擎按 ``status=interrupted`` 收尾并回提示符（会话保留）；
  在提示符处 Ctrl+C / EOF / ``/exit`` 均为正常退出。
- **退出码**：最后一轮状态的映射（0 干净 / 1 达限残留 / 2 运行错误或被中断），
  一轮都没跑过为 0。
- **记忆**：三 Agent 各自的追加式记忆由本模块在每轮结束后按**结构化状态**追加摘要
  （不含代码与思考内容）；下一轮把它们作为历史消息交给对应 Agent，接近预算时显式压缩。
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import agents as agents_mod
from .. import config as config_mod
from .. import render
from .. import rounds as rounds_mod
from .. import tools as tools_mod
from ..workflow import engine as engine_mod
from ..workflow import records as records_mod
from ..workflow import state as state_mod
from . import memory as memory_mod
from . import orchestrator as orch_mod
from . import state as session_state
from .state import SessionState

_EXIT_OK = 0
_EXIT_ISSUES = 1
_EXIT_ERROR = 2

#: 退出命令（``/exit`` 是对外文档的定稿写法；其余为常见手滑写法）
EXIT_COMMANDS = frozenset({"/exit", "exit", "/quit", "quit"})

#: 会话节点名 → 记忆文件所属 Agent（三份独立记忆）
_AGENT_OF_NODE: Mapping[str, str] = {
    "encode": "encoder",
    "review": "reviewer",
    "review_correctness": "reviewer",
    "review_security": "reviewer",
    "merge_reviews": "reviewer",
    "fix": "fixer",
}

#: 版本目录名 → 版本号（会话内跨轮连续递增，目录名是权威来源）
_VERSION_DIR_RE = re.compile(r"^code_v([0-9]+)$")

#: 运行状态 → 退出码（``crw chat`` 的退出码即最后一轮的状态映射）
_STATUS_EXIT_CODES: Mapping[str, int] = {
    state_mod.STATUS_SUCCESS: _EXIT_OK,
    state_mod.STATUS_RESIDUAL: _EXIT_ISSUES,
}


@dataclass(frozen=True)
class RoundSettings:
    """一组轮次设置（会话的**当前设置**，随档案跨轮保留）。"""

    max_iterations: int
    max_iterations_source: str
    rounds_mode: str


@dataclass(frozen=True)
class RoundOverrides:
    """命令行**显式**给出的轮次覆盖，按字段生效。

    两个字段相互独立：只给 ``--rounds-mode`` 时钉住的是"轮次语义"，轮次**数值**仍按
    消息提取 / 会话当前设置 / 配置默认的优先级链取值——把模式当成"轮次数值的命令行来源"
    会让配置默认被 argparse 的默认值静默顶掉，并误标来源。
    """

    max_iterations: int | None = None
    rounds_mode: str | None = None


@dataclass
class ChatIO:
    """会话交互的可注入接缝（与 :class:`crw.uninstall.UninstallIO` 同形的注入惯例）。

    ``ask`` 读入一行（``None`` = EOF / Ctrl+C / 输入流不可用）；``say`` 走 stdout（会话头、
    汇报与结论），``warn`` 走 stderr（过程提示与可读报错）；``renderer`` 构造
    运行观察者（会话级渲染）。
    """

    ask: Callable[[str], str | None]
    say: Callable[[str], None]
    warn: Callable[[str], None]
    renderer: Callable[[], engine_mod.RunObserver]


@dataclass(frozen=True)
class RoundOutcome:
    """一轮会话运行的结果视图（chat 据此更新会话档案与渲染汇报）。"""

    run_id: str
    run_dir: Path | None
    state: Mapping[str, Any]
    exit_code: int
    report_written: bool
    interrupted: bool


def run_chat(
    *,
    io: ChatIO,
    resolved: config_mod.ResolvedModel | None,
    clock: Callable[[], str],
    new_session_id_factory: Callable[[], str] | None = None,
    run_id_factory: engine_mod.RunIdFactory | None = None,
    rounds_limit: int | None = None,
    initial_rounds: RoundSettings | None = None,
    overrides: RoundOverrides | None = None,
    file_input: tuple[Path, str] | None = None,
    resume_session_id: str | None = None,
    severe_only: bool = False,
    no_exec: bool = False,
    confirm_deletions: engine_mod.DeletionConfirmer | None = None,
) -> int:
    """会话主循环；返回退出码（最后一轮状态的映射）。

    Args:
        io: 交互与渲染接缝（命令面装配真实实现，测试注入脚本化替身）。
        resolved: 本次会话的模型连接（编排节点与三 Agent 共用；``None`` 走环境变量回退）。
        clock: 时钟（会话档案与引擎时间戳共用，可注入固定值）。
        new_session_id_factory: 会话号生成器；缺省按注入时钟的日期 + 碰撞重试。
        run_id_factory: 运行号生成器（透传给每轮引擎；缺省引擎自带实现）。
        rounds_limit: 轮次夹取上限（``settings.json`` 的 ``rounds.limit``；缺省取 10）。
        initial_rounds: 会话的初始轮次设置（新建会话时用；恢复会话时以档案里的为准）。
        overrides: 命令行显式给出的轮次覆盖（按字段生效，优先级最高）。
        file_input: ``-f`` 起始的输入（路径与内容；仅新建会话的第 1 轮使用）。
        resume_session_id: ``--resume`` 的会话号（恢复记忆、代码版本与轮次设置）。
        severe_only: 是否放宽干净判定（只把【严重】当阻塞；随档案保留，恢复时取"档案或本次
            命令行"的并集）。
        no_exec: 是否关闭代码执行（按轮传入，不进档案——无法区分"显式关闭"与"未设置"）。
        confirm_deletions: 删除确认闸门（与 ``crw run`` 同一闸门）。

    Returns:
        退出码：0 干净 / 1 达限残留 / 2 运行错误或被中断；未跑过任何轮次为 0。
    """
    limit = rounds_limit if rounds_limit is not None else config_mod.DEFAULT_ROUNDS_LIMIT
    try:
        state = _start_session(
            resume_session_id=resume_session_id,
            resolved=resolved,
            clock=clock,
            new_session_id_factory=new_session_id_factory or _default_session_id_factory(clock),
            initial_rounds=initial_rounds,
            file_input=file_input,
            severe_only=severe_only,
        )
    except (OSError, ValueError) as exc:
        io.warn(f"crw chat：会话无法建立 — {exc}")
        return _EXIT_ERROR
    _print_session_header(io, state, resume=bool(resume_session_id), limit=limit)
    orchestrator_observer = io.renderer()
    orchestrator_runtime = agents_mod.build_runtime(
        resolved=resolved,
        emit=orchestrator_observer.on_event,
        on_delta=orchestrator_observer.on_delta,
    )
    orchestrator_registry = tools_mod.build_registry([Path.cwd()], names=(), allow_exec=False)
    exit_code = _EXIT_OK
    try:
        exit_code = _conversation_loop(
            io=io,
            state=state,
            orchestrator_runtime=orchestrator_runtime,
            orchestrator_registry=orchestrator_registry,
            orchestrator_observer=orchestrator_observer,
            resolved=resolved,
            clock=clock,
            run_id_factory=run_id_factory,
            rounds_limit=limit,
            overrides=overrides,
            file_input=file_input,
            severe_only=severe_only,
            no_exec=no_exec,
            confirm_deletions=confirm_deletions,
        )
    finally:
        try:
            _finish_session(io, state=state, clock=clock)
        except Exception as exc:  # noqa: BLE001 — 收尾（会话档案 / 总报告）失败绝不覆盖会话退出码
            io.warn(f"crw chat：会话收尾失败 — {render.mask_secrets(str(exc))}（运行记录不受影响）")
    return exit_code


def _default_session_id_factory(clock: Callable[[], str]) -> Callable[[], str]:
    """默认会话号生成：按注入时钟的日期 + 碰撞重试（与运行号同一口径）。"""

    def factory() -> str:
        day = "".join(clock()[:10].split("-"))
        if len(day) != 8 or not day.isdigit():
            raise ValueError(f"时钟返回值无法解析出日期：{clock()!r}")
        return session_state.new_session_id(day=day)

    return factory


def _start_session(
    *,
    resume_session_id: str | None,
    resolved: config_mod.ResolvedModel | None,
    clock: Callable[[], str],
    new_session_id_factory: Callable[[], str],
    initial_rounds: RoundSettings | None,
    file_input: tuple[Path, str] | None,
    severe_only: bool,
) -> SessionState:
    """恢复既有会话或新建会话（返回会话状态；失败抛可读异常）。"""
    if resume_session_id:
        if not session_state.is_valid_session_id(resume_session_id):
            raise ValueError(
                f"会话号格式非法：{resume_session_id!r}（应为 s-YYYYMMDD-xxxx；"
                "可用 crw sessions 查看现有会话）"
            )
        path = session_state.session_dir(resume_session_id)
        if not path.is_dir():
            raise ValueError(f"会话不存在：{resume_session_id}（可用 crw sessions 查看现有会话）")
        loaded = session_state.read_session(path)
        if loaded is None:
            raise ValueError(
                f"会话 {resume_session_id} 的 session.json 不可读，无法恢复"
                "（会话目录仍在：可查看其中的 rounds/ 运行记录）"
            )
        archived_id = str(loaded.get("session_id") or "")
        if archived_id != resume_session_id or not session_state.is_valid_session_id(archived_id):
            raise ValueError(
                f"会话 {resume_session_id} 的 session.json 与会话号不符"
                f"（内容为 {archived_id or '（缺失）'}，可能被手工改动），无法恢复；"
                "可用 crw sessions 确认现有会话，或新建会话。"
            )
        state = SessionState(**{str(key): value for key, value in loaded.items()})
        state["status"] = session_state.SESSION_STATUS_ACTIVE
        state["updated_at"] = clock()
        if not state.get("memory"):
            state["memory"] = session_state.memory_index(
                {agent: f"memory/{agent}.jsonl" for agent in memory_mod.MEMORY_AGENTS}
            )
        return state
    session_id = new_session_id_factory()
    session_state.create_session_dir(session_id)
    rounds = initial_rounds or RoundSettings(
        max_iterations=config_mod.DEFAULT_ROUNDS_DEFAULT,
        max_iterations_source=state_mod.SOURCE_DEFAULT,
        rounds_mode=rounds_mod.ROUNDS_UP_TO,
    )
    descriptor: dict[str, Any] = {}
    if file_input is not None:
        descriptor = {"kind": state_mod.INPUT_FILE, "source_path": str(file_input[0])}
    state = session_state.new_session_state(
        session_id=session_id,
        created_at=clock(),
        model=resolved.label() if resolved is not None else engine_mod.UNRESOLVED_MODEL_LABEL,
        settings={
            "max_iterations": rounds.max_iterations,
            "max_iterations_source": rounds.max_iterations_source,
            "rounds_mode": rounds.rounds_mode,
            "severe_only": severe_only,
        },
        input_descriptor=descriptor,
    )
    state["memory"] = session_state.memory_index(
        {agent: f"memory/{agent}.jsonl" for agent in memory_mod.MEMORY_AGENTS}
    )
    session_state.write_session(state)
    return state


def _conversation_loop(
    *,
    io: ChatIO,
    state: SessionState,
    orchestrator_runtime: Any,
    orchestrator_registry: Any,
    orchestrator_observer: engine_mod.RunObserver,
    resolved: config_mod.ResolvedModel | None,
    clock: Callable[[], str],
    run_id_factory: engine_mod.RunIdFactory | None,
    rounds_limit: int,
    overrides: RoundOverrides | None,
    file_input: tuple[Path, str] | None,
    severe_only: bool,
    no_exec: bool,
    confirm_deletions: engine_mod.DeletionConfirmer | None,
) -> int:
    """逐条消息驱动会话；返回退出码（最后一轮状态映射）。"""
    exit_code = _EXIT_OK
    while True:
        message = io.ask("你> ")
        if message is None:
            break
        text = message.strip()
        if not text:
            continue
        if text.lower() in EXIT_COMMANDS:
            break
        try:
            handling = _handle_message(
                io=io,
                text=text,
                state=state,
                orchestrator_runtime=orchestrator_runtime,
                orchestrator_registry=orchestrator_registry,
                orchestrator_observer=orchestrator_observer,
                resolved=resolved,
                clock=clock,
                run_id_factory=run_id_factory,
                rounds_limit=rounds_limit,
                overrides=overrides,
                file_input=file_input,
                severe_only=severe_only,
                no_exec=no_exec,
                confirm_deletions=confirm_deletions,
            )
        except KeyboardInterrupt:
            # 编排节点调用等"引擎之外"的位置被 Ctrl+C：按"中断本条消息"处理（会话保留）
            io.warn("crw chat：已中断本条消息（会话保留；可继续说，或 /exit 退出）。")
            continue
        if handling is not None:
            exit_code = handling
    return exit_code


def _handle_message(
    *,
    io: ChatIO,
    text: str,
    state: SessionState,
    orchestrator_runtime: Any,
    orchestrator_registry: Any,
    orchestrator_observer: engine_mod.RunObserver,
    resolved: config_mod.ResolvedModel | None,
    clock: Callable[[], str],
    run_id_factory: engine_mod.RunIdFactory | None,
    rounds_limit: int,
    overrides: RoundOverrides | None,
    file_input: tuple[Path, str] | None,
    severe_only: bool,
    no_exec: bool,
    confirm_deletions: engine_mod.DeletionConfirmer | None,
) -> int | None:
    """处理一条用户消息：编排 → 轮次设置 → 运行一轮 → 汇报；返回新的退出码（无需更新为 ``None``）。"""
    decision = _interpret_message(
        io,
        orchestrator_runtime,
        orchestrator_registry,
        orchestrator_observer,
        text,
        state,
        rounds_limit=rounds_limit,
    )
    if decision is None:
        return None
    if decision.reply:
        io.say(f"[编排] {decision.reply}")
    if not decision.starts_workflow:
        if decision.note:
            io.warn(f"[编排] {decision.note}")
        return None
    settings, clamp_note = _resolve_rounds(
        decision=decision, state=state, overrides=overrides, limit=rounds_limit
    )
    if not _has_code_for(decision.intent, state, file_input):
        io.warn(
            "crw chat：当前会话还没有代码版本，无法只审查 / 继续修复——"
            "请先给一个开发需求（或新建会话时用 -f 指定既有代码文件）。"
        )
        return None
    index = _next_round_index(state)
    # 本轮标注：夹取事实与编排层的说明都留痕（去重后按序连接）
    round_note = "；".join(dict.fromkeys(text for text in (clamp_note, decision.note) if text))
    io.say("")
    io.say(
        f"── 会话第 {index} 轮："
        + orch_mod.decision_notice(
            decision,
            rounds=settings.max_iterations,
            source=settings.max_iterations_source,
            rounds_mode=settings.rounds_mode,
        )
        + f"{f'；{round_note}' if round_note else ''} "
        + "─" * 4
    )
    outcome = _run_round(
        io=io,
        state=state,
        decision=decision,
        settings=settings,
        message=text,
        note=round_note,
        resolved=resolved,
        clock=clock,
        run_id_factory=run_id_factory,
        file_input=file_input,
        severe_only=severe_only,
        no_exec=no_exec,
        confirm_deletions=confirm_deletions,
    )
    if outcome is None:
        return None
    _render_round_report(io, state=state, outcome=outcome, decision=decision)
    return outcome.exit_code


def _interpret_message(
    io: ChatIO,
    runtime: Any,
    registry: Any,
    observer: engine_mod.RunObserver,
    message: str,
    state: Mapping[str, Any],
    *,
    rounds_limit: int,
) -> orch_mod.Decision | None:
    """编排节点解释一条消息；模型调用失败给可读报错并保留会话（返回 ``None``）。"""
    context = orch_mod.session_context(
        rounds=[item for item in state.get("rounds") or [] if isinstance(item, Mapping)],
        current_version=_current_version(state),
        version_files=[str(item) for item in state.get("version_files") or []],
    )
    observer.on_event({"node": "orchestrator", "event": "start"})
    try:
        decision = orch_mod.interpret(
            runtime,
            registry,
            message=message,
            context=context,
            rounds_limit=rounds_limit,
            rounds_default=_session_rounds_default(state),
        )
    except Exception as exc:  # noqa: BLE001 — 编排失败不得终止会话（可读报错后回提示符）
        observer.on_event({"node": "orchestrator", "event": "error", "detail": str(exc)})
        io.warn(f"crw chat：编排节点调用失败 — {render.mask_secrets(str(exc))}")
        io.warn("（会话保留；可重试或换个说法。）")
        return None
    observer.on_event(
        {
            "node": "orchestrator",
            "event": "end",
            "detail": f"意图：{orch_mod.intent_label(decision.intent)}",
        }
    )
    return decision


def _session_rounds_default(state: Mapping[str, Any]) -> int:
    """会话当前的轮次默认值（编排提示词里说明"未指定时的回退"）。"""
    settings = state.get("settings") if isinstance(state.get("settings"), Mapping) else {}
    value = settings.get("max_iterations")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return config_mod.DEFAULT_ROUNDS_DEFAULT


def _has_code_for(
    intent: str, state: Mapping[str, Any], file_input: tuple[Path, str] | None
) -> bool:
    """该意图是否有可作用的对象：审查 / 修复类意图需要既有代码（没有代码时反问，不空跑一轮）。"""
    if intent == orch_mod.INTENT_CODE:
        return True
    current = str(state.get("current_version_dir") or "")
    if current and Path(current).is_dir():
        return True
    return file_input is not None and not state.get("rounds")


def _resolve_rounds(
    *,
    decision: orch_mod.Decision,
    state: Mapping[str, Any],
    overrides: RoundOverrides | None,
    limit: int,
) -> tuple[RoundSettings, str]:
    """本轮轮次设置（优先级：命令行显式 > 消息提取 > 会话当前设置）。

    命令行的两个字段**分别**钉住：显式 ``--max-iter`` 钉住轮次数值（来源记 cli），
    显式 ``--rounds-mode`` 只钉住轮次语义——只给模式时，数值仍按优先级链从消息 / 会话
    设置 / 配置默认取值。消息提取值越界**夹到上限并标注**（意图理解 ≠ 用户显式参数）；
    命令行越界由命令面按参数错误拦下（本函数收到的命令行值必已合法）。

    **review_fix 的意图级默认**：用户要"审查之后修复"时不能被"审查干净"吞掉修复阶段——
    消息没给模式、且当前设置仍是默认的 ``up-to`` 时，本轮按 ``exact`` 执行（未给轮次则按
    1 轮：审查 → 修复 → 复审）。用户明确说了"没问题就不用改"（模型给出 ``up-to``）时照办。

    **review 的意图级默认**：``review`` = 只审查不改，未显式给轮次时本轮按 0 轮执行
    （首审即收尾，不进入修复），否则会话的默认轮次会把"只看不改"变成"接着修"。

    Returns:
        ``(设置, 标注)``；标注非空时表示轮次被夹取过、或本轮按 review_fix 的默认模式执行
        （进终端与会话档案）。
    """
    current = state.get("settings") if isinstance(state.get("settings"), Mapping) else {}
    base = RoundSettings(
        max_iterations=int(current.get("max_iterations") or config_mod.DEFAULT_ROUNDS_DEFAULT),
        max_iterations_source=str(current.get("max_iterations_source") or state_mod.SOURCE_DEFAULT),
        rounds_mode=str(current.get("rounds_mode") or rounds_mod.ROUNDS_UP_TO),
    )
    pinned = overrides.max_iterations if overrides is not None else None
    note = ""
    if pinned is not None:
        rounds, source = pinned, state_mod.SOURCE_CLI
    elif decision.rounds is not None:
        rounds = max(0, min(decision.rounds, limit))
        source = state_mod.SOURCE_REQUIREMENT
        if decision.rounds != rounds:
            note = f"消息里的轮次 {decision.rounds} 已按上限 {limit} 夹取为 {rounds}。"
    else:
        rounds, source = base.max_iterations, base.max_iterations_source
    pinned_mode = overrides.rounds_mode if overrides is not None else None
    effective_mode = pinned_mode or base.rounds_mode
    # review_fix 的意图级默认：消息没说模式、且设置里仍是 up-to 时，按"跑满"执行
    # （否则干净的审查会让修复阶段静默消失，与用户"审查后修复"的明确要求相悖）
    review_fix_default = (
        decision.rounds_mode is None
        and decision.intent == orch_mod.INTENT_REVIEW_FIX
        and effective_mode != rounds_mod.ROUNDS_EXACT
    )
    if review_fix_default:
        if decision.rounds is None and pinned is None:
            rounds, source = 1, state_mod.SOURCE_REQUIREMENT
        detail = "（审查干净也会走一轮修复）" if rounds > 0 else ""
        note = "；".join(
            item for item in (note, f"本轮按「审查→修复」默认跑满 {rounds} 轮{detail}") if item
        )
    mode = (
        rounds_mod.ROUNDS_EXACT
        if review_fix_default
        else (pinned_mode or decision.rounds_mode or base.rounds_mode)
    )
    # review 的意图级语义（"只审查不改"等价 rounds=0）：会话里默认的轮次
    # 设置不能把"只看不改"变成"接着修"。同一句话里明确给了轮次、或命令行钉住了轮次时，
    # 以那句话/那个参数为准（优先级链不变）。
    if decision.intent == orch_mod.INTENT_REVIEW and decision.rounds is None and pinned is None:
        if rounds != 0:
            note = "；".join(
                item for item in (note, "本轮按「只审查不改」执行，不进入修复") if item
            )
        rounds, source = 0, state_mod.SOURCE_REQUIREMENT
    return RoundSettings(
        max_iterations=rounds, max_iterations_source=source, rounds_mode=mode
    ), note


def _run_round(
    *,
    io: ChatIO,
    state: SessionState,
    decision: orch_mod.Decision,
    settings: RoundSettings,
    message: str,
    note: str,
    resolved: config_mod.ResolvedModel | None,
    clock: Callable[[], str],
    run_id_factory: engine_mod.RunIdFactory | None,
    file_input: tuple[Path, str] | None,
    severe_only: bool,
    no_exec: bool,
    confirm_deletions: engine_mod.DeletionConfirmer | None,
) -> RoundOutcome | None:
    """执行一轮：装配请求 → 引擎驱动 → 更新会话档案与记忆 → 返回结果视图。

    一轮的意外失败只警示、不终止会话（运行记录已由引擎落盘，是更强的一手证据）。
    """
    session_id = str(state["session_id"])
    session_path = session_state.session_dir(session_id)
    index = _next_round_index(state)
    requirement: str | None = None
    source_path: Path | None = None
    source_text: str | None = None
    source_dir: Path | None = None
    prior_files: Mapping[str, str] | None = None
    current_dir = str(state.get("current_version_dir") or "")
    if decision.intent == orch_mod.INTENT_CODE:
        # 在既有实现上继续开发：内联既有代码，起始版本号跨轮递增。
        # 传 source_path（而非 source_dir）只作来源追溯——物化会让编码产物压着同一号目录，
        # 新版本由编码输出定义，不需要先放一份副本。
        requirement = message
        if current_dir and Path(current_dir).is_dir():
            source_path = Path(current_dir)
            prior_files = engine_mod.read_version_files(Path(current_dir))
        elif file_input is not None:
            source_path = file_input[0]
            prior_files = {_input_name(file_input[0]): _normalize_text(file_input[1])}
    elif current_dir and Path(current_dir).is_dir():
        source_dir = Path(current_dir)
    elif file_input is not None:
        source_path, source_text = file_input[0], file_input[1]
    request = engine_mod.RunRequest(
        requirement=requirement,
        source_path=source_path,
        source_text=source_text,
        source_dir=source_dir,
        prior_files=prior_files,
        max_iterations=settings.max_iterations,
        max_iterations_source=settings.max_iterations_source,
        rounds_mode=settings.rounds_mode,
        severe_only=_effective_severe_only(state, cli_severe_only=severe_only),
        no_exec=no_exec,
        resolved=resolved,
        session_id=session_id,
        round_index=index,
        start_version=session_state.next_version(state),
        agent_memory=_memory_snapshot_safely(io, session_path, ts=clock()),
    )
    engine = engine_mod.WorkflowEngine(
        observer=io.renderer(),
        clock=clock,
        run_id_factory=run_id_factory,
        confirm_deletions=confirm_deletions,
        run_dir=session_path / "rounds" / f"round-{index}",
    )
    try:
        result = engine.execute(request)
    except Exception as exc:  # noqa: BLE001 — 会话层兜底：一轮的意外失败不终止会话
        io.warn(f"crw chat：本轮运行失败 — {render.mask_secrets(str(exc))}")
        return None
    outcome = RoundOutcome(
        run_id=result.run_id,
        run_dir=result.run_dir,
        state=result.state,
        exit_code=_STATUS_EXIT_CODES.get(result.status, _EXIT_ERROR),
        report_written=result.report_written,
        interrupted=result.interrupted or result.status == state_mod.STATUS_INTERRUPTED,
    )
    try:
        _update_session_after_round(
            io=io,
            state=state,
            outcome=outcome,
            decision=decision,
            settings=settings,
            index=index,
            message=message,
            note=note,
            clock=clock,
        )
    except (OSError, ValueError, TypeError) as exc:
        # 会话档案 / 记忆是会话层的记账：写不进去不改本轮结论（运行记录已由引擎落盘，是更强的一手证据）
        io.warn(f"crw chat：会话档案 / 记忆更新失败 — {exc}（运行记录已落盘，会话可继续）")
    return outcome


def _memory_snapshot_safely(
    io: ChatIO, session_path: Path, *, ts: str
) -> dict[str, list[dict[str, str]]]:
    """读记忆快照；失败按"空记忆"继续本轮（记忆是增强项，不阻塞工作流）。"""
    try:
        return _memory_snapshot(session_path, ts=ts)
    except (OSError, ValueError, TypeError) as exc:
        io.warn(f"crw chat：会话记忆读取失败，本轮按空记忆继续 — {exc}")
        return {}


def _effective_severe_only(state: Mapping[str, Any], *, cli_severe_only: bool) -> bool:
    """本轮是否放宽干净判定：本次命令行或档案里的设置取并集。

    并集方向与 ``--severe-only`` 的语义一致：命令行只能放宽判定，不能把档案里的放宽
    重新收紧——续跑 / 恢复会话是"接着干"，换判定标准会让结论不可比。
    """
    settings = state.get("settings") if isinstance(state.get("settings"), Mapping) else {}
    return bool(cli_severe_only) or bool(settings.get("severe_only"))


def _update_session_after_round(
    *,
    io: ChatIO,
    state: SessionState,
    outcome: RoundOutcome,
    decision: orch_mod.Decision,
    settings: RoundSettings,
    index: int,
    message: str,
    note: str,
    clock: Callable[[], str],
) -> None:
    """一轮结束后更新会话档案：轮次链、当前版本、当前设置；并按结构化状态追加记忆。"""
    run_state = outcome.state
    version_dir = str(run_state.get("current_version_dir") or "")
    files = run_state.get("version_files")
    if version_dir and Path(version_dir).is_dir():
        state["current_version_dir"] = version_dir
        state["version_files"] = [str(item) for item in files] if isinstance(files, list) else []
        version = _version_of_dir(version_dir)
        if version is not None:
            state["current_version"] = version
    settings_payload = dict(state.get("settings") or {})
    settings_payload.update(
        {
            "max_iterations": settings.max_iterations,
            "max_iterations_source": settings.max_iterations_source,
            "rounds_mode": settings.rounds_mode,
        }
    )
    state["settings"] = settings_payload
    state["updated_at"] = clock()
    conclusion = engine_mod.conclusion_text(
        status=str(run_state.get("status") or ""),
        review_round=int(run_state.get("review_round") or 0),
        fix_rounds=int(run_state.get("fix_rounds") or 0),
        findings=run_state.get("findings") if isinstance(run_state.get("findings"), list) else [],
        severe_only=bool(run_state.get("severe_only")),
    )
    session_state.record_round(
        state,
        index=index,
        run_id=outcome.run_id,
        intent=decision.intent,
        message=message,
        created_at=str(run_state.get("created_at") or clock()),
        finished_at=str(run_state.get("finished_at") or clock()),
        status=str(run_state.get("status") or ""),
        conclusion=conclusion,
        rounds_mode=settings.rounds_mode,
        max_iterations=settings.max_iterations,
        max_iterations_source=settings.max_iterations_source,
        version=_current_version(state) or None,
        review_round=int(run_state.get("review_round") or 0),
        fix_rounds=int(run_state.get("fix_rounds") or 0),
        note=note,
    )
    _append_memory(
        session_state.session_dir(str(state["session_id"])),
        state=run_state,
        round_index=index,
        message=message,
        intent=decision.intent,
        ts=str(run_state.get("finished_at") or clock()),
    )
    try:
        session_state.write_session(state)
    except (OSError, ValueError) as exc:
        io.warn(f"crw chat：会话档案写入失败 — {exc}（运行记录已落盘，会话可继续）")


def _append_memory(
    session_path: Path,
    *,
    state: Mapping[str, Any],
    round_index: int,
    message: str,
    intent: str,
    ts: str,
) -> None:
    """把本轮各 Agent 的结构化摘要追加进各自的记忆（代码与思考内容不进记忆）。"""
    entries = _memory_entries(state, round_index=round_index, message=message, intent=intent)
    for agent, exchanges in entries.items():
        if not exchanges:
            continue
        store = memory_mod.AgentMemory(memory_mod.memory_file(session_path, agent), agent=agent)
        for task, outcome in exchanges:
            store.append_exchange(round_index=round_index, task=task, outcome=outcome, ts=ts)


def _memory_entries(
    state: Mapping[str, Any], *, round_index: int, message: str, intent: str
) -> dict[str, list[tuple[str, str]]]:
    """本轮各 Agent 的记忆条目：``agent → [(任务要点, 结论要点)]``。

    条目由**结构化状态**（``history`` 的阶段摘要）生成：摘要本身不含代码（代码只在产物
    文件里），失败轮次的条目也在其中（如实记录）。
    """
    entries: dict[str, list[tuple[str, str]]] = {agent: [] for agent in memory_mod.MEMORY_AGENTS}
    history = state.get("history") if isinstance(state.get("history"), list) else []
    for item in history:
        if not isinstance(item, Mapping):
            continue
        node = str(item.get("node") or "")
        agent = _AGENT_OF_NODE.get(node)
        if agent is None:
            continue
        outcome = str(item.get("summary") or "").strip() or "（无摘要）"
        if agent == "encoder":
            task = f"第 {round_index} 轮：{orch_mod.intent_label(intent)} —— {message}"
        elif agent == "reviewer":
            task = f"第 {round_index} 轮：审查当前版本"
        else:
            task = f"第 {round_index} 轮：修复审查问题"
        entries[agent].append((task, outcome))
    return entries


def _memory_snapshot(session_path: Path, *, ts: str) -> dict[str, list[dict[str, str]]]:
    """三 Agent 的记忆快照（进 prompt 的历史消息）；接近预算时先做一次显式压缩。

    预算取 ``CRW_MAX_CONTEXT_TOKENS``（与运行时护栏同源）；压缩只追加 compaction 行、
    旧行留盘。
    """
    try:
        budget = config_mod.get_max_context_tokens()
    except config_mod.ConfigError:
        budget = config_mod.DEFAULT_MAX_CONTEXT_TOKENS
    snapshot: dict[str, list[dict[str, str]]] = {}
    for agent in memory_mod.MEMORY_AGENTS:
        store = memory_mod.AgentMemory(memory_mod.memory_file(session_path, agent), agent=agent)
        if store.needs_compaction(budget):
            store.compact(round_index=0, ts=ts or "unknown")
        history = store.prompt_history()
        if history:
            snapshot[agent] = history
    return snapshot


def _finish_session(io: ChatIO, *, state: SessionState, clock: Callable[[], str]) -> None:
    """收尾：写会话档案（status=ended + 时间）与会话总报告 ``session.md``，打印汇总。

    收尾失败只警示：会话档案与 ``session.md`` 是派生记录，每轮的运行记录才是更强的
    一手证据——写不进去给一行可读警示，不改变会话结论。
    """
    session_path = session_state.session_dir(str(state["session_id"]))
    state["status"] = session_state.SESSION_STATUS_ENDED
    state["updated_at"] = clock()
    rounds = [item for item in state.get("rounds") or [] if isinstance(item, Mapping)]
    last_status = str(rounds[-1].get("status") or "") if rounds else ""
    try:
        session_state.write_session(state)
    except KeyboardInterrupt:
        io.warn("crw chat：会话档案收尾被中断（运行记录不受影响）。")
    except (OSError, ValueError) as exc:
        io.warn(f"crw chat：会话档案写入失败 — {exc}")
    try:
        text = build_session_md(state, session_path=session_path)
        records_mod.atomic_write_text(session_path / "session.md", text)
    except KeyboardInterrupt:
        io.warn("crw chat：会话总报告收尾被中断（session.json 仍可用）。")
    except (OSError, ValueError, TypeError) as exc:
        io.warn(f"crw chat：会话总报告写入失败 — {exc}（session.json 仍可用）")
    label = engine_mod.status_label(last_status) if last_status else "未执行任何轮次"
    io.say("")
    io.say(f"[会话] 结束：共 {len(rounds)} 轮；最后状态：{label}")
    io.say(f"[会话] 会话记录：{session_path}（session.json / session.md / memory/ / rounds/）")


def build_session_md(state: Mapping[str, Any], *, session_path: Path) -> str:
    """会话总报告：摘要 → 逐轮结论演进 → 剩余建议 → 记忆索引。

    "剩余建议"取自**最后一轮**的运行记录（``state.json`` 的残留 findings）：会话档案里
    只存结论句，问题清单仍在各轮记录里（单一真相）。
    """
    rounds = [item for item in state.get("rounds") or [] if isinstance(item, Mapping)]
    session_id = str(state.get("session_id") or "（未知会话号）")
    lines: list[str] = [f"# 会话报告 · {_cell(session_id)}", ""]
    lines.append(f"**结论：{_session_conclusion(state, rounds)}**")
    lines.append("")
    lines.append("## 会话摘要")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("| --- | --- |")
    descriptor = state.get("input") if isinstance(state.get("input"), Mapping) else {}
    input_text = (
        state_mod.summarize_input(descriptor, limit=200) if descriptor else "（会话内逐轮给出需求）"
    )
    lines.append(f"| 起始输入 | {_cell(input_text)} |")
    lines.append(f"| 模型 | {_cell(state.get('model') or '（未记录）')} |")
    lines.append(
        f"| 开始 / 结束 | {_cell(state.get('created_at'))} → {_cell(state.get('updated_at'))} |"
    )
    lines.append(f"| 轮次数 | {len(rounds)} |")
    version = _current_version(state)
    files = state.get("version_files")
    file_count = (
        len([item for item in files if isinstance(item, str)]) if isinstance(files, list) else 0
    )
    if version:
        lines.append(f"| 当前代码版本 | code_v{version}/（{file_count} 个文件） |")
    settings = state.get("settings") if isinstance(state.get("settings"), Mapping) else {}
    lines.append(
        "| 当前轮次设置 | "
        f"上限 {_cell(settings.get('max_iterations'))}"
        f"（来源：{engine_mod.source_label(settings.get('max_iterations_source'))}）；"
        f"模式 {_cell(settings.get('rounds_mode'))} |"
    )
    lines.append(
        "| 干净判定 | "
        + (
            "无【严重】（--severe-only 放宽）"
            if settings.get("severe_only")
            else "无【严重】且无【一般】（默认判据）"
        )
        + " |"
    )
    lines.append("")
    lines.append("## 逐轮记录")
    lines.append("")
    if not rounds:
        lines.append("（本次会话未执行任何轮次）")
        lines.append("")
    for item in rounds:
        index = item.get("index")
        lines.append(
            f"### 第 {index} 轮（{orch_mod.intent_label(str(item.get('intent') or ''))}）"
            f"——{engine_mod.status_label(str(item.get('status') or ''))}"
        )
        lines.append("")
        lines.append(f"- 用户消息：{_cell(item.get('message'))}")
        lines.append(
            f"- 运行记录：{_cell(item.get('run_id'))}"
            f"（`rounds/round-{_cell(index)}/`，报告 report.md）"
        )
        lines.append(
            f"- 轮次使用：审查 {_cell(item.get('review_round'))} 轮 / "
            f"修复 {_cell(item.get('fix_rounds'))} 轮（上限 {_cell(item.get('max_iterations'))}"
            f"，来源：{engine_mod.source_label(item.get('max_iterations_source'))}，"
            f"模式 {_cell(item.get('rounds_mode'))}）"
        )
        if item.get("note"):
            lines.append(f"- 标注：{_cell(item.get('note'))}")
        lines.append(f"- 结论：{_cell(item.get('conclusion'))}")
        if item.get("version"):
            lines.append(f"- 会话代码版本：code_v{_cell(item.get('version'))}/")
        lines.append("")
    lines.extend(_remaining_suggestions(session_path, rounds))
    lines.append("## 记忆（各 Agent 的追加式记忆）")
    lines.append("")
    for agent in memory_mod.MEMORY_AGENTS:
        store = memory_mod.AgentMemory(memory_mod.memory_file(session_path, agent), agent=agent)
        all_lines = store.lines()
        effective = store.effective_lines()
        note = "" if len(all_lines) == len(effective) else "（已含一次压缩，旧行留盘）"
        lines.append(f"- `{agent}`：{len(all_lines)} 条记录（`memory/{agent}.jsonl`）{note}")
    lines.append("")
    return "\n".join(lines)


def _remaining_suggestions(session_path: Path, rounds: Sequence[Mapping[str, Any]]) -> list[str]:
    """剩余建议：最后一轮记录的残留 findings（读不到就空着——报告是收尾产物）。"""
    lines = ["## 剩余建议", ""]
    last_state: Mapping[str, Any] | None = None
    if rounds:
        candidate = session_path / "rounds" / f"round-{rounds[-1].get('index')}"
        with contextlib.suppress(OSError, ValueError):
            last_state = records_mod.read_state(candidate)
    findings = last_state.get("findings") if isinstance(last_state, Mapping) else None
    blockers = [
        item
        for item in (findings if isinstance(findings, list) else [])
        if isinstance(item, Mapping) and item.get("severity") in ("严重", "一般")
    ]
    if not blockers:
        lines.append("（无残留阻塞问题）")
        lines.append("")
        return lines
    for item in blockers:
        lines.append(
            f"- 【{_cell(item.get('severity'))}】`{_cell(item.get('location'))}` — "
            f"{_cell(item.get('issue'))}"
        )
    lines.append("")
    return lines


def _session_conclusion(state: Mapping[str, Any], rounds: Sequence[Mapping[str, Any]]) -> str:
    """会话结论句：最后一轮的结论 + 会话当前代码版本。"""
    if not rounds:
        return "未执行任何轮次。"
    conclusion = str(rounds[-1].get("conclusion") or "（无结论）")
    version = _current_version(state)
    tail = f"；会话当前代码版本 code_v{version}/" if version else ""
    return f"{conclusion}{tail}"


def _render_round_report(
    io: ChatIO, *, state: Mapping[str, Any], outcome: RoundOutcome, decision: orch_mod.Decision
) -> None:
    """每轮汇报（从结构化状态渲染，不由模型复述）。"""
    rounds = [item for item in state.get("rounds") or [] if isinstance(item, Mapping)]
    index = int(rounds[-1].get("index") or 0) if rounds else 0
    text = orch_mod.round_report(
        outcome.state,
        round_index=index,
        intent=decision.intent,
        run_dir=str(outcome.run_dir or "（未建立运行目录）"),
        report_written=outcome.report_written,
    )
    io.say("")
    io.say(text)


def _print_session_header(
    io: ChatIO, state: Mapping[str, Any], *, resume: bool, limit: int
) -> None:
    """会话头：会话号 / 模型 / 当前轮次设置与用法提示（会话模式的"结论面"）。"""
    session_id = str(state.get("session_id") or "")
    title = f"── 会话 {session_id}{'（恢复）' if resume else ''} "
    io.say(title + "─" * max(2, render.RULE_WIDTH - render.display_width(title)))
    io.say(f"[模型] {state.get('model') or '（未记录）'}")
    settings = state.get("settings") if isinstance(state.get("settings"), Mapping) else {}
    io.say(
        f"[轮次] 上限 {settings.get('max_iterations', '（未记录）')}"
        f"（来源：{engine_mod.source_label(settings.get('max_iterations_source'))}）；"
        f"模式 {settings.get('rounds_mode', rounds_mod.ROUNDS_UP_TO)}；"
        f"消息里可说「最多 N 轮 / 跑满 N 轮」（夹取上限 {limit}）"
    )
    rounds = [item for item in state.get("rounds") or [] if isinstance(item, Mapping)]
    if resume and rounds:
        version = _current_version(state)
        version_note = f"；当前代码 code_v{version}/" if version else ""
        io.say(f"[会话] 已恢复：已完成 {len(rounds)} 轮{version_note}")
    io.say("[会话] 直接说需求即可；/exit 或 Ctrl+C 退出；一轮进行中 Ctrl+C 可中断本轮（会话保留）")
    io.say(
        "[判定] "
        + (
            "仅【严重】阻塞（--severe-only 放宽）"
            if settings.get("severe_only")
            else "【严重】与【一般】都算阻塞，不阻塞的只有【建议】"
        )
    )
    io.say(f"会话记录：{session_state.session_dir(str(state.get('session_id') or ''))}")
    io.say("")


def _current_version(state: Mapping[str, Any]) -> int:
    """会话当前代码版本号（档案字段；缺失 / 非法按 0 处理）。"""
    value = state.get("current_version")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _next_round_index(state: Mapping[str, Any]) -> int:
    """下一个轮次序号：已记录轮次数与既有 ``round-<i>/`` 目录的最大序号取大者 + 1。

    以目录为准可避免"进程在落盘轮次前退出"留下的序号被重复使用（目录已存在的轮次
    记录绝不能被覆盖）。
    """
    recorded = len([item for item in state.get("rounds") or [] if isinstance(item, Mapping)])
    session_path = session_state.session_dir(str(state.get("session_id") or ""))
    existing = 0
    with contextlib.suppress(OSError):
        for item in (session_path / "rounds").iterdir():
            match = re.fullmatch(r"round-([0-9]+)", item.name)
            if item.is_dir() and match:
                existing = max(existing, int(match.group(1)))
    return max(recorded, existing) + 1


def _version_of_dir(path_text: str) -> int | None:
    """版本目录路径 → 版本号；不是版本目录时返回 ``None``。"""
    name = path_text.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    match = _VERSION_DIR_RE.fullmatch(name)
    return int(match.group(1)) if match else None


def _normalize_text(text: str) -> str:
    """``-f`` 内容归一（与引擎物化同口径：去掉首尾换行、单个结尾换行）。"""
    return text.strip("\n") + "\n"


def _input_name(path: Path) -> str:
    """``-f`` 输入在版本目录内的文件名（与引擎物化口径一致）。"""
    return path.name or "main.py"


def _cell(value: Any) -> str:
    """Markdown 表格单元格：转义竖线并折叠换行。"""
    return " ".join(str(value if value is not None else "").split()).replace("|", "\\|")


__all__ = [
    "EXIT_COMMANDS",
    "ChatIO",
    "RoundOutcome",
    "RoundSettings",
    "build_session_md",
    "run_chat",
]
