"""运行驱动：工作区建立、图执行、节点边界事件与快照、报告生成与退出码确定。

状态载体是图执行上下文本身，本模块只做生命周期组织，不维护第二份内存正本（Design §3.1、§4.5）。
节点输出经本模块的契约包装层校验、重试与落盘：节点自身不接触文件、不打印、不写日志（Design §3.1、§4.2.1）。
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import inspect
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .. import agents as agents_mod
from .. import config as config_mod
from .. import usage as usage_mod
from . import contracts, records
from . import graph as graph_mod
from . import state as state_mod
from .contracts import ContractViolation
from .graph import NodeFn
from .records import RoundReport, RunRecordError
from .state import RunState

#: 内置默认修复轮次上限（Design §4.4）
DEFAULT_MAX_ITERATIONS = 3

#: 节点名 → 显示名（终端阶段面板、报告、详情与退出码文案共用一处口径）
NODE_LABELS: dict[str, str] = {
    "encode": "编码",
    "review": "审查",
    "review_correctness": "审查·正确性",
    "review_security": "审查·安全性",
    "merge_reviews": "汇合",
    "fix": "修复",
}

#: 失败信息与事件 detail 的显示上限（沿用 Design §8.2 的 400 字符惯例）
MAX_ERROR_CHARS = 400

#: 未提供模型配置时的模型标识占位（仅程序化调用会走到）
UNRESOLVED_MODEL_LABEL = "（未解析模型）"

_EXIT_SUCCESS = 0
_EXIT_RESIDUAL = 1
_EXIT_ERROR = 2

Clock = Callable[[], str]
Timer = Callable[[], float]
RunIdFactory = Callable[[], str]
#: 缺省节点装配所需的运行时工厂：收到事件 / 增量回调，返回单任务 Agent 运行时
RuntimeFactory = Callable[[agents_mod.EventSink, agents_mod.DeltaSink], agents_mod.AgentRuntime]
#: 删除确认闸门（Design §4.2.2）：入参为待删清单（``path`` + 对应 findings 的理由），
#: 返回是否确认删除。``None`` = 非交互（一律保留并标注"待确认删除（未执行）"）。
DeletionConfirmer = Callable[[Sequence[Mapping[str, Any]]], bool]


def local_now() -> str:
    """默认时钟：本地时区 ISO 8601（秒级）。"""
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


class PersistenceError(RuntimeError):
    """运行记录落盘失败（Design §4.6：按运行失败处理，退出码 2）。"""


class _NodeFailure(RuntimeError):
    """节点失败：携带节点名、可读消息与时间信息，供 engine 写状态与时间线。"""

    def __init__(
        self,
        *,
        node: str,
        message: str,
        round_number: int,
        started_at: str,
        duration_ms: int,
        usage: Mapping[str, int] | None = None,
    ):
        super().__init__(message)
        self.node = node
        self.message = message
        self.round_number = round_number
        self.started_at = started_at
        self.duration_ms = duration_ms
        #: 失败前该节点已花掉的用量：钱已经花了，账要留在运行记录里
        self.usage = usage if usage is not None else usage_mod.empty()


class RunObserver(Protocol):
    """运行过程观察者（渲染层实现，Design §6.2 #6）；缺省不输出。"""

    def on_event(self, event: Mapping[str, Any]) -> None:
        """节点 / 引擎事件（比 log.jsonl 多带渲染所需附加键，如运行号）。"""

    def on_delta(self, node: str, text: str) -> None:
        """模型正文增量（渲染层据此流式输出；不落盘——正文由节点产物承载）。"""

    def on_review(
        self,
        *,
        round_number: int,
        version_label: str,
        findings: Sequence[Mapping[str, Any]],
        summary: str,
        view: str | None = None,
    ) -> None:
        """一轮审查结论（findings 清单，已过契约校验）；``view`` 非空表示并行分支视角。"""

    def on_fixes(
        self,
        *,
        round_number: int,
        fixes: Sequence[Mapping[str, Any]],
        findings: Sequence[Mapping[str, Any]],
    ) -> None:
        """一轮修复处置（与上一轮 findings 按 id 配对）。"""


class _NullObserver:
    """无输出观察者（测试与非交互场景的缺省）。"""

    def on_event(self, event: Mapping[str, Any]) -> None:
        """忽略事件。"""

    def on_delta(self, node: str, text: str) -> None:
        """忽略正文增量。"""

    def on_review(
        self,
        *,
        round_number: int,
        version_label: str,
        findings: Sequence[Mapping[str, Any]],
        summary: str,
        view: str | None = None,
    ) -> None:
        """忽略审查结论。"""

    def on_fixes(
        self,
        *,
        round_number: int,
        fixes: Sequence[Mapping[str, Any]],
        findings: Sequence[Mapping[str, Any]],
    ) -> None:
        """忽略修复处置。"""


@dataclass(frozen=True)
class RunRequest:
    """一次运行的输入与选项（会话模式复用同一结构，M4 追加会话字段）。"""

    requirement: str | None = None
    source_path: Path | None = None
    source_text: str | None = None
    source_dir: Path | None = None
    """既有版本目录（resume / 会话审查轮次：复制为新运行的起始版本目录，Design §4.2.2）。"""
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_iterations_source: str = state_mod.SOURCE_DEFAULT
    rounds_mode: str = state_mod.ROUNDS_UP_TO
    severe_only: bool = False
    no_exec: bool = False
    parallel_review: bool = False
    """并行双视角审查（Design §12.2 的扩展示例）：两个审查视角同超级步并行、汇合后判干净。"""
    resolved: config_mod.ResolvedModel | None = None
    """本次运行的模型连接（命令面经 ``config.resolve`` 解析）。

    缺省（``None``）时：完全注入 ``nodes`` 的运行不受影响；需要缺省装配时由运行时按
    环境变量回退（``CRW_`` → ``CRA_`` → ``SE_CodeAgent``）构建模型，无 key 则给可读运行失败。
    """

    parent_run_id: str | None = None
    session_id: str | None = None
    round_index: int | None = None
    start_version: int = 1
    """本次运行的起始版本号（会话轮次跨轮连续递增，Design §4.7）；一次性运行恒为 1。"""
    prior_files: Mapping[str, str] | None = None
    """既有实现的文件内容（会话轮次的编码上下文）：引擎放进状态瞬态键 ``prior_files``，
    编码节点把它内联进任务提示词（"按新需求修改"的依据，Design §4.7）。"""
    agent_memory: Mapping[str, Sequence[Mapping[str, str]]] | None = None
    """三 Agent 各自的会话记忆（``agent → [{"role", "text"}]``）：引擎放进状态瞬态键
    ``agent_memory``，各节点把它作为历史消息交给运行时（Design §4.7 的追加式记忆）。"""

    def input_descriptor(self) -> state_mod.InputDescriptor:
        """本次运行的输入描述（Design §4.3 的 ``input`` 字段）。

        有需求文本时以它为准（会话轮次"在既有版本上按新需求继续开发"即此形态：既要走
        编码起步，又要记下既有代码的来源）；否则 ``-f`` 记原文件路径、续跑记父运行的
        版本目录路径（"原文件 → 工作区"的映射口径随之变为"原版本目录 → 起始版本目录"）。
        """
        source = self.source_path if self.source_path is not None else self.source_dir
        if self.requirement:
            return state_mod.InputDescriptor(
                kind=state_mod.INPUT_REQUIREMENT,
                requirement=self.requirement,
                **({"source_path": str(source)} if source is not None else {}),
            )
        if source is not None:
            return state_mod.InputDescriptor(kind=state_mod.INPUT_FILE, source_path=str(source))
        if self.source_text is not None:
            # 程序化调用可能只给文本（CLI 侧恒给 source_path）：它仍会被物化为
            # code_v1/main.py，输入描述必须与物化结果同口径（kind=file）
            return state_mod.InputDescriptor(kind=state_mod.INPUT_FILE)
        return state_mod.InputDescriptor(kind=state_mod.INPUT_REQUIREMENT, requirement="")

    @property
    def model_label(self) -> str:
        """模型标识（写入 ``state["model"]``，Design §4.3）：``供应商/模型名``。"""
        return self.resolved.label() if self.resolved is not None else UNRESOLVED_MODEL_LABEL


@dataclass(frozen=True)
class RunResult:
    """一次运行的结果视图（命令面据此渲染与确定退出码）。"""

    run_id: str
    run_dir: Path | None
    status: str
    exit_code: int
    review_round: int
    fix_rounds: int
    clean: bool
    findings: tuple[Mapping[str, Any], ...]
    error_node: str | None
    error_message: str | None
    interrupted: bool
    state: Mapping[str, Any]
    report_written: bool = False
    """是否已产出 report.md（中断收尾按 §4.6 的写盘序列不做报告，命令面据此提示）。"""

    @property
    def succeeded(self) -> bool:
        """是否干净通过（退出码 0）。"""
        return self.exit_code == _EXIT_SUCCESS


@dataclass(frozen=True)
class _Active:
    """当前节点的执行上下文（按线程保存）：轮次、开始时间与计时起点。"""

    node: str
    round: int
    started_at: str
    started: float


@dataclass(frozen=True)
class _Listing:
    """节点产出的清单载荷：由包装层在 node end 事件之后交给观察者渲染。"""

    kind: str
    payload: Mapping[str, Any]


class WorkflowEngine:
    """运行驱动：组织运行目录、事件、快照、报告与退出码（Design §3.1）。

    Args:
        nodes: 节点覆盖表（逐键替换缺省装配），测试与真实装配共用同一接缝。
        runtime_factory: 缺省装配所需的 Agent 运行时工厂 ``(emit, on_delta) -> runtime``；
            缺省用 agents 层按 ``request.resolved`` 装配。完全注入 ``nodes`` 时不会被调用。
        clock: 时钟（返回 ISO 8601 字符串）；默认 :func:`local_now`。
        timer: 单调计时源（毫秒耗时）；默认 :func:`time.monotonic`。
        run_id_factory: 运行号生成器；默认按注入时钟的日期 + ``records.new_run_id`` 碰撞重试。
        observer: 过程观察者；默认不输出。
        confirm_deletions: 删除确认闸门（Design §4.2.2）：修复提出删除时引擎先打印待删
            清单并经它征询 ``y/N``；``None``（非交互 / 管道 / CI）或它抛异常 / 被中断时
            一律**保留文件**并标注"待确认删除（未执行）"。
    """

    def __init__(
        self,
        *,
        nodes: Mapping[str, NodeFn] | None = None,
        runtime_factory: RuntimeFactory | None = None,
        clock: Clock | None = None,
        timer: Timer | None = None,
        run_id_factory: RunIdFactory | None = None,
        observer: RunObserver | None = None,
        confirm_deletions: DeletionConfirmer | None = None,
        run_parent: Path | None = None,
        run_dir: Path | None = None,
    ) -> None:
        self._nodes = dict(nodes) if nodes else None
        self._runtime_factory = runtime_factory
        self._clock = clock or local_now
        self._timer = timer or time.monotonic
        self._run_id_factory = run_id_factory or self._default_run_id_factory
        self._observer: RunObserver = observer or _NullObserver()
        self._confirm_deletions = confirm_deletions
        self._run_parent = run_parent
        self._run_dir_override = run_dir
        self._run_dir: Path | None = None
        self._workspace: Path | None = None
        self._log_path: Path | None = None
        self._log_lock = threading.Lock()
        # 正在执行的节点上下文按**节点名**保存：并行双视角的两个分支会同时执行（子图在线程池
        # 里跑，工具事件来自别的线程），单槽记录会让事件被归到另一个分支的轮次上——按名归档
        # 既线程安全又与事件自带的 ``node`` 字段天然对齐
        self._active_nodes: dict[str, _Active] = {}
        self._latest: dict[str, Any] | None = None
        self._runtime: agents_mod.AgentRuntime | None = None
        self._no_exec = False
        self._parallel_review = False
        self._request: RunRequest | None = None
        #: 双视角分支的评述缓存（键 = 审查轮次 → 视角 → 评述文本）：汇合产物要合并两边的评述，
        #: 而分支结论此刻已落盘为 review_N_<view>.json，没必要再读回来
        self._branch_summaries: dict[int, dict[str, str]] = {}

    def _begin_node(self, name: str, round_number: int) -> _Active:
        """登记一个正在执行的节点（返回其上下文；同一节点名重复登记即覆盖，重试属同一节点）。"""
        active = _Active(
            node=name,
            round=round_number,
            started_at=self._clock(),
            started=self._timer(),
        )
        self._active_nodes[name] = active
        return active

    def _end_node(self, name: str) -> None:
        self._active_nodes.pop(name, None)

    def _any_active(self) -> _Active | None:
        """任一在建节点（中断与顶层兜底用）：单节点路径下即当前节点。

        先取快照再取值：并行双视角的分支在别的线程里增删这本字典，直接迭代会在尺寸变化时
        抛 ``RuntimeError``——而本函数专用于中断 / 兜底这类异常路径，正是最容易撞上的时刻。
        """
        values = list(self._active_nodes.values())
        return values[0] if values else None

    # ---- 对外入口 ----

    def execute(self, request: RunRequest) -> RunResult:
        """驱动一次完整运行：建目录 → 初始快照 → 图执行 → 收尾报告与退出码。

        错误与中断语义见 Design §4.6：节点失败 / 落盘失败 → ``status=failed`` + 退出码 2；
        Ctrl+C → 追加 interrupt 事件 → 原子写 ``state.json`` → 退出码 2（第二次 Ctrl+C 放弃写盘）。
        """
        try:
            run_id, run_dir = self._create_run()
        except (RunRecordError, OSError) as exc:
            return self._result({}, _EXIT_ERROR, message=f"运行目录创建失败：{exc}")
        self._run_dir = run_dir
        self._workspace = run_dir / "workspace"
        self._log_path = run_dir / "log.jsonl"
        self._request = request
        self._no_exec = request.no_exec
        self._parallel_review = request.parallel_review
        # 同一引擎实例被复用执行多次时，运行时必须按本次请求重建：它绑定了模型、上下文根
        # 与 no_exec，串用会让第二次运行拿着上一次的装配跑
        self._runtime = None
        self._branch_summaries = {}
        # 节点在建表同样重置：KeyboardInterrupt 等 BaseException 不会走 _end_node，
        # 引擎实例被复用执行下一次时不得残留上一次的节点上下文
        self._active_nodes = {}
        initial = state_mod.new_state(
            run_id=run_id,
            input_descriptor=request.input_descriptor(),
            workspace=self._workspace,
            max_iterations=request.max_iterations,
            max_iterations_source=request.max_iterations_source,
            rounds_mode=request.rounds_mode,
            severe_only=request.severe_only,
            parallel_review=request.parallel_review,
            model=request.model_label,
            created_at=self._clock(),
            parent_run_id=request.parent_run_id,
            session_id=request.session_id,
            round_index=request.round_index,
            version_base=request.start_version,
        )
        # 会话轮次的编码上下文与三 Agent 记忆：瞬态键（不进运行记录），只在本轮图执行内可见
        if request.prior_files:
            initial["prior_files"] = dict(request.prior_files)
        if request.agent_memory:
            # 形态不符的条目直接丢弃（与 runtime.history_from_state 同一口径）：
            # 记忆是增强项，坏数据不该把本轮运行变成失败
            initial["agent_memory"] = {
                str(agent): [dict(item) for item in lines if isinstance(item, Mapping)]
                for agent, lines in request.agent_memory.items()
                if isinstance(lines, Sequence) and not isinstance(lines, (str, bytes))
            }
        try:
            self._materialize_input_version(request, initial)
            # 快照取在物化之后：物化会就地写入 current_version_dir / version_files，
            # 若随后的首次落盘失败，失败快照必须带上这两项（否则记录与工作区脱节）
            self._latest = dict(initial)
            self._persist_state(initial)
            self._emit(
                {
                    "node": "engine",
                    "event": "start",
                    "detail": self._start_detail(request),
                    "run_id": run_id,
                    "status": state_mod.STATUS_RUNNING,
                    "model": request.model_label,
                }
            )
            final = self._drive(initial)
            return self._finish(final)
        except KeyboardInterrupt:
            return self._handle_interrupt(self._latest or initial)
        except _NodeFailure as failure:
            return self._handle_node_failure(self._latest or initial, failure)
        except PersistenceError as exc:
            return self._handle_persistence_failure(self._latest or initial, exc)
        except Exception as exc:  # noqa: BLE001 — 图执行层面的未预期异常转为可读运行失败
            active = self._any_active()
            failure = _NodeFailure(
                node=active.node if active else "engine",
                message=_readable_error(exc),
                round_number=active.round if active else 0,
                started_at=(active.started_at if active else "") or self._clock(),
                duration_ms=self._elapsed_ms(active) if active else 0,
            )
            return self._handle_node_failure(self._latest or initial, failure)

    # ---- 图执行 ----

    def _drive(self, initial: RunState) -> dict[str, Any]:
        """驱动图执行：节点边界快照原子写，返回最后一个超级步的状态。"""
        resolved = graph_mod.resolve_nodes(
            self._nodes, defaults=self._default_nodes(), parallel_review=self._parallel_review
        )
        wrapped = {name: self._wrap_node(name, fn) for name, fn in resolved.items()}
        app = graph_mod.build_graph(nodes=wrapped, parallel_review=self._parallel_review)
        recursion_limit = graph_mod.recursion_limit(
            max_iterations=int(initial["max_iterations"]), parallel_review=self._parallel_review
        )
        latest: dict[str, Any] = dict(initial)
        # 初始快照已在建立运行目录时落盘：以它的序列化文本为基线，跳过首个（相同的）产出
        last_text = state_mod.dumps(initial)
        # 崩溃护栏经 config 传入：langgraph 的 stream(..., **kwargs) 会把未知关键字吞进
        # 废弃参数桶（实测生效值仍是默认 10007），只有 config 位置参数真正生效。
        for snapshot_value in app.stream(
            initial, {"recursion_limit": recursion_limit}, stream_mode="values"
        ):
            # values 模式：先给出输入状态本身，此后每个超级步结束给出完整快照。
            # 空超级步（条件边求值）快照与上一份相同，跳过以保持"每节点写一次"的语义。
            latest = dict(snapshot_value)
            self._latest = latest
            text = state_mod.dumps(latest)
            if text == last_text:
                continue
            self._persist_state_text(text)
            last_text = text
        return latest

    def _default_nodes(self) -> dict[str, NodeFn] | None:
        """缺省节点装配（真实 Agent 节点 + 本层的汇合节点）；``nodes`` 已覆盖全部节点时返回 ``None``。

        运行时只在确需缺省装配时构建：完全注入节点的测试不触碰模型与配置。并行形态多出的
        汇合节点由本层提供（纯函数、无模型、无 IO），其余节点仍来自 agents 层——依赖方向
        不变（workflow → agents，Design §3.1）。
        """
        names = graph_mod.node_names(parallel_review=self._parallel_review)
        if self._nodes and set(names) <= set(self._nodes):
            return None
        runtime = self._make_runtime()
        nodes = agents_mod.build_nodes(
            runtime,
            roots=[self._workspace_path()],
            no_exec=self._no_exec,
            parallel_review=self._parallel_review,
        )
        if self._parallel_review:
            nodes[graph_mod.MERGE_NODE] = graph_mod.merge_review_node
        return nodes

    def _make_runtime(self) -> agents_mod.AgentRuntime:
        """构建（或复用）本次运行的 Agent 运行时。"""
        if self._runtime is not None:
            return self._runtime
        if self._runtime_factory is not None:
            self._runtime = self._runtime_factory(self._emit, self._delta)
        else:
            self._runtime = agents_mod.build_runtime(
                resolved=self._request.resolved if self._request else None,
                emit=self._emit,
                on_delta=self._delta,
            )
        return self._runtime

    def _delta(self, node: str, text: str) -> None:
        """模型正文增量 → 观察者（只渲染，不落盘；增量本身不是产物）。"""
        self._notify_observer(lambda: self._observer.on_delta(node, text))

    def _drain_notices(self, node: str | None = None) -> list[str]:
        """取走运行时的护栏触达记录（工具轮次 / 上下文预算），并入阶段摘要。

        优先按节点取（``drain_notices_for``）：并行分支各自只回收自己的记录；
        注入的运行时没有该方法时退回全局 drain（测试替身的既有口径）。
        """
        if self._runtime is None:
            return []
        try:
            if node is not None:
                drain_for = getattr(self._runtime, "drain_notices_for", None)
                if drain_for is not None:
                    return drain_for(node)
            return self._runtime.drain_notices()
        except Exception:  # noqa: BLE001 — 摘要附加信息不得影响运行结论
            return []

    def _drain_usage(self, node: str) -> dict[str, int]:
        """取走该节点本次任务的 token 用量账（注入的假运行时不回报用量时为空账）。

        用量采集是观测面而非判定面：取不到就按"未记录"处理，绝不影响运行结论与退出码。
        """
        drain = getattr(self._runtime, "drain_usage_for", None) if self._runtime else None
        if drain is None:
            return usage_mod.empty()
        try:
            return drain(node)
        except Exception:  # noqa: BLE001 — 观测数据不得把一次正常运行判成失败
            return usage_mod.empty()

    def _drain_residual_usage(self) -> dict[str, dict[str, int]]:
        """取走运行时里未被按节点回收的用量（被中断的节点、无引擎接入的调用方）。"""
        drain = getattr(self._runtime, "drain_usage", None) if self._runtime else None
        if drain is None:
            return {}
        try:
            return drain()
        except Exception:  # noqa: BLE001 — 同上：观测数据不影响收尾
            return {}

    def _emit(self, event: Mapping[str, Any]) -> None:
        """落盘一条事件并转交观察者（Design §5.1）；log 写入失败按运行失败处理。

        节点级事件（``encode`` / ``review`` / ``fix``）缺 ``iteration`` 时按当前节点轮次
        补齐——运行时只知道自己服务于哪个节点，轮次口径由引擎掌握（工具事件据此可归轮）。
        """
        payload = dict(event)
        active = self._active_nodes.get(str(payload.get("node") or ""))
        if active is not None and payload.get("iteration") is None:
            payload["iteration"] = active.round
        record = {"ts": self._clock(), **payload}
        if self._log_path is not None:
            try:
                # 并行双视角的两个分支在别的线程里同时发事件：逐事件加锁，
                # 保证日志行的写入互不穿插（行完整性 > 吞吐，日志是审计真相）
                with self._log_lock:
                    records.append_log_event(self._log_path, record)
            except OSError as exc:
                raise PersistenceError(f"log.jsonl 写入失败：{exc}") from exc
        self._notify_observer(lambda: self._observer.on_event(record))

    def _wrap_node(self, name: str, fn: NodeFn) -> NodeFn:
        """节点包装：事件、契约校验与重试、产物落盘、history 追加（Design §3.1、§4.6）。"""

        def wrapped(state: Mapping[str, Any]) -> Mapping[str, Any]:
            round_number = self._round_of(name, state)
            active = self._begin_node(name, round_number)
            started_at = active.started_at
            self._emit({"node": name, "event": "start", "iteration": round_number})
            try:
                update, summary, artifacts, listing = self._produce(name, fn, state)
            except PersistenceError:
                # 落盘失败按落盘失败归因（Design §4.6）：交 execute 的持久化失败分支处理，
                # 不在这里被归为"节点自身失败"——两者都退出 2，但原因定位必须准确。
                self._end_node(name)
                raise
            except Exception as exc:
                message = _readable_error(exc)
                # 护栏触达信息一并带上：节点在"工具/预算已达上限"之后失败时，这是最有力的线索
                message = _append_notices(message, self._drain_notices(name))
                duration = self._elapsed_ms(active)
                self._emit(
                    {"node": name, "event": "error", "iteration": round_number, "detail": message}
                )
                self._end_node(name)
                raise _NodeFailure(
                    node=name,
                    message=message,
                    round_number=round_number,
                    started_at=started_at,
                    duration_ms=duration,
                    usage=self._drain_usage(name),
                ) from exc
            duration = self._elapsed_ms(active)
            summary = _append_notices(summary, self._drain_notices(name))
            self._emit({"node": name, "event": "end", "iteration": round_number, "detail": summary})
            self._render_listing(listing)
            self._end_node(name)
            entry = state_mod.NodeEvent(
                node=name,
                round=round_number,
                started_at=started_at,
                duration_ms=duration,
                summary=summary,
                artifacts=list(artifacts),
            )
            node_usage = self._drain_usage(name)
            # 空账不写字段：假节点（脚本化、不经模型）的运行记录里不该出现"用量全 0"的误导行
            if usage_mod.is_empty(node_usage):
                return {**update, "history": [entry]}
            return {**update, "history": [entry], "usage": {name: node_usage}}

        return wrapped

    def _produce(
        self, name: str, fn: NodeFn, state: Mapping[str, Any]
    ) -> tuple[dict[str, Any], str, list[str], _Listing | None]:
        """按节点种类执行契约包装：调用 → 抽取 → 校验（失败重试一次）→ 产物落盘。

        种类由节点名决定（``encode`` / ``fix`` / ``review``（含并行视角分支）/ 汇合），两种图
        形态共用同一套包装逻辑——扩展形态只在"谁来写权威字段"上不同（Design §12.2）。
        """
        if name == "encode":
            return self._produce_encode(fn, state)
        if name == "fix":
            return self._produce_fix(fn, state)
        if name == graph_mod.MERGE_NODE:
            return self._produce_merge(fn, state)
        return self._produce_review(
            fn, state, node_name=name, view=graph_mod.view_of_review_node(name)
        )

    def _produce_encode(
        self, fn: NodeFn, state: Mapping[str, Any]
    ) -> tuple[dict[str, Any], str, list[str], _Listing | None]:
        """编码节点：抽取 1..n 个文件 → 落盘起始版本目录（Design §4.2.2）。

        目录号取 ``code_version(state)``：一次性运行为 ``code_v1/``；会话轮次沿用跨轮
        连续递增的起始版本号（第 2 轮的编码产物从上一轮结束后的下一号起，Design §4.7）。
        """
        output = _call_node(fn, state, retry_feedback=None)
        try:
            files, duplicates, _ = self._extract_version_files(output)
        except ContractViolation as exc:
            output = self._retry("encode", 1, fn, state, str(exc))
            files, duplicates, _ = self._extract_version_files(output)
        version = state_mod.code_version(state)
        version_dir = self._write_version(version, files)
        total_lines = sum(len(content.splitlines()) for content in files.values())
        summary = f"生成 {version_dir.name}/（{len(files)} 个文件，共 {total_lines} 行）"
        if duplicates:
            summary += f"（已忽略 {duplicates} 个重复声明）"
        update = {"current_version_dir": str(version_dir), "version_files": sorted(files)}
        return update, summary, self._version_artifacts(version_dir.name, files), None

    def _extract_version_files(
        self,
        output: Mapping[str, Any],
    ) -> tuple[dict[str, str], int, int]:
        """从编码节点输出抽取版本文件表（含 ``file=`` 声明与量级上限校验）。"""
        return contracts.collect_version_files(_node_text(output, "code_text"), require_path=True)

    def _produce_review(
        self,
        fn: NodeFn,
        state: Mapping[str, Any],
        *,
        node_name: str = "review",
        view: str | None = None,
    ) -> tuple[dict[str, Any], str, list[str], _Listing | None]:
        """审查节点：契约校验 → ``workspace/review_N[_<视角>].json`` 落盘 → clean 判定（Design §5.2）。

        一次审查覆盖版本内**全部文件**（文件清单来自状态的 ``version_files``，location 一律
        版本内相对路径，Design §4.2.2）。执行有效性门禁由审查节点交回的 ``review_evidence``
        决定（本轮是否至少一次成功工具事件）；不返回该键的节点（脚本化假节点 / 注入桩）按
        "内联送达记录"视为满足。并行双视角下每个视角只写自己的**独立 key**
        （``findings_<视角>``）：审查轮次推进与 clean 判定归汇合节点，否则同一超级步的
        两个分支会互相覆盖。
        """
        round_number = state_mod.round_number_for_review(state)
        version = state_mod.code_version(state)
        files = _version_files_of(state)
        version_label = f"{state_mod.code_version_dir_name(version)}/"
        output = _call_node(fn, state, retry_feedback=None)
        evidence = bool(output.get(agents_mod.REVIEW_EVIDENCE_KEY, True))
        try:
            result = contracts.parse_review_output(
                _node_text(output, "review_text"),
                review_round=round_number,
                code_version=version,
                files=files,
                has_tool_evidence=evidence,
            )
        except ContractViolation as exc:
            output = self._retry(node_name, round_number, fn, state, str(exc))
            evidence = bool(output.get(agents_mod.REVIEW_EVIDENCE_KEY, True)) or evidence
            result = contracts.parse_review_output(
                _node_text(output, "review_text"),
                review_round=round_number,
                code_version=version,
                files=files,
                has_tool_evidence=evidence,
            )
        path = self._workspace_path() / state_mod.review_file_name(round_number, view=view)
        self._write_artifact(path, json.dumps(result.contract, ensure_ascii=False, indent=2) + "\n")
        counts = contracts.severity_counts(result.findings)
        scope = f"·{graph_mod.VIEW_LABELS[view]}视角" if view else ""
        summary = (
            f"第 {round_number} 轮{scope}（{version_label}，{len(files)} 个文件）："
            f"{counts[contracts.SEVERE]} 严重 / "
            f"{counts[contracts.MODERATE]} 一般 / {counts[contracts.ADVICE]} 建议"
        )
        listing = _Listing(
            kind="review",
            payload={
                "round_number": round_number,
                "version_label": version_label,
                "findings": list(result.findings),
                "summary": result.summary,
                "view": graph_mod.VIEW_LABELS.get(view) if view else None,
            },
        )
        if view is not None:
            # 分支只交回自己的独立 key；评述缓存供汇合产物合并两边评述用
            self._branch_summaries.setdefault(round_number, {})[view] = result.summary
            return (
                {graph_mod.findings_key_for_view(view): result.findings},
                summary,
                [self._relative(path)],
                listing,
            )
        if contracts.judge_clean(result.findings, severe_only=bool(state.get("severe_only"))):
            # "建议不影响判定"只在确有【建议】时说明；0 findings 时加括号属噪音
            tail = "（建议不影响判定）" if counts[contracts.ADVICE] else ""
            summary = f"{summary} —— 干净通过{tail}"
        summary = f"{summary}{self._convergence_tail(state, result.findings)}"
        update = {
            "findings": result.findings,
            "clean": contracts.judge_clean(
                result.findings, severe_only=bool(state.get("severe_only"))
            ),
            "review_round": round_number,
        }
        return update, summary, [self._relative(path)], listing

    def _produce_merge(
        self, fn: NodeFn, state: Mapping[str, Any]
    ) -> tuple[dict[str, Any], str, list[str], _Listing | None]:
        """汇合节点：合并双视角 findings → ``review_N.json`` → 当轮 clean 判定（Design §12.2）。

        ``findings`` 由本步**显式覆盖、无条件写**（未运行分支的旧值不得残留）；分支原始结论
        已分别落 ``review_N_<视角>.json``，合并结果落 ``review_N.json``——"这条结论由谁提出"
        可从 findings 的 ``views`` 字段追溯。
        """
        round_number = state_mod.round_number_for_review(state)
        version = state_mod.code_version(state)
        files = _version_files_of(state)
        version_label = f"{state_mod.code_version_dir_name(version)}/"
        output = _call_node(fn, state, retry_feedback=None)
        merged = [item for item in (output.get("findings") or []) if isinstance(item, Mapping)]
        contract_object = {
            "iteration": round_number,
            "code_version": version,
            "files": sorted(files),
            "findings": merged,
            "summary": self._merged_summary(round_number),
        }
        path = self._workspace_path() / state_mod.review_file_name(round_number)
        self._write_artifact(path, json.dumps(contract_object, ensure_ascii=False, indent=2) + "\n")
        counts = contracts.severity_counts(merged)
        clean = contracts.judge_clean(merged, severe_only=bool(state.get("severe_only")))
        summary = (
            f"第 {round_number} 轮（{version_label}，{len(files)} 个文件）："
            f"{counts[contracts.SEVERE]} 严重 / "
            f"{counts[contracts.MODERATE]} 一般 / {counts[contracts.ADVICE]} 建议（双视角合并）"
        )
        if clean:
            tail = "（建议不影响判定）" if counts[contracts.ADVICE] else ""
            summary = f"{summary} —— 干净通过{tail}"
        summary = f"{summary}{self._convergence_tail(state, merged)}"
        listing = _Listing(
            kind="review",
            payload={
                "round_number": round_number,
                "version_label": version_label,
                "findings": list(merged),
                "summary": str(contract_object["summary"]),
                "view": None,
            },
        )
        update = {"findings": merged, "clean": clean, "review_round": round_number}
        return update, summary, [self._relative(path)], listing

    def _merged_summary(self, round_number: int) -> str:
        """汇合产物的评述：把两个视角各自的整体评价并成一句（设计上不做二次模型总结）。"""
        cached = self._branch_summaries.get(round_number) or {}
        parts = [
            f"{graph_mod.VIEW_LABELS.get(view, view)}视角："
            f"{str(cached.get(view) or '').strip() or '（无补充评述）'}"
            for view in graph_mod.REVIEW_VIEWS
        ]
        return "；".join(parts)

    def _convergence_tail(
        self, state: Mapping[str, Any], findings: Sequence[Mapping[str, Any]]
    ) -> str:
        """未收敛标注（Design §4.4）：与上一轮问题集合一致时在阶段摘要标注，不提前中止。"""
        previous = state.get("findings") or []
        if not isinstance(previous, list) or not previous or not findings:
            return ""
        if not_converged({"findings": previous}, {"findings": list(findings)}):
            return "（未收敛：与上一轮问题集合一致，已记入报告）"
        return ""

    def _produce_fix(
        self, fn: NodeFn, state: Mapping[str, Any]
    ) -> tuple[dict[str, Any], str, list[str], _Listing | None]:
        """修复节点：契约校验 → ``code_vN+1/`` 与 ``fixes_N.json`` 落盘（Design §4.2.2 / §5.2）。

        修复必须给出版本内的全部文件（声明删除的除外）；删除只是**提出**——引擎先打印
        待删清单并经确认闸门征询 ``y/N``，拒绝 / 中断 / 非交互一律保留文件并在摘要与报告
        标注"待确认删除（未执行）"。"本轮未产生改动"按整版本 ``路径 → sha256`` 比较。
        """
        round_number = state_mod.round_number_for_fix(state)
        current_version = state_mod.code_version(state)
        new_version = current_version + 1
        findings = list(state.get("findings") or [])
        current_files = _version_files_of(state)
        output = _call_node(fn, state, retry_feedback=None)
        try:
            result = contracts.parse_fix_output(
                _node_text(output, "fix_text"), findings=findings, current_files=current_files
            )
        except ContractViolation as exc:
            output = self._retry("fix", round_number, fn, state, str(exc))
            result = contracts.parse_fix_output(
                _node_text(output, "fix_text"), findings=findings, current_files=current_files
            )
        # 删除确认闸门（Design §4.2.2）：确认前零删除；未确认的文件从上一版本目录保留下来
        deleted_confirmed, carried = self._resolve_deletions(
            result.deleted, findings, current_version
        )
        files = dict(result.files)
        files.update(carried)
        # 围栏抽取会剥掉内容首尾的换行，而磁盘上的版本内容是归一化的（单个结尾换行）：
        # 先归一再算指纹，"本轮未产生改动"的整版本比较才不会因这个表示差误报
        files = {path: _normalize_code(content) for path, content in files.items()}
        previous = self._previous_fingerprint(state)
        changed = _fingerprint(files) != previous
        version_dir = self._write_version(new_version, files)
        fixes_path = self._workspace_path() / state_mod.fixes_file_name(round_number)
        payload = {
            "round": round_number,
            "version_dir": version_dir.name,
            "files": sorted(files),
            "fixes": result.fixes,
            "unknown_ids": list(result.unknown_ids),
            "deleted": {
                "proposed": sorted(result.deleted),
                "confirmed": sorted(deleted_confirmed),
                "pending": sorted(path for path in result.deleted if path not in deleted_confirmed),
            },
        }
        self._write_artifact(fixes_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        statuses = [str(item.get("status")) for item in result.fixes]
        parts = [f"第 {round_number} 轮："]
        parts.append(f"修复 {statuses.count('fixed')} 项")
        if statuses.count("partial"):
            parts.append(f" / 部分修复 {statuses.count('partial')} 项")
        if statuses.count("skipped"):
            parts.append(f" / 跳过 {statuses.count('skipped')} 项")
        parts.append(f" → {version_dir.name}/（{len(files)} 个文件）")
        if not changed:
            parts.append("（本轮未产生改动）")
        if deleted_confirmed:
            parts.append(f"（已确认删除：{', '.join(sorted(deleted_confirmed))}）")
        pending = [path for path in result.deleted if path not in deleted_confirmed]
        if pending:
            parts.append(f"（待确认删除（未执行）：{', '.join(sorted(pending))}）")
        if result.duplicate_declarations:
            parts.append(f"（已忽略 {result.duplicate_declarations} 个重复声明）")
        if result.extra_fences:
            # Design §4.2.1：多余围栏忽略但留痕（写进阶段摘要即进 log.jsonl 与时间线）
            parts.append(f"（已忽略 {result.extra_fences} 个多余围栏）")
        summary = "".join(parts)
        if result.unknown_ids:
            summary = f"{summary}（未识别处置 {len(result.unknown_ids)} 条，见报告）"
        listing = _Listing(
            kind="fix",
            payload={
                "round_number": round_number,
                "fixes": list(result.fixes),
                "findings": findings,
            },
        )
        update = {
            "current_version_dir": str(version_dir),
            "version_files": sorted(files),
            "fix_rounds": round_number,
        }
        artifacts = [self._relative(fixes_path), *self._version_artifacts(version_dir.name, files)]
        return update, summary, artifacts, listing

    def _resolve_deletions(
        self,
        deleted: Sequence[str],
        findings: Sequence[Mapping[str, Any]],
        current_version: int,
    ) -> tuple[list[str], dict[str, str]]:
        """落实删除确认：返回 ``(已确认删除, 从上一版本保留下来的文件)``。

        确认闸门（``confirm_deletions``）缺失 / 返回否 / 抛异常时一律保留；保留的内容从
        上一版本目录读出——修复 Agent 对声明删除的文件可不给内容，版本完整性由这里兜底。
        """
        if not deleted:
            return [], {}
        previous_dir = self._workspace_path() / state_mod.code_version_dir_name(current_version)
        requests: list[dict[str, Any]] = []
        for path in deleted:
            reasons = [
                f"{finding.get('id')}【{finding.get('severity')}】{finding.get('issue')}"
                for finding in findings
                if _finding_in_file(finding, str(path))
            ]
            requests.append({"path": path, "reasons": reasons})
        confirmed: list[str] = []
        if self._confirm_deletions is not None and requests:
            try:
                if self._confirm_deletions(requests):
                    confirmed = list(deleted)
            except Exception:  # noqa: BLE001 — 确认闸门故障按"未确认"处理，删除永不静默发生
                confirmed = []
        carried: dict[str, str] = {}
        for path in deleted:
            if path in confirmed:
                continue
            try:
                # 与写入侧同一防线：拼接进文件系统路径前再验一次（模型输入路径是
                # 本项目风险最高的一类输入，读取边界不依赖上游自觉）
                safe_path = contracts.validate_version_path(path)
            except ContractViolation as exc:
                raise PersistenceError(f"待删除路径不合法：{exc}") from exc
            source = previous_dir / safe_path
            try:
                carried[path] = source.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                if isinstance(exc, FileNotFoundError):
                    # 上一版本里已不存在该文件（记录与磁盘脱节）：没有内容可"保留"，
                    # 继续按待确认标注即可，不把无操作升级成整轮运行失败
                    continue
                raise PersistenceError(
                    f"待删除文件 {path} 保留失败（无法从上一版本读出）：{exc}"
                ) from exc
        return confirmed, carried

    # ---- 收尾 ----

    def _finish(self, latest: Mapping[str, Any]) -> RunResult:
        """正常收尾：判定状态与退出码、写最终状态与报告（Design §4.4、§8.1）。"""
        state = dict(latest)
        clean = bool(state.get("clean"))
        fix_rounds = int(state.get("fix_rounds", 0))
        max_iterations = int(state.get("max_iterations", 0))
        if clean:
            status, exit_code, next_action = (
                state_mod.STATUS_SUCCESS,
                _EXIT_SUCCESS,
                state_mod.NEXT_DONE,
            )
        elif fix_rounds >= max_iterations:
            status, exit_code, next_action = (
                state_mod.STATUS_RESIDUAL,
                _EXIT_RESIDUAL,
                state_mod.NEXT_FIX_AGAIN,
            )
        else:
            # 条件边只有"干净"与"达限"两条出口（Design §4.4）：走到这里说明编排自检不通过，
            # 如实按运行失败处理，而不是编一个假的收尾结论。
            raise RuntimeError(
                f"工作流在未达轮次上限（{max_iterations}）且未干净时收尾：编排状态自检失败。"
            )
        state.update(
            {
                "status": status,
                "next_action": next_action,
                "finished_at": self._clock(),
            }
        )
        self._latest = state
        return self._finalize(state, exit_code)

    def _handle_node_failure(self, latest: Mapping[str, Any], failure: _NodeFailure) -> RunResult:
        """节点失败收尾（Design §4.6）：写 status/error + 时间线条目 + 报告，退出码 2。"""
        state = dict(latest)
        entry = state_mod.NodeEvent(
            node=failure.node,
            round=failure.round_number,
            started_at=failure.started_at,
            duration_ms=failure.duration_ms,
            summary=f"节点失败：{failure.message}",
            artifacts=[],
        )
        state.update(
            {
                "status": state_mod.STATUS_FAILED,
                "error": {"node": failure.node, "message": failure.message},
                "finished_at": self._clock(),
                "history": [*list(state.get("history") or []), entry],
                **self._usage_update(state, {failure.node: failure.usage}),
            }
        )
        self._latest = state
        return self._finalize(state, _EXIT_ERROR)

    def _usage_update(
        self, state: Mapping[str, Any], by_node: Mapping[str, Mapping[str, int]]
    ) -> dict[str, Any]:
        """把 ``节点 → 账`` 并入状态的用量表（绕过图 reducer 的收尾路径用）。

        全 0 的账不写字段：假节点 / 未回报用量的场景下，记录里应看不到"用量为 0"，
        否则读报告的人会以为这次运行真的只花了 0 token。
        """
        additions = {
            node: record
            for node, record in by_node.items()
            if record is not None and not usage_mod.is_empty(record)
        }
        if not additions:
            return {}
        return {"usage": usage_mod.merge_by_node(state.get("usage"), additions)}

    def _handle_interrupt(self, latest: Mapping[str, Any]) -> RunResult:
        """中断收尾（Design §4.6 / §8.2）：interrupt 事件 → 原子写 state.json → 退出码 2。

        会话轮次的状态写 ``interrupted``（§8.2 的会话轮次口径：中断的本轮可被会话识别为
        "被打断"而不是"运行失败"）；一次性运行按 §4.6 保持 ``failed`` + ``error.node=interrupt``。
        第二次 Ctrl+C 放弃写盘（"尽力"语义），仍按中断结论返回退出码 2。
        """
        state = dict(latest)
        session_round = self._request is not None and self._request.session_id is not None
        state.update(
            {
                "status": (
                    state_mod.STATUS_INTERRUPTED if session_round else state_mod.STATUS_FAILED
                ),
                "error": {"node": "interrupt", "message": "运行被用户中断（Ctrl+C）。"},
                "finished_at": self._clock(),
                **self._usage_update(state, self._drain_residual_usage()),
            }
        )
        self._latest = state
        active = self._any_active()
        node = active.node if active else "engine"
        try:
            self._emit(
                {
                    "node": node,
                    "event": "interrupt",
                    "detail": "用户中断（Ctrl+C）；正在写入运行记录。",
                }
            )
            self._persist_state(state)
        except KeyboardInterrupt:
            pass  # 第二次 Ctrl+C：放弃写盘
        except PersistenceError:
            pass  # 尽力语义：中断结论不因写盘失败而改变
        return self._result(state, _EXIT_ERROR, interrupted=True)

    def _handle_persistence_failure(self, latest: Mapping[str, Any], exc: Exception) -> RunResult:
        """落盘失败收尾（Design §4.6）：status=failed + 可读原因 + 已保留产物路径，退出码 2。

        失败归因固定为引擎级（``node="engine"``）：磁盘故障不是某个节点的产出问题，
        报成"审查节点失败"会把排查方向带偏。
        """
        state = dict(latest)
        message = f"{exc}（已保留产物：{self._run_dir}）"
        state.update(
            {
                "status": state_mod.STATUS_FAILED,
                "error": {"node": "engine", "message": message},
                "finished_at": self._clock(),
            }
        )
        self._latest = state
        # 尽力留痕与再写一次失败状态：写好可供 crw show / log 诊断；仍失败则放弃
        # （消息已带可读原因与保留路径，不会静默）
        with contextlib.suppress(PersistenceError, OSError):
            self._emit({"node": "engine", "event": "error", "detail": message})
        with contextlib.suppress(PersistenceError, OSError, KeyboardInterrupt):
            self._persist_state(state)
        return self._result(state, _EXIT_ERROR)

    def _finalize(self, state: Mapping[str, Any], exit_code: int) -> RunResult:
        """统一收尾写盘：state.json → report.md → 引擎 end 事件（Design §8.2）。

        收尾写盘**再被打断**（如失败收尾途中第二次 Ctrl+C）不改变本次运行的结论：
        按 §4.6 的"尽力"语义返回既有状态（写没写盘由磁盘上的记录为准），不让
        BaseException 带着裸 traceback 逃出引擎。
        """
        report_written = False
        try:
            self._persist_state(state)
            self._write_report(state)
            report_written = True
        except PersistenceError as exc:
            return self._handle_persistence_failure(state, exc)
        except KeyboardInterrupt:
            return self._result(state, exit_code)
        self._emit(
            {
                "node": "engine",
                "event": "end",
                "detail": str(state.get("status") or ""),
                "status": state.get("status"),
            }
        )
        return self._result(state, exit_code, report_written=report_written)

    def _write_report(self, state: Mapping[str, Any]) -> None:
        """生成 report.md（结构见 Design §4.5）；写入失败按运行失败处理。"""
        run_dir = self._run_dir_path()
        # report.md 自身尚不存在，与已落盘产物一起列出（设计口径：产物清单含报告）
        artifacts = [run_dir / "report.md", *records.run_artifacts(run_dir)]
        text = build_report(
            state=state,
            rounds=records.read_rounds(run_dir),
            run_dir=run_dir,
            artifacts=artifacts,
            parent_residual=self._parent_residual(state),
        )
        try:
            records.atomic_write_text(run_dir / "report.md", text)
        except OSError as exc:
            raise PersistenceError(
                f"report.md 写入失败：{exc}（状态已保留：{run_dir / 'state.json'}）"
            ) from exc

    def _parent_residual(self, state: Mapping[str, Any]) -> str | None:
        """续跑来源的残留摘要（Design §4.4 的报告口径）：读父运行的 state.json，读不到则留空。

        报告是收尾产物：父运行记录被清理 / 损坏时只是少一行标注，不得影响本次运行的结论。
        """
        parent = str(state.get("parent_run_id") or "")
        if not parent or not records.is_valid_run_id(parent):
            return None
        try:
            parent_dir = records.find_run_dir(parent)
            if parent_dir is None:
                return None
            parent_state = records.read_state(parent_dir)
            if not isinstance(parent_state, Mapping):
                return None
            findings = parent_state.get("findings")
            conclusion = conclusion_text(
                status=str(parent_state.get("status") or ""),
                review_round=int(parent_state.get("review_round") or 0),
                fix_rounds=int(parent_state.get("fix_rounds") or 0),
                findings=findings if isinstance(findings, list) else [],
                severe_only=bool(parent_state.get("severe_only")),
            )
        except (OSError, TypeError, ValueError):
            # 父记录被清理 / 字段被手改成非法值时少一行标注即可，报告本身必须照样产出
            return None
        return f"{parent}：{conclusion}"

    # ---- 基础设施 ----

    def _materialize_input_version(self, request: RunRequest, initial: RunState) -> None:
        """把运行输入物化为起始版本目录（``-f`` 单文件包成目录，续跑 / 会话轮次复制既有版本）。

        目录名取 ``request.start_version``（Design §4.7 的跨轮连续递增）：会话第 2 轮从
        上一轮的版本号继续，报告里的"第几版"在会话内始终指向同一份代码。
        需求文本输入没有初始版本（编码节点产出起始版本目录），此处不动作。

        **同时给出需求与既有来源**（``source_dir`` / ``source_text``）时**不物化**：该组合
        下入口路由走编码起步（需求优先，见 :meth:`RunRequest.input_descriptor`），既有代码
        只作编码上下文（``prior_files`` 内联进任务提示词）——物化会把一份"不在新版本清单里"
        的旧文件留在起始版本目录里，反而制造误读。命令面（会话的"改需求"轮次）本来就按
        "不物化 + ``prior_files``"走，这里让程序化调用取得同一语义。
        """
        if request.requirement and (
            request.source_dir is not None or request.source_text is not None
        ):
            if not initial.get("prior_files"):
                initial["prior_files"] = self._read_prior_source(request)
            return
        if request.source_dir is not None:
            files = read_version_files(request.source_dir)
        elif request.source_text is not None:
            name = request.source_path.name if request.source_path is not None else "main.py"
            files = {name: _normalize_code(request.source_text)}
        else:
            return
        version_dir = self._write_version(
            state_mod.normalize_version_base(request.start_version), files
        )
        initial["current_version_dir"] = str(version_dir)
        initial["version_files"] = sorted(files)

    def _read_prior_source(self, request: RunRequest) -> dict[str, str]:
        """既有来源（版本目录或文本）的读取口径：与物化共用同一套读出规则。"""
        if request.source_dir is not None:
            return read_version_files(request.source_dir)
        name = request.source_path.name if request.source_path is not None else "main.py"
        return {name: _normalize_code(request.source_text or "")}

    def _write_version(self, version: int, files: Mapping[str, str]) -> Path:
        """落盘一个版本目录：全部文件（原子写）→ ``manifest.json``，返回目录路径。

        manifest 最后写：它是"该版本包含哪些文件"的权威声明，出现半写文件比缺文件更误导。
        落盘失败按持久化失败处理（Design §4.6），不产生半更新的状态引用。
        """
        version_dir = self._workspace_path() / state_mod.code_version_dir_name(version)
        # 纵深防御：路径在契约层已校验过（拒绝绝对路径 / 盘符 / `..`），这里落盘前再验一次——
        # "模型输出拼进文件系统路径"是本项目风险最高的一类输入，写盘边界不能只靠上游自觉
        normalized = {
            contracts.validate_version_path(path): _normalize_code(content)
            for path, content in files.items()
        }
        try:
            for path, content in normalized.items():
                self._write_artifact(version_dir / path, content)
            manifest = contracts.manifest_payload(version=version, files=normalized)
            self._write_artifact(
                version_dir / "manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            )
        except PersistenceError as exc:
            raise PersistenceError(f"版本目录 {version_dir.name}/ 写入失败：{exc}") from exc
        return version_dir

    def _read_version_files(self, directory: Path) -> dict[str, str]:
        """读出版本目录的代码文件（委托模块级 :func:`read_version_files`）。"""
        return read_version_files(directory)

    def _previous_fingerprint(self, state: Mapping[str, Any]) -> dict[str, str]:
        """上一版本的 ``路径 → sha256`` 指纹（"本轮未产生改动"的比较基准）。

        上一版本目录缺失 / 不可读时返回空表——比较结果为"有改动"，宁可漏标注也不误标。
        """
        version = state_mod.code_version(state)
        previous_dir = self._workspace_path() / state_mod.code_version_dir_name(version)
        if not previous_dir.is_dir():
            return {}
        try:
            return _fingerprint(read_version_files(previous_dir))
        except PersistenceError:
            return {}

    def _version_artifacts(self, dir_name: str, files: Mapping[str, str]) -> list[str]:
        """版本内文件的产物名（相对运行目录、posix 风格，报告与时间线共用）。"""
        return sorted(f"workspace/{dir_name}/{path}" for path in files)

    def _default_run_id_factory(self) -> str:
        """默认运行号生成：按注入时钟的日期 + 碰撞重试（Design §4.3）。"""
        day = "".join(self._clock()[:10].split("-"))
        if len(day) != 8 or not day.isdigit():
            raise RunRecordError(f"时钟返回值无法解析出日期：{self._clock()!r}")
        return records.new_run_id(day=day)

    def _create_run(self) -> tuple[str, Path]:
        """创建运行目录并返回 ``(运行号, 运行目录)``。

        - 默认：``~/.crw/runs/<运行号>/``（编号碰撞时重新生成；并发 / 注入固定编号都可能撞）。
        - ``run_parent``：换个父目录（会话轮次在会话目录下的 ``rounds/`` 里新开记录）。
        - ``run_dir``：由调用方指定目录（会话轮次用 ``round-<i>/`` 命名，运行号只进
          ``state.json``）；目录已存在即报可读错误，不覆盖既有轮次记录。
        """
        if self._run_dir_override is not None:
            run_id = self._run_id_factory()
            return run_id, records.create_run_dir_at(self._run_dir_override)
        parent = self._run_parent
        last_error: Exception | None = None
        for _ in range(5):
            run_id = self._run_id_factory()
            try:
                return run_id, records.create_run_dir(run_id, parent=parent)
            except FileExistsError as exc:
                if records.run_dir(run_id, parent=parent).is_dir():
                    last_error = exc
                    continue
                raise RunRecordError(
                    f"无法创建运行目录（{records.run_dir(run_id, parent=parent)}）：{exc}"
                ) from exc
        raise RunRecordError(f"运行号连续碰撞，无法创建运行目录：{last_error}")

    def _persist_state(self, state: Mapping[str, Any]) -> None:
        """原子写 state.json（节点边界与失败时各一次，Design §8.2）。"""
        self._persist_state_text(state_mod.dumps(state))

    def _persist_state_text(self, text: str) -> None:
        """原子写已序列化的状态快照（序列化失败不产生半写文件）。"""
        try:
            records.atomic_write_text(self._run_dir_path() / "state.json", text)
        except OSError as exc:
            raise PersistenceError(f"state.json 写入失败：{exc}") from exc

    def _write_artifact(self, path: Path, text: str) -> None:
        """原子写节点产物（代码版本 / 审查结论 / 修复处置）。

        与 state.json / report.md 同口径：落盘失败按"落盘失败"报告（Design §4.6），
        而不是被归为节点自身的失败——两者都退出 2，但原因定位必须准确。
        """
        try:
            records.atomic_write_text(path, text)
        except OSError as exc:
            raise PersistenceError(f"产物 {path.name} 写入失败：{exc}") from exc

    def _notify_observer(self, callback: Callable[[], None]) -> None:
        """执行渲染回调并隔离输出流故障。

        渲染是呈现层：终端已关闭 / 下游退出（``BrokenPipeError`` 等 ``OSError``）时无处可报，
        运行记录（state.json / log.jsonl）仍是真相来源——这类故障不得改变运行结论，也不得
        被误报为节点失败（Design §4.6 只把落盘失败列为运行失败）。渲染器自身的编码缺陷
        （非 ``OSError``）照常抛出，不在此掩盖。
        """
        try:
            callback()
        except OSError:
            pass

    def _retry(
        self, node: str, round_number: int, fn: NodeFn, state: Mapping[str, Any], feedback: str
    ) -> Mapping[str, Any]:
        """契约校验失败后的重试（Design §4.6）：发 retry 事件、附错误清单重试一次。"""
        self._emit(
            {
                "node": node,
                "event": "retry",
                "iteration": round_number,
                "detail": f"契约校验未通过，携错误清单重试一次：{feedback}",
            }
        )
        return _call_node(fn, state, retry_feedback=feedback)

    def _render_listing(self, listing: _Listing | None) -> None:
        """把清单载荷交给观察者渲染（node end 事件之后，Design §6.2 #6 的阶段序列）。"""
        if listing is None:
            return
        if listing.kind == "review":
            self._notify_observer(lambda: self._observer.on_review(**listing.payload))
        else:
            self._notify_observer(lambda: self._observer.on_fixes(**listing.payload))

    def _result(
        self,
        state: Mapping[str, Any],
        exit_code: int,
        *,
        interrupted: bool = False,
        message: str | None = None,
        report_written: bool = False,
    ) -> RunResult:
        """构造结果视图；``message`` 供运行目录都无法创建时给出可读原因。"""
        error = state.get("error") if isinstance(state.get("error"), Mapping) else {}
        error_node = error.get("node") if isinstance(error, Mapping) else None
        error_message = error.get("message") if isinstance(error, Mapping) else None
        if message is not None:
            error_message = message
            error_node = error_node or "engine"
        return RunResult(
            run_id=str(state.get("run_id") or ""),
            run_dir=self._run_dir,
            status=str(state.get("status") or state_mod.STATUS_FAILED),
            exit_code=exit_code,
            review_round=int(state.get("review_round", 0)),
            fix_rounds=int(state.get("fix_rounds", 0)),
            clean=bool(state.get("clean")),
            findings=tuple(state.get("findings") or ()),
            error_node=str(error_node) if error_node else None,
            error_message=str(error_message) if error_message else None,
            interrupted=interrupted,
            state=dict(state),
            report_written=report_written,
        )

    def _start_detail(self, request: RunRequest) -> str:
        """运行开始事件的 detail：输入摘要 + 轮次上限与来源（log 预览与渲染头行共用）。

        ``-f`` 输入额外标注"原文件名 → 工作区名"映射（Design §4.5 的报告口径）；续跑、
        会话轮次与并行审查各加一句，使终端头行与日志能一眼看出这一轮是怎么起的。
        """
        descriptor = request.input_descriptor()
        start_label = state_mod.code_version_dir_name(
            state_mod.normalize_version_base(request.start_version)
        )
        if descriptor.get("kind") == state_mod.INPUT_FILE:
            source = str(descriptor.get("source_path") or "")
            if request.source_dir is not None:
                summary = f"父运行版本目录：{source} → {start_label}/"
            else:
                summary = f"文件：{source} → {start_label}/{Path(source).name}"
        else:
            summary = state_mod.summarize_input(descriptor, limit=80)
            source = str(descriptor.get("source_path") or "")
            if source:
                # 会话轮次：需求驱动编码起步、既有代码来自上一版本或用户文件（Design §4.7）；
                # 沿用输入摘要的公开口径，让"版本目录 / 文件"两种来源在头行上一眼可分
                label = state_mod.summarize_input(
                    {"kind": state_mod.INPUT_FILE, "source_path": source}, limit=120
                )
                summary = f"{summary}（基于 {label} 继续开发 → {start_label}/）"
        criterion = "，判定已放宽为仅【严重】阻塞（--severe-only）" if request.severe_only else ""
        parts = [
            (
                f"{summary}；修复轮次上限 {request.max_iterations}"
                f"（来源：{source_label(request.max_iterations_source)}）{criterion}"
            )
        ]
        if request.rounds_mode == state_mod.ROUNDS_EXACT:
            parts.append("轮次模式 exact（跑满指定轮次）")
        if request.parent_run_id:
            parts.append(f"续跑自 {request.parent_run_id}")
        if request.session_id:
            rounds = f"第 {request.round_index} 轮" if request.round_index else "本轮"
            parts.append(f"会话 {request.session_id} 的{rounds}")
        if request.parallel_review:
            parts.append("并行双视角审查（正确性 + 安全性）")
        return "；".join(parts)

    def _round_of(self, name: str, state: Mapping[str, Any]) -> int:
        """节点所属轮次：编码恒为 1；修复按已完成修复数；审查与汇合按已完成的审查数。"""
        if name == "fix":
            return state_mod.round_number_for_fix(state)
        if name == "encode":
            return 1
        return state_mod.round_number_for_review(state)

    def _elapsed_ms(self, active: _Active) -> int:
        return max(0, int((self._timer() - active.started) * 1000))

    def _workspace_path(self) -> Path:
        if self._workspace is None:  # pragma: no cover — execute 前调用属编程缺陷
            raise PersistenceError("运行工作区尚未建立。")
        return self._workspace

    def _run_dir_path(self) -> Path:
        if self._run_dir is None:  # pragma: no cover — execute 前调用属编程缺陷
            raise PersistenceError("运行目录尚未建立。")
        return self._run_dir

    def _relative(self, path: Path) -> str:
        """产物路径 → 相对运行目录的展示形式（正斜杠，跨平台一致）。"""
        run_dir = self._run_dir_path()
        with contextlib.suppress(ValueError):
            return path.relative_to(run_dir).as_posix()
        return str(path)


def build_report(
    *,
    state: Mapping[str, Any],
    rounds: Sequence[RoundReport],
    run_dir: Path,
    artifacts: Sequence[Path],
    parent_residual: str | None = None,
) -> str:
    """生成运行报告（结构见 Design §4.5）：摘要 → 时间线 → 逐轮问题清单 → 产物清单。"""
    status = str(state.get("status") or "")
    review_round = int(state.get("review_round", 0))
    fix_rounds = int(state.get("fix_rounds", 0))
    findings = state.get("findings") if isinstance(state.get("findings"), list) else []
    severe_only = bool(state.get("severe_only"))
    input_descriptor = state.get("input") if isinstance(state.get("input"), Mapping) else {}
    conclusion = conclusion_text(
        status=status,
        review_round=review_round,
        fix_rounds=fix_rounds,
        findings=findings,
        severe_only=severe_only,
    )

    lines: list[str] = [f"# 运行报告 · {_cell(state.get('run_id') or '（未知运行号）')}", ""]
    lines.append(f"**结论：{conclusion}。**")
    lines.append("")
    lines.append("## 运行摘要")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 输入 | {_cell(state_mod.summarize_input(input_descriptor, limit=200))} |")
    lines.append(f"| 模型 | {_cell(state.get('model') or '（未记录）')} |")
    lines.append(f"| 状态 | {status_label(status)} |")
    if state.get("parallel_review"):
        lines.append(
            "| 审查方式 | 双视角并行（正确性 + 安全性，汇合后按 (location, issue) 去重） |"
        )
    lines.append(f"| 审查 / 修复轮次 | {review_round} / {fix_rounds} |")
    lines.append(
        f"| 轮次上限 | {_cell(state.get('max_iterations'))}（来源：{source_label(state.get('max_iterations_source'))}） |"
    )
    lines.append(
        "| 干净判定 | "
        + (
            "无【严重】且无【一般】（默认判据）"
            if not severe_only
            else "无【严重】（--severe-only 放宽）"
        )
        + " |"
    )
    lines.append(
        f"| 开始 / 结束 | {_cell(state.get('created_at'))} → {_cell(state.get('finished_at'))} |"
    )
    if state.get("parent_run_id"):
        lines.append(f"| 父运行 | {_cell(state.get('parent_run_id'))} |")
        if parent_residual:
            lines.append(f"| 父运行残留 | {_cell(parent_residual)} |")
    if state.get("session_id"):
        lines.append(
            f"| 会话 / 轮次 | {_cell(state.get('session_id'))} / 第 {_cell(state.get('round_index'))} 轮 |"
        )
    lines.append("")

    lines.append("## 时间线")
    lines.append("")
    history = state.get("history") if isinstance(state.get("history"), list) else []
    if history:
        lines.append("| # | 节点 | 轮次 | 开始时间 | 耗时(ms) | 摘要 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for index, entry in enumerate(history, start=1):
            lines.append(
                f"| {index} | {node_label(entry.get('node'))} | {_cell(entry.get('round', ''))} | "
                f"{_cell(entry.get('started_at', ''))} | {_cell(entry.get('duration_ms', ''))} | "
                f"{_cell(entry.get('summary', ''))} |"
            )
    else:
        lines.append("（无节点事件）")
    lines.append("")

    usage_section = _usage_section(state)
    if usage_section:
        lines.extend(usage_section)

    lines.append("## 逐轮问题清单")
    lines.append("")
    if rounds:
        for index, item in enumerate(rounds):
            previous = rounds[index - 1] if index else None
            lines.extend(
                _round_section(
                    item,
                    status=status,
                    not_converged=bool(previous and not_converged(previous.review, item.review)),
                )
            )
    else:
        lines.append("（无审查结论）")
    lines.append("")

    lines.append("## 产物清单")
    lines.append("")
    lines.append(f"- 运行目录：`{run_dir}`")
    source_path = (
        input_descriptor.get("source_path") if isinstance(input_descriptor, Mapping) else None
    )
    mapped_name = Path(str(source_path)).name if source_path else ""
    start_dir = state_mod.code_version_dir_name(_version_base_of(state))
    for path in artifacts:
        relative = _relative_display(path, run_dir)
        if source_path and mapped_name and relative == f"workspace/{start_dir}/{mapped_name}":
            lines.append(f"- `{relative}`（原文件 → 工作区：`{source_path}`）")
        else:
            lines.append(f"- `{relative}`")
    lines.append("")

    error = state.get("error") if isinstance(state.get("error"), Mapping) else None
    if error:
        lines.append("## 错误")
        lines.append("")
        lines.append(f"- 节点：{node_label(error.get('node'))}")
        lines.append(f"- 说明：{_cell(error.get('message'))}")
        lines.append("")
    return "\n".join(lines)


def _append_notices(summary: str, notices: Sequence[str]) -> str:
    """把运行时的护栏触达记录并入阶段摘要（进 log.jsonl 与时间线，Design §4.6）。"""
    if not notices:
        return summary
    return f"{summary}（{'；'.join(notices)}，结论可能不完整）"


#: 用量小节的表头（列序与 :func:`crw.usage.cells` 的单元一一对应）
_USAGE_HEADER = ("节点", "调用", "输入", "缓存命中", "输出", "其中思考")
_USAGE_DIVIDER = "| --- | --- | --- | --- | --- | --- |"


def _usage_section(state: Mapping[str, Any]) -> list[str]:
    """Token 用量小节：按节点一行 + 合计行（无用量数据时整节省略）。

    节点行顺序 = 用量首次记进状态的顺序（``usage`` 按键并入、先写的键在前），与时间线的
    阅读顺序一致；未回报的调用由 :func:`crw.usage.cells` 写成"未回报"而不是 0。
    """
    by_node = state.get("usage")
    if not isinstance(by_node, Mapping):
        return []
    totals = usage_mod.totals(by_node)
    if usage_mod.is_empty(totals):
        return []
    lines = ["## Token 用量", "", f"| {' | '.join(_USAGE_HEADER)} |", _USAGE_DIVIDER]
    for node, record in by_node.items():
        cells = usage_mod.cells(usage_mod.merge(usage_mod.empty(), record))
        # 节点名经 _cell 转义：state.json 是"可能被手工改过"的输入，含 | 或换行的键不得串列
        lines.append("| " + " | ".join([_cell(node_label(node)), *cells]) + " |")
    lines.append("| " + " | ".join(["**合计**", *usage_mod.cells(totals)]) + " |")
    lines.append("")
    note = usage_mod.note(totals)
    hint = usage_mod.hint(totals)
    if note:
        lines.append(f"注：{note}。")
    if hint:
        lines.append(f"提示：{hint}")
    if note or hint:
        lines.append("")
    return lines


def _cell(value: Any) -> str:
    """Markdown 表格单元格：转义竖线并折叠换行（摘要 / 模型标识等可能含任意字符）。"""
    return " ".join(str(value if value is not None else "").split()).replace("|", "\\|")


def _round_section(item: RoundReport, *, status: str, not_converged: bool = False) -> list[str]:
    """单轮问题清单：标题固定为"第 i 轮审查（基于 code_vi/）"，问题按文件分组（Design §4.5）。

    ``not_converged`` 为真时插入"未收敛"标注（Design §4.4）：问题集合与上一轮一致说明
    修复没有改变问题清单——如实标出，但不改变流程与退出码。
    """
    lines: list[str] = [f"### 第 {item.round_number} 轮审查（基于 `{item.version_label}`）", ""]
    findings = item.review.get("findings")
    findings = findings if isinstance(findings, list) else []
    if not findings:
        lines.append("本轮无问题。")
        lines.append("")
        return lines
    summary = item.review.get("summary")
    if isinstance(summary, str) and summary.strip():
        lines.append(f"{summary}")
        lines.append("")
    if not_converged:
        lines.append(
            "> ⚠ **未收敛**：本轮问题集合与上一轮完全相同（location + issue 集合一致），"
            "修复未改变问题清单；报告如实标注，运行未因此提前中止。"
        )
        lines.append("")
    dispositions = _dispositions(item)
    lines.extend(_deletion_notes(item))
    for file_name, group in _group_by_file(findings, item.files):
        if item.files:
            lines.append(f"#### `{file_name}`")
            lines.append("")
        for finding in group:
            if not isinstance(finding, Mapping):
                continue
            severity = finding.get("severity") or "未分级"
            location = finding.get("location") or "（未定位）"
            issue = finding.get("issue") or ""
            lines.append(f"- **【{severity}】** `{location}` — {issue}{_views_note(finding)}")
            suggestion = finding.get("suggestion")
            if isinstance(suggestion, str) and suggestion.strip():
                lines.append(f"  - 修改建议：{suggestion}")
            if finding.get("uncertain"):
                lines.append("  - 备注：结论存在不确定性，需人工确认。")
            lines.append(f"  - 处置：{dispositions.get(str(finding.get('id')), '—')}")
        lines.append("")
    unknown = item.fixes.get("unknown_ids") if isinstance(item.fixes, Mapping) else None
    if isinstance(unknown, list) and unknown:
        lines.append(
            f"  - 未识别的处置（已忽略）：{', '.join(str(item_id) for item_id in unknown)}"
        )
    lines.append("")
    return lines


def _group_by_file(findings: Sequence[Any], files: Sequence[str]) -> list[tuple[str, list[Any]]]:
    """findings 按文件分组（Design §4.2.2 的"逐轮清单按文件分组"）。

    分组键 = location 的文件部分（``相对路径:行号`` 的 ``:`` 前段，或整文件级 location）；
    版本文件清单给出分组次序，location 对不上清单的条目归入末尾的"其他"组（清单为空时
    不分组，保持原有的平铺形态）。
    """
    if not files:
        return [("", list(findings))]
    groups: dict[str, list[Any]] = {name: [] for name in files}
    other: list[Any] = []
    for finding in findings:
        if not isinstance(finding, Mapping):
            other.append(finding)
            continue
        location = str(finding.get("location") or "")
        path = location.rsplit(":", 1)[0] if ":" in location else location
        target = groups.get(path)
        if target is None:
            target = groups.get(location)
        if target is None:
            other.append(finding)
        else:
            target.append(finding)
    result = [(name, groups[name]) for name in files if groups[name]]
    if other:
        result.append(("其他", other))
    return result


def _deletion_notes(item: RoundReport) -> list[str]:
    """删除处置的报告标注（Design §4.2.2）：已确认删除 / 待确认删除（未执行）。"""
    if not isinstance(item.fixes, Mapping):
        return []
    deleted = item.fixes.get("deleted")
    if not isinstance(deleted, Mapping):
        return []
    notes: list[str] = []

    def _names(key: str) -> str:
        values = deleted.get(key)
        return ", ".join(str(value) for value in values) if isinstance(values, list) else ""

    confirmed = _names("confirmed")
    pending = _names("pending")
    if confirmed:
        notes.append(f"> 已确认删除：{confirmed}。")
        notes.append("")
    if pending:
        notes.append(
            f"> ⚠ **待确认删除（未执行）**：{pending}——删除需要你在终端确认（y/N），"
            "未确认的文件已原样保留在本版本中。"
        )
        notes.append("")
    return notes


def _views_note(finding: Mapping[str, Any]) -> str:
    """并行双视角的来源标注（findings 的附加字段 ``views``，单视角审查时为空串）。

    汇合节点写入的是显示名；此处再按显示名表映射一次，手改成内部键（``correctness``）的
    记录也能渲染成中文，不至于同一份报告里两种口径并存。
    """
    views = finding.get("views")
    if not isinstance(views, list) or not views:
        return ""
    labels = [graph_mod.VIEW_LABELS.get(str(view), str(view)) for view in views]
    return f"（视角：{' + '.join(labels)}）"


def _dispositions(item: RoundReport) -> dict[str, str]:
    """finding_id → 处置文本（来自 ``fixes_N.json``；末轮无处置时说明原因）。"""
    if not isinstance(item.fixes, Mapping):
        return {}
    fixes = item.fixes.get("fixes")
    if not isinstance(fixes, list):
        return {}
    mapping: dict[str, str] = {}
    for fix in fixes:
        if not isinstance(fix, Mapping):
            continue
        finding_id = str(fix.get("finding_id"))
        note = str(fix.get("note") or "").strip()
        mapping[finding_id] = f"**{fix.get('status')}**{f' — {note}' if note else ''}"
    return mapping


def conclusion_text(
    *,
    status: str,
    review_round: int,
    fix_rounds: int,
    findings: Sequence[Mapping[str, Any]] = (),
    severe_only: bool = False,
) -> str:
    """收尾结论句（报告与终端渲染共用单一来源，避免两处口径漂移）。

    阻塞性结论数与干净判定同源（``contracts.blocking_count``）：默认判据把【严重】与
    【一般】都算阻塞，``severe_only`` 放宽为只算【严重】——残留可能全部由【一般】构成，
    文案因此不得写死"【严重】"。
    """
    if status == state_mod.STATUS_SUCCESS:
        return f"干净通过（共 {review_round} 轮审查 / {fix_rounds} 轮修复）"
    if status == state_mod.STATUS_RESIDUAL:
        blocking = contracts.blocking_count(findings, severe_only=severe_only)
        kind = "项【严重】残留" if severe_only else "项阻塞问题（【严重】或【一般】）残留"
        return (
            f"达修复轮次上限，仍有 {blocking} {kind}"
            f"（共 {review_round} 轮审查 / {fix_rounds} 轮修复）"
        )
    if status == state_mod.STATUS_INTERRUPTED:
        return "运行被中断（Ctrl+C），已产出产物保留"
    if status == state_mod.STATUS_FAILED:
        return "运行失败，已产出产物保留；失败细节见 state.json 与 log.jsonl"
    return f"运行未收尾（状态：{status or '未知'}）"


def status_label(status: str) -> str:
    """运行状态的显示名（报告、列表与详情共用）；``unknown`` = 由日志也读不出来的降级态。"""
    return {
        state_mod.STATUS_RUNNING: "运行中",
        state_mod.STATUS_SUCCESS: "干净通过",
        state_mod.STATUS_RESIDUAL: "达限残留",
        state_mod.STATUS_FAILED: "运行失败",
        state_mod.STATUS_INTERRUPTED: "已中断",
        "unknown": "未知（记录损坏）",
    }.get(status, status or "未知")


def source_label(source: Any) -> str:
    """轮次取值来源的显示名（Design §4.4 的来源枚举）；缺失时给可读占位而非 ``None``。"""
    if source is None or not str(source).strip():
        return "（未记录）"
    return {
        state_mod.SOURCE_CLI: "命令行",
        state_mod.SOURCE_REQUIREMENT: "需求文本提取",
        state_mod.SOURCE_CONFIG: "配置默认",
        state_mod.SOURCE_PARENT: "继承父运行",
        state_mod.SOURCE_DEFAULT: "内置默认",
    }.get(str(source), str(source))


def node_label(node: Any) -> str:
    """节点名的显示名（编码 / 审查 / 修复；并行形态含视角与汇合）。"""
    return NODE_LABELS.get(str(node), str(node))


def not_converged(previous: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    """未收敛判定（Design §4.4）：两份审查契约的 ``(location, issue)`` 集合完全相同。

    Args:
        previous / current: ``review_N.json`` 形态的契约对象（含 ``findings`` 数组）。

    Returns:
        上一轮与本轮问题集合一致（且两轮都非空）时为真——报告据此标注"未收敛"，
        但仍然**不提前中止**（标注 ≠ 改变流程）。
    """
    before = previous.get("findings") if isinstance(previous.get("findings"), list) else []
    after = current.get("findings") if isinstance(current.get("findings"), list) else []
    if not before or not after:
        return False
    return contracts.finding_keys(before) == contracts.finding_keys(after)


def _relative_display(path: Path, run_dir: Path) -> str:
    with contextlib.suppress(ValueError):
        return path.relative_to(run_dir).as_posix()
    return str(path)


def _normalize_code(code: str) -> str:
    """代码文本归一：去掉首尾换行、统一以单个换行结尾（行内空白原样保留）。

    围栏抽取会剥掉内容首尾的换行，磁盘版本则统一为"单个结尾换行"——两处表示一致，
    整版本逐字节比较才有意义。
    """
    return code.strip("\n") + "\n"


def read_version_files(directory: Path) -> dict[str, str]:
    """读出版本目录的代码文件：以 ``manifest.json`` 清单为准（Design §4.2.2）。

    manifest 是引擎落盘时写的权威清单；按它读而不是全目录递归，才能把"审查 / 修复
    Agent 执行代码时落进版本目录的副产物"（`__pycache__`、被测程序自己写出的
    数据文件等，真实演示暴露）挡在版本内容之外——否则 resume 复制、指纹比较与
    修复覆盖检查都会把副产物当版本文件。缺 manifest（旧目录 / 手工介入）时退回
    全目录读取，并排除字节码与隐藏 / 临时文件。

    会话层（命令面）读取既有版本内容时复用本函数：版本内容的唯一权威口径只在此处。
    """
    manifest_files = _manifest_file_names(directory)
    names = sorted(manifest_files) if manifest_files is not None else None
    if names is None:
        names = [
            path.relative_to(directory).as_posix()
            for path in sorted(directory.rglob("*"))
            if path.is_file()
            and path.name != "manifest.json"
            and not path.name.startswith(".")
            and not path.name.endswith(".tmp")
            and path.suffix != ".pyc"
            and "__pycache__" not in path.relative_to(directory).parts
        ]
    files: dict[str, str] = {}
    for name in names:
        try:
            files[name] = (directory / name).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise PersistenceError(f"版本文件 {name} 读取失败：{exc}") from exc
    return files


def _manifest_file_names(directory: Path) -> list[str] | None:
    """读 manifest.json 的文件清单；缺失 / 损坏返回 ``None``（调用方退回全目录读取）。"""
    manifest_path = directory / "manifest.json"
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    entries = data.get("files") if isinstance(data, Mapping) else None
    if not isinstance(entries, list):
        return None
    names = [
        str(item.get("path"))
        for item in entries
        if isinstance(item, Mapping) and isinstance(item.get("path"), str) and item.get("path")
    ]
    return names or None


def _version_files_of(state: Mapping[str, Any]) -> list[str]:
    """读取状态的版本文件清单（``version_files``）；缺失或含空项时给可用子集。"""
    files = state.get("version_files")
    if not isinstance(files, list):
        return []
    return [str(item) for item in files if isinstance(item, str) and item.strip()]


def _version_base_of(state: Mapping[str, Any]) -> int:
    """状态的起始版本号（展示与报告用）：坏值按 1 处理（口径见 state.normalize_version_base）。"""
    return state_mod.normalize_version_base(state.get("version_base"))


def _fingerprint(files: Mapping[str, str]) -> dict[str, str]:
    """版本指纹：``路径 → sha256``（"本轮未产生改动"的整版本比较口径，Design §4.2.2）。"""
    return {
        path: hashlib.sha256(content.encode("utf-8")).hexdigest() for path, content in files.items()
    }


def _finding_in_file(finding: Mapping[str, Any], path: str) -> bool:
    """finding 的 location 是否指向版本内该文件（``path`` 或 ``path:行号``）。"""
    location = str(finding.get("location") or "")
    return location == path or location.startswith(f"{path}:")


def _node_text(output: Mapping[str, Any], key: str) -> str:
    """读取节点交回的原始文本（节点输出协议的暂存键，Design §3.1）。"""
    value = output.get(key) if isinstance(output, Mapping) else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"节点未按协议返回 {key}（应为非空字符串）")
    return value


def _call_node(
    fn: NodeFn, state: Mapping[str, Any], *, retry_feedback: str | None
) -> Mapping[str, Any]:
    """按节点协议调用：声明 ``retry_feedback`` 的节点回传错误清单，其余节点忽略重试反馈。"""
    if retry_feedback is not None and _accepts_retry_feedback(fn):
        return fn(state, retry_feedback=retry_feedback)
    return fn(state)


def _accepts_retry_feedback(fn: NodeFn) -> bool:
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover — 内置函数等无签名对象
        return False
    if "retry_feedback" in parameters:
        return True
    return any(item.kind is item.VAR_KEYWORD for item in parameters.values())


def _readable_error(exc: Exception) -> str:
    """异常 → 可读报错（常量截断，避免 traceback 成为唯一线索，Design §4.6）。"""
    if isinstance(exc, RecursionError) or "recursion limit" in str(exc).lower():
        return "工作流超出递归护栏（recursion_limit）：可能存在未收敛的循环。"
    text = f"{type(exc).__name__}: {exc}"
    if len(text) <= MAX_ERROR_CHARS:
        return text
    return f"{text[:MAX_ERROR_CHARS]}…（已截断）"
