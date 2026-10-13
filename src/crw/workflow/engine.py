"""运行驱动：工作区建立、图执行、节点边界事件与快照、报告生成与退出码确定。

状态载体是图执行上下文本身，本模块只做生命周期组织，不维护第二份内存正本（Design §3.1、§4.5）。
节点输出经本模块的契约包装层校验、重试与落盘：节点自身不接触文件、不打印、不写日志（Design §3.1、§4.2.1）。
"""

from __future__ import annotations

import contextlib
import datetime
import inspect
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .. import agents as agents_mod
from .. import config as config_mod
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


def local_now() -> str:
    """默认时钟：本地时区 ISO 8601（秒级）。"""
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


class PersistenceError(RuntimeError):
    """运行记录落盘失败（Design §4.6：按运行失败处理，退出码 2）。"""


class _NodeFailure(RuntimeError):
    """节点失败：携带节点名、可读消息与时间信息，供 engine 写状态与时间线。"""

    def __init__(
        self, *, node: str, message: str, round_number: int, started_at: str, duration_ms: int
    ):
        super().__init__(message)
        self.node = node
        self.message = message
        self.round_number = round_number
        self.started_at = started_at
        self.duration_ms = duration_ms


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
        file_name: str,
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
        file_name: str,
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
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_iterations_source: str = state_mod.SOURCE_DEFAULT
    rounds_mode: str = state_mod.ROUNDS_UP_TO
    strict: bool = False
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

    def input_descriptor(self) -> state_mod.InputDescriptor:
        """本次运行的输入描述（Design §4.3 的 ``input`` 字段）。"""
        if self.source_path is not None:
            return state_mod.InputDescriptor(
                kind=state_mod.INPUT_FILE, source_path=str(self.source_path)
            )
        return state_mod.InputDescriptor(
            kind=state_mod.INPUT_REQUIREMENT, requirement=self.requirement or ""
        )

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
    ) -> None:
        self._nodes = dict(nodes) if nodes else None
        self._runtime_factory = runtime_factory
        self._clock = clock or local_now
        self._timer = timer or time.monotonic
        self._run_id_factory = run_id_factory or self._default_run_id_factory
        self._observer: RunObserver = observer or _NullObserver()
        self._run_dir: Path | None = None
        self._workspace: Path | None = None
        self._log_path: Path | None = None
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
            strict=request.strict,
            parallel_review=request.parallel_review,
            model=request.model_label,
            created_at=self._clock(),
            parent_run_id=request.parent_run_id,
            session_id=request.session_id,
            round_index=request.round_index,
        )
        self._latest = dict(initial)
        try:
            if request.source_text is not None:
                code_path = self._workspace / state_mod.code_file_name(1)
                self._write_artifact(code_path, _normalize_code(request.source_text))
                initial["current_code"] = str(code_path)
            self._persist_state(initial)
            self._emit(
                {
                    "node": "engine",
                    "event": "start",
                    "detail": self._start_detail(request),
                    "run_id": run_id,
                    "status": state_mod.STATUS_RUNNING,
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

    def _drain_notices(self) -> list[str]:
        """取走运行时的护栏触达记录（工具轮次 / 上下文预算），并入阶段摘要。"""
        if self._runtime is None:
            return []
        try:
            return self._runtime.drain_notices()
        except Exception:  # noqa: BLE001 — 摘要附加信息不得影响运行结论
            return []

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
                message = _append_notices(message, self._drain_notices())
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
                ) from exc
            duration = self._elapsed_ms(active)
            summary = _append_notices(summary, self._drain_notices())
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
            return {**update, "history": [entry]}

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
        return self._produce_review(fn, state, view=graph_mod.view_of_review_node(name))

    def _produce_encode(
        self, fn: NodeFn, state: Mapping[str, Any]
    ) -> tuple[dict[str, Any], str, list[str], _Listing | None]:
        """编码节点：抽取单文件代码 → 落盘 ``workspace/code_v1.py``（Design §4.2.1）。"""
        output = _call_node(fn, state, retry_feedback=None)
        try:
            code = _single_code_fence(_node_text(output, "code_text"))
        except ContractViolation as exc:
            output = self._retry("encode", 1, fn, state, str(exc))
            code = _single_code_fence(_node_text(output, "code_text"))
        normalized = _normalize_code(code)
        path = self._workspace_path() / state_mod.code_file_name(1)
        self._write_artifact(path, normalized)
        summary = f"生成 {path.name}（{len(normalized.splitlines())} 行）"
        return {"current_code": str(path)}, summary, [self._relative(path)], None

    def _produce_review(
        self, fn: NodeFn, state: Mapping[str, Any], *, view: str | None = None
    ) -> tuple[dict[str, Any], str, list[str], _Listing | None]:
        """审查节点：契约校验 → ``workspace/review_N[_<视角>].json`` 落盘 → clean 判定（Design §5.2）。

        执行有效性门禁由审查节点交回的 ``review_evidence`` 决定（本轮是否至少一次成功
        工具事件）；不返回该键的节点（脚本化假节点 / 注入桩）按"内联送达记录"视为满足。
        并行双视角下每个视角只写自己的**独立 key**（``findings_<视角>``）：审查轮次推进与
        clean 判定归汇合节点，否则同一超级步的两个分支会互相覆盖。
        """
        round_number = state_mod.round_number_for_review(state)
        version = state_mod.code_version(state)
        file_name = state_mod.code_file_name(version)
        output = _call_node(fn, state, retry_feedback=None)
        evidence = bool(output.get(agents_mod.REVIEW_EVIDENCE_KEY, True))
        try:
            result = contracts.parse_review_output(
                _node_text(output, "review_text"),
                review_round=round_number,
                code_version=version,
                file_name=file_name,
                has_tool_evidence=evidence,
            )
        except ContractViolation as exc:
            output = self._retry("review", round_number, fn, state, str(exc))
            evidence = bool(output.get(agents_mod.REVIEW_EVIDENCE_KEY, True)) or evidence
            result = contracts.parse_review_output(
                _node_text(output, "review_text"),
                review_round=round_number,
                code_version=version,
                file_name=file_name,
                has_tool_evidence=evidence,
            )
        path = self._workspace_path() / state_mod.review_file_name(round_number, view=view)
        self._write_artifact(path, json.dumps(result.contract, ensure_ascii=False, indent=2) + "\n")
        counts = contracts.severity_counts(result.findings)
        scope = f"·{graph_mod.VIEW_LABELS[view]}视角" if view else ""
        summary = (
            f"第 {round_number} 轮{scope}（{file_name}）：{counts[contracts.SEVERE]} 严重 / "
            f"{counts[contracts.MODERATE]} 一般 / {counts[contracts.ADVICE]} 建议"
        )
        listing = _Listing(
            kind="review",
            payload={
                "round_number": round_number,
                "file_name": file_name,
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
        if contracts.judge_clean(result.findings, strict=bool(state.get("strict"))):
            # "建议不影响判定"只在确有【建议】时说明；0 findings 时加括号属噪音
            tail = "（建议不影响判定）" if counts[contracts.ADVICE] else ""
            summary = f"{summary} —— 干净通过{tail}"
        summary = f"{summary}{self._convergence_tail(state, result.findings)}"
        update = {
            "findings": result.findings,
            "clean": contracts.judge_clean(result.findings, strict=bool(state.get("strict"))),
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
        file_name = state_mod.code_file_name(version)
        output = _call_node(fn, state, retry_feedback=None)
        merged = [item for item in (output.get("findings") or []) if isinstance(item, Mapping)]
        contract_object = {
            "iteration": round_number,
            "code_version": version,
            "file": file_name,
            "findings": merged,
            "summary": self._merged_summary(round_number),
        }
        path = self._workspace_path() / state_mod.review_file_name(round_number)
        self._write_artifact(path, json.dumps(contract_object, ensure_ascii=False, indent=2) + "\n")
        counts = contracts.severity_counts(merged)
        clean = contracts.judge_clean(merged, strict=bool(state.get("strict")))
        summary = (
            f"第 {round_number} 轮（{file_name}）：{counts[contracts.SEVERE]} 严重 / "
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
                "file_name": file_name,
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
        """修复节点：契约校验 → ``code_vN+1.py`` 与 ``fixes_N.json`` 落盘（Design §5.2）。"""
        round_number = state_mod.round_number_for_fix(state)
        current_version = state_mod.code_version(state)
        new_version = current_version + 1
        findings = list(state.get("findings") or [])
        output = _call_node(fn, state, retry_feedback=None)
        try:
            result = contracts.parse_fix_output(_node_text(output, "fix_text"), findings=findings)
        except ContractViolation as exc:
            output = self._retry("fix", round_number, fn, state, str(exc))
            result = contracts.parse_fix_output(_node_text(output, "fix_text"), findings=findings)
        previous_path = Path(str(state.get("current_code") or ""))
        previous = previous_path.read_text(encoding="utf-8") if previous_path.is_file() else ""
        code = _normalize_code(result.code)
        code_path = self._workspace_path() / state_mod.code_file_name(new_version)
        self._write_artifact(code_path, code)
        fixes_path = self._workspace_path() / state_mod.fixes_file_name(round_number)
        payload = {
            "round": round_number,
            "code_file": code_path.name,
            "fixes": result.fixes,
            "unknown_ids": list(result.unknown_ids),
        }
        self._write_artifact(fixes_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        statuses = [str(item.get("status")) for item in result.fixes]
        parts = [f"第 {round_number} 轮："]
        parts.append(f"修复 {statuses.count('fixed')} 项")
        if statuses.count("partial"):
            parts.append(f" / 部分修复 {statuses.count('partial')} 项")
        if statuses.count("skipped"):
            parts.append(f" / 跳过 {statuses.count('skipped')} 项")
        parts.append(f" → {code_path.name}")
        if code == previous:
            parts.append("（本轮未产生改动）")
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
        update = {"current_code": str(code_path), "fix_rounds": round_number}
        return update, summary, [self._relative(fixes_path), self._relative(code_path)], listing

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
            }
        )
        self._latest = state
        return self._finalize(state, _EXIT_ERROR)

    def _handle_interrupt(self, latest: Mapping[str, Any]) -> RunResult:
        """中断收尾（Design §4.6）：interrupt 事件 → 原子写 state.json → 退出码 2。

        第二次 Ctrl+C 放弃写盘（"尽力"语义），仍按中断结论返回退出码 2。
        """
        state = dict(latest)
        state.update(
            {
                "status": state_mod.STATUS_FAILED,
                "error": {"node": "interrupt", "message": "运行被用户中断（Ctrl+C）。"},
                "finished_at": self._clock(),
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
        """统一收尾写盘：state.json → report.md → 引擎 end 事件（Design §8.2）。"""
        try:
            self._persist_state(state)
            self._write_report(state)
        except PersistenceError as exc:
            return self._handle_persistence_failure(state, exc)
        self._emit(
            {
                "node": "engine",
                "event": "end",
                "detail": str(state.get("status") or ""),
                "status": state.get("status"),
            }
        )
        return self._result(state, exit_code, report_written=True)

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
            parent_state = records.read_state(records.run_dir(parent))
            if not isinstance(parent_state, Mapping):
                return None
            findings = parent_state.get("findings")
            conclusion = conclusion_text(
                status=str(parent_state.get("status") or ""),
                review_round=int(parent_state.get("review_round") or 0),
                fix_rounds=int(parent_state.get("fix_rounds") or 0),
                findings=findings if isinstance(findings, list) else [],
                strict=bool(parent_state.get("strict")),
            )
        except (OSError, TypeError, ValueError):
            # 父记录被清理 / 字段被手改成非法值时少一行标注即可，报告本身必须照样产出
            return None
        return f"{parent}：{conclusion}"

    # ---- 基础设施 ----

    def _default_run_id_factory(self) -> str:
        """默认运行号生成：按注入时钟的日期 + 碰撞重试（Design §4.3）。"""
        day = "".join(self._clock()[:10].split("-"))
        if len(day) != 8 or not day.isdigit():
            raise RunRecordError(f"时钟返回值无法解析出日期：{self._clock()!r}")
        return records.new_run_id(day=day)

    def _create_run(self) -> tuple[str, Path]:
        """创建运行目录；编号碰撞（并发 / 注入固定编号）时重新生成。

        只有"该运行号目录已存在"才算碰撞；父路径不可创建（如数据目录位置被文件占用）
        属目录创建失败，直接给可读原因，不伪装成重试。
        """
        last_error: Exception | None = None
        for _ in range(5):
            run_id = self._run_id_factory()
            try:
                return run_id, records.create_run_dir(run_id)
            except FileExistsError as exc:
                if records.run_dir(run_id).is_dir():
                    last_error = exc
                    continue
                raise RunRecordError(
                    f"无法创建运行目录（{records.run_dir(run_id)}）：{exc}"
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

        ``-f`` 输入额外标注"原文件名 → 工作区名"映射（Design §4.5 的报告口径）；续跑与并行
        审查各加一句，使终端头行与日志能一眼看出这一轮是怎么起的。
        """
        descriptor = request.input_descriptor()
        if descriptor.get("kind") == state_mod.INPUT_FILE:
            source = str(descriptor.get("source_path") or "")
            summary = f"文件：{source} → {state_mod.code_file_name(1)}"
        else:
            summary = state_mod.summarize_input(descriptor, limit=80)
        strict = "，--strict 已启用" if request.strict else ""
        parts = [
            (
                f"{summary}；修复轮次上限 {request.max_iterations}"
                f"（来源：{source_label(request.max_iterations_source)}）{strict}"
            )
        ]
        if request.parent_run_id:
            parts.append(f"续跑自 {request.parent_run_id}")
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
    strict = bool(state.get("strict"))
    input_descriptor = state.get("input") if isinstance(state.get("input"), Mapping) else {}
    conclusion = conclusion_text(
        status=status,
        review_round=review_round,
        fix_rounds=fix_rounds,
        findings=findings,
        strict=strict,
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
        "| 干净判定 | " + ("无【严重】且无【一般】（--strict）" if strict else "无【严重】") + " |"
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
    for path in artifacts:
        relative = _relative_display(path, run_dir)
        if relative == f"workspace/{state_mod.code_file_name(1)}" and source_path:
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


def _cell(value: Any) -> str:
    """Markdown 表格单元格：转义竖线并折叠换行（摘要 / 模型标识等可能含任意字符）。"""
    return " ".join(str(value if value is not None else "").split()).replace("|", "\\|")


def _round_section(item: RoundReport, *, status: str, not_converged: bool = False) -> list[str]:
    """单轮问题清单：标题固定为"第 i 轮审查（基于 code_vi.py）"，含"问题 ← 处置"对照。

    ``not_converged`` 为真时插入"未收敛"标注（Design §4.4）：问题集合与上一轮一致说明
    修复没有改变问题清单——如实标出，但不改变流程与退出码。
    """
    lines: list[str] = [f"### 第 {item.round_number} 轮审查（基于 `{item.file_name}`）", ""]
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
    for finding in findings:
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
    unknown = item.fixes.get("unknown_ids") if isinstance(item.fixes, Mapping) else None
    if isinstance(unknown, list) and unknown:
        lines.append(
            f"  - 未识别的处置（已忽略）：{', '.join(str(item_id) for item_id in unknown)}"
        )
    lines.append("")
    return lines


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
    strict: bool = False,
) -> str:
    """收尾结论句（报告与终端渲染共用单一来源，避免两处口径漂移）。

    阻塞性结论数按 ``strict`` 计入【一般】（``contracts.blocking_count``），与干净判定同源。
    """
    if status == state_mod.STATUS_SUCCESS:
        return f"干净通过（共 {review_round} 轮审查 / {fix_rounds} 轮修复）"
    if status == state_mod.STATUS_RESIDUAL:
        blocking = contracts.blocking_count(findings, strict=strict)
        kind = "阻塞问题（--strict：含【一般】）" if strict else "【严重】"
        return (
            f"达修复轮次上限，仍有 {blocking} 项{kind}残留"
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
    """代码文本归一：去掉首尾空行、统一以单个换行结尾。"""
    return code.strip("\n") + "\n"


def _single_code_fence(text: str) -> str:
    """从节点输出抽取唯一的代码围栏（Design §4.2.1）。"""
    blocks = contracts.extract_fence(text, contracts.CODE_LABELS)
    if len(blocks) != 1:
        raise ContractViolation([f"需要恰好一个 ```python 围栏，实际 {len(blocks)} 个"])
    if not blocks[0].strip():
        raise ContractViolation(["代码围栏内容为空"])
    return blocks[0]


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
