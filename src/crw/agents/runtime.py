"""Agent 运行时：LangChain ``create_agent`` 封装、工具轮次限制、错误协议与过程事件回调。

**单任务节点语义**：一次 :meth:`AgentRuntime.run` 只服务一个节点任务，
运行时不持有跨轮对话正本——跨轮记忆由会话层按 Agent 分别持有，
经 ``history`` 参数以历史消息形式进入本次请求，运行时不落盘、不解释其内容。

LangChain / LangGraph 类型（消息、图对象、流事件）不越出本模块：对上层只返回
:class:`TaskResult` 与字符串回调。

框架映射：
- 模型客户端由 :func:`crw.llm.create_model` 构建（OpenAI 兼容）；SDK / 框架异常经
  :func:`crw.llm.translate_failure` 翻译为 ``LLMError``，本模块不 import SDK 类型；
- 工具经 ``StructuredTool`` 桥接注册表：模型可见 schema 与注册表逐字段一致（保真），
  执行统一委托 ``registry.dispatch``（``错误：`` 结构化回传协议不变），成功执行次数
  计入 :attr:`TaskResult.tool_events`——这是审查契约"执行有效性门禁"的依据；
- 工具轮次上限与上下文预算护栏由 middleware 实现（达限对下一次模型调用去工具并注入
  收尾指令；框架内建 Limit 类 middleware 的 exit 语义不等价，不采用），并显式传
  ``recursion_limit`` 作崩溃护栏；
- 护栏触达的事实经 :meth:`AgentRuntime.drain_notices` 交给上层（引擎把它并入阶段摘要，
  进 log.jsonl 与时间线）。
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ToolCallRequest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool, tool

from .. import config, llm
from .. import usage as usage_mod
from ..tools import ERROR_PREFIX, EXEC_TOOL_NAME, Tool, ToolRegistry, arguments_preview

#: 单个节点任务允许的最大工具轮次，超过则强制要求模型给出结论
MAX_TOOL_ROUNDS = 10
#: 每个工具轮的图节点数（模型 + 工具）；middleware 引入新节点时须同步
_NODES_PER_TOOL_ROUND = 2
#: 上下文预算护栏的收尾余量（字符估算 → token 的折中系数，见 _estimate_tokens）
_ESTIMATE_FACTOR = 0.6

#: 过程事件回调（工具调用 / 重试等；由引擎落 log.jsonl 并转渲染）
EventSink = Callable[[Mapping[str, Any]], None]
#: 正文增量回调：``(节点名, 增量文本)``——渲染层据此流式输出（``--no-stream`` 由渲染层缓存）
DeltaSink = Callable[[str, str], None]
#: 节点函数类型（``(state[, retry_feedback=...]) -> Mapping``）：agents 层共用的别名，
#: 与 workflow 层的同名别名同构——agents 不得向上 import workflow，故本层自带定义）
NodeFn = Callable[..., Mapping[str, Any]]

#: 审查节点交回引擎的执行证据键（执行有效性门禁；不在状态 schema 内）
REVIEW_EVIDENCE_KEY = "review_evidence"

#: ToolNode 对参数校验失败等分支的兜底文案前缀（langgraph prebuilt）；仅用于改写时剥离
_FRAMEWORK_ERROR_PREFIX = "Error:"
#: finish_reason="length" 时附加到最终回答：长度截断的结论若静默当最终结论，用户无法察觉
_LENGTH_TRUNCATION_NOTICE = "\n\n⚠ 回答因达到输出长度上限被截断，可能不完整。"
#: 供应商原生的工具调用标记：模型想调用工具、但该调用未被解析成 tool_calls 而原样落进正文时
#: 出现（实测 DeepSeek 系偶发）。这种正文不是结论——必须显式失败，不能当审查结果展示。
_TOOL_MARKUP_MARKERS = (
    "｜｜DSML｜｜",
    "<||DSML||>",
    "<tool_calls>",
    "</tool_calls>",
    "<|tool_calls|>",
)
#: 收官回答不可用（空 / 带工具调用标记）时的重发次数（2 = 最多两次尝试）
_ANSWER_RETRY_ATTEMPTS = 2


@dataclass(frozen=True)
class TaskResult:
    """一次单任务调用的结果。"""

    text: str
    """模型给出的最终正文（节点把它作为暂存键交回引擎）。"""

    tool_events: int = 0
    """成功执行的工具调用次数（审查契约"执行有效性门禁"的依据）。"""

    truncated: bool = False
    """正文是否因达到输出长度上限被截断。"""


def _ignore_event(event: Mapping[str, Any]) -> None:
    """缺省事件回调（未注入观察者时不输出）。"""


def _forced_conclusion_directive(reason: str) -> str:
    """护栏触达时注入的收尾指令（请求侧注入，不落入任何持久化数据）。"""
    return (
        f"{reason}。请立即基于已获取的信息给出最终结论，不要再尝试调用工具；"
        "若结论基于的信息不完整，请在结论中说明这一点。"
    )


def _tool_markup_message(text: str) -> str | None:
    """正文里是否残留供应商原生的工具调用标记；有则给出可读的失败原因。"""
    if any(marker in text for marker in _TOOL_MARKUP_MARKERS):
        return (
            "模型未按工具协议返回：正文里带的是工具调用标记而不是结论（该次调用未被解析为"
            "工具调用）。这段回答不作结论，请重发；若反复出现，可换用别的模型（--model / crw config）。"
        )
    return None


@dataclass(frozen=True)
class _Answer:
    """本轮收官回答：``problem`` 非空表示该回答不可用（可原样重发一次），此时 ``text`` 为空。"""

    text: str
    message: AIMessage | None
    problem: str | None = None


def _retry_prompt(user_prompt: str, problem: str | None, *, keep_tools: bool = False) -> str:
    """本次尝试的用户消息：首次原文直传，重发时附上"上一轮不可用 + 怎么收尾"的要求。

    只有**重发**（``problem`` 非空）才追加这段要求：它同时是"去掉工具"这一变体的说明，
    首次尝试仍带工具，若把它拼进首次消息，模型会照做（实测有节点据此判定"本轮禁止调用
    工具"，放弃 read_file / run_python 而凭清单重写整份代码）。

    ``keep_tools``（上一轮已有成功工具事件时的重发）：整项目审查这类任务的证据**只在工具
    结果里**，重发不带工具就等于让模型在"什么都没读过"的上下文里硬给结论（实测它如实回答
    "未能读取任何文件"，整批白花）。此时保留工具，并明确要求按需重读后再收尾。

    追加在**用户消息**（而不是系统提示词）末尾：系统提示词是缓存前缀，必须保持不变。
    """
    if problem is None:
        return user_prompt
    if keep_tools:
        return (
            f"{user_prompt}\n\n（上一轮回答不可用：{problem}。"
            "本轮仍可读取文件：请按需重新读取要审查的内容，"
            "然后用文字直接给出最终结论，并在结论里说明覆盖范围。）"
        )
    return (
        f"{user_prompt}\n\n（上一轮回答不可用：{problem}。"
        "请直接用文字给出最终结论，不要再调用工具；信息不完整请在结论中说明。）"
    )


def _final_answer(final_state: Mapping[str, Any] | None) -> _Answer:
    """从图状态里取本轮最终回答，并判定它是否可用。

    三类不可用（都按"协议层偶发故障"处理：原样重发一次，仍失败则按运行失败收尾）：
    未返回最终状态、收尾响应缺最终回答、正文为空或残留供应商工具调用标记。
    """
    if final_state is None:
        return _Answer(text="", message=None, problem="模型调用未返回最终状态。")
    messages = final_state.get("messages") or []
    message = messages[-1] if messages else None
    if not isinstance(message, AIMessage) or message.tool_calls:
        return _Answer(text="", message=None, problem="模型调用的收尾响应缺少最终回答。")
    text = message.text
    if not text.strip():
        return _Answer(text="", message=message, problem="模型返回了空回答。")
    markup = _tool_markup_message(text)
    if markup is not None:
        return _Answer(text="", message=message, problem=markup)
    return _Answer(text=text, message=message)


def history_from_state(state: Mapping[str, Any], agent: str) -> list[dict[str, str]]:
    """从运行状态的瞬态键取该 Agent 的会话记忆；缺失 / 形态不符返回空表。

    记忆由会话层按 Agent 落盘并随 ``RunRequest`` 注入状态（引擎只做搬运，不解释内容）；
    一次性运行没有该键，因此也就没有历史消息。
    """
    memory = state.get("agent_memory")
    if not isinstance(memory, Mapping):
        return []
    lines = memory.get(agent)
    if not isinstance(lines, Sequence):
        return []
    history: list[dict[str, str]] = []
    for item in lines:
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role") or "")
        text = item.get("text")
        if role in ("user", "assistant") and isinstance(text, str) and text.strip():
            history.append({"role": role, "text": text})
    return history


def _history_messages(history: Sequence[Mapping[str, str]]) -> list[BaseMessage]:
    """会话记忆 → 历史消息：``user`` → HumanMessage、``assistant`` → AIMessage。

    记忆是"该 Agent 自己的历史"：以消息序列而非拼接文本进入请求，
    追加式前缀（系统提示词 + 历史）逐轮稳定，供应商的上下文缓存才能命中。
    """
    messages: list[BaseMessage] = []
    for item in history:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        if str(item.get("role") or "") == "assistant":
            messages.append(AIMessage(content=text))
        else:
            messages.append(HumanMessage(content=text))
    return messages


def _estimate_tokens(messages: Sequence[BaseMessage]) -> int:
    """字符数粗估（中英混合折中 × 1.2 安全余量）。

    中英混合按 ``字符数//2`` 折中：英文占比高时估算偏高（多触发——安全方向），中文
    占比高时估算偏低（少触发、可能连续超限）。取 0.6 系数（= ``//2`` 的 1.2 倍）向
    "宁早收尾"倾斜；本护栏的用途是止损而非精确计量。
    """
    total_chars = 0
    for message in messages:
        total_chars += len(message.text)
        for call in getattr(message, "tool_calls", None) or []:
            total_chars += len(str(call.get("name") or "")) + len(str(call.get("args") or {}))
    return int(total_chars * _ESTIMATE_FACTOR)


def _rounds_this_task(messages: Sequence[BaseMessage]) -> int:
    """本次任务的工具轮数：带 ``tool_calls`` 的 AI 消息记一轮。

    会话记忆（历史消息）只有正文、不带 ``tool_calls``，因此全量计数天然只覆盖本次任务
    的工具轮次（工具轮次上限按"任务"计数，不按会话累计）。
    """
    return sum(1 for message in messages if isinstance(message, AIMessage) and message.tool_calls)


def _error_tool_message(call: Mapping[str, Any], content: str) -> ToolMessage:
    """按 tool_call 构造错误结果消息（id / name 与调用一一对应）。"""
    return ToolMessage(
        content=content,
        name=str(call.get("name") or ""),
        tool_call_id=str(call.get("id") or ""),
    )


def _normalize_framework_error(message: BaseMessage) -> BaseMessage:
    """框架生成的错误 ToolMessage（``status="error"``）统一为 ``错误：`` 前缀回传模型。

    以 ``status`` 语义字段判别而非文案前缀：工具的正常输出（``status="success"``）即使
    恰好以 ``Error:`` 开头也不被改写；框架升版更换文案时该门禁依旧成立。已带 ``错误：``
    前缀的消息原样返回，避免重复叠加。
    """
    if (
        not isinstance(message, ToolMessage)
        or message.status != "error"
        or not isinstance(message.content, str)
        or message.content.startswith(ERROR_PREFIX)
    ):
        return message
    content = message.content
    if content.startswith(_FRAMEWORK_ERROR_PREFIX):
        content = content[len(_FRAMEWORK_ERROR_PREFIX) :].lstrip()
    return message.model_copy(update={"content": ERROR_PREFIX + content})


class _ToolBridgeMiddleware(AgentMiddleware):
    """工具调用桥接：过程事件、``错误：`` 结构化协议与成功次数计数。

    三条分支（未知工具名 / 框架参数校验失败 / 执行异常）统一归入 ``错误：`` 前缀回传
    模型；成功执行（结果不以 ``错误：`` 开头）计入 ``counter["ok"]``，供审查节点的执行
    有效性门禁判定。
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        self.counter: dict[str, int] = {"ok": 0}
        # 由 run() 按任务设置/清除；成功与异常路径都在 finally 里清理
        self.callbacks: tuple[EventSink | None, str] | None = None

    def _emit_tool_event(self, name: str, arguments: str) -> None:
        sink, node = self.callbacks or (None, "")
        if sink is not None:
            # 参数用预览形态：write_file 的完整正文可达数十 KB，事件 detail 是
            # 可读线索（终端工具行与日志消费同一份），不是产物的副本
            sink(
                {"node": node, "event": "tool", "detail": f"{name}({arguments_preview(arguments)})"}
            )

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Any]) -> Any:
        """模型通道错误前缀归一：框架合成的错误提示（如畸形 JSON 参数）不经工具层，
        绕过 wrap_tool_call，在请求侧统一改写。"""
        normalized = [_normalize_framework_error(message) for message in request.messages]
        if all(same is original for same, original in zip(normalized, request.messages)):
            return handler(request)  # 无框架错误消息：原请求直通，不产生额外列表
        return handler(request.override(messages=normalized))

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Any]
    ) -> Any:
        call = request.tool_call
        name = str(call.get("name") or "")
        arguments = json.dumps(call.get("args") or {}, ensure_ascii=False)
        self._emit_tool_event(name, arguments)
        if request.tool is None:
            # 未知工具名：复用注册表的结构化错误文案，模型可自行纠正策略
            response: Any = _error_tool_message(call, self._registry.dispatch(name, arguments))
        else:
            try:
                response = handler(request)
            except Exception as exc:  # noqa: BLE001 — 设计要求：工具异常必须结构化回传模型
                response = _error_tool_message(call, f"{ERROR_PREFIX}工具 '{name}' 执行失败：{exc}")
            else:
                if isinstance(response, ToolMessage):
                    response = _normalize_framework_error(response)
        if (
            isinstance(response, ToolMessage)
            and isinstance(response.content, str)
            and not response.content.startswith(ERROR_PREFIX)
        ):
            self.counter["ok"] += 1
        # 非 ToolMessage 返回值（框架版本相关的其它形态）：不记成功——执行有效性门禁
        # 宁可少认一次成功（保守方向），也不能把未知形态虚增成"工具已执行"
        return response


class _GuardrailMiddleware(AgentMiddleware):
    """任务级护栏：工具轮次上限与上下文预算。

    触达任一上限后，对**后续每一次模型调用**去工具并注入收尾指令（保留已达结论，并在
    结论中标注可能不完整）；触达事实经 ``notices`` 交给上层写入阶段摘要。官方
    ModelCallLimit / ToolCallLimit 的 exit 语义不等价（不注入、不去工具），不采用。
    """

    def __init__(self, *, max_rounds: int, max_context_tokens: int) -> None:
        self.max_rounds = max_rounds
        self.max_context_tokens = max_context_tokens
        self.notices: list[str] = []

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Any]) -> Any:
        rounds = _rounds_this_task(request.state.get("messages", []))
        prompt_tokens = len(str(request.system_message.content)) if request.system_message else 0
        estimated = _estimate_tokens(request.messages) + int(prompt_tokens * _ESTIMATE_FACTOR)
        reason: str | None = None
        if rounds >= self.max_rounds:
            reason = f"工具调用已达上限（{self.max_rounds} 轮）"
        elif estimated > self.max_context_tokens:
            reason = f"上下文预算已达上限（估算 {estimated} tokens > {self.max_context_tokens}）"
        if reason is None:
            return handler(request)
        if reason not in self.notices:
            self.notices.append(reason)
        if request.system_message is None:
            # create_agent 的 system_prompt 由本模块恒传，缺失属装配缺陷：显式失效而非
            # 伪造 user 消息兜底（伪造 user 会构成新的轮次边界）
            raise RuntimeError("模型请求缺少系统提示词：护栏收尾指令无法经 system 通道注入")
        request = request.override(
            tools=[],
            tool_choice=None,
            system_message=SystemMessage(
                content=f"{request.system_message.content}\n\n{_forced_conclusion_directive(reason)}"
            ),
        )
        return handler(request)


class AgentRuntime:
    """单任务 Agent 运行时：装配模型与工具、执行一次任务并返回纯数据结果。

    Args:
        model: 注入的聊天模型（仅测试用假模型）；与 ``resolved`` 二者取一。
        resolved: 配置层解析出的连接参数；缺省时走环境变量回退（``llm.create_model``）。
        emit: 过程事件回调（工具调用等）；缺省不输出。
        on_delta: 正文增量回调 ``(节点名, 文本)``；缺省不回调。
        max_tool_rounds: 单任务工具轮次上限。
        max_context_tokens: 单任务上下文预算；缺省取 ``CRW_MAX_CONTEXT_TOKENS``。
    """

    def __init__(
        self,
        *,
        model: BaseChatModel | None = None,
        resolved: config.ResolvedModel | None = None,
        emit: EventSink | None = None,
        on_delta: DeltaSink | None = None,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
        max_context_tokens: int | None = None,
    ) -> None:
        if model is not None:
            self._chat_model: BaseChatModel = model
        elif resolved is not None:
            self._chat_model = llm.create_model(
                resolved.model,
                api_key=resolved.api_key,
                base_url=resolved.base_url,
                thinking=resolved.thinking,
            )
        else:
            self._chat_model = llm.create_model()
        self._emit: EventSink = emit or _ignore_event
        self._on_delta = on_delta
        self.max_tool_rounds = max_tool_rounds
        self.max_context_tokens = (
            max_context_tokens
            if max_context_tokens is not None
            else config.get_max_context_tokens()
        )
        self._graphs: dict[tuple[str, int], Any] = {}
        self._guardrails: dict[tuple[str, int], _GuardrailMiddleware] = {}
        self._bridges: dict[tuple[str, int], _ToolBridgeMiddleware] = {}
        #: 各节点累计的 token 用量（``节点名 → 账``，见 :mod:`crw.usage`）：与护栏记录同样
        #: 由上层按节点取走（并行分支各自回收自己的），引擎把它写进运行记录
        self._usage: dict[str, dict[str, int]] = {}
        # 并行双视角的两个分支在别的线程里共用本 runtime：图的按需编译（_graph_for 的
        # 读-建-写）与护栏记录的清空 / 取用都必须串行，避免重复编译与记录错位
        self._lock = threading.RLock()

    def run(
        self,
        *,
        node: str,
        system_prompt: str,
        user_prompt: str,
        registry: ToolRegistry,
        history: Sequence[Mapping[str, str]] = (),
    ) -> TaskResult:
        """执行一次节点任务：历史消息 + 单轮 user 输入 → 自主调用工具 → 返回最终正文。

        Args:
            node: 节点名（``encode`` / ``review`` / ``fix``；用于事件与增量路由）。
            system_prompt: 该节点的系统提示词（常量，缓存前缀友好）。
            user_prompt: 本次任务的用户消息（含本轮上下文与输出契约）。
            registry: 该节点的工具注册表（``--no-exec`` 时其中不含 ``run_python``）。
            history: 该 Agent 的会话记忆（``[{"role", "text"}]``）：作为
                **历史消息**放在本轮 user 之前——追加式记忆的前缀逐轮稳定，供应商的
                上下文缓存因此可以命中已发送部分；空表（一次性运行）时只有本次任务的消息。

        Raises:
            LLMError: 模型调用失败（已翻译为可读中文，不含 SDK 类型）。
            RuntimeError: 模型未给出最终回答等协议级失败。
        """
        graph, bridge, _ = self._graph_for(node, system_prompt, registry)
        with self._lock:
            # 清空**全部**变体的护栏记录（含重发用的无工具变体）：drain_notices 遍历全部
            # guardrail，只清当前变体会让残留在别处的触达记录被后续任务误报为本轮触达
            for item in self._guardrails.values():
                item.notices.clear()
        memory = _history_messages(history)
        tool_events = 0
        problem: str | None = None
        had_evidence = False  # 上一轮是否有成功工具事件（决定重发是否保留工具）
        for attempt in range(1, _ANSWER_RETRY_ATTEMPTS + 1):
            # 重发默认换成"无工具 + 直接要结论"：实测这类故障多发于多轮工具探索之后，
            # 原样重发很可能再撞一次；去掉工具并显式要求收尾才能稳定拿到结论。
            # 例外是上一轮靠工具读到过内容——证据只活在工具结果里，不带工具的重发等于
            # 从零开始，只能得到"什么都没读到"的空结论（整项目审查形态实测踩过）。
            retry = attempt > 1
            keep_tools = retry and had_evidence
            graph, bridge, _ = self._graph_for(
                node, system_prompt, registry, tools_enabled=not retry or had_evidence
            )
            bridge.counter["ok"] = 0
            bridge.callbacks = (self._emit, node)
            fed: list[BaseMessage] = [
                *memory,
                HumanMessage(content=_retry_prompt(user_prompt, problem, keep_tools=keep_tools)),
            ]
            try:
                final_state = self._drive(graph, fed, node)
            except Exception as exc:
                translated = llm.translate_failure(exc)
                if translated is exc:
                    raise  # 意外缺陷原样上抛（命令面以"意外错误"展示），不伪装成请求失败
                raise translated from exc
            finally:
                bridge.callbacks = None
            tool_events = max(tool_events, bridge.counter["ok"])
            had_evidence = bridge.counter["ok"] > 0
            answer = _final_answer(final_state)
            if answer.problem is None:
                break
            problem = answer.problem
            if attempt == _ANSWER_RETRY_ATTEMPTS:
                raise llm.LLMError(problem)
            # "正文里带回工具调用标记 / 收官回答为空"是协议层的偶发故障（实测 DeepSeek 系），
            # 不是内容问题：重发一次，并把这次重发记进事件流（与节点的契约重试同为 retry 事件）。
            self._emit({"node": node, "event": "retry", "detail": problem})
        truncated = answer.message is not None and _finish_reason(answer.message) == "length"
        text = answer.text
        if truncated:
            text += _LENGTH_TRUNCATION_NOTICE
        return TaskResult(text=text, tool_events=tool_events, truncated=truncated)

    def drain_notices(self) -> list[str]:
        """取走并清空护栏触达记录（引擎并入阶段摘要，进 log.jsonl 与时间线）。"""
        notices: list[str] = []
        with self._lock:
            for guardrail in self._guardrails.values():
                notices.extend(guardrail.notices)
                guardrail.notices.clear()
        return notices

    def drain_notices_for(self, node: str) -> list[str]:
        """只取该节点变体的护栏触达记录（并行分支各自只回收自己的）。

        并行双视角下两个分支共用本 runtime：全局 drain 会把**另一个仍在运行的分支**
        刚触达的记录一并取走并归到当前节点上。按节点（缓存键首项）过滤后，
        每条触达只会进入产生它的那个节点的阶段摘要。
        """
        notices: list[str] = []
        with self._lock:
            for key, guardrail in self._guardrails.items():
                if key[0] != node:
                    continue
                notices.extend(guardrail.notices)
                guardrail.notices.clear()
        return notices

    def drain_usage_for(self, node: str) -> dict[str, int]:
        """取走并清空该节点已累计的 token 用量（无调用时为空账，不是 ``None``）。"""
        with self._lock:
            record = self._usage.pop(node, None)
        return usage_mod.merge(usage_mod.empty(), record)

    def drain_usage(self) -> dict[str, dict[str, int]]:
        """取走并清空**全部**节点的用量账（``crw review`` 的批量审查与中断收尾用）。

        返回 ``节点名 → 账``；一次性运行由引擎按节点取走，本方法给没有引擎接入的
        调用方（``crw review`` 逐文件审查、被中断时残留的本节点）兜底取用。
        """
        with self._lock:
            drained, self._usage = self._usage, {}
        return {
            node: usage_mod.merge(usage_mod.empty(), record) for node, record in drained.items()
        }

    def model_label(self) -> str:
        """当前模型标识（命令面写 ``state["model"]`` 用；注入假模型时为类名）。"""
        name = getattr(self._chat_model, "model_name", "") or getattr(self._chat_model, "model", "")
        return str(name) or type(self._chat_model).__name__

    # ---- 内部装配 ----

    def _drive(self, graph: Any, fed: Sequence[BaseMessage], node: str) -> dict[str, Any] | None:
        """驱动图执行并返回最后一份状态快照。

        ``stream_mode="messages"`` 提供正文增量（工具轮伴随的少量文本同样渲染，已知
        取舍）；``"values"`` 提供每步后的状态快照，末次即最终状态。护栏经 config 位置
        参数传入——kwargs 会被框架吞进废弃参数桶而静默失效（实测过该行为）。

        每次模型调用落进状态的新 AI 消息都记一笔用量（并即时发出 usage 事件，与工具
        事件在同一时间线上交错）：输入 / 缓存命中 / 输出 token 只有在**调用当时**拿得到，
        等节点收尾再汇总就分不清是哪一次调用花的。
        """
        limit = _NODES_PER_TOOL_ROUND * self.max_tool_rounds + 5
        final_state: dict[str, Any] | None = None
        reported = len(fed)  # 喂进去的历史消息与本轮 user 不带本次调用的用量，跳过
        for mode, payload in graph.stream(
            {"messages": list(fed)}, {"recursion_limit": limit}, stream_mode=["messages", "values"]
        ):
            if mode == "messages":
                chunk = payload[0]
                if isinstance(chunk, AIMessageChunk) and self._on_delta is not None:
                    text = chunk.text
                    if text:
                        self._on_delta(node, text)
            else:
                final_state = payload
                reported = self._record_usage(node, payload.get("messages") or [], reported)
        return final_state

    def _record_usage(self, node: str, messages: Sequence[Any], reported: int) -> int:
        """为状态里新增的 AI 消息记用量，返回已处理的条数。

        只处理 ``messages[reported:]``：values 模式每个超级步都给完整消息列表，靠
        "已处理条数"避免重复入账。列表被框架裁短（本栈的装配不会——消息通道只追加，
        记忆压缩在会话层做）时**游标不回落**：宁可漏记几条（少报花费），也不能把
        已入账的调用重复记一遍（多报花费）。
        """
        if len(messages) < reported:
            return reported
        for message in messages[reported:]:
            if not isinstance(message, AIMessage):
                continue
            record = usage_mod.from_usage_metadata(getattr(message, "usage_metadata", None))
            self._emit({"node": node, "event": "usage", "detail": usage_mod.log_detail(record)})
            with self._lock:
                self._usage[node] = usage_mod.merge(self._usage.get(node), record)
        return len(messages)

    def _graph_for(
        self,
        node: str,
        system_prompt: str,
        registry: ToolRegistry,
        *,
        tools_enabled: bool = True,
    ) -> tuple[Any, _ToolBridgeMiddleware, _GuardrailMiddleware]:
        """按节点取（或建）编译后的 agent 图与配套 middleware（并发安全）。

        图谱缓存按 (节点, 工具集) 维度：同一运行内每个节点的系统提示词与工具集都是常量，
        重复调用（如 ``crw review`` 逐文件审查）复用同一图，避免重复编译；``tools_enabled=False``
        的"无工具"变体只在回答重发时用到。并行双视角的两个分支会同时到达这里（不同线程），
        读-建-写整体加锁：重复编译浪费事小，**两个分支各自拿到半初始化的缓存条目**才是隐患。
        缓存键用 ``id(registry)`` 是安全的：缓存里的编译图经 ``_bridge_tool`` 的闭包
        **强引用**着该 registry，条目存活期内其 id 不可能被回收复用；registry 生命周期
        与 runtime 一致（同一运行内按节点构造一次），不会出现"新 registry 命中旧图"。
        """
        key = (node, id(registry), tools_enabled)
        with self._lock:
            cached = self._graphs.get(key)
            if cached is not None:
                return cached, self._bridges[key], self._guardrails[key]
            bridge = _ToolBridgeMiddleware(registry)
            guardrail = _GuardrailMiddleware(
                max_rounds=self.max_tool_rounds, max_context_tokens=self.max_context_tokens
            )
            graph = create_agent(
                model=self._chat_model,
                tools=(
                    [_bridge_tool(registry, tool_def) for tool_def in registry.definitions()]
                    if tools_enabled
                    else []
                ),
                system_prompt=system_prompt,
                middleware=[bridge, guardrail],
            )
            self._graphs[key] = graph
            self._bridges[key] = bridge
            self._guardrails[key] = guardrail
            return graph, bridge, guardrail


def _bridge_tool(registry: ToolRegistry, tool_def: Tool) -> StructuredTool:
    """经 ``@tool`` 声明单个注册表工具的 LangChain 桥接：执行统一委托 ``registry.dispatch``。

    ``args_schema`` 直接采用注册表的 JSON schema（模型可见 schema 逐字段保真，``@tool``
    的类型标注推断在此不适用）；dict schema 下参数不经框架 pydantic 校验直达 dispatch——
    参数缺失 / 类型不符等仍由注册表以 ``错误：…`` 结构化文本兜底。
    """
    name = tool_def.name

    @tool(name, description=tool_def.description, args_schema=tool_def.parameters)
    def _dispatch(**kwargs: Any) -> str:
        return registry.dispatch(name, json.dumps(kwargs, ensure_ascii=False))

    return _dispatch


def _finish_reason(message: AIMessage) -> str | None:
    """服务端 ``finish_reason``（length 截断需向用户明示）。"""
    return message.response_metadata.get("finish_reason")


__all__ = [
    "EXEC_TOOL_NAME",
    "MAX_TOOL_ROUNDS",
    "REVIEW_EVIDENCE_KEY",
    "AgentRuntime",
    "DeltaSink",
    "EventSink",
    "TaskResult",
    "history_from_state",
]
