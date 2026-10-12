"""模型客户端：ChatOpenAI（OpenAI 兼容多供应商）构建、SDK 重试映射与异常翻译。

模型客户端与重试策略都在本层完成：客户端构建（base_url / api_key / 温度 / 请求级
超时 / SDK 重试 / 认证快速失败指引）与 SDK / 框架异常 → :class:`LLMError` 的翻译，
上层（agents / cli）不接触 SDK 类型。

重试语义：瞬时错误由 SDK ``max_retries`` 承担（指数退避与 5xx 语义以 openai SDK
实现为准，仅覆盖建连阶段）；401/403 认证错误 SDK 本就不重试（快速失败），失败时附
key 排查指引；流式中途断开不重试——增量已渲染，重发会让用户看到重复文本。
"""

from __future__ import annotations

import sys
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAIError

from . import config

#: 请求级超时：不用 SDK 默认的 600s，避免失败请求长时间挂起
REQUEST_TIMEOUT_SECONDS = 60
#: 输出温度：偏低使同类代码的审查结论更稳定
TEMPERATURE = 0.2
#: 瞬时错误重试次数（映射到 SDK，仅建连期）
MAX_RETRIES = 3
#: ``crw config test`` 的自检超时：快速反馈，不等完整请求超时
PROBE_TIMEOUT_SECONDS = 15
#: 连通性自检只发一次 max_tokens=1 的最小请求
PROBE_MAX_TOKENS = 1

_AUTH_STATUSES = (401, 403)
_CONTEXT_LENGTH_HINTS = (
    "context length",
    "context_length",
    "maximum context",
    "max context",
    "too many tokens",
    "reduce the length",
)


class LLMError(RuntimeError):
    """LLM 请求失败（网络 / 超时 / 认证 / 上下文超限等）。

    对上层屏蔽 openai SDK 与 LangGraph 的异常类型，消息为可读中文。
    """


def _key_hint() -> str:
    """key 设置方式按当前平台给出（默认供应商 DeepSeek 的变量名）。"""
    if sys.platform == "win32":
        return 'Windows：setx CRW_DEEPSEEK_API_KEY "你的key"，然后新开终端生效'
    return 'macOS/Linux：export CRW_DEEPSEEK_API_KEY="你的key"（写入 ~/.bashrc 或 ~/.zshrc 持久化）'


def _auth_guidance() -> str:
    """认证失败的排查指引（401/403 快速失败时随错误给出）。"""
    return (
        "认证失败（HTTP 401/403）：API key 无效、未激活或无权访问该模型，已停止重试。"
        "请用 crw config 检查该供应商的 key（或其环境变量，如 CRW_DEEPSEEK_API_KEY）；"
        f"设置方法（{_key_hint()}）。"
    )


def create_model(
    model_name: str | None = None,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> ChatOpenAI:
    """构建 OpenAI 兼容的 ChatOpenAI 客户端。

    显式校验 key 非空：SDK 在 ``api_key`` 为 None 时会回退读取 ``OPENAI_API_KEY``，
    可能把 OpenAI 的密钥发往 ``CRW_BASE_URL`` 指定的第三方接入点。

    Args:
        model_name: 模型名；省略时取环境变量回退（``CRW_MODEL`` → ``CRA_MODEL``）。
        api_key: 显式 key（``crw config`` 解析结果）；省略时走环境变量回退链。
        base_url: 显式接入点；省略时走环境变量回退。
    """
    key = api_key if api_key is not None else config.get_api_key()
    if not key:
        raise LLMError(
            "未配置 API key：请运行 crw config，或设置环境变量 CRW_DEEPSEEK_API_KEY"
            "（使用环境变量回退方式时默认供应商为 DeepSeek）。"
        )
    return ChatOpenAI(
        model=model_name if model_name is not None else config.get_model(),
        api_key=key,
        base_url=base_url if base_url is not None else config.get_base_url(),
        temperature=TEMPERATURE,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
    )


def translate_failure(exc: Exception) -> Exception:
    """SDK / 框架异常 → :class:`LLMError` 的翻译（上层不 import SDK 类型）。

    :class:`LLMError` 原样返回；401/403 附 key 排查指引（快速失败语义由 SDK 保证）；
    请求体超上下文窗口的 400 给中文可读报错；recursion_limit 护栏给可读提示；其余
    SDK 异常统一包装为 :class:`LLMError`；非 SDK / 框架异常原样返回——那是意外缺陷，
    应以"意外错误"示人而非伪装成请求失败。
    """
    if isinstance(exc, LLMError):
        return exc
    translated: LLMError | None = None
    if isinstance(exc, APIStatusError) and exc.status_code in _AUTH_STATUSES:
        translated = LLMError(f"{_auth_guidance()}（{exc}）")
    elif isinstance(exc, APIStatusError) and _is_context_length_error(exc):
        translated = LLMError(
            "请求超出模型上下文窗口（HTTP 400）：本轮内容过长。"
            "可改用更大上下文窗口的模型（--model / crw config），"
            "或缩小审查范围（分段审查、减少修复轮次）后重试。"
        )
    elif isinstance(exc, APIStatusError):
        # 非认证、非上下文超限的 HTTP 失败（429 / 5xx 等）：中文引导 + 保留 SDK 原文便于排查
        translated = LLMError(f"模型请求失败（HTTP {exc.status_code}）：{exc}")
    elif isinstance(exc, (APITimeoutError, APIConnectionError, OpenAIError)):
        translated = LLMError(str(exc))
    elif isinstance(exc, GraphRecursionError):
        translated = LLMError(
            "工具调用超出框架护栏（recursion_limit），本轮作废；若反复出现请反馈该问题。"
        )
    if translated is None:
        return exc
    translated.__cause__ = exc  # 返回而非 raise：原始异常保留在 __cause__ 供排查
    return translated


def invoke_once(model: BaseChatModel, messages: list[BaseMessage], **kwargs: Any) -> AIMessage:
    """非流式单次调用（``--no-stream`` 与自检路径使用），异常经翻译上抛。"""
    try:
        return model.invoke(messages, **kwargs)
    except Exception as exc:
        translated = translate_failure(exc)
        if translated is exc:
            raise
        raise translated from exc


def _is_context_length_error(exc: APIStatusError) -> bool:
    """400 且报错文本指向上下文长度/ token 超限（各供应商文案不一，按关键词判定）。"""
    if exc.status_code != 400:
        return False
    text = f"{exc}".lower()
    return any(hint in text for hint in _CONTEXT_LENGTH_HINTS)


def probe(base_url: str, api_key: str, model_name: str) -> tuple[str, str]:
    """连通性自检（``crw config test``）：发一次 ``max_tokens=1`` 的最小请求。

    Returns:
        ``(状态, 详情)``：ok / auth（401/403，key 无效）/ timeout（超时或网络不可达）/
        error（其余失败）。探测用短超时且 ``max_retries=0``——自检的意义是快速反馈，
        不承担业务请求的重试语义。异常分类用 SDK 原始类型在本层完成（:class:`LLMError`
        的文本面向用户排查，不适合程序化判定），详情截断至 200 字符防服务端响应体刷屏。
    """
    if not api_key:
        # 与 create_model 的显式校验一致：空 key 不应发到服务端换回误导性的 401
        return "error", f"API key 为空；请用 crw config 补充，或设置环境变量 {_key_hint()}。"
    try:
        model = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=TEMPERATURE,
            timeout=PROBE_TIMEOUT_SECONDS,
            max_retries=0,
        )
        model.bind(max_tokens=PROBE_MAX_TOKENS).invoke([HumanMessage(content="ping")])
    except APIStatusError as exc:
        if exc.status_code in _AUTH_STATUSES:
            return "auth", str(exc)[:200]
        return "error", str(exc)[:200]  # 客户端构建期抛出同样归入可报告状态，不逃逸为 traceback
    except (APITimeoutError, APIConnectionError) as exc:
        return "timeout", str(exc)[:200]
    except Exception as exc:  # noqa: BLE001 — 自检兜底：任何失败都归入可报告状态而非上抛
        return "error", str(exc)[:200]
    return "ok", ""
