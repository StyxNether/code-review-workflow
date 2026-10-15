"""llm 层：客户端构建、SDK / 框架异常翻译与连通自检（全部离线，无网络）。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.errors import GraphRecursionError
from openai import APIConnectionError, APIStatusError, APITimeoutError

from crw import llm

#: 测试占位值（明显假值，非真实凭据）
PLACEHOLDER_KEY = "placeholder-key"


def _http_error(status_code: int, message: str = "http error", **kwargs: Any) -> APIStatusError:
    """构造带 status_code 的 ``APIStatusError``（离线构造，response 只需极简对象）。"""
    response = SimpleNamespace(status_code=status_code, request=SimpleNamespace(), headers={})
    return APIStatusError(message, response=response, body=None, **kwargs)


class TestCreateModel:
    def test_explicit_connection_parameters_win(self) -> None:
        model = llm.create_model(
            "some-model", api_key=PLACEHOLDER_KEY, base_url="https://api.example.com/v1"
        )

        assert model.model_name == "some-model"
        assert model.openai_api_base == "https://api.example.com/v1"
        assert model.temperature == llm.TEMPERATURE
        assert model.request_timeout == llm.REQUEST_TIMEOUT_SECONDS
        assert model.max_retries == llm.MAX_RETRIES

    def test_env_fallback_uses_crw_then_cra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """环境变量回退链：CRW_* 优先，未设时回落到 CRA_*（既有工具链零重配）。"""
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-legacy")
        monkeypatch.setenv("CRA_MODEL", "legacy-model")
        monkeypatch.setenv("CRA_BASE_URL", "https://legacy.example.com")
        model = llm.create_model()
        assert model.model_name == "legacy-model"
        assert model.openai_api_base == "https://legacy.example.com"

        monkeypatch.setenv("CRW_DEEPSEEK_API_KEY", PLACEHOLDER_KEY)
        monkeypatch.setenv("CRW_MODEL", "crw-model")
        monkeypatch.setenv("CRW_BASE_URL", "https://crw.example.com")
        model = llm.create_model()
        assert model.model_name == "crw-model"
        assert model.openai_api_base == "https://crw.example.com"

    def test_missing_key_fails_fast_with_guidance(self) -> None:
        """无 key 时快速失败：不把 None 交给 SDK（SDK 会回退读 OPENAI_API_KEY）。"""
        with pytest.raises(llm.LLMError, match="未配置 API key"):
            llm.create_model()

        with pytest.raises(llm.LLMError, match="crw config"):
            llm.create_model("some-model", api_key="")


class TestTranslateFailure:
    def test_llm_error_passes_through(self) -> None:
        error = llm.LLMError("原始错误")

        assert llm.translate_failure(error) is error

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_errors_get_key_guidance(self, status: int) -> None:
        translated = llm.translate_failure(_http_error(status))

        assert isinstance(translated, llm.LLMError)
        assert "认证失败" in str(translated)
        assert "CRW_DEEPSEEK_API_KEY" in str(translated)
        assert "crw config" in str(translated)
        assert isinstance(translated.__cause__, APIStatusError)

    @pytest.mark.parametrize(
        "message",
        [
            "This model's maximum context length is 65536 tokens",
            "context_length_exceeded",
            "too many tokens in request",
        ],
    )
    def test_context_length_400_is_translated_readably(self, message: str) -> None:
        """模型端上下文超限：中文可读报错 + 可操作建议。"""
        translated = llm.translate_failure(_http_error(400, message))

        assert isinstance(translated, llm.LLMError)
        assert "上下文窗口" in str(translated)
        assert "--model" in str(translated)

    def test_other_400_is_not_mistaken_for_context_length(self) -> None:
        translated = llm.translate_failure(_http_error(400, "invalid request: bad temperature"))

        assert isinstance(translated, llm.LLMError)
        assert "上下文窗口" not in str(translated)

    @pytest.mark.parametrize(
        "error",
        [
            APITimeoutError(SimpleNamespace()),
            APIConnectionError(message="connection reset", request=SimpleNamespace()),
        ],
    )
    def test_network_errors_are_wrapped(self, error: Exception) -> None:
        translated = llm.translate_failure(error)

        assert isinstance(translated, llm.LLMError)
        assert translated.__cause__ is error

    def test_other_http_statuses_keep_the_sdk_detail(self) -> None:
        """非认证 / 非上下文超限的 HTTP 失败：中文引导 + 保留 SDK 原文（便于排查）。"""
        translated = llm.translate_failure(_http_error(429, "rate limit exceeded"))

        assert isinstance(translated, llm.LLMError)
        assert "HTTP 429" in str(translated)
        assert "rate limit exceeded" in str(translated)

    def test_recursion_guard_is_readable(self) -> None:
        translated = llm.translate_failure(GraphRecursionError("recursion limit reached"))

        assert isinstance(translated, llm.LLMError)
        assert "护栏" in str(translated)

    def test_unrelated_exception_is_returned_unchanged(self) -> None:
        """意外缺陷原样返回：不得伪装成"请求失败"，否则排查方向被带偏。"""
        error = ValueError("编程缺陷")

        assert llm.translate_failure(error) is error


class _FakeModel(BaseChatModel):
    """假模型：``invoke`` 返回固定消息或抛出预置异常。"""

    reply: str = "答案"
    error: Exception | None = None

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        if self.error is not None:
            raise self.error
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])


class TestInvokeOnce:
    def test_returns_answer(self) -> None:
        message = llm.invoke_once(_FakeModel(), [HumanMessage(content="hi")])

        assert message.text == "答案"

    def test_translates_sdk_failure(self) -> None:
        model = _FakeModel(error=_http_error(401))

        with pytest.raises(llm.LLMError, match="认证失败"):
            llm.invoke_once(model, [HumanMessage(content="hi")])


class TestProbe:
    def test_empty_key_is_reported_without_a_request(self) -> None:
        status, detail = llm.probe("https://api.example.com", "", "some-model")

        assert status == "error"
        assert "API key 为空" in detail

    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (_http_error(401), "auth"),
            (_http_error(500), "error"),
            (APITimeoutError(SimpleNamespace()), "timeout"),
            (APIConnectionError(message="no route", request=SimpleNamespace()), "timeout"),
        ],
    )
    def test_classifies_failures(
        self, monkeypatch: pytest.MonkeyPatch, error: Exception, expected: str
    ) -> None:
        """探测结果分类（自检只回字符串状态，调用方按状态渲染）。"""

        class _ProbeModel(_FakeModel):
            pass

        monkeypatch.setattr(llm, "ChatOpenAI", lambda **kwargs: _ProbeModel(error=error))

        status, _ = llm.probe("https://api.example.com", PLACEHOLDER_KEY, "some-model")

        assert status == expected

    def test_ok_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm, "ChatOpenAI", lambda **kwargs: _FakeModel())

        assert llm.probe("https://api.example.com", PLACEHOLDER_KEY, "some-model") == ("ok", "")

    def test_probe_uses_short_timeout_and_no_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def _factory(**kwargs: Any) -> _FakeModel:
            captured.update(kwargs)
            return _FakeModel()

        monkeypatch.setattr(llm, "ChatOpenAI", _factory)

        llm.probe("https://api.example.com", PLACEHOLDER_KEY, "some-model")

        assert captured["max_retries"] == 0  # 自检不承担重试语义
        assert captured["timeout"] == llm.PROBE_TIMEOUT_SECONDS


class TestThinkingParams:
    """思考挡位 → 客户端构造参数（构造不发请求，可离线断言落点）。"""

    def test_no_thinking_sends_no_params(self) -> None:
        """未配置挡位：不发送任何思考参数。"""
        model = llm.create_model(
            "some-model", api_key=PLACEHOLDER_KEY, base_url="https://api.example.com/v1"
        )

        assert model.reasoning_effort is None
        assert model.extra_body is None

    def test_effort_level_lands_on_reasoning_effort(self) -> None:
        model = llm.create_model(
            "some-model",
            api_key=PLACEHOLDER_KEY,
            base_url="https://api.example.com/v1",
            thinking="high",
        )

        assert model.reasoning_effort == "high"

    def test_off_level_lands_on_thinking_type_disabled(self) -> None:
        model = llm.create_model(
            "some-model",
            api_key=PLACEHOLDER_KEY,
            base_url="https://api.example.com/v1",
            thinking="off",
        )

        assert model.extra_body == {"thinking": {"type": "disabled"}}

    def test_max_composes_both_channels(self) -> None:
        model = llm.create_model(
            "some-model",
            api_key=PLACEHOLDER_KEY,
            base_url="https://api.example.com/v1",
            thinking="max",
        )

        assert model.reasoning_effort == "high"
        assert model.extra_body == {"thinking": {"type": "enabled"}}

    def test_illegal_level_is_a_readable_llm_error(self) -> None:
        with pytest.raises(llm.LLMError, match="思考挡位不合法"):
            llm.create_model(
                "some-model",
                api_key=PLACEHOLDER_KEY,
                base_url="https://api.example.com/v1",
                thinking="budget:0",
            )


class TestConstructionGuards:
    def test_construction_validation_error_becomes_readable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """构造期 pydantic 校验失败（如参数超出 SDK 取值域）翻译为可读 LLMError，不裸抛 traceback。"""
        from pydantic import BaseModel

        class _Strict(BaseModel):
            value: int

        def _broken(**kwargs: object) -> object:
            _Strict(value="not-an-int")  # 触发 ValidationError（pydantic 内部抛）
            raise AssertionError("unreachable")

        monkeypatch.setattr(llm, "ChatOpenAI", _broken)

        with pytest.raises(llm.LLMError, match="模型客户端构造失败"):
            llm.create_model("m", api_key=PLACEHOLDER_KEY, base_url="https://api.example.com/v1")
