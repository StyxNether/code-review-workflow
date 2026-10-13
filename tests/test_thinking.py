"""thinking 模块：内置等级映射、budget 形态判定与 thinking 配置归一化。"""

from __future__ import annotations

import pytest

from crw import thinking

# ---- 等级 → 参数映射 ----


@pytest.mark.parametrize("name", ["off", "disabled"])
def test_off_family_sends_thinking_disabled(name: str) -> None:
    """off / disabled 走 thinking.type=disabled 通道（DeepSeek 等兼容层的 Anthropic 风格参数）。"""
    assert thinking.resolve_level_params(name) == {"extra_body": {"thinking": {"type": "disabled"}}}


def test_none_sends_reasoning_effort_none() -> None:
    """none 走 reasoning_effort=none 通道（OpenAI / DeepSeek 均以 none 表示关闭思考）。"""
    assert thinking.resolve_level_params("none") == {"reasoning_effort": "none"}


@pytest.mark.parametrize("name", ["minimal", "low", "medium", "high"])
def test_effort_family_sends_reasoning_effort(name: str) -> None:
    """minimal..high 走 langchain-openai 一等参数 reasoning_effort。"""
    assert thinking.resolve_level_params(name) == {"reasoning_effort": name}


def test_max_composes_effort_and_thinking_enabled() -> None:
    """max 复合注入：effort 顶格 high，另以 thinking.type=enabled 声明开启。"""
    assert thinking.resolve_level_params("max") == {
        "reasoning_effort": "high",
        "extra_body": {"thinking": {"type": "enabled"}},
    }


def test_enabled_sends_enable_thinking_switch() -> None:
    assert thinking.resolve_level_params("enabled") == {"extra_body": {"enable_thinking": True}}


@pytest.mark.parametrize(("name", "budget"), [("budget:1", 1), ("budget:1024", 1024)])
def test_budget_level_sends_int_thinking_budget(name: str, budget: int) -> None:
    """budget:<n> 解析为 int，经 extra_body.thinking_budget 发送。"""
    assert thinking.resolve_level_params(name) == {"extra_body": {"thinking_budget": budget}}


def test_level_input_is_stripped() -> None:
    assert thinking.resolve_level_params("  high\t") == {"reasoning_effort": "high"}


def test_unknown_level_passes_through() -> None:
    """未知名称按用户自定义供应商参数原样透传，由兼容层解释。"""
    assert thinking.resolve_level_params("analysis") == {"extra_body": {"analysis": True}}


def test_resolve_returns_fresh_dicts_per_call() -> None:
    """两次调用结果相等但对象互不共享，杜绝可变默认值被串改。"""
    first = thinking.resolve_level_params("max")
    second = thinking.resolve_level_params("max")
    assert first == second
    assert first is not second
    assert first["extra_body"] is not second["extra_body"]
    assert first["extra_body"]["thinking"] is not second["extra_body"]["thinking"]


# ---- 非法挡位 ----


@pytest.mark.parametrize(
    "name",
    ["budget:0", "budget:-3", "budget:abc", "budget:", "budget:+5", "budget:１"],
)
def test_illegal_budget_forms_raise(name: str) -> None:
    with pytest.raises(thinking.ThinkingLevelError):
        thinking.resolve_level_params(name)
    assert not thinking.is_budget_level(name)


@pytest.mark.parametrize("name", ["", "   "])
def test_empty_level_raises(name: str) -> None:
    with pytest.raises(thinking.ThinkingLevelError):
        thinking.resolve_level_params(name)
    assert not thinking.is_builtin_level(name)


def test_budget_error_message_is_actionable() -> None:
    """报错信息须点名 budget 形态，便于用户自查。"""
    with pytest.raises(thinking.ThinkingLevelError) as excinfo:
        thinking.resolve_level_params("budget:0")
    assert "budget" in str(excinfo.value)


def test_errors_are_value_errors() -> None:
    """两个错误类型挂 ValueError 下，调用方可统一按值错误捕获。"""
    assert issubclass(thinking.ThinkingLevelError, ValueError)
    assert issubclass(thinking.ThinkingConfigError, ValueError)


# ---- 挡位判定与文案 ----


@pytest.mark.parametrize("name", [*thinking.BUILTIN_LEVELS, " budget:5 ", "budget:9"])
def test_is_builtin_level_positive(name: str) -> None:
    assert thinking.is_builtin_level(name)


@pytest.mark.parametrize("name", ["", "   ", "OFF", "unknown", "budget:0", "budget:abc"])
def test_is_builtin_level_negative(name: str) -> None:
    assert not thinking.is_builtin_level(name)


def test_available_levels_text_lists_everything() -> None:
    assert thinking.available_levels_text().split(" / ") == [
        *thinking.BUILTIN_LEVELS,
        "budget:<n>",
    ]


# ---- thinking 配置归一化 ----


def test_normalize_none_means_unconfigured() -> None:
    assert thinking.normalize_thinking_config(None) is None


def test_normalize_empty_object_allows_omission() -> None:
    assert thinking.normalize_thinking_config({}) == {"levels": None, "selected": None}


def test_normalize_levels_and_selected() -> None:
    assert thinking.normalize_thinking_config({"levels": ["low", "high"], "selected": "high"}) == {
        "levels": ["low", "high"],
        "selected": "high",
    }


def test_normalize_only_levels() -> None:
    assert thinking.normalize_thinking_config({"levels": ["off", "low"]}) == {
        "levels": ["off", "low"],
        "selected": None,
    }


def test_normalize_only_selected_off() -> None:
    """levels 未声明时仅允许 off：没有可选清单就没有可选项。"""
    assert thinking.normalize_thinking_config({"selected": "off"}) == {
        "levels": None,
        "selected": "off",
    }


def test_normalize_off_selected_needs_no_levels_entry() -> None:
    """off 不依赖清单：即便 levels 未列出 off 也合法。"""
    assert thinking.normalize_thinking_config({"levels": ["low"], "selected": "off"}) == {
        "levels": ["low"],
        "selected": "off",
    }


def test_normalize_returns_copied_levels() -> None:
    """返回的 levels 是副本，与入参不共享可变列表。"""
    levels = ["low"]
    result = thinking.normalize_thinking_config({"levels": levels, "selected": "low"})
    assert result == {"levels": ["low"], "selected": "low"}
    result["levels"].append("high")
    assert levels == ["low"]


def test_normalize_selected_outside_levels_raises() -> None:
    with pytest.raises(thinking.ThinkingConfigError):
        thinking.normalize_thinking_config({"levels": ["low", "high"], "selected": "max"})


def test_normalize_selected_without_levels_raises() -> None:
    with pytest.raises(thinking.ThinkingConfigError):
        thinking.normalize_thinking_config({"selected": "high"})


def test_normalize_empty_levels_raises() -> None:
    with pytest.raises(thinking.ThinkingConfigError):
        thinking.normalize_thinking_config({"levels": []})


@pytest.mark.parametrize("levels", [["low", 3], ["low", True], "low", 42])
def test_normalize_non_string_levels_raise(levels: object) -> None:
    with pytest.raises(thinking.ThinkingConfigError):
        thinking.normalize_thinking_config({"levels": levels})


def test_normalize_duplicate_levels_raise() -> None:
    with pytest.raises(thinking.ThinkingConfigError):
        thinking.normalize_thinking_config({"levels": ["low", "high", "low"]})


@pytest.mark.parametrize("selected", [3, True, ["low"]])
def test_normalize_non_string_selected_raises(selected: object) -> None:
    with pytest.raises(thinking.ThinkingConfigError):
        thinking.normalize_thinking_config({"selected": selected})


def test_normalize_unknown_key_raises_and_names_it() -> None:
    """拼错字段名应当场暴露：报错并点名未知字段，不静默忽略。"""
    with pytest.raises(thinking.ThinkingConfigError) as excinfo:
        thinking.normalize_thinking_config({"level": "low"})
    assert "level" in str(excinfo.value)


@pytest.mark.parametrize("value", ["off", ["off"], 3, True])
def test_normalize_non_object_value_raises(value: object) -> None:
    with pytest.raises(thinking.ThinkingConfigError):
        thinking.normalize_thinking_config(value)


# ---- 生效挡位 ----


def test_selected_level_none_when_unconfigured() -> None:
    assert thinking.selected_level(None) is None
    assert thinking.selected_level({}) is None
    assert thinking.selected_level({"levels": ["low"], "selected": None}) is None


def test_selected_level_returns_off_verbatim() -> None:
    """off 是显式关闭而非未配置，原样返回供调用方走 off 通道。"""
    assert thinking.selected_level({"selected": "off"}) == "off"


def test_selected_level_returns_level_name() -> None:
    assert thinking.selected_level({"levels": ["low"], "selected": "low"}) == "low"
