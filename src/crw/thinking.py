"""内置思考等级表：把挡位名映射为 OpenAI 兼容客户端的发送参数。

``reasoning_effort`` 是 langchain-openai 的一等构造参数；``thinking`` /
``enable_thinking`` / ``thinking_budget`` 等非一等参数一律经 ``extra_body`` 透传，
由各供应商的 OpenAI 兼容层在请求体中接收。本模块只做纯逻辑，不依赖任何 crw 模块。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: 内置等级名（budget:<n> 数值预算形式单独判定，不在此元组）
BUILTIN_LEVELS: tuple[str, ...] = (
    "off",
    "none",
    "disabled",
    "on",
    "minimal",
    "low",
    "medium",
    "high",
    "max",
    "enabled",
    "enable-off",
)

#: 数值预算挡位的前缀，完整形态为 ``budget:<正整数>``
_BUDGET_PREFIX = "budget:"


class ThinkingLevelError(ValueError):
    """思考挡位名不合法（空串或 budget 形态非法）；消息为可读中文，可直接展示给用户。"""


class ThinkingConfigError(ValueError):
    """models.json 的 thinking 配置不合法（结构、类型或取值）；消息为可读中文。"""


def _is_ascii_positive_int(text: str) -> bool:
    """判断文本是否为非空 ASCII 十进制正整数字面量（拒收全角数字、符号与零）。"""
    return bool(text) and text.isascii() and text.isdigit() and int(text) >= 1


def is_budget_level(name: str) -> bool:
    """判断是否为 ``budget:<n>`` 预算形态；n 须为正整数，``budget:0`` 等为假。"""
    stripped = name.strip()
    if not stripped.startswith(_BUDGET_PREFIX):
        return False
    return _is_ascii_positive_int(stripped[len(_BUDGET_PREFIX) :])


def is_builtin_level(name: str) -> bool:
    """判断是否为可用挡位：内置名或合法 budget 形态；先去首尾空白，空串为假。"""
    stripped = name.strip()
    if not stripped:
        return False
    return stripped in BUILTIN_LEVELS or is_budget_level(stripped)


def available_levels_text() -> str:
    """单行列出全部可用挡位（内置名 + budget 形态），供参数报错与配置菜单复用。"""
    return " / ".join((*BUILTIN_LEVELS, "budget:<n>"))


def resolve_level_params(level: str) -> dict[str, Any]:
    """挡位名 → ChatOpenAI 构造参数；每次调用新建返回值，不共享可变对象。

    关闭族有两个通道（``thinking.type=disabled`` 与 ``reasoning_effort=none``），按名分派
    ——``off`` / ``disabled`` 走前者（DeepSeek 等兼容层支持 Anthropic 风格参数），
    ``none`` 走后者（OpenAI / DeepSeek 的 ``reasoning_effort`` 取值，``none`` 即关闭思考）；
    ``on`` 是 ``off`` 的开启对偶（``thinking.type=enabled``，Kimi k2.6 等用此通道开启）；
    minimal..high 走 ``reasoning_effort`` 一等参数；max 复合注入（effort 顶格 high，另以
    ``thinking.type=enabled`` 声明开启）；enabled / enable-off 是 ``enable_thinking``
    布尔开关的开与关（Qwen 系的百炼兼容层）；budget:<n> 以 ``thinking_budget`` 传数值预算；
    其余名称视为用户自定义的供应商参数，原样透传给兼容层解释。
    """
    name = level.strip()
    if not name:
        raise ThinkingLevelError(f"思考挡位不能为空；可用挡位：{available_levels_text()}")
    if name.startswith(_BUDGET_PREFIX):
        digits = name[len(_BUDGET_PREFIX) :]
        if not _is_ascii_positive_int(digits):
            raise ThinkingLevelError(
                f"budget 挡位形如 budget:<正整数>（如 budget:2048），得到：{level!r}"
            )
        return {"extra_body": {"thinking_budget": int(digits)}}
    if name in ("off", "disabled"):
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    if name == "on":
        return {"extra_body": {"thinking": {"type": "enabled"}}}
    if name == "none":
        return {"reasoning_effort": "none"}
    if name in ("minimal", "low", "medium", "high"):
        return {"reasoning_effort": name}
    if name == "max":
        return {
            "reasoning_effort": "high",
            "extra_body": {"thinking": {"type": "enabled"}},
        }
    if name == "enabled":
        return {"extra_body": {"enable_thinking": True}}
    if name == "enable-off":
        return {"extra_body": {"enable_thinking": False}}
    return {"extra_body": {name: True}}


def normalize_thinking_config(value: Any) -> dict[str, Any] | None:
    """归一化 models.json 模型条目的 ``thinking`` 键；未配置（None）返回 None。

    合法形态为 ``{"levels": [挡位名...], "selected": 挡位名}``，两键皆可省略；
    selected 只能是 ``off`` 或 levels 中的一项（levels 未声明时仅允许 ``off``）；
    未知字段直接报错——拼错字段名应当场暴露，而非静默忽略。
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ThinkingConfigError(
            'thinking 配置须为对象，形如 {"levels": [...], "selected": "..."}，'
            f"得到 {type(value).__name__}"
        )
    unknown = sorted(set(value) - {"levels", "selected"})
    if unknown:
        raise ThinkingConfigError(
            f"thinking 含未知字段：{'、'.join(unknown)}；仅支持 levels / selected"
        )
    levels = value.get("levels")
    if levels is not None:
        if not isinstance(levels, list) or not all(isinstance(item, str) for item in levels):
            raise ThinkingConfigError("thinking.levels 须为字符串数组")
        if not levels:
            raise ThinkingConfigError("thinking.levels 不能为空数组；未声明时应省略该字段")
        duplicated = sorted({item for item in levels if levels.count(item) > 1})
        if duplicated:
            raise ThinkingConfigError(f"thinking.levels 含重复挡位：{'、'.join(duplicated)}")
    selected = value.get("selected")
    if selected is not None:
        if not isinstance(selected, str):
            raise ThinkingConfigError("thinking.selected 须为字符串挡位名")
        if selected != "off" and (levels is None or selected not in levels):
            if levels is None:
                raise ThinkingConfigError(
                    f"thinking.selected={selected!r}：levels 未声明时仅允许 off"
                )
            raise ThinkingConfigError(
                f"thinking.selected={selected!r} 不在 levels 内；可选：{'、'.join(levels)}"
            )
    return {"levels": None if levels is None else list(levels), "selected": selected}


def selected_level(thinking: Mapping[str, Any] | None) -> str | None:
    """取归一化结果中的生效挡位；None / selected 缺失均视为未配置，不发送思考参数。

    ``off`` 是显式关闭而非未配置：原样返回，由调用方决定走 off 通道发关闭参数。
    """
    if thinking is None:
        return None
    return thinking.get("selected")
