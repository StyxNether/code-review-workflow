"""供应商配置：``~/.crw/models.json`` 读写、内置预设、环境变量回退与解析。

数据目录经 ``CRW_HOME`` 解析（默认 ``~/.crw``）；key 回退链 CRW_ → CRA_ → SE_CodeAgent，
解析优先级为 ``--model/--provider`` > 配置文件 ``default`` > 环境变量（Design §8.3）。
本模块是 key 的唯一读取点——仓库内任何文件、日志、测试样例都不得出现真实密钥。
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

#: 数据目录环境变量：测试与多环境隔离用（Design §9.1）
HOME_ENV = "CRW_HOME"

#: 默认数据目录名（用户主目录下）
DEFAULT_HOME_NAME = ".crw"

#: 内置默认模型与接入点（沿用官方短名）
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"

#: run_python 超时秒数默认值（Design §6.2 #3 保留 CRW_EXEC_TIMEOUT）
DEFAULT_EXEC_TIMEOUT = 10

#: 单节点请求预算（Design §4.6 的任务级护栏）
DEFAULT_MAX_CONTEXT_TOKENS = 100_000

#: 与既有工具链兼容的历史变量（key 回退链末端，前作亦用它）
LEGACY_KEY_ENV = "SE_CodeAgent"

_KEY_PREFIX = "CRW_"
_LEGACY_PREFIX = "CRA_"


class ConfigError(Exception):
    """配置存在但无法使用（损坏、引用失效、参数无法解析）；调用方按运行错误退出 2。"""


# ---- 供应商预设（离线内置，不联网拉取清单）----


@dataclass(frozen=True)
class ProviderPreset:
    """供应商预设：官方接入点 + 预填模型清单 + key 环境变量名（CRW_ 统一格式）。

    预填项均可在交互中修改；模型名有时效性，``crw config test`` 可兜底验证。
    环境变量名与既有工具链的 ``CRA_*`` 一一对应，保证"零重配"。
    """

    name: str
    base_url: str
    models: tuple[str, ...]
    env_var: str


PROVIDER_PRESETS: tuple[ProviderPreset, ...] = (
    ProviderPreset(
        "DeepSeek",
        "https://api.deepseek.com",
        ("deepseek-flash", "deepseek-v4-pro"),
        "CRW_DEEPSEEK_API_KEY",
    ),
    ProviderPreset(
        "Kimi (Moonshot)",
        "https://api.moonshot.cn/v1",
        ("kimi-k3", "kimi-k2.6"),
        "CRW_MOONSHOT_API_KEY",
    ),
    ProviderPreset(
        "OpenAI",
        "https://api.openai.com/v1",
        ("gpt-6-astra", "gpt-6-luna"),
        "CRW_OPENAI_API_KEY",
    ),
    ProviderPreset(
        "阿里云百炼",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ("qwen3.8-max", "qwen3.8-flash"),
        "CRW_DASHSCOPE_API_KEY",
    ),
    ProviderPreset(
        "Z.ai Coding Plan",
        "https://api.z.ai/api/coding/paas/v4",
        ("glm-5.3", "glm-5.3-flash"),
        "CRW_ZAI_API_KEY",
    ),
    ProviderPreset(
        "Z.ai API",
        "https://api.z.ai/api/paas/v4",
        ("glm-5.3", "glm-5.3-flash"),
        "CRW_ZAI_API_KEY",
    ),
    ProviderPreset(
        "BigModel Coding Plan",
        "https://open.bigmodel.cn/api/coding/paas/v4",
        ("glm-5.3", "glm-5.3-flash"),
        "CRW_ZHIPUAI_API_KEY",
    ),
    ProviderPreset(
        "BigModel API",
        "https://open.bigmodel.cn/api/paas/v4",
        ("glm-5.3", "glm-5.3-flash"),
        "CRW_ZHIPUAI_API_KEY",
    ),
)


def data_dir() -> Path:
    """数据目录根：``CRW_HOME`` 覆盖，默认 ``~/.crw``（Design §8.3、§9.1）。

    配置（models.json）与运行记录同源解析：运行记录经 ``records.runs_root()`` 取
    ``data_dir()/runs``，测试由 conftest 统一指向临时目录。
    """
    override = os.environ.get(HOME_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / DEFAULT_HOME_NAME


def config_path() -> Path:
    """models.json 路径（``data_dir()/models.json``，Design §8.3）。"""
    return data_dir() / "models.json"


def env_registry_path() -> Path:
    """本程序写入的环境变量登记文件（``data_dir()/env-vars.json``）。

    卸载时据此区分"程序自己设置的变量"与"用户已有的同名变量"——只清前者，
    绝不触碰后者（使用者可能在此前已自行设置 CRW_DEEPSEEK_API_KEY 等变量）。
    """
    return data_dir() / "env-vars.json"


# ---- 前作配置探测（Design §6.2 #2 / D7：仅在零配置时提示导入，不读进常态解析路径）----

#: 前作数据目录名（用户主目录下）
LEGACY_HOME_NAME = ".cra"


def legacy_config_path() -> Path:
    """前作配置路径 ``~/.cra/models.json``（只用于"探测存在性 + 提示导入"）。"""
    return Path.home() / LEGACY_HOME_NAME / "models.json"


def load_legacy_config() -> dict[str, Any] | None:
    """读取前作配置；不存在 / 损坏 / 顶层非对象返回 ``None``。

    导入是可选的便利功能：前作文件读不了就当它不存在，不得让 ``crw config`` 因此报错。
    """
    try:
        data = json.loads(legacy_config_path().read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def importable_providers(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """把前作配置的供应商条目规范为本项目形态（只保留结构完整者，丢弃损坏条目）。

    两者 ``models.json`` 结构同源（``default`` + ``providers[{name, base_url, models, api_key?}]``）：
    差异只在数据目录与 key 环境变量前缀，条目本身可直接沿用；此处只做结构过滤，
    避免把半截条目带进新配置（整表校验由调用方执行）。
    """
    entries = data.get("providers") if isinstance(data, Mapping) else None
    result: list[dict[str, Any]] = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        base_url = entry.get("base_url")
        models = entry.get("models")
        if not isinstance(name, str) or not name or "/" in name:
            continue
        if not isinstance(base_url, str) or not base_url:
            continue
        if (
            not isinstance(models, list)
            or not models
            or not all(isinstance(item, str) and item for item in models)
        ):
            continue
        item: dict[str, Any] = {"name": name, "base_url": base_url, "models": list(models)}
        api_key = entry.get("api_key")
        if isinstance(api_key, str):
            item["api_key"] = api_key
        result.append(item)
    return result


def env_var_for_provider(name: str) -> str:
    """自定义供应商 → key 环境变量名：``CRW_供应商名大写_API_KEY``。

    规则：前缀 + 供应商名大写、空格与非字母数字转 ``_`` + ``_API_KEY`` 后缀；非
    ASCII 字符剥离（前缀已保证变量名不以数字开头）。退化输入（纯非 ASCII / 纯
    符号名）以名称摘要生成唯一变量名——不同供应商不得共用同一兜底名，否则 key
    串用会把一家凭据发给另一家的接入点。
    """
    parts: list[str] = []
    for char in name.upper():
        if char.isascii() and char.isalnum():
            parts.append(char)
        elif parts and parts[-1] != "_":
            parts.append("_")
    stem = "".join(parts).strip("_")
    if not stem:
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8].upper()
        return f"{_KEY_PREFIX}{digest}_API_KEY"
    return f"{_KEY_PREFIX}{stem}_API_KEY"


def provider_env_var(name: str) -> str:
    """供应商 key 的主环境变量名：预设取官方短名，自定义按归一化规则生成。"""
    for preset in PROVIDER_PRESETS:
        if preset.name == name:
            return preset.env_var
    return env_var_for_provider(name)


def legacy_env_var(name: str) -> str:
    """同一供应商在既有工具链中的变量名（``CRW_`` → ``CRA_``，回退链第二环）。

    前缀不一致时显式报错而不是静默拼出一个取不到 key 的变量名（预设表扩展时的护栏）。
    """
    stem = provider_env_var(name)
    if not stem.startswith(_KEY_PREFIX):
        raise ConfigError(f"预设环境变量名不符合 {_KEY_PREFIX} 前缀约定：{stem!r}")
    return _LEGACY_PREFIX + stem.removeprefix(_KEY_PREFIX)


def resolve_provider_api_key(name: str, entry_api_key: str | None) -> str | None:
    """供应商 key 回退链：条目 api_key → ``CRW_*`` → ``CRA_*`` → ``SE_CodeAgent``。

    key 是否与供应商匹配由 API 认证环节暴露，此处只做存在性回退。
    """
    if entry_api_key:
        return entry_api_key
    return (
        os.environ.get(provider_env_var(name))
        or os.environ.get(legacy_env_var(name))
        or os.environ.get(LEGACY_KEY_ENV)
    )


# ---- 环境变量登记（设置时登记、卸载时只清登记在案的变量）----


def registered_env_vars() -> list[str]:
    """读取登记清单；文件缺失或损坏返回空表（卸载侧据此保守跳过清理）。"""
    path = env_registry_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    names = data.get("env_vars") if isinstance(data, dict) else None
    if not isinstance(names, list):
        return []
    return [name for name in names if isinstance(name, str) and name]


def register_env_var(name: str) -> None:
    """登记一个由本程序写入的环境变量名（去重、原子替换）。

    Windows 为 setx 持久写入；POSIX 为当前进程即时设置（持久化取决于用户是否执行
    ``export``，登记项在卸载时用于打印待删行指引）。写盘失败上抛 ``OSError``，由
    调用方提示——登记缺失会导致卸载时无法自动清理该变量。
    """
    names = registered_env_vars()
    if name in names:
        return
    names.append(name)
    payload = json.dumps({"env_vars": names}, ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(env_registry_path(), payload)


@dataclass(frozen=True)
class ResolvedModel:
    """一次 LLM 连接的解析结果；``provider`` 为 None 表示来自环境变量回退。

    ``api_key`` 恒为非空字符串：所有返回路径都保证 key 存在（缺失时抛
    :class:`ConfigError` 或返回 ``None``），调用方无须再做空值回退。
    """

    model: str
    base_url: str
    api_key: str
    provider: str | None

    def label(self) -> str:
        """模型标识（写入 ``state["model"]``）：``供应商/模型名``（Design §4.3）。

        环境变量回退路径没有供应商名，用接入点主机名代替，保证标识仍可辨识。
        """
        owner = self.provider or (urlsplit(self.base_url).hostname or "环境变量")
        return f"{owner}/{self.model}"


# ---- 环境变量回退路径（models.json 不存在或未覆盖相应项时生效）----


def get_api_key() -> str | None:
    """默认供应商（DeepSeek）的 key 回退：CRW_ → CRA_ → SE_CodeAgent。"""
    return resolve_provider_api_key("DeepSeek", None)


def get_model() -> str:
    """模型名：CRW_MODEL → CRA_MODEL → 内置默认。"""
    return os.environ.get("CRW_MODEL") or os.environ.get("CRA_MODEL") or DEFAULT_MODEL


def get_base_url() -> str:
    """OpenAI 兼容接入点：CRW_BASE_URL → CRA_BASE_URL → 内置默认。"""
    return os.environ.get("CRW_BASE_URL") or os.environ.get("CRA_BASE_URL") or DEFAULT_BASE_URL


def _get_int_env(name: str, default: int) -> int:
    """读取整数型环境变量：未设置或空白取默认值，非法值给可读报错。"""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"环境变量 {name} 必须是整数，当前值：{raw!r}") from exc


def get_exec_timeout() -> int:
    """run_python 超时秒数：``CRW_EXEC_TIMEOUT``，默认 10。"""
    return _get_int_env("CRW_EXEC_TIMEOUT", DEFAULT_EXEC_TIMEOUT)


def get_max_context_tokens() -> int:
    """单节点请求的上下文预算：``CRW_MAX_CONTEXT_TOKENS``，默认 100000（Design §4.6）。

    必须为正：0 / 负数会让护栏恒触发（等于静默禁用工具），属配置错误而非可运行配置。
    """
    value = _get_int_env("CRW_MAX_CONTEXT_TOKENS", DEFAULT_MAX_CONTEXT_TOKENS)
    if value <= 0:
        raise ConfigError(
            f"环境变量 CRW_MAX_CONTEXT_TOKENS 必须是正整数（单节点上下文预算），当前值：{value}"
        )
    return value


# ---- models.json：加载、校验与保存 ----


def load_raw() -> dict[str, Any]:
    """读取 models.json；文件不存在返回空配置，损坏抛 :class:`ConfigError`。

    结构校验在读取时执行：损坏的配置（类型不符、default 格式非法、供应商名含
    "/"）在入口处即报错，避免下游各自防御。
    """
    path = config_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"配置文件 {path} 读取失败：{exc}") from exc
    validate_config(data)
    return data


def save_raw(data: dict[str, Any]) -> None:
    """校验后原子写入 models.json（先写临时文件再替换，防写一半损坏）。

    POSIX 上尽力收紧文件权限为 0600（该文件可能含明文 key）；权限收紧失败不阻塞
    写入，权限与备份注意事项由 ``crw config`` 在写入时向用户提示。
    """
    validate_config(data)
    _atomic_write_text(config_path(), json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def validate_config(data: Any) -> None:
    """models.json 结构校验；不合法抛 :class:`ConfigError`。

    ``api_key`` 允许缺失或空串（回退环境变量）；其余字段类型不符、供应商名含 "/"、
    ``default`` 不是"供应商/模型"形态均视为配置损坏。
    """
    if not isinstance(data, dict):
        raise ConfigError("配置文件结构无效：顶层必须是 JSON 对象。")
    providers = data.get("providers", [])
    if not isinstance(providers, list):
        raise ConfigError("配置文件结构无效：providers 必须是数组。")
    for index, entry in enumerate(providers):
        where = f"providers[{index}]"
        if not isinstance(entry, dict):
            raise ConfigError(f"配置文件结构无效：{where} 必须是对象。")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ConfigError(f"配置文件结构无效：{where} 缺少非空 name。")
        if "/" in name:
            raise ConfigError(f"配置文件结构无效：供应商名 '{name}' 不得含 '/'。")
        base_url = entry.get("base_url")
        if not isinstance(base_url, str) or not base_url:
            raise ConfigError(f"配置文件结构无效：供应商 '{name}' 缺少非空 base_url。")
        models = entry.get("models")
        if (
            not isinstance(models, list)
            or not models
            or not all(isinstance(item, str) and item for item in models)
        ):
            raise ConfigError(f"配置文件结构无效：供应商 '{name}' 的 models 必须是非空字符串数组。")
        api_key = entry.get("api_key")
        if api_key is not None and not isinstance(api_key, str):
            raise ConfigError(f"配置文件结构无效：供应商 '{name}' 的 api_key 必须是字符串。")
    default = data.get("default")
    if default is not None:
        if not isinstance(default, str) or "/" not in default:
            raise ConfigError(f"配置文件结构无效：default {default!r} 应为 '供应商/模型'。")
        provider_name, _, model_name = default.partition("/")
        if not provider_name or not model_name:
            raise ConfigError(f"配置文件结构无效：default {default!r} 的供应商与模型名均不能为空。")
        # 引用完整性与供应商重名：问题在 crw config 写入时即暴露（手改文件在加载时
        # 同样报错），不必等到运行期 resolve() 才失败
        entry = next((item for item in providers if item.get("name") == provider_name), None)
        if entry is None:
            raise ConfigError(f"配置文件结构无效：default 引用的供应商 '{provider_name}' 不存在。")
        if model_name not in entry["models"]:
            raise ConfigError(
                f"配置文件结构无效：default 引用的模型 '{model_name}' 不在供应商 "
                f"'{provider_name}' 的清单内。"
            )
    names = [item.get("name") for item in providers]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ConfigError(f"配置文件结构无效：供应商名重复：{', '.join(duplicates)}。")
    collisions = conflicting_keyless_env_vars(providers)
    if collisions:
        detail = "；".join(
            f"{var} ← {', '.join(names)}" for var, names in sorted(collisions.items())
        )
        raise ConfigError(
            "配置文件结构无效：以下供应商未设置 api_key，且回退到同一个 key 环境变量"
            f"（接入点主机不同，会互相读到对方的凭据）：{detail}。"
            "请为其中一方设置 api_key，或修改供应商名使变量名可区分。"
        )


def conflicting_keyless_env_vars(providers: list[dict[str, Any]]) -> dict[str, list[str]]:
    """未设 api_key 的供应商中，回退环境变量名相同但接入点主机不同的分组。

    环境变量名归一化是有损映射（"DeepSeek" 与 "deepseek" 同名）；两个都依赖环境变量
    回退的供应商撞名时，会把同一凭据发往不同接入点（key 串用）。同一主机（同一供应商
    的多端点，如 Z.ai / BigModel 的 Coding Plan 与 API 双预设共用官方变量名）属设计内
    共享，不算冲突。主变量名与历史变量名（CRW_/CRA_）都参与判定——回退链上任一环
    撞名都会造成串用。回退链末端共享的 ``SE_CodeAgent`` 是有意保留的最后回退，不属
    本函数的判定范围。
    """
    by_var: dict[str, list[tuple[str, str]]] = {}
    for entry in providers:
        if entry.get("api_key"):
            continue
        host = urlsplit(entry.get("base_url", "")).hostname or ""
        for var in (provider_env_var(entry["name"]), legacy_env_var(entry["name"])):
            by_var.setdefault(var, []).append((entry["name"], host))
    return {
        var: [name for name, _ in entries]
        for var, entries in by_var.items()
        if len({host for _, host in entries}) > 1
    }


def _resolve_provider(
    providers: list[dict[str, Any]], provider: str, model: str | None
) -> ResolvedModel:
    """解析到指定供应商：``--model`` 须在其清单内匹配，缺省取 ``models[0]``。"""
    entry = next((item for item in providers if item.get("name") == provider), None)
    if entry is None:
        raise ConfigError(f"供应商 '{provider}' 未在 {config_path()} 中配置，请先运行 crw config。")
    models = entry["models"]
    if model is not None and model not in models:
        raise ConfigError(
            f"模型 '{model}' 不在供应商 '{provider}' 的清单内（{', '.join(models)}）；"
            "可用 crw config 添加，或从清单中选择。"
        )
    model_name = model if model is not None else models[0]
    api_key = resolve_provider_api_key(provider, entry.get("api_key"))
    if not api_key:
        raise ConfigError(
            f"供应商 '{provider}' 未配置 api_key，环境变量 {provider_env_var(provider)}、"
            f"{legacy_env_var(provider)} 与 {LEGACY_KEY_ENV} 也未设置；"
            "请运行 crw config 补充 key，或设置环境变量。"
        )
    return ResolvedModel(
        model=model_name, base_url=entry["base_url"], api_key=api_key, provider=provider
    )


def resolve(*, model: str | None = None, provider: str | None = None) -> ResolvedModel | None:
    """按配置优先级解析本次使用的模型连接（Design §8.3）。

    Args:
        model: ``--model`` 覆盖值（须能在已配置清单中定位）。
        provider: ``--provider`` 覆盖值。

    Returns:
        :class:`ResolvedModel`；返回 ``None`` 表示无任何可用配置（调用方打印
        ``crw config`` 引导后退出 2）。配置存在但损坏、``default`` 引用失效或
        ``--model/--provider`` 无法解析时抛 :class:`ConfigError`。
    """
    data = load_raw()
    providers = data.get("providers") or []
    if provider is not None:
        return _resolve_provider(providers, provider, model)
    if model is not None and providers:
        owners = sorted({entry["name"] for entry in providers if model in entry["models"]})
        if len(owners) > 1:
            raise ConfigError(
                f"模型 '{model}' 在多个供应商中存在（{', '.join(owners)}），"
                "请同时用 --provider 指定供应商。"
            )
        if owners:
            return _resolve_provider(providers, owners[0], model)
        raise ConfigError(
            f"模型 '{model}' 不在任何已配置供应商的清单内，请检查拼写或用 crw config 添加。"
        )
    default = data.get("default")
    if default:
        provider_name, _, model_name = str(default).partition("/")
        return _resolve_provider(providers, provider_name, model_name)
    # 环境变量回退；无供应商清单时 --model 直接覆盖 CRW_MODEL
    api_key = get_api_key()
    if not api_key:
        return None
    return ResolvedModel(
        model=model or get_model(),
        base_url=get_base_url(),
        api_key=api_key,
        provider=None,
    )


def test_provider(entry: dict[str, Any]) -> tuple[str, str]:
    """单个供应商的连通性自检（``crw config test``）：发一次 ``max_tokens=1`` 的请求。

    key 走与 :func:`resolve` 相同的回退链；条目缺 key 时直接给出可读结论，不发起请求。

    Returns:
        ``(状态, 详情)``：ok / auth（401/403）/ timeout / error，语义见 :func:`crw.llm.probe`。
    """
    # llm 在模块顶层 import 本模块（取环境变量回退与数据目录），此处按需 import 避免成环
    from . import llm

    name = str(entry.get("name") or "")
    models = entry.get("models")
    model_name = str(models[0]) if isinstance(models, list) and models else ""
    api_key = resolve_provider_api_key(name, entry.get("api_key"))
    if not api_key:
        return "error", (
            f"未配置 key（条目与环境变量 {provider_env_var(name)} / {legacy_env_var(name)} / "
            f"{LEGACY_KEY_ENV} 均缺）"
        )
    return llm.probe(
        base_url=str(entry.get("base_url") or ""), api_key=api_key, model_name=model_name
    )


def _atomic_write_text(path: Path, text: str) -> None:
    """原子写文本（临时文件 + 替换，POSIX 尽力 0600）。

    本模块自带这一原语而不复用 ``workflow.records``：records 依赖本模块取数据目录
    （``records → config``），反向 import 会成环；两处目录与权限语义也不同（用户配置
    目录 vs 运行记录目录）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    if os.name == "posix":
        try:
            tmp.chmod(0o600)
        except OSError:
            pass  # 权限收紧失败不阻塞写入（提示已在交互层给出）
    os.replace(tmp, path)
