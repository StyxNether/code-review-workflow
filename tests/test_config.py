"""配置层：models.json 读写校验、预设与变量名、key 回退链、解析优先级（Design §8.3）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from crw import config

#: 测试占位 key（明显假值，非真实凭据）
PLACEHOLDER_KEY = "placeholder-key"


def _provider(
    name: str = "DeepSeek", *, api_key: str = "", base_url: str | None = None
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": name,
        "base_url": base_url or "https://api.example.com/v1",
        "models": ["model-a"],
        "api_key": api_key,
    }
    return entry


class TestDataDir:
    def test_honours_crw_home(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv(config.HOME_ENV, str(tmp_path / "home"))

        assert config.data_dir() == tmp_path / "home"
        assert config.config_path() == tmp_path / "home" / "models.json"
        assert config.env_registry_path() == tmp_path / "home" / "env-vars.json"

    def test_defaults_to_user_directory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(config.HOME_ENV, raising=False)

        assert config.data_dir() == Path.home() / ".crw"

    def test_blank_override_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(config.HOME_ENV, "   ")

        assert config.data_dir() == Path.home() / ".crw"


class TestEnvVarNames:
    def test_presets_use_official_short_names(self) -> None:
        assert config.provider_env_var("DeepSeek") == "CRW_DEEPSEEK_API_KEY"
        assert config.provider_env_var("Kimi (Moonshot)") == "CRW_MOONSHOT_API_KEY"
        assert config.provider_env_var("OpenAI") == "CRW_OPENAI_API_KEY"
        assert config.provider_env_var("阿里云百炼") == "CRW_DASHSCOPE_API_KEY"
        assert config.provider_env_var("Z.ai API") == "CRW_ZAI_API_KEY"
        assert config.provider_env_var("BigModel API") == "CRW_ZHIPUAI_API_KEY"

    def test_legacy_var_is_the_cra_counterpart(self) -> None:
        """与既有工具链的 CRA_* 一一对应：老机器上的 key 零重配可用。"""
        for preset in config.PROVIDER_PRESETS:
            legacy = config.legacy_env_var(preset.name)
            assert legacy == preset.env_var.replace("CRW_", "CRA_", 1)

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("my provider", "CRW_MY_PROVIDER_API_KEY"),
            ("deepseek", "CRW_DEEPSEEK_API_KEY"),
            ("a--b", "CRW_A_B_API_KEY"),
            ("_x_", "CRW_X_API_KEY"),
        ],
    )
    def test_custom_provider_normalisation(self, name: str, expected: str) -> None:
        assert config.env_var_for_provider(name) == expected

    def test_non_ascii_name_gets_a_unique_digest_variable(self) -> None:
        """纯非 ASCII 名以摘要兜底：不同供应商不得共用同一变量（防 key 串用）。"""
        first = config.env_var_for_provider("供应商甲")
        second = config.env_var_for_provider("供应商乙")

        assert first != second
        assert first.startswith("CRW_") and first.endswith("_API_KEY")


class TestKeyFallback:
    def test_entry_key_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CRW_DEEPSEEK_API_KEY", "placeholder-env")

        assert config.resolve_provider_api_key("DeepSeek", PLACEHOLDER_KEY) == PLACEHOLDER_KEY

    def test_chain_crw_then_cra_then_legacy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SE_CodeAgent", "placeholder-legacy-file")
        assert config.resolve_provider_api_key("DeepSeek", None) == "placeholder-legacy-file"

        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-cra")
        assert config.resolve_provider_api_key("DeepSeek", None) == "placeholder-cra"

        monkeypatch.setenv("CRW_DEEPSEEK_API_KEY", "placeholder-crw")
        assert config.resolve_provider_api_key("DeepSeek", None) == "placeholder-crw"

    def test_no_source_returns_none(self) -> None:
        assert config.resolve_provider_api_key("DeepSeek", None) is None

    def test_env_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CRA_MODEL", "legacy-model")
        assert config.get_model() == "legacy-model"
        monkeypatch.setenv("CRW_MODEL", "crw-model")
        assert config.get_model() == "crw-model"
        monkeypatch.delenv("CRW_MODEL")
        monkeypatch.delenv("CRA_MODEL")
        assert config.get_model() == config.DEFAULT_MODEL

    def test_int_env_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert config.get_exec_timeout() == config.DEFAULT_EXEC_TIMEOUT
        assert config.get_max_context_tokens() == config.DEFAULT_MAX_CONTEXT_TOKENS

        monkeypatch.setenv("CRW_EXEC_TIMEOUT", "30")
        monkeypatch.setenv("CRW_MAX_CONTEXT_TOKENS", "1234")
        assert config.get_exec_timeout() == 30
        assert config.get_max_context_tokens() == 1234

    @pytest.mark.parametrize("name", ["CRW_EXEC_TIMEOUT", "CRW_MAX_CONTEXT_TOKENS"])
    def test_illegal_int_env_is_a_readable_config_error(
        self, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        monkeypatch.setenv(name, "十秒")

        with pytest.raises(config.ConfigError, match=name):
            config.get_exec_timeout() if "EXEC" in name else config.get_max_context_tokens()


class TestValidate:
    def test_missing_file_is_empty_config(self) -> None:
        assert config.load_raw() == {}

    def test_broken_json_is_a_config_error(self) -> None:
        config.config_path().parent.mkdir(parents=True, exist_ok=True)
        config.config_path().write_text("{ 不是 JSON", encoding="utf-8")

        with pytest.raises(config.ConfigError, match="读取失败"):
            config.load_raw()

    @pytest.mark.parametrize(
        ("data", "match"),
        [
            ([], "顶层必须是 JSON 对象"),
            ({"providers": {}}, "providers 必须是数组"),
            ({"providers": [{"name": "", "base_url": "u", "models": ["m"]}]}, "缺少非空 name"),
            ({"providers": [{"name": "a/b", "base_url": "u", "models": ["m"]}]}, "不得含"),
            ({"providers": [{"name": "a", "models": ["m"]}]}, "缺少非空 base_url"),
            ({"providers": [{"name": "a", "base_url": "u", "models": []}]}, "models 必须是"),
            (
                {"providers": [{"name": "a", "base_url": "u", "models": ["m"], "api_key": 1}]},
                "必须是字符串",
            ),
            ({"providers": [], "default": "only-provider"}, "应为 '供应商/模型'"),
            ({"providers": [], "default": "p/m"}, "引用的供应商 'p' 不存在"),
            (
                {
                    "providers": [_provider("p")],
                    "default": "p/missing",
                },
                "不在供应商 'p' 的清单内",
            ),
            (
                {"providers": [_provider("p"), _provider("p")]},
                "供应商名重复",
            ),
        ],
    )
    def test_invalid_shapes(self, data: Any, match: str) -> None:
        with pytest.raises(config.ConfigError, match=match):
            config.validate_config(data)

    def test_accepts_a_minimal_valid_config(self) -> None:
        config.validate_config({"providers": [_provider()], "default": "DeepSeek/model-a"})

    def test_keyless_providers_sharing_a_variable_with_different_hosts_are_rejected(self) -> None:
        """同一个回退变量 + 不同接入点主机 = key 串用，加载即报错（Design §8.3）。"""
        data = {
            "providers": [
                _provider("DeepSeek", base_url="https://api.deepseek.com"),
                _provider("deepseek", base_url="https://other.example.com/v1"),
            ]
        }

        with pytest.raises(config.ConfigError, match="互相读到对方的凭据"):
            config.validate_config(data)

    def test_same_host_sharing_is_allowed(self) -> None:
        """同一供应商的多端点（Coding Plan 与 API）共用官方变量名属设计内共享。"""
        data = {
            "providers": [
                _provider("Z.ai API", base_url="https://api.z.ai/api/paas/v4"),
                _provider("Z.ai Coding Plan", base_url="https://api.z.ai/api/coding/paas/v4"),
            ]
        }

        assert config.conflicting_keyless_env_vars(data["providers"]) == {}


class TestSave:
    def test_roundtrip_and_atomicity(self) -> None:
        data = {"providers": [_provider(api_key=PLACEHOLDER_KEY)], "default": "DeepSeek/model-a"}

        config.save_raw(data)

        assert config.load_raw() == data
        assert not list(config.config_path().parent.glob("*.tmp"))  # 不留临时文件

    def test_rejects_invalid_before_writing(self) -> None:
        with pytest.raises(config.ConfigError):
            config.save_raw({"providers": [{"name": "a"}]})

        assert not config.config_path().exists()


class TestResolve:
    def test_provider_and_model(self) -> None:
        config.save_raw(
            {
                "providers": [_provider(api_key=PLACEHOLDER_KEY)],
                "default": "DeepSeek/model-a",
            }
        )

        resolved = config.resolve()

        assert resolved is not None
        assert (resolved.provider, resolved.model, resolved.api_key) == (
            "DeepSeek",
            "model-a",
            PLACEHOLDER_KEY,
        )
        assert resolved.label() == "DeepSeek/model-a"

    def test_explicit_provider_with_model(self) -> None:
        config.save_raw({"providers": [_provider(api_key=PLACEHOLDER_KEY)]})

        resolved = config.resolve(provider="DeepSeek", model="model-a")

        assert resolved is not None and resolved.model == "model-a"

    def test_explicit_model_in_an_unknown_list_is_rejected(self) -> None:
        config.save_raw({"providers": [_provider(api_key=PLACEHOLDER_KEY)]})

        with pytest.raises(config.ConfigError, match="不在供应商"):
            config.resolve(provider="DeepSeek", model="nope")

    def test_model_owned_by_several_providers_needs_provider(self) -> None:
        config.save_raw(
            {
                "providers": [
                    _provider("p1", api_key=PLACEHOLDER_KEY),
                    _provider("p2", api_key=PLACEHOLDER_KEY),
                ]
            }
        )

        with pytest.raises(config.ConfigError, match="请同时用 --provider"):
            config.resolve(model="model-a")

    def test_model_without_any_providers_is_an_env_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """无供应商清单时 ``--model`` 直接覆盖 ``CRW_MODEL``；连 key 都没有则为"未配置"。"""
        assert config.resolve(model="ghost") is None

        monkeypatch.setenv("CRW_DEEPSEEK_API_KEY", PLACEHOLDER_KEY)
        resolved = config.resolve(model="ghost")

        assert resolved is not None and resolved.model == "ghost"

    def test_unknown_model_with_providers_is_a_config_error(self) -> None:
        config.save_raw({"providers": [_provider(api_key=PLACEHOLDER_KEY)]})

        with pytest.raises(config.ConfigError, match="不在任何已配置供应商"):
            config.resolve(model="ghost")

    def test_default_pointing_at_a_keyless_provider_is_a_config_error(self) -> None:
        config.save_raw({"providers": [_provider()], "default": "DeepSeek/model-a"})

        with pytest.raises(config.ConfigError, match="未配置 api_key"):
            config.resolve()

    def test_keyless_provider_without_default_falls_back_to_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """无 ``default`` 时不拦截在供应商条目上：环境变量回退仍可用，否则返回 None。"""
        config.save_raw({"providers": [_provider()]})
        assert config.resolve() is None

        monkeypatch.setenv("CRW_DEEPSEEK_API_KEY", PLACEHOLDER_KEY)
        resolved = config.resolve()

        assert resolved is not None and resolved.provider is None

    def test_returns_none_without_any_configuration(self) -> None:
        assert config.resolve() is None

    def test_env_fallback_when_no_models_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CRW_DEEPSEEK_API_KEY", PLACEHOLDER_KEY)
        monkeypatch.setenv("CRW_MODEL", "crw-model")

        resolved = config.resolve()

        assert resolved is not None
        assert resolved.provider is None
        assert resolved.model == "crw-model"
        assert resolved.label() == "api.deepseek.com/crw-model"  # 无供应商名时用接入点主机


class TestEnvRegistry:
    def test_register_is_idempotent_and_readable(self) -> None:
        config.register_env_var("CRW_DEEPSEEK_API_KEY")
        config.register_env_var("CRW_DEEPSEEK_API_KEY")
        config.register_env_var("CRW_MOONSHOT_API_KEY")

        assert config.registered_env_vars() == [
            "CRW_DEEPSEEK_API_KEY",
            "CRW_MOONSHOT_API_KEY",
        ]

    def test_missing_or_broken_registry_is_empty(self) -> None:
        assert config.registered_env_vars() == []

        config.env_registry_path().parent.mkdir(parents=True, exist_ok=True)
        config.env_registry_path().write_text("{ 坏", encoding="utf-8")

        assert config.registered_env_vars() == []

    def test_non_list_payload_is_empty(self) -> None:
        config.env_registry_path().parent.mkdir(parents=True, exist_ok=True)
        config.env_registry_path().write_text(json.dumps({"env_vars": "oops"}), encoding="utf-8")

        assert config.registered_env_vars() == []


class TestProviderProbe:
    def test_missing_key_is_reported_without_probing(self) -> None:
        status, detail = config.test_provider(_provider())

        assert status == "error"
        assert "未配置 key" in detail

    def test_probe_delegates_to_llm_with_resolved_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def _probe(*, base_url: str, api_key: str, model_name: str) -> tuple[str, str]:
            captured.update(base_url=base_url, api_key=api_key, model_name=model_name)
            return "ok", ""

        from crw import llm

        monkeypatch.setattr(llm, "probe", _probe)

        assert config.test_provider(_provider(api_key=PLACEHOLDER_KEY)) == ("ok", "")
        assert captured == {
            "base_url": "https://api.example.com/v1",
            "api_key": PLACEHOLDER_KEY,
            "model_name": "model-a",
        }


def test_context_budget_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    """0 / 负数预算会让护栏恒触发（静默禁用工具）：按配置错误报出，而不是照单执行。"""
    monkeypatch.setenv("CRW_MAX_CONTEXT_TOKENS", "0")

    with pytest.raises(config.ConfigError, match="CRW_MAX_CONTEXT_TOKENS"):
        config.get_max_context_tokens()


def test_legacy_env_var_rejects_a_malformed_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    """预设表若给出不符合前缀约定的变量名，历史变量名的推导显式报错（不静默拼错名）。"""

    class _Preset(config.ProviderPreset):
        pass

    monkeypatch.setattr(
        config,
        "PROVIDER_PRESETS",
        (_Preset("X", "https://x.example.com", ("m",), "WRONG_X_API_KEY"),),
    )

    with pytest.raises(config.ConfigError, match="前缀约定"):
        config.legacy_env_var("X")


# ---- 模型条目两态与思考挡位（Design §8.3 的 M3.5 规格） ----


def _thinking_config(**overrides: object) -> dict:
    entry: dict = {
        "default": "DeepSeek/deepseek-flash",
        "providers": [
            {
                "name": "DeepSeek",
                "base_url": "https://api.deepseek.com",
                "api_key": "placeholder-key",
                "models": [
                    {
                        "name": "deepseek-flash",
                        "thinking": {"levels": ["off", "low", "high"], "selected": "high"},
                    }
                ],
            }
        ],
    }
    entry.update(overrides)
    return entry


class TestThinkingModelEntries:
    def test_object_form_is_accepted(self) -> None:
        config.validate_config(_thinking_config())

    def test_string_form_still_means_no_thinking(self) -> None:
        data = _thinking_config()
        data["providers"][0]["models"] = ["deepseek-flash"]

        config.validate_config(data)
        assert config.model_thinking_config(["deepseek-flash"], "deepseek-flash") is None

    @pytest.mark.parametrize(
        ("models", "match"),
        [
            ([{"thinking": {"levels": ["high"]}}], "models"),  # 缺 name
            ([{"name": "m", "extra": 1}], "未知字段"),
            ([{"name": "m", "thinking": {"levels": []}}], "不能为空数组"),
            ([{"name": "m", "thinking": {"levels": ["high"], "selected": "low"}}], "不在 levels"),
            ([{"name": "m", "thinking": {"levels": ["high", "high"]}}], "重复挡位"),
            ([{"name": "m"}, {"name": "m"}], "模型名重复"),
            ([{"name": "m", "thinking": "high"}], "thinking 配置须为对象"),
        ],
    )
    def test_invalid_object_forms_are_rejected(self, models: list, match: str) -> None:
        data = _thinking_config()
        data["providers"][0]["models"] = models
        data.pop("default")

        with pytest.raises(config.ConfigError, match=match):
            config.validate_config(data)

    def test_resolve_reports_the_selected_level(self) -> None:
        """解析结果带上生效挡位，标签按 ``供应商/模型（思考：…）`` 回显。"""
        config.save_raw(_thinking_config())

        resolved = config.resolve()

        assert resolved is not None
        assert resolved.thinking == "high"
        assert resolved.label() == "DeepSeek/deepseek-flash（思考：high）"

    def test_unconfigured_level_keeps_the_plain_label(self) -> None:
        """未配置 selected（或纯字符串条目）：标签不带括号，行为与 M3 一致。"""
        data = _thinking_config()
        data["providers"][0]["models"] = [
            {"name": "deepseek-flash", "thinking": {"levels": ["high"]}}
        ]
        config.save_raw(data)

        resolved = config.resolve()

        assert resolved is not None and resolved.thinking is None
        assert resolved.label() == "DeepSeek/deepseek-flash"


def test_exec_timeout_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    """CRW_EXEC_TIMEOUT 必须为正整数：0 / 负数按配置错误报出（与上下文预算同口径）。"""
    monkeypatch.setenv("CRW_EXEC_TIMEOUT", "0")
    with pytest.raises(config.ConfigError, match="CRW_EXEC_TIMEOUT"):
        config.get_exec_timeout()

    monkeypatch.setenv("CRW_EXEC_TIMEOUT", "-5")
    with pytest.raises(config.ConfigError, match="正整数"):
        config.get_exec_timeout()


def test_test_provider_survives_invalid_model_entries() -> None:
    """非法模型条目（无可用名称）不炸栈：给可读结论（正常加载会先被 validate_config 拦下）。"""
    status, detail = config.test_provider(
        {"name": "X", "base_url": "https://x.example.invalid", "models": [123]}
    )

    assert status == "error" and "未配置 key" in detail


def test_importable_providers_keeps_object_entries() -> None:
    """前作配置导入：对象形态的模型条目原样沿用（结构同源口径），不再整条静默丢弃。"""
    data = {
        "providers": [
            {
                "name": "DeepSeek",
                "base_url": "https://api.deepseek.com",
                "models": [
                    "deepseek-flash",
                    {"name": "deepseek-v4-pro", "thinking": {"levels": ["high"]}},
                ],
            }
        ]
    }

    entries = config.importable_providers(data)

    assert entries[0]["models"] == [
        "deepseek-flash",
        {"name": "deepseek-v4-pro", "thinking": {"levels": ["high"]}},
    ]


def _preset_levels(preset: config.ProviderPreset, model: str) -> list[str] | None:
    payload = config.model_thinking_config(list(preset.models), model)
    return payload["levels"] if payload else None


def test_presets_prefill_thinking_levels_per_user_rulings() -> None:
    """预设回填锁住 2026-10-14 的四项用户裁决。

    裁决：GLM 接受"不可关闭思考"→ 不预填；Kimi k2.6 新增 on 映射（thinking.type 开启）；
    百炼新增 enable-off 映射（enable_thinking 关闭）；OpenAI 按 5.6 系列口径预填。
    """
    presets = {preset.name: preset for preset in config.PROVIDER_PRESETS}

    assert _preset_levels(presets["DeepSeek"], "deepseek-flash") == [
        "off",
        "none",
        "low",
        "high",
        "max",
    ]
    assert _preset_levels(presets["Kimi (Moonshot)"], "kimi-k2.6") == ["off", "on"]
    assert _preset_levels(presets["Kimi (Moonshot)"], "kimi-k3") == ["low", "high"]
    assert _preset_levels(presets["阿里云百炼"], "qwen3.8-max") == ["enabled", "enable-off"]
    assert _preset_levels(presets["OpenAI"], "gpt-6-astra") == ["none", "low", "medium", "high"]
    # GLM（Z.ai / BigModel）：思考恒开、无可配置项 → 条目保持纯字符串（不预填）
    for name in ("Z.ai Coding Plan", "Z.ai API", "BigModel Coding Plan", "BigModel API"):
        assert all(isinstance(item, str) for item in presets[name].models)


# ---- settings.json：运行默认值（Design §8.3）----


def test_settings_defaults_when_the_file_is_missing() -> None:
    assert config.load_settings() == {
        "rounds": {
            "default": config.DEFAULT_ROUNDS_DEFAULT,
            "limit": config.DEFAULT_ROUNDS_LIMIT,
        }
    }
    assert config.rounds_settings() == (
        config.DEFAULT_ROUNDS_DEFAULT,
        config.DEFAULT_ROUNDS_LIMIT,
    )
    assert config.configured_rounds_default() is None  # 未显式配置 → 来源记内置默认


def test_settings_round_trip_and_merge() -> None:
    config.save_settings({"rounds": {"default": 5, "limit": 8}})

    assert config.rounds_settings() == (5, 8)
    assert config.configured_rounds_default() == 5
    assert config.settings_path().is_file()
    assert config.load_settings()["rounds"] == {"default": 5, "limit": 8}


def test_partial_rounds_block_is_rejected() -> None:
    """给出 rounds 就必须给全两个键：配一半是歧义状态，按配置错误报出（Design §8.3）。"""
    config.settings_path().parent.mkdir(parents=True, exist_ok=True)
    config.settings_path().write_text('{"rounds": {"limit": 12}}', encoding="utf-8")

    with pytest.raises(config.ConfigError, match="缺少 default"):
        config.load_settings()


@pytest.mark.parametrize(
    "payload",
    [
        {"rounds": {"default": 30, "limit": 20}},  # default > limit 且超天花板
        {"rounds": {"default": 4, "limit": 3}},  # default > limit
        {"rounds": {"default": 3, "limit": 21}},  # limit 超天花板
        {"rounds": {"default": -1, "limit": 5}},  # 负数
        {"rounds": {"default": True, "limit": 5}},  # 布尔不算整数
        {"rounds": {"default": "3", "limit": 5}},  # 字符串
        {"rounds": {"default": 3}},  # 缺项
        {"rounds": {"default": 3, "limit": 5, "mode": "exact"}},  # 未知键
        {"unknown": {}},  # 顶层未知键
        [],
    ],
)
def test_invalid_settings_are_config_errors(payload: object) -> None:
    """违规按配置错误处理（退出码 2 的上游）；不静默夹取（Design §8.3）。"""
    config.settings_path().parent.mkdir(parents=True, exist_ok=True)
    config.settings_path().write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(config.ConfigError):
        config.load_settings()


def test_corrupt_settings_file_reports_the_path() -> None:
    config.settings_path().parent.mkdir(parents=True, exist_ok=True)
    config.settings_path().write_text("{不是 JSON", encoding="utf-8")

    with pytest.raises(config.ConfigError, match="读取失败"):
        config.rounds_settings()


def test_settings_ceiling_is_twenty() -> None:
    """上限天花板 = 20（改动须评审，防"把上限改到无界"）。"""
    assert config.ROUNDS_LIMIT_CEILING == 20
    config.save_settings({"rounds": {"default": 20, "limit": 20}})
    assert config.rounds_settings() == (20, 20)
