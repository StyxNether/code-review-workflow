"""pytest 全局夹具：数据目录重定向、真实网络兜底阻断与引擎 / 模型注入接缝。"""

from __future__ import annotations

import datetime
import os
import socket
from collections.abc import Callable
from pathlib import Path

import pytest

from crw import config as config_mod

#: 固定时钟（引擎注入接缝，Design §9.1）：断言可复现
FIXED_TIME = "2026-10-12T10:00:00+08:00"

#: 固定日期部分（由固定时钟派生，运行号前缀用；改 FIXED_TIME 即同步）
FIXED_DAY = datetime.datetime.fromisoformat(FIXED_TIME).strftime("%Y%m%d")

#: 测试用占位 key（`placeholder-*` 构造，不带任何真实供应商前缀）
PLACEHOLDER_KEY = "placeholder-key-for-tests"

#: 会被配置层读取的凭证类环境变量前缀（测试须隔离，避免读到开发机上的真实 key）
_CREDENTIAL_PREFIXES = ("CRW_", "CRA_")
_LEGACY_KEY_ENV = "SE_CodeAgent"
#: openai SDK 自身的回退变量：本项目的客户端恒显式传 key，但隔离它们能让"未配置"分支
#: 在不同开发机上表现一致（不因本机是否配过 OpenAI 而分叉）
_SDK_ENV_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "OPENAI_ORG_ID",
    "OPENAI_ORGANIZATION",
)


@pytest.fixture(autouse=True)
def _isolate_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """把配置与运行记录根目录重定向到临时目录（Design §9.1）：测试不写真实用户目录。"""
    monkeypatch.setenv("CRW_HOME", str(tmp_path / "crw-home"))


@pytest.fixture(autouse=True)
def _isolate_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """清掉开发机上可能存在的 key 环境变量：测试的行为不得取决于本机是否配了 key。

    配置层的回退链（CRW_ → CRA_ → SE_CodeAgent）会读它们；不清掉的话，同一份用例
    在有 key 的机器上会走"已配置"分支、换台机器又走"未配置"分支。
    """
    for name in [*(list(os.environ)), *_SDK_ENV_VARS]:
        if name == "CRW_HOME":
            # 数据目录重定向变量（_isolate_data_dir 设置）必须保留：
            # 前缀删除若不慎命中它，测试会写进真实 ~/.crw（顺序无关的显式豁免）
            continue
        if (
            name == _LEGACY_KEY_ENV
            or name.startswith(_CREDENTIAL_PREFIXES)
            or name in _SDK_ENV_VARS
        ):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def configured_provider(tmp_path: Path) -> Callable[[str], dict[str, object]]:
    """写一份最小可用的 models.json（含默认供应商与占位 key），返回其内容。

    测试里的"已配置"状态经真实配置层落地（走 ``config.resolve`` 的真实代码路径），
    而不是把 resolve 本身 mock 掉。
    """

    def _write(model: str = "deepseek-flash", provider: str = "DeepSeek") -> dict[str, object]:
        data: dict[str, object] = {
            "default": f"{provider}/{model}",
            "providers": [
                {
                    "name": provider,
                    "base_url": "https://api.example.invalid/v1",
                    "models": [model],
                    "api_key": PLACEHOLDER_KEY,
                }
            ],
        }
        config_mod.save_raw(data)
        return data

    return _write


@pytest.fixture
def fake_chat_model(monkeypatch: pytest.MonkeyPatch) -> object:
    """把 ``crw.llm.create_model`` 换成本地脚本化假模型（测试禁真实网络）。

    真实路径（配置解析 → 运行时装配 → create_agent）全部照走，只有"造模型客户端"
    这一步被替换；用例向返回的假模型注入脚本化响应即可驱动端到端流程。
    """
    import fakes

    from crw import llm as llm_mod

    model = fakes.ScriptedChatModel(responses=[])
    monkeypatch.setattr(llm_mod, "create_model", lambda *args, **kwargs: model)
    return model


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """禁止测试内建连与域名解析（LLM 一律 mock，连通性验证走 crw config test）。

    拦截面为可替换的出站建连、解析与发送入口；已知漏口：子进程不继承本闸门
    （M1 起 run_python 的子进程执行需另行约束），以及平台专有发送接口。
    """

    def _forbidden(*args: object, **kwargs: object) -> None:
        raise RuntimeError("测试禁止真实网络调用")

    monkeypatch.setattr(socket, "create_connection", _forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", _forbidden)
    monkeypatch.setattr(socket, "getnameinfo", _forbidden)
    monkeypatch.setattr(socket, "gethostbyname", _forbidden)
    monkeypatch.setattr(socket, "gethostbyname_ex", _forbidden)
    monkeypatch.setattr(socket, "gethostbyaddr", _forbidden)
    monkeypatch.setattr(socket.socket, "connect", _forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", _forbidden)
    monkeypatch.setattr(socket.socket, "sendto", _forbidden)
    for name in ("sendmsg", "sendmmsg"):
        if hasattr(socket.socket, name):
            monkeypatch.setattr(socket.socket, name, _forbidden)


@pytest.fixture
def fixed_clock() -> Callable[[], str]:
    """固定时钟：时间戳与运行号日期可断言（Design §9.1 的注入接口）。"""

    def _clock() -> str:
        return FIXED_TIME

    return _clock


@pytest.fixture
def run_id_sequence() -> Callable[[], str]:
    """递增的确定性运行号生成器（r-<固定日期>-0001、0002…；末段为 4 位小写十六进制）。"""
    counter = {"value": 0}

    def _factory() -> str:
        counter["value"] += 1
        return f"r-{FIXED_DAY}-{counter['value']:04x}"

    return _factory
