"""pytest 全局夹具：数据目录重定向、真实网络兜底阻断与引擎注入接缝。"""

from __future__ import annotations

import datetime
import socket
from collections.abc import Callable
from pathlib import Path

import pytest

#: 固定时钟（引擎注入接缝，Design §9.1）：断言可复现
FIXED_TIME = "2026-10-12T10:00:00+08:00"

#: 固定日期部分（由固定时钟派生，运行号前缀用；改 FIXED_TIME 即同步）
FIXED_DAY = datetime.datetime.fromisoformat(FIXED_TIME).strftime("%Y%m%d")


@pytest.fixture(autouse=True)
def _isolate_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """把配置与运行记录根目录重定向到临时目录（Design §9.1）：测试不写真实用户目录。"""
    monkeypatch.setenv("CRW_HOME", str(tmp_path / "crw-home"))


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
