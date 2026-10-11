"""pytest 全局夹具：兜底阻断真实网络，保证测试不发起任何网络调用。"""

from __future__ import annotations

import socket

import pytest


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """禁止测试内建连与域名解析（LLM 一律 mock，连通性验证走 crw config test）。

    拦截面为 Python 层可替换的出站建连与解析入口；已知漏口：C 层直连解析系统调用的
    平台专有发送接口（sendmsg / sendmmsg 等），故本闸门是"兜底"而非密不透风的沙箱。
    """

    def _forbidden(*args: object, **kwargs: object) -> None:
        raise RuntimeError("测试禁止真实网络调用")

    monkeypatch.setattr(socket, "create_connection", _forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", _forbidden)
    monkeypatch.setattr(socket, "gethostbyname", _forbidden)
    monkeypatch.setattr(socket, "gethostbyname_ex", _forbidden)
    monkeypatch.setattr(socket, "gethostbyaddr", _forbidden)
    monkeypatch.setattr(socket.socket, "connect", _forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", _forbidden)
    monkeypatch.setattr(socket.socket, "sendto", _forbidden)
