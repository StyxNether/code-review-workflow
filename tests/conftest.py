"""pytest 全局夹具：数据目录重定向与真实网络兜底阻断。"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest


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
