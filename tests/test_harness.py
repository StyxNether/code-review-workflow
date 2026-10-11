"""测试基础设施自测：网络闸门与数据目录重定向（Design §9.1）。

闸门与重定向都由 autouse 夹具提供，若被误改套件会静默失去约束，故在此锁定其行为。
出网目标一律用保留网段 **TEST-NET-1**（``192.0.2.0/24``，永不路由）与 ``.invalid`` 域名：
万一闸门失效，用例虽然会红，但不会真的打到外部主机。
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest

_ORIGINAL_CREATE_CONNECTION = socket.create_connection


@pytest.mark.parametrize(
    "attempt",
    [
        lambda: socket.create_connection(("192.0.2.1", 80), timeout=0.1),
        lambda: socket.getaddrinfo("example.invalid", 80),
        lambda: socket.getnameinfo(("192.0.2.1", 80), 0),
        lambda: socket.gethostbyname("example.invalid"),
        lambda: socket.gethostbyname_ex("example.invalid"),
        lambda: socket.socket().connect_ex(("192.0.2.1", 80)),
        lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"x", ("192.0.2.1", 53)),
    ],
)
def test_network_gate_blocks_outbound_calls(attempt: object) -> None:
    """出站建连、解析与发送入口在测试内一律被拒。"""
    assert socket.create_connection is not _ORIGINAL_CREATE_CONNECTION, "网络闸门未安装"

    with pytest.raises(RuntimeError, match="测试禁止真实网络调用"):
        attempt()  # type: ignore[operator]


def test_data_dir_is_redirected_into_the_test_tmp_dir(tmp_path: Path) -> None:
    """数据目录经 CRW_HOME 指向本用例的临时目录，测试不写真实 ~/.crw（Design §9.1）。"""
    crw_home = os.environ.get("CRW_HOME")

    assert crw_home, "CRW_HOME 应由 conftest 的 autouse 夹具统一注入"
    assert Path(crw_home).is_relative_to(tmp_path)
    assert Path(crw_home) != Path.home() / ".crw"
