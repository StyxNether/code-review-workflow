"""测试共用辅助：预埋缺陷样例的路径与读取、节点级用例的状态与围栏解析。

单一来源：节点状态构造与输出解析的口径一旦分叉（默认字段增删、围栏语义变化），
分散在多份测试里的副本会各自漂移，所以集中在这里供各测试文件 import。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: 预埋缺陷样例目录（README 快速开始与 Design §3 引用同一路径；ruff 已排除该目录）
BUGGY_SAMPLES = Path(__file__).parent / "fixtures" / "buggy_samples"


def sample_path(name: str) -> Path:
    """缺陷样例文件路径（如 ``syntax_error.py`` / ``divide_by_zero.py``）。"""
    return BUGGY_SAMPLES / name


def read_sample(name: str) -> str:
    """按 UTF-8 读取缺陷样例内容（``-f`` 输入的 CLI 侧行为）。"""
    return sample_path(name).read_text(encoding="utf-8")


def node_state(
    *,
    kind: str = "file",
    max_iterations: int = 3,
    version_dir: object = "",
    files: Mapping[str, object] | None = None,
    **overrides: object,
) -> dict[str, Any]:
    """构造节点可用的最小状态（只含节点读取的字段；版本 = 目录 + 文件清单）。"""
    base: dict[str, Any] = {
        "input": {"kind": kind, "requirement": "写一个计算器"},
        "current_version_dir": str(version_dir),
        "version_files": sorted(files or {}),
        "findings": [],
        "clean": False,
        "fix_rounds": 0,
        "review_round": 0,
        "max_iterations": max_iterations,
        "severe_only": False,
    }
    base.update(overrides)
    return base


def requirement_state(requirement: str) -> dict[str, Any]:
    """需求文本输入的节点状态。"""
    return node_state(kind="requirement", input={"kind": "requirement", "requirement": requirement})


def write_version(tmp_path: Path, files: Mapping[str, str], *, version: int = 1) -> str:
    """把一组文件写进版本目录（``tmp_path/code_vN/``），返回目录路径（节点读 current_version_dir）。"""
    directory = tmp_path / f"code_v{version}"
    for name, text in files.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return str(directory)


def fenced(text: str, label: str) -> str:
    """取文本中第一个指定标签围栏的内容（围栏 info 允许 ``file=…`` 等后缀）。

    找不到围栏时带上下文报错：裸 IndexError 无法区分"围栏缺失"与"标签写错"。
    """
    parts = text.split(f"```{label}", 1)
    assert len(parts) == 2, f"文本中没有 ```{label} 围栏（开头：{text[:60]!r}）"
    body = parts[1]
    body = body.split("\n", 1)[1] if "\n" in body else ""
    return body.split("```", 1)[0]


def review_payload(review_text: str) -> dict[str, Any]:
    """审查输出文本 → 契约载荷。"""
    return json.loads(fenced(review_text, "json"))


def fix_payload(fix_text: str) -> dict[str, Any]:
    """修复输出文本 → fixes 载荷。"""
    return json.loads(fenced(fix_text, "json"))


__all__ = [
    "BUGGY_SAMPLES",
    "fenced",
    "fix_payload",
    "node_state",
    "read_sample",
    "requirement_state",
    "review_payload",
    "sample_path",
    "write_version",
]
