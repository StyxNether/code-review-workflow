"""测试共用辅助：预埋缺陷样例的路径与读取（单一来源，避免路径口径分散）。"""

from __future__ import annotations

from pathlib import Path

#: 预埋缺陷样例目录（README 快速开始与 Design §3 引用同一路径；ruff 已排除该目录）
BUGGY_SAMPLES = Path(__file__).parent / "fixtures" / "buggy_samples"


def sample_path(name: str) -> Path:
    """缺陷样例文件路径（如 ``syntax_error.py`` / ``divide_by_zero.py``）。"""
    return BUGGY_SAMPLES / name


def read_sample(name: str) -> str:
    """按 UTF-8 读取缺陷样例内容（``-f`` 输入的 CLI 侧行为）。"""
    return sample_path(name).read_text(encoding="utf-8")
