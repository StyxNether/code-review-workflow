"""轮次与代码版本口径的纯数据叶子层：取值枚举 + 从状态映射派生数字。

**为什么单独一层**：``rounds_mode`` 的取值、代码版本目录名与审查 / 修复序号的派生规则，
workflow / agents / session / cli 四层都要用，而 agents 是下层、不得向上依赖 workflow
——把这部分**纯派生计算**放在叶子模块，agents 因此只依赖叶子层。本模块自己不依赖任何
crw 模块、不含框架类型，入参一律是 ``Mapping[str, Any]``。

**与配置项 ``rounds`` 是两回事**：`~/.crw/settings.json` 里的 ``rounds`` 是用户配置（修复轮次
上限的来源）；本模块处理的是**状态里已经定下来的口径**（轮次语义取值、版本号与
序号的算法），不读配置、也不决定用哪个取值。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

# ---- 取值枚举 ----

ROUNDS_UP_TO = "up-to"
ROUNDS_EXACT = "exact"

#: 版本目录名的语法：``code_v`` + ASCII 十进制数字，无前后缀（判定与生成同源）
_VERSION_DIR_NAME_RE = re.compile(r"code_v([0-9]+)")


def code_version_dir_name(version: int) -> str:
    """代码版本目录名：``code_v<version>``（一个版本 = 一个目录 + 目录内的 manifest.json）。"""
    return f"code_v{version}"


def version_of_dir_name(name: str) -> int | None:
    """目录名 → 版本号；不是版本目录名时返回 ``None``。

    版本目录名**文法**的唯一判定处：写侧生成、读侧扫描、渲染与提示词都经它判断，
    避免同一份命名规则在多处各写一遍正则而慢慢分叉（路径→目录名的切分由调用方负责）。
    """
    match = _VERSION_DIR_NAME_RE.fullmatch(name)
    return int(match.group(1)) if match else None


def is_code_version_dir_name(name: str) -> bool:
    """该目录名是否代码版本目录（与 :func:`code_version_dir_name` 同一份命名规则）。"""
    return version_of_dir_name(name) is not None


def round_number_for_review(state: Mapping[str, Any]) -> int:
    """本轮审查序号（= 已完成审查数 + 1，1-based）。"""
    return int(state.get("review_round", 0)) + 1


def round_number_for_fix(state: Mapping[str, Any]) -> int:
    """本轮修复序号（= 已完成修复数 + 1，1-based）。"""
    return int(state.get("fix_rounds", 0)) + 1


def normalize_version_base(value: Any) -> int:
    """起始版本号归一：非法值（缺失 / 非整数 / 小于 1）一律按 1 处理。

    请求侧与状态侧共用这一处规则：坏值只降级成"从第 1 版起"，
    不让一个手改过的字段把版本目录名或报告渲染弄崩。
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return 1
    return value


def code_version(state: Mapping[str, Any]) -> int:
    """当前代码版本号 = 起始版本号 + 已完成修复轮数。

    一次性运行的起始版本号恒为 1（即 ``fix_rounds + 1``）；会话轮次由命令面
    给出跨轮连续递增的起始版本号，报告里的"第几版"因此在会话内始终指向同一份代码。
    """
    try:
        fix_rounds = int(state.get("fix_rounds", 0))
    except (TypeError, ValueError):
        fix_rounds = 0
    return normalize_version_base(state.get("version_base")) + fix_rounds
