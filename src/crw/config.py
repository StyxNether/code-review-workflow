"""供应商配置：``~/.crw/models.json`` 读写、内置预设、环境变量回退与连通性自检。

数据目录经 ``CRW_HOME`` 解析（默认 ``~/.crw``）；key 回退链 CRW_ → CRA_ → SE_CodeAgent，
解析优先级为命令行覆盖 > 配置文件 > 环境变量（Design §8.3）。
"""

from __future__ import annotations

import os
from pathlib import Path

#: 数据目录环境变量：测试与多环境隔离用（Design §9.1）
HOME_ENV = "CRW_HOME"

#: 默认数据目录名（用户主目录下）
DEFAULT_HOME_NAME = ".crw"


def data_dir() -> Path:
    """数据目录根：``CRW_HOME`` 覆盖，默认 ``~/.crw``（Design §8.3、§9.1）。

    配置（models.json / settings.json）与运行记录同源解析：运行记录经
    ``records.runs_root()`` 取 ``data_dir()/runs``，测试由 conftest 统一指向临时目录。
    """
    override = os.environ.get(HOME_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / DEFAULT_HOME_NAME
