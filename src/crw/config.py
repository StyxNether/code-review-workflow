"""供应商配置：``~/.crw/models.json`` 读写、内置预设、环境变量回退与连通性自检。

数据目录经 ``CRW_HOME`` 解析（默认 ``~/.crw``）；key 回退链 CRW_ → CRA_ → SE_CodeAgent，
解析优先级为命令行覆盖 > 配置文件 > 环境变量（Design §8.3）。
"""
