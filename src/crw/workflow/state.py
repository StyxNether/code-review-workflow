"""运行状态 schema 与快照的纯序列化 / 反序列化。

字段定义见 Design §4.3；本模块只做数据与序列化，文件 IO 归 records、时序归 engine（Design §3.1）。
"""
