"""会话层：编排/路由节点、会话态与按 Agent 的追加式记忆（Design §3、§4.7、§5.4）。

模块分工：

- ``state.py``：会话态 schema、会话号与目录、``session.json`` 的读写与列表；
- ``memory.py``：三 Agent 各自的追加式记忆（``memory/<agent>.jsonl``，只追加 + 阈值压缩）；
- ``orchestrator.py``：把用户消息解释为 ``{intent, rounds, rounds_mode, reply}``（§5.4 契约）
  并选择下一节点；每轮汇报从结构化状态渲染；
- ``chat.py``：会话循环——读入消息、驱动工作流、中断与退出语义、会话档案与会话总报告。

协作 Agent 始终只有编码 / 审查 / 修复三个（Design D12）：编排节点不产出代码与结论、
不计入 Agent 数，三 Agent 工作流保持纯净（不依赖会话即 ``crw run`` 可用）。
"""
