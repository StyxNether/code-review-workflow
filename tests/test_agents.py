"""Agent 层：运行时（工具桥接 / 协议 / 护栏）与三节点装配（全假模型，无网络）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fakes
import pytest

from crw import agents as agents_mod
from crw import llm
from crw import rounds as rounds_mod
from crw import usage as usage_mod
from crw.tools import EXEC_TOOL_NAME, build_registry

SYSTEM_PROMPT = "你是审查员。"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """上下文根（模拟运行工作区）。"""
    root = tmp_path / "workspace"
    root.mkdir()
    return root


def make_runtime(model: Any, **kwargs: Any) -> agents_mod.AgentRuntime:
    """经装配入口构造运行时（与生产路径同一构造点，只是注入假模型）。"""
    return agents_mod.build_runtime(model=model, **kwargs)


def last_prompt(model: Any, index: int = -1) -> str:
    """第 ``index`` 次请求的 user 消息文本（首条为系统提示词，由框架注入）。"""
    return str(model.requests[index][-1].content)


def review_registry(workspace: Path) -> Any:
    """审查节点的工具集（与生产装配同源）。"""
    return build_registry([workspace], names=agents_mod.NODE_TOOL_NAMES["review"])


class TestRuntimeBasics:
    def test_returns_text_and_counts_successful_tools(self, workspace: Path) -> None:
        (workspace / "code_v1.py").write_text("print(1)\n", encoding="utf-8")
        model = fakes.ScriptedChatModel(
            responses=[
                fakes.ai_tool_call("read_file", path="code_v1.py"),
                fakes.ai_text("结论：没问题。"),
            ]
        )
        runtime = make_runtime(model)

        result = runtime.run(
            node="review",
            system_prompt=SYSTEM_PROMPT,
            user_prompt="审查 code_v1.py",
            registry=review_registry(workspace),
        )

        assert result.text == "结论：没问题。"
        assert result.tool_events == 1

    def test_emits_tool_events_with_the_node_name(self, workspace: Path) -> None:
        (workspace / "code_v1.py").write_text("print(1)\n", encoding="utf-8")
        events: list[dict[str, Any]] = []
        model = fakes.ScriptedChatModel(
            responses=[
                fakes.ai_tool_call("read_file", path="code_v1.py"),
                fakes.ai_text("结论。"),
            ]
        )
        runtime = make_runtime(model, emit=events.append)

        runtime.run(
            node="review",
            system_prompt=SYSTEM_PROMPT,
            user_prompt="审查",
            registry=review_registry(workspace),
        )

        tools = [event for event in events if event["event"] == "tool"]
        assert tools and tools[0]["node"] == "review"
        assert "read_file" in tools[0]["detail"] and "code_v1.py" in tools[0]["detail"]

    def test_forwards_text_deltas(self, workspace: Path) -> None:
        deltas: list[tuple[str, str]] = []
        model = fakes.ScriptedChatModel(responses=[fakes.ai_text("一。二。三。")])
        runtime = make_runtime(model, on_delta=lambda node, text: deltas.append((node, text)))

        runtime.run(
            node="fix",
            system_prompt=SYSTEM_PROMPT,
            user_prompt="修复",
            registry=review_registry(workspace),
        )

        assert "".join(text for _, text in deltas) == "一。二。三。"
        assert len(deltas) > 1  # 真的是逐片转发，而不是整段一次性回传
        assert {node for node, _ in deltas} == {"fix"}

    def test_each_run_is_independent(self, workspace: Path) -> None:
        """单任务语义：第二次 run 只带自己的 user 消息（不持有跨轮正本，Design §6.2 #5）。"""
        model = fakes.ScriptedChatModel(
            responses=[fakes.ai_text("第一次。"), fakes.ai_text("第二次。")]
        )
        runtime = make_runtime(model)

        first = runtime.run(
            node="review",
            system_prompt=SYSTEM_PROMPT,
            user_prompt="甲",
            registry=review_registry(workspace),
        )
        second = runtime.run(
            node="review",
            system_prompt=SYSTEM_PROMPT,
            user_prompt="乙",
            registry=review_registry(workspace),
        )

        assert (first.text, second.text) == ("第一次。", "第二次。")
        # 第二次请求 = 系统提示词 + 本轮 user 消息（不携带上一轮的对话）
        assert len(model.requests[1]) == 2
        assert "甲" not in str(model.requests[1][-1].content)

    def test_model_label_falls_back_to_the_class_name(self) -> None:
        runtime = make_runtime(fakes.ScriptedChatModel(responses=[]))

        assert runtime.model_label() == "ScriptedChatModel"

    def test_first_attempt_keeps_the_prompt_and_the_tools(self, workspace: Path) -> None:
        """首次尝试：提示词原样、工具照给。

        回归防线：曾经把"上一轮不可用 / 不要再调用工具"无条件拼在首次消息末尾，模型照做后
        放弃 read_file / run_python，凭清单重写整份代码（试用实测）。
        """
        model = fakes.ScriptedChatModel(responses=[fakes.ai_text("结论。")])
        runtime = make_runtime(model)

        runtime.run(
            node="fix",
            system_prompt=SYSTEM_PROMPT,
            user_prompt="修复这个问题",
            registry=review_registry(workspace),
        )

        prompt = last_prompt(model)
        assert "不要再调用工具" not in prompt
        assert "上一轮回答不可用" not in prompt
        assert prompt == "修复这个问题"
        assert model.bound_tools[0]  # 首次尝试带着工具

    def test_empty_answer_is_retried_once(self, workspace: Path) -> None:
        """空回答按协议层偶发故障处理：原样重发一次（实测多出现在护栏去工具后的收官调用）。"""
        events: list[dict[str, Any]] = []
        model = fakes.ScriptedChatModel(
            responses=[fakes.ai_text("   "), fakes.ai_text("重发后的结论。")]
        )
        runtime = make_runtime(model, emit=events.append)

        result = runtime.run(
            node="review",
            system_prompt=SYSTEM_PROMPT,
            user_prompt="审查",
            registry=review_registry(workspace),
        )

        assert result.text == "重发后的结论。"
        assert [event["event"] for event in events].count("retry") == 1
        assert [] in model.bound_tools  # 重发走无工具图
        assert "不要再调用工具" in last_prompt(model)

    def test_empty_answer_twice_is_a_readable_failure(self, workspace: Path) -> None:
        model = fakes.ScriptedChatModel(responses=[fakes.ai_text("   "), fakes.ai_text("")])
        runtime = make_runtime(model)

        with pytest.raises(llm.LLMError, match="空回答"):
            runtime.run(
                node="review",
                system_prompt=SYSTEM_PROMPT,
                user_prompt="审查",
                registry=review_registry(workspace),
            )


class TestErrorProtocol:
    def test_unknown_tool_returns_structured_error(self, workspace: Path) -> None:
        """未注册工具：``错误：`` 文本回传模型，不中断运行（Design §8.4）。"""
        model = fakes.ScriptedChatModel(
            responses=[fakes.ai_tool_call("deploy", target="prod"), fakes.ai_text("改用静态分析。")]
        )
        runtime = make_runtime(model)

        result = runtime.run(
            node="review",
            system_prompt=SYSTEM_PROMPT,
            user_prompt="审查",
            registry=review_registry(workspace),
        )

        assert result.text == "改用静态分析。"
        assert result.tool_events == 0  # 失败的工具不计入执行证据
        tool_messages = [message for message in model.requests[1] if message.type == "tool"]
        assert tool_messages and tool_messages[0].content.startswith("错误：未注册的工具")

    def test_escaping_path_is_rejected_as_a_tool_error(self, workspace: Path) -> None:
        model = fakes.ScriptedChatModel(
            responses=[
                fakes.ai_tool_call("read_file", path="../../../etc/passwd"),
                fakes.ai_text("无法读取，按静态分析给出结论。"),
            ]
        )
        runtime = make_runtime(model)

        result = runtime.run(
            node="review",
            system_prompt=SYSTEM_PROMPT,
            user_prompt="审查",
            registry=review_registry(workspace),
        )

        assert result.tool_events == 0
        tool_messages = [message for message in model.requests[1] if message.type == "tool"]
        assert tool_messages and "越出可用范围" in str(tool_messages[0].content)

    def test_sdk_failure_is_translated(self, workspace: Path) -> None:
        from types import SimpleNamespace

        from openai import APIStatusError

        response = SimpleNamespace(status_code=401, request=SimpleNamespace(), headers={})
        model = fakes.ScriptedChatModel(
            responses=[APIStatusError("unauthorized", response=response, body=None)]
        )
        runtime = make_runtime(model)

        with pytest.raises(llm.LLMError, match="认证失败"):
            runtime.run(
                node="review",
                system_prompt=SYSTEM_PROMPT,
                user_prompt="审查",
                registry=review_registry(workspace),
            )


class TestGuardrails:
    def test_tool_round_limit_drops_tools_and_records_a_notice(self, workspace: Path) -> None:
        """工具轮次上限：下一轮去工具 + 注入收尾指令，并把事实记进 notices（Design §4.6）。"""
        (workspace / "code_v1.py").write_text("print(1)\n", encoding="utf-8")
        model = fakes.ScriptedChatModel(
            responses=[
                fakes.ai_tool_call("read_file", path="code_v1.py"),
                fakes.ai_text("基于已读内容给出结论。"),
            ]
        )
        runtime = make_runtime(model, max_tool_rounds=1)

        result = runtime.run(
            node="review",
            system_prompt=SYSTEM_PROMPT,
            user_prompt="审查",
            registry=review_registry(workspace),
        )

        assert result.text == "基于已读内容给出结论。"
        assert any(not tools for tools in model.bound_tools)  # 末次调用没有工具可绑
        notices = runtime.drain_notices()
        assert any("工具调用已达上限" in notice for notice in notices)
        assert runtime.drain_notices() == []  # 取走即清空

    def test_context_budget_stops_tools_from_the_first_request(self, workspace: Path) -> None:
        """任务级预算护栏：估算超阈值即停止工具 + 收尾（Design §4.6）。"""
        model = fakes.ScriptedChatModel(responses=[fakes.ai_text("结论（可能不完整）。")])
        runtime = make_runtime(model, max_context_tokens=1)

        result = runtime.run(
            node="review",
            system_prompt=SYSTEM_PROMPT,
            user_prompt="审查" * 50,
            registry=review_registry(workspace),
        )

        assert result.text == "结论（可能不完整）。"
        assert model.bound_tools[0] == []  # 第一次请求就已去工具
        assert any("上下文预算" in notice for notice in runtime.drain_notices())

    def test_budget_setting_comes_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CRW_MAX_CONTEXT_TOKENS", "4321")

        runtime = make_runtime(fakes.ScriptedChatModel(responses=[]))

        assert runtime.max_context_tokens == 4321


class TestNodeAssembly:
    def test_encode_node_stages_the_model_text(self, workspace: Path) -> None:
        model = fakes.ScriptedChatModel(responses=[fakes.ai_text(fakes.file_fence("print(1)"))])
        nodes = agents_mod.build_nodes(make_runtime(model), roots=[workspace])

        output = nodes["encode"]({"input": {"kind": "requirement", "requirement": "写一个计算器"}})

        assert "```python" in output["code_text"]
        assert "写一个计算器" in str(last_prompt(model, 0))

    def test_review_node_reports_tool_evidence(self, workspace: Path) -> None:
        """有工具事件 → review_evidence 为真；一次工具都没调 → 为假（Design §5.2 门禁依据）。"""
        (workspace / "code_v1.py").write_text("print(1)\n", encoding="utf-8")
        state = {
            "current_code": str(workspace / "code_v1.py"),
            "review_round": 0,
            "fix_rounds": 0,
            "severe_only": False,
        }

        with_tools = fakes.ScriptedChatModel(
            responses=[
                fakes.ai_tool_call("read_file", path="code_v1.py"),
                fakes.ai_text(fakes.review_fence([], "读过了，没问题。")),
            ]
        )
        output = agents_mod.build_nodes(make_runtime(with_tools), roots=[workspace])["review"](
            state
        )
        assert output["review_evidence"] is True

        without_tools = fakes.ScriptedChatModel(
            responses=[fakes.ai_text(fakes.review_fence([], "没问题。"))]
        )
        output = agents_mod.build_nodes(make_runtime(without_tools), roots=[workspace])["review"](
            state
        )
        assert output["review_evidence"] is False

    def test_review_node_resets_evidence_per_round(self, workspace: Path) -> None:
        """第二轮（无 retry_feedback 的新调用）不继承上一轮的工具证据。"""
        (workspace / "code_v1.py").write_text("print(1)\n", encoding="utf-8")
        model = fakes.ScriptedChatModel(
            responses=[
                fakes.ai_tool_call("read_file", path="code_v1.py"),
                fakes.ai_text(fakes.review_fence([], "第一轮结论。")),
                fakes.ai_text(fakes.review_fence([], "第二轮结论。")),
            ]
        )
        review = agents_mod.build_nodes(make_runtime(model), roots=[workspace])["review"]
        state = {
            "current_code": str(workspace / "code_v1.py"),
            "review_round": 0,
            "fix_rounds": 0,
            "severe_only": False,
        }

        first = review(state)
        second = review({**state, "review_round": 1, "fix_rounds": 1})

        assert first["review_evidence"] is True
        assert second["review_evidence"] is False

    def test_retry_feedback_reaches_the_prompt(self, workspace: Path) -> None:
        model = fakes.ScriptedChatModel(
            responses=[fakes.ai_text(fakes.review_fence([], "重试后的结论。"))]
        )
        review = agents_mod.build_nodes(make_runtime(model), roots=[workspace])["review"]

        review(
            {
                "current_code": str(workspace / "code_v1.py"),
                "review_round": 1,
                "fix_rounds": 0,
                "severe_only": False,
            },
            retry_feedback="findings 必须为数组",
        )

        assert "findings 必须为数组" in str(last_prompt(model, 0))

    def test_fix_node_writes_only_the_changed_files(self, workspace: Path) -> None:
        """修复节点用 ``write_file`` 落盘：只写改动的文件，未改动的原样保留（Design §4.2.2）。

        上下文根 = 本轮产出的版本目录，所以模型给的路径不带 ``code_vN/`` 前缀；节点只交回
        fixes 围栏（产物内容不再走围栏），版本完整性由引擎按磁盘内容校验。
        """
        version = workspace / "code_v2"
        version.mkdir()
        (version / "main.py").write_text("print(1)\n", encoding="utf-8")
        (version / "helper.py").write_text("X = 1\n", encoding="utf-8")
        model = fakes.ScriptedChatModel(
            responses=[
                fakes.ai_tool_call("write_file", path="main.py", content="print(2)\n"),
                fakes.ai_text(
                    fakes.fix_contract(
                        [{"finding_id": "f-1", "status": "fixed", "note": "已改为打印 2。"}]
                    )
                ),
            ]
        )
        fix = agents_mod.build_nodes(make_runtime(model), roots=[workspace])["fix"]

        output = fix(
            {
                "current_version_dir": str(version),
                "version_files": ["helper.py", "main.py"],
                "findings": [
                    {"id": "f-1", "severity": "严重", "location": "main.py:1", "issue": "输出错了"}
                ],
                "fix_rounds": 0,
            }
        )

        assert (version / "main.py").read_text(encoding="utf-8") == "print(2)\n"
        assert (version / "helper.py").read_text(encoding="utf-8") == "X = 1\n"
        assert "```json" in output["fix_text"] and "f-1" in output["fix_text"]
        # 写入根是版本目录：模型无需再"整份重贴"未改动的文件
        assert "不要重写" in str(model.requests[0][0].content)
        assert "write_file" in model.bound_tools[0]

    def test_fix_node_without_a_target_dir_fails_readably(self, workspace: Path) -> None:
        """缺目标版本目录属装配缺陷：显式失败，不在一个不存在的根上"假装修复"。"""
        model = fakes.ScriptedChatModel(responses=[])
        fix = agents_mod.build_nodes(make_runtime(model), roots=[workspace])["fix"]

        with pytest.raises(ValueError, match="缺少目标版本目录"):
            fix({"current_version_dir": "", "version_files": [], "findings": []})

    def test_no_exec_is_visible_to_the_nodes(self, workspace: Path) -> None:
        """``--no-exec`` 时节点看到的是不含 run_python 的工具集（提示词口径随之切换）。"""
        model = fakes.ScriptedChatModel(responses=[fakes.ai_text(fakes.review_fence([], "好。"))])
        nodes = agents_mod.build_nodes(make_runtime(model), roots=[workspace], no_exec=True)

        nodes["review"](
            {
                "current_code": str(workspace / "code_v1.py"),
                "review_round": 1,
                "fix_rounds": 0,
                "severe_only": False,
            }
        )

        bound = [name for tools in model.bound_tools for name in tools]
        assert "run_python" not in bound  # 执行工具整体消失
        assert "read_file" in bound  # 读文件仍可用（只做静态审查）
        assert "已关闭代码执行" in str(last_prompt(model, 0))

    def test_exec_tool_is_registered_by_default(self, workspace: Path) -> None:
        model = fakes.ScriptedChatModel(responses=[fakes.ai_text(fakes.review_fence([], "好。"))])
        nodes = agents_mod.build_nodes(make_runtime(model), roots=[workspace])

        nodes["review"](
            {
                "current_code": str(workspace / "code_v1.py"),
                "review_round": 1,
                "fix_rounds": 0,
                "severe_only": False,
            }
        )

        assert EXEC_TOOL_NAME in model.bound_tools[0]


class TestStandaloneReview:
    def test_inline_prompt_carries_numbered_content(self, workspace: Path) -> None:
        model = fakes.ScriptedChatModel(responses=[fakes.ai_text("结论。")])
        runtime, registry = agents_mod.build_review_runtime(model=model, roots=[workspace])

        text = agents_mod.review_once(
            runtime,
            registry,
            display_name="sample.py",
            content="a = 1\nb = 2\n",
            as_json=False,
            max_lines=100,
        )

        assert text == "结论。"
        prompt = str(last_prompt(model, 0))
        assert "sample.py" in prompt
        assert "   1 | a = 1" in prompt
        assert "【严重|一般|建议】sample.py:行号" in prompt  # 文本模式：人类可读格式

    def test_json_mode_asks_for_the_files_shape(self, workspace: Path) -> None:
        model = fakes.ScriptedChatModel(responses=[fakes.ai_text("{}")])
        runtime, registry = agents_mod.build_review_runtime(model=model, roots=[workspace])

        agents_mod.review_once(
            runtime,
            registry,
            display_name="sample.py",
            content="print(1)\n",
            as_json=True,
            max_lines=100,
        )

        assert '"files"' in str(last_prompt(model, 0))

    def test_long_file_is_truncated_and_flagged(self, workspace: Path) -> None:
        model = fakes.ScriptedChatModel(responses=[fakes.ai_text("结论。")])
        runtime, registry = agents_mod.build_review_runtime(model=model, roots=[workspace])
        content = "".join(f"line {index}\n" for index in range(50))

        agents_mod.review_once(
            runtime,
            registry,
            display_name="big.py",
            content=content,
            as_json=False,
            max_lines=10,
        )

        prompt = str(last_prompt(model, 0))
        assert "仅含前 10 行" in prompt
        # 精确锁定截断边界：前 10 行（line 0..line 9）在，第 11 行（"line 10"）不在
        assert "line 9" in prompt
        assert "line 10" not in prompt

    def test_review_graph_is_reused_across_files(self, workspace: Path) -> None:
        """逐文件审查复用同一图（避免重复编译）：两次请求都来自同一节点图。"""
        model = fakes.ScriptedChatModel(responses=[fakes.ai_text("一。"), fakes.ai_text("二。")])
        runtime, registry = agents_mod.build_review_runtime(model=model, roots=[workspace])

        for name in ("a.py", "b.py"):
            agents_mod.review_once(
                runtime,
                registry,
                display_name=name,
                content="print(1)\n",
                as_json=False,
                max_lines=100,
            )

        assert len(model.requests) == 2

    def test_review_runtime_defaults_to_the_shared_round_budget(self, workspace: Path) -> None:
        """缺省工具轮次预算仍是运行时的 10（本层不为单一命令改默认）。"""
        model = fakes.ScriptedChatModel(responses=[])
        runtime, _ = agents_mod.build_review_runtime(model=model, roots=[workspace])

        assert runtime.max_tool_rounds == agents_mod.MAX_TOOL_ROUNDS == 10

    def test_review_runtime_can_raise_its_round_budget(self, workspace: Path) -> None:
        """整项目形态的高预算经装配入口透传（一次会话要顺链读多个文件，10 轮不够）。"""
        model = fakes.ScriptedChatModel(responses=[])
        runtime, _ = agents_mod.build_review_runtime(
            model=model, roots=[workspace], max_tool_rounds=30
        )

        assert runtime.max_tool_rounds == 30

    def test_project_prompt_lists_files_without_their_content(self, workspace: Path) -> None:
        """整项目任务只给清单（路径 + 行数），正文留给审查者自己读。"""
        model = fakes.ScriptedChatModel(responses=[fakes.ai_text("跨文件结论。")])
        runtime, registry = agents_mod.build_review_runtime(model=model, roots=[workspace])

        text = agents_mod.review_project_once(
            runtime,
            registry,
            listing=(("main.py", 12), ("pkg/util.py", 5)),
            as_json=False,
            allow_exec=True,
            total_lines=17,
        )

        assert text == "跨文件结论。"
        prompt = str(last_prompt(model, 0))
        assert "main.py（12 行）" in prompt and "pkg/util.py（5 行）" in prompt
        assert "顺着 import / 调用链" in prompt
        assert "必查项" in prompt
        assert "没读完就如实说明覆盖了清单里的哪些文件" in prompt
        assert "run_python" in prompt  # allow_exec=True：给出核实手段

    def test_project_prompt_reports_omitted_files_and_static_mode(self, workspace: Path) -> None:
        """闸门外的文件必须出现在任务里；``--no-exec`` 时改为静态审查口径。"""
        model = fakes.ScriptedChatModel(responses=[fakes.ai_text("结论。")])
        runtime, registry = agents_mod.build_review_runtime(model=model, roots=[workspace])

        agents_mod.review_project_once(
            runtime,
            registry,
            listing=(("main.py", 3),),
            as_json=True,
            allow_exec=False,
            omitted=("huge.py",),
            total_lines=3,
        )

        prompt = str(last_prompt(model, 0))
        assert "未列入清单" in prompt and "huge.py" in prompt
        assert "--no-exec" in prompt
        assert '"files"' in prompt  # JSON 契约形态


def test_build_nodes_covers_every_graph_node(workspace: Path) -> None:
    from crw.workflow import graph as graph_mod

    nodes = agents_mod.build_nodes(
        make_runtime(fakes.ScriptedChatModel(responses=[])), roots=[workspace]
    )

    assert set(nodes) == set(graph_mod.NODE_NAMES)


def test_state_helpers_used_by_the_nodes() -> None:
    """节点读的轮次 / 版本口径与状态层同源（避免两处各算一份）。"""
    state = {"review_round": 2, "fix_rounds": 1}

    assert rounds_mod.code_version_dir_name(rounds_mod.code_version(state)) == "code_v2"
    assert rounds_mod.round_number_for_review(state) == 3


class TestToolMarkupGuard:
    """模型把工具调用标记当正文返回时的兜底（实测 DeepSeek 系偶发）。"""

    MARKUP = "<｜｜DSML｜｜ calls>\n<｜｜DSML｜｜ invoke name='read_file'>"

    def test_markup_answer_is_retried_once(self, workspace: Path) -> None:
        events: list[dict[str, Any]] = []
        model = fakes.ScriptedChatModel(
            responses=[fakes.ai_text(self.MARKUP), fakes.ai_text("重发后的结论。")]
        )
        runtime = make_runtime(model, emit=events.append)

        result = runtime.run(
            node="review",
            system_prompt=SYSTEM_PROMPT,
            user_prompt="审查",
            registry=review_registry(workspace),
        )

        assert result.text == "重发后的结论。"
        assert any(event["event"] == "retry" for event in events)  # 重发留痕（Design §8.2）
        assert len(model.requests) == 2
        # 重发改为"无工具 + 直接要结论"：多轮探索是这类故障的温床，原样重发很可能再撞一次
        assert [] in model.bound_tools
        assert "不要再调用工具" in last_prompt(model)

    def test_retry_keeps_tools_when_the_lost_round_had_read_content(self, workspace: Path) -> None:
        """证据只在工具结果里时，重发必须带着工具（整项目审查形态的实测缺陷）。

        不带工具的重发会让模型在"什么都没读过"的上下文里收尾——它如实回答"未能读取任何
        文件"，前一轮读进来的内容全部白花，整批结论不可用。
        """
        (workspace / "code_v1.py").write_text("print(1)\n", encoding="utf-8")
        events: list[dict[str, Any]] = []
        model = fakes.ScriptedChatModel(
            responses=[
                fakes.ai_tool_call("read_file", path="code_v1.py"),
                fakes.ai_text(self.MARKUP),
                fakes.ai_text("重发后基于已读内容给出结论。"),
            ]
        )
        runtime = make_runtime(model, emit=events.append)

        result = runtime.run(
            node="review",
            system_prompt=SYSTEM_PROMPT,
            user_prompt="审查",
            registry=review_registry(workspace),
        )

        assert result.text == "重发后基于已读内容给出结论。"
        assert len(model.requests) == 3  # 工具调用 → 不可用回答 → 带工具重发
        assert [] not in model.bound_tools  # 重发没有走"去工具"变体
        assert model.bound_tools and model.bound_tools[0]
        assert "请按需重新读取要审查的内容" in last_prompt(model)
        assert "不要再调用工具" not in last_prompt(model)

    def test_markup_twice_fails_readably(self, workspace: Path) -> None:
        model = fakes.ScriptedChatModel(
            responses=[fakes.ai_text(self.MARKUP), fakes.ai_text("<tool_calls></tool_calls>")]
        )
        runtime = make_runtime(model)

        with pytest.raises(llm.LLMError, match="未按工具协议返回"):
            runtime.run(
                node="review",
                system_prompt=SYSTEM_PROMPT,
                user_prompt="审查",
                registry=review_registry(workspace),
            )


def test_ordinary_text_is_not_mistaken_for_markup(workspace: Path) -> None:
    model = fakes.ScriptedChatModel(responses=[fakes.ai_text("结论：没有发现问题。")])
    runtime = make_runtime(model)

    result = runtime.run(
        node="review",
        system_prompt=SYSTEM_PROMPT,
        user_prompt="审查",
        registry=review_registry(workspace),
    )

    assert result.text == "结论：没有发现问题。"


def test_task_prompt_json_examples_are_parseable() -> None:
    """提示词里给模型看的 JSON 示例必须自身合法：示例错了会诱导模型产出非法契约。

    （真实缺陷回归：审查任务提示词的示例曾多一个右花括号，导致合法输出被判违约。）
    """
    import json
    import re

    from crw.agents import prompts

    review_prompt = prompts.review_task_prompt(
        version_label="code_v1",
        files=["main.py", "util.py"],
        review_round=1,
        code_version=1,
        allow_exec=True,
    )
    examples = re.findall(r"\{.*\}", review_prompt, flags=re.DOTALL)
    assert examples
    json.loads(examples[0])

    inline = prompts.inline_review_prompt(
        display_name="a.py", content="print(1)\n", as_json=True, max_lines=10
    )
    inline_examples = re.findall(r"\{.*\}", inline, flags=re.DOTALL)
    assert inline_examples
    json.loads(inline_examples[0])


def test_unknown_review_view_is_rejected() -> None:
    """拼错的视角显式报错：静默退回基座提示词会让两个审查者都失去职责边界且无人发现。"""
    from crw.agents import prompts as prompts_mod

    with pytest.raises(ValueError, match="未知的审查视角"):
        prompts_mod.reviewer_system_prompt("correctnes")


def test_guardrail_notices_are_cleared_across_runs(workspace: Path) -> None:
    """护栏触达记录按任务清空（含重发变体的 guardrail）：上一任务的触达不残留到下一任务。"""
    model = fakes.ScriptedChatModel(responses=[fakes.ai_text("一。"), fakes.ai_text("二。")])
    runtime = make_runtime(model, max_context_tokens=1)

    runtime.run(
        node="review",
        system_prompt=SYSTEM_PROMPT,
        user_prompt="任务一",
        registry=review_registry(workspace),
    )
    first = runtime.drain_notices()
    runtime.run(
        node="review",
        system_prompt=SYSTEM_PROMPT,
        user_prompt="任务二",
        registry=review_registry(workspace),
    )
    second = runtime.drain_notices()

    assert first and second  # 预算=1，两次都触达
    assert len(second) == len(first)  # 不累积：第二次只含本任务的记录


def test_parallel_tool_calls_in_one_message_are_counted(workspace: Path) -> None:
    """同一条消息里的多个并行工具调用按真实 index 回放：两个都要被计数（fakes 的 index 回归）。"""
    from langchain_core.messages import AIMessage

    (workspace / "a.py").write_text("print(1)\n", encoding="utf-8")
    (workspace / "b.py").write_text("print(2)\n", encoding="utf-8")
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "read_file", "args": {"path": "a.py"}, "id": "call-a"},
            {"name": "read_file", "args": {"path": "b.py"}, "id": "call-b"},
        ],
    )
    model = fakes.ScriptedChatModel(responses=[message, fakes.ai_text("结论。")])
    runtime = make_runtime(model)

    result = runtime.run(
        node="review",
        system_prompt=SYSTEM_PROMPT,
        user_prompt="审查",
        registry=review_registry(workspace),
    )

    assert result.tool_events == 2


def test_fix_task_prompt_scopes_paths_to_the_version_dir() -> None:
    """修复任务的上下文根就是本轮产出的版本目录：清单与 write_file 都用不带前缀的相对路径。

    旧形态在这里踩过真实的坑——清单带 ``code_v1/`` 前缀而产出要求不带前缀，模型照清单写
    带前缀的路径、覆盖检查判缺全体文件；根换成版本目录后两种口径合一。
    """
    from crw.agents import prompts

    prompt = prompts.fix_task_prompt(
        version_label="code_v1",
        out_label="code_v2",
        files=["main.py", "todos.py"],
        findings=[fakes.finding(location="main.py:1")],
        allow_exec=True,
    )

    assert "- main.py" in prompt and "- todos.py" in prompt
    assert "code_v1/ 已复制为 code_v2/" in prompt  # 来源与产出去向都写明
    assert "本轮的版本目录 code_v2/（上下文根" in prompt
    assert "只把你改动的文件写回去" in prompt
    assert "未改动的文件不要写" in prompt


def test_drain_notices_for_only_takes_the_named_node(workspace: Path) -> None:
    """按节点取护栏记录：并行分支各自只回收自己的，不误取仍在运行的另一个分支。"""
    model = fakes.ScriptedChatModel(responses=[fakes.ai_text("一。"), fakes.ai_text("二。")])
    runtime = make_runtime(model, max_context_tokens=1)
    registry = review_registry(workspace)

    runtime.run(
        node="review_correctness",
        system_prompt=SYSTEM_PROMPT,
        user_prompt="任务一",
        registry=registry,
    )

    # 只按节点过滤：先要别的节点拿不到、再拿自己的，取走后清空
    assert runtime.drain_notices_for("review_security") == []
    notices = runtime.drain_notices_for("review_correctness")
    assert notices
    assert runtime.drain_notices_for("review_correctness") == []


# ---- 用量采集（每次模型调用的输入 / 缓存命中 / 输出 token）----


def test_each_model_call_is_recorded_and_emitted(workspace: Path) -> None:
    """一次任务里的每次调用都记一笔，并即时发出 usage 事件（与工具事件同一条时间线）。"""
    (workspace / "code_v1.py").write_text("print(1)\n", encoding="utf-8")
    events: list[dict[str, Any]] = []
    model = fakes.ScriptedChatModel(
        responses=[
            fakes.ai_tool_call(
                "read_file",
                path="code_v1.py",
                usage=fakes.call_usage(input_tokens=1000, output_tokens=40, cached_tokens=512),
            ),
            fakes.ai_text(
                "结论：没问题。",
                usage=fakes.call_usage(input_tokens=1200, output_tokens=60, cached_tokens=1024),
            ),
        ]
    )
    runtime = make_runtime(model, emit=events.append)

    runtime.run(
        node="review",
        system_prompt=SYSTEM_PROMPT,
        user_prompt="审查 code_v1.py",
        registry=review_registry(workspace),
    )

    usage_events = [event for event in events if event["event"] == "usage"]
    assert [event["detail"] for event in usage_events] == [
        {"input_tokens": 1000, "output_tokens": 40, "reasoning_tokens": 0, "cached_tokens": 512},
        {"input_tokens": 1200, "output_tokens": 60, "reasoning_tokens": 0, "cached_tokens": 1024},
    ]
    assert all(event["node"] == "review" for event in usage_events)

    record = runtime.drain_usage_for("review")
    assert record["calls"] == 2
    assert record["input_tokens"] == 2200
    assert record["cached_tokens"] == 1536
    assert record["cache_reported"] == 2


def test_model_without_usage_yields_call_count_only(workspace: Path) -> None:
    """假模型不带 usage（供应商未回报同理）：只记调用次数，不崩也不虚增 token。"""
    model = fakes.ScriptedChatModel(responses=[fakes.ai_text("结论。")])
    events: list[dict[str, Any]] = []
    runtime = make_runtime(model, emit=events.append)

    runtime.run(
        node="fix",
        system_prompt=SYSTEM_PROMPT,
        user_prompt="修复",
        registry=review_registry(workspace),
    )

    (event,) = [item for item in events if item["event"] == "usage"]
    assert event["detail"] == {}  # 未回报：日志里如实留一条空 detail
    record = runtime.drain_usage_for("fix")
    assert record["calls"] == 1
    assert record["with_usage"] == 0
    assert record["input_tokens"] == 0


def test_history_messages_are_not_billed_to_this_task(workspace: Path) -> None:
    """会话记忆的历史消息不带本次调用的用量：不得被计进本轮账。"""
    model = fakes.ScriptedChatModel(
        responses=[
            fakes.ai_text(
                "结论。",
                usage=fakes.call_usage(input_tokens=50, output_tokens=10, cached_tokens=20),
            )
        ]
    )
    runtime = make_runtime(model)

    runtime.run(
        node="review",
        system_prompt=SYSTEM_PROMPT,
        user_prompt="审查",
        registry=review_registry(workspace),
        history=[
            {"role": "user", "text": "上一轮任务"},
            {"role": "assistant", "text": "上一轮结论"},
        ],
    )

    record = runtime.drain_usage_for("review")
    assert record["calls"] == 1
    assert record["input_tokens"] == 50


def test_usage_is_keyed_per_node_and_draining_clears(workspace: Path) -> None:
    """按节点归账：取走自己的不碰别的节点，取走后清空；drain_usage 一次取全部。"""
    model = fakes.ScriptedChatModel(
        responses=[
            fakes.ai_text("编码产出。", usage=fakes.call_usage(input_tokens=7, output_tokens=3)),
            fakes.ai_text("审查结论。", usage=fakes.call_usage(input_tokens=11, output_tokens=5)),
        ]
    )
    runtime = make_runtime(model)
    registry = review_registry(workspace)

    runtime.run(node="encode", system_prompt=SYSTEM_PROMPT, user_prompt="写", registry=registry)
    runtime.run(node="review", system_prompt=SYSTEM_PROMPT, user_prompt="审", registry=registry)

    assert runtime.drain_usage_for("review")["input_tokens"] == 11
    assert runtime.drain_usage_for("review") == usage_mod.empty()
    everything = runtime.drain_usage()
    assert set(everything) == {"encode"}
    assert everything["encode"]["output_tokens"] == 3
    assert runtime.drain_usage() == {}


def test_retry_attempt_bills_both_attempts(workspace: Path) -> None:
    """收官回答不可用而重发：两次的调用与 token 都记账（钱确实花了两遍）。"""
    model = fakes.ScriptedChatModel(
        responses=[
            fakes.ai_text("   ", usage=fakes.call_usage(input_tokens=30, output_tokens=2)),
            fakes.ai_text(
                "结论。", usage=fakes.call_usage(input_tokens=40, output_tokens=8, cached_tokens=30)
            ),
        ]
    )
    runtime = make_runtime(model)

    runtime.run(
        node="review",
        system_prompt=SYSTEM_PROMPT,
        user_prompt="审",
        registry=review_registry(workspace),
    )

    record = runtime.drain_usage_for("review")
    assert record["calls"] == 2
    assert record["input_tokens"] == 70
    assert record["with_usage"] == 2


def test_shrunk_message_list_is_not_rebilled(workspace: Path) -> None:
    """消息列表被裁短（本栈不会发生，属版本漂移防御）：游标不回落，漏记而不重复记账。"""
    runtime = make_runtime(fakes.ScriptedChatModel(responses=[]))
    used = fakes.call_usage(input_tokens=10, output_tokens=1)
    first = [fakes.ai_text("一", usage=used), fakes.ai_text("二", usage=used)]

    assert runtime._record_usage("review", first, 0) == 2
    grown_but_shorter = [first[0], fakes.ai_text("三", usage=used)]
    assert runtime._record_usage("review", grown_but_shorter, 2) == 2  # 裁短：不回落

    with_new = [*grown_but_shorter, fakes.ai_text("四", usage=used)]
    assert runtime._record_usage("review", with_new, 2) == 3

    record = runtime.drain_usage_for("review")
    assert record["calls"] == 3  # 一、二、四入账；"三"被漏记——少报花费好于多报花费


def test_reviewer_prompt_constrains_verification_scripts_and_monologue() -> None:
    """审查提示词里的降本约束：核实脚本小而准、围栏外最多两三句（防过程独白膨胀输出）。"""
    from crw.agents import prompts

    text = prompts.REVIEWER_SYSTEM_PROMPT
    assert "核实脚本要**小而准**" in text
    assert "不要把待审代码整份抄进" in text
    assert "围栏之外**最多两三句**" in text
    assert "我先读取文件" in text  # 明确点出被禁止的过程独白形态
    task = prompts.review_task_prompt(
        version_label="code_v1",
        files=["main.py"],
        review_round=1,
        code_version=1,
        allow_exec=True,
    )
    assert "最小输入与断言" in task


def test_fixer_prompt_limits_repeated_inline_self_verification() -> None:
    """修复提示词：只写改动文件、自验一次就够（重复贴版本与反复重跑都按输出计费）。"""
    from crw.agents import prompts

    assert "只自验一次、只验改动涉及的部分" in prompts.FIXER_SYSTEM_PROMPT
    assert "只写你实际改动的文件" in prompts.FIXER_SYSTEM_PROMPT
    assert "不要输出代码围栏" in prompts.FIXER_SYSTEM_PROMPT
    task = prompts.fix_task_prompt(
        version_label="code_v1",
        out_label="code_v2",
        files=["main.py"],
        findings=[fakes.finding()],
        allow_exec=True,
    )
    assert "一次就够，不要逐文件反复重跑" in task
