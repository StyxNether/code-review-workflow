"""编排 / 路由节点（``session/orchestrator.py``）用例：意图契约、校验夹取、反问兜底与汇报渲染。

假模型驱动真实运行时（禁真实网络），脚本按"编排节点 → 三 Agent"的调用顺序回放。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fakes import ScriptedChatModel, ai_text

from crw import agents as agents_mod
from crw import rounds as rounds_mod
from crw import tools as tools_mod
from crw.session import orchestrator as orch
from crw.workflow import state as state_mod


def _orchestrator_reply(payload: Any, *, fenced: bool = True, prose: str = "") -> Any:
    """编排节点的响应：一个 ```json 围栏（或裸 JSON / 非 JSON 文本）。"""
    if isinstance(payload, str):
        body = payload
    else:
        body = json.dumps(payload, ensure_ascii=False)
    text = f"{prose}\n```json\n{body}\n```\n" if fenced else body
    return ai_text(text)


def _runtime() -> Any:
    return agents_mod.build_runtime(emit=lambda event: None, on_delta=lambda node, text: None)


def _registry(tmp_path: Path) -> Any:
    return tools_mod.build_registry([tmp_path], names=(), allow_exec=False)


def _interpret(
    model: ScriptedChatModel,
    tmp_path: Path,
    *,
    message: str = "写一个计算器",
    rounds_limit: int = 10,
    rounds_default: int = 3,
    context: dict[str, Any] | None = None,
) -> orch.Decision:
    return orch.interpret(
        _runtime(),
        _registry(tmp_path),
        message=message,
        context=context or {"rounds": [], "current_version": 0, "version_files": []},
        rounds_limit=rounds_limit,
        rounds_default=rounds_default,
    )


def test_valid_payload_becomes_a_decision(
    fake_chat_model: ScriptedChatModel, tmp_path: Path
) -> None:
    """合法输出：意图 / 轮次 / 模式 / reply 原样采用（未知字段留档，不改写）。"""
    fake_chat_model.load(
        [
            _orchestrator_reply(
                {
                    "intent": "review_fix",
                    "rounds": 2,
                    "rounds_mode": "exact",
                    "reply": "好的，再检查两轮。",
                    "extra_field": {"keep": True},
                }
            )
        ]
    )

    decision = _interpret(fake_chat_model, tmp_path)

    assert decision.intent == "review_fix"
    assert decision.rounds == 2
    assert decision.rounds_mode == rounds_mod.ROUNDS_EXACT
    assert decision.reply == "好的，再检查两轮。"
    assert decision.note == ""
    assert decision.raw["extra_field"] == {"keep": True}
    assert decision.starts_workflow is True


def test_ask_decision_does_not_start_the_workflow(
    fake_chat_model: ScriptedChatModel, tmp_path: Path
) -> None:
    fake_chat_model.load([_orchestrator_reply({"intent": "ask", "reply": "你想改哪一部分？"})])

    decision = _interpret(fake_chat_model, tmp_path, message="帮我看看")

    assert decision.intent == orch.INTENT_ASK
    assert decision.reply == "你想改哪一部分？"
    assert decision.starts_workflow is False


def test_rounds_are_clamped_to_the_limit_with_a_note(
    fake_chat_model: ScriptedChatModel, tmp_path: Path
) -> None:
    """消息提取值越界**夹取而非报错**（意图理解 ≠ 用户显式参数，Design §4.4）。"""
    fake_chat_model.load([_orchestrator_reply({"intent": "code", "rounds": 99, "reply": "好"})])

    decision = _interpret(fake_chat_model, tmp_path, rounds_limit=5)

    assert decision.rounds == 5
    assert "夹取" in decision.note


def test_negative_rounds_are_lifted_to_zero(
    fake_chat_model: ScriptedChatModel, tmp_path: Path
) -> None:
    fake_chat_model.load([_orchestrator_reply({"intent": "code", "rounds": -3, "reply": "好"})])

    decision = _interpret(fake_chat_model, tmp_path)

    assert decision.rounds == 0
    assert "下限" in decision.note


def test_non_integer_rounds_and_mode_fall_back_to_unspecified(
    fake_chat_model: ScriptedChatModel, tmp_path: Path
) -> None:
    """非整数的 rounds 与非枚举的 rounds_mode 按"未指定"处理（由调用方走优先级链）。"""
    fake_chat_model.load(
        [
            _orchestrator_reply(
                {"intent": "code", "rounds": "三轮", "rounds_mode": "turbo", "reply": "好"}
            )
        ]
    )

    decision = _interpret(fake_chat_model, tmp_path)

    assert decision.rounds is None
    assert decision.rounds_mode is None


def test_ask_without_reply_is_repaired_by_retry(
    fake_chat_model: ScriptedChatModel, tmp_path: Path
) -> None:
    """``ask`` 必须给出 reply；首次缺失 → 携原因重试一次（重试请求里带上错误原因）。"""
    fake_chat_model.load(
        [
            _orchestrator_reply({"intent": "ask"}),
            _orchestrator_reply({"intent": "ask", "reply": "请说明是要改哪一段。"}),
        ]
    )

    decision = _interpret(fake_chat_model, tmp_path)

    assert decision.reply == "请说明是要改哪一段。"
    retry_prompt = "\n".join(str(item.content) for item in fake_chat_model.requests[-1])
    assert "ask 必须给出非空的 reply" in retry_prompt


def test_unparseable_output_falls_back_to_a_clarifying_ask(
    fake_chat_model: ScriptedChatModel, tmp_path: Path
) -> None:
    """分不清意图一律按 ask 反问（不静默挑路）；两次都解析不出时给标准反问 + 标注。"""
    fake_chat_model.load([ai_text("我觉得这是个不错的想法。"), ai_text("嗯，是的。")])

    decision = _interpret(fake_chat_model, tmp_path)

    assert decision.intent == orch.INTENT_ASK
    assert decision.reply == orch.FALLBACK_ASK
    assert "无法解析" in decision.note
    assert decision.starts_workflow is False


def test_invalid_intent_is_rejected_and_asked_instead(
    fake_chat_model: ScriptedChatModel, tmp_path: Path
) -> None:
    fake_chat_model.load(
        [
            _orchestrator_reply({"intent": "refactor", "reply": "好"}),
            _orchestrator_reply({"intent": "refactor"}),
        ]
    )

    decision = _interpret(fake_chat_model, tmp_path)

    assert decision.intent == orch.INTENT_ASK
    assert decision.note


def test_bare_json_and_prose_preamble_are_accepted(
    fake_chat_model: ScriptedChatModel, tmp_path: Path
) -> None:
    """无围栏的裸 JSON、围栏前后带说明文字都能解析（宽容剥离，与既有契约同口径）。"""
    fake_chat_model.load(
        [_orchestrator_reply('{"intent": "review", "reply": "好的"}', fenced=False)]
    )
    assert _interpret(fake_chat_model, tmp_path).intent == orch.INTENT_REVIEW

    fake_chat_model.load(
        [_orchestrator_reply({"intent": "code", "reply": "好"}, prose="我理解为开发需求。")]
    )
    assert _interpret(fake_chat_model, tmp_path).intent == orch.INTENT_CODE


def test_session_context_is_rendered_into_the_user_message(
    fake_chat_model: ScriptedChatModel, tmp_path: Path
) -> None:
    """会话现状（轮次数 / 当前版本与文件 / 上一轮结论）进用户消息，系统提示词保持常量。"""
    fake_chat_model.load([_orchestrator_reply({"intent": "review_fix", "reply": "好"})])

    _interpret(
        fake_chat_model,
        tmp_path,
        context={
            "rounds": [{"index": 1, "conclusion": "干净通过"}],
            "current_version": 2,
            "version_files": ["main.py", "todo.py"],
            "last_conclusion": "干净通过",
        },
    )

    request = fake_chat_model.requests[-1]
    joined = "\n".join(str(item.content) for item in request)
    assert "已进行 1 轮" in joined
    assert "code_v2/" in joined and "main.py" in joined
    assert "上一轮结论：干净通过" in joined
    assert orch.ORCHESTRATOR_SYSTEM_PROMPT in joined  # 系统提示词为常量（缓存前缀）


def test_decision_notice_reports_the_effective_settings() -> None:
    """本轮口径说明：本条消息按什么处理 + 轮次取值与来源（用户可当场纠正，Design §11）。"""
    decision = orch.Decision(intent=orch.INTENT_CODE, reply="好", rounds=99, note="已夹取")

    notice = orch.decision_notice(
        decision, rounds=10, source=state_mod.SOURCE_REQUIREMENT, rounds_mode="up-to"
    )

    assert "写/改代码" in notice
    assert "轮次上限 10（来源：需求文本提取）" in notice
    assert "已夹取" not in notice  # 标注由调用方统一附加一次（不在这里重复）


def test_round_report_renders_from_structured_state() -> None:
    """每轮汇报从结构化状态渲染（含轮次使用情况、结论与 next_action），不由模型复述。"""
    state = {
        "status": state_mod.STATUS_RESIDUAL,
        "review_round": 2,
        "fix_rounds": 1,
        "max_iterations": 1,
        "max_iterations_source": state_mod.SOURCE_REQUIREMENT,
        "findings": [{"severity": "严重", "location": "main.py:1", "issue": "除零"}],
        "next_action": state_mod.NEXT_FIX_AGAIN,
        "current_version_dir": str(Path("runs") / "r-1" / "workspace" / "code_v2"),
        "version_files": ["main.py"],
    }

    text = orch.round_report(
        state, round_index=2, intent=orch.INTENT_CODE, run_dir="rounds/round-2", report_written=True
    )

    assert "第 2 轮（写/改代码）" in text
    assert "审查 2 轮 / 修复 1 轮（上限 1，来源：需求文本提取）" in text
    assert "达修复轮次上限" in text
    assert "code_v2/（1 个文件）" in text
    assert "可继续修复" in text
    assert "rounds/round-2（报告 report.md）" in text


@pytest.mark.parametrize(
    ("intent", "label"),
    [
        (orch.INTENT_CODE, "写/改代码"),
        (orch.INTENT_REVIEW, "只审查"),
        (orch.INTENT_REVIEW_FIX, "再走一轮审查→修复"),
        (orch.INTENT_ASK, "提问"),
    ],
)
def test_intent_labels_cover_every_intent(intent: str, label: str) -> None:
    assert orch.intent_label(intent) == label


@pytest.mark.parametrize("value", ["²", "𝟑", "3.5", "", "  "])
def test_rounds_with_non_decimal_text_are_treated_as_unspecified(
    fake_chat_model: ScriptedChatModel, tmp_path: Path, value: str
) -> None:
    """``rounds`` 的值是"isdigit 为真但 int() 会抛错"的字符时按未指定处理，不让会话崩掉。"""
    fake_chat_model.load([_orchestrator_reply({"intent": "code", "rounds": value, "reply": "好"})])

    decision = _interpret(fake_chat_model, tmp_path)

    assert decision.intent == orch.INTENT_CODE
    assert decision.rounds is None
