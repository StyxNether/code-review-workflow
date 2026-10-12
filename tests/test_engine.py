"""运行驱动：三条路径、落盘、报告、失败与中断（Design §4.4、§4.5、§4.6、§8.1、§8.2）。"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from support import read_sample, sample_path

from crw.workflow import contracts, records
from crw.workflow import engine as engine_mod
from crw.workflow import state as state_mod


def file_request(name: str, **kwargs: object) -> engine_mod.RunRequest:
    """以仓库内缺陷样例为输入的运行请求（-f 语义：CLI 读文件、引擎落盘）。"""
    return engine_mod.RunRequest(
        source_path=sample_path(name),
        source_text=read_sample(name),
        **kwargs,  # type: ignore[arg-type]
    )


def scripted_review(*responses: object) -> Callable[..., dict[str, str]]:
    """按调用次序给出固定输出的审查节点：元素为 findings 列表或原始文本（坏输出用）。

    调用次数超出给定序列时重复最后一项，便于构造"每轮同类问题"的场景。
    """
    if not responses:
        raise ValueError("scripted_review 至少需要一个响应（findings 列表或原始文本）")
    calls = {"count": 0}

    def node(state: Mapping[str, Any], retry_feedback: str | None = None) -> dict[str, str]:
        index = min(calls["count"], len(responses) - 1)
        calls["count"] += 1
        response = responses[index]
        if isinstance(response, str):
            return {"review_text": response}
        payload = {"findings": response, "summary": "脚本化审查结论。"}
        return {"review_text": f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```"}

    return node


def scripted_fix(code: str, fixes: Sequence[Mapping[str, Any]]) -> Callable[..., dict[str, str]]:
    """返回固定代码与 fixes 清单的修复节点。"""

    def node(state: Mapping[str, Any], retry_feedback: str | None = None) -> dict[str, str]:
        payload = {"fixes": list(fixes)}
        return {
            "fix_text": (
                f"```python\n{code}\n```\n\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
            )
        }

    return node


def failing_node(
    message: str, error: type[BaseException] = RuntimeError
) -> Callable[..., dict[str, str]]:
    def node(state: Mapping[str, Any], retry_feedback: str | None = None) -> dict[str, str]:
        raise error(message)

    return node


def _raise_file_exists(run_id: str) -> Path:
    """模拟"目标目录已存在但不是运行目录"（如数据目录位置被文件占用）。"""
    raise FileExistsError(183, "Cannot create a file when that file already exists")


def severe_finding(issue: str = "除零风险", location: str = "code_v1.py:5") -> dict[str, Any]:
    return {
        "severity": "严重",
        "location": location,
        "issue": issue,
        "suggestion": "加校验",
        "uncertain": False,
    }


def make_engine(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str], **kwargs: Any
) -> engine_mod.WorkflowEngine:
    kwargs.setdefault("clock", fixed_clock)
    kwargs.setdefault("run_id_factory", run_id_sequence)
    return engine_mod.WorkflowEngine(**kwargs)


def read_state(result: engine_mod.RunResult) -> dict[str, Any]:
    assert result.run_dir is not None
    state = records.read_state(result.run_dir)
    assert state is not None
    return state


def read_events(result: engine_mod.RunResult) -> list[dict[str, Any]]:
    assert result.run_dir is not None
    return records.read_events(result.run_dir / "log.jsonl")


def report_text(result: engine_mod.RunResult) -> str:
    assert result.run_dir is not None
    return (result.run_dir / "report.md").read_text(encoding="utf-8")


# ---- 路径一：需求文本 → 首审干净（干净早退）----


def test_requirement_run_is_clean_on_the_first_review(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    engine = make_engine(fixed_clock, run_id_sequence)

    result = engine.execute(engine_mod.RunRequest(requirement="写一个命令行计算器"))

    assert result.exit_code == 0
    assert result.status == state_mod.STATUS_SUCCESS
    assert (result.review_round, result.fix_rounds) == (1, 0)
    assert result.clean is True
    assert result.run_id == "r-20261012-0001"


def test_requirement_run_persists_the_designed_directory_contract(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """运行目录与产物按 Design §4.5 落盘。"""
    result = make_engine(fixed_clock, run_id_sequence).execute(
        engine_mod.RunRequest(requirement="写一个命令行计算器")
    )
    assert result.run_dir is not None

    assert sorted(
        path.relative_to(result.run_dir).as_posix() for path in result.run_dir.rglob("*")
    ) == [
        "log.jsonl",
        "report.md",
        "state.json",
        "workspace",
        "workspace/code_v1.py",
        "workspace/review_1.json",
    ]
    ast.parse((result.run_dir / "workspace" / "code_v1.py").read_text(encoding="utf-8"))


def test_requirement_run_state_snapshot_matches_design(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    result = make_engine(fixed_clock, run_id_sequence).execute(
        engine_mod.RunRequest(requirement="写一个命令行计算器")
    )

    state = read_state(result)

    assert set(state) == set(state_mod.PERSISTED_FIELDS)
    assert state["status"] == state_mod.STATUS_SUCCESS
    assert state["next_action"] == state_mod.NEXT_DONE
    assert state["clean"] is True
    assert state["findings"] == []
    assert state["created_at"] == "2026-10-12T10:00:00+08:00"
    assert state["finished_at"] == "2026-10-12T10:00:00+08:00"
    assert state["input"]["kind"] == state_mod.INPUT_REQUIREMENT
    assert state["parent_run_id"] is None
    assert state["session_id"] is None and state["round_index"] is None
    assert state["review_round"] == state["fix_rounds"] + 1  # 审查发生后的不变量
    assert [entry["node"] for entry in state["history"]] == ["encode", "review"]
    assert state["history"][0]["artifacts"] == ["workspace/code_v1.py"]
    assert all(entry["duration_ms"] >= 0 for entry in state["history"])


def test_requirement_run_log_covers_lifecycle_and_nodes(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    result = make_engine(fixed_clock, run_id_sequence).execute(
        engine_mod.RunRequest(requirement="写一个命令行计算器")
    )

    events = read_events(result)
    kinds = [(event["node"], event["event"]) for event in events]

    assert kinds == [
        ("engine", "start"),
        ("encode", "start"),
        ("encode", "end"),
        ("review", "start"),
        ("review", "end"),
        ("engine", "end"),
    ]
    assert events[1]["iteration"] == 1
    assert events[-1]["detail"] == state_mod.STATUS_SUCCESS


# ---- 路径二：代码文件 → 修复循环 → 复审干净 ----


def test_syntax_fixture_runs_one_fix_round_then_passes(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    result = make_engine(fixed_clock, run_id_sequence).execute(file_request("syntax_error.py"))
    assert result.run_dir is not None

    assert result.exit_code == 0
    assert (result.review_round, result.fix_rounds) == (2, 1)
    workspace = result.run_dir / "workspace"
    first_review = json.loads((workspace / "review_1.json").read_text(encoding="utf-8"))
    second_review = json.loads((workspace / "review_2.json").read_text(encoding="utf-8"))
    assert [item["severity"] for item in first_review["findings"]] == ["严重"]
    assert first_review["iteration"] == 1 and first_review["file"] == "code_v1.py"
    assert second_review["findings"] == []
    ast.parse((workspace / "code_v2.py").read_text(encoding="utf-8"))


def test_syntax_fixture_persists_fix_dispositions(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    result = make_engine(fixed_clock, run_id_sequence).execute(file_request("syntax_error.py"))
    assert result.run_dir is not None

    payload = json.loads(
        (result.run_dir / "workspace" / "fixes_1.json").read_text(encoding="utf-8")
    )

    assert payload["round"] == 1
    assert payload["code_file"] == "code_v2.py"
    assert payload["fixes"][0]["finding_id"] == "f-1"
    assert payload["fixes"][0]["status"] == "fixed"


def test_file_input_is_copied_into_the_workspace(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """-f 输入由命令面读取、引擎复制为 workspace/code_v1.py，原始路径记入 input（Design §4.2.1）。"""
    result = make_engine(fixed_clock, run_id_sequence).execute(file_request("syntax_error.py"))

    state = read_state(result)
    assert result.run_dir is not None
    assert (
        state["input"]["source_path"]
        .replace("\\", "/")
        .endswith("tests/fixtures/buggy_samples/syntax_error.py")
    )
    assert (result.run_dir / "workspace" / "code_v1.py").read_text(encoding="utf-8") == read_sample(
        "syntax_error.py"
    )


def test_report_documents_rounds_dispositions_and_input_mapping(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    result = make_engine(fixed_clock, run_id_sequence).execute(file_request("syntax_error.py"))

    text = report_text(result)

    assert "# 运行报告" in text and "结论：干净通过" in text
    assert "## 运行摘要" in text and "## 时间线" in text
    assert "## 逐轮问题清单" in text and "## 产物清单" in text
    assert "### 第 1 轮审查（基于 `code_v1.py`）" in text
    assert "### 第 2 轮审查（基于 `code_v2.py`）" in text
    assert "**fixed**" in text  # 问题 ← 处置对照
    assert "原文件 → 工作区" in text
    assert "`report.md`" in text  # 产物清单含报告自身


# ---- 路径三：代码文件 → 同类【严重】直至达限（达限残留）----


def test_divide_by_zero_hits_the_limit_and_reports_residual(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    result = make_engine(fixed_clock, run_id_sequence).execute(
        file_request("divide_by_zero.py", max_iterations=1)
    )
    assert result.run_dir is not None

    assert result.exit_code == 1
    assert result.status == state_mod.STATUS_RESIDUAL
    assert (result.review_round, result.fix_rounds) == (2, 1)
    assert result.clean is False
    state = read_state(result)
    assert state["next_action"] == state_mod.NEXT_FIX_AGAIN
    assert contracts.residual_severe(state["findings"]) == 1

    workspace = result.run_dir / "workspace"
    assert (workspace / "code_v2.py").read_text(encoding="utf-8") == (
        workspace / "code_v1.py"
    ).read_text(encoding="utf-8")
    assert "**skipped**" in report_text(result)


def test_unchanged_round_is_annotated(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """修复未产生改动时不失败、继续复审，并在时间线与报告标注（Design §4.2）。"""
    result = make_engine(fixed_clock, run_id_sequence).execute(
        file_request("divide_by_zero.py", max_iterations=2)
    )

    state = read_state(result)
    fix_entry = next(entry for entry in state["history"] if entry["node"] == "fix")

    assert "本轮未产生改动" in fix_entry["summary"]
    assert "本轮未产生改动" in report_text(result)


def test_residual_report_lists_the_last_round_without_disposition(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    result = make_engine(fixed_clock, run_id_sequence).execute(
        file_request("divide_by_zero.py", max_iterations=1)
    )

    text = report_text(result)

    assert "结论：达修复轮次上限，仍有 1 项【严重】残留" in text
    assert "### 第 2 轮审查（基于 `code_v2.py`）" in text
    assert "处置：—" in text  # 末轮无修复轮，未处置


def test_max_iter_zero_reviews_once_without_fixing(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """--max-iter 0 = 只审不修（Design §4.4）。"""
    result = make_engine(fixed_clock, run_id_sequence).execute(
        file_request("divide_by_zero.py", max_iterations=0)
    )
    assert result.run_dir is not None

    assert result.exit_code == 1
    assert (result.review_round, result.fix_rounds) == (1, 0)
    assert not (result.run_dir / "workspace" / "code_v2.py").exists()
    assert not (result.run_dir / "workspace" / "fixes_1.json").exists()


# ---- 注入节点：固定 findings 序列与契约接缝 ----


def test_injected_review_sequence_drives_the_loop(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """测试经 nodes= 注入固定 findings 序列（Design §9.1 的注入接缝）。"""
    nodes = {"review": scripted_review([severe_finding()], [])}

    result = make_engine(fixed_clock, run_id_sequence, nodes=nodes).execute(
        engine_mod.RunRequest(requirement="写一个计算器")
    )

    assert result.exit_code == 0
    assert (result.review_round, result.fix_rounds) == (2, 1)


def test_injected_review_sequence_can_force_residual(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    nodes = {"review": scripted_review([severe_finding()])}

    result = make_engine(fixed_clock, run_id_sequence, nodes=nodes).execute(
        engine_mod.RunRequest(requirement="写一个计算器", max_iterations=2)
    )

    assert result.exit_code == 1
    assert (result.review_round, result.fix_rounds) == (3, 2)
    assert contracts.residual_severe(read_state(result)["findings"]) == 1


def test_strict_mode_blocks_on_moderate_findings(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """--strict 加严判定：无【严重】但有【一般】时仍判不干净（Design §4.4）。"""
    moderate = [{"severity": "一般", "location": "code_v1.py:3", "issue": "静默吞异常"}]
    nodes = {"review": scripted_review(moderate)}

    lenient = make_engine(fixed_clock, run_id_sequence, nodes=nodes).execute(
        engine_mod.RunRequest(requirement="写一个计算器", max_iterations=0)
    )
    strict = make_engine(fixed_clock, run_id_sequence, nodes=nodes).execute(
        engine_mod.RunRequest(requirement="写一个计算器", max_iterations=0, strict=True)
    )

    assert lenient.exit_code == 0
    assert strict.exit_code == 1
    assert read_state(strict)["strict"] is True


def test_strict_residual_report_counts_blocking_findings(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """--strict 下残留可能全部由【一般】构成：报告口径与判定同源，不得写"0 项【严重】"。"""
    moderate = [{"severity": "一般", "location": "code_v1.py:3", "issue": "静默吞异常"}]
    nodes = {"review": scripted_review(moderate)}

    result = make_engine(fixed_clock, run_id_sequence, nodes=nodes).execute(
        engine_mod.RunRequest(requirement="写一个计算器", max_iterations=0, strict=True)
    )

    text = report_text(result)
    assert "仍有 1 项阻塞问题（--strict：含【一般】）残留" in text
    assert "仍有 0 项" not in text
    assert contracts.residual_severe(result.findings) == 0  # runs 列表的"残留严重数"仍只数严重


def test_contract_violation_retries_once_with_feedback(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """契约校验失败 → 附错误清单重试一次（Design §4.6）；重试不计审查轮次。"""
    seen: list[str | None] = []

    def review(state: Mapping[str, Any], retry_feedback: str | None = None) -> dict[str, str]:
        seen.append(retry_feedback)
        if retry_feedback is None:
            return {"review_text": "第一次没有任何围栏的乱输出"}
        return {"review_text": '```json\n{"findings": [], "summary": "重试后干净。"}\n```'}

    result = make_engine(fixed_clock, run_id_sequence, nodes={"review": review}).execute(
        engine_mod.RunRequest(requirement="写一个计算器")
    )

    assert result.exit_code == 0
    assert (result.review_round, result.fix_rounds) == (1, 0)  # 重试不计轮次
    assert seen[0] is None and "恰好一个 ```json 围栏" in str(seen[1])
    kinds = [(event["node"], event["event"]) for event in read_events(result)]
    assert ("review", "retry") in kinds


def test_repeated_contract_violation_fails_the_node(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    nodes = {"review": scripted_review("坏输出一", "坏输出二")}

    result = make_engine(fixed_clock, run_id_sequence, nodes=nodes).execute(
        engine_mod.RunRequest(requirement="写一个计算器")
    )

    assert result.exit_code == 2
    assert result.status == state_mod.STATUS_FAILED
    assert result.error_node == "review"
    assert "契约校验未通过" in report_text(result) or "围栏" in (result.error_message or "")


def test_fix_contract_reports_unknown_ids_in_the_report(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """未知 finding_id 忽略并在报告标注（Design §5.2 边界）。"""
    nodes = {
        "review": scripted_review([severe_finding()], []),
        "fix": scripted_fix(
            "print(1)\n",
            [
                {"finding_id": "f-1", "status": "fixed", "note": "已修"},
                {"finding_id": "f-9", "status": "fixed", "note": "无中生有"},
            ],
        ),
    }

    result = make_engine(fixed_clock, run_id_sequence, nodes=nodes).execute(
        engine_mod.RunRequest(requirement="写一个计算器")
    )

    assert result.exit_code == 0
    assert "未识别的处置（已忽略）：f-9" in report_text(result)


# ---- 失败与中断（Design §4.6）----


def test_node_failure_marks_run_failed_with_readable_diagnostics(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    nodes = {"review": failing_node("审查模型调用失败：连接超时")}

    result = make_engine(fixed_clock, run_id_sequence, nodes=nodes).execute(
        engine_mod.RunRequest(requirement="写一个计算器")
    )

    assert result.exit_code == 2
    assert result.status == state_mod.STATUS_FAILED
    assert result.error_node == "review"
    state = read_state(result)
    assert state["error"] == {
        "node": "review",
        "message": "RuntimeError: 审查模型调用失败：连接超时",
    }
    assert state["history"][-1]["summary"].startswith("节点失败")
    assert ("review", "error") in [(event["node"], event["event"]) for event in read_events(result)]
    assert "## 错误" in report_text(result)


def test_recursion_guard_error_is_translated(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    nodes = {"review": failing_node("maximum recursion depth exceeded", RecursionError)}

    result = make_engine(fixed_clock, run_id_sequence, nodes=nodes).execute(
        engine_mod.RunRequest(requirement="写一个计算器")
    )

    assert result.exit_code == 2
    assert "递归护栏" in (result.error_message or "")


def test_recursion_limit_guard_actually_stops_a_non_converging_graph(
    monkeypatch: pytest.MonkeyPatch,
    fixed_clock: Callable[[], str],
    run_id_sequence: Callable[[], str],
) -> None:
    """护栏真正生效：图不收敛时被 recursion_limit 截停并给出可读报错。

    回归护栏的传递方式——langgraph 的 ``stream(..., **kwargs)`` 会静默吞掉
    ``recursion_limit`` 关键字（实测生效值回落为默认 10007），必须经 config 位置参数传入；
    断言节点执行次数贴着 ``2 × max_iter + 8`` 才证明生效（被吞时会跑到上万次）。
    """
    from langgraph.graph import END, START, StateGraph

    from crw.workflow import graph as graph_mod
    from crw.workflow.state import RunState

    calls = {"count": 0}

    def looping(state: Mapping[str, Any]) -> Mapping[str, Any]:
        calls["count"] += 1
        return {}

    builder = StateGraph(RunState)
    builder.add_node("encode", looping)
    builder.add_edge(START, "encode")
    builder.add_edge("encode", "encode")  # 自环：任何状态都回到自己，永不收尾
    builder.add_edge("encode", END)
    looping_graph = builder.compile()
    monkeypatch.setattr(graph_mod, "build_graph", lambda **kwargs: looping_graph)

    result = make_engine(fixed_clock, run_id_sequence).execute(
        engine_mod.RunRequest(requirement="写一个计算器", max_iterations=1)
    )

    assert result.exit_code == 2
    assert result.status == state_mod.STATUS_FAILED
    assert "递归护栏" in (result.error_message or "")
    assert calls["count"] <= 2 * 1 + 8


def test_interrupt_writes_state_and_exits_two(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """Ctrl+C：log 追加 interrupt 事件 → 原子写 state.json → 退出码 2（Design §4.6）。"""
    nodes = {"review": failing_node("用户打断", KeyboardInterrupt)}

    result = make_engine(fixed_clock, run_id_sequence, nodes=nodes).execute(
        engine_mod.RunRequest(requirement="写一个计算器")
    )

    assert result.exit_code == 2
    assert result.interrupted is True
    assert result.status == state_mod.STATUS_FAILED
    assert result.error_node == "interrupt"
    state = read_state(result)
    assert state["error"]["node"] == "interrupt"
    assert state["finished_at"] == "2026-10-12T10:00:00+08:00"
    assert [event["event"] for event in read_events(result)][-1] == "interrupt"
    assert result.report_written is False  # 中断只写 log + state（Design §4.6 的写盘序列）
    assert result.run_dir is not None
    assert not (result.run_dir / "report.md").exists()


def test_second_interrupt_abandons_persistence(
    monkeypatch: pytest.MonkeyPatch,
    fixed_clock: Callable[[], str],
    run_id_sequence: Callable[[], str],
) -> None:
    """第二次 Ctrl+C：放弃写盘（"尽力"语义），仍按中断结论退出 2（Design §4.6）。"""
    real_write = records.atomic_write_text
    state_writes = {"count": 0}

    def flaky_write(path: Path, text: str, **kwargs: Any) -> None:
        if path.name == "state.json":
            state_writes["count"] += 1
            if state_writes["count"] >= 2:  # 中断收尾时的写盘再被打断
                raise KeyboardInterrupt
        real_write(path, text, **kwargs)

    monkeypatch.setattr(records, "atomic_write_text", flaky_write)
    nodes = {"review": failing_node("用户打断", KeyboardInterrupt)}

    result = make_engine(fixed_clock, run_id_sequence, nodes=nodes).execute(
        engine_mod.RunRequest(requirement="写一个计算器")
    )

    assert result.exit_code == 2
    assert result.interrupted is True
    assert result.run_dir is not None
    on_disk = json.loads((result.run_dir / "state.json").read_text(encoding="utf-8"))
    assert on_disk["status"] == state_mod.STATUS_RUNNING  # 放弃写盘：保留中断前的快照


def test_state_write_failure_is_a_readable_run_failure(
    monkeypatch: pytest.MonkeyPatch,
    fixed_clock: Callable[[], str],
    run_id_sequence: Callable[[], str],
) -> None:
    real_write = records.atomic_write_text

    def failing_write(path: Path, text: str, **kwargs: Any) -> None:
        if path.name == "state.json":
            raise OSError("磁盘已满")
        real_write(path, text, **kwargs)

    monkeypatch.setattr(records, "atomic_write_text", failing_write)

    result = make_engine(fixed_clock, run_id_sequence).execute(
        engine_mod.RunRequest(requirement="写一个计算器")
    )

    assert result.exit_code == 2
    assert result.status == state_mod.STATUS_FAILED
    assert "state.json 写入失败" in (result.error_message or "")
    assert "磁盘已满" in (result.error_message or "")


def test_report_write_failure_points_to_state_json(
    monkeypatch: pytest.MonkeyPatch,
    fixed_clock: Callable[[], str],
    run_id_sequence: Callable[[], str],
) -> None:
    real_write = records.atomic_write_text

    def failing_write(path: Path, text: str, **kwargs: Any) -> None:
        if path.name == "report.md":
            raise OSError("只读目录")
        real_write(path, text, **kwargs)

    monkeypatch.setattr(records, "atomic_write_text", failing_write)

    result = make_engine(fixed_clock, run_id_sequence).execute(
        engine_mod.RunRequest(requirement="写一个计算器")
    )

    assert result.exit_code == 2
    assert "report.md 写入失败" in (result.error_message or "")
    assert "state.json" in (result.error_message or "")


def test_artifact_write_failure_is_classified_as_persistence_failure(
    monkeypatch: pytest.MonkeyPatch,
    fixed_clock: Callable[[], str],
    run_id_sequence: Callable[[], str],
) -> None:
    """节点产物（代码 / 审查结论）写盘失败按"落盘失败"定位，不误报为节点自身失败。"""
    real_write = records.atomic_write_text

    def failing_write(path: Path, text: str, **kwargs: Any) -> None:
        if path.name == "review_1.json":
            raise OSError("磁盘已满")
        real_write(path, text, **kwargs)

    monkeypatch.setattr(records, "atomic_write_text", failing_write)

    result = make_engine(fixed_clock, run_id_sequence).execute(
        engine_mod.RunRequest(requirement="写一个计算器")
    )

    assert result.exit_code == 2
    assert result.error_node == "engine"  # 磁盘故障归引擎级，不误报成"审查节点失败"
    assert "产物 review_1.json 写入失败" in (result.error_message or "")
    assert "磁盘已满" in (result.error_message or "")
    state = read_state(result)
    assert state["error"]["node"] == "engine"
    assert not any("节点失败" in entry["summary"] for entry in state["history"])
    assert ("engine", "error") in [(event["node"], event["event"]) for event in read_events(result)]


def test_run_directory_creation_with_unusable_parent_path_is_readable(
    monkeypatch: pytest.MonkeyPatch,
    fixed_clock: Callable[[], str],
    run_id_sequence: Callable[[], str],
) -> None:
    """父路径不可创建（如数据目录位置被文件占用）不算编号碰撞，直接给可读原因。"""
    monkeypatch.setattr(records, "create_run_dir", _raise_file_exists)

    result = make_engine(fixed_clock, run_id_sequence).execute(
        engine_mod.RunRequest(requirement="写一个计算器")
    )

    assert result.exit_code == 2
    message = result.error_message or ""
    assert "运行目录创建失败" in message
    assert "连续碰撞" not in message


def test_broken_output_stream_does_not_fail_the_run(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """渲染层输出流断开（BrokenPipeError 等）不改变运行结论，也不误报为节点失败。"""

    class BrokenObserver:
        """模拟下游已退出的终端：所有渲染调用都抛 BrokenPipeError。"""

        def on_event(self, event: Mapping[str, Any]) -> None:
            raise BrokenPipeError("下游已关闭")

        def on_review(self, **kwargs: Any) -> None:
            raise BrokenPipeError("下游已关闭")

        def on_fixes(self, **kwargs: Any) -> None:
            raise BrokenPipeError("下游已关闭")

    result = make_engine(fixed_clock, run_id_sequence, observer=BrokenObserver()).execute(
        file_request("syntax_error.py")
    )

    assert result.exit_code == 0
    assert result.status == state_mod.STATUS_SUCCESS
    assert (result.review_round, result.fix_rounds) == (2, 1)


def test_state_snapshot_is_written_at_every_node_boundary(
    monkeypatch: pytest.MonkeyPatch,
    fixed_clock: Callable[[], str],
    run_id_sequence: Callable[[], str],
) -> None:
    """state.json 在每个节点结束与失败时各原子写一次（Design §8.2）。"""
    real_write = records.atomic_write_text
    snapshots: list[dict[str, Any]] = []

    def spying_write(path: Path, text: str, **kwargs: Any) -> None:
        if path.name == "state.json":
            snapshots.append(json.loads(text))
        real_write(path, text, **kwargs)

    monkeypatch.setattr(records, "atomic_write_text", spying_write)

    result = make_engine(fixed_clock, run_id_sequence).execute(file_request("syntax_error.py"))

    assert [item["status"] for item in snapshots] == [
        state_mod.STATUS_RUNNING,
        state_mod.STATUS_RUNNING,
        state_mod.STATUS_RUNNING,
        state_mod.STATUS_RUNNING,
        state_mod.STATUS_SUCCESS,
    ]
    assert [(item["review_round"], item["fix_rounds"]) for item in snapshots] == [
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 1),
        (2, 1),
    ]
    assert result.review_round == 2


def test_run_id_collision_retries_with_a_new_id(
    fixed_clock: Callable[[], str],
) -> None:
    """编号碰撞（目录已存在）时重新生成运行号（Design §4.3）。"""
    existing = "r-20261012-0001"
    records.create_run_dir(existing)
    ids = iter([existing, "r-20261012-0002"])

    result = make_engine(fixed_clock, ids.__next__).execute(
        engine_mod.RunRequest(requirement="写一个计算器")
    )

    assert result.run_id == "r-20261012-0002"
    assert result.exit_code == 0


def test_run_directory_creation_failure_is_readable(
    monkeypatch: pytest.MonkeyPatch,
    fixed_clock: Callable[[], str],
    run_id_sequence: Callable[[], str],
) -> None:
    def broken_create(run_id: str) -> Path:
        raise OSError("磁盘只读")

    monkeypatch.setattr(records, "create_run_dir", broken_create)

    result = make_engine(fixed_clock, run_id_sequence).execute(
        engine_mod.RunRequest(requirement="写一个计算器")
    )

    assert result.exit_code == 2
    assert "运行目录创建失败" in (result.error_message or "")
    assert result.run_dir is None


# ---- 选项与来源记录 ----


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (state_mod.SOURCE_CLI, "命令行"),
        (state_mod.SOURCE_DEFAULT, "内置默认"),
        (state_mod.SOURCE_PARENT, "继承父运行"),
    ],
)
def test_max_iterations_source_is_recorded_verbatim(
    fixed_clock: Callable[[], str],
    run_id_sequence: Callable[[], str],
    source: str,
    expected: str,
) -> None:
    result = make_engine(fixed_clock, run_id_sequence).execute(
        engine_mod.RunRequest(
            requirement="写一个计算器", max_iterations=2, max_iterations_source=source
        )
    )

    state = read_state(result)
    assert state["max_iterations"] == 2
    assert state["max_iterations_source"] == source
    assert f"来源：{expected}" in report_text(result)


def test_run_request_records_labels_for_future_milestones(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """会话字段与模型标识按入参写入（M1 不实现会话，但记录结构须可扩展，Design §4.7）。"""
    result = make_engine(fixed_clock, run_id_sequence).execute(
        engine_mod.RunRequest(
            requirement="写一个计算器",
            model_label="some-provider/some-model",
            parent_run_id="r-20261011-0001",
            session_id="s-20261012-0001",
            round_index=2,
            rounds_mode=state_mod.ROUNDS_UP_TO,
        )
    )

    state = read_state(result)
    assert state["model"] == "some-provider/some-model"
    assert state["parent_run_id"] == "r-20261011-0001"
    assert state["session_id"] == "s-20261012-0001"
    assert state["round_index"] == 2
    assert state["rounds_mode"] == state_mod.ROUNDS_UP_TO


def test_extra_fences_are_ignored_but_recorded(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """Design §4.2.1：修复输出的多余围栏忽略并记 log（写进阶段摘要即进 log.jsonl）。"""

    def chatty_fix(state: Mapping[str, Any], retry_feedback: str | None = None) -> dict[str, str]:
        fixes_body = json.dumps(
            {"fixes": [{"finding_id": "f-1", "status": "fixed", "note": "已修"}]},
            ensure_ascii=False,
        )
        return {
            "fix_text": (
                "```text\n旁白\n```\n\n"
                "```python\nprint(1)\n```\n\n"
                f"```json\n{fixes_body}\n```\n\n"
                "```json\n{}\n```\n"
            )
        }

    nodes = {"review": scripted_review([severe_finding()], []), "fix": chatty_fix}

    result = make_engine(fixed_clock, run_id_sequence, nodes=nodes).execute(
        engine_mod.RunRequest(requirement="写一个计算器")
    )

    assert result.exit_code == 0
    state = read_state(result)
    fix_entry = next(entry for entry in state["history"] if entry["node"] == "fix")
    assert "已忽略 2 个多余围栏" in fix_entry["summary"]
    assert any(
        "已忽略 2 个多余围栏" in str(event.get("detail"))
        for event in read_events(result)
        if event.get("event") == "end"
    )


def test_observer_defects_that_are_not_output_failures_still_surface(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """渲染器的编码缺陷（非 OSError）不得被吞：照常暴露为可读运行失败。"""

    class BuggyObserver:
        def on_event(self, event: Mapping[str, Any]) -> None:
            return None

        def on_review(self, **kwargs: Any) -> None:
            raise TypeError("渲染器缺陷")

        def on_fixes(self, **kwargs: Any) -> None:
            return None

    result = make_engine(fixed_clock, run_id_sequence, observer=BuggyObserver()).execute(
        engine_mod.RunRequest(requirement="写一个计算器")
    )

    assert result.exit_code == 2
    assert "渲染器缺陷" in (result.error_message or "")


def test_clean_stage_line_mentions_advice_only_when_present(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """阶段摘要的"（建议不影响判定）"只在确有【建议】时出现（0 findings 时不加括号噪音）。"""
    advice = [{"severity": "建议", "location": "code_v1.py", "issue": "缺少模块文档"}]

    plain = make_engine(fixed_clock, run_id_sequence).execute(
        engine_mod.RunRequest(requirement="写一个计算器")
    )
    with_advice = make_engine(
        fixed_clock, run_id_sequence, nodes={"review": scripted_review(advice)}
    ).execute(engine_mod.RunRequest(requirement="写一个计算器"))

    plain_review = next(
        entry for entry in read_state(plain)["history"] if entry["node"] == "review"
    )
    advice_review = next(
        entry for entry in read_state(with_advice)["history"] if entry["node"] == "review"
    )
    assert "建议不影响判定" not in plain_review["summary"]
    assert "建议不影响判定" in advice_review["summary"]
