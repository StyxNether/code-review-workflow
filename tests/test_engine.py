"""运行驱动：三条路径、落盘、报告、失败与中断（Design §4.4、§4.5、§4.6、§8.1、§8.2）。"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import fakes
import pytest
from support import read_sample, sample_path

from crw import config as config_mod
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
    """返回固定代码与 fixes 清单的修复节点（输出版本内全部文件，首文件替换为给定代码）。"""

    def node(state: Mapping[str, Any], retry_feedback: str | None = None) -> dict[str, str]:
        raw = str(state.get("current_version_dir") or "")
        # 空值直接走兜底：Path("") 等价于当前目录，会把仓库整棵树当版本读
        directory = Path(raw) if raw.strip() else None
        files: dict[str, str] = {}
        if directory is not None and directory.is_dir():
            for item in sorted(directory.rglob("*")):
                if item.is_file() and item.name != "manifest.json":
                    files[item.relative_to(directory).as_posix()] = item.read_text(encoding="utf-8")
        if not files:
            files = {"main.py": ""}
        first = min(files)
        files[first] = code
        payload = {"fixes": list(fixes)}
        fences = "\n\n".join(
            f"```python file={name}\n{content}\n```" for name, content in sorted(files.items())
        )
        return {"fix_text": f"{fences}\n\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```"}

    return node


def failing_node(
    message: str, error: type[BaseException] = RuntimeError
) -> Callable[..., dict[str, str]]:
    def node(state: Mapping[str, Any], retry_feedback: str | None = None) -> dict[str, str]:
        raise error(message)

    return node


def _raise_file_exists(run_id: str, *, parent: Path | None = None) -> Path:
    """模拟"目标目录已存在但不是运行目录"（如数据目录位置被文件占用）。"""
    raise FileExistsError(183, "Cannot create a file when that file already exists")


def severe_finding(issue: str = "除零风险", location: str = "main.py:5") -> dict[str, Any]:
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
    """构造引擎：**恒注入**脚本化假节点整表（生产缺省装配是真实 Agent 节点，需模型与网络）。

    ``nodes`` 逐键并入假节点整表，用例只需给出关心的一两个键；本 helper 不支持"完全不注入"
    ——要跑真实装配请直接构造 ``WorkflowEngine``（那是 CLI 路径，由 test_cli 覆盖）。
    """
    kwargs.setdefault("clock", fixed_clock)
    kwargs.setdefault("run_id_factory", run_id_sequence)
    overrides = dict(kwargs.pop("nodes", None) or {})
    kwargs["nodes"] = {**fakes.SCRIPTED_NODES(), **overrides}
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
        "workspace/code_v1",
        "workspace/code_v1/main.py",
        "workspace/code_v1/manifest.json",
        "workspace/review_1.json",
    ]
    manifest = json.loads(
        (result.run_dir / "workspace" / "code_v1" / "manifest.json").read_text(encoding="utf-8")
    )
    assert [entry["path"] for entry in manifest["files"]] == ["main.py"]
    ast.parse((result.run_dir / "workspace" / "code_v1" / "main.py").read_text(encoding="utf-8"))


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
    assert state["history"][0]["artifacts"] == ["workspace/code_v1/main.py"]
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
    assert first_review["iteration"] == 1 and first_review["files"] == ["syntax_error.py"]
    assert second_review["findings"] == []
    ast.parse((workspace / "code_v2" / "syntax_error.py").read_text(encoding="utf-8"))


def test_syntax_fixture_persists_fix_dispositions(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    result = make_engine(fixed_clock, run_id_sequence).execute(file_request("syntax_error.py"))
    assert result.run_dir is not None

    payload = json.loads(
        (result.run_dir / "workspace" / "fixes_1.json").read_text(encoding="utf-8")
    )

    assert payload["round"] == 1
    assert payload["version_dir"] == "code_v2"
    assert payload["files"] == ["syntax_error.py"]
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
    assert (result.run_dir / "workspace" / "code_v1" / "syntax_error.py").read_text(
        encoding="utf-8"
    ) == read_sample("syntax_error.py")


def test_report_documents_rounds_dispositions_and_input_mapping(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    result = make_engine(fixed_clock, run_id_sequence).execute(file_request("syntax_error.py"))

    text = report_text(result)

    assert "# 运行报告" in text and "结论：干净通过" in text
    assert "## 运行摘要" in text and "## 时间线" in text
    assert "## 逐轮问题清单" in text and "## 产物清单" in text
    assert "### 第 1 轮审查（基于 `code_v1/`）" in text
    assert "### 第 2 轮审查（基于 `code_v2/`）" in text
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
    assert (workspace / "code_v2" / "divide_by_zero.py").read_text(encoding="utf-8") == (
        workspace / "code_v1" / "divide_by_zero.py"
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
    assert "### 第 2 轮审查（基于 `code_v2/`）" in text
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
    assert not (result.run_dir / "workspace" / "code_v2").exists()
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
    def broken_create(run_id: str, *, parent: Path | None = None) -> Path:
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
    """会话字段与模型标识按入参写入（会话未实现，但记录结构须可扩展，Design §4.7）。"""
    result = make_engine(fixed_clock, run_id_sequence).execute(
        engine_mod.RunRequest(
            requirement="写一个计算器",
            resolved=config_mod.ResolvedModel(
                model="some-model",
                base_url="https://example.invalid/v1",
                api_key="placeholder-key",
                provider="some-provider",
            ),
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
                "```python file=main.py\nprint(1)\n```\n\n"
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


# ---- 护栏触达信息并入阶段摘要 / 失败原因（Design §4.6）----


def test_notices_are_appended_to_the_stage_summary() -> None:
    assert engine_mod._append_notices("第 1 轮：0 严重", ["工具调用已达上限（1 轮）"]) == (
        "第 1 轮：0 严重（工具调用已达上限（1 轮），结论可能不完整）"
    )


def test_no_notices_leaves_the_summary_untouched() -> None:
    assert engine_mod._append_notices("第 1 轮：0 严重", []) == "第 1 轮：0 严重"


def test_the_runtime_is_rebuilt_for_every_run(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """同一引擎实例被复用执行两次时，运行时按请求重建（不串用上次的模型 / no_exec / 上下文根）。"""
    built: list[str] = []

    class _StubRuntime:
        def run(self, **kwargs: Any) -> Any:
            raise RuntimeError("stub 运行时：本用例只验证装配是否重建")

        def drain_notices(self) -> list[str]:
            return []

    def factory(emit: Any, delta: Any) -> Any:
        built.append("built")
        return _StubRuntime()

    # 只覆盖编码节点 → 审查 / 修复走缺省装配，必然经运行时工厂（make_engine 会注入整表，故此处直接构造）
    engine = engine_mod.WorkflowEngine(
        nodes={"encode": fakes.fake_encode},
        runtime_factory=factory,
        clock=fixed_clock,
        run_id_factory=run_id_sequence,
    )

    first = engine.execute(engine_mod.RunRequest(requirement="写一个计算器"))
    second = engine.execute(engine_mod.RunRequest(requirement="写一个计算器"))

    assert first.exit_code == engine_mod._EXIT_ERROR
    assert second.exit_code == engine_mod._EXIT_ERROR
    assert len(built) == 2  # 第二次运行没有复用第一次的运行时


# ---- M3：并行双视角审查（Design §12.2 的扩展示例）----


def make_parallel_engine(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str], **kwargs: Any
) -> engine_mod.WorkflowEngine:
    """构造并行形态引擎：同样注入脚本化节点整表（含汇合节点，与生产装配同形）。"""
    kwargs.setdefault("clock", fixed_clock)
    kwargs.setdefault("run_id_factory", run_id_sequence)
    overrides = dict(kwargs.pop("nodes", None) or {})
    kwargs["nodes"] = {**fakes.PARALLEL_SCRIPTED_NODES(), **overrides}
    return engine_mod.WorkflowEngine(**kwargs)


def test_parallel_review_runs_two_views_and_merges_them(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """两视角同轮并行、分支结论各落一份、汇合去重合并（Design §12.2 的汇合规则）。"""
    engine = make_parallel_engine(fixed_clock, run_id_sequence)

    # 以缺陷样例为输入：脚本化修复对语义型缺陷只会标注未采纳，代码逐轮不变，
    # 因此两个视角每轮都报同一批位置——正好覆盖去重与"取较严重者"
    result = engine.execute(
        file_request("divide_by_zero.py", parallel_review=True, max_iterations=1)
    )

    assert result.exit_code == 1  # 安全性视角独有的一项【严重】无法修掉 → 达限残留
    state = read_state(result)
    assert state["parallel_review"] is True
    assert state["review_round"] == state["fix_rounds"] + 1  # 汇合完成时不变式成立
    workspace = result.run_dir / "workspace"
    assert (workspace / "review_1_correctness.json").is_file()
    assert (workspace / "review_1_security.json").is_file()
    assert (workspace / "review_1.json").is_file()

    merged = state["findings"]
    assert merged, "两个视角各自都报出了问题，合并结果不应为空"
    assert [item["id"] for item in merged] == [f"f-{index}" for index in range(1, len(merged) + 1)]
    shared = [item for item in merged if item["views"] == ["正确性", "安全性"]]
    assert shared, "同一 (location, issue) 应当被合并为一条并记下来源视角"
    assert all(item["severity"] == "严重" for item in shared), "取较严重者（正确性侧更严重）"
    security_only = [item for item in merged if item["views"] == ["安全性"]]
    assert len(security_only) == 1 and security_only[0]["severity"] == "严重"
    # 汇合结果即干净判定依据：安全性独有的一项是【严重】→ 不得干净通过
    assert state["clean"] is False
    assert contracts.judge_clean(merged, strict=False) is state["clean"]


def test_parallel_review_branch_keys_are_not_persisted(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """分支 key 是图执行期的中间态：不写进运行记录（产物落 review_N_<视角>.json）。"""
    engine = make_parallel_engine(fixed_clock, run_id_sequence)

    result = engine.execute(
        engine_mod.RunRequest(requirement="写一个命令行计算器", parallel_review=True)
    )

    state = read_state(result)
    assert "findings_correctness" not in state
    assert "findings_security" not in state
    assert result.run_dir is not None
    assert (result.run_dir / "workspace" / "review_1_correctness.json").is_file()


def test_parallel_review_timeline_names_views_and_merge(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """时间线 / 日志把两个视角与汇合分别记名（可审计：这一轮是谁在审、谁在合并）。"""
    engine = make_parallel_engine(fixed_clock, run_id_sequence)

    result = engine.execute(
        engine_mod.RunRequest(requirement="写一个命令行计算器", parallel_review=True)
    )

    state = read_state(result)
    nodes = [entry["node"] for entry in state["history"]]
    assert {"review_correctness", "review_security", "merge_reviews"} <= set(nodes)
    assert nodes.count("merge_reviews") == state["review_round"]
    report = report_text(result)
    assert "双视角并行（正确性 + 安全性，汇合后按 (location, issue) 去重）" in report
    summary = next(
        entry["summary"] for entry in state["history"] if entry["node"] == "merge_reviews"
    )
    assert "双视角合并" in summary


def test_parallel_review_second_round_fans_out_again(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """修复后重新扇出到两个视角（同一超级步），汇合后再判干净。"""
    engine = make_parallel_engine(fixed_clock, run_id_sequence)

    result = engine.execute(
        engine_mod.RunRequest(
            requirement="写一个命令行计算器", parallel_review=True, max_iterations=1
        )
    )

    state = read_state(result)
    assert state["fix_rounds"] == 1
    assert state["review_round"] == 2
    assert (result.run_dir / "workspace" / "review_2_correctness.json").is_file()
    assert (result.run_dir / "workspace" / "review_2.json").is_file()


# ---- M3：未收敛标注（Design §4.4）----


def test_unchanged_finding_set_is_flagged_as_not_converged(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """同轮与上轮 (location, issue) 集合一致 → 报告与阶段摘要标注，但不提前中止。"""
    review = scripted_review([severe_finding()], [severe_finding()])
    engine = make_engine(fixed_clock, run_id_sequence, nodes={"review": review})

    result = engine.execute(file_request("divide_by_zero.py", max_iterations=1))

    assert result.exit_code == 1  # 仍按达限残留收尾：标注不改变流程
    state = read_state(result)
    assert state["review_round"] == 2
    summaries = [str(entry["summary"]) for entry in state["history"] if entry["node"] == "review"]
    assert "未收敛" not in summaries[0]  # 首轮没有可比较的上一轮
    assert "未收敛" in summaries[-1]  # 次轮集合与首轮一致 → 标注
    assert "**未收敛**" in report_text(result)


def test_changed_finding_set_is_not_flagged(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """问题清单有变化（哪怕仍是同一级别）就不算未收敛。"""
    review = scripted_review([severe_finding()], [severe_finding(issue="换了一条问题")])
    engine = make_engine(fixed_clock, run_id_sequence, nodes={"review": review})

    result = engine.execute(file_request("divide_by_zero.py", max_iterations=1))

    assert all("未收敛" not in str(entry["summary"]) for entry in read_state(result)["history"])
    assert "**未收敛**" not in report_text(result)


# ---- M3：续跑的父运行标注（Design §4.4 / §4.5）----


def test_resumed_run_records_parent_and_reports_parent_residual(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """续跑新运行记 parent_run_id，报告标注父运行与父残留摘要。"""
    engine = make_engine(fixed_clock, run_id_sequence)
    parent = engine.execute(file_request("divide_by_zero.py", max_iterations=1))
    assert parent.exit_code == 1  # 父运行：达限残留

    version_dir = Path(str(read_state(parent)["current_version_dir"]))
    resumed = engine.execute(
        engine_mod.RunRequest(
            parent_run_id=parent.run_id,
            source_dir=version_dir,
            max_iterations=1,
            max_iterations_source=state_mod.SOURCE_PARENT,
        )
    )

    state = read_state(resumed)
    assert state["parent_run_id"] == parent.run_id
    assert state["max_iterations_source"] == state_mod.SOURCE_PARENT
    report = report_text(resumed)
    assert f"| 父运行 | {parent.run_id} |" in report
    assert "| 父运行残留 | " in report and "达修复轮次上限" in report
    assert parent.run_id in f"{state.get('parent_run_id')}"


def test_resumed_run_notes_the_source_in_the_start_event(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    engine = make_engine(fixed_clock, run_id_sequence)
    parent = engine.execute(file_request("divide_by_zero.py", max_iterations=0))
    version_dir = Path(str(read_state(parent)["current_version_dir"]))

    resumed = engine.execute(
        engine_mod.RunRequest(
            parent_run_id=parent.run_id,
            source_dir=version_dir,
        )
    )

    first = read_events(resumed)[0]
    assert first["event"] == "start" and f"续跑自 {parent.run_id}" in str(first["detail"])


def test_report_survives_a_corrupt_parent_record(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """父记录被清理 / 字段非法时少一行标注即可：报告本身必须照样产出（注释口径即契约）。"""
    engine = make_engine(fixed_clock, run_id_sequence)
    parent = engine.execute(file_request("divide_by_zero.py", max_iterations=0))
    parent_dir = parent.run_dir
    assert parent_dir is not None
    version_dir = Path(str(read_state(parent)["current_version_dir"]))
    (parent_dir / "state.json").write_text(
        json.dumps({"status": "residual", "review_round": "坏值", "findings": []}), encoding="utf-8"
    )

    resumed = engine.execute(
        engine_mod.RunRequest(
            parent_run_id=parent.run_id,
            source_dir=version_dir,
            max_iterations=0,
        )
    )

    assert resumed.report_written is True
    report = report_text(resumed)
    assert f"| 父运行 | {parent.run_id} |" in report
    assert "| 父运行残留 |" not in report  # 读不出父状态 → 少一行标注，不影响报告


# ---- M3.5：多文件版本与删除确认闸门（Design §4.2.2） ----


def two_file_encode(state: Mapping[str, Any]) -> dict[str, str]:
    """编码节点：产出两个文件（main.py 与 legacy.py）。"""
    return {
        "code_text": (
            "```python file=main.py\nprint(1)\n```\n\n```python file=legacy.py\nprint(2)\n```\n"
        )
    }


def deleting_fix(state: Mapping[str, Any], retry_feedback: str | None = None) -> dict[str, str]:
    """修复节点：只给出 main.py，并提出删除 legacy.py（Design §4.2.2 的"申请-确认"）。"""
    files = fakes.read_version_files(state)
    keep = {name: content for name, content in files.items() if name != "legacy.py"}
    payload = {
        "fixes": [{"finding_id": "f-1", "status": "fixed", "note": "删除冗余模块。"}],
        "deleted": ["legacy.py"],
    }
    fences = "\n\n".join(f"```python file={name}\n{content}\n```" for name, content in keep.items())
    return {"fix_text": f"{fences}\n\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```\n"}


def _deletion_run(
    fixed_clock: Callable[[], str],
    run_id_sequence: Callable[[], str],
    confirm: engine_mod.DeletionConfirmer | None,
) -> tuple[engine_mod.RunResult, Path]:
    nodes = {
        "encode": two_file_encode,
        "review": scripted_review([severe_finding(location="legacy.py:1")], []),
        "fix": deleting_fix,
    }
    engine = make_engine(fixed_clock, run_id_sequence, confirm_deletions=confirm, nodes=nodes)
    result = engine.execute(engine_mod.RunRequest(requirement="写两个文件并清理冗余"))
    assert result.run_dir is not None
    return result, result.run_dir / "workspace"


def test_confirmed_deletion_drops_the_file_from_the_new_version(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """确认删除：新版本目录不再包含该文件，摘要与报告标注"已确认删除"，退出码不受影响。"""
    result, workspace = _deletion_run(fixed_clock, run_id_sequence, lambda requests: True)

    assert result.exit_code == 0  # 删除是工程清洁度，不改变判定结论
    assert sorted(path.name for path in (workspace / "code_v2").iterdir()) == [
        "main.py",
        "manifest.json",
    ]
    fix_entry = next(entry for entry in read_state(result)["history"] if entry["node"] == "fix")
    assert "已确认删除：legacy.py" in fix_entry["summary"]
    payload = json.loads((workspace / "fixes_1.json").read_text(encoding="utf-8"))
    assert payload["deleted"] == {
        "proposed": ["legacy.py"],
        "confirmed": ["legacy.py"],
        "pending": [],
    }
    assert "已确认删除：legacy.py" in report_text(result)


@pytest.mark.parametrize("confirm", [lambda requests: False, None])
def test_unconfirmed_deletion_keeps_the_file_and_annotates(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str], confirm: Any
) -> None:
    """拒绝 / 非交互（无确认闸门）：文件保留（从上一版本带回），终端摘要与报告标注"待确认删除（未执行）"。"""
    result, workspace = _deletion_run(fixed_clock, run_id_sequence, confirm)

    assert result.exit_code == 0
    assert sorted(path.name for path in (workspace / "code_v2").iterdir()) == [
        "legacy.py",
        "main.py",
        "manifest.json",
    ]
    assert (workspace / "code_v2" / "legacy.py").read_text(encoding="utf-8") == "print(2)\n"
    fix_entry = next(entry for entry in read_state(result)["history"] if entry["node"] == "fix")
    assert "待确认删除（未执行）：legacy.py" in fix_entry["summary"]
    payload = json.loads((workspace / "fixes_1.json").read_text(encoding="utf-8"))
    assert payload["deleted"] == {
        "proposed": ["legacy.py"],
        "confirmed": [],
        "pending": ["legacy.py"],
    }
    assert "待确认删除（未执行）" in report_text(result)


def test_deletion_confirmer_failure_is_treated_as_unconfirmed(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """确认闸门自身抛错时按"未确认"处理：删除永不静默发生。"""

    def broken(requests: Any) -> bool:
        raise RuntimeError("终端不可用")

    result, workspace = _deletion_run(fixed_clock, run_id_sequence, broken)

    assert result.exit_code == 0
    assert (workspace / "code_v2" / "legacy.py").is_file()


def test_version_reading_ignores_pycache_artifacts(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """审查 Agent 执行核实时落下的 __pycache__ / *.pyc 不属于版本内容：

    字节码不是 UTF-8 文本，读进来会把版本复制（resume）变成运行失败，
    整版本指纹比较也会失真（真实缺陷回归）。
    """
    engine = make_engine(fixed_clock, run_id_sequence)
    parent = engine.execute(file_request("divide_by_zero.py", max_iterations=0))
    assert parent.run_dir is not None
    version_dir = Path(str(read_state(parent)["current_version_dir"]))
    cache = version_dir / "__pycache__"
    cache.mkdir()
    (cache / "divide_by_zero.cpython-313.pyc").write_bytes(b"\xf3\r\r\n\x00bytecode")

    resumed = engine.execute(
        engine_mod.RunRequest(parent_run_id=parent.run_id, source_dir=version_dir, max_iterations=0)
    )

    # 除零样例复审必然残留（假节点不改动），退出码 1 是预期结论；关键是版本复制成功
    assert resumed.exit_code == 1
    assert resumed.run_dir is not None
    copied = resumed.run_dir / "workspace" / "code_v1"
    assert sorted(path.name for path in copied.iterdir()) == ["divide_by_zero.py", "manifest.json"]


# ---- M3.5 自审修复的回归 ----


def test_source_text_only_request_is_recorded_as_file_input(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """只给 source_text 的程序化调用：输入描述与物化结果同口径（kind=file + code_v1/main.py）。"""
    result = make_engine(fixed_clock, run_id_sequence).execute(
        engine_mod.RunRequest(source_text="print(1)\n", max_iterations=0)
    )

    assert result.run_dir is not None
    state = read_state(result)
    assert state["input"]["kind"] == state_mod.INPUT_FILE
    assert (result.run_dir / "workspace" / "code_v1" / "main.py").read_text(
        encoding="utf-8"
    ) == "print(1)\n"


def test_deleting_an_absent_file_keeps_running(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """待删文件在上一版本目录里已不存在（记录与磁盘脱节）：跳过保留、按待确认标注，不升级为运行失败。"""

    def fix_removing_the_file(
        state: Mapping[str, Any], retry_feedback: str | None = None
    ) -> dict[str, str]:
        directory = Path(str(state["current_version_dir"]))
        (directory / "legacy.py").unlink()  # 模拟磁盘上的文件已不在
        files = fakes.read_version_files(state)
        keep = {name: content for name, content in files.items() if name != "legacy.py"}
        payload = {
            "fixes": [{"finding_id": "f-1", "status": "fixed", "note": "清理冗余模块。"}],
            "deleted": ["legacy.py"],
        }
        fences = "\n\n".join(
            f"```python file={name}\n{content}\n```" for name, content in keep.items()
        )
        return {
            "fix_text": f"{fences}\n\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```\n"
        }

    nodes = {
        "encode": two_file_encode,
        "review": scripted_review([severe_finding(location="legacy.py:1")], []),
        "fix": fix_removing_the_file,
    }
    result = make_engine(fixed_clock, run_id_sequence, nodes=nodes).execute(
        engine_mod.RunRequest(requirement="写两个文件并清理冗余")
    )

    assert result.exit_code == 0
    assert result.run_dir is not None
    assert sorted(path.name for path in (result.run_dir / "workspace" / "code_v2").iterdir()) == [
        "main.py",
        "manifest.json",
    ]


def test_version_read_follows_the_manifest(
    fixed_clock: Callable[[], str], run_id_sequence: Callable[[], str]
) -> None:
    """版本读取以 manifest 清单为准：执行核实落进版本目录的副产物不进版本内容。

    真实演示暴露：审查 Agent 跑被测程序时把数据文件写进了版本目录，resume 复制后
    变成"版本里的文件"，修复覆盖检查随之失败。
    """
    engine = make_engine(fixed_clock, run_id_sequence)
    parent = engine.execute(file_request("divide_by_zero.py", max_iterations=0))
    assert parent.run_dir is not None
    version_dir = Path(str(read_state(parent)["current_version_dir"]))
    (version_dir / "todos.json").write_text("[]\n", encoding="utf-8")  # 执行副产物
    cache = version_dir / "__pycache__"
    cache.mkdir()
    (cache / "x.cpython-313.pyc").write_bytes(b"\x00\x01")

    resumed = engine.execute(
        engine_mod.RunRequest(parent_run_id=parent.run_id, source_dir=version_dir, max_iterations=0)
    )

    assert resumed.run_dir is not None
    copied = resumed.run_dir / "workspace" / "code_v1"
    assert sorted(path.name for path in copied.iterdir()) == [
        "divide_by_zero.py",
        "manifest.json",
    ]
    resumed_state = read_state(resumed)
    assert resumed_state["version_files"] == ["divide_by_zero.py"]
