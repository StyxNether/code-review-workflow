"""终端渲染：输出通道分流、阶段行、清单分区与列表 / 详情（Design §8.1、§8.4）。"""

from __future__ import annotations

import builtins
from io import StringIO
from pathlib import Path

import pytest

from crw import render
from crw.workflow import engine as engine_mod
from crw.workflow import records
from crw.workflow import state as state_mod


def finding(severity: str = "严重", **extra: object) -> dict:
    return {
        "id": "f-1",
        "severity": severity,
        "location": "code_v1.py:5",
        "issue": "除零风险",
        "suggestion": "加校验",
        **extra,
    }


def make_renderer() -> tuple[render.RunRenderer, StringIO, StringIO]:
    out, err = StringIO(), StringIO()
    return render.RunRenderer(out=out, err=err), out, err


# ---- 宽度与对齐 ----


@pytest.mark.parametrize(
    ("text", "expected"), [("abc", 3), ("中文", 4), ("中文abc", 7), ("", 0), ("，。", 4)]
)
def test_display_width_counts_east_asian_chars_as_two(text: str, expected: int) -> None:
    assert render.display_width(text) == expected


def test_pad_aligns_by_display_width() -> None:
    assert render.pad("中文", 6) == "中文  "
    assert render.pad("abcdef", 3) == "abcdef"


# ---- 事件与清单 ----


def test_engine_start_renders_header_and_input_on_stderr_only() -> None:
    renderer, out, err = make_renderer()

    renderer.on_event(
        {
            "node": "engine",
            "event": "start",
            "detail": "需求：写一个计算器；修复轮次上限 3（来源：内置默认）",
            "run_id": "r-20261012-0001",
        }
    )

    assert "r-20261012-0001" in err.getvalue()
    assert "需求：写一个计算器" in err.getvalue()
    assert out.getvalue() == ""


def test_node_end_renders_a_stage_line_on_stderr() -> None:
    renderer, out, err = make_renderer()

    renderer.on_event(
        {
            "node": "review",
            "event": "end",
            "iteration": 1,
            "detail": "第 1 轮（code_v1.py）：1 严重 / 0 一般 / 0 建议",
        }
    )

    assert "[审查] 第 1 轮（code_v1.py）：1 严重" in err.getvalue()
    assert out.getvalue() == ""


def test_node_start_and_engine_end_stay_silent() -> None:
    """M1 简版：阶段标题由结束事件给出；收尾结论由命令面用 RunResult 渲染。"""
    renderer, out, err = make_renderer()

    renderer.on_event({"node": "review", "event": "start", "iteration": 1})
    renderer.on_event({"node": "engine", "event": "end", "detail": "success"})

    assert out.getvalue() == ""
    assert err.getvalue() == ""


@pytest.mark.parametrize(
    ("event", "needle"),
    [
        ({"node": "review", "event": "retry", "detail": "契约校验未通过"}, "重试"),
        ({"node": "fix", "event": "error", "detail": "模型调用失败"}, "[修复] 错误：模型调用失败"),
        ({"node": "review", "event": "interrupt", "detail": "用户中断（Ctrl+C）"}, "用户中断"),
        ({"node": "review", "event": "tool", "detail": "read_file(code_v1.py)"}, "工具："),
    ],
)
def test_retry_error_interrupt_and_tool_go_to_stderr(event: dict, needle: str) -> None:
    renderer, out, err = make_renderer()

    renderer.on_event(event)

    assert needle in err.getvalue()
    assert out.getvalue() == ""


def test_review_listing_goes_to_stdout_with_suggestions() -> None:
    renderer, out, err = make_renderer()

    renderer.on_review(
        round_number=1, file_name="code_v1.py", findings=[finding()], summary="发现 1 项问题。"
    )

    text = out.getvalue()
    assert "【严重】code_v1.py:5 — 除零风险" in text
    assert "建议：加校验" in text
    assert "评述：发现 1 项问题。" in text
    assert err.getvalue() == ""


def test_review_listing_without_findings_omits_the_summary_line() -> None:
    renderer, out, _ = make_renderer()

    renderer.on_review(
        round_number=2, file_name="code_v2.py", findings=[], summary="本轮未发现问题。"
    )

    assert out.getvalue().strip() == "（本轮无问题）"


def test_fix_dispositions_go_to_stdout() -> None:
    renderer, out, _ = make_renderer()

    renderer.on_fixes(
        round_number=1,
        fixes=[{"finding_id": "f-1", "status": "skipped", "note": "需业务语义判断"}],
        findings=[finding()],
    )

    assert "【修复】f-1 → skipped：需业务语义判断" in out.getvalue()


# ---- 收尾结论 ----


def result_for(
    status: str, *, exit_code: int, run_dir: Path | None = None, **kwargs: object
) -> engine_mod.RunResult:
    defaults: dict = {
        "run_id": "r-20261012-0001",
        "run_dir": run_dir or Path("C:/runs/r-20261012-0001"),
        "status": status,
        "exit_code": exit_code,
        "review_round": 2,
        "fix_rounds": 1,
        "clean": exit_code == 0,
        "findings": (),
        "error_node": None,
        "error_message": None,
        "interrupted": False,
        "state": {},
        "report_written": True,
    }
    defaults.update(kwargs)
    return engine_mod.RunResult(**defaults)


def test_success_conclusion_goes_to_stdout() -> None:
    out, err = StringIO(), StringIO()

    render.render_result(result_for(state_mod.STATUS_SUCCESS, exit_code=0), out=out, err=err)

    assert "完成：干净通过（共 2 轮审查 / 1 轮修复）" in out.getvalue()
    assert "运行记录" in out.getvalue()
    assert err.getvalue() == ""


def test_residual_conclusion_counts_severe_findings() -> None:
    out, err = StringIO(), StringIO()

    render.render_result(
        result_for(state_mod.STATUS_RESIDUAL, exit_code=1, findings=(finding(), finding("建议"))),
        out=out,
        err=err,
    )

    assert "达修复轮次上限，仍有 1 项【严重】残留" in out.getvalue()
    assert err.getvalue() == ""


def test_strict_residual_conclusion_counts_blocking_findings() -> None:
    """--strict 下残留可能全部是【一般】：文案与判定口径一致（不得写"0 项【严重】"）。"""
    moderate = {"id": "f-1", "severity": "一般", "location": "code_v1.py:3", "issue": "静默吞异常"}
    out, err = StringIO(), StringIO()

    render.render_result(
        result_for(
            state_mod.STATUS_RESIDUAL, exit_code=1, findings=(moderate,), state={"strict": True}
        ),
        out=out,
        err=err,
    )

    assert "仍有 1 项阻塞问题（--strict：含【一般】）残留" in out.getvalue()
    assert "0 项" not in out.getvalue()


def test_interrupted_status_label_is_live_for_session_records() -> None:
    """STATUS_INTERRUPTED 属会话轮次口径（Design §8.2）：标签与结论分支须可渲染。"""
    assert engine_mod.status_label(state_mod.STATUS_INTERRUPTED) == "已中断"
    out, err = StringIO(), StringIO()

    render.render_result(result_for(state_mod.STATUS_INTERRUPTED, exit_code=2), out=out, err=err)

    assert "已中断" in err.getvalue()


@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        ({"error_node": "review", "error_message": "RuntimeError: 调用失败"}, "运行失败 — [审查]"),
        (
            {
                "error_node": "interrupt",
                "error_message": "运行被用户中断（Ctrl+C）",
                "interrupted": True,
            },
            "已中断",
        ),
        ({"error_node": "engine", "error_message": "state.json 写入失败"}, "运行失败 — state.json"),
    ],
)
def test_failures_report_readably_on_stderr(kwargs: dict, needle: str) -> None:
    out, err = StringIO(), StringIO()

    render.render_result(
        result_for(state_mod.STATUS_FAILED, exit_code=2, **kwargs), out=out, err=err
    )

    assert "crw run：" in err.getvalue()
    assert needle in err.getvalue()
    assert "运行记录" in err.getvalue()
    assert out.getvalue() == ""


# ---- 列表与详情 ----


def summary(**kwargs: object) -> records.RunSummary:
    defaults: dict = {
        "run_id": "r-20261012-0001",
        "created_at": "2026-10-12T10:00:00+08:00",
        "input_summary": "需求：写一个计算器",
        "status": state_mod.STATUS_SUCCESS,
        "fix_rounds": 1,
        "residual_severe": 0,
        "path": Path("C:/runs/r-20261012-0001"),
        "stale": False,
    }
    defaults.update(kwargs)
    return records.RunSummary(**defaults)


def test_runs_table_lists_the_six_designed_columns() -> None:
    out = StringIO()

    render.render_runs(
        [
            summary(),
            summary(
                run_id="r-20261012-0002", status="residual", fix_rounds=None, residual_severe=None
            ),
        ],
        out=out,
    )

    lines = out.getvalue().splitlines()
    assert lines[0].split() == ["运行号", "时间", "输入摘要", "状态", "修复轮数", "残留严重数"]
    assert lines[1].startswith("---")
    assert "r-20261012-0001" in lines[2]
    assert "达限残留" in lines[3]
    assert "?" in lines[3]  # 降级派生不可得时以 ? 标注


def test_runs_table_without_records_says_so() -> None:
    out = StringIO()

    render.render_runs([], out=out)

    assert "暂无运行记录" in out.getvalue()


def test_detail_renders_summary_timeline_rounds_and_artifacts(tmp_path: Path) -> None:
    detail = records.RunDetail(
        summary=summary(),
        state={
            "run_id": "r-20261012-0001",
            "input": {
                "kind": "file",
                "source_path": "tests/fixtures/buggy_samples/syntax_error.py",
            },
            "model": "内置假节点（未接模型）",
            "status": state_mod.STATUS_SUCCESS,
            "review_round": 2,
            "fix_rounds": 1,
            "max_iterations": 3,
            "max_iterations_source": state_mod.SOURCE_CLI,
            "created_at": "2026-10-12T10:00:00+08:00",
            "finished_at": "2026-10-12T10:00:01+08:00",
            "history": [
                {
                    "node": "review",
                    "round": 1,
                    "started_at": "2026-10-12T10:00:00+08:00",
                    "duration_ms": 12,
                    "summary": "第 1 轮（code_v1.py）：1 严重 / 0 一般 / 0 建议",
                }
            ],
        },
        events=(),
        rounds=(
            records.RoundReport(
                round_number=1,
                file_name="code_v1.py",
                review={"findings": [finding()]},
                fixes={"fixes": [{"finding_id": "f-1", "status": "fixed", "note": "已修"}]},
            ),
            records.RoundReport(round_number=2, file_name="code_v2.py", review={"findings": []}),
        ),
        artifacts=(tmp_path / "report.md",),
    )
    out = StringIO()

    render.render_detail(detail, out=out)

    text = out.getvalue()
    assert "syntax_error.py → code_v1.py" in text
    assert "[状态] 干净通过" in text
    assert "上限 3（来源：命令行）" in text
    assert "时间线：" in text and "第 1 轮（code_v1.py）：1 严重" in text
    assert "逐轮审查：" in text
    assert "← 处置：fixed" in text
    assert "产物路径：" in text and "report.md" in text


def test_detail_marks_interrupted_runs(tmp_path: Path) -> None:
    detail = records.RunDetail(
        summary=summary(status=state_mod.STATUS_FAILED),
        state={
            "run_id": "r-20261012-0001",
            "input": {"kind": "requirement", "requirement": "写一个计算器"},
            "status": state_mod.STATUS_FAILED,
            "error": {"node": "interrupt", "message": "运行被用户中断（Ctrl+C）。"},
            "history": [],
        },
        events=(),
        rounds=(),
        artifacts=(tmp_path / "state.json",),
    )
    out = StringIO()

    render.render_detail(detail, out=out)

    assert "[已中断] interrupt：运行被用户中断（Ctrl+C）。" in out.getvalue()


# ---- 交互输入 ----


def test_ask_user_prompts_on_stderr_and_strips_input(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """提示语写 stderr（stdout 只留结论），输入去空白。"""
    monkeypatch.setattr(builtins, "input", lambda: "  2  ")

    assert render.ask_user("选择：") == "2"

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "选择："


@pytest.mark.parametrize("error", [EOFError, KeyboardInterrupt, OSError, RuntimeError])
def test_ask_user_treats_input_failures_as_cancel(
    monkeypatch: pytest.MonkeyPatch, error: type[BaseException]
) -> None:
    """stdin 被捕获 / 关闭 / 中断一律按取消处理，不以 traceback 退出。"""

    def broken(*args: object) -> str:
        raise error

    monkeypatch.setattr(builtins, "input", broken)

    assert render.ask_user("选择：") is None


def test_runs_table_flags_stale_running_records() -> None:
    """陈旧 running 标注"可能已中断"（Design §8.2）。"""
    out = StringIO()

    render.render_runs(
        [summary(status=state_mod.STATUS_RUNNING, stale=True, fix_rounds=0, residual_severe=0)],
        out=out,
    )

    assert "运行中（可能已中断）" in out.getvalue()


def test_detail_flags_stale_running_records() -> None:
    detail = records.RunDetail(
        summary=summary(status=state_mod.STATUS_RUNNING, stale=True),
        state={"run_id": "r-20261012-0001", "status": state_mod.STATUS_RUNNING, "history": []},
        events=(),
        rounds=(),
        artifacts=(),
    )
    out = StringIO()

    render.render_detail(detail, out=out)

    assert "[状态] 运行中（可能已中断）" in out.getvalue()


def test_interrupted_run_is_not_promised_a_report() -> None:
    """中断收尾不写报告（Design §4.6 的写盘序列）：提示语不得指向不存在的 report.md。"""
    out, err = StringIO(), StringIO()

    render.render_result(
        result_for(
            state_mod.STATUS_FAILED,
            exit_code=2,
            error_node="interrupt",
            error_message="运行被用户中断（Ctrl+C）",
            interrupted=True,
            report_written=False,
        ),
        out=out,
        err=err,
    )

    assert "已中断，未生成报告" in err.getvalue()
    assert "state.json / log.jsonl" in err.getvalue()
    assert "报告 report.md）" not in err.getvalue()


def test_completed_run_points_to_the_report() -> None:
    out, err = StringIO(), StringIO()

    render.render_result(
        result_for(state_mod.STATUS_SUCCESS, exit_code=0, report_written=True), out=out, err=err
    )

    assert "(报告 report.md)" in out.getvalue().replace("（", "(").replace("）", ")")
