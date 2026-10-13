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


def test_node_start_renders_a_stage_banner_and_engine_end_stays_silent() -> None:
    """阶段面板：节点开始即打一条横幅（Agent 切换可辨）；收尾结论由命令面用 RunResult 渲染。"""
    renderer, out, err = make_renderer()

    renderer.on_event({"node": "review", "event": "start", "iteration": 1})
    renderer.on_event({"node": "engine", "event": "end", "detail": "success"})

    assert "── [审查] 第 1 轮 " in err.getvalue()
    assert out.getvalue() == ""


@pytest.mark.parametrize(
    ("node", "iteration", "needle"),
    [
        ("encode", 1, "[编码] 生成初始代码"),
        ("fix", 2, "[修复] 第 2 轮"),
        ("review_correctness", 1, "[审查·正确性] 第 1 轮"),
        ("merge_reviews", 1, "[汇合] 第 1 轮：合并双视角结论"),
    ],
)
def test_stage_banner_labels_every_agent(node: str, iteration: int, needle: str) -> None:
    renderer, _, err = make_renderer()

    renderer.on_event({"node": node, "event": "start", "iteration": iteration})

    assert needle in err.getvalue()


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
        round_number=1, version_label="code_v1/", findings=[finding()], summary="发现 1 项问题。"
    )

    text = out.getvalue()
    assert "【严重】code_v1.py:5 — 除零风险" in text
    assert "建议：加校验" in text
    assert "评述：发现 1 项问题。" in text
    assert err.getvalue() == ""


def test_review_listing_without_findings_omits_the_summary_line() -> None:
    renderer, out, _ = make_renderer()

    renderer.on_review(
        round_number=2, version_label="code_v2/", findings=[], summary="本轮未发现问题。"
    )

    assert out.getvalue().strip() == "（本轮无问题）"


def test_fix_dispositions_go_to_stdout() -> None:
    renderer, out, _ = make_renderer()

    renderer.on_fixes(
        round_number=1,
        fixes=[{"finding_id": "f-1", "status": "skipped", "note": "需业务语义判断"}],
        findings=[finding()],
    )

    assert "f-1 → 未采纳：需业务语义判断" in out.getvalue()


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


def test_runs_table_lists_the_designed_columns() -> None:
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
    assert lines[0].split() == [
        "运行号",
        "时间",
        "输入摘要",
        "状态",
        "修复轮数",
        "残留严重数",
        "模型",
    ]
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
                version_label="code_v1/",
                files=["main.py"],
                review={"findings": [finding()]},
                fixes={"fixes": [{"finding_id": "f-1", "status": "fixed", "note": "已修"}]},
            ),
            records.RoundReport(
                round_number=2, version_label="code_v2/", files=["main.py"], review={"findings": []}
            ),
        ),
        artifacts=(tmp_path / "report.md",),
    )
    out = StringIO()

    render.render_detail(detail, out=out)

    text = out.getvalue()
    assert "syntax_error.py → code_v1/syntax_error.py" in text
    assert "[状态] 干净通过" in text
    assert "上限 3（来源：命令行）" in text
    assert "时间线：" in text and "第 1 轮（code_v1.py）：1 严重" in text
    assert "逐轮审查：" in text
    assert "← 处置：已修复" in text
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


# ---- 流式正文与 --no-stream（Design §8.1）----


def make_stream_renderer(stream: bool) -> tuple[render.RunRenderer, StringIO, StringIO]:
    out, err = StringIO(), StringIO()
    return render.RunRenderer(stream=stream, out=out, err=err), out, err


def test_streamed_text_is_written_as_it_arrives() -> None:
    renderer, _, err = make_stream_renderer(True)

    renderer.on_delta("review", "第一段")
    assert err.getvalue() == "第一段"  # 未等到阶段结束就已落笔

    renderer.on_delta("review", "第二段")
    assert err.getvalue() == "第一段第二段"


def test_no_stream_buffers_text_until_the_stage_ends() -> None:
    renderer, _, err = make_stream_renderer(False)

    renderer.on_delta("review", "第一段")
    renderer.on_delta("review", "第二段")
    assert err.getvalue() == ""  # 整体输出：阶段结束前不出正文

    renderer.on_event(
        {"node": "review", "event": "end", "iteration": 1, "detail": "第 1 轮：0 严重"}
    )

    assert "第一段第二段" in err.getvalue()
    assert err.getvalue().index("第一段第二段") < err.getvalue().index("[审查] 第 1 轮")


def test_retried_text_is_not_rendered_again() -> None:
    """契约重试：重试后的正文不再渲染（前次已流出，重复渲染只会让人看到两段叙述）。"""
    renderer, _, err = make_stream_renderer(True)

    renderer.on_delta("review", "第一次的输出")
    renderer.on_event({"node": "review", "event": "retry", "iteration": 1, "detail": "输出缺围栏"})
    renderer.on_delta("review", "第二次的输出")
    renderer.on_event({"node": "review", "event": "end", "iteration": 1, "detail": "第 1 轮：干净"})

    text = err.getvalue()
    assert "第一次的输出" in text
    assert "第二次的输出" not in text
    assert "重试：输出缺围栏" in text


def test_buffer_drops_text_from_a_retried_stage_in_no_stream_mode() -> None:
    renderer, _, err = make_stream_renderer(False)

    renderer.on_delta("review", "第一次的输出")
    renderer.on_event({"node": "review", "event": "retry", "iteration": 1, "detail": "输出缺围栏"})
    renderer.on_event({"node": "review", "event": "end", "iteration": 1, "detail": "第 1 轮：干净"})

    assert "第一次的输出" not in err.getvalue()


def test_error_event_flushes_pending_text_before_the_report_line() -> None:
    renderer, _, err = make_stream_renderer(False)

    renderer.on_delta("encode", "正在写代码……")
    renderer.on_event({"node": "encode", "event": "error", "iteration": 1, "detail": "超时"})

    text = err.getvalue()
    assert "正在写代码……" in text
    assert text.index("正在写代码……") < text.index("[编码] 错误：超时")


# ---- crw review 的报告渲染（Design §6.4 / §8.1）----


def test_review_text_report_lists_every_file() -> None:
    """逐位对齐：失败的位（None）标"结论不可用"，成功位嵌结论原文。"""
    files = [("a.py", "内容一"), ("b.py", "内容二")]

    text = render.review_text_report(files, ["结论一", None])

    assert text.startswith("# crw 审查报告")
    assert "## a.py" in text and "结论一" in text
    assert "## b.py" in text and "[该文件审查失败，结论不可用]" in text


def test_review_json_report_renders_findings_and_skips_duplicate_summary() -> None:
    merged = {
        "files": [
            {
                "file": "a.py",
                "findings": [
                    {
                        "severity": "严重",
                        "location": "a.py:1",
                        "issue": "问题",
                        "suggestion": "建议",
                        "uncertain": True,
                    }
                ],
                "summary": "存在阻塞问题。",
            }
        ],
        "summary": "存在阻塞问题。",
    }

    text = render.review_json_report(merged)

    assert "- **【严重】** `a.py:1` — 问题（不确定）" in text
    assert "修改建议：建议" in text
    assert "## 总体评价" not in text  # 单文件不重复顶层 summary


def test_review_json_report_keeps_the_overall_section_when_nothing_was_reviewable() -> None:
    """files 为空时"总体评价"是唯一承载结论的地方（Design §8.1 的报告不能空壳）。"""
    text = render.review_json_report({"files": [], "summary": "没有可用的审查结果。"})

    assert "## 总体评价" in text
    assert "没有可用的审查结果。" in text


def test_review_json_report_keeps_the_overall_section_for_multiple_files() -> None:
    merged = {
        "files": [
            {"file": "a.py", "findings": [], "summary": "干净。"},
            {"file": "b.py", "findings": [], "summary": "也干净。"},
        ],
        "summary": "a.py：干净。；b.py：也干净。",
    }

    text = render.review_json_report(merged)

    assert "## 总体评价" in text
    assert "a.py：干净。；b.py：也干净。" in text


def test_streamed_text_without_trailing_newline_is_not_glued_to_the_stage_line() -> None:
    """流式正文末尾没有换行时，紧随其后的阶段行另起一行（实测模型正文有时不以换行收尾）。"""
    renderer, _, err = make_stream_renderer(True)

    renderer.on_delta("review", "结论写到一半就停了")
    renderer.on_event({"node": "review", "event": "end", "iteration": 1, "detail": "第 1 轮：干净"})

    lines = [line for line in err.getvalue().splitlines() if line.strip()]
    assert lines[0] == "结论写到一半就停了"
    assert lines[1] == "[审查] 第 1 轮：干净"


def test_streamed_text_ending_with_newline_keeps_a_single_newline() -> None:
    renderer, _, err = make_stream_renderer(True)

    renderer.on_delta("review", "整段正文\n")
    renderer.on_event({"node": "review", "event": "end", "iteration": 1, "detail": "第 1 轮：干净"})

    assert err.getvalue() == "整段正文\n[审查] 第 1 轮：干净\n"


# ---- M3：围栏抑制 / 分级着色 / 阶段排版 ----


def test_streamed_code_fence_is_not_dumped_to_the_terminal() -> None:
    """产出的代码不进终端（只留一行形状摘要）：生产代码整份回显会刷屏。"""
    renderer, _, err = make_renderer()

    renderer.on_event({"node": "encode", "event": "start", "iteration": 1})
    renderer.on_delta("encode", "实现说明。\n\n```python\n")
    renderer.on_delta("encode", "print('这一行不该出现在终端')\n" * 3)
    renderer.on_delta("encode", "```\n")
    renderer.on_event(
        {"node": "encode", "event": "end", "iteration": 1, "detail": "生成 code_v1.py"}
    )

    text = err.getvalue()
    assert "不该出现在终端" not in text
    assert "  ⋯ 已收到代码块（3 行，内容不在此显示）" in text
    assert "实现说明。" in text


def test_streamed_json_fence_is_not_dumped_to_the_terminal() -> None:
    renderer, _, err = make_renderer()

    renderer.on_delta("review", "我先读了文件。\n")
    renderer.on_delta(
        "review", '```json\n{"findings": [{"severity": "严重", "issue": "除零"}]}\n```\n'
    )

    text = err.getvalue()
    assert "findings" not in text
    assert "  ⋯ 已收到 JSON 结论（1 行，内容不在此显示）" in text
    assert "我先读了文件。" in text


def test_fence_marker_split_across_deltas_is_still_suppressed() -> None:
    """标记可能被切在任意位置（`` ` `` / ` `` ` / info 串分片）：仍不得漏出代码。"""
    renderer, _, err = make_renderer()

    for piece in ("说明\n", "`", "``", "python", "\n", "SECRET = 1\n", "`", "``", "`\n", "结束。"):
        renderer.on_delta("encode", piece)
    renderer.on_event(
        {"node": "encode", "event": "end", "iteration": 1, "detail": "生成 code_v1.py"}
    )

    text = err.getvalue()
    assert "SECRET" not in text
    assert "说明" in text and "结束。" in text


def test_code_fence_is_suppressed_in_no_stream_mode_too() -> None:
    renderer, _, err = make_renderer()
    renderer.stream = False

    renderer.on_delta("fix", "修复说明。\n```python\nHIDDEN = 1\n```\n")
    renderer.on_event({"node": "fix", "event": "end", "iteration": 1, "detail": "修复 1 项"})

    text = err.getvalue()
    assert "HIDDEN" not in text
    assert "修复说明。" in text
    assert "已收到代码块（1 行，内容不在此显示）" in text


def test_unterminated_fence_still_gets_a_summary_line() -> None:
    """围栏未闭合（模型截断 / 乱输出）：同样不泄漏内容（完整性由契约层判定，不由渲染层断言）。"""
    renderer, _, err = make_renderer()

    renderer.on_delta("encode", "```python\nprint(1)\n")
    renderer.on_event(
        {"node": "encode", "event": "end", "iteration": 1, "detail": "生成 code_v1.py"}
    )

    text = err.getvalue()
    assert "print(1)" not in text
    assert "已收到代码块（1 行，内容不在此显示）" in text


def test_prose_is_streamed_incrementally() -> None:
    """围栏抑制不得退化成整段缓存：普通正文仍逐片写出。"""
    renderer, _, err = make_renderer()

    renderer.on_delta("review", "第一片")
    first = err.getvalue()
    renderer.on_delta("review", "，第二片")

    assert first == "第一片"
    assert err.getvalue() == "第一片，第二片"


def test_severity_and_stage_are_colored_when_enabled() -> None:
    out, err = StringIO(), StringIO()
    renderer = render.RunRenderer(out=out, err=err, color=True)

    renderer.on_event({"node": "review", "event": "start", "iteration": 1})
    renderer.on_review(
        round_number=1,
        version_label="code_v1/",
        findings=[finding(), finding("建议", id="f-2", location="code_v1.py:9")],
        summary="",
    )

    assert "\x1b[" in err.getvalue()  # 阶段横幅着色
    assert "\x1b[1;31m" in out.getvalue()  # 【严重】红
    assert "\x1b[2m" in out.getvalue()  # 【建议】暗


def test_colors_are_off_by_default_for_non_tty_streams() -> None:
    renderer, out, err = make_renderer()

    renderer.on_event({"node": "review", "event": "start", "iteration": 1})
    renderer.on_review(round_number=1, version_label="code_v1/", findings=[finding()], summary="")

    assert "\x1b[" not in err.getvalue()
    assert "\x1b[" not in out.getvalue()


def test_stage_banner_is_separated_by_a_blank_line() -> None:
    renderer, _, err = make_renderer()

    renderer.on_event({"node": "encode", "event": "start", "iteration": 1})

    assert err.getvalue().startswith("\n── [编码]")


def test_parallel_view_listing_shows_a_counts_line() -> None:
    renderer, out, _ = make_renderer()

    renderer.on_review(
        round_number=1,
        version_label="code_v1/",
        findings=[finding()],
        summary="",
        view="正确性",
    )

    assert "· 正确性视角：1 严重 / 0 一般 / 0 建议" in out.getvalue()
    assert "【严重】" not in out.getvalue()  # 分支只报规模，合并结果才展开逐条


def test_degraded_detail_renders_log_summary_and_artifacts(tmp_path: Path) -> None:
    """state.json 损坏：降级展示（日志时间线 + 产物清单 + 逐轮审查），并说明数据来源。"""
    detail = records.RunDetail(
        summary=summary(status=state_mod.STATUS_SUCCESS),
        state={},
        events=(
            {"ts": "2026-10-12T10:00:00+08:00", "node": "review", "event": "start", "iteration": 1},
            {
                "ts": "2026-10-12T10:00:04+08:00",
                "node": "review",
                "event": "tool",
                "iteration": 1,
                "detail": 'read_file({"path": "code_v1.py"})',
            },
            {
                "ts": "2026-10-12T10:00:09+08:00",
                "node": "review",
                "event": "end",
                "iteration": 1,
                "detail": "第 1 轮（code_v1.py）：1 严重 / 0 一般 / 0 建议",
            },
            {
                "ts": "2026-10-12T10:00:09+08:00",
                "node": "engine",
                "event": "end",
                "detail": "success",
            },
        ),
        rounds=(
            records.RoundReport(
                round_number=1,
                version_label="code_v1/",
                files=["main.py"],
                review={"findings": [finding()]},
            ),
        ),
        artifacts=(tmp_path / "state.json",),
        degraded=True,
        degraded_reason="state.json 不可读或不是 JSON 对象（摘要取自 log.jsonl 与产物清单）",
    )
    out = StringIO()

    render.render_detail(detail, out=out)

    text = out.getvalue()
    assert "（记录降级展示）" in text
    assert "[来源] 以下摘要取自 log.jsonl 与产物清单" in text
    assert "2026-10-12T10:00:00+08:00 → 2026-10-12T10:00:09+08:00（取自日志）" in text
    assert "9000" in text  # 由 start / end 事件时间戳配对的耗时
    assert "read_file(" in text  # 工具调用明细
    assert "第 1 轮（code_v1.py）：1 严重" in text
    assert "state.json" in text


def test_detail_marks_not_converged_rounds(tmp_path: Path) -> None:
    """未收敛标注（Design §4.4）：与上一轮问题集合一致时在详情与报告里如实标出。"""
    detail = records.RunDetail(
        summary=summary(status=state_mod.STATUS_RESIDUAL),
        state={
            "run_id": "r-20261012-0001",
            "input": {"kind": "requirement", "requirement": "写一个计算器"},
            "status": state_mod.STATUS_RESIDUAL,
            "history": [],
        },
        events=(),
        rounds=(
            records.RoundReport(
                round_number=1,
                version_label="code_v1/",
                files=["main.py"],
                review={"findings": [finding()]},
            ),
            records.RoundReport(
                round_number=2,
                version_label="code_v2/",
                files=["main.py"],
                review={"findings": [finding()]},
            ),
        ),
        artifacts=(),
    )
    out = StringIO()

    render.render_detail(detail, out=out)

    assert "未收敛：本轮问题集合与上一轮完全相同" in out.getvalue()


def test_parallel_branches_do_not_interleave_their_prose() -> None:
    """并行分支同时到达的正文按节点缓存、各在自己的阶段结束时整块输出（不逐句交错）。"""
    renderer, _, err = make_renderer()

    renderer.on_event({"node": "review_correctness", "event": "start", "iteration": 1})
    renderer.on_event({"node": "review_security", "event": "start", "iteration": 1})
    renderer.on_delta("review_correctness", "正确性甲。")
    renderer.on_delta("review_security", "安全性甲。")
    renderer.on_delta("review_correctness", "正确性乙。")
    renderer.on_delta("review_security", "安全性乙。")
    renderer.on_event(
        {"node": "review_security", "event": "end", "iteration": 1, "detail": "安全性：0 / 0 / 0"}
    )
    renderer.on_event(
        {
            "node": "review_correctness",
            "event": "end",
            "iteration": 1,
            "detail": "正确性：1 / 0 / 0",
        }
    )

    text = err.getvalue()
    assert "正确性甲。正确性乙。" in text
    assert "安全性甲。安全性乙。" in text
    # 块的先后与阶段结束顺序一致（先前结束的先整块输出）
    assert text.index("安全性甲。安全性乙。") < text.index("正确性甲。正确性乙。")


def test_fence_closer_without_a_trailing_newline_is_recognized() -> None:
    """末行无换行是常态：闭合标记照样算闭合（否则每次收尾都错报"未闭合"）。"""
    renderer, _, err = make_renderer()

    renderer.on_delta("encode", "```python\nprint(1)\n```")
    renderer.on_event(
        {"node": "encode", "event": "end", "iteration": 1, "detail": "生成 code_v1.py"}
    )

    text = err.getvalue()
    assert "print(1)" not in text
    assert "已收到代码块（1 行，内容不在此显示）" in text
    assert "未闭合" not in text


class _BrokenStream(StringIO):
    """写入即抛 OSError 的流（模拟下游关闭管道：``crw show | head``）。"""

    def write(self, text: str) -> int:  # type: ignore[override]
        raise OSError(22, "Invalid argument")


def test_render_detail_survives_a_closed_output_stream() -> None:
    """展示层故障不改变命令结论（Design §4.6 的同口径）：管道被下游关掉不报成运行失败。"""
    detail = records.RunDetail(
        summary=summary(),
        state={"run_id": "r-20261012-0001", "status": state_mod.STATUS_SUCCESS, "history": []},
        events=(),
        rounds=(),
        artifacts=(),
    )

    render.render_detail(detail, out=_BrokenStream())  # 不抛出
    render.render_result(
        result_for(state_mod.STATUS_SUCCESS, exit_code=0), out=_BrokenStream(), err=_BrokenStream()
    )
    render.render_runs([summary()], out=_BrokenStream())


def test_render_does_not_swallow_programming_errors() -> None:
    """渲染器自身的缺陷（非 OSError）照常暴露，不被"管道容错"掩盖。"""

    class _Boom(StringIO):
        def write(self, text: str) -> int:  # type: ignore[override]
            raise RuntimeError("渲染器缺陷")

    with pytest.raises(RuntimeError):
        render.render_runs([summary()], out=_Boom())


def test_branch_prose_is_buffered_even_before_the_second_stage_starts() -> None:
    """分支节点一律按节点缓存（不依赖"已开启阶段数"的时序启发式）：只开一个阶段也不抢先流出去。"""
    renderer, _, err = make_renderer()

    renderer.on_event({"node": "review_correctness", "event": "start", "iteration": 1})
    renderer.on_delta("review_correctness", "甲句。")
    first = err.getvalue()
    renderer.on_event(
        {"node": "review_correctness", "event": "end", "iteration": 1, "detail": "正确性：0/0/0"}
    )

    assert "甲句。" not in first  # 阶段结束前不写出（否则并行时与另一分支交错）
    assert "甲句。" in err.getvalue()


def test_fix_status_labels_cover_the_contract_enum() -> None:
    """处置标签表的键集必须与契约层的状态枚举一致（新增状态时靠本用例拦住漂移）。"""
    from crw.workflow import contracts as contracts_mod

    assert set(render._STATUS_LABEL) == set(contracts_mod.FIX_STATUSES)
    assert set(render._SEVERITY_STYLE) == set(contracts_mod.SEVERITIES)


def test_engine_level_error_events_are_rendered() -> None:
    """引擎级 error / interrupt 事件照常渲染（只有 start / end 由命令面接管结论）。"""
    renderer, _, err = make_renderer()

    renderer.on_event({"node": "engine", "event": "error", "detail": "state.json 写入失败"})

    assert "错误：state.json 写入失败" in err.getvalue()


def test_crlf_fence_markers_are_suppressed_too() -> None:
    """CRLF 正文里的围栏（含非 python/json 标签）同样被抑制，不漏出正文。"""
    renderer, _, err = make_renderer()

    renderer.on_delta("review", "说明。\r\n```text\r\n秘而不宣\r\n```\r\n收尾。")

    text = err.getvalue()
    assert "秘而不宣" not in text
    assert "已收到 text 围栏" in text
