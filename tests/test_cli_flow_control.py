"""命令面的流程控制面：续跑（resume）/ 图导出（graph）/ 降级展示（show）/ 并行审查开关。

与 test_cli.py 的分工：那边覆盖参数面与 run / runs / show / review / config 的主路径，
这里覆盖 resume / graph / show 三条命令面与两个开关，两者都用假模型驱动真实装配（无真实网络）。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import fakes
import pytest
from support import sample_path

from crw import config as config_mod
from crw import rounds as rounds_mod
from crw.cli import main
from crw.workflow import graph as graph_mod
from crw.workflow import records
from crw.workflow import state as state_mod


def write_parent_record(
    run_id: str = "r-20261012-0001",
    *,
    code: str = "print(1)\n",
    max_iterations: int = 2,
    severe_only: bool = False,
    parallel_review: bool = False,
    with_code: bool = True,
) -> Path:
    """在测试数据目录里造一条父运行记录（state.json + workspace/code_v2/）。"""
    run_dir = records.create_run_dir(run_id)
    workspace = run_dir / "workspace"
    current_version_dir = ""
    if with_code:
        version_dir = workspace / "code_v2"
        version_dir.mkdir()
        (version_dir / "main.py").write_text(code, encoding="utf-8")
        current_version_dir = str(version_dir)
    state = {
        "run_id": run_id,
        "parent_run_id": None,
        "input": {"kind": "requirement", "requirement": "写一个命令行计算器"},
        "workspace": str(workspace),
        "review_round": 2,
        "fix_rounds": 1,
        "max_iterations": max_iterations,
        "max_iterations_source": state_mod.SOURCE_CLI,
        "rounds_mode": rounds_mod.ROUNDS_UP_TO,
        "parallel_review": parallel_review,
        "session_id": None,
        "round_index": None,
        "current_version_dir": current_version_dir,
        "version_files": ["main.py"] if with_code else [],
        "findings": [fakes.finding(location="main.py:1")],
        "clean": False,
        "severe_only": severe_only,
        "history": [],
        "status": state_mod.STATUS_RESIDUAL,
        "next_action": state_mod.NEXT_FIX_AGAIN,
        "error": None,
        "model": "DeepSeek/deepseek-flash",
        "created_at": "2026-10-12T09:00:00+08:00",
        "finished_at": "2026-10-12T09:01:00+08:00",
    }
    records.atomic_write_text(run_dir / "state.json", state_mod.dumps(state))
    return run_dir


def test_resume_inherits_parent_rounds_and_records_parent_run_id(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """续跑：父运行最新版本目录复制为新运行 code_v1/、记 parent_run_id、轮次来源标 parent。"""
    write_parent_record(max_iterations=2)
    configured_provider()
    fake_chat_model.load(
        [
            fakes.ai_tool_call("read_file", path="code_v1/main.py"),
            fakes.ai_text(fakes.review_fence([], "已读完文件，未发现阻塞问题。")),
        ]
    )

    assert main(["resume", "r-20261012-0001"]) == 0
    capsys.readouterr()

    child = next(item for item in records.list_runs() if item.run_id != "r-20261012-0001")
    state = records.read_state(child.path)
    assert state is not None
    assert state["parent_run_id"] == "r-20261012-0001"
    assert state["max_iterations"] == 2  # 未显式给出 → 继承父运行
    assert state["max_iterations_source"] == state_mod.SOURCE_PARENT
    assert state["input"]["kind"] == state_mod.INPUT_FILE
    assert (child.path / "workspace" / "code_v1" / "main.py").read_text(
        encoding="utf-8"
    ) == "print(1)\n"
    report = (child.path / "report.md").read_text(encoding="utf-8")
    assert "| 父运行 | r-20261012-0001 |" in report
    assert "| 父运行残留 |" in report


def test_resume_inherits_severe_only_and_explicit_rounds_win(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """父运行的判定标准一并继承；命令行显式给出轮次时以命令行为准（来源标 cli）。"""
    write_parent_record(max_iterations=2, severe_only=True)
    configured_provider()
    fake_chat_model.load(
        [
            fakes.ai_tool_call("read_file", path="code_v1/main.py"),
            fakes.ai_text(fakes.review_fence([], "已读完文件。")),
        ]
    )

    assert main(["resume", "r-20261012-0001", "--max-iter", "0"]) == 0
    capsys.readouterr()

    child = next(item for item in records.list_runs() if item.run_id != "r-20261012-0001")
    state = records.read_state(child.path)
    assert state is not None
    assert state["max_iterations"] == 0
    assert state["max_iterations_source"] == state_mod.SOURCE_CLI
    assert state["severe_only"] is True


def test_resume_of_a_pre_severe_only_record_judges_strictly(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """旧记录没有 ``severe_only`` 字段：按缺省 False（加严）解释——口径统一，不是缺陷。

    同一条记录续跑时判据比它当初更严，测试把这条预期钉住：
    不得因为字段缺席就退回"只数【严重】"的旧默认。
    """
    run_dir = write_parent_record()
    legacy = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    legacy.pop("severe_only")
    (run_dir / "state.json").write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
    configured_provider()
    fake_chat_model.load(
        [
            fakes.ai_tool_call("read_file", path="code_v1/main.py"),
            fakes.ai_text(fakes.review_fence([], "已读完文件。")),
        ]
    )

    assert main(["resume", "r-20261012-0001"]) == 0
    capsys.readouterr()

    child = next(item for item in records.list_runs() if item.run_id != "r-20261012-0001")
    state = records.read_state(child.path)
    assert state is not None
    assert state["severe_only"] is False


def test_resume_can_end_at_the_round_limit_with_exit_code_one(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """续跑的"达限残留"路径（退出码 1）在命令面可达，而不只靠 run 侧同函数推定。"""
    write_parent_record(max_iterations=1)
    configured_provider()
    fake_chat_model.load(
        [
            fakes.ai_tool_call("read_file", path="code_v1/main.py"),
            fakes.ai_text(
                fakes.review_fence(
                    [fakes.finding(location="main.py:1")], "仍有除零风险。", files=["main.py"]
                )
            ),
        ]
    )

    assert main(["resume", "r-20261012-0001", "--max-iter", "0"]) == 1

    captured = capsys.readouterr()
    assert "达修复轮次上限" in captured.out


def test_resume_without_a_code_version_is_readable(
    configured_provider: Callable[[], dict[str, object]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """父运行没有 current_code（建码前就失败）：给可读原因并退出 2，不新建运行。"""
    write_parent_record(with_code=False)
    configured_provider()

    assert main(["resume", "r-20261012-0001"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "没有可续跑的代码版本" in captured.err
    assert len(records.list_runs()) == 1


def test_resume_of_a_corrupt_parent_record_is_readable(
    configured_provider: Callable[[], dict[str, object]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = write_parent_record()
    records.atomic_write_text(run_dir / "state.json", "{损坏")
    configured_provider()

    assert main(["resume", "r-20261012-0001"]) == 2

    assert "记录已损坏" in capsys.readouterr().err


def test_graph_prints_to_stdout_and_writes_a_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["graph"]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("flowchart TD")
    assert graph_mod.mermaid_flowchart() == printed

    target = tmp_path / "graph.mmd"
    assert main(["graph", "-o", str(target)]) == 0
    assert target.read_text(encoding="utf-8") == printed


def test_graph_exports_the_parallel_shape_on_request(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["graph", "--parallel-review"]) == 0

    assert capsys.readouterr().out == graph_mod.mermaid_flowchart(parallel_review=True)


def test_show_degrades_and_exits_2_when_state_is_corrupt(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """记录损坏时降级展示 + 退出码 2：人读拿得到信息，脚本消费能察觉异常。"""
    run_dir = write_parent_record("r-20261012-0002")
    records.append_log_event(
        run_dir / "log.jsonl",
        {"ts": "2026-10-12T09:00:00+08:00", "node": "review", "event": "end", "detail": "已完成"},
    )
    records.atomic_write_text(run_dir / "state.json", "{损坏")

    assert main(["show", "r-20261012-0002"]) == 2

    captured = capsys.readouterr()
    assert "（记录降级展示）" in captured.out
    assert "取自 log.jsonl 与产物清单" in captured.out
    assert "记录损坏，已降级展示" in captured.err


def test_run_parallel_review_records_the_shape_and_branch_artifacts(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--parallel-review``：两视角各审一轮、分支结论各落一份、状态记下审查形态。"""
    configured_provider()
    # 两个分支并发执行，按调用顺序回放会把响应错配给另一个分支 → 按分支标记路由
    fake_chat_model.route(
        {
            "你只负责【正确性】视角": [
                fakes.ai_tool_call("read_file", path="code_v1/divide_by_zero.py"),
                fakes.ai_text(
                    fakes.review_fence(
                        [fakes.finding(location="divide_by_zero.py:7")],
                        files=["divide_by_zero.py"],
                    )
                ),
            ],
            "你只负责【安全性】视角": [
                fakes.ai_tool_call("read_file", path="code_v1/divide_by_zero.py"),
                fakes.ai_text(
                    fakes.review_fence(
                        [fakes.finding(location="divide_by_zero.py:9")],
                        files=["divide_by_zero.py"],
                    )
                ),
            ],
        }
    )

    assert (
        main(
            [
                "run",
                "-f",
                str(sample_path("divide_by_zero.py")),
                "--parallel-review",
                "--max-iter",
                "0",
            ]
        )
        == 1
    )
    capsys.readouterr()

    (record,) = records.list_runs()
    state = records.read_state(record.path)
    assert state is not None
    assert state["parallel_review"] is True
    assert {item["id"] for item in state["findings"]} == {"f-1", "f-2"}
    workspace = record.path / "workspace"
    assert (workspace / "review_1_correctness.json").is_file()
    assert (workspace / "review_1_security.json").is_file()
    assert (workspace / "review_1.json").is_file()


def test_config_can_import_the_previous_tool_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """零配置时提示导入旧工具的配置（``~/.cra/models.json``）：显式同意才写本配置（不改旧文件）。"""
    legacy = tmp_path / "legacy-models.json"
    legacy.write_text(
        json.dumps(
            {
                "default": "LegacyProvider/legacy-model",
                "providers": [
                    {
                        "name": "LegacyProvider",
                        "base_url": "https://legacy.example.invalid/v1",
                        "models": ["legacy-model"],
                        "api_key": "placeholder-legacy-key",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_mod, "legacy_config_path", lambda: legacy)
    answers = iter(["y", "0"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))

    assert main(["config"]) == 0

    captured = capsys.readouterr()
    assert "检测到可导入的配置" in captured.out
    assert "已导入 1 个供应商。" in captured.out
    data = config_mod.load_raw()
    assert [entry["name"] for entry in data["providers"]] == ["LegacyProvider"]
    assert data["default"] == "LegacyProvider/legacy-model"
    assert legacy.read_text(encoding="utf-8").count("api_key") == 1  # 旧配置文件原样保留


def test_config_import_is_skipped_when_the_user_declines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    legacy = tmp_path / "legacy-models.json"
    legacy.write_text(
        json.dumps(
            {"providers": [{"name": "X", "base_url": "https://x.invalid", "models": ["m"]}]}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_mod, "legacy_config_path", lambda: legacy)
    answers = iter(["n", "0"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))

    assert main(["config"]) == 0

    assert "已跳过导入。" in capsys.readouterr().out
    assert not config_mod.config_path().exists()


def test_config_import_rejects_a_self_conflicting_legacy_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """旧工具配置本身违反本工具的配置规则（此处为供应商重名）时不产生半导入状态。

    真冲突场景（同名供应商回退到同一 key 变量）只有在两边都无 key 时成立，而导入本身也要求
    本工具当前零供应商——可复现的冲突形态是"旧配置文件内部自相矛盾"（重名 / 撞变量）：
    整表校验先行，失败即给逐条原因且一个供应商都不写盘。
    """
    legacy = tmp_path / "legacy-models.json"
    legacy.write_text(
        json.dumps(
            {
                "providers": [
                    {"name": "Dup", "base_url": "https://a.invalid", "models": ["m1"]},
                    {"name": "Dup", "base_url": "https://b.invalid", "models": ["m2"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_mod, "legacy_config_path", lambda: legacy)
    answers = iter(["y", "0"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))

    assert main(["config"]) == 0

    captured = capsys.readouterr()
    assert "检测到可导入的配置" in captured.out
    assert "导入失败" in captured.err
    assert "供应商名重复" in captured.err  # 逐条原因，不是笼统的"不符合规则"
    assert not config_mod.config_path().exists()  # 一个都没写盘（无半导入状态）


def test_uninstall_command_wires_through_the_injected_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """命令面 → ``run_uninstall`` 的接线与退出码映射（真实 IO 打桩，绝不真实卸载）。"""
    from test_uninstall import FakeIO

    from crw import uninstall as uninstall_mod

    io = FakeIO(tools=[], answers=[None])
    monkeypatch.setattr(uninstall_mod, "default_io", lambda: io.to_io())

    assert main(["uninstall"]) == 0
    assert io.removed == [] and io.spawns == [] and io.scripts == []  # 取消 = 零删除


def test_import_hint_appears_when_nothing_is_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """零配置 + 存在旧工具配置：运行指引里补一行导入提示。"""
    legacy = tmp_path / "legacy-models.json"
    legacy.write_text(json.dumps({"providers": []}), encoding="utf-8")
    monkeypatch.setattr(config_mod, "legacy_config_path", lambda: legacy)

    assert main(["run", "写一个计算器"]) == 2

    assert "检测到可导入的配置" in capsys.readouterr().err


def test_resume_inherits_the_parallel_review_shape(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """审查形态随会话继续：从并行运行的记录续跑仍走双视角（否则结论不可比）。"""
    write_parent_record(parallel_review=True)
    configured_provider()
    fake_chat_model.route(
        {
            "你只负责【正确性】视角": [
                fakes.ai_tool_call("read_file", path="code_v1/main.py"),
                fakes.ai_text(fakes.review_fence([], "正确性视角已读完文件。")),
            ],
            "你只负责【安全性】视角": [
                fakes.ai_tool_call("read_file", path="code_v1/main.py"),
                fakes.ai_text(fakes.review_fence([], "安全性视角已读完文件。")),
            ],
        }
    )

    assert main(["resume", "r-20261012-0001", "--max-iter", "0"]) == 0
    capsys.readouterr()

    child = next(item for item in records.list_runs() if item.run_id != "r-20261012-0001")
    state = records.read_state(child.path)
    assert state is not None
    assert state["parallel_review"] is True
    assert (child.path / "workspace" / "review_1_correctness.json").is_file()
    assert (child.path / "workspace" / "review_1_security.json").is_file()


def test_resume_falls_back_to_default_when_inherited_rounds_is_out_of_range(
    configured_provider: Callable[[], dict[str, object]],
    fake_chat_model: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """继承来的轮次上限同样要过 0–10 的硬闸；越界即**回退内置默认**（不是夹到上限 10）。

    手改过的 ``state.json`` 写 999 不得绕过"防无界循环与无界费用"的护栏。
    """
    write_parent_record(max_iterations=999)
    configured_provider()
    fake_chat_model.load(
        [
            fakes.ai_tool_call("read_file", path="code_v1/main.py"),
            fakes.ai_text(fakes.review_fence([], "已读完文件。")),
        ]
    )

    assert main(["resume", "r-20261012-0001"]) == 0

    captured = capsys.readouterr()
    assert "轮次上限（999）不合法" in captured.err
    child = next(item for item in records.list_runs() if item.run_id != "r-20261012-0001")
    state = records.read_state(child.path)
    assert state is not None
    assert state["max_iterations"] == 3  # 退回内置默认
