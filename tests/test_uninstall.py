"""crw uninstall：分级确认、确认前零删除与各级清理的接缝行为（Design §6.2 #8 / §8.1 / §9.1）。

全部用例经注入的假 IO 驱动：不真实卸载、不启动后台任务、不触碰真实环境与网络。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from crw import config, uninstall
from crw import config as config_mod

PACKAGE = uninstall.PACKAGE_NAME


class FakeIO:
    """卸载流程的假 IO：记录调用、按预设答案回答、可注入失败；不触碰任何真实系统。"""

    def __init__(
        self,
        *,
        tools: list[str] | None = None,
        answers: list[str | None] | None = None,
        platform: str = "linux",
        environ: dict[str, str] | None = None,
        pid: int = 4242,
    ) -> None:
        self.tools = list(tools or [])
        self.answers = list(answers or [])
        self.platform = platform
        self.pid = pid
        self.environ = dict(os.environ) if environ is None else environ
        self.prompts: list[str] = []
        self.says: list[str] = []
        self.fails: list[str] = []
        self.scripts: list[list[str]] = []
        self.spawns: list[list[str]] = []
        self.removed: list[Path] = []
        self.spawn_ok = True
        self.script_code = 0
        self.script_error: OSError | None = None
        self.remove_error: OSError | None = None

    def ask(self, prompt: str) -> str | None:
        self.prompts.append(prompt)
        return self.answers.pop(0) if self.answers else None

    def say(self, text: str) -> None:
        self.says.append(text)

    def fail(self, text: str) -> None:
        self.fails.append(text)

    def list_uv_tools(self) -> list[str]:
        return list(self.tools)

    def run_script(self, argv: list[str]) -> int:
        self.scripts.append(argv)
        if self.script_error is not None:
            raise self.script_error
        return self.script_code

    def spawn_task(self, argv: list[str]) -> bool:
        self.spawns.append(argv)
        return self.spawn_ok

    def remove_tree(self, path: Path) -> None:
        self.removed.append(path)
        if self.remove_error is not None:
            raise self.remove_error

    def to_io(self) -> uninstall.UninstallIO:
        """装配成模块要求的注入对象。"""
        return uninstall.UninstallIO(
            ask=self.ask,
            say=self.say,
            fail=self.fail,
            list_uv_tools=self.list_uv_tools,
            run_script=self.run_script,
            spawn_task=self.spawn_task,
            remove_tree=self.remove_tree,
            environ=self.environ,
            platform=self.platform,
            pid=self.pid,
        )

    def stdout(self) -> str:
        return "\n".join(self.says)

    def stderr(self) -> str:
        return "\n".join(self.fails)


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """把数据目录显式重定向到 tmp（不依赖 conftest 夹具的执行顺序）。"""
    target = tmp_path / "crw-home"
    monkeypatch.setenv(config.HOME_ENV, str(target))
    return target


def _make_home(home: Path) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    return home


def _write_registry(names: list[str]) -> None:
    """直接写登记文件（模拟被手改的登记文件，含非法项）。"""
    path = config.env_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"env_vars": names}), encoding="utf-8")


def _spawned_script(io: FakeIO) -> str:
    """后台任务 argv 里的脚本（powershell / sh 的最后一段）。"""
    assert io.spawns, "未启动后台任务"
    return io.spawns[0][-1]


# ---- 接缝与纯函数 ----


def test_default_io_wires_real_system_seams() -> None:
    io = uninstall.default_io()

    assert io.environ is os.environ
    assert io.platform == sys.platform
    assert all(
        callable(item)
        for item in (
            io.ask,
            io.say,
            io.fail,
            io.list_uv_tools,
            io.run_script,
            io.spawn_task,
            io.remove_tree,
        )
    )


def test_collect_inventory_is_read_only(home: Path) -> None:
    data = _make_home(home)
    io = FakeIO(tools=[PACKAGE, "other-tool"])

    inventory = uninstall.collect_inventory(io.to_io())

    assert inventory.uv_tools == [PACKAGE, "other-tool"]
    assert inventory.other_uv_tools == ["other-tool"]
    assert inventory.data_dir == data
    assert inventory.data_dir_exists is True
    assert inventory.home_override == str(data)
    assert inventory.registered_env_vars == []
    assert io.prompts == [] and io.says == [] and io.fails == []
    assert io.spawns == [] and io.scripts == [] and io.removed == []


def test_collect_inventory_accepts_injected_paths(home: Path) -> None:
    io = FakeIO(tools=[])
    custom = home.parent / "custom-data"

    inventory = uninstall.collect_inventory(io.to_io(), data_dir=custom, home_override="  ")

    assert inventory.data_dir == custom
    assert inventory.data_dir_exists is False
    assert inventory.home_override == "  "


def test_should_remove_uv_path_entry_only_without_other_tools() -> None:
    assert uninstall._should_remove_uv_path_entry([]) is True
    assert uninstall._should_remove_uv_path_entry(["other-tool"]) is False


def test_data_dir_guard_requires_default_name_or_explicit_override(home: Path) -> None:
    default_like = home.parent / config.DEFAULT_HOME_NAME
    custom = home.parent / "custom-data"

    assert uninstall._data_dir_block_reason(default_like, "") is None
    assert uninstall._data_dir_block_reason(custom, "") is not None
    assert uninstall._data_dir_block_reason(custom, str(custom)) is None


def test_data_dir_guard_never_deletes_repository(home: Path) -> None:
    repo_like = home.parent / config.DEFAULT_HOME_NAME
    (repo_like / ".git").mkdir(parents=True)

    reason = uninstall._data_dir_block_reason(repo_like, str(repo_like))

    assert reason is not None
    assert "Git" in reason


# ---- 后台脚本与命令形态 ----


def test_windows_command_waits_for_this_process_before_uninstalling() -> None:
    io = FakeIO(tools=[PACKAGE], platform="win32")

    command = uninstall._program_cleanup_command(io.to_io(), remove_path=False)

    assert command[0] == "powershell"
    script = command[-1]
    assert f"uv tool uninstall {PACKAGE}" in script
    assert "Get-Process -Id" in script
    assert "'.local\\bin'" not in script


# ---- 分级确认的行为 ----


def test_cancel_at_first_prompt_touches_nothing(home: Path) -> None:
    _make_home(home)
    config.register_env_var("CRW_DEEPSEEK_API_KEY")
    io = FakeIO(tools=[PACKAGE], answers=[None])

    code = uninstall.run_uninstall(io=io.to_io())

    assert code == 0
    assert io.spawns == [] and io.scripts == [] and io.removed == []


def test_all_tiers_declined_reports_and_changes_nothing(home: Path) -> None:
    _make_home(home)
    config.register_env_var("CRW_DEEPSEEK_API_KEY")
    io = FakeIO(
        tools=[PACKAGE],
        answers=["n", "n", "n", "n"],
        environ={
            **os.environ,
            "CRW_DEEPSEEK_API_KEY": "placeholder-registered",
            "SE_CodeAgent": "user-own",
        },
    )
    before = dict(io.environ)

    code = uninstall.run_uninstall(io=io.to_io())

    assert code == 0
    assert io.spawns == [] and io.scripts == [] and io.removed == []
    assert io.environ == before
    assert "已取消" in io.stdout()
    assert "已保留" in io.stdout()


def test_tool_not_installed_skips_program_and_path_without_error(home: Path) -> None:
    io = FakeIO(tools=["other-tool"], answers=["y"])

    code = uninstall.run_uninstall(io=io.to_io())

    assert code == 0
    assert io.spawns == [] and io.scripts == [] and io.removed == []
    assert io.fails == []
    out = io.stdout()
    assert "uv run" in out
    assert "跳过程序卸载" in out


def test_declined_path_tier_leaves_path_block_out(home: Path) -> None:
    io = FakeIO(tools=[PACKAGE], answers=["y", "n"], platform="win32")

    code = uninstall.run_uninstall(io=io.to_io())

    assert code == 0
    assert "'.local\\bin'" not in _spawned_script(io)
    assert "已保留 PATH" in io.stdout()


def test_env_tier_only_clears_registered_names(home: Path) -> None:
    config.register_env_var("CRW_DEEPSEEK_API_KEY")
    io = FakeIO(
        tools=[],
        answers=["y"],
        environ={
            **os.environ,
            "CRW_DEEPSEEK_API_KEY": "placeholder-registered",
            "CRW_MOONSHOT_API_KEY": "user-own",
            "SE_CodeAgent": "user-own-legacy",
        },
    )

    code = uninstall.run_uninstall(io=io.to_io())

    assert code == 0
    assert "CRW_DEEPSEEK_API_KEY" not in io.environ
    assert io.environ["CRW_MOONSHOT_API_KEY"] == "user-own"
    assert io.environ["SE_CodeAgent"] == "user-own-legacy"
    out = io.stdout()
    assert "CRW_DEEPSEEK_API_KEY" in out
    assert "export" in out  # POSIX：打印待手工删除的 shell 配置行
    assert io.scripts == [] and io.spawns == [] and io.removed == []


def test_env_tier_windows_script_excludes_invalid_names(home: Path) -> None:
    _write_registry(["CRW_OK_VAR", "1BAD-NAME", "bad name"])
    io = FakeIO(
        tools=[],
        answers=["y"],
        platform="win32",
        environ={**os.environ, "CRW_OK_VAR": "placeholder", "1BAD-NAME": "user-value"},
    )

    code = uninstall.run_uninstall(io=io.to_io())

    assert code == 2
    assert len(io.scripts) == 1
    argv = io.scripts[0]
    assert argv[0] == "powershell"
    script = argv[-1]
    assert "CRW_OK_VAR" in script
    assert "1BAD-NAME" not in script and "bad name" not in script
    assert "[Environment]::SetEnvironmentVariable" in script
    assert "'User'" in script
    err = io.stderr()
    assert "手工" in err and "1BAD-NAME" in err and "bad name" in err
    assert "CRW_OK_VAR" not in io.environ
    assert io.environ["1BAD-NAME"] == "user-value"


@pytest.mark.parametrize("failure", ["raises", "nonzero"])
def test_env_cleanup_failure_returns_2(home: Path, failure: str) -> None:
    config.register_env_var("CRW_DEEPSEEK_API_KEY")
    io = FakeIO(tools=[], answers=["y"], platform="win32")
    if failure == "raises":
        io.script_error = OSError("powershell 不可用")
    else:
        io.script_code = 1

    code = uninstall.run_uninstall(io=io.to_io())

    assert code == 2
    assert "手工" in io.stderr()


def test_data_tier_only_removes_data_dir(home: Path) -> None:
    _make_home(home)
    config.register_env_var("CRW_DEEPSEEK_API_KEY")
    io = FakeIO(
        tools=[],
        answers=["n", "y"],
        environ={**os.environ, "CRW_DEEPSEEK_API_KEY": "placeholder-registered"},
    )
    target = config.data_dir()

    code = uninstall.run_uninstall(io=io.to_io())

    assert code == 0
    assert io.removed == [target]
    assert io.scripts == [] and io.spawns == []
    assert io.environ["CRW_DEEPSEEK_API_KEY"] == "placeholder-registered"
    assert io.fails == []
    printed = uninstall._display_path(target)
    assert target.is_absolute()
    assert f"将删除数据目录：{printed}" in io.stdout()
    assert f"用户数据已删除：{printed}" in io.stdout()


def test_out_of_scope_data_dir_is_warned_and_kept(
    home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    custom = tmp_path / "custom-data"
    custom.mkdir()
    monkeypatch.setattr(config, "data_dir", lambda: custom)
    io = FakeIO(tools=[], answers=["y"], environ={})  # 未显式设置 CRW_HOME

    code = uninstall.run_uninstall(io=io.to_io())

    assert code == 0
    assert io.removed == []
    err = io.stderr()
    assert "跳过" in err
    assert config.HOME_ENV in err


def test_missing_data_dir_is_skipped(home: Path) -> None:
    io = FakeIO(tools=[], answers=["y"])

    code = uninstall.run_uninstall(io=io.to_io())

    assert code == 0
    assert io.removed == []
    assert "未发现用户数据目录" in io.stdout()


def test_data_removal_failure_returns_2(home: Path) -> None:
    _make_home(home)
    io = FakeIO(tools=[], answers=["y"])
    io.remove_error = OSError("拒绝访问")

    code = uninstall.run_uninstall(io=io.to_io())

    assert code == 2
    assert "用户数据删除失败" in io.stderr()
    assert "手工删除" in io.stderr()


def test_spawn_failure_returns_2_with_manual_command(home: Path) -> None:
    io = FakeIO(tools=[PACKAGE], answers=["y", "y"])
    io.spawn_ok = False

    code = uninstall.run_uninstall(io=io.to_io())

    assert code == 2
    assert f"uv tool uninstall {PACKAGE}" in io.stderr()
    assert "手工" in io.stderr()


def test_corrupt_registry_warns_and_skips_env_tier(home: Path) -> None:
    _make_home(home)
    config.env_registry_path().write_text("{ 损坏", encoding="utf-8")
    io = FakeIO(tools=[], answers=["n"])

    code = uninstall.run_uninstall(io=io.to_io())

    assert code == 2  # 报错文案配成功退出码会让调用方把"可能漏清变量"当成完全成功
    assert io.scripts == []
    err = io.stderr()
    assert "已损坏" in err and "手工" in err and "未完成清理" in err


# ---- PATH 条目 ----


def test_shared_path_entry_is_kept_and_script_has_no_path_block(home: Path) -> None:
    io = FakeIO(tools=[PACKAGE, "other-tool"], answers=["y"], platform="win32")

    code = uninstall.run_uninstall(io=io.to_io())

    assert code == 0
    assert len(io.spawns) == 1
    script = _spawned_script(io)
    assert f"uv tool uninstall {PACKAGE}" in script
    assert "'.local\\bin'" not in script
    assert "GetEnvironmentVariable" not in script
    assert "共享" in io.stdout()


def test_last_uv_tool_removes_path_entry_on_windows(home: Path) -> None:
    io = FakeIO(tools=[PACKAGE], answers=["y", "y"], platform="win32")

    code = uninstall.run_uninstall(io=io.to_io())

    assert code == 0
    script = _spawned_script(io)
    assert "'.local\\bin'" in script
    assert "uv tool list" in script
    assert "[Environment]::GetEnvironmentVariable('Path', 'User')" in script
    assert "后台卸载任务中一并移除" in io.stdout()


def test_posix_path_entry_is_printed_not_executed(home: Path) -> None:
    io = FakeIO(tools=[PACKAGE], answers=["y", "y"], platform="linux")

    code = uninstall.run_uninstall(io=io.to_io())

    assert code == 0
    script = _spawned_script(io)
    assert script.startswith("while kill -0")
    assert "'.local\\bin'" not in script
    out = io.stdout()
    assert f'export PATH="$HOME/{uninstall.UV_BIN_DIRNAME}:$PATH"' in out
    assert io.scripts == []


# ---- 固定顺序与整体流程 ----


def test_confirmation_order_is_fixed_and_happy_path(home: Path) -> None:
    data = _make_home(home)
    config.register_env_var("CRW_DEEPSEEK_API_KEY")
    io = FakeIO(tools=[PACKAGE], answers=["y", "y", "y", "y"], platform="linux")

    code = uninstall.run_uninstall(io=io.to_io())

    assert code == 0
    assert io.fails == []
    labels = ["uv 工具", "PATH", "环境变量", "数据目录"]
    assert len(io.prompts) == 4
    for prompt, label in zip(io.prompts, labels, strict=True):
        assert label in prompt
    script = _spawned_script(io)
    assert "kill -0" in script and f"uv tool uninstall {PACKAGE}" in script
    assert io.removed == [data]
    assert "卸载流程结束" in io.stdout()


def test_env_registry_corrupt_is_distinguished_from_empty() -> None:
    """登记文件"损坏"与"合法空清单"分开判定：只有前者算未完成的清理项（退出码 2）。"""
    registry = config_mod.env_registry_path()
    registry.parent.mkdir(parents=True, exist_ok=True)
    io = FakeIO()

    registry.write_text('{"env_vars": []}', encoding="utf-8")
    assert uninstall.collect_inventory(io).registry_corrupt is False

    registry.write_text("{坏 JSON", encoding="utf-8")
    inventory = uninstall.collect_inventory(io)
    assert inventory.registry_corrupt is True
    assert inventory.registered_env_vars == []


def test_data_dir_guard_refuses_home_and_its_ancestors() -> None:
    """数据目录保险：用户主目录及其上级一律拒绝删除（极端 CRW_HOME 配置的兜底）。"""
    home = Path.home()

    assert uninstall._data_dir_block_reason(home, str(home)) is not None
    root = Path(home.anchor)
    assert uninstall._data_dir_block_reason(root, str(root)) is not None
