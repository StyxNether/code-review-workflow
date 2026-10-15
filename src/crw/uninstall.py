"""``crw uninstall``：分级确认后的卸载清理（Design §6.2 #8 / §8.1）。

清理分四级、顺序固定：① 程序 / 垫片（``uv tool uninstall code-review-workflow``，由
"等本进程退出"的分离后台任务执行——运行中的 venv 解释器文件被锁）→ ② PATH 条目
（``~/.local/bin``，仅当卸载后已无其他 uv 工具）→ ③ 本程序登记的环境变量
（``~/.crw/env-vars.json``，只清登记在案者）→ ④ 用户数据目录（``~/.crw``，含可能
带明文 key 的 ``models.json``）。确认前零删除：清单先行、逐级提问，被拒绝或取消的
一级不动；未以 uv tool 安装时跳过 ①② 并说明。

可测试性接缝（Design §9.1）：输入、输出、外部命令与删除动作一律经
:class:`UninstallIO` 注入，真实实现集中在 :func:`default_io`——除此之外模块内不
触碰终端、``subprocess``、``shutil.rmtree`` 与 ``os.environ``。输出通道按 §8.4
分流：清单 / 菜单 / 结论走 stdout，错误与警示走 stderr。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import config as config_mod

#: 发行名 / uv 工具名：uv tool list 与卸载命令的目标（与 pyproject 的 project.name 一致）
PACKAGE_NAME = "code-review-workflow"

#: CLI 命令名（面向用户的文案用）
PROG = "crw"

#: uv 垫片目录（用户主目录下），PATH 条目展示与清理共用
UV_BIN_DIRNAME = ".local/bin"

#: 退出码：正常完成 / 用户取消为 0；存在未完成的清理失败项为 2（Design §8.1）
_EXIT_OK = 0
_EXIT_ERROR = 2

#: 合法环境变量名的完整形态；登记文件可被手改，拼进 PowerShell 脚本前必须校验
_ENV_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: 视为确认的输入；其余取值（含回车、None）一律按拒绝 / 取消处理
_AFFIRMATIVE = ("y", "yes")


@dataclass
class UninstallIO:
    """卸载流程的可注入 IO / 环境接缝（Design §9.1）；除 :func:`default_io` 外不碰真实系统。

    字段语义：``ask`` 读用户输入（``None`` = 取消 / EOF / 中断）；``say`` 走 stdout、
    ``fail`` 走 stderr（Design §8.4 的通道分流）；``list_uv_tools`` 读 uv 工具清单；
    ``run_script`` 同步执行外部命令并返回退出码（启动失败抛 ``OSError``，由调用方提示）；
    ``spawn_task`` 启动分离后台任务并返回是否成功；``remove_tree`` 递归删除目录；
    ``environ`` 为进程环境（清理登记变量用，默认 ``os.environ``）；``platform`` 决定
    脚本形态（默认 ``sys.platform``，测试可覆盖）；``pid`` 供"等本进程退出"的后台脚本使用
    （默认 ``os.getpid()``，同样可注入）。
    """

    ask: Callable[[str], str | None]
    say: Callable[[str], None]
    fail: Callable[[str], None]
    list_uv_tools: Callable[[], list[str]]
    run_script: Callable[[list[str]], int]
    spawn_task: Callable[[list[str]], bool]
    remove_tree: Callable[[Path], None]
    environ: MutableMapping[str, str] = field(default_factory=lambda: os.environ)
    platform: str = field(default_factory=lambda: sys.platform)
    pid: int = field(default_factory=os.getpid)


@dataclass(frozen=True)
class Inventory:
    """确认前的只读盘点结果（"确认前零删除"的保证面）。"""

    uv_tools: list[str]
    other_uv_tools: list[str]
    registered_env_vars: list[str]
    registry_path: Path
    registry_exists: bool
    registry_corrupt: bool
    """登记文件存在但不可解析（与"合法但为空清单"区分：只有前者算未完成的清理项）。"""
    data_dir: Path
    data_dir_exists: bool
    home_override: str


def default_io() -> UninstallIO:
    """真实环境的 IO 装配：``subprocess`` / ``shutil.rmtree`` / ``input`` / ``print`` 只在此出现。"""
    return UninstallIO(
        ask=_ask,
        say=_say,
        fail=_fail,
        list_uv_tools=_uv_tool_names,
        run_script=_run_external,
        spawn_task=_spawn_detached,
        remove_tree=shutil.rmtree,
        environ=os.environ,
        platform=sys.platform,
    )


def collect_inventory(
    io: UninstallIO,
    *,
    data_dir: Path | None = None,
    home_override: str | None = None,
) -> Inventory:
    """只读盘点：uv 工具清单、登记变量、数据目录位置与存在性；不提问、不打印、零删除。

    ``data_dir`` / ``home_override`` 供测试注入（默认取 :func:`crw.config.data_dir` 与
    进程环境里 ``CRW_HOME`` 的当前值）；登记清单经 :func:`crw.config.registered_env_vars`
    读取——它必须在删除数据目录之前读出，否则无从区分"本程序设置的变量"与用户自己的。
    """
    data_path = data_dir if data_dir is not None else config_mod.data_dir()
    override = (
        home_override if home_override is not None else io.environ.get(config_mod.HOME_ENV, "")
    )
    tools = list(io.list_uv_tools())
    registry = config_mod.env_registry_path()
    return Inventory(
        uv_tools=tools,
        other_uv_tools=[name for name in tools if name != PACKAGE_NAME],
        registered_env_vars=list(config_mod.registered_env_vars()),
        registry_path=registry,
        registry_exists=registry.exists(),
        registry_corrupt=_registry_parse_failed(registry),
        data_dir=data_path,
        data_dir_exists=data_path.exists(),
        home_override=override,
    )


def run_uninstall(*, io: UninstallIO | None = None) -> int:
    """``crw uninstall`` 的唯一入口：盘点 → 分级确认 → 逐级清理；返回退出码。

    退出码：正常完成或用户取消为 0；存在未完成的清理失败项（后台任务启动失败、
    环境变量清理失败、数据目录删除失败）为 2，原因说明走 stderr。
    """
    io = io if io is not None else default_io()
    inventory = collect_inventory(io)
    _print_manifest(io, inventory)
    failed = False

    # ① ② 程序 / 垫片与 PATH 条目：未以 uv tool 安装则整段跳过并说明（Design §8.1）
    if PACKAGE_NAME in inventory.uv_tools:
        failed = _run_program_tiers(io, inventory) or failed
    else:
        io.say(
            f"未在 uv tool list 中找到 {PACKAGE_NAME}（尚未以 uv tool 安装，或 uv 不可用）："
            "跳过程序卸载与 PATH 清理；以 uv run 方式使用本项目无需卸载程序本体。"
        )
    failed = _run_env_tier(io, inventory) or failed
    failed = _run_data_tier(io, inventory) or failed

    if failed:
        io.fail("卸载未全部完成：请按上方提示处理未完成的清理项。")
        return _EXIT_ERROR
    io.say("卸载流程结束；项目文件夹（仓库克隆）不由本程序删除，如需请自行删除克隆目录。")
    return _EXIT_OK


# ---- 真实 IO 实现（仅 default_io 引用）----


def _say(text: str) -> None:
    """结论通道：stdout（Design §8.4）。"""
    print(text)


def _fail(text: str) -> None:
    """错误与警示通道：stderr（Design §8.4）。"""
    print(text, file=sys.stderr)


def _ask(prompt: str) -> str | None:
    """终端输入：EOF / Ctrl+C / 输入流不可用一律按取消（``None``）处理（与 cli 同口径）。"""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt, OSError, RuntimeError):
        return None


def _uv_tool_names() -> list[str]:
    """``uv tool list`` 的工具名清单；uv 不可用或命令失败返回空表。

    输出形态：每个工具首行 "名称 vX.Y.Z"，随后若干 "- 别名/可执行文件" 行——
    非 "-" 开头的非空行首词即工具名。
    """
    try:
        result = subprocess.run(["uv", "tool", "list"], capture_output=True, text=True, check=False)
    except OSError:
        return []
    if result.returncode != 0:
        return []
    names: list[str] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("-"):
            continue
        names.append(stripped.split()[0])
    return names


def _run_external(argv: list[str]) -> int:
    """同步执行外部命令并返回退出码；启动失败抛 ``OSError``（由调用方转成可读提示）。"""
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    return result.returncode


def _spawn_detached(argv: list[str], *, platform: str | None = None) -> bool:
    """启动分离的后台任务（不等待、不占终端），返回是否成功启动。

    本进程退出后任务仍须存活（venv 解释器文件要被解锁后才能删净），故必须脱离当前
    控制台与进程组；``uv`` / ``powershell`` / ``sh`` 不可用（``OSError``）时返回
    ``False``，由调用方给出可读的手工卸载指令。
    """
    kwargs: dict[str, object] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    target = sys.platform if platform is None else platform
    if target == "win32":
        kwargs["creationflags"] = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
        kwargs["close_fds"] = True
    else:
        kwargs["start_new_session"] = True
    try:
        subprocess.Popen(argv, **kwargs)
    except OSError:
        return False
    return True


# ---- 分级清理 ----


def _confirmed(answer: str | None) -> bool:
    """仅显式 y / yes 视为确认；回车、取消与其它输入一律按拒绝处理（确认前零删除）。"""
    return answer is not None and answer.strip().lower() in _AFFIRMATIVE


def _print_manifest(io: UninstallIO, inventory: Inventory) -> None:
    """先打印完整清理清单再逐级提问（"先看后做"），结论走 stdout（Design §8.4）。"""
    io.say(f"{PROG} 卸载程序")
    io.say("以下为本程序可清理的内容，每一级单独确认；如需保留某一级，拒绝该级即可。")
    if PACKAGE_NAME in inventory.uv_tools:
        io.say(f"  ① 程序 / 垫片：uv 工具 {PACKAGE_NAME}（本进程退出后由后台任务卸载）")
    else:
        io.say(f"  ① 程序 / 垫片：uv tool list 中无 {PACKAGE_NAME}，将跳过")
    io.say(
        f"  ② PATH 条目：~/{UV_BIN_DIRNAME}（仅当卸载后已无其他 uv 工具；"
        "Windows 自动移除，POSIX 打印待删行）"
    )
    if inventory.registered_env_vars:
        io.say("  ③ 本程序登记的环境变量：" + ", ".join(inventory.registered_env_vars))
    else:
        io.say("  ③ 本程序登记的环境变量：（无登记）")
    io.say(
        f"  ④ 用户数据目录：{_display_path(inventory.data_dir)}"
        "（含可能带明文 key 的 models.json / settings.json、runs/ 运行记录、sessions/ 会话、env-vars.json）"
    )


def _run_program_tiers(io: UninstallIO, inventory: Inventory) -> bool:
    """①② 程序 / 垫片与 PATH 条目；返回是否发生清理失败。

    Windows 的 PATH 清理就是后台脚本的一部分（条目移除须发生在工具卸载之后），故 ②
    的确认必须先于 ① 的任务生成；POSIX 只打印待删的 shell 配置行，绝不代改 rc。
    """
    if not _confirmed(io.ask(f"卸载 uv 工具 {PACKAGE_NAME}（程序本体与 crw 垫片）？（y/N）")):
        io.say(f"已取消：保留 uv 工具 {PACKAGE_NAME}，PATH 条目不动。")
        return False
    remove_path = _should_remove_uv_path_entry(inventory.other_uv_tools)
    path_approved = False
    if remove_path:
        hint = (
            "Windows 下由后台任务自动移除"
            if io.platform == "win32"
            else "POSIX 下打印待删行，不自动改 rc"
        )
        prompt = (
            f"卸载后已无其他 uv 工具，是否一并移除 PATH 中的 ~/{UV_BIN_DIRNAME} 条目"
            f"（{hint}）？（y/N）"
        )
        path_approved = _confirmed(io.ask(prompt))
    else:
        io.say(
            f"检测到其他 uv 工具（{', '.join(inventory.other_uv_tools)}），"
            f"PATH 中的 ~/{UV_BIN_DIRNAME} 条目由它们共享，予以保留。"
        )
    if not io.spawn_task(_program_cleanup_command(io, remove_path=path_approved)):
        io.fail("后台清理任务启动失败，程序尚未卸载。")
        io.fail(f"请稍后手工执行：uv tool uninstall {PACKAGE_NAME}")
        return True
    io.say("程序将在本进程退出后完成卸载（运行中的文件被占用，须等本进程退出）。")
    if remove_path and path_approved:
        if io.platform == "win32":
            io.say(f"PATH 中的 ~/{UV_BIN_DIRNAME} 条目将在后台卸载任务中一并移除。")
        else:
            io.say(
                "卸载完成后，请从 shell 配置文件（~/.bashrc / ~/.zshrc 等）删除如下 PATH 行（如有）："
            )
            io.say(f'  export PATH="$HOME/{UV_BIN_DIRNAME}:$PATH"')
    elif remove_path:
        io.say(f"已保留 PATH 中的 ~/{UV_BIN_DIRNAME} 条目。")
    return False


def _registry_parse_failed(registry: Path) -> bool:
    """登记文件存在但无法解析出字符串数组清单（合法但为空清单不算损坏）。"""
    if not registry.exists():
        return False
    try:
        data = json.loads(registry.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return True
    names = data.get("env_vars") if isinstance(data, dict) else None
    return not isinstance(names, list)


def _run_env_tier(io: UninstallIO, inventory: Inventory) -> bool:
    """③ 本程序登记的环境变量：确认后清理；返回是否发生失败。"""
    names = inventory.registered_env_vars
    if not names:
        if inventory.registry_corrupt:
            # 登记文件在却解析不出清单：与"确实无登记"（含合法空清单）分开提示，避免
            # 本程序设置过的变量因文件损坏被静默漏清理；按"未完成的清理项"计（退出码 2）
            io.fail(
                f"环境变量登记文件 {inventory.registry_path} 已损坏，"
                "无法识别本程序设置过的变量；请手工核对（本次按未完成清理计）。"
            )
            return True
        io.say("未发现本程序登记的环境变量；您自行设置的变量不会被动。")
        return False
    prompt = f"清除本程序登记的环境变量（{', '.join(names)}）？只清登记在案的名称（y/N）"
    if not _confirmed(io.ask(prompt)):
        io.say("已保留本程序登记的环境变量。")
        return False
    return not _cleanup_registered_env_vars(io, names)


def _run_data_tier(io: UninstallIO, inventory: Inventory) -> bool:
    """④ 用户数据目录：确认后删除；返回是否发生失败（安全跳过不算失败）。"""
    display = _display_path(inventory.data_dir)
    if not inventory.data_dir_exists:
        io.say(f"未发现用户数据目录 {display}，跳过数据清理。")
        return False
    reason = _data_dir_block_reason(inventory.data_dir, inventory.home_override)
    if reason is not None:
        io.fail(f"出于安全考虑，数据目录删除已跳过：{reason}。")
        io.fail(f"如确需清理，请核对后手工删除：{display}")
        return False
    prompt = f"删除用户数据目录 {display}（含可能带明文 key 的 models.json / settings.json、runs/ 运行记录、sessions/ 会话等）？（y/N）"
    if not _confirmed(io.ask(prompt)):
        io.say(f"已保留用户数据目录：{display}")
        return False
    # 打印绝对路径后才动删除（Design §6.2 #8）
    io.say(f"将删除数据目录：{display}")
    try:
        io.remove_tree(inventory.data_dir)
    except OSError as exc:
        io.fail(f"用户数据删除失败：{exc}")
        io.fail(f"请稍后手工删除该目录：{display}")
        return True
    io.say(f"用户数据已删除：{display}")
    return False


# ---- 环境变量清理 ----


def _windows_env_cleanup_script(names: Sequence[str]) -> str:
    """Windows 用户环境变量清理脚本：逐个删除登记在案的变量。

    .NET 的 SetEnvironmentVariable 写回注册表时会广播 WM_SETTINGCHANGE，新开的终端
    立即生效（reg delete 无此效果）。调用方已把名单过滤为合法标识符——登记文件可被
    手改，不得把任意内容拼进脚本。
    """
    quoted = ", ".join(f"'{name}'" for name in names)
    return (
        "$ErrorActionPreference = 'Stop'\n"
        f"foreach ($name in @({quoted})) {{\n"
        "  [Environment]::SetEnvironmentVariable($name, $null, 'User')\n"
        "}\n"
    )


def _cleanup_registered_env_vars(io: UninstallIO, names: Sequence[str]) -> bool:
    """清理本程序登记的环境变量；返回是否已全部处理（须手工收尾的项算未完成）。

    Windows：前台 PowerShell 清用户环境（同步执行、完成即生效）；POSIX：本程序从不
    自动写 shell 配置文件，持久化的 export 行只能由用户删除（打印待删行指引）。两者
    都只处理登记在案的名单，并从当前进程环境移除——用户自己设置的同名变量绝不触碰。
    """
    valid = [name for name in names if _ENV_NAME_PATTERN.fullmatch(name)]
    invalid = [name for name in names if not _ENV_NAME_PATTERN.fullmatch(name)]
    if invalid:
        io.fail(
            "以下登记项不是合法变量名（登记文件可能被手改），无法自动清理，请手工检查："
            + ", ".join(invalid)
        )
    if not valid:
        return not invalid
    if io.platform == "win32":
        command = [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            _windows_env_cleanup_script(valid),
        ]
        try:
            code = io.run_script(command)
        except OSError as exc:
            io.fail(f"环境变量清理失败：{exc}")
            io.fail(f"请手工删除下列用户环境变量：{', '.join(valid)}")
            return False
        if code != 0:
            io.fail(f"环境变量清理失败（PowerShell 退出码 {code}）。")
            io.fail(f"请手工删除下列用户环境变量：{', '.join(valid)}")
            return False
    else:
        io.say("若下列变量的 export 已写入 shell 配置文件，请手工删除对应行（本程序不自动改 rc）：")
        for name in valid:
            io.say(f"  - 含 {name} 的 export 行")
    for name in valid:
        io.environ.pop(name, None)  # 当前进程立即生效（与设置侧对称）
    if io.platform == "win32":
        io.say(f"已删除 {len(valid)} 个本程序设置的环境变量（用户环境）。")
    else:
        io.say(f"已从当前进程清除 {len(valid)} 个变量；持久化的 export 行请按上方指引手工删除。")
    return not invalid


# ---- 数据目录删除的两道保险 ----


def _data_dir_block_reason(data_dir: Path, home_override: str) -> str | None:
    """数据目录删除的放行判定；返回拒绝原因（``None`` = 允许删除）。

    三道保险（Design §6.2 #8）：① 目录名必须是默认的 ``.crw``，或 ``CRW_HOME`` 被显式
    设置（只有用户主动指定的目录才算数）；② 绝不删除仓库目录——即使命中第 ① 条，含
    ``.git`` 的目录与当前工作目录及其上级也一律拒绝，防误删代码仓库；③ 绝不删除
    **文件系统根**——第 ① 条把"设了 CRW_HOME"当成用户意图，而 `CRW_HOME=/` 这类误配
    恰好是"设了但指错了"，主目录与上级两道拦不住跨盘的根（Windows 上主目录在另一块盘时
    `home.is_relative_to(根)` 为假），确认前必须拦下。
    """
    if data_dir.name != config_mod.DEFAULT_HOME_NAME and not home_override.strip():
        return (
            f"目录名不是 {config_mod.DEFAULT_HOME_NAME}，且未显式设置 {config_mod.HOME_ENV}，"
            "无法确认它是本程序的数据目录"
        )
    try:
        target = data_dir.resolve()
    except OSError:
        target = data_dir
    if target.parent == target:
        return "该目录是文件系统根目录（本程序绝不删除）"
    if (target / ".git").exists():
        return "该目录是一个 Git 仓库（本程序绝不删除仓库目录）"
    try:
        home = Path.home().resolve()
    except OSError:  # pragma: no cover — 解析不出主目录时跳过该道保险
        home = None
    if home is not None and (target == home or home.is_relative_to(target)):
        return "该目录是用户主目录或其上级（本程序绝不删除）"
    cwd = _current_dir()
    if cwd is not None and (cwd == target or cwd.is_relative_to(target)):
        return "该目录是当前工作目录或其上级（删除会连带当前工作目录）"
    return None


def _current_dir() -> Path | None:
    """当前工作目录；取不到（目录已被删等）返回 ``None``，跳过相关保护判定。"""
    try:
        return Path.cwd()
    except OSError:
        return None


def _display_path(path: Path) -> Path:
    """绝对路径展示：删除前须打印绝对路径；解析失败退回原路径（展示不能阻碍流程）。"""
    try:
        return path.resolve()
    except OSError:
        return path


# ---- 分离后台任务（等本进程退出再执行 uv tool uninstall）----


def _should_remove_uv_path_entry(other_tool_names: Sequence[str]) -> bool:
    """PATH 条目移除判定：卸载后已无其他 uv 工具才移除 ``~/.local/bin`` 条目。

    条目被其他 uv 工具共享时保留（删净自己、不破坏他人）；``uv tool uninstall`` 只删
    工具环境与垫片、不动 PATH，PATH 清理须本命令自理。
    """
    return not other_tool_names


def _program_cleanup_command(io: UninstallIO, *, remove_path: bool) -> list[str]:
    """后台清理任务的完整命令行：按平台选择解释器与脚本形态。"""
    if io.platform == "win32":
        script = _windows_program_cleanup_script(io.pid, remove_path=remove_path)
        return ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]
    return ["sh", "-c", _posix_program_cleanup_script(io.pid)]


def _windows_program_cleanup_script(parent_pid: int, *, remove_path: bool) -> str:
    """Windows 后台清理脚本（分离 PowerShell 执行）。

    先等父进程退出（运行中的 venv 解释器文件被锁，父进程退出后才能删净），再执行
    ``uv tool uninstall``；``remove_path`` 时在卸载后复查 ``uv tool list``——彼时已无
    其他 uv 工具才移除用户 PATH 的 ``~/.local/bin`` 条目（条目共享则保留）。
    """
    script = (
        "$ErrorActionPreference = 'SilentlyContinue'\n"
        f"while (Get-Process -Id {parent_pid} -ErrorAction SilentlyContinue) "
        "{ Start-Sleep -Milliseconds 500 }\n"
        f"uv tool uninstall {PACKAGE_NAME}\n"
    )
    if remove_path:
        script += (
            "$names = @(uv tool list 2>$null | Where-Object { "
            "$_ -and -not $_.Trim().StartsWith('-') })\n"
            "if ($LASTEXITCODE -eq 0 -and $names.Count -eq 0) {\n"
            "  $bin = Join-Path $env:USERPROFILE '.local\\bin'\n"
            "  $user = [Environment]::GetEnvironmentVariable('Path', 'User')\n"
            "  if ($user) {\n"
            "    $entries = $user -split ';'\n"
            "    $kept = @($entries | Where-Object {\n"
            "      $_ -and $_.TrimEnd('\\').ToLowerInvariant() -ne "
            "$bin.TrimEnd('\\').ToLowerInvariant()\n"
            "    })\n"
            "    if ($kept.Count -lt $entries.Count) {\n"
            "      [Environment]::SetEnvironmentVariable('Path', ($kept -join ';'), 'User')\n"
            "    }\n"
            "  }\n"
            "}\n"
        )
    return script


def _posix_program_cleanup_script(parent_pid: int) -> str:
    """POSIX 后台清理脚本（``sh -c`` 分离执行）：等父进程退出后执行 ``uv tool uninstall``。

    PATH 清理不自动执行：POSIX 的 PATH 条目写在 shell 配置文件里，由主流程在确认时
    打印待删行指引（绝不代改 rc）。
    """
    return (
        f"while kill -0 {parent_pid} 2>/dev/null; do sleep 0.5; done\n"
        f"uv tool uninstall {PACKAGE_NAME}\n"
    )
