"""代码执行工具：run_python。

**本机直接运行，非沙箱**：子进程拥有当前用户全部权限，仅用于审查 / 修复过程中验证
可信代码片段；风险以"限时 + 进程树强杀 + 输出截断 + 环境净化 + 限制明示"控制。

路径以**上下文根**解析（Design §8.4），子进程工作目录 = 上下文根；子进程输出捕获
显式 ``encoding='utf-8', errors='replace'``（Windows 默认 cp936，否则 UTF-8 源码的
中文输出会解码崩溃），并设 ``PYTHONIOENCODING=utf-8`` 保持子进程侧一致；环境经净化
后才继承（剔除凭证变量与本程序自有变量，防被审查代码经环境读走 key 后经输出回传
模型上下文）。

超时等失败以异常上抛，由注册表 dispatch 统一兜底为结构化错误文本回传模型。
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from .. import config
from . import fs

#: stdout/stderr 各自截断上限（展示用）
MAX_OUTPUT_CHARS = 2000
#: 读取阶段的存储上限：超出后仍继续排空管道（防子进程写阻塞）但不再留存，
#: 内存占用因此有界——"先全量读入再截断"会让长输出的子进程把内存撑爆
MAX_CAPTURE_CHARS = 64 * 1024
_READ_CHUNK_CHARS = 4096
#: 模型可请求的单次超时上限，防止长挂拖死会话
MAX_TIMEOUT_SECONDS = 60
#: 强杀后回收子进程的有限等待；再超时则放弃回收（杀树已尽力）
KILL_REAP_SECONDS = 5

_TRUNCATION_NOTICE = "…（输出超过 {limit} 字符，已截断）"
_PYTHON_SUFFIX = ".py"

#: 凭证类变量前缀：本程序与既有工具链的变量一律不进子进程环境（Design D9）
_CREDENTIAL_PREFIXES = ("CRW_", "CRA_")


def _is_credential_var(name: str) -> bool:
    """环境变量名是否属"不传给子进程"的一类（独立成函数便于测试与扩展名单）。

    大小写不敏感比对：Windows 环境变量名不区分大小写；连字符先归一为下划线（Windows
    变量名合法字符，归一后覆盖 MY-API-KEY 型命名）；POSIX 侧为大小写敏感，不敏感过滤
    只会多剔除无关变量，属安全方向的超集。名单：本程序与既有工具链的变量前缀
    （``CRW_`` / ``CRA_``）、历史变量 ``SE_CodeAgent``、一切 ``*API_KEY*`` / ``*APIKEY*``
    型命名。名称黑名单无法覆盖"名字无害、值是凭证"的变量——这是本控制的已知边界。
    """
    normalized = name.upper().replace("-", "_")
    if normalized.startswith(_CREDENTIAL_PREFIXES) or normalized == "SE_CODEAGENT":
        return True
    return "API_KEY" in normalized or "APIKEY" in normalized


def _sanitized_env() -> dict[str, str]:
    """父环境剔除凭证变量后的副本（key 不得进入子进程环境）。

    被审查 / 被生成的代码可 ``print(os.environ)`` 读走环境变量，其输出会随运行日志与
    模型上下文留存——剔除代理自身凭证与该通道上的一切 key 形态变量；非凭证类变量
    （PATH 等）原样保留。
    """
    return {name: value for name, value in os.environ.items() if not _is_credential_var(name)}


def run_python(
    roots: Sequence[Path],
    code: str | None = None,
    path: str | None = None,
    timeout: int | None = None,
) -> str:
    """执行 Python 代码并返回退出码与 stdout/stderr（各截断至 2000 字符）。

    ``code``（内联代码）与 ``path``（相对上下文根的 .py 文件）二选一；内联执行是修复
    Agent 验证候选代码的方式（候选尚未落盘）。默认超时取 ``CRW_EXEC_TIMEOUT``（10s），
    可用 ``timeout`` 参数覆盖（上限 60s）。
    """
    if (code is None) == (path is None):
        return f"{fs.ERROR_PREFIX}code 与 path 必须二选一（恰好提供一个）。"
    if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, int)):
        # 模型可能把整数参数写成字符串：给出可读原因，而不是让比较抛 TypeError
        return (
            f"{fs.ERROR_PREFIX}timeout 必须是 1~{MAX_TIMEOUT_SECONDS} 的整数秒，当前值：{timeout!r}"
        )
    if timeout is not None and not 0 < timeout <= MAX_TIMEOUT_SECONDS:
        return (
            f"{fs.ERROR_PREFIX}timeout 必须是 1~{MAX_TIMEOUT_SECONDS} 的整数秒，当前值：{timeout}"
        )

    roots_resolved = fs.resolve_roots(roots)
    if not roots_resolved:
        return f"{fs.ERROR_PREFIX}未提供上下文根，无法执行（运行工作区不可用）。"
    workdir = roots_resolved[0]
    if code is not None:
        command = [sys.executable, "-c", code]
    else:
        assert path is not None  # 二选一校验已保证
        target, error = fs.resolve_in_roots(roots_resolved, path)
        if error is not None:
            return error
        assert target is not None  # resolve_in_roots 的错误分支已返回
        if not target.exists():
            return f"{fs.ERROR_PREFIX}文件不存在：{path!r}"
        if not target.is_file():
            return f"{fs.ERROR_PREFIX}路径不是文件：{path!r}（列目录请用 list_dir）"
        # Windows 文件系统大小写不敏感：UPPER.PY 也是 Python 文件
        if target.suffix.lower() != _PYTHON_SUFFIX:
            return (
                f"{fs.ERROR_PREFIX}{target.name} 不是 Python 文件（仅支持 .py），无法执行。"
                "如需查看其内容请改用 read_file。"
            )
        command = [sys.executable, str(target)]
        owner = next((root for root in roots_resolved if target.is_relative_to(root)), None)
        workdir = owner if owner is not None else roots_resolved[0]

    effective_timeout = timeout if timeout is not None else config.get_exec_timeout()
    if not 0 < effective_timeout <= MAX_TIMEOUT_SECONDS:
        # 配置来源（CRW_EXEC_TIMEOUT）与模型传参同标准校验：非法配置不应伪装成"执行超时"，
        # 而应给出指向配置的可读错误
        source = "timeout 参数" if timeout is not None else "环境变量 CRW_EXEC_TIMEOUT"
        return (
            f"{fs.ERROR_PREFIX}{source}必须是 1~{MAX_TIMEOUT_SECONDS} 的整数秒，"
            f"当前值：{effective_timeout}"
        )
    exit_code, stdout, stderr = _run_subprocess(command, effective_timeout, cwd=workdir)
    return _format_result(exit_code=exit_code, stdout=stdout, stderr=stderr)


def _run_subprocess(command: list[str], timeout_seconds: int, *, cwd: Path) -> tuple[int, str, str]:
    """运行子进程并收集输出；任何退出路径都杀树并有限回收。

    超时强杀整棵进程树后以异常上抛，由 dispatch 兜底；Ctrl+C 等中断同样先杀树、再有限
    等待回收管道，然后上抛原始异常——不留残留进程与管道；杀树失败时回收等待超时后放弃
    （有限等待，不阻塞中断响应）。
    """
    environment = {**_sanitized_env(), "PYTHONIOENCODING": "utf-8"}
    popen_kwargs: dict[str, Any] = {}
    if sys.platform != "win32":
        # POSIX：独立进程组启动，支撑 killpg 整树强杀（见 _kill_process_tree）
        popen_kwargs["start_new_session"] = True
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        cwd=str(cwd),
        **popen_kwargs,
    )
    readers = _start_readers(process)
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        kill_detail = _kill_process_tree(process)
        # 杀树成功后读取线程随即自然结束；失败（进程仍在跑）只有限等待后放弃，
        # 超时信息本身已回传模型，残余输出不再有价值
        _join_readers(readers, KILL_REAP_SECONDS)
        detail = f"（进程树终止失败：{kill_detail}）" if kill_detail else ""
        raise RuntimeError(
            f"执行超时（{timeout_seconds}s），已强制终止进程树{detail}。"
            "若代码含死循环或长阻塞，请修复后再试。"
        ) from None
    except BaseException:
        # 中断 / 请求失败等路径：先杀树并回收管道再上抛，不掩盖中断本身
        with suppress(Exception):
            _kill_process_tree(process)
        _join_readers(readers, KILL_REAP_SECONDS)
        raise
    _join_readers(readers, KILL_REAP_SECONDS)
    return (
        process.returncode if process.returncode is not None else -1,
        _reader_text(readers[0]),
        _reader_text(readers[1]),
    )


def _start_readers(process: subprocess.Popen[str]) -> tuple[Any, Any]:
    """为 stdout / stderr 各起一个读取线程：有界留存 + 持续排空。

    排空是必需的：子进程写到管道满会阻塞，只"停止读取"会让它卡死；因此超限后继续读、
    只是不再留存。线程是守护线程，主流程异常退出时不会被拖住。
    """
    readers = []
    for stream in (process.stdout, process.stderr):
        reader = {"text": "", "stream": stream, "thread": None}

        def _pump(reader: dict[str, Any] = reader) -> None:
            chunks: list[str] = []
            size = 0
            source = reader["stream"]
            while True:
                chunk = source.read(_READ_CHUNK_CHARS)
                if not chunk:
                    break
                if size < MAX_CAPTURE_CHARS:
                    keep = chunk[: MAX_CAPTURE_CHARS - size]
                    chunks.append(keep)
                    size += len(keep)
            reader["text"] = "".join(chunks)

        thread = threading.Thread(target=_pump, daemon=True)
        reader["thread"] = thread
        thread.start()
        readers.append(reader)
    return readers[0], readers[1]


def _join_readers(readers: tuple[Any, Any], timeout: float) -> None:
    """有限等待读取线程收尾（不阻塞中断响应）。"""
    for reader in readers:
        thread = reader["thread"]
        if thread is not None:
            thread.join(timeout)


def _reader_text(reader: Any) -> str:
    return str(reader.get("text") or "")


def _kill_process_tree(process: subprocess.Popen[str]) -> str | None:
    """强杀进程树；返回失败说明，成功（或进程已自行退出）返回 None。

    Windows 下用 ``taskkill /T /F`` 连子进程一起杀（``proc.kill()`` 只杀直接子进程，
    孙进程会残留）；POSIX 下子进程以独立进程组启动（``start_new_session``），对整组
    SIGKILL 同样覆盖孙进程（子孙若自建进程组则仍不可及，与 taskkill /T 的边界一致）。
    """
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",  # 中文版 taskkill 输出为 GBK，只作错误详情，容错解码
                check=False,  # 失败详情在下方按进程是否仍存活判定，不走异常路径
            )
        except Exception as killed:  # noqa: BLE001 — 杀树是兜底路径，任何失败都转为详情文本
            return f"taskkill 调用失败：{killed}"
        if result.returncode != 0 and process.poll() is None:
            detail = (result.stderr or result.stdout or "").strip()
            return detail or f"taskkill 退出码 {result.returncode}"
        return None
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        return None  # 进程已自行退出：视为成功（与 Windows 分支的判定一致）
    except OSError as exc:  # 权限等异常：转为详情文本，不替换掉调用方报告的超时/中断
        return f"killpg 失败：{exc}"
    return None


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """单路输出截断（防刷屏）；截断处附提示让模型知道内容不完整。"""
    if len(text) <= limit:
        return text
    return text[:limit] + _TRUNCATION_NOTICE.format(limit=limit)


def _format_result(exit_code: int, stdout: str, stderr: str) -> str:
    """归一为模型可读的三段结构；空输出显式标注（空与非空对模型含义不同）。"""
    stdout = _truncate(stdout.rstrip("\n"))
    stderr = _truncate(stderr.rstrip("\n"))
    return (
        f"退出码：{exit_code}\n"
        f"stdout：\n{stdout if stdout else '（空）'}\n"
        f"stderr：\n{stderr if stderr else '（空）'}"
    )
