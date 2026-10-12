"""文件类工具：list_dir 与 read_file；路径以上下文根解析，越界一律拒绝。

所有 handler 返回字符串（错误也是字符串，数据封装，由注册表统一交给模型）：
- 相对路径以**上下文根**解析（运行工作区；`crw review` 独立执行时为审查目标所在目录）；
- 规范化后越出根（含 ``..``）、绝对路径、``~`` 展开一律结构化拒绝（Design §8.4）；
- 隐藏文件（任一路径成分以 ``.`` 开头）与数据目录内路径拒绝读取——防凭证类文件
  进入模型上下文与运行日志；
- 行号口径（1-based、右对齐竖线分隔）与审查 prompt 的代码附行号方式一致。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .. import config

#: 单次读取上限，超出提示模型用 start/end 分段
MAX_READ_LINES = 400
#: 单文件读取上限：读取本身须有界（先判大小再读，防大文件进内存）
MAX_READ_BYTES = 10 * 1024 * 1024
#: 二进制嗅探采样窗口：文件头部含 null 字节即判为二进制
TEXT_SNIFF_BYTES = 8192

#: list_dir 忽略这些目录与所有点开头的隐藏目录
IGNORED_DIRS = {".venv", "__pycache__", ".git", ".pytest_cache", ".ruff_cache", ".mypy_cache"}

#: 结构化错误文本的统一前缀（注册表与运行时按它区分"工具失败"与"正常结果"）
ERROR_PREFIX = "错误："


def split_lines(text: str) -> list[str]:
    """按真实换行符（``\\r\\n``、``\\r``、``\\n``）切分行；尾部换行不产生空尾行。

    不用 ``str.splitlines()``：它把 ``\\x0c``（form feed）、U+2028 等 Unicode 换行类
    字符也当行界，而行号口径（编辑器、审查 location）只认真实换行——含这些字符的
    文件会让 read_file 与审查 prompt 的行号整体漂移。
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def resolve_roots(roots: Sequence[Path]) -> list[Path]:
    """上下文根归一为绝对路径（去重、保持顺序）。"""
    resolved: list[Path] = []
    for root in roots:
        item = Path(root).expanduser().resolve()
        if item not in resolved:
            resolved.append(item)
    return resolved


def resolve_in_roots(roots: Sequence[Path], raw: str) -> tuple[Path | None, str | None]:
    """把工具传入的路径解析到上下文根之内。

    Returns:
        ``(路径, None)`` 解析成功；``(None, 错误文本)`` 拒绝（绝对路径 / ``~`` /
        越界 / 隐藏文件 / 数据目录）。
    """
    resolved_roots = resolve_roots(roots)
    text = (raw or ".").strip()
    if not text:
        text = "."
    if text.startswith("~"):
        return None, f"{ERROR_PREFIX}不支持 ~ 展开，请给出相对上下文根的路径：{raw!r}"
    candidate = Path(text)
    if candidate.is_absolute():
        return None, (
            f"{ERROR_PREFIX}不接受绝对路径，请给出相对上下文根的路径（如 code_v1.py）：{raw!r}"
        )
    # 多根（`crw review` 的目录并集）逐个试：优先取已存在的候选，其次取第一个未越界的候选
    # ——"文件在第二个根里"与"文件不存在但路径合法"两类情形都要能准确定位 / 准确拒绝。
    options: list[tuple[Path, Path]] = []
    for root in resolved_roots:
        candidate_path = (root / candidate).resolve()
        if candidate_path.is_relative_to(root):
            options.append((root, candidate_path))
    if not options:
        available = "、".join(str(root) for root in resolved_roots)
        return None, (
            f"{ERROR_PREFIX}路径越出可用范围（上下文根：{available}）：{raw!r}"
            "。相对路径基于上下文根解析，不允许 .. 回溯到根之外。"
        )
    owner, target = next(((root, path) for root, path in options if path.exists()), options[0])
    relative = target.relative_to(owner)
    if any(part.startswith(".") for part in relative.parts):
        return None, f"{ERROR_PREFIX}隐藏文件 / 隐藏目录不可读取：{raw!r}"
    if _in_forbidden_data_dir(target, owner):
        return None, f"{ERROR_PREFIX}数据目录（配置与运行记录）内的路径不可读取：{raw!r}"
    return target, None


def list_dir(roots: Sequence[Path], path: str = ".") -> str:
    """返回目录的两层树（本层 + 每个子目录的一层），忽略忽略名单与隐藏目录。"""
    root, error = resolve_in_roots(roots, path)
    if error is not None:
        return error
    assert root is not None  # resolve_in_roots 的错误分支已返回
    if not root.exists():
        return f"{ERROR_PREFIX}路径不存在：{path!r}"
    if not root.is_dir():
        return f"{ERROR_PREFIX}路径不是目录：{path!r}（读文件请用 read_file）"
    try:
        entries = sorted(root.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
    except OSError as exc:
        return f"{ERROR_PREFIX}无法列出目录 {path!r}：{exc}"
    relative_display = "." if root == resolve_roots(roots)[0] else _relative_display(root, roots)
    lines = [relative_display]
    for entry in entries:
        if _is_skipped_entry(entry):
            continue
        if entry.is_dir():
            lines.append(f"├── {entry.name}/")
            lines.extend(_list_child_dir(entry))
        else:
            lines.append(f"├── {entry.name}")
    return "\n".join(lines)


def read_file(
    roots: Sequence[Path], path: str, start: int | None = None, end: int | None = None
) -> str:
    """带行号读取文本文件；1-based 闭区间 ``[start, end]``，单次上限 ``MAX_READ_LINES`` 行。

    二进制文件返回结构化错误（null 字节嗅探：``errors='replace'`` 会把二进制读成乱码
    刷爆上下文，嗅探才能给模型可操作的提示）。
    """
    target, error = resolve_in_roots(roots, path)
    if error is not None:
        return error
    assert target is not None  # resolve_in_roots 的错误分支已返回
    if not target.exists():
        return f"{ERROR_PREFIX}文件不存在：{path!r}"
    if not target.is_file():
        return f"{ERROR_PREFIX}路径不是文件：{path!r}（列目录请用 list_dir）"
    try:
        # 先判大小再读：上限必须在读取之前生效，否则大文件已整文件进内存
        size = target.stat().st_size
    except OSError as exc:
        return f"{ERROR_PREFIX}无法读取文件 {path!r}：{exc}"
    if size > MAX_READ_BYTES:
        return (
            f"{ERROR_PREFIX}{target.name} 约 {size / 1048576:.1f} MB，超过单次读取上限"
            f"（{MAX_READ_BYTES // 1048576} MB）。请分段读取所需部分。"
        )
    try:
        raw = target.read_bytes()
    except OSError as exc:
        return f"{ERROR_PREFIX}无法读取文件 {path!r}：{exc}"
    if b"\x00" in raw:
        return (
            f"{ERROR_PREFIX}{target.name} 是二进制文件（含 null 字节），无法按文本读取。"
            "如需了解其结构请改用其他方式。"
        )
    text = raw.decode("utf-8", errors="replace")
    lines = split_lines(text)
    total = len(lines)
    if total == 0:
        # 空文件没有行区间可言，按区间规则会误报"start 不能大于 end"
        return f"{path}（空文件，共 0 行）"
    start_line, end_line = _normalize_range(start, end, total)
    range_error = _range_error(start, end, start_line, end_line, total)
    if range_error is not None:
        return range_error
    selected = lines[start_line - 1 : end_line]
    numbered = "\n".join(
        f"{index:>4} | {line}" for index, line in enumerate(selected, start=start_line)
    )
    output = f"{path}（共 {total} 行，显示 {start_line}-{end_line} 行）\n{numbered}"
    if end_line < total and end_line - start_line + 1 == MAX_READ_LINES:
        output += f"\n（已按单次 {MAX_READ_LINES} 行上限截断：请用 start={end_line + 1} 继续读取）"
    return output


def looks_textual(path: Path) -> bool:
    """采样文件头部判断是否为文本文件（与 read_file 同判据）。

    嗅探读取失败（权限等）按"是文本"处理：后续真正的读取会用可读错误如实报告原因，
    不在这里把 IO 故障伪装成"二进制文件"误导用户。
    """
    try:
        with path.open("rb") as handle:
            return b"\x00" not in handle.read(TEXT_SNIFF_BYTES)
    except OSError:
        return True


def file_line(path: str, line: int) -> str:
    """``路径:行号`` 的展示形式（审查 location 与提示词口径一致）。"""
    return f"{path}:{line}"


def _list_child_dir(directory: Path) -> list[str]:
    """子目录的一层内容（第二层），同样应用忽略规则；不递归第三层。"""
    try:
        children = sorted(directory.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
    except OSError:
        return ["│   └── （无法读取该子目录）"]
    lines: list[str] = []
    for child in children:
        if _is_skipped_entry(child):
            continue
        suffix = "/" if child.is_dir() else ""
        lines.append(f"│   ├── {child.name}{suffix}")
    if not lines:
        lines.append("│   ├── （空目录）")
    return lines


def _relative_display(target: Path, roots: Sequence[Path]) -> str:
    """target 相对某个上下文根的展示形式（不回显主机绝对路径）。"""
    for root in resolve_roots(roots):
        if target.is_relative_to(root):
            return target.relative_to(root).as_posix() or "."
    return "."


def _is_skipped_entry(entry: Path) -> bool:
    """列表条目是否跳过：忽略名单目录与一切点开头的条目。

    隐藏**文件**同样跳过：它们不可被 read_file 读取（Design §8.4 的拒绝名单），列出来只会
    引诱模型发起注定被拒的调用。
    """
    if entry.name.startswith("."):
        return True
    return entry.is_dir() and entry.name in IGNORED_DIRS


def _data_dirs() -> list[Path]:
    """需拒读的数据目录：``CRW_HOME`` / ``~/.crw`` / ``~/.cra``（Design §8.4）。"""
    candidates = [config.data_dir(), Path.home() / ".crw", Path.home() / ".cra"]
    resolved: list[Path] = []
    for item in candidates:
        path = item.expanduser()
        try:
            path = path.resolve()
        except OSError:  # pragma: no cover — 极端路径（父目录不可解析）
            continue
        if path not in resolved:
            resolved.append(path)
    return resolved


def _in_forbidden_data_dir(target: Path, owner: Path) -> bool:
    """目标是否落在数据目录内（``owner`` = 实际包含该目标的那个上下文根）。

    上下文根本身位于数据目录内时不适用（运行工作区即 ``~/.crw/runs/<id>/workspace``）：
    否则工作区文件一律读不到；此时越界由根检查负责。豁免只按**包含目标的那个根**判定，
    不因"另一个根恰好在数据目录内"而对所有根放宽。
    """
    for data_dir in _data_dirs():
        if target.is_relative_to(data_dir) and not owner.is_relative_to(data_dir):
            return True
    return False


def _normalize_range(start: int | None, end: int | None, total: int) -> tuple[int, int]:
    """把可选的 start/end 归一为 1-based 闭区间；end 超出文件按 EOF 截齐。"""
    start_line = 1 if start is None else start
    end_line = min(start_line + MAX_READ_LINES - 1, total) if end is None else end
    end_line = min(end_line, total)
    return start_line, end_line


def _range_error(
    start: int | None, end: int | None, start_line: int, end_line: int, total: int
) -> str | None:
    """非法区间返回结构化错误文本，合法区间返回 None。"""
    if (start is not None and start < 1) or (end is not None and end < 1):
        return (
            f"{ERROR_PREFIX}start/end 必须是 1-based 正整数（文件共 {total} 行），"
            f"当前 start={start}, end={end}"
        )
    if start is not None and start_line > total:
        return f"{ERROR_PREFIX}start={start} 超出文件范围（文件共 {total} 行）"
    if start_line > end_line:
        return (
            f"{ERROR_PREFIX}start 不能大于 end（1-based 闭区间，文件共 {total} 行），"
            f"当前 start={start}, end={end}"
        )
    if end_line - start_line + 1 > MAX_READ_LINES:
        return (
            f"{ERROR_PREFIX}请求区间 {start_line}-{end_line} 共 {end_line - start_line + 1} 行，"
            f"超过单次 {MAX_READ_LINES} 行上限，请缩小 start/end 范围分段读取"
        )
    return None


__all__ = [
    "ERROR_PREFIX",
    "IGNORED_DIRS",
    "MAX_READ_BYTES",
    "MAX_READ_LINES",
    "TEXT_SNIFF_BYTES",
    "file_line",
    "list_dir",
    "looks_textual",
    "read_file",
    "resolve_in_roots",
    "resolve_roots",
    "split_lines",
]
