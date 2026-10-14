"""文件类工具：list_dir / read_file / write_file；路径以上下文根解析，越界一律拒绝。

所有 handler 返回字符串（错误也是字符串，数据封装，由注册表统一交给模型）：
- 相对路径以**上下文根**解析（运行工作区；`crw review` 独立执行时为审查目标所在目录）；
- 规范化后越出根（含 ``..``）、绝对路径、``~`` 展开一律结构化拒绝（Design §8.4）；
- 隐藏文件（任一路径成分以 ``.`` 开头）与数据目录内路径拒绝读写——防凭证类文件
  进入模型上下文与运行日志；
- 行号口径（1-based、右对齐竖线分隔）与审查 prompt 的代码附行号方式一致；
- ``write_file`` 自带原子写（临时文件 + ``os.replace``）：本层不得 import workflow 层
  （依赖方向 §3.1），因此不复用 ``workflow.records`` 的写盘原语。
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Sequence
from pathlib import Path

from .. import config

#: 单次读取上限，超出提示模型用 start/end 分段
MAX_READ_LINES = 400
#: 单文件读取上限：读取本身须有界（先判大小再读，防大文件进内存）
MAX_READ_BYTES = 10 * 1024 * 1024
#: 单文件写入上限：与读取同量级——写盘的有界性由工具层自己保证，不依赖上层契约
MAX_WRITE_BYTES = 10 * 1024 * 1024
#: 二进制嗅探采样窗口：文件头部含 null 字节即判为二进制
TEXT_SNIFF_BYTES = 8192

#: list_dir 忽略这些目录与所有点开头的隐藏目录
IGNORED_DIRS = {
    ".venv",
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    # 常见的大体积构建 / 依赖目录：列出来只会刷爆模型上下文，对审查没有价值
    "node_modules",
    "dist",
    "build",
    "vendor",
}

#: read_file 返回文本的字符上限：行数上限（400 行）挡不住"少换行的大文件"（单行 JSON /
#: minified 产物），再设字符闸门保证回传模型的内容有界
MAX_READ_OUTPUT_CHARS = 60_000
#: list_dir 树形输出的行数上限（超大目录截断并提示按子目录深入）
MAX_LIST_LINES = 400

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
        return None, f"{ERROR_PREFIX}隐藏文件 / 隐藏目录不可读写：{raw!r}"
    if _in_forbidden_data_dir(target, owner):
        return None, f"{ERROR_PREFIX}数据目录（配置与运行记录）内的路径不可读写：{raw!r}"
    return target, None


def list_dir(roots: Sequence[Path], path: str = ".") -> str:
    """返回目录的两层树（本层 + 每个子目录的一层），忽略忽略名单与隐藏目录。

    输出有行数上限：超大目录（真实仓库常见）截断并提示用 ``path`` 定位子目录——
    列表刷爆模型上下文比"看不全"更糟。
    """
    if not isinstance(path, str) or not path.strip():
        return f"{ERROR_PREFIX}path 必须是非空字符串（相对上下文根）"
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
            lines.extend(_list_child_dir(entry, budget=MAX_LIST_LINES - len(lines)))
        else:
            lines.append(f"├── {entry.name}")
        if len(lines) >= MAX_LIST_LINES:
            lines.append(
                f"{ERROR_PREFIX[2:]}（目录过大，已按 {MAX_LIST_LINES} 行上限截断：请用 path 指定子目录）"
            )
            break
    return "\n".join(lines)


def _int_or_error(value: object, name: str) -> str | None:
    """行号参数校验：模型可能把整数写成字符串，给出可读原因而不是让比较抛 TypeError。"""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return f"{ERROR_PREFIX}{name} 必须是 1-based 整数行号，当前值：{value!r}"
    return None


def read_file(
    roots: Sequence[Path], path: str, start: int | None = None, end: int | None = None
) -> str:
    """带行号读取文本文件；1-based 闭区间 ``[start, end]``，单次上限 ``MAX_READ_LINES`` 行。

    输出另有 ``MAX_READ_OUTPUT_CHARS`` 字符上限：行数上限挡不住少换行的大文件
    （单行 JSON / minified 产物），字符闸门保证回传模型的内容有界。二进制文件返回
    结构化错误（null 字节嗅探：``errors='replace'`` 会把二进制读成乱码刷爆上下文，
    嗅探才能给模型可操作的提示）。
    """
    if not isinstance(path, str) or not path.strip():
        return f"{ERROR_PREFIX}path 必须是非空字符串（相对上下文根）"
    for value, name in ((start, "start"), (end, "end")):
        type_error = _int_or_error(value, name)
        if type_error is not None:
            return type_error
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
    if len(output) > MAX_READ_OUTPUT_CHARS:
        output = (
            output[:MAX_READ_OUTPUT_CHARS]
            + f"\n（输出超过 {MAX_READ_OUTPUT_CHARS} 字符上限，已截断；"
            "该文件换行很少，请改用 offset 无法细分的部分——建议对超大单行文件用其他方式处理）"
        )
    return output


def write_file(roots: Sequence[Path], path: str, content: str) -> str:
    """把文本写入上下文根内的文件（原子写）；返回写入摘要。

    路径解析与拒绝名单同 ``read_file``（Design §8.4）：越界 / 绝对路径 / ``~`` / 隐藏文件
    与数据目录一律拒绝——所以调用方给什么根就只能写进什么根。修复节点的根是本轮产出的
    版本目录，写入因此天然限定在该目录之内（Design §4.2.2）。父目录不存在时按需创建
    （版本内的新文件可以是子目录里的新条目）；已有文件整份覆盖。
    """
    if not isinstance(path, str) or not path.strip():
        return f"{ERROR_PREFIX}path 必须是非空字符串（相对上下文根）"
    if not isinstance(content, str):
        return (
            f"{ERROR_PREFIX}content 必须是字符串（文件的完整内容），收到：{type(content).__name__}"
        )
    target, error = resolve_in_roots(roots, path)
    if error is not None:
        return error
    assert target is not None  # resolve_in_roots 的错误分支已返回
    if target.is_dir():
        return f"{ERROR_PREFIX}路径是目录，无法写入：{path!r}"
    if target.name == "manifest.json":
        # manifest 是引擎落盘时写的权威文件清单（Design §4.2.2）：写它既会被重算覆盖，
        # 也给了"把文件藏出清单"的可乘之机，直接在入口拒掉并说明原因
        return f"{ERROR_PREFIX}manifest.json 由引擎维护（版本的权威文件清单），不可写入"
    if content and not content.endswith("\n"):
        content += "\n"  # 与版本落盘的归一化口径一致（整版本逐字节比较才有意义）
    try:
        data = content.encode("utf-8")
    except UnicodeEncodeError as exc:
        # JSON 里的孤立代理字符（U+D800 那一类码位）能走到这里：编码失败也必须回传结构化
        # 错误，不能让异常冲出 handler（注册表兜底会把归因推成"工具执行失败"）
        return f"{ERROR_PREFIX}content 无法按 UTF-8 编码，请给出合法文本：{exc}"
    if len(data) > MAX_WRITE_BYTES:
        return (
            f"{ERROR_PREFIX}{path!r} 约 {len(data) / 1048576:.1f} MB，超过单次写入上限"
            f"（{MAX_WRITE_BYTES // 1048576} MB）。请缩小写入范围。"
        )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(target, content)
    except OSError as exc:
        return f"{ERROR_PREFIX}写入 {path!r} 失败：{exc}"
    return f"已写入 {path}（{len(split_lines(content))} 行，{len(data)} 字节）"


def _atomic_write_text(path: Path, text: str) -> None:
    """原子写文本（临时文件 + ``os.replace``；失败尽力清理临时文件）。

    与 ``workflow.records`` 的实现同语义但**自带一份**：工具层不得向上 import（§3.1），
    而"不产生半写文件"是 write_file 对自己落盘的承诺，不能依赖调用方谨慎。临时名带 pid：
    并发（同一轮里两个写盘）不得互相踩同一个临时文件。
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name == "posix":
            with contextlib.suppress(OSError):
                tmp.chmod(0o600)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


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


def _list_child_dir(directory: Path, *, budget: int) -> list[str]:
    """子目录的一层内容（第二层），同样应用忽略规则；不递归第三层。

    ``budget`` 是剩余可用的行数预算：单个大目录（几万条目）不得在 list_dir 的
    外层截断检查之前把整份清单塞进输出——预算内截断并给提示。
    """
    try:
        children = sorted(directory.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
    except OSError:
        return ["│   └── （无法读取该子目录）"]
    lines: list[str] = []
    for child in children:
        if _is_skipped_entry(child):
            continue
        if len(lines) >= max(0, budget):
            lines.append("│   └── …（条目过多已截断）")
            break
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
    "MAX_WRITE_BYTES",
    "TEXT_SNIFF_BYTES",
    "file_line",
    "list_dir",
    "looks_textual",
    "read_file",
    "resolve_in_roots",
    "resolve_roots",
    "split_lines",
    "write_file",
]
