"""工具层：上下文根路径语义、文件与执行工具、环境净化、注册表协议（Design §8.4、D9）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from crw import config
from crw.tools import build_registry, exec_py, fs


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """上下文根（模拟运行工作区）。"""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


class TestResolveInRoots:
    def test_relative_path_resolves_inside_the_root(self, root: Path) -> None:
        (root / "code_v1.py").write_text("print(1)\n", encoding="utf-8")

        target, error = fs.resolve_in_roots([root], "code_v1.py")

        assert error is None and target == root / "code_v1.py"

    def test_dot_means_the_root(self, root: Path) -> None:
        assert fs.resolve_in_roots([root], ".") == (root, None)
        assert fs.resolve_in_roots([root], "") == (root, None)

    @pytest.mark.parametrize("raw", ["../secret.py", "a/../../escape.py", ".."])
    def test_parent_traversal_is_rejected(self, root: Path, raw: str) -> None:
        target, error = fs.resolve_in_roots([root], raw)

        assert target is None
        assert error is not None and "越出可用范围" in error

    def test_absolute_path_is_rejected(self, root: Path) -> None:
        target, error = fs.resolve_in_roots([root], str(root / "code_v1.py"))

        assert target is None
        assert error is not None and "不接受绝对路径" in error

    def test_home_expansion_is_rejected(self, root: Path) -> None:
        target, error = fs.resolve_in_roots([root], "~/notes.txt")

        assert target is None
        assert error is not None and "~" in error

    @pytest.mark.parametrize("raw", [".env", "sub/.hidden", ".git/config"])
    def test_hidden_paths_are_rejected(self, root: Path, raw: str) -> None:
        target, error = fs.resolve_in_roots([root], raw)

        assert target is None
        assert error is not None and "隐藏" in error

    def test_second_root_is_accepted(self, tmp_path: Path, root: Path) -> None:
        other = tmp_path / "other"
        other.mkdir()
        (other / "b.py").write_text("print(2)\n", encoding="utf-8")

        target, error = fs.resolve_in_roots([root, other], "b.py")

        assert error is None and target == other / "b.py"

    def test_data_directory_outside_the_root_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """数据目录（配置与运行记录）不因根放宽而可读：防凭证类文件进模型上下文。"""
        data_home = tmp_path / "crw-data"
        data_home.mkdir()
        monkeypatch.setenv(config.HOME_ENV, str(data_home))
        wide_root = tmp_path  # 根放宽到临时目录，数据目录落在其中

        target, error = fs.resolve_in_roots([wide_root], "crw-data/models.json")

        assert target is None
        assert error is not None and "数据目录" in error

    def test_data_directory_containing_the_root_is_allowed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """运行工作区本身就在数据目录内（~/.crw/runs/<id>/workspace）：不得自锁。"""
        data_home = tmp_path / "crw-data"
        workspace = data_home / "runs" / "r-20261012-0001" / "workspace"
        workspace.mkdir(parents=True)
        monkeypatch.setenv(config.HOME_ENV, str(data_home))

        target, error = fs.resolve_in_roots([workspace], "code_v1.py")

        assert error is None and target == workspace / "code_v1.py"


class TestListDir:
    def test_two_level_tree_skips_hidden_and_ignored(self, root: Path) -> None:
        (root / "code_v1.py").write_text("print(1)\n", encoding="utf-8")
        (root / ".env").write_text("SECRET=1\n", encoding="utf-8")
        (root / "__pycache__").mkdir()
        sub = root / "pkg"
        sub.mkdir()
        (sub / "mod.py").write_text("x = 1\n", encoding="utf-8")

        text = fs.list_dir([root])

        assert "code_v1.py" in text
        assert "pkg/" in text and "mod.py" in text
        assert ".env" not in text
        assert "__pycache__" not in text

    def test_missing_path_and_non_directory_are_readable(self, root: Path) -> None:
        (root / "code_v1.py").write_text("print(1)\n", encoding="utf-8")

        assert "路径不存在" in fs.list_dir([root], "nope")
        assert "路径不是目录" in fs.list_dir([root], "code_v1.py")


class TestReadFile:
    def test_line_numbers_are_one_based(self, root: Path) -> None:
        (root / "code_v1.py").write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")

        text = fs.read_file([root], "code_v1.py")

        assert "1 | a = 1" in text
        assert "3 | c = 3" in text
        assert "共 3 行" in text

    def test_range_selection(self, root: Path) -> None:
        (root / "code_v1.py").write_text("a\nb\nc\nd\n", encoding="utf-8")

        text = fs.read_file([root], "code_v1.py", start=2, end=3)

        assert "2 | b" in text and "3 | c" in text
        assert "1 | a" not in text and "4 | d" not in text

    @pytest.mark.parametrize(
        ("start", "end", "match"),
        [(0, None, "1-based"), (1, 0, "1-based"), (99, None, "超出文件范围"), (2, 1, "不能大于")],
    )
    def test_illegal_ranges(
        self, root: Path, start: int | None, end: int | None, match: str
    ) -> None:
        (root / "code_v1.py").write_text("a\nb\n", encoding="utf-8")

        assert match in fs.read_file([root], "code_v1.py", start=start, end=end)

    def test_range_larger_than_the_limit_is_rejected(self, root: Path) -> None:
        (root / "code_v1.py").write_text(
            "".join(f"line {i}\n" for i in range(600)), encoding="utf-8"
        )

        text = fs.read_file([root], "code_v1.py", start=1, end=500)

        assert "超过单次" in text

    def test_default_read_is_truncated_at_the_line_limit(self, root: Path) -> None:
        (root / "code_v1.py").write_text(
            "".join(f"line {i}\n" for i in range(600)), encoding="utf-8"
        )

        text = fs.read_file([root], "code_v1.py")

        assert f"1-{fs.MAX_READ_LINES}" in text
        assert "继续读取" in text

    def test_binary_file_is_reported(self, root: Path) -> None:
        (root / "blob.bin").write_bytes(b"\x00\x01\x02")

        assert "二进制文件" in fs.read_file([root], "blob.bin")

    def test_oversized_file_is_reported_without_reading(
        self, monkeypatch: pytest.MonkeyPatch, root: Path
    ) -> None:
        monkeypatch.setattr(fs, "MAX_READ_BYTES", 10)
        (root / "big.py").write_text("print('x' * 100)\n", encoding="utf-8")

        assert "超过单次读取上限" in fs.read_file([root], "big.py")

    def test_empty_file_is_readable(self, root: Path) -> None:
        (root / "empty.py").write_text("", encoding="utf-8")

        assert "空文件" in fs.read_file([root], "empty.py")

    def test_missing_and_directory_are_readable(self, root: Path) -> None:
        (root / "pkg").mkdir()

        assert "文件不存在" in fs.read_file([root], "nope.py")
        assert "路径不是文件" in fs.read_file([root], "pkg")

    def test_looks_textual(self, root: Path) -> None:
        text_file = root / "a.py"
        text_file.write_text("print(1)\n", encoding="utf-8")
        binary = root / "b.bin"
        binary.write_bytes(b"\x00\x01")

        assert fs.looks_textual(text_file) is True
        assert fs.looks_textual(binary) is False


class TestSplitLines:
    @pytest.mark.parametrize("text", ["a\nb\n", "a\r\nb\r\n", "a\rb\r"])
    def test_real_newlines(self, text: str) -> None:
        assert fs.split_lines(text) == ["a", "b"]

    def test_unicode_line_separators_do_not_split(self) -> None:
        """U+2028 / form feed 不是真实换行：行号口径必须与编辑器一致。"""
        assert fs.split_lines("a\x0cb\u2028c") == ["a\x0cb\u2028c"]

    def test_empty_text(self) -> None:
        assert fs.split_lines("") == []


class TestRunPython:
    def test_code_and_path_are_mutually_exclusive(self, root: Path) -> None:
        assert "二选一" in exec_py.run_python([root])
        assert "二选一" in exec_py.run_python([root], code="print(1)", path="code_v1.py")

    def test_inline_code_runs_in_the_context_root(self, root: Path) -> None:
        (root / "data.txt").write_text("hello\n", encoding="utf-8")

        result = exec_py.run_python(
            [root], code="print(open('data.txt', encoding='utf-8').read().strip())"
        )

        assert "退出码：0" in result
        assert "hello" in result

    def test_path_must_be_a_python_file_inside_the_root(self, root: Path, tmp_path: Path) -> None:
        (root / "note.txt").write_text("hi\n", encoding="utf-8")
        outside = tmp_path / "outside.py"
        outside.write_text("print('x')\n", encoding="utf-8")

        assert "不是 Python 文件" in exec_py.run_python([root], path="note.txt")
        assert "越出可用范围" in exec_py.run_python([root], path="../outside.py")

    def test_script_path_runs_with_the_root_as_cwd(self, root: Path) -> None:
        (root / "script.py").write_text(
            "import pathlib\nprint(pathlib.Path.cwd().name)\n", encoding="utf-8"
        )

        result = exec_py.run_python([root], path="script.py")

        assert "退出码：0" in result and "workspace" in result

    def test_timeout_parameter_is_validated(self, root: Path) -> None:
        assert "1~60" in exec_py.run_python([root], code="print(1)", timeout=0)
        assert "1~60" in exec_py.run_python([root], code="print(1)", timeout=120)

    def test_illegal_configured_timeout_is_a_readable_error(
        self, monkeypatch: pytest.MonkeyPatch, root: Path
    ) -> None:
        monkeypatch.setenv("CRW_EXEC_TIMEOUT", "0")

        result = exec_py.run_python([root], code="print(1)")

        assert "CRW_EXEC_TIMEOUT" in result

    def test_output_is_truncated(self, root: Path) -> None:
        result = exec_py.run_python([root], code="print('x' * 5000)")

        assert "已截断" in result

    def test_non_zero_exit_is_reported(self, root: Path) -> None:
        result = exec_py.run_python([root], code="raise SystemExit(3)")

        assert "退出码：3" in result

    def test_stderr_is_captured(self, root: Path) -> None:
        result = exec_py.run_python([root], code="import sys; sys.stderr.write('boom')")

        assert "boom" in result


class TestEnvSanitisation:
    @pytest.mark.parametrize(
        "name",
        [
            "CRW_DEEPSEEK_API_KEY",
            "CRW_MODEL",
            "CRA_DEEPSEEK_API_KEY",
            "SE_CodeAgent",
            "MY_API_KEY",
            "MY-API-KEY",
            "some_apikey",
            "openai_api_key",
        ],
    )
    def test_credential_like_names_are_stripped(self, name: str) -> None:
        assert exec_py._is_credential_var(name) is True

    @pytest.mark.parametrize("name", ["PATH", "HOME", "PYTHONPATH", "TEMP"])
    def test_ordinary_names_survive(self, name: str) -> None:
        assert exec_py._is_credential_var(name) is False

    def test_child_process_cannot_see_credentials(
        self, monkeypatch: pytest.MonkeyPatch, root: Path
    ) -> None:
        """被审查代码可以 print(os.environ)：本程序与既有工具链的变量都不得进子进程。"""
        monkeypatch.setenv("CRW_DEEPSEEK_API_KEY", "placeholder-crw")
        monkeypatch.setenv("CRA_DEEPSEEK_API_KEY", "placeholder-cra")
        monkeypatch.setenv("SE_CodeAgent", "placeholder-legacy")
        monkeypatch.setenv("CRW_HOME", str(root))
        monkeypatch.setenv("SOME_API_KEY", "placeholder-generic")
        monkeypatch.setenv("CRW_MARKER", "visible-if-leaked")

        result = exec_py.run_python(
            [root],
            code="import os; print('|'.join(sorted(k for k in os.environ if k.startswith(('CRW_', 'CRA_')) or 'API' in k or 'CodeAgent' in k)))",
        )

        assert "placeholder" not in result
        assert "CRW_MARKER" not in result
        assert "退出码：0" in result

    def test_sanitised_env_keeps_ordinary_variables(self) -> None:
        environment = exec_py._sanitized_env()

        assert "PATH" in environment or sys.platform == "win32"


class TestRegistry:
    def test_default_registry_has_the_three_tools(self, root: Path) -> None:
        assert build_registry([root]).names() == ["list_dir", "read_file", "run_python"]

    def test_no_exec_drops_the_execution_tool(self, root: Path) -> None:
        """``--no-exec`` 的唯一闸口在注册表组装层（Design §8.1）。"""
        names = build_registry([root], allow_exec=False).names()

        assert names == ["list_dir", "read_file"]
        assert all(
            "run_python" not in schema["function"]["name"]
            for schema in build_registry([root], allow_exec=False).schemas()
        )

    def test_per_node_tool_selection(self, root: Path) -> None:
        assert build_registry([root], names=()).names() == []
        assert build_registry([root], names=("read_file", "run_python")).names() == [
            "read_file",
            "run_python",
        ]

    def test_dispatch_runs_a_tool(self, root: Path) -> None:
        (root / "code_v1.py").write_text("print(1)\n", encoding="utf-8")

        result = build_registry([root]).dispatch("read_file", json.dumps({"path": "code_v1.py"}))

        assert "1 | print(1)" in result

    def test_dispatch_reports_unknown_tool_readably(self, root: Path) -> None:
        result = build_registry([root]).dispatch("deploy", "{}")

        assert result.startswith(fs.ERROR_PREFIX)
        assert "未注册的工具" in result and "read_file" in result

    def test_dispatch_reports_bad_arguments_readably(self, root: Path) -> None:
        registry = build_registry([root])

        assert "不是合法 JSON" in registry.dispatch("read_file", "{oops")
        assert "必须是 JSON 对象" in registry.dispatch("read_file", "[1, 2]")
        assert "执行失败" in registry.dispatch("read_file", json.dumps({"path": 12}))

    def test_tool_errors_never_raise(self, root: Path) -> None:
        """工具异常一律结构化回传模型：运行不因工具失败而崩（Design §8.4）。"""
        registry = build_registry([root])

        result = registry.dispatch("run_python", json.dumps({"code": "import sys; sys.exit(1)"}))

        assert isinstance(result, str)

    def test_arguments_preview_is_truncated(self) -> None:
        from crw.tools import arguments_preview

        assert arguments_preview('{"path": "a.py"}') == '{"path": "a.py"}'
        assert arguments_preview("x" * 500).endswith("…")


class TestRunPythonBoundaries:
    def test_timeout_must_be_an_integer(self, root: Path) -> None:
        """模型可能把整数参数写成字符串 / 布尔：给可读错误，而不是让比较抛 TypeError。"""
        assert "整数秒" in exec_py.run_python([root], code="print(1)", timeout="10")  # type: ignore[arg-type]
        assert "整数秒" in exec_py.run_python([root], code="print(1)", timeout=True)  # type: ignore[arg-type]

    def test_timeout_kills_the_process_tree_via_the_registry(self, root: Path) -> None:
        """超时强杀仍然有效（本轮把 communicate 换成 wait + 读取线程，契约不变）。

        走注册表入口：超时以异常上抛、由 dispatch 结构化为 ``错误：`` 文本回传模型，
        即模型实际看到的那条路径。
        """
        result = build_registry([root]).dispatch(
            "run_python", json.dumps({"code": "while True: pass", "timeout": 1})
        )

        assert result.startswith(fs.ERROR_PREFIX)
        assert "执行超时" in result

    def test_missing_context_root_is_a_structured_error(self) -> None:
        assert "未提供上下文根" in exec_py.run_python([], code="print(1)")

    def test_huge_output_is_bounded_before_display_truncation(self, root: Path) -> None:
        """读取阶段就有上限：超限后继续排空管道但不再留存，返回文本仍是展示级截断。"""
        result = exec_py.run_python([root], code="print('x' * 500000)")

        assert "已截断" in result
        assert len(result) < exec_py.MAX_CAPTURE_CHARS  # 远小于子进程实际输出

    def test_large_output_does_not_block_the_child(self, root: Path) -> None:
        """子进程写满管道也不应卡死（读取线程持续排空）。"""
        result = exec_py.run_python(
            [root],
            code="import sys\nfor _ in range(2000): sys.stdout.write('y' * 1000)\nprint('done')",
            timeout=30,
        )

        assert "退出码：0" in result
