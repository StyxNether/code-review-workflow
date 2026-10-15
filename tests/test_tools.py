"""工具层：上下文根路径语义、文件与执行工具、环境净化、注册表协议。"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import pytest

from crw import config
from crw.tools import ALL_TOOL_NAMES, WRITE_FILE_TOOL, build_registry, exec_py, fs


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


def _raise_os_error(*args: object, **kwargs: object) -> None:
    """顶替 ``os.replace``：模拟替换阶段的文件系统故障。"""
    raise OSError("模拟替换失败")


def _temporary_files(directory: Path) -> list[str]:
    """目录树里残留的临时文件名——原子写失败后必须为空（不得留下半写内容）。"""
    return sorted(item.name for item in directory.rglob("*.tmp"))


class TestWriteFile:
    """修复节点的写盘入口：落盘边界 = 上下文根，且不产生半写文件。"""

    def test_writes_content_and_reports_lines_and_bytes(self, root: Path) -> None:
        """返回文本要报出路径、行数与落盘字节数（摘要形态不限，数字必须对得上磁盘）。"""
        result = fs.write_file([root], "main.py", "a = 1\nb = 2\nc = 3")

        written = (root / "main.py").read_bytes()
        assert written == b"a = 1\nb = 2\nc = 3\n"
        assert not result.startswith(fs.ERROR_PREFIX)
        assert "main.py" in result
        assert "3" in result  # 三行
        assert "18" in result  # 15 个可见字符 + 两个内部换行 + 补上的结尾换行

    def test_byte_count_is_utf8_bytes_not_characters(self, root: Path) -> None:
        """字节数按 UTF-8 算：4 个汉字是 12 字节再加补上的结尾换行。"""
        result = fs.write_file([root], "a.py", "中文内容")

        assert (root / "a.py").read_bytes() == "中文内容\n".encode()
        assert "13" in result and "4" not in result

    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            ("print(1)", b"print(1)\n"),
            ("print(2)\n", b"print(2)\n"),
            ("print(3)\n\n", b"print(3)\n\n"),
        ],
    )
    def test_trailing_newline_is_added_exactly_once(
        self, root: Path, content: str, expected: bytes
    ) -> None:
        """缺结尾换行补一个、已有则不重复补——与版本落盘的归一化同口径，逐字节比较才有意义。"""
        fs.write_file([root], "a.py", content)

        assert (root / "a.py").read_bytes() == expected

    def test_new_nested_path_creates_parent_directories(self, root: Path, tmp_path: Path) -> None:
        """版本内的新文件可以落在子目录里；父目录只建在上下文根之内。"""
        result = fs.write_file([root], "pkg/sub/mod.py", "x = 1")

        assert not result.startswith(fs.ERROR_PREFIX)
        assert (root / "pkg" / "sub" / "mod.py").read_text(encoding="utf-8") == "x = 1\n"
        assert [item.name for item in tmp_path.iterdir()] == ["workspace"]

    def test_existing_file_is_replaced_wholesale(self, root: Path) -> None:
        fs.write_file([root], "a.py", "print('这一行应当整份消失')\n")

        result = fs.write_file([root], "a.py", "x = 1")

        assert not result.startswith(fs.ERROR_PREFIX)
        assert (root / "a.py").read_bytes() == b"x = 1\n"

    def test_two_roots_confine_the_write_to_the_owner_root(
        self, root: Path, tmp_path: Path
    ) -> None:
        """多上下文根时写入落在其中的一个根内，绝不落在根之外（修复节点的根即版本目录）。"""
        other = tmp_path / "other"
        other.mkdir()

        result = fs.write_file([root, other], "a.py", "print(1)")

        assert not result.startswith(fs.ERROR_PREFIX)
        assert (root / "a.py").is_file()
        assert list(other.iterdir()) == []

    def test_absolute_path_is_rejected(self, root: Path, tmp_path: Path) -> None:
        outside = tmp_path / "outside.py"

        result = fs.write_file([root], str(outside), "print(1)")

        assert result.startswith(fs.ERROR_PREFIX)
        assert "不接受绝对路径" in result
        assert not outside.exists()

    @pytest.mark.parametrize("raw", ["../escape.py", "pkg/../../escape.py", "..", "~/notes.txt"])
    def test_paths_escaping_the_root_are_rejected(
        self, root: Path, tmp_path: Path, raw: str
    ) -> None:
        result = fs.write_file([root], raw, "print(1)")

        assert result.startswith(fs.ERROR_PREFIX)
        assert list(root.iterdir()) == []
        assert not (tmp_path / "escape.py").exists()
        assert not (tmp_path / "notes.txt").exists()

    @pytest.mark.parametrize("raw", [".env", "pkg/.hidden", ".git/config"])
    def test_hidden_targets_are_rejected(self, root: Path, raw: str) -> None:
        """拒绝名单与读取同口径：凭证类隐藏文件不得被模型写进上下文根。"""
        result = fs.write_file([root], raw, "SECRET=1")

        assert result.startswith(fs.ERROR_PREFIX)
        assert "隐藏" in result
        assert list(root.iterdir()) == []

    def test_data_directory_target_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """数据目录（配置与运行记录）内的路径不可写：模型不得改写 key 文件或运行记录。"""
        data_home = tmp_path / "crw-data"
        data_home.mkdir()
        monkeypatch.setenv(config.HOME_ENV, str(data_home))

        result = fs.write_file([tmp_path], "crw-data/models.json", '{"api_key": "x"}')

        assert result.startswith(fs.ERROR_PREFIX)
        assert "数据目录" in result
        assert not (data_home / "models.json").exists()

    def test_directory_target_is_rejected(self, root: Path) -> None:
        (root / "pkg").mkdir()

        result = fs.write_file([root], "pkg", "print(1)")

        assert "路径是目录" in result
        assert list((root / "pkg").iterdir()) == []

    @pytest.mark.parametrize("content", [123, None, ["print(1)"], b"bytes"])
    def test_non_string_content_is_rejected(self, root: Path, content: object) -> None:
        result = fs.write_file([root], "a.py", content)

        assert "content 必须是字符串" in result
        assert list(root.iterdir()) == []

    @pytest.mark.parametrize("path", ["", "   ", 12, None])
    def test_missing_or_non_string_path_is_rejected(self, root: Path, path: object) -> None:
        result = fs.write_file([root], path, "print(1)")

        assert "path 必须是非空字符串" in result
        assert list(root.iterdir()) == []

    def test_oversized_content_is_rejected_before_writing(
        self, monkeypatch: pytest.MonkeyPatch, root: Path
    ) -> None:
        """写入的有界性由工具层自己保证（不依赖上层契约）：超限既不落盘也不留临时文件。"""
        monkeypatch.setattr(fs, "MAX_WRITE_BYTES", 64)

        result = fs.write_file([root], "a.py", "x" * 100)

        assert "超过单次写入上限" in result
        assert not (root / "a.py").exists()
        assert _temporary_files(root) == []

    def test_failed_replace_keeps_the_previous_file_and_leaves_no_temp(
        self, monkeypatch: pytest.MonkeyPatch, root: Path
    ) -> None:
        """原子写的承诺：替换失败时旧内容原样保留、临时文件被清掉，失败以结构化文本回传。"""
        (root / "a.py").write_text("old\n", encoding="utf-8")
        monkeypatch.setattr(os, "replace", _raise_os_error)

        result = fs.write_file([root], "a.py", "new content\n")

        assert result.startswith(fs.ERROR_PREFIX)
        assert (root / "a.py").read_text(encoding="utf-8") == "old\n"
        assert _temporary_files(root) == []

    def test_interrupted_write_propagates_and_leaves_no_temp_file(
        self, monkeypatch: pytest.MonkeyPatch, root: Path
    ) -> None:
        """落盘原子性对中断同样成立：异常上抛（不伪装成写入成功）且不留半写临时文件。"""

        def _interrupt(*args: object, **kwargs: object) -> None:
            raise KeyboardInterrupt("模拟中断")

        monkeypatch.setattr(os, "replace", _interrupt)

        with pytest.raises(KeyboardInterrupt):
            fs.write_file([root], "a.py", "print(1)")

        assert not (root / "a.py").exists()
        assert _temporary_files(root) == []


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

        # 配置层给可读的"必须是正整数"结论（不再伪装成执行超时）
        assert "CRW_EXEC_TIMEOUT" in result and "正整数" in result

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
        """普通变量原样保留：按大小写无关判定（Windows 上键名常见为 Path）。"""
        environment = exec_py._sanitized_env()

        names = {name.upper() for name in environment}
        assert "PATH" in names
        if sys.platform == "win32":
            assert "SYSTEMROOT" in names  # Windows 子进程正常运行所必需


class TestRegistry:
    def test_default_registry_has_the_four_tools(self, root: Path) -> None:
        """默认装配含 write_file：修复节点只写改动的文件（写盘形态）。"""
        assert build_registry([root]).names() == [
            "list_dir",
            "read_file",
            "write_file",
            "run_python",
        ]
        assert build_registry([root]).names() == list(ALL_TOOL_NAMES)

    def test_write_file_schema_requires_path_and_content(self, root: Path) -> None:
        schema = next(
            item
            for item in build_registry([root]).schemas()
            if item["function"]["name"] == WRITE_FILE_TOOL
        )

        assert schema["function"]["parameters"]["required"] == ["path", "content"]
        assert set(schema["function"]["parameters"]["properties"]) == {"path", "content"}

    def test_no_exec_drops_the_execution_tool(self, root: Path) -> None:
        """``--no-exec`` 的唯一闸口在注册表组装层：只关执行，不动写盘。"""
        names = build_registry([root], allow_exec=False).names()

        assert names == ["list_dir", "read_file", "write_file"]
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
        assert build_registry([root], names=(WRITE_FILE_TOOL,)).names() == ["write_file"]

    def test_dispatch_runs_a_tool(self, root: Path) -> None:
        (root / "code_v1.py").write_text("print(1)\n", encoding="utf-8")

        result = build_registry([root]).dispatch("read_file", json.dumps({"path": "code_v1.py"}))

        assert "1 | print(1)" in result

    def test_dispatch_writes_through_the_registered_handler(self, root: Path) -> None:
        """模型走的正是注册表入口：write_file 经 dispatch 落盘并回摘要。"""
        registry = build_registry([root])

        result = registry.dispatch(
            "write_file", json.dumps({"path": "pkg/mod.py", "content": "VALUE = 42"})
        )

        assert not result.startswith(fs.ERROR_PREFIX)
        assert "pkg/mod.py" in result
        assert (root / "pkg" / "mod.py").read_bytes() == b"VALUE = 42\n"

    def test_dispatch_reports_bad_write_arguments_readably(self, root: Path) -> None:
        """缺参数 / 非法 JSON / 越界路径都以 ``错误：`` 文本回传，不抛出。"""
        registry = build_registry([root])

        assert "执行失败" in registry.dispatch("write_file", "{}")
        assert "content" in registry.dispatch("write_file", json.dumps({"path": "a.py"}))
        assert "不是合法 JSON" in registry.dispatch("write_file", "{oops")
        assert "必须是 JSON 对象" in registry.dispatch("write_file", '["a.py", "print(1)"]')
        assert registry.dispatch(
            "write_file", json.dumps({"path": "../escape.py", "content": "print(1)"})
        ).startswith(fs.ERROR_PREFIX)
        assert list(root.iterdir()) == []

    def test_dispatch_reports_unknown_tool_readably(self, root: Path) -> None:
        result = build_registry([root]).dispatch("deploy", "{}")

        assert result.startswith(fs.ERROR_PREFIX)
        assert "未注册的工具" in result and "read_file" in result

    def test_dispatch_reports_bad_arguments_readably(self, root: Path) -> None:
        registry = build_registry([root])

        assert "不是合法 JSON" in registry.dispatch("read_file", "{oops")
        assert "必须是 JSON 对象" in registry.dispatch("read_file", "[1, 2]")
        # 参数类型错误的正是"可读"的要义：给出期望形态，而不是让工具内部抛 TypeError
        assert "必须是非空字符串" in registry.dispatch("read_file", json.dumps({"path": 12}))

    def test_tool_errors_never_raise(self, root: Path) -> None:
        """工具异常一律结构化回传模型：运行不因工具失败而崩。"""
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


class TestRecheckGuards:
    """复核轮修复的行为回归。"""

    def test_list_dir_caps_a_single_huge_subdirectory(self, root: Path) -> None:
        """单个子目录的条目数也受预算约束：不得等到外层检查才发现已塞入几万行。"""
        huge = root / "huge"
        huge.mkdir()
        for index in range(fs.MAX_LIST_LINES + 200):
            (huge / f"item{index:04d}.txt").write_text("x", encoding="utf-8")

        result = fs.list_dir([root])

        assert len(result.splitlines()) <= fs.MAX_LIST_LINES + 5
        assert "条目过多已截断" in result


class TestToolBoundaryGuards:
    """工具输入校验与输出上限的行为回归。"""

    def test_read_file_rejects_non_integer_line_range(self, root: Path) -> None:
        (root / "a.py").write_text("print(1)\n", encoding="utf-8")

        result = fs.read_file([root], "a.py", start="10")  # type: ignore[arg-type]

        assert "start 必须是" in result

    def test_read_file_caps_output_for_few_newline_files(self, root: Path) -> None:
        """少换行的大文件（单行）也必须有字符上限：400 行闸门挡不住它。"""
        (root / "big.py").write_text(
            "x = 1  # " + "y" * (fs.MAX_READ_OUTPUT_CHARS + 10_000), encoding="utf-8"
        )

        result = fs.read_file([root], "big.py")

        assert "字符上限" in result
        assert len(result) <= fs.MAX_READ_OUTPUT_CHARS + 300

    def test_list_dir_ignores_bulky_build_dirs_and_caps_lines(self, root: Path) -> None:
        (root / "node_modules").mkdir()
        (root / "node_modules" / "x.js").write_text("x", encoding="utf-8")
        for index in range(fs.MAX_LIST_LINES + 100):
            (root / f"f{index:04d}.py").write_text("x\n", encoding="utf-8")

        result = fs.list_dir([root])

        assert "node_modules" not in result
        assert f"已按 {fs.MAX_LIST_LINES} 行上限截断" in result

    def test_run_python_passes_large_code_via_stdin(self, root: Path) -> None:
        """大候选代码改走 stdin：规避 Windows 命令行长度上限与进程命令行的暴露面。"""
        code = "# pad\n" * 2000 + "print('stdin-path-ok')\n"  # 远超命令行阈值

        result = exec_py.run_python([root], code=code)

        assert "stdin-path-ok" in result
        assert "退出码：0" in result

    def test_run_python_does_not_leave_bytecode_in_the_workspace(self, root: Path) -> None:
        """子进程不写 __pycache__（PYTHONDONTWRITEBYTECODE）：核实执行不得污染工作区 /
        版本目录 —— 字节码一旦落进产物目录，就会随 resume 与 --out 一起传播出去。"""
        (root / "helper.py").write_text("VALUE = 42\n", encoding="utf-8")

        result = exec_py.run_python(
            [root],
            code="import sys; sys.path.insert(0, '.'); import helper; print(helper.VALUE)\n",
        )

        assert "42" in result
        assert not (root / "__pycache__").exists()


def test_reader_that_overshot_the_reap_window_is_marked_incomplete() -> None:
    """读取线程超窗未收尾：输出必须带"未收集完"标注，不得报成"无输出"。

    子进程退出后管道仍可被孙进程占着，`_join_readers` 只等 KILL_REAP_SECONDS；把
    未收尾的空缓冲直接交给模型，等于让模型以为程序什么都没打印（失败必须如实上报）。
    """
    hold = threading.Event()
    thread = threading.Thread(target=lambda: hold.wait(5), daemon=True)
    thread.start()
    try:
        pending = {"text": "", "thread": thread}
        first = exec_py._reader_text(pending)
        assert "未收集完" in first
        assert exec_py._reader_text({"text": "已收集", "thread": thread}).startswith("已收集")

        hold.set()
        thread.join(5)
    finally:
        hold.set()
    assert exec_py._reader_text({"text": "已收集", "thread": thread}) == "已收集"
