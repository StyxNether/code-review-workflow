"""模块布局测试：清单里的模块均可导入、带 docstring，且包目录的模块集合与清单一致。"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import crw

# 模块清单：新增 / 删除模块时必须同步本表，否则 test_module_list_matches_package_tree 会失败。
MODULES = (
    "crw",
    "crw.cli",
    "crw.render",
    "crw.llm",
    "crw.config",
    "crw.thinking",
    "crw.usage",
    "crw.rounds",
    "crw.workflow",
    "crw.workflow.graph",
    "crw.workflow.state",
    "crw.workflow.engine",
    "crw.workflow.contracts",
    "crw.workflow.records",
    "crw.agents",
    "crw.agents.encoder",
    "crw.agents.reviewer",
    "crw.agents.fixer",
    "crw.agents.runtime",
    "crw.agents.prompts",
    "crw.session",
    "crw.session.state",
    "crw.session.memory",
    "crw.session.orchestrator",
    "crw.session.chat",
    "crw.tools",
    "crw.tools.fs",
    "crw.tools.exec_py",
    "crw.uninstall",
)


def _modules_on_disk() -> set[str]:
    """枚举包目录下的模块名（包目录自身的 ``__init__`` 归一为包名）。"""
    package_root = Path(str(crw.__file__)).parent
    found: set[str] = set()
    for path in package_root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        parts = list(path.relative_to(package_root.parent).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        found.add(".".join(parts))
    return found


@pytest.mark.parametrize("module_name", MODULES)
def test_module_is_importable(module_name: str) -> None:
    assert importlib.import_module(module_name).__name__ == module_name


@pytest.mark.parametrize("module_name", MODULES)
def test_module_documents_its_responsibility(module_name: str) -> None:
    module = importlib.import_module(module_name)

    assert (module.__doc__ or "").strip()


def test_module_list_matches_package_tree() -> None:
    """包目录实际模块集合与本清单一致（多、少、改名都即刻暴露）。"""
    assert _modules_on_disk() == set(MODULES)
