"""模块骨架测试：Design §3 列出的模块均可导入且带模块 docstring。"""

from __future__ import annotations

import importlib

import pytest

MODULES = (
    "crw",
    "crw.cli",
    "crw.render",
    "crw.llm",
    "crw.config",
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
    "crw.tools",
    "crw.tools.fs",
    "crw.tools.exec_py",
)


@pytest.mark.parametrize("module_name", MODULES)
def test_module_is_importable(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


@pytest.mark.parametrize("module_name", MODULES)
def test_module_documents_its_responsibility(module_name: str) -> None:
    module = importlib.import_module(module_name)

    assert (module.__doc__ or "").strip()
