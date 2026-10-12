"""图组装：节点 / 边集合、条件路由与装配接缝（Design §4.1、§4.4、§12.2）。

脚本化假节点的行为用例见 test_scripted_nodes.py（假节点已移出生产代码、只作测试注入）。
"""

from __future__ import annotations

import fakes
import pytest
from support import node_state as state_for

from crw.workflow import graph as graph_mod

# ---- 组装与路由 ----


def test_resolve_nodes_replaces_key_by_key() -> None:
    """覆盖表逐键替换：未给出的键仍用缺省装配（Design §12.2 第 3 步）。"""

    def my_review(state: dict) -> dict:
        return {"review_text": "```json\n{}\n```"}

    resolved = graph_mod.resolve_nodes({"review": my_review}, defaults=fakes.SCRIPTED_NODES())

    assert resolved["review"] is my_review
    assert resolved["encode"] is fakes.fake_encode
    assert resolved["fix"] is fakes.fake_fix


def test_resolve_nodes_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="未知节点名"):
        graph_mod.resolve_nodes({"deploy": fakes.fake_encode}, defaults=fakes.SCRIPTED_NODES())


def test_resolve_nodes_requires_a_complete_set() -> None:
    """既无缺省装配、覆盖表又不全：显式报错（不许静默少接一个节点）。"""
    with pytest.raises(ValueError, match="缺少节点装配"):
        graph_mod.resolve_nodes({"review": fakes.fake_review})


def test_build_graph_without_any_assembly_is_rejected() -> None:
    """无缺省装配、又不给覆盖表：显式报错，而不是编译出一个少节点的图（Design §12.2 第 3 步）。"""
    with pytest.raises(ValueError, match="缺少节点装配"):
        graph_mod.build_graph()


def test_build_graph_exposes_the_designed_nodes_and_edges() -> None:
    """节点 / 边集合与 Design §4.1 一致（M3 的 crw graph 快照测试以此为基线）。"""
    compiled = graph_mod.build_graph(defaults=fakes.SCRIPTED_NODES())
    structure = compiled.get_graph()

    assert set(structure.nodes) == {"__start__", "__end__", "encode", "review", "fix"}
    edges = {(edge.source, edge.target) for edge in structure.edges}
    assert edges == {
        ("__start__", "encode"),
        ("__start__", "review"),
        ("encode", "review"),
        ("review", "fix"),
        ("review", "__end__"),
        ("fix", "review"),
    }


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("requirement", "encode"), ("file", "review")],
)
def test_route_entry(kind: str, expected: str) -> None:
    """需求文本从编码起步；代码文件直接进入审查（Design §4.4）。"""
    assert graph_mod.route_entry(state_for(kind=kind)) == expected


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"clean": True}, "__end__"),
        ({"clean": True, "fix_rounds": 3, "max_iterations": 3}, "__end__"),
        ({"clean": False, "fix_rounds": 3, "max_iterations": 3}, "__end__"),
        ({"clean": False, "fix_rounds": 0, "max_iterations": 0}, "__end__"),
        ({"clean": False, "fix_rounds": 1, "max_iterations": 3}, "fix"),
    ],
)
def test_route_after_review(overrides: dict, expected: str) -> None:
    """干净 / 达限收尾，否则进入修复（Design §4.4 的循环语义）。"""
    assert graph_mod.route_after_review(state_for(**overrides)) == expected
