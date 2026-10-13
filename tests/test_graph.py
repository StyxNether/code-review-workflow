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


# ---- M3：图形态快照与 mermaid 导出（Design §8.1、§12.2）----


def test_parallel_graph_exposes_the_designed_nodes_and_edges() -> None:
    """并行双视角形态的节点 / 边集合（Design §12.2 的落地形态）。"""
    compiled = graph_mod.build_graph(
        defaults={**fakes.PARALLEL_SCRIPTED_NODES(), "merge_reviews": graph_mod.merge_review_node},
        parallel_review=True,
    )
    structure = compiled.get_graph()

    assert set(structure.nodes) == {
        "__start__",
        "__end__",
        "encode",
        "review_correctness",
        "review_security",
        "merge_reviews",
        "fix",
    }
    edges = {(edge.source, edge.target) for edge in structure.edges}
    assert edges == {
        ("__start__", "encode"),
        ("__start__", "review_correctness"),
        ("__start__", "review_security"),
        ("encode", "review_correctness"),
        ("encode", "review_security"),
        ("review_correctness", "merge_reviews"),
        ("review_security", "merge_reviews"),
        ("merge_reviews", "fix"),
        ("merge_reviews", "__end__"),
        ("fix", "review_correctness"),
        ("fix", "review_security"),
    }


def test_parallel_entry_router_fans_out_for_file_inputs() -> None:
    """入口路由返回目标清单即扇出（同一超级步并行两个视角）。"""
    assert graph_mod.route_entry_parallel(state_for(kind="requirement")) == ["encode"]
    assert graph_mod.route_entry_parallel(state_for(kind="file")) == [
        "review_correctness",
        "review_security",
    ]


def test_mermaid_single_snapshot() -> None:
    """``crw graph`` 的导出内容由快照锁定（节点 / 边集合与实现同源）。"""
    assert graph_mod.mermaid_flowchart() == SNAPSHOT_SINGLE


def test_mermaid_parallel_snapshot() -> None:
    assert graph_mod.mermaid_flowchart(parallel_review=True) == SNAPSHOT_PARALLEL


@pytest.mark.parametrize("parallel", [False, True])
def test_mermaid_matches_the_compiled_graph(parallel: bool) -> None:
    """导出图里的节点与边必须与编译出的图一致（漂移即失败，不靠人眼看）。"""
    nodes = fakes.PARALLEL_SCRIPTED_NODES() if parallel else fakes.SCRIPTED_NODES()
    if parallel:
        nodes = {**nodes, "merge_reviews": graph_mod.merge_review_node}
    compiled = graph_mod.build_graph(defaults=nodes, parallel_review=parallel)
    declared_nodes = {
        entry[0] for entry in _node_lines(graph_mod.mermaid_flowchart(parallel_review=parallel))
    }
    declared_edges = _edge_lines(graph_mod.mermaid_flowchart(parallel_review=parallel))

    # mermaid 的 start / finish 即图的 __start__ / __end__（README/Design 的口径映射）
    assert declared_nodes - {"start", "finish"} == set(compiled.get_graph().nodes) - {
        "__start__",
        "__end__",
    }
    compiled_edges = {(edge.source, edge.target) for edge in compiled.get_graph().edges}
    assert declared_edges == compiled_edges


def _node_lines(content: str) -> list[tuple[str, str]]:
    """mermaid 里的节点声明行（``id[标签]`` 形态）。"""
    import re

    pattern = re.compile(
        r"^ {4}(?P<id>[A-Za-z_][A-Za-z0-9_]*)(?P<bracket>\[|\()(?P<label>.+)(?:\]|\))$"
    )
    found: list[tuple[str, str]] = []
    for line in content.splitlines():
        match = pattern.match(line)
        if match:
            found.append((match.group("id"), match.group("label")))
    return found


def _edge_lines(content: str) -> set[tuple[str, str]]:
    """mermaid 里的边（含条件边上的中文标签，标签不参与比对）。"""
    import re

    pattern = re.compile(
        r"^ {4}(?P<src>[A-Za-z_][A-Za-z0-9_]*)(?: -- .+ --> | --> )(?P<dst>[A-Za-z_][A-Za-z0-9_]*)$"
    )
    edges: set[tuple[str, str]] = set()
    for line in content.splitlines():
        match = pattern.match(line)
        if match:
            src, dst = match.group("src"), match.group("dst")
            edges.add(
                (("__start__" if src == "start" else src), ("__end__" if dst == "finish" else dst))
            )
    return edges


def test_recursion_limit_is_recomputed_per_graph_shape() -> None:
    """护栏按形态复算：并行形态每轮多一个汇合超级步（Design §4.4）。"""
    assert graph_mod.recursion_limit(max_iterations=3) == 14
    assert graph_mod.recursion_limit(max_iterations=3, parallel_review=True) == 17


def test_agents_layer_node_names_match_the_graph_declaration() -> None:
    """节点名 / 视角 / 显示名三处声明必须同口径（任一处漏改都会"能跑但错位"）。"""
    from crw import agents as agents_mod
    from crw import render as render_mod
    from crw.agents import prompts as prompts_mod

    views = set(graph_mod.VIEW_LABELS)
    assert set(agents_mod.PARALLEL_REVIEW_NODES) == set(graph_mod.REVIEW_NODE_FOR_VIEW.values())
    assert set(agents_mod.PARALLEL_REVIEW_NODES.values()) == views
    assert set(prompts_mod.VIEW_DISPLAY) == views
    assert set(prompts_mod._VIEW_SYSTEM_PROMPTS) == views
    assert set(render_mod._NODE_STYLE) >= set(graph_mod.PARALLEL_NODE_NAMES)


SNAPSHOT_SINGLE = """flowchart TD
    start(["开始"])
    encode["编码 Agent：生成初始代码 code_v1.py"]
    review["审查 Agent：读文件 / 运行核实 → findings"]
    fix["修复 Agent：按 findings 逐条处置 → code_vN+1.py"]
    finish(["收尾：report.md（干净通过退出码 0 / 达限残留退出码 1）"])

    start -- "需求文本" --> encode
    start -- "代码文件（-f）/ 续跑（resume）" --> review
    encode --> review
    review -- "干净通过 或 已达修复轮次上限" --> finish
    review -- "未干净，且未达修复轮次上限" --> fix
    fix --> review
"""

SNAPSHOT_PARALLEL = """flowchart TD
    start(["开始"])
    encode["编码 Agent：生成初始代码 code_v1.py"]
    review_correctness["审查 Agent·正确性视角：并行分支，写 findings_correctness"]
    review_security["审查 Agent·安全性视角：并行分支，写 findings_security"]
    merge_reviews["汇合：按 (location, issue) 去重取较严重者、统一重编号 id、审查轮次 +1"]
    fix["修复 Agent：按合并 findings 逐条处置 → code_vN+1.py"]
    finish(["收尾：report.md（干净通过退出码 0 / 达限残留退出码 1）"])

    start -- "需求文本" --> encode
    start -- "代码文件（-f）/ 续跑（resume）" --> review_correctness
    start -- "代码文件（-f）/ 续跑（resume）" --> review_security
    encode -- "扇出：同一超级步并行" --> review_correctness
    encode -- "扇出：同一超级步并行" --> review_security
    review_correctness --> merge_reviews
    review_security --> merge_reviews
    merge_reviews -- "干净通过 或 已达修复轮次上限" --> finish
    merge_reviews -- "未干净，且未达修复轮次上限" --> fix
    fix -- "扇出：同一超级步并行" --> review_correctness
    fix -- "扇出：同一超级步并行" --> review_security
"""
