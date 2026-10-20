"""Observer·Cross-Checker 병렬 후 Spec-Judge를 실행하는 research 서브그래프."""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.state import AgentState

from .cross_checker import cross_checker_node
from .observer import observer_node
from .spec_judge import spec_judge_node


def build_research_subgraph(deps: Any):
    """부모 그래프와 같은 AgentState를 사용하는 compiled subgraph를 만든다."""
    graph = StateGraph(AgentState)
    graph.add_node("observer", partial(observer_node, deps=deps))
    graph.add_node("cross_checker", partial(cross_checker_node, deps=deps))
    graph.add_node("spec_judge", partial(spec_judge_node, deps=deps))
    graph.add_edge(START, "observer")
    graph.add_edge(START, "cross_checker")
    graph.add_edge(["observer", "cross_checker"], "spec_judge")
    graph.add_edge("spec_judge", END)
    return graph.compile()
