"""제품 이미지 검사 Supervisor 그래프 조립."""

from functools import partial

from langgraph.graph import END, START, StateGraph

from .deps import SupervisorDeps
from .nodes.analysis import analysis_node
from .nodes.chat_answer import chat_answer_node
from .nodes.finalize import finalize_node
from .nodes.gate import gate_node
from .nodes.hitl import hitl_node
from .nodes.plan import plan_node
from .nodes.replan_check import replan_check_node
from .nodes.route import route_node
from .nodes.validate import validate_node
from .nodes.write import write_node
from .state import AgentState
from .subgraphs.research import build_research_subgraph


def _after_route(state: AgentState) -> str:
    return state.get("route", "chat")


def _after_replan(state: AgentState) -> str:
    return "analysis" if state.get("research_sufficient", True) else "research"


def _after_hitl(state: AgentState) -> str:
    approval = state.get("approval") or {}
    return "write" if approval.get("decision") == "reject" else "validate"


def _after_validate(state: AgentState) -> str:
    if not state.get("validation_passed"):
        return "write"
    approval = state.get("approval") or {}
    if approval.get("required") and approval.get("decision") is None:
        # 재작성 상한 뒤 low confidence로 강제 통과한 보고서도 사람 승인을 거친다.
        return "hitl"
    return "finalize"


def build_graph(deps: SupervisorDeps, checkpointer=None):
    """Base 순서와 노드 시그니처를 유지한 그래프를 컴파일한다."""
    graph = StateGraph(AgentState)
    graph.add_node("route", partial(route_node, deps=deps))
    graph.add_node("chat_answer", partial(chat_answer_node, deps=deps))
    graph.add_node("plan", partial(plan_node, deps=deps))
    graph.add_node("gate", partial(gate_node, deps=deps))
    graph.add_node("research", build_research_subgraph(deps))
    graph.add_node("replan_check", partial(replan_check_node, deps=deps))
    graph.add_node("analysis", partial(analysis_node, deps=deps))
    graph.add_node("write", partial(write_node, deps=deps))
    graph.add_node("hitl", partial(hitl_node, deps=deps))
    graph.add_node("validate", partial(validate_node, deps=deps))
    graph.add_node("finalize", partial(finalize_node, deps=deps))

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        _after_route,
        {"chat": "chat_answer", "needs_item": "chat_answer", "inspection": "plan"},
    )
    graph.add_edge("chat_answer", END)
    graph.add_edge("plan", "gate")
    graph.add_edge("gate", "research")
    graph.add_edge("research", "replan_check")
    graph.add_conditional_edges(
        "replan_check",
        _after_replan,
        {"research": "research", "analysis": "analysis"},
    )
    graph.add_edge("analysis", "write")
    graph.add_edge("write", "hitl")
    graph.add_conditional_edges(
        "hitl", _after_hitl, {"write": "write", "validate": "validate"}
    )
    graph.add_conditional_edges(
        "validate",
        _after_validate,
        {"write": "write", "hitl": "hitl", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


__all__ = ["AgentState", "build_graph"]
