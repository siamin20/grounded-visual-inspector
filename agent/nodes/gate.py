"""실행 가능한 품목과 task만 통과시키는 코드 규칙 게이트."""

from langsmith import traceable

from agent.deps import SupervisorDeps
from agent.state import AgentState

from .common import KNOWN_ITEMS, event


ALLOWED_KINDS = frozenset({"observe", "cross_check", "spec_judge"})


@traceable(name="gate_node")
async def gate_node(state: AgentState, deps: SupervisorDeps) -> dict:
    del deps
    tasks = state.get("tasks", [])
    if state.get("item") not in KNOWN_ITEMS:
        kept: list[dict] = []
        reason = "item_required"
    else:
        kept = [task for task in tasks if task.get("kind") in ALLOWED_KINDS]
        reason = "rule_filter"
    removed = len(tasks) - len(kept)
    return {
        "tasks": kept,
        "gate_removed": removed,
        "events": event(state, "GateApplied", removed=removed, reason=reason),
    }
