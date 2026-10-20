"""관찰이 불충분하면 center 크롭을 단 한 번 추가한다."""

from langsmith import traceable

from agent.deps import SupervisorDeps
from agent.state import AgentState

from .common import event


@traceable(name="replan_check_node")
async def replan_check_node(state: AgentState, deps: SupervisorDeps) -> dict:
    del deps
    unreadable = any(
        "판독 불가" in str(observation.get("text", ""))
        or (
            not observation.get("features")
            and observation.get("anomaly_seen") is None
        )
        for observation in state.get("observations", [])
    )
    rounds = int(state.get("replan_rounds", 0))
    if unreadable and rounds == 0:
        task = {
            "id": "task-observe-center",
            "agent": "research",
            "kind": "observe",
            "region": "center",
            "status": "pending",
        }
        return {
            "tasks": [*state.get("tasks", []), task],
            "replan_rounds": 1,
            "research_sufficient": False,
            "events": event(
                state,
                "ReplanCheck",
                round=rounds,
                is_sufficient=False,
                gap_task=task["id"],
            ),
        }
    return {
        "research_sufficient": True,
        "events": event(
            state, "ReplanCheck", round=rounds, is_sufficient=True
        ),
    }
