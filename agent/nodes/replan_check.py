"""관찰이 불충분하면 center 크롭을 단 한 번 추가한다."""

from langsmith import traceable

from agent.deps import SupervisorDeps
from agent.state import AgentState

from .common import event, observer_status


@traceable(name="replan_check_node")
async def replan_check_node(state: AgentState, deps: SupervisorDeps) -> dict:
    status = observer_status(state.get("observations", []))
    rounds = int(state.get("replan_rounds", 0))
    ablation = state.get("ablation") or getattr(deps, "ablation", "")
    if status == "O?" and rounds == 0 and ablation == "no_crops":
        return {
            "research_sufficient": True,
            "events": event(
                state,
                "ReplanCheck",
                round=rounds,
                is_sufficient=True,
                observer_status=status,
                replan_skipped=True,
                skip_reason="no_crops_full_only",
                ablation=ablation,
            ),
        }
    if status == "O?" and rounds == 0:
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
                observer_status=status,
            ),
        }
    return {
        "research_sufficient": True,
        "events": event(
            state,
            "ReplanCheck",
            round=rounds,
            is_sufficient=True,
            observer_status=status,
        ),
    }
