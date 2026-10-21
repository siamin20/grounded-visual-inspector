"""MCP 검사 절차를 반영해 research task를 만든다."""

import json

from langsmith import traceable

from agent.deps import SupervisorDeps
from agent.state import AgentState

from .common import _usage_updates, event, load_lot


@traceable(name="plan_node")
async def plan_node(state: AgentState, deps: SupervisorDeps) -> dict:
    lot = state.get("lot") or load_lot(state.get("image_path"))
    server = await deps.mcp.find_prompt("inspection_prompt")
    procedure = await deps.mcp.get_prompt(
        "inspection_prompt",
        {"item": state["item"], "lot": json.dumps(lot, ensure_ascii=False)},
        server=server,
    )
    result, usage = deps.fast_llm.plan_inspection(state["item"], procedure)
    tasks = [dict(task) for task in result.get("tasks", [])]
    ablation = state.get("ablation") or deps.ablation
    if ablation == "no_cross":
        tasks = [task for task in tasks if task.get("kind") != "cross_check"]
    return {
        "lot": lot,
        "ablation": ablation,
        "tasks": tasks,
        "needs_analysis": False,
        **_usage_updates(state, usage),
        "events": event(state, "PlanCreated", task_count=len(tasks)),
    }
