"""위험 조치와 낮은 확신도의 승인을 중단·재개한다."""

from langsmith import traceable
from langgraph.types import interrupt

from agent.deps import SupervisorDeps
from agent.state import AgentState

from .common import event


@traceable(name="hitl_node")
async def hitl_node(state: AgentState, deps: SupervisorDeps) -> dict:
    report = state.get("report") or {}
    raw_actions = report.get("actions")
    actions = {
        action for action in raw_actions if isinstance(action, str)
    } if isinstance(raw_actions, list) else set()
    required = (
        report.get("verdict") == "REJECT"
        and bool(actions & {"폐기", "라인정지"})
    ) or report.get("confidence") == "low"
    approval = {"required": required, "decision": None, "note": ""}
    updates: dict = {"approval": approval}

    if not required:
        return updates
    previous = state.get("approval") or {}
    rejected = int(state.get("hitl_rejections", 0))
    if previous.get("required") and previous.get("decision") == "approve":
        # 검증 반려로 보고서를 다시 써도 같은 요청에서 승인을 반복하지 않는다.
        approval.update(decision="approve", note=str(previous.get("note", "")))
        return updates
    if rejected:
        # 사용자 반려를 반영한 재작성은 재중단 없이 검증으로 보낸다.
        approval.update(decision="approve", note=str(previous.get("note", "")))
        return updates

    events = list(state.get("events", []))
    if deps.hitl_mode != "interactive" and not any(
        value.get("type") == "ApprovalRequired"
        for value in events
        if isinstance(value, dict)
    ):
        events = event(state, "ApprovalRequired", required=True)

    if deps.hitl_mode == "interactive":
        resumed = interrupt({"type": "ApprovalRequired", "report": report})
        if not isinstance(resumed, dict) or resumed.get("decision") not in {
            "approve", "reject"
        }:
            raise ValueError("invalid approval decision")
        approval["decision"] = resumed["decision"]
        approval["note"] = str(resumed.get("note", ""))
        events.append({
            "type": "ApprovalResolved",
            "decision": approval["decision"],
            "note": approval["note"],
        })
        updates["events"] = events
        if approval["decision"] == "reject":
            updates["validation_feedback"] = "사람이 반려: 조치를 보수적으로 재작성"
            updates["hitl_rejections"] = 1
        return updates

    if deps.hitl_mode == "auto_approve":
        approval["decision"] = "approve"
        events.append({"type": "ApprovalResolved", "decision": "approve"})
    elif deps.hitl_mode == "auto_reject":
        approval["decision"] = "reject"
        approval["note"] = "조치를 보수적으로 재작성"
        updates["validation_feedback"] = "사람이 반려: 조치를 보수적으로 재작성"
        updates["hitl_rejections"] = 1
        events.append({"type": "ApprovalResolved", "decision": "reject"})
    if events != state.get("events", []):
        updates["events"] = events
    return updates
