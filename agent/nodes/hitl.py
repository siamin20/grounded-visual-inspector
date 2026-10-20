"""Phase 4 interrupt를 연결하기 전의 HITL 정책 자리."""

from langsmith import traceable

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
    events = event(state, "ApprovalRequired", required=required)
    updates: dict = {"approval": approval, "events": events}

    if not required:
        return updates
    if deps.hitl_mode == "auto_approve":
        approval["decision"] = "approve"
        events.append({"type": "ApprovalResolved", "decision": "approve"})
    elif deps.hitl_mode == "auto_reject":
        rejected = int(state.get("hitl_rejections", 0))
        if rejected == 0:
            approval["decision"] = "reject"
            approval["note"] = "조치를 보수적으로 재작성"
            updates["validation_feedback"] = "사람이 반려: 조치를 보수적으로 재작성"
            updates["hitl_rejections"] = 1
            events.append({"type": "ApprovalResolved", "decision": "reject"})
        else:
            # 자동 반려는 재작성 루프를 한 번만 증명한다.
            approval["decision"] = "approve"
            approval["note"] = "자동 반려 1회 후 통과"
            events.append({"type": "ApprovalResolved", "decision": "approve"})
    # interactive의 실제 interrupt는 Phase 4에서 추가한다.
    return updates
