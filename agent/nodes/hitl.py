"""위험 조치와 낮은 확신도의 승인을 중단·재개한다."""

from langsmith import traceable
from langgraph.types import interrupt

from agent.deps import SupervisorDeps, validate_hitl_mode
from agent.state import AgentState

from .common import event


@traceable(name="hitl_node")
async def hitl_node(state: AgentState, deps: SupervisorDeps) -> dict:
    hitl_mode = validate_hitl_mode(getattr(deps, "hitl_mode", "interactive"))
    report = state.get("report") or {}
    raw_actions = report.get("actions")
    actions = {
        action for action in raw_actions if isinstance(action, str)
    } if isinstance(raw_actions, list) else set()
    required = (
        bool(state.get("reapproval_required"))
        or report.get("verdict") == "ABSTAIN"
        or (
            report.get("verdict") == "REJECT"
            and bool(actions & {"폐기", "라인정지"})
        )
        or report.get("confidence") == "low"
    )
    approval = {"required": required, "decision": None, "note": ""}
    updates: dict = {
        "approval": approval,
        "reapproval_required": False,
    }

    if not required:
        return updates
    previous = state.get("approval") or {}
    rejected = int(state.get("hitl_rejections", 0))
    if (
        not state.get("reapproval_required")
        and previous.get("required")
        and previous.get("decision") == "approve"
    ):
        # 같은 초안으로 노드가 재진입한 경우에만 이미 끝난 결정을 보존한다.
        # write 노드는 새 초안마다 approval을 초기화하므로 재작성본에는 적용되지 않는다.
        approval.update(decision="approve", note=str(previous.get("note", "")))
        return updates

    events = list(state.get("events", []))
    if hitl_mode != "interactive":
        events = event(state, "ApprovalRequired", required=True, report=report)

    fresh_human_review = hitl_mode == "interactive" or (
        hitl_mode == "auto_reject" and rejected >= 1
    )
    if fresh_human_review:
        # auto_reject는 첫 초안의 반려만 자동화한다. 그 반려로 내용이
        # 달라진 다음 초안은 실제 사람의 새 결정을 받아야 한다.
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

    if hitl_mode == "auto_approve":
        approval["decision"] = "approve"
        events.append({"type": "ApprovalResolved", "decision": "approve"})
    elif hitl_mode == "auto_reject":
        approval["decision"] = "reject"
        approval["note"] = "조치를 보수적으로 재작성"
        updates["validation_feedback"] = "사람이 반려: 조치를 보수적으로 재작성"
        updates["hitl_rejections"] = 1
        events.append({"type": "ApprovalResolved", "decision": "reject"})
    if events != state.get("events", []):
        updates["events"] = events
    return updates
