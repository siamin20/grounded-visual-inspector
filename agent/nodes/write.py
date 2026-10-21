"""판정 근거와 조치 가이드를 보고서 JSON으로 작성한다."""

import json
import re

from langgraph.config import get_stream_writer
from langsmith import traceable

from agent.deps import SupervisorDeps
from agent.guide_actions import parse_guide_actions, resolve_reject_actions
from agent.state import AgentState

from .common import _usage_updates, event
from .report_contract import (
    deterministic_summary,
    make_pre_validation_snapshot,
    state_lot,
)


_VERDICTS = {"PASS", "REJECT", "ABSTAIN"}
_ABSTAIN_REASONS = {
    "insufficient_visual_evidence",
    "ungrounded_anomaly",
    "evidence_conflict",
    "validation_failure",
}
_REPORT_KEYS = {
    "verdict",
    "defect_type",
    "confidence",
    "abstain_reason",
    "observations",
    "spec_clauses",
    "actions",
    "lot",
    "summary",
}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _parse_report(draft: str) -> dict:
    content = draft.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content)
    parsed = json.loads(content, parse_constant=_reject_json_constant)
    if not isinstance(parsed, dict):
        raise ValueError("보고서는 JSON 객체여야 합니다.")
    return parsed


def _split_report_observations(report: dict) -> dict:
    """각 관찰을 단일 호출 ID와 연결하고 정확한 중복만 제거한다."""
    observations = report.get("observations")
    if not isinstance(observations, list):
        return report

    normalized_observations: list = []
    seen: set[tuple[str, str]] = set()
    for observation in observations:
        if not isinstance(observation, dict) or not isinstance(
            observation.get("text"), str
        ):
            normalized_observations.append(observation)
            continue

        text = observation["text"]
        raw_ids = observation.get("source_call_id")
        if isinstance(raw_ids, list) and raw_ids:
            source_ids = raw_ids
        elif isinstance(raw_ids, str):
            source_ids = [raw_ids]
        else:
            # 잘못된 형식은 숨기지 않고 validate가 거부하게 둔다.
            normalized_observations.append(dict(observation))
            continue

        for source_id in source_ids:
            normalized = {
                "text": text,
                "source_call_id": source_id,
            }
            if not isinstance(source_id, str):
                normalized_observations.append(normalized)
                continue
            key = (text, source_id)
            if key in seen:
                continue
            normalized_observations.append(normalized)
            seen.add(key)

    normalized_report = dict(report)
    normalized_report["observations"] = normalized_observations
    return normalized_report


def _enforce_decision_contract(
    report: dict,
    state: AgentState,
    allowed_guide_actions: list[str] | None = None,
) -> dict:
    """LLM이 코드 판정과 조치 불변조건을 바꾸지 못하게 한다."""
    normalized = {
        key: value for key, value in report.items() if key in _REPORT_KEYS
    }
    verdict = state.get("verdict")
    if verdict not in _VERDICTS:
        verdict = "ABSTAIN"
    normalized["verdict"] = verdict

    confidence = state.get("confidence")
    normalized["confidence"] = (
        "low"
        if verdict == "ABSTAIN"
        else confidence if confidence in {"high", "low"} else "low"
    )

    if verdict == "PASS":
        normalized.update(
            defect_type=None,
            abstain_reason=None,
            spec_clauses=[],
            actions=["통과"],
        )
    elif verdict == "ABSTAIN":
        reason = state.get("abstain_reason")
        if reason not in _ABSTAIN_REASONS:
            reason = "validation_failure"
        normalized.update(
            defect_type=None,
            abstain_reason=reason,
            spec_clauses=[],
            actions=["재검사"],
        )
    else:
        actions = resolve_reject_actions(
            normalized.get("actions"),
            allowed_guide_actions or [],
        )
        judgement = state.get("judgement") or {}
        violations = judgement.get("violations")
        normalized.update(
            defect_type=state.get("defect_type"),
            abstain_reason=None,
            spec_clauses=(
                list(dict.fromkeys(
                    clause_id for clause_id in violations
                    if isinstance(clause_id, str)
                ))
                if isinstance(violations, list)
                else []
            ),
            # 가이드가 비었으면 빈 조치를 남겨 validate가 반려하게 한다.
            actions=actions,
        )
    normalized["lot"] = state_lot(state)
    normalized["summary"] = deterministic_summary(normalized)
    return normalized


@traceable(name="write_node")
async def write_node(state: AgentState, deps: SupervisorDeps) -> dict:
    previous_approval = state.get("approval") or {}
    reapproval_required = bool(
        previous_approval.get("required")
        and previous_approval.get("decision") in {"approve", "reject"}
    )
    guide_text = ""
    if state.get("verdict") == "REJECT" and state.get("defect_type"):
        server, uri_template = await deps.mcp.find_resource_template("get_guide")
        guide_text = await deps.mcp.read_resource(
            uri_template.format(defect_type=state["defect_type"]), server=server
        )

    attempt = int(state.get("validation_attempts", 0)) + 1
    context = {
        "query": state.get("query", ""),
        "item": state.get("item"),
        "lot": state.get("lot", {}),
        "observations": state.get("observations", []),
        "cross_check": state.get("cross_check", {}),
        "spec_hits": state.get("spec_hits", []),
        "judgement": state.get("judgement", {}),
        "verdict": state.get("verdict"),
        "defect_type": state.get("defect_type"),
        "confidence": state.get("confidence"),
        "abstain_reason": state.get("abstain_reason"),
        "guide_text": guide_text,
    }
    writer = get_stream_writer()
    draft, usage = await deps.quality_llm.write_report(
        context,
        attempt,
        state.get("validation_feedback"),
        on_token=lambda piece: writer(
            {"type": "AnswerToken", "delta": piece, "attempt": attempt}
        ),
    )
    feedback = None
    try:
        allowed_guide_actions = parse_guide_actions(guide_text)
        report = _enforce_decision_contract(
            _split_report_observations(_parse_report(draft)),
            state,
            allowed_guide_actions=allowed_guide_actions,
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        report = {}
        feedback = "JSON 파싱 실패"
    enforced_draft = json.dumps(report, ensure_ascii=False)

    usage_updates = _usage_updates(state, usage)
    result = {
        "draft": enforced_draft,
        "report": report,
        # 새 초안은 과거 초안의 승인 결정을 이어받지 않는다.
        "approval": {"required": False, "decision": None, "note": ""},
        "reapproval_required": reapproval_required,
        "validation_attempts": attempt,
        "validation_terminal": False,
        "validation_feedback": feedback,
        "last_prompt": usage.get("prompt"),
        **usage_updates,
        "events": event(state, "DraftWritten", attempt=attempt),
    }
    # H3은 별도 모델 호출이 아닌 동일 실행의 validate 전·후를 비교한다.
    # 재작성 시에는 첫 초안을 덮어쓰지 않는다.
    if attempt == 1 and not state.get("pre_validation_snapshot"):
        result["pre_validation_snapshot"] = make_pre_validation_snapshot(
            report,
            usage=usage_updates,
        )
    return result
