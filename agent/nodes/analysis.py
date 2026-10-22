"""Observer·Cross-Checker·Spec-Judge를 합의 규칙으로 조합한다."""

from langsmith import traceable

from agent.deps import SupervisorDeps
from agent.state import AgentState

from .common import (
    cross_status,
    event,
    normalized_judgement_status,
    observer_status,
    positive_observer_regions,
)


def _abstain(reason: str) -> tuple[str, str, None, str]:
    return "ABSTAIN", "low", None, reason


@traceable(name="analysis_node")
async def analysis_node(state: AgentState, deps: SupervisorDeps) -> dict:
    observations = state.get("observations", [])
    observer = observer_status(observations)
    observed_regions = positive_observer_regions(observations)
    cross_check = state.get("cross_check") or {}
    ablation = state.get("ablation") or getattr(deps, "ablation", "")
    cross = cross_status(cross_check, skipped=ablation == "no_cross")
    cross_regions = {
        str(region)
        for region in (cross_check.get("regions") or [])
        if region in {"tl", "tr", "bl", "br", "center"}
    }
    positive_overlap = bool(observed_regions & cross_regions)
    judgement = state.get("judgement") or {}
    raw_violations = judgement.get("violations")
    violations = list(raw_violations) if isinstance(raw_violations, list) else []
    judged = normalized_judgement_status(judgement)

    if cross == "C-skip":
        agree = (observer, judged) in {("O+", "V+"), ("O-", "V0")}
    else:
        agree = (
            (observer == "O+" and cross == "C+" and positive_overlap)
            or (observer == "O-" and cross == "C-")
        )

    if judged == "V?":
        verdict, confidence, defect_type, abstain_reason = _abstain(
            "validation_failure"
        )
    elif cross == "C-skip":
        if observer == "O+" and judged == "V+":
            verdict, confidence = "REJECT", "high"
            defect_type = judgement.get("defect_type")
            abstain_reason = None
        elif observer == "O-" and judged == "V0":
            verdict, confidence, defect_type, abstain_reason = (
                "PASS", "high", None, None
            )
        elif observer == "O+" and judged == "V0":
            verdict, confidence, defect_type, abstain_reason = _abstain(
                "ungrounded_anomaly"
            )
        elif judged == "V+":
            verdict, confidence, defect_type, abstain_reason = _abstain(
                "evidence_conflict"
            )
        else:
            verdict, confidence, defect_type, abstain_reason = _abstain(
                "insufficient_visual_evidence"
            )
    elif observer == "O+" and cross == "C+" and judged == "V+":
        if positive_overlap:
            verdict, confidence = "REJECT", "high"
            defect_type = judgement.get("defect_type")
            abstain_reason = None
        else:
            verdict, confidence, defect_type, abstain_reason = _abstain(
                "evidence_conflict"
            )
    elif judged == "V+" and (
        (observer == "O+" and cross in {"C-", "C?"})
        or (observer in {"O?", "O-"} and cross == "C+")
    ):
        verdict, confidence, defect_type, abstain_reason = _abstain(
            "evidence_conflict"
        )
    elif judged == "V0" and (observer == "O+" or cross == "C+"):
        verdict, confidence, defect_type, abstain_reason = _abstain(
            "ungrounded_anomaly"
        )
    elif judged == "V+":
        verdict, confidence, defect_type, abstain_reason = _abstain(
            "evidence_conflict"
        )
    elif observer == "O-" and cross == "C-":
        verdict, confidence, defect_type, abstain_reason = (
            "PASS", "high", None, None
        )
    elif observer == "O-" and cross == "C?":
        verdict, confidence, defect_type, abstain_reason = (
            "PASS", "low", None, None
        )
    else:
        verdict, confidence, defect_type, abstain_reason = _abstain(
            "insufficient_visual_evidence"
        )

    return {
        "verdict": verdict,
        "defect_type": defect_type,
        "confidence": confidence,
        "abstain_reason": abstain_reason,
        "observer_status": observer,
        "cross_status": cross,
        "events": event(
            state,
            "AnalysisDecided",
            verdict=verdict,
            confidence=confidence,
            abstain_reason=abstain_reason,
            observer_status=observer,
            cross_status=cross,
            judgement_status=judged,
            agree=agree,
            violations=violations,
            cross_check_skipped=cross == "C-skip",
        ),
    }
