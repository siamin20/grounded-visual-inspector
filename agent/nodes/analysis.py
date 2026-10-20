"""Observer·Cross-Checker·Spec-Judge를 합의 규칙으로 조합한다."""

from langsmith import traceable

from agent.deps import SupervisorDeps
from agent.state import AgentState

from .common import event


@traceable(name="analysis_node")
async def analysis_node(state: AgentState, deps: SupervisorDeps) -> dict:
    del deps
    observations = state.get("observations", [])
    anomaly_seen = any(obs.get("anomaly_seen") is True for obs in observations)
    observed_regions = {
        region
        for observation in observations
        for region in observation.get("regions", [])
    }

    cross_check = state.get("cross_check") or {}
    cross_check_skipped = not bool(cross_check)
    if cross_check_skipped:
        agree = True
    else:
        same_anomaly = anomaly_seen == cross_check.get("anomaly_seen")
        region_overlap = bool(observed_regions & set(cross_check.get("regions", [])))
        agree = same_anomaly and (not anomaly_seen or region_overlap)

    judgement = state.get("judgement") or {}
    violations = list(judgement.get("violations") or [])
    if agree and violations:
        verdict, confidence = "REJECT", "high"
    elif agree:
        verdict, confidence = "PASS", "high"
    else:
        verdict = "REJECT" if anomaly_seen or violations else "PASS"
        confidence = "low"
    defect_type = judgement.get("defect_type") if verdict == "REJECT" else None

    return {
        "verdict": verdict,
        "defect_type": defect_type,
        "confidence": confidence,
        "events": event(
            state,
            "AnalysisDecided",
            verdict=verdict,
            confidence=confidence,
            agree=agree,
            violations=violations,
            cross_check_skipped=cross_check_skipped,
        ),
    }
