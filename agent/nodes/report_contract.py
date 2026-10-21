"""보고서에 노출되는 요약과 로트를 코드 규칙으로 고정한다."""

from copy import deepcopy
import hashlib
import json


def state_lot(state: dict) -> dict:
    """로트는 상태의 조회값만 사용하고 작성 모델의 값은 버린다."""
    lot = state.get("lot", {})
    return deepcopy(lot) if isinstance(lot, dict) else {}


def make_pre_validation_snapshot(
    report: dict,
    *,
    cost_usd: float,
    tokens: int,
) -> dict:
    """validate 직전 첫 보고서를 재작성과 분리해 고정한다."""

    frozen_report = deepcopy(report)
    canonical = json.dumps(
        frozen_report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "version": 1,
        "stage": "before_validate",
        "report": frozen_report,
        "report_sha256": hashlib.sha256(canonical).hexdigest(),
        "cost_usd": float(cost_usd),
        "tokens": int(tokens),
    }


def deterministic_summary(report: dict) -> str:
    """인용한 관찰과 코드 판정 필드만으로 결정론적 요약을 만든다."""
    observations = report.get("observations")
    texts: list[str] = []
    if isinstance(observations, list):
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            text = observation.get("text")
            if (
                isinstance(text, str)
                and text.strip()
                and text.strip() not in texts
            ):
                texts.append(text.strip())
    evidence = " / ".join(texts) if texts else "유효한 관찰 근거 없음"

    verdict = report.get("verdict")
    if verdict == "PASS":
        return f"관찰 근거({evidence})를 바탕으로 PASS로 판정하고 통과를 요청합니다."
    if verdict == "REJECT":
        defect_type = report.get("defect_type")
        defect = (
            defect_type
            if isinstance(defect_type, str) and defect_type
            else "유형 미상"
        )
        clauses = report.get("spec_clauses")
        clause_values = clauses if isinstance(clauses, list) else []
        clause_text = ", ".join(
            clause_id for clause_id in clause_values if isinstance(clause_id, str)
        ) or "검증된 조항 없음"
        actions = report.get("actions")
        action_values = actions if isinstance(actions, list) else []
        action_text = ", ".join(
            action for action in action_values if isinstance(action, str)
        ) or "조치 미정"
        return (
            f"관찰 근거({evidence})와 위반 조항({clause_text})을 바탕으로 "
            f"{defect} 결함을 REJECT로 판정하고 {action_text} 조치를 요청합니다."
        )
    if verdict == "ABSTAIN":
        reason = report.get("abstain_reason")
        reason_text = (
            reason if isinstance(reason, str) and reason else "validation_failure"
        )
        return (
            f"관찰 근거({evidence})만으로 확정할 수 없어 {reason_text} 사유로 "
            "ABSTAIN하고 재검사를 요청합니다."
        )
    return f"관찰 근거({evidence})에 대한 판정이 유효하지 않습니다."
