"""보고서의 근거·조항·유형·스키마를 LLM 없이 검증한다."""

from difflib import SequenceMatcher
import json
import math
import os
import re

from langsmith import traceable

from agent.deps import SupervisorDeps
from agent.guide_actions import load_guide_actions
from agent.state import AgentState

from .common import event, load_type_map
from .report_contract import deterministic_summary, state_lot


REQUIRED_KEYS = {
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
ALLOWED_ACTIONS = {"재작업", "폐기", "라인정지", "통과", "재검사"}
REJECT_ACTIONS = {"재작업", "폐기", "라인정지"}
ALLOWED_ABSTAIN_REASONS = {
    "insufficient_visual_evidence",
    "ungrounded_anomaly",
    "evidence_conflict",
    "validation_failure",
}


def normalize(text: str) -> str:
    """공백·구두점을 제거한 소문자 비교용 문자열을 만든다."""
    return re.sub(r"[^\w]", "", str(text).lower(), flags=re.UNICODE)


def _is_grounded(candidate: str, sources: list[str], threshold: float) -> bool:
    normalized = normalize(candidate)
    if not normalized:
        return False
    for source in sources:
        actual = normalize(source)
        if normalized in actual:
            return True
        if SequenceMatcher(None, normalized, actual).ratio() >= threshold:
            return True
    return False


def _validation_threshold() -> float:
    """환경변수가 잘못되어 무조건 인용을 허용하지 않는다."""
    try:
        threshold = float(os.environ.get("VALIDATE_THRESH", "0.85"))
    except ValueError:
        return 0.85
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        return 0.85
    return threshold


def _has_nonfinite_number(value) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_has_nonfinite_number(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_nonfinite_number(item) for item in value)
    return False


def _schema_errors(report: dict) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_KEYS - set(report))
    if missing:
        errors.append(f"필수 키 누락: {', '.join(missing)}")
    extra = sorted(set(report) - REQUIRED_KEYS)
    if extra:
        errors.append(f"unexpected keys: {', '.join(extra)}")
    if _has_nonfinite_number(report):
        errors.append("NaN/Infinity is not allowed in reports")
    if report.get("verdict") not in {"PASS", "REJECT", "ABSTAIN"}:
        errors.append("verdict 값이 PASS/REJECT/ABSTAIN이 아님")
    if report.get("confidence") not in {"high", "low"}:
        errors.append("confidence 값이 high/low가 아님")
    reason = report.get("abstain_reason")
    if reason is not None and reason not in ALLOWED_ABSTAIN_REASONS:
        errors.append("abstain_reason 값이 허용 목록에 없음")
    if report.get("defect_type") is not None and not isinstance(
        report.get("defect_type"), str
    ):
        errors.append("defect_type은 문자열 또는 null이어야 함")
    observations = report.get("observations")

    def valid_source_id(value) -> bool:
        return isinstance(value, str) and bool(value.strip())

    if not isinstance(observations, list) or any(
        not isinstance(value, dict)
        or set(value) != {"text", "source_call_id"}
        or not isinstance(value.get("text"), str)
        or not valid_source_id(value.get("source_call_id"))
        for value in (observations or [])
    ):
        errors.append("observations 스키마 불일치")
    clauses = report.get("spec_clauses")
    if not isinstance(clauses, list):
        errors.append("spec_clauses는 배열이어야 함")
    elif any(not isinstance(clause_id, str) for clause_id in clauses):
        errors.append("spec_clauses의 조항 ID는 문자열이어야 함")
    actions = report.get("actions")
    if (
        not isinstance(actions, list)
        or not actions
        or any(
            not isinstance(action, str) or action not in ALLOWED_ACTIONS
            for action in actions
        )
    ):
        errors.append("actions가 허용된 조치 목록이 아님")
    if not isinstance(report.get("lot"), dict):
        errors.append("lot은 객체여야 함")
    if not isinstance(report.get("summary"), str) or not report.get("summary", "").strip():
        errors.append("summary가 비어 있음")
    return errors


def _invariant_errors(report: dict) -> list[str]:
    """판정값별 공개 보고서 불변조건을 검사한다."""
    verdict = report.get("verdict")
    defect_type = report.get("defect_type")
    confidence = report.get("confidence")
    reason = report.get("abstain_reason")
    clauses = report.get("spec_clauses")
    actions = report.get("actions")
    errors: list[str] = []

    if verdict == "PASS":
        if not isinstance(report.get("observations"), list) or not report.get(
            "observations"
        ):
            errors.append("PASS는 observations 1개 이상이 필요함")
        if defect_type is not None:
            errors.append("PASS는 defect_type이 null이어야 함")
        if reason is not None:
            errors.append("PASS는 abstain_reason이 null이어야 함")
        if clauses != []:
            errors.append("PASS는 spec_clauses가 비어야 함")
        if actions != ["통과"]:
            errors.append("PASS는 actions=[통과]여야 함")
    elif verdict == "REJECT":
        if not isinstance(report.get("observations"), list) or not report.get(
            "observations"
        ):
            errors.append("REJECT는 observations 1개 이상이 필요함")
        if not isinstance(defect_type, str) or not defect_type:
            errors.append("REJECT는 defect_type이 필요함")
        if reason is not None:
            errors.append("REJECT는 abstain_reason이 null이어야 함")
        if not isinstance(clauses, list) or not clauses:
            errors.append("REJECT는 검증된 spec_clauses가 필요함")
        if (
            not isinstance(actions, list)
            or not actions
            or any(
                not isinstance(action, str) or action not in REJECT_ACTIONS
                for action in actions
            )
        ):
            errors.append("REJECT 조치는 재작업/폐기/라인정지만 허용됨")
    elif verdict == "ABSTAIN":
        if defect_type is not None:
            errors.append("ABSTAIN은 defect_type이 null이어야 함")
        if confidence != "low":
            errors.append("ABSTAIN은 confidence=low여야 함")
        if reason not in ALLOWED_ABSTAIN_REASONS:
            errors.append("ABSTAIN은 허용된 abstain_reason이 필요함")
        if clauses != []:
            errors.append("ABSTAIN은 spec_clauses가 비어야 함")
        if actions != ["재검사"]:
            errors.append("ABSTAIN은 actions=[재검사]여야 함")
    return errors


def _validation_failure_report(state: AgentState) -> dict:
    """재작성 상한 뒤에도 외부 근거를 새로 만들지 않는 유효 보고서를 만든다."""
    observations: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for observation in state.get("observations", []):
        if not isinstance(observation, dict):
            continue
        text = observation.get("text")
        call_id = observation.get("call_id")
        if (
            not isinstance(text, str)
            or not text.strip()
            or not isinstance(call_id, str)
            or not call_id.strip()
        ):
            continue
        key = (text, call_id)
        if key not in seen:
            observations.append({"text": text, "source_call_id": call_id})
            seen.add(key)
    cross_check = state.get("cross_check") or {}
    text = cross_check.get("summary")
    call_id = cross_check.get("call_id")
    if (
        isinstance(text, str)
        and text.strip()
        and isinstance(call_id, str)
        and call_id.strip()
    ):
        key = (text, call_id)
        if key not in seen:
            observations.append({"text": text, "source_call_id": call_id})

    report = {
        "verdict": "ABSTAIN",
        "defect_type": None,
        "confidence": "low",
        "abstain_reason": "validation_failure",
        "observations": observations,
        "spec_clauses": [],
        "actions": ["재검사"],
        "lot": state_lot(state),
    }
    report["summary"] = deterministic_summary(report)
    return report


def validate_report(state: AgentState, deps: SupervisorDeps) -> list[str]:
    report = state.get("report") or {}
    if not isinstance(report, dict):
        return ["report는 객체여야 함"]
    errors = _schema_errors(report)
    errors.extend(_invariant_errors(report))

    # analysis는 코드 규칙이 내린 결정이다. 작성 LLM이 보고서에서 이를
    # 바꾸면 검증·HITL을 우회할 수 있으므로 상태와의 일치도 스키마 계약으로 본다.
    for key in ("verdict", "defect_type", "confidence", "abstain_reason"):
        if report.get(key) != state.get(key):
            errors.append(
                f"코드 판정과 보고서 {key} 불일치: "
                f"{state.get(key)} != {report.get(key)}"
            )
    if report.get("lot") != state_lot(state):
        errors.append("상태와 보고서 lot 불일치")
    if isinstance(report.get("summary"), str) and report.get(
        "summary"
    ) != deterministic_summary(report):
        errors.append("보고서 summary가 검증 가능한 관찰·판정 요약과 불일치")

    sources_by_id: dict[str, list[str]] = {}
    actual_texts: list[str] = []
    for source in state.get("observations", []):
        if not isinstance(source, dict):
            continue
        source_id = source.get("call_id")
        source_text = source.get("text")
        if isinstance(source_text, str) and source_text.strip():
            actual_texts.append(source_text)
            if isinstance(source_id, str) and source_id.strip():
                sources_by_id.setdefault(source_id, []).append(source_text)
    cross_check = state.get("cross_check") or {}
    if cross_check.get("summary"):
        actual_texts.append(str(cross_check["summary"]))
        cross_id = cross_check.get("call_id")
        if isinstance(cross_id, str) and cross_id.strip():
            sources_by_id.setdefault(cross_id, []).append(
                str(cross_check["summary"])
            )
    threshold = _validation_threshold()
    report_observations = report.get("observations", [])
    for observation in (
        report_observations if isinstance(report_observations, list) else []
    ):
        if not isinstance(observation, dict):
            continue
        candidate = observation.get("text", "")
        source_id = observation.get("source_call_id")
        if not isinstance(source_id, str) or source_id not in sources_by_id:
            errors.append(f"존재하지 않는 관찰 출처: {source_id}")
        elif not _is_grounded(candidate, sources_by_id[source_id], threshold):
            errors.append(f"관찰문과 출처가 일치하지 않음: {source_id}")
    for observation in report.get("observations", []) if isinstance(report.get("observations"), list) else []:
        if isinstance(observation, dict) and not _is_grounded(
            observation.get("text", ""), actual_texts, threshold
        ):
            errors.append(f"근거 없는 관찰: {observation.get('text', '')}")

    clauses = report.get("spec_clauses")
    if isinstance(clauses, list):
        judged = {
            value
            for value in (state.get("judgement") or {}).get("violations") or []
            if isinstance(value, str)
        }
        for clause_id in clauses:
            if not isinstance(clause_id, str):
                continue
            if deps.vector_store is None or not deps.vector_store.has(
                clause_id, state.get("item", "")
            ):
                errors.append(f"존재하지 않는 기준 조항: {clause_id}")
            elif clause_id not in judged:
                errors.append(f"Spec-Judge가 위반으로 판정하지 않은 조항: {clause_id}")

    defect_type = report.get("defect_type")
    allowed_types = set(load_type_map().get(state.get("item", ""), {})) - {"good"}
    if defect_type is not None and defect_type not in allowed_types:
        errors.append(f"품목에 없는 결함 유형: {defect_type}")
    if report.get("verdict") == "REJECT":
        allowed_actions = load_guide_actions(defect_type)
        report_actions = report.get("actions")
        if not allowed_actions:
            errors.append(f"결함 유형의 유효한 조치 가이드가 없음: {defect_type}")
        elif isinstance(report_actions, list) and any(
            action not in allowed_actions for action in report_actions
        ):
            errors.append(f"조치 가이드에 없는 actions: {report_actions}")
    return errors


@traceable(name="validate_node")
async def validate_node(state: AgentState, deps: SupervisorDeps) -> dict:
    errors = validate_report(state, deps)
    if (state.get("ablation") or deps.ablation) == "no_validate":
        return {
            "validation_errors": [],
            "validation_feedback": None,
            "validation_passed": True,
            "events": event(
                state,
                "Validation",
                verdict="accepted",
                errors=[],
                would_fail=errors,
            ),
        }

    attempts = int(state.get("validation_attempts", 0))
    if errors and attempts < 3:
        return {
            "validation_errors": errors,
            "validation_feedback": "; ".join(errors),
            "validation_passed": False,
            "events": event(
                state, "Validation", verdict="rejected", errors=errors
            ),
        }
    if errors:
        report = _validation_failure_report(state)
        return {
            "draft": json.dumps(report, ensure_ascii=False),
            "report": report,
            "verdict": "ABSTAIN",
            "defect_type": None,
            "confidence": "low",
            "abstain_reason": "validation_failure",
            "approval": {"required": True, "decision": None, "note": ""},
            "validation_errors": errors,
            "validation_feedback": None,
            "validation_passed": True,
            "events": event(
                state, "Validation", verdict="abstained", errors=errors
            ),
        }
    return {
        "validation_errors": [],
        "validation_feedback": None,
        "validation_passed": True,
        "events": event(state, "Validation", verdict="accepted", errors=[]),
    }
