"""보고서의 근거·조항·유형·스키마를 LLM 없이 검증한다."""

from difflib import SequenceMatcher
import os
import re

from langsmith import traceable

from agent.deps import SupervisorDeps
from agent.state import AgentState

from .common import event, load_type_map


REQUIRED_KEYS = {
    "verdict",
    "defect_type",
    "confidence",
    "observations",
    "spec_clauses",
    "actions",
    "lot",
    "summary",
}
ALLOWED_ACTIONS = {"재작업", "폐기", "라인정지", "통과"}


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


def _schema_errors(report: dict) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_KEYS - set(report))
    if missing:
        errors.append(f"필수 키 누락: {', '.join(missing)}")
    if report.get("verdict") not in {"PASS", "REJECT"}:
        errors.append("verdict 값이 PASS/REJECT가 아님")
    if report.get("confidence") not in {"high", "low"}:
        errors.append("confidence 값이 high/low가 아님")
    observations = report.get("observations")

    def valid_source_id(value) -> bool:
        return isinstance(value, str) or (
            isinstance(value, list)
            and bool(value)
            and all(isinstance(source_id, str) for source_id in value)
        )

    if not isinstance(observations, list) or any(
        not isinstance(value, dict)
        or not isinstance(value.get("text"), str)
        or not valid_source_id(value.get("source_call_id"))
        for value in (observations or [])
    ):
        errors.append("observations 스키마 불일치")
    if not isinstance(report.get("spec_clauses"), list):
        errors.append("spec_clauses는 배열이어야 함")
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


def validate_report(state: AgentState, deps: SupervisorDeps) -> list[str]:
    report = state.get("report") or {}
    errors = _schema_errors(report)

    # analysis는 코드 규칙이 내린 결정이다. 작성 LLM이 보고서에서 이를
    # 바꾸면 검증·HITL을 우회할 수 있으므로 상태와의 일치도 스키마 계약으로 본다.
    for key in ("verdict", "defect_type", "confidence"):
        if report.get(key) != state.get(key):
            errors.append(
                f"코드 판정과 보고서 {key} 불일치: "
                f"{state.get(key)} != {report.get(key)}"
            )

    actual_texts = [
        str(observation.get("text", ""))
        for observation in state.get("observations", [])
    ]
    cross_check = state.get("cross_check") or {}
    if cross_check.get("summary"):
        actual_texts.append(str(cross_check["summary"]))
    threshold = float(os.environ.get("VALIDATE_THRESH", "0.85"))
    for observation in report.get("observations", []) if isinstance(report.get("observations"), list) else []:
        if isinstance(observation, dict) and not _is_grounded(
            observation.get("text", ""), actual_texts, threshold
        ):
            errors.append(f"근거 없는 관찰: {observation.get('text', '')}")

    clauses = report.get("spec_clauses")
    if isinstance(clauses, list):
        judged = set((state.get("judgement") or {}).get("violations") or [])
        for clause_id in clauses:
            if deps.vector_store is None or not deps.vector_store.has(
                str(clause_id), state.get("item", "")
            ):
                errors.append(f"존재하지 않는 기준 조항: {clause_id}")
            elif clause_id not in judged:
                errors.append(f"Spec-Judge가 위반으로 판정하지 않은 조항: {clause_id}")

    defect_type = report.get("defect_type")
    allowed_types = load_type_map().get(state.get("item", ""), {})
    if defect_type is not None and defect_type not in allowed_types:
        errors.append(f"품목에 없는 결함 유형: {defect_type}")
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
        report = dict(state.get("report") or {})
        report["confidence"] = "low"
        approval = dict(state.get("approval") or {})
        approval["required"] = True
        return {
            "report": report,
            "confidence": "low",
            "approval": approval,
            "validation_errors": errors,
            "validation_feedback": None,
            "validation_passed": True,
            "events": event(
                state, "Validation", verdict="forced", errors=errors
            ),
        }
    return {
        "validation_errors": [],
        "validation_feedback": None,
        "validation_passed": True,
        "events": event(state, "Validation", verdict="accepted", errors=[]),
    }
