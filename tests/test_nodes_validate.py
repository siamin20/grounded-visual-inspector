"""validate의 네 규칙과 재작성 상한 계약 테스트."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent.guide_actions import load_guide_actions
from agent.nodes.report_contract import deterministic_summary
from agent.nodes.validate import validate_node, validate_report


class ClauseStore:
    """품목 소유권까지 확인할 수 있는 최소 조항 저장소."""

    clauses = {
        ("SCREW-03", "screw"),
        ("SCREW-04", "screw"),
        ("BOTTLE-03", "bottle"),
    }

    def has(self, clause_id: str, item: str) -> bool:
        return (clause_id, item) in self.clauses


def _deps(*, ablation: str = "") -> SimpleNamespace:
    return SimpleNamespace(vector_store=ClauseStore(), ablation=ablation)


def test_validate_rejects_unknown_fields_and_nonfinite_numbers(
    inspection_state: dict,
) -> None:
    unknown = deepcopy(inspection_state)
    unknown["report"]["model_score"] = 0.99
    unknown["report"]["observations"][0]["score"] = 0.8
    unknown_errors = validate_report(unknown, _deps())

    nonfinite = deepcopy(inspection_state)
    nonfinite["report"]["lot"] = {"temperature": float("nan")}
    nonfinite["lot"] = {"temperature": float("nan")}
    nonfinite_errors = validate_report(nonfinite, _deps())

    assert any("unexpected keys" in error for error in unknown_errors)
    assert any("observations" in error for error in unknown_errors)
    assert any("NaN/Infinity" in error for error in nonfinite_errors)


def test_validate_rule_1_accepts_normalized_substring_and_rejects_invention(
    inspection_state: dict,
) -> None:
    state = deepcopy(inspection_state)
    state["observations"][0]["text"] = (
        "검사 결과: 목 부위에 선형 긁힘이 보입니다."
    )
    state["report"]["observations"][0]["text"] = "목부위에, 선형긁힘이"
    assert not any(
        "근거 없는 관찰" in error for error in validate_report(state, _deps())
    )

    state["report"]["observations"][0]["text"] = "우측 상단에 균열이 있습니다."
    assert any(
        "근거 없는 관찰" in error for error in validate_report(state, _deps())
    )


def test_validate_rule_1_honors_similarity_threshold_boundary(
    monkeypatch: pytest.MonkeyPatch,
    inspection_state: dict,
) -> None:
    state = deepcopy(inspection_state)
    state["observations"][0]["text"] = "abcdefghik"
    state["cross_check"] = {}
    state["report"]["observations"][0]["text"] = "abcdefghij"

    monkeypatch.setenv("VALIDATE_THRESH", "0.9")
    at_boundary = validate_report(state, _deps())
    assert not any("근거 없는 관찰" in error for error in at_boundary)

    monkeypatch.setenv("VALIDATE_THRESH", "0.9001")
    above_boundary = validate_report(state, _deps())
    assert any("근거 없는 관찰" in error for error in above_boundary)


@pytest.mark.parametrize("threshold", ["-1", "0", "nan", "inf", "invalid"])
def test_validate_invalid_threshold_cannot_make_invented_text_pass(
    monkeypatch: pytest.MonkeyPatch,
    inspection_state: dict,
    threshold: str,
) -> None:
    state = deepcopy(inspection_state)
    state["report"]["observations"][0]["text"] = (
        "완전히 지어낸 폐기 사유입니다."
    )
    monkeypatch.setenv("VALIDATE_THRESH", threshold)

    errors = validate_report(state, _deps())

    invented = state["report"]["observations"][0]["text"]
    assert errors
    assert any(invented in value for value in errors)


def test_validate_rejects_empty_decision_evidence_and_forged_source_ids(
    inspection_state: dict,
) -> None:
    empty = deepcopy(inspection_state)
    empty["report"]["observations"] = []
    assert any(
        "observations 1" in error for error in validate_report(empty, _deps())
    )

    forged = deepcopy(inspection_state)
    forged["report"]["observations"][0]["source_call_id"] = "made-up-call"
    assert any(
        "made-up-call" in error for error in validate_report(forged, _deps())
    )

    mismatched = deepcopy(inspection_state)
    mismatched["cross_check"]["summary"] = "교차 걸과가 없습니다."
    mismatched["report"]["observations"][0]["source_call_id"] = "cc-full"
    assert any(
        "cc-full" in error for error in validate_report(mismatched, _deps())
    )


@pytest.mark.parametrize(
    ("clause_id", "expected_error"),
    [
        ("MADE-UP-99", "존재하지 않는 기준 조항"),
        ("BOTTLE-03", "존재하지 않는 기준 조항"),
        ("SCREW-03", "Spec-Judge가 위반으로 판정하지 않은 조항"),
    ],
)
def test_validate_rule_2_checks_clause_existence_item_and_judgement(
    inspection_state: dict,
    clause_id: str,
    expected_error: str,
) -> None:
    state = deepcopy(inspection_state)
    state["report"]["spec_clauses"] = [clause_id]

    errors = validate_report(state, _deps())

    assert any(expected_error in error for error in errors)


@pytest.mark.parametrize("defect_type", ["invented_type", "good"])
def test_validate_rule_3_rejects_type_outside_item_map(
    inspection_state: dict,
    defect_type: str,
) -> None:
    state = deepcopy(inspection_state)
    state["defect_type"] = defect_type
    state["report"]["defect_type"] = defect_type

    errors = validate_report(state, _deps())

    assert any("품목에 없는 결함 유형" in error for error in errors)


@pytest.mark.parametrize(
    ("case", "expected_error"),
    [
        ("missing_summary", "필수 키 누락"),
        ("invalid_verdict", "verdict 값"),
        ("invalid_confidence", "confidence 값"),
        ("invalid_observations", "observations 스키마"),
        ("list_source_id", "observations 스키마"),
        ("invalid_clauses", "spec_clauses는 배열"),
        ("invalid_actions", "actions가 허용된 조치"),
        ("invalid_actions_type", "actions가 허용된 조치"),
        ("invalid_lot", "lot은 객체"),
        ("blank_summary", "summary가 비어 있음"),
    ],
)
def test_validate_rule_4_rejects_invalid_report_schema(
    inspection_state: dict,
    case: str,
    expected_error: str,
) -> None:
    state = deepcopy(inspection_state)
    report = state["report"]
    if case == "missing_summary":
        del report["summary"]
    elif case == "invalid_verdict":
        report["verdict"] = "UNKNOWN"
    elif case == "invalid_confidence":
        report["confidence"] = "medium"
    elif case == "invalid_observations":
        report["observations"] = [
            {"text": "관찰", "source_call_id": []}
        ]
    elif case == "list_source_id":
        report["observations"] = [
            {"text": "관찰", "source_call_id": ["obs-full", "cc-full"]}
        ]
    elif case == "invalid_clauses":
        report["spec_clauses"] = "SCREW-04"
    elif case == "invalid_actions":
        report["actions"] = []
    elif case == "invalid_actions_type":
        report["actions"] = 42
    elif case == "invalid_lot":
        report["lot"] = []
    elif case == "blank_summary":
        report["summary"] = "   "

    errors = validate_report(state, _deps())

    assert any(expected_error in error for error in errors)


@pytest.mark.asyncio
async def test_validate_replaces_invalid_report_with_abstain_at_third_attempt(
    inspection_state: dict,
) -> None:
    state = deepcopy(inspection_state)
    state["report"]["observations"][0]["text"] = "근거에 없는 균열"
    state["validation_attempts"] = 2

    retry = await validate_node(state, _deps())
    assert retry["validation_passed"] is False
    assert retry["validation_terminal"] is False
    assert retry["events"][-1]["passed"] is False
    assert retry["events"][-1]["verdict"] == "rejected"

    state["validation_attempts"] = 3
    abstained = await validate_node(state, _deps())
    assert abstained["validation_passed"] is False
    assert abstained["validation_terminal"] is True
    assert abstained["events"][-1]["passed"] is False
    assert abstained["verdict"] == "ABSTAIN"
    assert abstained["defect_type"] is None
    assert abstained["confidence"] == "low"
    assert abstained["abstain_reason"] == "validation_failure"
    expected_report = {
        "verdict": "ABSTAIN",
        "defect_type": None,
        "confidence": "low",
        "abstain_reason": "validation_failure",
        "observations": [
            {
                "text": state["observations"][0]["text"],
                "source_call_id": "obs-full",
            },
            {
                "text": state["cross_check"]["summary"],
                "source_call_id": "cc-full",
            },
        ],
        "spec_clauses": [],
        "actions": ["재검사"],
        "lot": {},
    }
    expected_report["summary"] = deterministic_summary(expected_report)
    assert abstained["report"] == expected_report
    assert abstained["approval"] == {
        "required": True,
        "decision": None,
        "note": "",
    }
    assert abstained["validation_errors"]
    assert abstained["events"][-1]["verdict"] == "abstained"

    state.update(abstained)
    assert validate_report(state, _deps()) == []


@pytest.mark.parametrize(
    ("updates", "expected_error"),
    [
        (
            {
                "verdict": "PASS",
                "defect_type": "scratch_neck",
                "spec_clauses": ["SCREW-04"],
                "actions": ["재작업"],
            },
            "PASS는 defect_type",
        ),
        (
            {
                "verdict": "REJECT",
                "defect_type": "scratch_neck",
                "spec_clauses": [],
                "actions": ["재작업"],
            },
            "REJECT는 검증된 spec_clauses",
        ),
        (
            {
                "verdict": "ABSTAIN",
                "defect_type": None,
                "confidence": "high",
                "abstain_reason": "validation_failure",
                "spec_clauses": [],
                "actions": ["재검사"],
            },
            "ABSTAIN은 confidence=low",
        ),
    ],
)
def test_validate_enforces_verdict_invariants(
    inspection_state: dict,
    updates: dict,
    expected_error: str,
) -> None:
    state = deepcopy(inspection_state)
    state.update({
        key: value
        for key, value in updates.items()
        if key in {"verdict", "defect_type", "confidence", "abstain_reason"}
    })
    state["report"].update(updates)

    errors = validate_report(state, _deps())

    assert any(expected_error in error for error in errors)


def test_validate_rejects_lot_and_summary_not_derived_from_state(
    inspection_state: dict,
) -> None:
    state = deepcopy(inspection_state)
    state["lot"] = {"lot_id": "LOT-001", "line": "L1"}
    state["report"]["lot"] = {"lot_id": "FORGED", "line": "L1"}
    state["report"]["summary"] = "관찰에 없는 긴급 폐기 사유입니다."

    errors = validate_report(state, _deps())

    assert "상태와 보고서 lot 불일치" in errors
    assert any("summary" in error for error in errors)


def test_validate_accepts_exact_lot_and_deterministic_summary(
    inspection_state: dict,
) -> None:
    state = deepcopy(inspection_state)
    state["lot"] = {"lot_id": "LOT-001", "meta": {"line": "L1"}}
    state["report"]["lot"] = {"lot_id": "LOT-001", "meta": {"line": "L1"}}
    state["report"]["summary"] = deterministic_summary(state["report"])

    assert validate_report(state, _deps()) == []


def test_validate_rejects_action_outside_defect_guide(
    inspection_state: dict,
) -> None:
    state = deepcopy(inspection_state)
    state.update(item="bottle", defect_type="broken_large")
    state["judgement"] = {
        "violations": ["BOTTLE-03"],
        "defect_type": "broken_large",
    }
    state["report"].update(
        defect_type="broken_large",
        spec_clauses=["BOTTLE-03"],
        actions=["재작업"],
    )
    state["report"]["summary"] = deterministic_summary(state["report"])

    errors = validate_report(state, _deps())
    assert any("조치 가이드에 없는 actions" in error for error in errors)

    state["report"]["actions"] = ["폐기"]
    state["report"]["summary"] = deterministic_summary(state["report"])
    assert validate_report(state, _deps()) == []


@pytest.mark.parametrize(
    ("defect_type", "expected_actions"),
    [
        ("bent", ["재작업", "폐기"]),
        ("broken_large", ["폐기", "라인정지"]),
        ("broken_small", ["폐기", "라인정지"]),
        ("color", ["재작업", "폐기"]),
        ("contamination", ["폐기", "라인정지"]),
        ("crack", ["폐기", "라인정지"]),
        ("cut", ["폐기", "재작업"]),
        ("hole", ["폐기", "라인정지"]),
        ("print", ["재작업", "폐기"]),
        ("scratch", ["재작업", "폐기"]),
        ("scratch_head", ["재작업", "폐기"]),
        ("scratch_neck", ["재작업", "폐기"]),
        ("thread_side", ["재작업", "폐기"]),
        ("thread_top", ["재작업", "폐기"]),
    ],
)
def test_every_defect_guide_has_expected_actions(
    defect_type: str,
    expected_actions: list[str],
) -> None:
    assert load_guide_actions(defect_type) == expected_actions


@pytest.mark.asyncio
async def test_no_validate_keeps_failures_only_in_would_fail(
    inspection_state: dict,
) -> None:
    state = deepcopy(inspection_state)
    state["ablation"] = "no_validate"
    state["report"]["spec_clauses"] = ["MADE-UP-99"]

    result = await validate_node(state, _deps())

    assert result["validation_passed"] is True
    assert result["validation_terminal"] is True
    assert result["events"][-1]["skipped"] is True
    assert result["validation_errors"] == []
    assert any(
        "존재하지 않는 기준 조항" in error
        for error in result["events"][-1]["would_fail"]
    )
