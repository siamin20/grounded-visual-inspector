from __future__ import annotations

import pytest

from eval.evaluators import (
    binary_metrics,
    citation_fidelity,
    confidence_flag_validity,
    cost_latency,
    mcnemar,
    type_metrics,
)


def _row(
    row_id: str,
    is_defect: bool,
    verdict: str,
    **overrides: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "id": row_id,
        "is_defect": is_defect,
        "label": "scratch_neck" if is_defect else "good",
        "item": "screw",
        "verdict": verdict,
        "error": "",
    }
    row.update(overrides)
    return row


def test_binary_metrics_matches_hand_calculation() -> None:
    rows = [
        _row("tp-1", True, "REJECT"),
        _row("tp-2", True, "REJECT"),
        _row("tn", False, "PASS"),
        _row("fp", False, "REJECT"),
        _row("fn", True, "PASS"),
        _row("failed", True, "REJECT", error="timeout"),
        _row("invalid", True, "UNKNOWN"),
    ]

    result = binary_metrics(rows)

    assert result["n"] == 5
    assert (result["tp"], result["tn"], result["fp"], result["fn"]) == (2, 1, 1, 1)
    assert result["accuracy"] == pytest.approx(3 / 5)
    assert result["balanced_accuracy"] == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["fpr"] == pytest.approx(1 / 2)
    assert result["fnr"] == pytest.approx(1 / 3)


def test_binary_metrics_handles_empty_and_zero_denominators() -> None:
    empty = binary_metrics([])
    assert empty["n"] == 0
    for name in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "fpr", "fnr"):
        assert empty[name] == 0.0

    positives_only = binary_metrics([_row("only", True, "REJECT")])
    assert positives_only["recall"] == 1.0
    assert positives_only["fpr"] == 0.0
    assert positives_only["balanced_accuracy"] == 0.5


@pytest.fixture
def small_type_map() -> dict[str, dict[str, list[str]]]:
    return {
        "screw": {
            "good": ["good", "정상"],
            "scratch_neck": ["scratch_neck", "목 긁힘"],
            "thread_side": ["thread_side", "측면 나사산 손상"],
        }
    }


def test_type_metrics_normalizes_aliases_and_matches_hand_calculation(
    small_type_map: dict[str, dict[str, list[str]]],
) -> None:
    rows = [
        _row("s1", True, "REJECT", label="scratch_neck", defect_type="  목 긁힘  "),
        _row("s2", True, "REJECT", label="scratch_neck", defect_type="thread_side"),
        _row("t1", True, "REJECT", label="thread_side", defect_type="측면 나사산 손상"),
        _row("t2", True, "REJECT", label="thread_side", defect_type=None),
        _row("normal", False, "PASS", label="good", defect_type="scratch_neck"),
        _row("failed", True, "REJECT", label="scratch_neck", defect_type="scratch_neck", error="failed"),
    ]

    result = type_metrics(rows, small_type_map)

    assert result["n"] == 4
    assert result["macro_f1"] == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert result["per_type"]["scratch_neck"] == {
        "tp": 1,
        "fp": 0,
        "fn": 1,
        "support": 2,
        "precision": 1.0,
        "recall": 0.5,
        "f1": pytest.approx(2 / 3),
    }
    assert result["per_type"]["thread_side"] == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "support": 2,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }


def test_type_metrics_handles_empty_rows_and_unknown_prediction(
    small_type_map: dict[str, dict[str, list[str]]],
) -> None:
    assert type_metrics([], small_type_map) == {"n": 0, "macro_f1": 0.0, "per_type": {}}

    result = type_metrics(
        [_row("unknown", True, "REJECT", label="scratch_neck", defect_type="not-in-map")],
        small_type_map,
    )
    assert result["n"] == 1
    assert result["macro_f1"] == 0.0
    assert result["per_type"]["scratch_neck"]["fn"] == 1


def test_mcnemar_exact_binomial_and_strata() -> None:
    rows_a: list[dict[str, object]] = []
    rows_b: list[dict[str, object]] = []

    for index in range(10):
        is_defect = index >= 5
        rows_a.append(_row(f"a-only-{index}", is_defect, "REJECT" if is_defect else "PASS"))
        rows_b.append(_row(f"a-only-{index}", is_defect, "PASS" if is_defect else "REJECT"))
    for index, is_defect in enumerate((False, True)):
        rows_a.append(_row(f"b-only-{index}", is_defect, "PASS" if is_defect else "REJECT"))
        rows_b.append(_row(f"b-only-{index}", is_defect, "REJECT" if is_defect else "PASS"))

    rows_a.extend([_row("both-correct", False, "PASS"), _row("both-wrong", True, "PASS")])
    rows_b.extend([_row("both-correct", False, "PASS"), _row("both-wrong", True, "PASS")])

    result = mcnemar(rows_a, rows_b)

    assert result["overall"] == {
        "n": 14,
        "both_correct": 1,
        "both_wrong": 1,
        "b": 10,
        "c": 2,
        "p_value": pytest.approx(0.03857421875),
    }
    for stratum in ("normal", "defect"):
        assert result[stratum]["n"] == 7
        assert result[stratum]["b"] == 5
        assert result[stratum]["c"] == 1
        assert result[stratum]["p_value"] == pytest.approx(0.21875)


def test_mcnemar_skips_unpaired_errors_and_mismatched_truth() -> None:
    rows_a = [
        _row("valid", False, "PASS"),
        _row("error", False, "PASS", error="timeout"),
        _row("mismatch", False, "PASS"),
        _row("a-only", True, "REJECT"),
    ]
    rows_b = [
        _row("valid", False, "PASS"),
        _row("error", False, "PASS"),
        _row("mismatch", True, "REJECT"),
        _row("b-only", True, "REJECT"),
    ]

    result = mcnemar(rows_a, rows_b)

    assert result["overall"]["n"] == 1
    assert result["overall"]["p_value"] == 1.0
    assert result["normal"]["n"] == 1
    assert result["defect"]["n"] == 0
    assert result["defect"]["p_value"] == 1.0


def test_confidence_flag_validity_matches_hand_calculation() -> None:
    rows = [
        _row("low-wrong-1", True, "PASS", confidence="low"),
        _row("low-wrong-2", False, "REJECT", confidence=" LOW "),
        _row("high-right-1", True, "REJECT", confidence="high"),
        _row("high-right-2", False, "PASS", confidence="high"),
        _row("ignored", True, "PASS", confidence="low", error="failed"),
    ]

    result = confidence_flag_validity(rows)

    assert result == {
        "total_count": 4,
        "total_errors": 2,
        "overall_error_rate": 0.5,
        "low_count": 2,
        "low_errors": 2,
        "low_error_rate": 1.0,
        "error_rate_ratio": 2.0,
    }


def test_confidence_flag_validity_handles_no_rows_or_errors() -> None:
    assert confidence_flag_validity([]) == {
        "total_count": 0,
        "total_errors": 0,
        "overall_error_rate": 0.0,
        "low_count": 0,
        "low_errors": 0,
        "low_error_rate": 0.0,
        "error_rate_ratio": 0.0,
    }
    result = confidence_flag_validity([_row("right", False, "PASS", confidence="low")])
    assert result["overall_error_rate"] == 0.0
    assert result["low_error_rate"] == 0.0
    assert result["error_rate_ratio"] == 0.0


def test_citation_fidelity_uses_raw_contiguous_eight_character_ngrams() -> None:
    rows = [
        {
            "observations_text": ["prefix-ABCDEFGH-suffix"],
            "report": {
                "observations": [
                    {"text": "xxABCDEFGHyy"},
                    {"text": "xxabcdefghyy"},
                    {"text": "ABCDEFG"},
                ]
            },
            "error": "",
        },
        {
            "observations_text": ["1234", {"text": "5678"}],
            "report": {"observations": [{"text": "12345678"}]},
            "error": "",
        },
        {
            "observations_text": "ABCDEFGH",
            "report": {"observations": [{"text": "ABCDEFGH"}]},
            "error": "timeout",
        },
    ]

    result = citation_fidelity(rows)

    assert result == {
        "total_citations": 4,
        "matched_citations": 1,
        "citation_fidelity": 0.25,
    }


def test_citation_fidelity_handles_missing_or_empty_reports() -> None:
    assert citation_fidelity([]) == {
        "total_citations": 0,
        "matched_citations": 0,
        "citation_fidelity": 0.0,
    }
    assert citation_fidelity([{"report": {}}, {"report": None}])["total_citations"] == 0


def test_cost_latency_matches_hand_calculation_and_includes_error_attempts() -> None:
    rows = [
        {"latency_s": 1, "tokens": 10, "usd": 0.1, "approval_required": True, "validation_attempts": 0},
        {"latency_s": 2, "tokens": 20, "usd": 0.2, "approval_required": False, "validation_attempts": 1},
        {"latency_s": 3, "tokens": 30, "usd": 0.3, "approval_required": "true", "validation_attempts": 2},
        {
            "latency_s": 4,
            "tokens": 40,
            "usd": 0.4,
            "approval_required": None,
            "validation_attempts": 3,
            "error": "failed",
        },
    ]

    result = cost_latency(rows)

    assert result["n"] == 4
    assert result["latency_p50_s"] == 2.5
    assert result["latency_p95_s"] == pytest.approx(3.85)
    assert result["mean_tokens"] == 25.0
    assert result["mean_usd"] == pytest.approx(0.25)
    assert result["hitl_rate"] == 0.5
    assert result["mean_validation_attempts"] == 1.5


def test_cost_latency_handles_empty_and_malformed_values() -> None:
    assert cost_latency([]) == {
        "n": 0,
        "latency_p50_s": 0.0,
        "latency_p95_s": 0.0,
        "mean_tokens": 0.0,
        "mean_usd": 0.0,
        "hitl_rate": 0.0,
        "mean_validation_attempts": 0.0,
    }

    result = cost_latency(
        [
            {
                "latency_s": "not-a-number",
                "tokens": -10,
                "usd": float("nan"),
                "approval_required": "false",
                "validation_attempts": -1,
            }
        ]
    )
    assert result == {
        "n": 1,
        "latency_p50_s": 0.0,
        "latency_p95_s": 0.0,
        "mean_tokens": 0.0,
        "mean_usd": 0.0,
        "hitl_rate": 0.0,
        "mean_validation_attempts": 0.0,
    }
