from __future__ import annotations

import pytest

from eval.evaluators import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_PERMUTATION_SAMPLES,
    DEFAULT_RESAMPLING_SEED,
    EvaluationDataError,
    binary_metrics,
    citation_fidelity,
    confidence_flag_validity,
    cost_latency,
    mcnemar,
    paired_balanced_accuracy_test,
    paired_macro_f1_test,
    type_metrics,
    wilson_interval,
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
        "split": "dev",
        "verdict": verdict,
        "error": "",
    }
    row.update(overrides)
    return row


def test_wilson_interval_matches_hand_values_and_boundary_contract() -> None:
    assert wilson_interval(0, 0) == (None, None)
    assert wilson_interval(0, 10) == pytest.approx((0.0, 0.2775327999))
    assert wilson_interval(10, 10) == pytest.approx((0.7224672001, 1.0))
    assert wilson_interval(5, 10) == pytest.approx((0.2365930905, 0.7634069095))


@pytest.mark.parametrize(
    ("successes", "total", "confidence"),
    [(-1, 1, 0.95), (2, 1, 0.95), (0, -1, 0.95), (0, 1, 0.0), (0, 1, 1.0)],
)
def test_wilson_interval_rejects_invalid_inputs(
    successes: int, total: int, confidence: float
) -> None:
    with pytest.raises(ValueError):
        wilson_interval(successes, total, confidence)


def test_binary_metrics_keeps_abstain_error_invalid_and_missing_in_denominator() -> None:
    expected = [
        _row("n-pass", False, "PASS"),
        _row("n-reject", False, "PASS"),
        _row("n-abstain", False, "PASS"),
        _row("n-error", False, "PASS"),
        _row("d-reject", True, "REJECT"),
        _row("d-pass", True, "REJECT"),
        _row("d-missing", True, "REJECT"),
        _row("d-invalid", True, "REJECT"),
    ]
    rows = [
        _row("n-pass", False, "PASS"),
        _row("n-reject", False, "REJECT"),
        _row("n-abstain", False, "ABSTAIN"),
        _row("n-error", False, "PASS", error="timeout"),
        _row("d-reject", True, "REJECT"),
        _row("d-pass", True, "PASS"),
        _row("d-invalid", True, "UNKNOWN"),
    ]

    result = binary_metrics(rows, expected)

    assert result["n"] == result["expected_n"] == 8
    assert result["observed_n"] == result["result_n"] == 7
    assert (result["missing_n"], result["error_n"], result["abstain_n"], result["invalid_n"]) == (1, 1, 1, 1)
    assert (result["tp"], result["tn"], result["fp"], result["fn"]) == (1, 1, 3, 3)
    assert result["accuracy"] == pytest.approx(2 / 8)
    assert result["balanced_accuracy"] == pytest.approx(0.25)
    assert result["fpr"] == pytest.approx(3 / 4)
    assert result["fnr"] == pytest.approx(3 / 4)
    assert result["coverage"] == pytest.approx(4 / 8)
    assert result["abstain_rate"] == pytest.approx(1 / 8)
    assert result["selective_accuracy"] == pytest.approx(2 / 4)
    assert result["selective_precision"] == pytest.approx(1 / 2)
    assert result["selective_recall"] == pytest.approx(1 / 2)
    assert result["selective_f1"] == pytest.approx(1 / 2)


def test_binary_metrics_handles_empty_and_zero_denominators() -> None:
    empty = binary_metrics([])
    assert empty["n"] == 0
    for name in (
        "accuracy",
        "balanced_accuracy",
        "fpr",
        "fnr",
        "coverage",
        "abstain_rate",
        "selective_accuracy",
        "selective_precision",
        "selective_recall",
        "selective_f1",
    ):
        assert empty[name] == 0.0

    positives_only = binary_metrics([_row("only", True, "REJECT")])
    assert positives_only["selective_recall"] == 1.0
    assert positives_only["fpr"] == 0.0
    assert positives_only["balanced_accuracy"] == 0.5


@pytest.mark.parametrize(
    ("rows", "field"),
    [
        (
            [
                _row("normal", False, "ABSTAIN"),
                _row("defect", True, "ABSTAIN"),
            ],
            "abstain_n",
        ),
        (
            [
                _row("normal", False, "PASS", error="timeout"),
                _row("defect", True, "REJECT", error="timeout"),
            ],
            "error_n",
        ),
        ([], "missing_n"),
    ],
)
def test_binary_metrics_all_unresolved_states_are_zero_accuracy(
    rows: list[dict[str, object]], field: str
) -> None:
    expected = [
        _row("normal", False, "PASS"),
        _row("defect", True, "REJECT"),
    ]

    result = binary_metrics(rows, expected)

    assert result["accuracy"] == 0.0
    assert result["balanced_accuracy"] == 0.0
    assert result["coverage"] == 0.0
    assert result[field] == 2


def test_binary_metrics_does_not_normalize_malformed_verdict_or_error_flag() -> None:
    expected = [
        _row("normal", False, "PASS"),
        _row("defect", True, "REJECT"),
    ]
    rows = [
        _row("normal", False, " pass "),
        _row("defect", True, "REJECT", error=False),
    ]

    result = binary_metrics(rows, expected)

    assert result["invalid_n"] == 1
    assert result["error_n"] == 1
    assert result["accuracy"] == 0.0
    assert result["coverage"] == 0.0


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

    assert result["n"] == 5
    assert result["error_n"] == 1
    assert result["macro_f1"] == pytest.approx(0.5)
    assert result["per_type"]["scratch_neck"] == {
        "tp": 1,
        "fp": 0,
        "fn": 2,
        "support": 3,
        "precision": 1.0,
        "recall": pytest.approx(1 / 3),
        "f1": 0.5,
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
    empty = type_metrics([], small_type_map)
    assert empty["n"] == 0
    assert empty["macro_f1"] == 0.0
    assert empty["per_type"] == {}

    result = type_metrics(
        [_row("unknown", True, "REJECT", label="scratch_neck", defect_type="not-in-map")],
        small_type_map,
    )
    assert result["n"] == 1
    assert result["macro_f1"] == 0.0
    assert result["per_type"]["scratch_neck"]["fn"] == 1


def test_type_metrics_accepts_prediction_only_from_clean_reject(
    small_type_map: dict[str, dict[str, list[str]]],
) -> None:
    expected = [
        _row("pass", True, "REJECT", label="scratch_neck"),
        _row("abstain", True, "REJECT", label="scratch_neck"),
        _row("error", True, "REJECT", label="scratch_neck"),
        _row("missing", True, "REJECT", label="scratch_neck"),
        _row("reject", True, "REJECT", label="scratch_neck"),
    ]
    rows = [
        _row("pass", True, "PASS", defect_type="scratch_neck"),
        _row("abstain", True, "ABSTAIN", defect_type="scratch_neck"),
        _row("error", True, "REJECT", defect_type="scratch_neck", error="timeout"),
        _row("reject", True, "REJECT", defect_type="scratch_neck"),
    ]

    result = type_metrics(rows, small_type_map, expected)

    assert result["n"] == 5
    assert result["missing_n"] == 1
    assert result["error_n"] == 1
    assert result["abstain_n"] == 1
    assert result["per_type"]["scratch_neck"]["tp"] == 1
    assert result["per_type"]["scratch_neck"]["fn"] == 4


def test_type_metrics_keeps_entire_missing_type_in_macro_support(
    small_type_map: dict[str, dict[str, list[str]]],
) -> None:
    expected = [
        _row("scratch", True, "REJECT", label="scratch_neck"),
        _row("thread", True, "REJECT", label="thread_side"),
    ]
    rows = [_row("scratch", True, "REJECT", label="scratch_neck", defect_type="목 긁힘")]

    result = type_metrics(rows, small_type_map, expected)

    assert result["per_type"]["scratch_neck"]["f1"] == 1.0
    assert result["per_type"]["thread_side"]["support"] == 1
    assert result["per_type"]["thread_side"]["f1"] == 0.0
    assert result["macro_f1"] == 0.5


def test_type_metrics_rejects_alias_collision_within_item() -> None:
    type_map = {
        "screw": {
            "good": ["good"],
            "scratch_neck": ["공통 별칭"],
            "thread_side": ["공통 별칭"],
        }
    }
    expected = [_row("one", True, "REJECT", label="scratch_neck")]

    with pytest.raises(EvaluationDataError, match="동의어가 충돌"):
        type_metrics([], type_map, expected)


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


def test_mcnemar_keeps_missing_errors_and_all_registered_ids() -> None:
    expected = [
        _row("valid", False, "PASS"),
        _row("error", False, "PASS"),
        _row("a-only", True, "REJECT"),
        _row("b-only", True, "REJECT"),
        _row("both-missing", True, "REJECT"),
    ]
    rows_a = [
        _row("valid", False, "PASS"),
        _row("error", False, "PASS", error="timeout"),
        _row("a-only", True, "REJECT"),
    ]
    rows_b = [
        _row("valid", False, "PASS"),
        _row("error", False, "PASS"),
        _row("b-only", True, "REJECT"),
    ]

    result = mcnemar(rows_a, rows_b, expected)

    assert result["overall"]["n"] == 5
    assert result["overall"]["both_correct"] == 1
    assert result["overall"]["both_wrong"] == 1
    assert result["overall"]["b"] == 1
    assert result["overall"]["c"] == 2
    assert result["overall"]["p_value"] == 1.0
    assert result["normal"]["n"] == 2
    assert result["defect"]["n"] == 3
    assert result["defect"]["p_value"] == 1.0
    assert mcnemar(list(reversed(rows_a)), list(reversed(rows_b)), expected) == result


def test_paired_balanced_accuracy_exact_test_and_bootstrap_hand_value() -> None:
    expected = [
        _row("normal-1", False, "PASS"),
        _row("normal-2", False, "PASS"),
        _row("defect-1", True, "REJECT"),
        _row("defect-2", True, "REJECT"),
    ]
    rows_a = [
        _row("normal-1", False, "ABSTAIN"),
        _row("normal-2", False, "PASS", error="timeout"),
        # defect-1은 누락이며 defect-2는 명시적 오답이다.
        _row("defect-2", True, "PASS"),
    ]
    rows_b = [
        _row("normal-1", False, "PASS"),
        _row("normal-2", False, "PASS"),
        _row("defect-1", True, "REJECT"),
        _row("defect-2", True, "REJECT"),
    ]

    result = paired_balanced_accuracy_test(
        rows_a,
        rows_b,
        expected,
        permutation_samples=50,
        bootstrap_samples=50,
    )

    assert result["n"] == 4
    assert result["strata"] == {"normal": 2, "defect": 2}
    assert result["a"] == 0.0
    assert result["b"] == 1.0
    assert result["delta"] == 1.0
    assert (result["delta_ci95_low"], result["delta_ci95_high"]) == (1.0, 1.0)
    assert result["p_value"] == pytest.approx(2 / 16)
    assert result["permutation_method"] == "exact"
    assert result["permutation_samples"] == 16
    assert result["requested_permutation_samples"] == 50
    assert result["swappable_pairs"] == 4
    assert result["confidence_level"] == 0.95
    assert result["bootstrap_method"] == "paired_stratified_percentile"
    assert result["seed"] == DEFAULT_RESAMPLING_SEED


def test_paired_balanced_accuracy_monte_carlo_is_deterministic_and_order_invariant() -> None:
    expected: list[dict[str, object]] = []
    rows_a: list[dict[str, object]] = []
    rows_b: list[dict[str, object]] = []
    for index in range(18):
        is_defect = index >= 9
        expected.append(
            _row(f"sample-{index:02d}", is_defect, "REJECT" if is_defect else "PASS")
        )
        rows_a.append(
            _row(f"sample-{index:02d}", is_defect, "PASS" if is_defect else "REJECT")
        )
        rows_b.append(
            _row(f"sample-{index:02d}", is_defect, "REJECT" if is_defect else "PASS")
        )

    options = {"seed": 7, "permutation_samples": 200, "bootstrap_samples": 100}
    first = paired_balanced_accuracy_test(rows_a, rows_b, expected, **options)
    second = paired_balanced_accuracy_test(
        list(reversed(rows_a)),
        list(reversed(rows_b)),
        list(reversed(expected)),
        **options,
    )

    assert first == second
    assert first["permutation_method"] == "monte_carlo"
    assert first["permutation_samples"] == 200
    assert first["delta"] == 1.0
    assert 0.0 < first["p_value"] <= 1.0


def test_paired_macro_f1_uses_type_predictions_not_binary_mcnemar(
    small_type_map: dict[str, dict[str, list[str]]],
) -> None:
    expected: list[dict[str, object]] = []
    rows_a: list[dict[str, object]] = []
    rows_b: list[dict[str, object]] = []
    for index in range(12):
        label = "scratch_neck" if index < 6 else "thread_side"
        expected.append(_row(f"type-{index:02d}", True, "REJECT", label=label))
        # 이진 verdict는 두 조건 모두 정답이지만 A의 유형은 전부 틀린다.
        rows_a.append(
            _row(
                f"type-{index:02d}",
                True,
                "REJECT",
                label=label,
                defect_type="unknown-type",
            )
        )
        rows_b.append(
            _row(
                f"type-{index:02d}",
                True,
                "REJECT",
                label=label,
                defect_type=label,
            )
        )

    assert mcnemar(rows_a, rows_b, expected)["overall"]["p_value"] == 1.0
    result = paired_macro_f1_test(
        rows_a,
        rows_b,
        small_type_map,
        expected,
        permutation_samples=50,
        bootstrap_samples=50,
    )

    assert result["n"] == 12
    assert result["type_n"] == 2
    assert result["strata"] == {
        "screw/scratch_neck": 6,
        "screw/thread_side": 6,
    }
    assert result["a"] == 0.0
    assert result["b"] == 1.0
    assert result["delta"] == 1.0
    assert (result["delta_ci95_low"], result["delta_ci95_high"]) == (1.0, 1.0)
    assert result["p_value"] == pytest.approx(2 / 4096)
    assert result["permutation_method"] == "exact"
    assert result["permutation_samples"] == 4096
    assert result["swappable_pairs"] == 12
    assert result["confidence_level"] == 0.95


def test_paired_macro_f1_keeps_missing_error_abstain_and_is_order_invariant(
    small_type_map: dict[str, dict[str, list[str]]],
) -> None:
    expected = [
        _row("missing", True, "REJECT", label="scratch_neck"),
        _row("error", True, "REJECT", label="scratch_neck"),
        _row("abstain", True, "REJECT", label="thread_side"),
        _row("pass", True, "REJECT", label="thread_side"),
    ]
    rows_a = [
        _row(
            "error",
            True,
            "REJECT",
            label="scratch_neck",
            defect_type="scratch_neck",
            error="timeout",
        ),
        _row(
            "abstain",
            True,
            "ABSTAIN",
            label="thread_side",
            defect_type="thread_side",
        ),
        _row(
            "pass",
            True,
            "PASS",
            label="thread_side",
            defect_type="thread_side",
        ),
    ]
    rows_b = [
        _row(row["id"], True, "REJECT", label=row["label"], defect_type=row["label"])
        for row in expected
    ]
    options = {"seed": 11, "permutation_samples": 100, "bootstrap_samples": 50}

    first = paired_macro_f1_test(
        rows_a, rows_b, small_type_map, expected, **options
    )
    second = paired_macro_f1_test(
        list(reversed(rows_a)),
        list(reversed(rows_b)),
        small_type_map,
        list(reversed(expected)),
        **options,
    )

    assert first == second
    assert first["a"] == 0.0
    assert first["b"] == 1.0
    assert first["delta"] == 1.0


def test_paired_tests_expose_fixed_default_resampling_contract() -> None:
    assert DEFAULT_RESAMPLING_SEED == 20261020
    assert DEFAULT_PERMUTATION_SAMPLES > 0
    assert DEFAULT_BOOTSTRAP_SAMPLES > 0


def test_paired_tests_reuse_id_and_metadata_integrity(
    small_type_map: dict[str, dict[str, list[str]]],
) -> None:
    binary_expected = [
        _row("normal", False, "PASS"),
        _row("defect", True, "REJECT"),
    ]
    drift = _row("normal", False, "PASS", item="bottle")
    with pytest.raises(EvaluationDataError, match="item"):
        paired_balanced_accuracy_test(
            [drift],
            [],
            binary_expected,
            permutation_samples=5,
            bootstrap_samples=5,
        )

    type_expected = [_row("defect", True, "REJECT", label="scratch_neck")]
    duplicate = [
        _row("defect", True, "REJECT", defect_type="scratch_neck"),
        _row("defect", True, "REJECT", defect_type="scratch_neck"),
    ]
    with pytest.raises(EvaluationDataError, match="중복"):
        paired_macro_f1_test(
            duplicate,
            [],
            small_type_map,
            type_expected,
            permutation_samples=5,
            bootstrap_samples=5,
        )


@pytest.mark.parametrize("field", ["item", "label", "is_defect", "split"])
def test_evaluators_reject_canonical_metadata_mismatch(field: str) -> None:
    expected = [_row("one", False, "PASS")]
    observed = _row("one", False, "PASS")
    observed[field] = {
        "item": "bottle",
        "label": "scratch_neck",
        "is_defect": True,
        "split": "heldout",
    }[field]

    with pytest.raises(EvaluationDataError, match=field):
        binary_metrics([observed], expected)
    with pytest.raises(EvaluationDataError, match=field):
        mcnemar([observed], [], expected)


def test_evaluators_reject_duplicate_unknown_and_non_bool_truth() -> None:
    expected = [_row("one", False, "PASS")]
    with pytest.raises(EvaluationDataError, match="중복"):
        binary_metrics([_row("one", False, "PASS")] * 2, expected)
    with pytest.raises(EvaluationDataError, match="사전 등록에 없는"):
        binary_metrics([_row("other", False, "PASS")], expected)

    invalid_expected = _row("one", False, "PASS")
    invalid_expected["is_defect"] = 0
    with pytest.raises(EvaluationDataError, match="bool is_defect"):
        binary_metrics([], [invalid_expected])

    inconsistent = _row("one", False, "PASS", label="scratch_neck")
    with pytest.raises(EvaluationDataError, match="모순"):
        binary_metrics([], [inconsistent])

    whitespace_id = _row(" one ", False, "PASS")
    with pytest.raises(EvaluationDataError, match="앞뒤 공백"):
        binary_metrics([whitespace_id])

    empty_id = _row("one", False, "PASS")
    empty_id["id"] = ""
    with pytest.raises(EvaluationDataError, match="id"):
        binary_metrics([empty_id])


@pytest.mark.parametrize("field", ["item", "label", "is_defect", "split"])
def test_evaluators_reject_missing_result_metadata(field: str) -> None:
    expected = [_row("one", False, "PASS")]
    observed = _row("one", False, "PASS")
    observed.pop(field)

    with pytest.raises(EvaluationDataError, match=field):
        binary_metrics([observed], expected)


def test_confidence_flag_validity_matches_hand_calculation() -> None:
    rows = [
        _row("low-wrong-1", True, "PASS", confidence="low"),
        _row("low-wrong-2", False, "REJECT", confidence=" LOW "),
        _row("high-right-1", True, "REJECT", confidence="high"),
        _row("high-right-2", False, "PASS", confidence="high"),
        _row("high-wrong", True, "PASS", confidence="high"),
        _row("invalid-confidence", False, "PASS", confidence="medium"),
        _row("error", True, "PASS", confidence="low", error="failed"),
        _row("abstain", True, "ABSTAIN", confidence="low"),
    ]

    result = confidence_flag_validity(rows)

    assert result["total_count"] == 8
    assert result["total_errors"] == 5
    assert result["overall_error_rate"] == pytest.approx(5 / 8)
    assert result["selected_count"] == 6
    assert result["selected_errors"] == 3
    assert result["selected_error_rate"] == pytest.approx(1 / 2)
    assert (result["low_count"], result["low_errors"], result["low_error_rate"]) == (
        2,
        2,
        1.0,
    )
    assert result["non_low_count"] == 3
    assert result["non_low_errors"] == 1
    assert result["non_low_error_rate"] == pytest.approx(1 / 3)
    assert result["enrichment_ratio"] == pytest.approx(3.0)
    assert result["enrichment_available"] is True
    # 기존 low/종단 전체 비율 키는 보조 지표로 호환 유지한다.
    assert result["error_rate_ratio"] == pytest.approx(1.6)
    assert result["abstain_count"] == 1
    assert result["error_count"] == 1
    assert result["missing_count"] == 0
    assert result["invalid_count"] == 0
    assert result["invalid_confidence_count"] == 1
    assert result["low_error_ci95_low"] == pytest.approx(0.3423802275)
    assert result["low_error_ci95_high"] == 1.0


def test_confidence_flag_validity_handles_no_rows_or_errors() -> None:
    empty = confidence_flag_validity([])
    assert empty["total_count"] == 0
    assert empty["overall_error_rate"] == 0.0
    assert empty["selected_count"] == 0
    assert empty["selected_error_rate"] is None
    assert empty["low_error_rate"] is None
    assert empty["non_low_error_rate"] is None
    assert empty["enrichment_ratio"] is None
    assert empty["enrichment_available"] is False
    assert empty["low_error_ci95_low"] is None
    assert empty["low_error_ci95_high"] is None

    result = confidence_flag_validity([_row("right", False, "PASS", confidence="low")])
    assert result["overall_error_rate"] == 0.0
    assert result["low_error_rate"] == 0.0
    assert result["non_low_error_rate"] is None
    assert result["error_rate_ratio"] is None
    assert result["enrichment_ratio"] is None


def test_confidence_flag_validity_separates_missing_error_abstain_and_invalid() -> None:
    expected = [
        _row("selected", False, "PASS"),
        _row("abstain", True, "REJECT"),
        _row("error", True, "REJECT"),
        _row("invalid", False, "PASS"),
        _row("missing", True, "REJECT"),
    ]
    rows = [
        _row("selected", False, "PASS", confidence="high"),
        _row("abstain", True, "ABSTAIN", confidence="low"),
        _row("error", True, "REJECT", confidence="high", error="timeout"),
        _row("invalid", False, "UNKNOWN", confidence="low"),
    ]

    result = confidence_flag_validity(rows, expected)

    assert result["total_count"] == 5
    assert result["selected_count"] == 1
    assert result["abstain_count"] == 1
    assert result["error_count"] == 1
    assert result["invalid_count"] == 1
    assert result["missing_count"] == 1
    assert result["low_count"] == 0
    assert result["low_error_rate"] is None
    assert result["enrichment_ratio"] is None


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
        "mean_input_tokens": None,
        "mean_output_tokens": None,
        "mean_usd": 0.0,
        "mean_compute_proxy_usd": 0.0,
        "known_estimated_usd_sum": 0,
        "known_cost_n": 0,
        "unknown_usage_n": 0,
        "hitl_rate": 0.0,
        "mean_validation_attempts": 0.0,
    }

    with pytest.raises(EvaluationDataError, match="0 이상"):
        cost_latency(
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
