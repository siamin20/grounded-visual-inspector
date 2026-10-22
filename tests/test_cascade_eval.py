from __future__ import annotations

from copy import deepcopy

import pytest

from tools import cascade_eval


def _expected(sample_id: str, *, is_defect: bool = False) -> dict:
    return {
        "id": sample_id,
        "item": "screw",
        "label": "scratch_head" if is_defect else "good",
        "is_defect": is_defect,
        "split": "dev",
    }


def _row(
    sample_id: str,
    condition: str,
    verdict: str,
    *,
    is_defect: bool = False,
    error: str | None = None,
) -> dict:
    return {
        **_expected(sample_id, is_defect=is_defect),
        "condition": condition,
        "verdict": verdict,
        "error": error,
    }


@pytest.mark.parametrize(
    ("b0", "after", "expected"),
    [
        ("REJECT", "PASS", ("REJECT", "B0")),
        ("REJECT", "REJECT", ("REJECT", "B0")),
        ("REJECT", "ABSTAIN", ("REJECT", "B0")),
        ("PASS", "PASS", ("PASS", "A")),
        ("ABSTAIN", "PASS", ("PASS", "A")),
        ("PASS", "REJECT", ("ABSTAIN", None)),
        ("PASS", "ABSTAIN", ("ABSTAIN", None)),
        ("ABSTAIN", "REJECT", ("ABSTAIN", None)),
        ("ABSTAIN", "ABSTAIN", ("ABSTAIN", None)),
    ],
)
def test_cascade_policy_decision_table(
    b0: str,
    after: str,
    expected: tuple[str, str | None],
) -> None:
    assert cascade_eval.cascade_verdict(
        {"verdict": b0, "error": None},
        {"verdict": after, "error": None},
    ) == expected


def test_error_or_invalid_verdict_is_not_usable() -> None:
    assert cascade_eval.cascade_verdict(
        {"verdict": "REJECT", "error": "timeout"},
        {"verdict": "PASS", "error": None},
    ) == ("ABSTAIN", None)
    assert cascade_eval.cascade_verdict(
        {"verdict": "INVALID", "error": None},
        {"verdict": "REJECT", "error": None},
    ) == ("ABSTAIN", None)


def test_upstream_error_is_preserved_in_derived_row() -> None:
    expected = [_expected("one")]
    b0 = [_row("one", "B0", "REJECT", error="timeout")]
    after = [_row("one", "A", "PASS")]
    derived, audit = cascade_eval.build_cascade_rows(b0, after, expected)
    assert derived[0]["verdict"] == "ABSTAIN"
    assert derived[0]["error"] == "upstream_b0_unusable"
    assert audit[0]["cascade_error"] == "upstream_b0_unusable"


def test_policy_output_is_invariant_to_ground_truth_fields() -> None:
    b0 = _row("sample", "B0", "ABSTAIN")
    after = _row("sample", "A", "PASS")
    before = cascade_eval.cascade_verdict(b0, after)
    b0["label"] = "scratch_head"
    b0["is_defect"] = True
    after["label"] = "scratch_head"
    after["is_defect"] = True
    assert cascade_eval.cascade_verdict(b0, after) == before


def test_build_rows_requires_exact_paired_roster() -> None:
    expected = [_expected("one"), _expected("two", is_defect=True)]
    b0 = [_row("one", "B0", "REJECT")]
    after = [
        _row("one", "A", "PASS"),
        _row("two", "A", "ABSTAIN", is_defect=True),
    ]
    with pytest.raises(cascade_eval.CascadeDataError, match="roster"):
        cascade_eval.build_cascade_rows(b0, after, expected)


def test_build_rows_rejects_duplicate_and_metadata_mismatch() -> None:
    expected = [_expected("one")]
    duplicate = [
        _row("one", "B0", "REJECT"),
        _row("one", "B0", "REJECT"),
    ]
    with pytest.raises(cascade_eval.CascadeDataError, match="중복 id"):
        cascade_eval.build_cascade_rows(
            duplicate,
            [_row("one", "A", "PASS")],
            expected,
        )

    wrong = _row("one", "A", "PASS")
    wrong["item"] = "bottle"
    with pytest.raises(cascade_eval.CascadeDataError, match="item"):
        cascade_eval.build_cascade_rows(
            [_row("one", "B0", "REJECT")],
            [wrong],
            expected,
        )


def test_audit_rows_omit_ground_truth_and_paths() -> None:
    expected = [_expected("one")]
    _, audit = cascade_eval.build_cascade_rows(
        [_row("one", "B0", "ABSTAIN")],
        [_row("one", "A", "PASS")],
        expected,
    )
    assert audit == [{
        "id": "one",
        "b0_verdict": "ABSTAIN",
        "a_verdict": "PASS",
        "cascade_verdict": "PASS",
        "selected_source": "A",
        "cascade_error": None,
    }]
    assert all(
        key not in audit[0]
        for key in ("label", "is_defect", "path", "image_path")
    )


@pytest.mark.parametrize(
    "change",
    [
        {"status": "incomplete"},
        {"split": "heldout"},
        {"dry_run": True},
    ],
)
def test_manifest_contract_is_complete_real_dev_only(change: dict) -> None:
    manifest = {
        "status": "complete",
        "split": "dev",
        "dry_run": False,
        "planned_conditions": ["B0", "A"],
        "conditions": {
            "B0": {"status": "complete"},
            "A": {"status": "complete"},
        },
    }
    manifest.update(change)
    with pytest.raises(cascade_eval.CascadeDataError):
        cascade_eval._require_manifest_contract(manifest)


def test_output_flags_and_warning_are_not_configurable(tmp_path) -> None:
    artifact = {
        "run_id": "test-run",
        "analysis_type": cascade_eval.ANALYSIS_TYPE,
        "policy_id": cascade_eval.POLICY_ID,
        "post_hoc_dev": True,
        "primary_hypothesis": False,
        "heldout_validated": False,
        "claim_status": cascade_eval.CLAIM_STATUS,
        "warning": cascade_eval.WARNING,
        "safety": {"defect_b0_abstain_to_a_plus_pass_n": 1},
        "metrics": {
            key: {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "tn": 0,
                "normal_n": 1,
                "tp": 0,
                "defect_n": 1,
                "abstain_n": 2,
                "coverage": 0.0,
            }
            for key in ("B0", "A", "A+")
        },
        "comparison": {
            "accuracy_delta_vs_b0": 0.0,
            "balanced_accuracy_delta_vs_b0": 0.0,
            "mcnemar_vs_b0": {"overall": {"p_value": 1.0}},
        },
    }
    json_path, markdown_path = cascade_eval.write_outputs(artifact, tmp_path)
    assert json_path.exists()
    markdown = markdown_path.read_text(encoding="utf-8")
    assert cascade_eval.MARKDOWN_WARNING in markdown
    assert cascade_eval.WARNING in markdown
    assert "실제 결함 1건" in markdown
    assert "지연·계산 자원은 아직 미측정" in markdown
    payload = __import__("json").loads(json_path.read_text(encoding="utf-8"))
    assert payload["post_hoc_dev"] is True
    assert payload["primary_hypothesis"] is False
    assert payload["heldout_validated"] is False

    with pytest.raises(cascade_eval.CascadeDataError, match="덮어쓸"):
        cascade_eval.write_outputs(deepcopy(artifact), tmp_path)


def test_small_golden_metrics_match_fixed_policy() -> None:
    expected = [
        _expected("normal"),
        _expected("defect", is_defect=True),
    ]
    b0 = [
        _row("normal", "B0", "ABSTAIN"),
        _row("defect", "B0", "REJECT", is_defect=True),
    ]
    after = [
        _row("normal", "A", "PASS"),
        _row("defect", "A", "PASS", is_defect=True),
    ]
    derived, _ = cascade_eval.build_cascade_rows(b0, after, expected)
    metrics = cascade_eval.binary_metrics(derived, expected)
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert cascade_eval.mcnemar(b0, derived, expected)["overall"]["p_value"] == 1.0
    risky_audit = [{
        "id": "defect",
        "b0_verdict": "ABSTAIN",
        "cascade_verdict": "PASS",
    }]
    assert cascade_eval.summarize_safety(risky_audit, expected) == {
        "defect_b0_abstain_to_a_plus_pass_n": 1,
    }
