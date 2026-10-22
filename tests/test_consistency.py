from __future__ import annotations

from argparse import Namespace
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest

from eval import run_eval
from eval.consistency import (
    ConsistencyError,
    _validate_run_contracts,
    analyze_repetition_rows,
    build_execution_plan,
    build_selection,
    load_selection,
)
from eval.provenance import ProvenanceError, _validate_consistency_settings


ROOT = Path(__file__).resolve().parents[1]
SELECTION_PATH = ROOT / "eval" / "consistency_dev_ids.json"


def _consistency_args(**overrides) -> Namespace:
    values = {
        "condition": "A",
        "split": "dev",
        "items": None,
        "limit": None,
        "out": str(ROOT / "eval" / "results"),
        "dry_run": True,
        "push_dataset": False,
        "final_run": False,
        "freeze_manifest": None,
        "run_id": None,
        "suite_conditions": "A",
        "sample_ids_file": str(SELECTION_PATH),
        "consistency_repeat": 1,
    }
    values.update(overrides)
    return Namespace(**values)


def _manifest(repeat: int, selection_record: dict) -> dict:
    return {
        "run_id": f"r6-repeat-{repeat}",
        "split": "dev",
        "planned_conditions": ["A"],
        "expected_n": 20,
        "settings": {
            "temperature": 0.0,
            "items": None,
            "limit": None,
            "final_run": False,
            "freeze_manifest": None,
            "sample_ids_file": deepcopy(selection_record),
            "consistency_repeat": repeat,
        },
        "backends": {"llm": "openai", "vlm": "openai", "embed": "openai"},
        "models": {"llm": "model-a", "vlm": "model-v", "embed": "model-e"},
        "pricing": {"version": "fixed"},
        "repository": {"commit": "a" * 40},
        "artifacts": {"dataset": "same"},
        "runtime": {"python": "3.11"},
        "immutable_sha256": str(repeat) * 64,
        "manifest_sha256": chr(96 + repeat) * 64,
        "conditions": {
            "A": {
                "status": "complete",
                "observed_model_ids": ["openai:model-a", "openai:model-v"],
                "estimated_spend_usd": 1.0,
            }
        },
    }


def test_fixed_selection_matches_current_dataset_and_strata() -> None:
    expected = build_selection()
    selection = load_selection(SELECTION_PATH)

    assert selection.payload == expected
    assert len(selection.ids) == len(set(selection.ids)) == 20
    assert sum(expected["stratum_counts"].values()) == 20
    assert {sample_id.split("-", 1)[0] for sample_id in selection.ids} == {
        "bottle",
        "hazelnut",
        "metal_nut",
        "screw",
    }
    dataset = json.loads((ROOT / "eval" / "dataset.json").read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in dataset["samples"]}
    selected = [by_id[sample_id] for sample_id in selection.ids]
    assert all(row["split"] == "dev" for row in selected)
    assert Counter(
        f"{row['item']}/{'defect' if row['is_defect'] else 'normal'}"
        for row in selected
    ) == expected["stratum_counts"]


def test_selection_tampering_duplicate_or_unknown_id_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "eval").mkdir(parents=True)
    dataset = repo / "eval" / "dataset.json"
    dataset.write_bytes((ROOT / "eval" / "dataset.json").read_bytes())
    payload = build_selection(dataset)
    payload["ids"][1] = payload["ids"][0]
    selection_path = repo / "eval" / "selection.json"
    selection_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConsistencyError, match="계약과 다릅니다"):
        load_selection(selection_path, dataset_path=dataset, repo_root=repo)

    payload = build_selection(dataset)
    payload["ids"].pop()
    selection_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConsistencyError, match="계약과 다릅니다"):
        load_selection(selection_path, dataset_path=dataset, repo_root=repo)

    payload = build_selection(dataset)
    payload["ids"][0] = "heldout-or-unknown-id"
    selection_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConsistencyError, match="계약과 다릅니다"):
        load_selection(selection_path, dataset_path=dataset, repo_root=repo)


def test_execution_plan_is_dev_a_temperature_zero_three_times() -> None:
    selection = load_selection(SELECTION_PATH)
    plan = build_execution_plan(selection)

    assert plan["evidence_status"] == "PLAN_ONLY_NO_MODEL_CALLS"
    assert plan["split"] == "dev"
    assert plan["condition"] == "A"
    assert plan["temperature"] == 0.0
    assert [entry["repeat"] for entry in plan["repetitions"]] == [1, 2, 3]
    assert len({tuple(entry["argv"]) for entry in plan["repetitions"]}) == 3
    for repeat, entry in enumerate(plan["repetitions"], start=1):
        argv = entry["argv"]
        assert argv[argv.index("--split") + 1] == "dev"
        assert argv[argv.index("--condition") + 1] == "A"
        assert argv[argv.index("--consistency-repeat") + 1] == str(repeat)
        assert "--dry-run" not in argv


def test_conservative_rate_keeps_fixed_denominator_and_marks_fake() -> None:
    ids = [f"sample-{index:02d}" for index in range(20)]
    repetitions = [
        [{"id": sample_id, "verdict": "PASS", "error": None} for sample_id in ids]
        for _ in range(3)
    ]
    repetitions[1][0]["verdict"] = "REJECT"
    repetitions[0][1]["error"] = "provider error"
    repetitions[2] = [row for row in repetitions[2] if row["id"] != ids[2]]
    repetitions[1][3]["verdict"] = "MAYBE"

    report = analyze_repetition_rows(
        repetitions,
        selection={"ids": ids},
        dry_run=True,
    )

    assert report["evidence_status"] == "DRYRUN_NOT_PERFORMANCE"
    assert report["sample_n"] == 20
    assert report["verdict_variation_count"] == 4
    assert report["verdict_variation_rate"] == 0.2
    assert report["threshold_assessment"] == "NOT_APPLICABLE_DRYRUN"
    assert report["strict_unstable_count"] == 4
    assert report["strict_instability_rate"] == 0.2
    assert report["valid_triplet_count"] == 17
    assert report["observed_change_count"] == 1
    assert report["observed_change_rate"] == pytest.approx(1 / 17)
    assert report["missing_occurrences"] == 1
    assert report["error_occurrences"] == 1
    assert report["invalid_verdict_occurrences"] == 1


def test_postprocessor_rejects_outside_or_duplicate_result_ids() -> None:
    ids = [f"sample-{index:02d}" for index in range(20)]
    rows = [{"id": sample_id, "verdict": "PASS", "error": None} for sample_id in ids]

    outside = deepcopy(rows)
    outside[0]["id"] = "outside"
    with pytest.raises(ConsistencyError, match="선정 외 ID"):
        analyze_repetition_rows([outside, rows, rows], selection={"ids": ids}, dry_run=True)

    duplicate = deepcopy(rows)
    duplicate[1]["id"] = duplicate[0]["id"]
    with pytest.raises(ConsistencyError, match="중복 ID"):
        analyze_repetition_rows([duplicate, rows, rows], selection={"ids": ids}, dry_run=True)


def test_run_contract_rejects_repeat_setting_or_model_mixing() -> None:
    selection = load_selection(SELECTION_PATH)
    manifests = [_manifest(index, selection.manifest_record()) for index in (1, 2, 3)]

    records = _validate_run_contracts(manifests, selection=selection)
    assert [record["repeat"] for record in records] == [1, 2, 3]

    repeated = deepcopy(manifests)
    repeated[2]["settings"]["consistency_repeat"] = 2
    with pytest.raises(ConsistencyError, match="반복 번호"):
        _validate_run_contracts(repeated, selection=selection)

    mixed_model = deepcopy(manifests)
    mixed_model[2]["models"]["vlm"] = "other-model"
    with pytest.raises(ConsistencyError, match="provenance"):
        _validate_run_contracts(mixed_model, selection=selection)

    mixed_setting = deepcopy(manifests)
    mixed_setting[2]["settings"]["validation_similarity_threshold"] = 0.99
    with pytest.raises(ConsistencyError, match="실험 설정"):
        _validate_run_contracts(mixed_setting, selection=selection)

    mixed_ids = deepcopy(manifests)
    mixed_ids[2]["settings"]["sample_ids_file"]["ids"][0] = "other-id"
    with pytest.raises(ConsistencyError, match="선정 파일"):
        _validate_run_contracts(mixed_ids, selection=selection)


def test_incomplete_registered_run_is_kept_for_conservative_aggregation() -> None:
    selection = load_selection(SELECTION_PATH)
    manifests = [_manifest(index, selection.manifest_record()) for index in (1, 2, 3)]
    manifests[1]["conditions"]["A"]["status"] = "incomplete"

    records = _validate_run_contracts(manifests, selection=selection)

    assert records[1]["status"] == "incomplete"


def test_manifest_consistency_settings_are_fail_closed() -> None:
    selection = load_selection(SELECTION_PATH)
    manifest = {
        "settings": {
            "temperature": 0.0,
            "items": None,
            "limit": None,
            "final_run": False,
            "freeze_manifest": None,
            "sample_ids_file": selection.manifest_record(),
            "consistency_repeat": 1,
        },
        "split": "dev",
        "seed": 20261020,
        "expected_n": 20,
    }
    _validate_consistency_settings(manifest, planned_conditions=["A"])

    heldout = deepcopy(manifest)
    heldout["split"] = "heldout"
    with pytest.raises(ProvenanceError, match="dev/A 20건"):
        _validate_consistency_settings(heldout, planned_conditions=["A"])

    wrong_ids = deepcopy(manifest)
    wrong_ids["settings"]["sample_ids_file"]["ids"][0] = "wrong"
    with pytest.raises(ProvenanceError, match="digest"):
        _validate_consistency_settings(wrong_ids, planned_conditions=["A"])


def test_run_eval_consistency_guard_rejects_heldout_and_filter_mixing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_BACKEND", "fake")
    monkeypatch.setenv("VLM_BACKEND", "fake")
    monkeypatch.setenv("EMBED_BACKEND", "fake")

    with pytest.raises(SystemExit, match="dev/A"):
        run_eval._guard_run_request(_consistency_args(split="heldout"))
    with pytest.raises(SystemExit, match="--items"):
        run_eval._guard_run_request(_consistency_args(items="screw"))
    with pytest.raises(SystemExit, match=r"1\|2\|3"):
        run_eval._guard_run_request(_consistency_args(consistency_repeat=None))
    with pytest.raises(SystemExit, match="LangSmith Dataset"):
        run_eval._guard_run_request(_consistency_args(push_dataset=True))

    run_eval._guard_run_request(_consistency_args())


def test_run_eval_loads_exact_selection_order() -> None:
    selection = load_selection(SELECTION_PATH)

    rows = run_eval._load_samples(
        split="dev",
        items=None,
        limit=None,
        sample_ids=selection.ids,
    )

    assert [row["id"] for row in rows] == selection.ids
    assert len(rows) == 20


def test_parser_accepts_explicit_consistency_roster() -> None:
    args = run_eval._parser().parse_args(
        [
            "--condition",
            "A",
            "--split",
            "dev",
            "--sample-ids-file",
            "eval/consistency_dev_ids.json",
            "--consistency-repeat",
            "3",
        ]
    )

    assert args.sample_ids_file == "eval/consistency_dev_ids.json"
    assert args.consistency_repeat == 3
