from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import pytest

from eval.provenance import (
    LEGACY_RESULT_ROW_KEYS_V2,
    ProvenanceError,
    atomic_write_json,
    build_run_manifest,
    canonical_sha256,
    collect_repository_provenance,
    complete_condition,
    create_run_manifest,
    fail_condition,
    join_run_manifest,
    load_run_manifest,
    new_run_id,
    pause_condition,
    record_condition_progress,
    record_condition_setup_usage,
    resume_condition,
    sha256_file,
    sha256_tree,
    start_condition as _start_condition,
    validate_run_id,
    verify_run_manifest,
)


def _no_setup_usage() -> dict:
    return {
        "tokens": 0,
        "input_tokens": None,
        "output_tokens": None,
        "token_accounting": "total_only",
        "estimated_usd": 0.0,
        "compute_proxy_usd": 0.0,
        "successful_provider_requests": 0,
        "successful_vlm_requests": 0,
        "unknown_token_usage_successes": 0,
        "unobserved_model_successes": 0,
        "observed_model_response_count": 0,
        "observed_model_ids": [],
        "latency_s": 0.0,
        "usage_status": "not_applicable",
    }


def start_condition(run_dir, manifest, *, condition: str, **kwargs):
    result = _start_condition(run_dir, manifest, condition=condition, **kwargs)
    record_condition_setup_usage(
        run_dir,
        condition=condition,
        setup_usage=_no_setup_usage(),
    )
    return result


@pytest.fixture
def synthetic_repository(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "repository"
    (root / "eval").mkdir(parents=True)
    (root / "data" / "specs").mkdir(parents=True)
    (root / "data" / "guides").mkdir(parents=True)
    (root / "data" / "mvtec" / "screw" / "test" / "good").mkdir(parents=True)
    (root / "data" / "mvtec" / "screw" / "train" / "good").mkdir(parents=True)
    (root / "agent").mkdir(parents=True)
    (root / "eval" / "dataset.json").write_text(
        json.dumps({"samples": [
            {
                "id": "dev-1", "split": "dev", "item": "screw",
                "path": "data/mvtec/screw/test/good/001.png",
                "label": "good", "is_defect": False,
            },
            {
                "id": "dev-2", "split": "dev", "item": "screw",
                "path": "data/mvtec/screw/test/good/002.png",
                "label": "good", "is_defect": False,
            },
        ]}),
        encoding="utf-8",
    )
    (root / "eval" / "type_map.json").write_text(
        json.dumps({"screw": {"good": ["good"]}}),
        encoding="utf-8",
    )
    (root / "data" / "specs" / "screw.md").write_text("# 나사 기준\n", encoding="utf-8")
    (root / "data" / "guides" / "scratch.md").write_text("# 조치 가이드\n", encoding="utf-8")
    (root / "data" / "lots.json").write_text("{}\n", encoding="utf-8")
    first_image = root / "data" / "mvtec" / "screw" / "test" / "good" / "001.png"
    second_image = root / "data" / "mvtec" / "screw" / "test" / "good" / "002.png"
    reference_image = root / "data" / "mvtec" / "screw" / "train" / "good" / "000.png"
    first_image.write_bytes(b"first-image")
    second_image.write_bytes(b"second-image")
    reference_image.write_bytes(b"reference-image")
    (root / "agent" / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
    provenance = collect_repository_provenance(
        root,
        source_paths=[root / "agent" / "core.py"],
        package_names=(),
        git_state={
            "commit": "a" * 40,
            "dirty": False,
            "diff_sha256": "b" * 64,
        },
        evaluation_images=[
            {"id": "dev-1", "path": first_image},
            {"id": "dev-2", "path": second_image},
        ],
        reference_images=[{"item": "screw", "path": reference_image}],
    )
    return root, provenance


def _manifest(
    provenance: dict,
    *,
    run_id: str = "20261021T120000000000Z-abcd1234",
    planned_conditions: tuple[str, ...] = ("B0",),
    dry_run: bool = False,
    backends: dict[str, str] | None = None,
    models: dict[str, str] | None = None,
    settings: dict | None = None,
    budget: object = 15.0,
) -> dict:
    return build_run_manifest(
        run_id=run_id,
        split="dev",
        planned_conditions=planned_conditions,
        seed=20261020,
        expected_n=2,
        dry_run=dry_run,
        suite_budget_estimated_usd=budget,
        provenance=provenance,
        backends=backends
        or {
            "llm": "openai",
            "vlm": "openai",
            "embed": "openai",
        },
        models=models
        or {
            "llm_fast": "gpt-fast",
            "llm_quality": "gpt-quality",
            "vlm_observer": "gpt-vision",
            "embed": "embedding-model",
        },
        settings=settings or {"temperature": 0},
        pricing={"mode": "estimated", "version": "2026-10-21"},
        created_at="2026-10-21T12:00:00Z",
    )


def _legacy_v2_manifest(manifest: dict) -> dict:
    """기존 schema v2 서명을 보존한 회귀 테스트용 manifest를 만든다."""
    legacy = json.loads(json.dumps(manifest))
    legacy["schema_version"] = 2
    immutable_keys = (
        "schema_version",
        "run_id",
        "split",
        "dry_run",
        "planned_conditions",
        "seed",
        "expected_n",
        "suite_budget_estimated_usd",
        "backends",
        "models",
        "settings",
        "pricing",
        "repository",
        "artifacts",
        "runtime",
    )
    legacy["immutable_sha256"] = canonical_sha256(
        {key: legacy[key] for key in immutable_keys}
    )
    unsigned = dict(legacy)
    unsigned.pop("manifest_sha256", None)
    legacy["manifest_sha256"] = canonical_sha256(unsigned)
    return legacy


def _create_run(
    tmp_path: Path,
    provenance: dict,
    **manifest_overrides,
) -> tuple[Path, dict]:
    results = tmp_path / "results"
    results.mkdir()
    manifest = _manifest(provenance, **manifest_overrides)
    run_dir = results / manifest["run_id"]
    create_run_manifest(run_dir, manifest)
    return run_dir, manifest


def _write_condition_outputs(
    run_dir: Path,
    manifest: dict,
    condition: str,
    *,
    result_name: str | None = None,
    complete_capture: bool = True,
) -> tuple[Path, Path]:
    current = load_run_manifest(run_dir)
    record = current["conditions"][condition]
    processed = record["processed_samples"]
    condition_spend = float(record["estimated_spend_usd"])
    result_path = run_dir / (result_name or f"{condition}_dev.jsonl")
    rows = []
    fake = manifest["backends"]["llm"] == "fake" and manifest["backends"]["vlm"] == "fake"
    after = condition not in {"B0", "B1"}
    for index in range(processed):
        rows.append({
            "run_id": manifest["run_id"],
            "provenance_sha256": manifest["immutable_sha256"],
            "dry_run": manifest["dry_run"],
            "id": f"dev-{index + 1}",
            "item": "screw",
            "label": "good",
            "is_defect": False,
            "split": manifest["split"],
            "condition": condition,
            "verdict": "PASS",
            "defect_type": None,
            "confidence": "high",
            "abstain_reason": None,
            "approval_required": False,
            "validation_attempts": 1,
            "validation_rejections": 0,
            "retry_count": 0,
            "replan_rounds": 0,
            "successful_provider_requests": 0 if fake else 1,
            "successful_vlm_requests": 0 if fake else 1,
            "unknown_token_usage_successes": 0,
            "unobserved_model_successes": 0,
            "observed_model_response_count": 0 if fake else 1,
            "validation_errors_first": [],
            "would_fail": [],
            "observer_status": "O-" if after else None,
            "cross_status": "C-" if after else None,
            "observations": [{
                "call_id": "obs-full",
                "region": "full",
                "text": "이상 없음",
                "anomaly_seen": False,
                "regions": [],
                "features": ["균일한 윤곽"],
                "model": "openai:response-model-1",
            }] if after else [],
            "cross_check": {
                "call_id": "cc-full",
                "summary": "이상 없음",
                "anomaly_seen": False,
                "regions": [],
                "features": [],
                "checks": {
                    "scratch": False,
                    "break_or_crack": False,
                    "contamination": False,
                    "discoloration": False,
                    "bend_or_deformation": False,
                    "thread_damage": False,
                },
                "model": "openai:response-model-1",
            } if after else {},
            "observations_text": "이상 없음",
            "rag_query_text": "이상 없음" if after else "",
            "spec_hits": [],
            "judgement": {
                "violations": [],
                "defect_type": None,
                "rationale": "위반 없음",
                "status": "none",
            } if after else {},
            "judgement_corrected": False,
            "raw_judgement": {
                "violations": [],
                "defect_type": None,
                "rationale": "위반 없음",
            } if after else {},
            "pre_validation_snapshot": {},
            "report": {"verdict": "PASS"},
            "latency_s": 0.1,
            "tokens": 1,
            "input_tokens": 1,
            "output_tokens": 0,
            "token_accounting": "estimated_split" if fake else "provider_split",
            "estimated_usd": condition_spend if index == 0 else 0.0,
            "billed_usd": None,
            "compute_proxy_usd": 0.0,
            "cost_basis": "fake_no_billing" if fake else "provider_token_estimate_v1",
            "observed_model_ids": ["openai:response-model-1"],
            "langsmith_run_id": None,
            "usage_status": "fake_estimate_no_billing" if fake else "known_split",
            "error": None,
        })
    if manifest["schema_version"] == 2:
        rows = [
            {key: row[key] for key in LEGACY_RESULT_ROW_KEYS_V2}
            for row in rows
        ]
    result_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary_path = run_dir / f"summary_{condition}_dev.json"
    atomic_write_json(
        summary_path,
        {
            "run_id": manifest["run_id"],
            "condition": condition,
            "split": manifest["split"],
            "dry_run": manifest["dry_run"],
            "provenance_sha256": manifest["immutable_sha256"],
            "processed_samples": processed,
            "condition_estimated_usd": condition_spend,
            "suite_estimated_usd": current["suite_estimated_spend_usd"],
            "complete_capture": complete_capture,
            "setup_usage": current["conditions"][condition]["setup_usage"],
            "usage_unknown_count": 0,
        },
    )
    return result_path, summary_path


def _finish_condition(
    run_dir: Path,
    manifest: dict,
    condition: str,
    *,
    result_name: str | None = None,
    complete_capture: bool = True,
) -> tuple[Path, Path, dict]:
    current = load_run_manifest(run_dir)
    if current["conditions"][condition]["processed_samples"] == 0:
        record_condition_progress(
            run_dir,
            condition=condition,
            processed_samples=current["expected_n"],
            estimated_spend_usd=0.0,
        )
    result_path, summary_path = _write_condition_outputs(
        run_dir,
        manifest,
        condition,
        result_name=result_name,
        complete_capture=complete_capture,
    )
    completed = complete_condition(
        run_dir,
        condition=condition,
        result_path=result_path,
        summary_path=summary_path,
        complete_capture=complete_capture,
        observed_model_ids=["openai:response-model-1"],
        ended_at="2026-10-21T12:10:00Z",
    )
    return result_path, summary_path, completed


def _rebind_output_hash(
    run_dir: Path,
    *,
    condition: str,
    key: str,
    path: Path,
) -> None:
    """산출물과 manifest hash를 함께 바꾼 내부 불일치 위조를 만든다."""
    payload = load_run_manifest(run_dir)
    payload["conditions"][condition][key]["sha256"] = sha256_file(path)
    _write_resigned_manifest(run_dir, payload)


def _write_resigned_manifest(run_dir: Path, payload: dict) -> None:
    """test용 변조가 단순 manifest hash 오류에서 멈추지 않게 한다."""
    manifest_path = run_dir / "manifest.json"
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    payload["manifest_sha256"] = canonical_sha256(unsigned)
    atomic_write_json(manifest_path, payload)


def test_run_id_is_unique_and_path_safe() -> None:
    fixed = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)
    run_id = new_run_id(now=fixed, random_suffix="abcd1234")

    assert run_id == "20261021T120000000000Z-abcd1234"
    assert validate_run_id(run_id) == run_id
    for invalid in ("", "../escape", "/absolute", "two words", ".", "..", True):
        with pytest.raises(ProvenanceError):
            validate_run_id(invalid)


def test_file_and_tree_hashes_change_with_content(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    first = tree / "a.txt"
    second = tree / "b.txt"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")

    first_file_hash = sha256_file(first)
    first_tree_hash = sha256_tree(tree)
    first.write_text("changed", encoding="utf-8")

    assert sha256_file(first) != first_file_hash
    assert sha256_tree(tree) != first_tree_hash


def test_repository_provenance_uses_only_relative_paths(
    synthetic_repository: tuple[Path, dict],
) -> None:
    root, provenance = synthetic_repository
    artifacts = provenance["artifacts"]
    records = [artifacts["dataset"], artifacts["type_map"], artifacts["lots"]]
    records.extend(artifacts["specs"])
    records.extend(artifacts["guides"])
    records.extend(artifacts["source"])
    records.extend(artifacts["evaluation_images"])
    records.extend(artifacts["reference_images"])

    assert all(not Path(record["path"]).is_absolute() for record in records)
    assert str(root) not in json.dumps(provenance)
    assert len(artifacts["source_tree_sha256"]) == 64


def test_suite_lifecycle_progress_and_duplicate_condition_are_fail_closed(
    tmp_path: Path,
    synthetic_repository: tuple[Path, dict],
) -> None:
    root, provenance = synthetic_repository
    run_dir, manifest = _create_run(
        tmp_path,
        provenance,
        planned_conditions=("B0", "A"),
    )

    start_condition(
        run_dir,
        manifest,
        condition="B0",
        started_at="2026-10-21T12:01:00Z",
    )
    with pytest.raises(ProvenanceError, match="실행 중"):
        start_condition(run_dir, manifest, condition="A")
    progress = record_condition_progress(
        run_dir,
        condition="B0",
        processed_samples=2,
        estimated_spend_usd=0.2,
        updated_at="2026-10-21T12:02:00Z",
    )
    assert progress["suite_estimated_spend_usd"] == pytest.approx(0.2)
    with pytest.raises(ProvenanceError, match="processed_samples"):
        record_condition_progress(
            run_dir,
            condition="B0",
            processed_samples=1,
            estimated_spend_usd=0.2,
        )
    with pytest.raises(ProvenanceError, match="estimated_spend_usd"):
        record_condition_progress(
            run_dir,
            condition="B0",
            processed_samples=2,
            estimated_spend_usd=0.1,
        )
    _, _, first_done = _finish_condition(run_dir, manifest, "B0")
    assert first_done["status"] == "incomplete"

    start_condition(
        run_dir,
        manifest,
        condition="A",
        started_at="2026-10-21T12:11:00Z",
    )
    record_condition_progress(
        run_dir,
        condition="A",
        processed_samples=2,
        estimated_spend_usd=0.3,
        updated_at="2026-10-21T12:12:00Z",
    )
    _, _, all_done = _finish_condition(run_dir, manifest, "A")

    assert all_done["status"] == "complete"
    assert all_done["suite_estimated_spend_usd"] == pytest.approx(0.5)
    verified = verify_run_manifest(
        run_dir,
        repo_root=root,
        verify_inputs=True,
    )
    assert verified["status"] == "complete"
    with pytest.raises(ProvenanceError, match="이미"):
        start_condition(run_dir, manifest, condition="B0")


def test_paused_checkpoint_requires_exact_roster_prefix(
    tmp_path: Path,
    synthetic_repository: tuple[Path, dict],
) -> None:
    root, provenance = synthetic_repository
    run_dir, manifest = _create_run(
        tmp_path,
        provenance,
        planned_conditions=("B0", "A"),
    )
    start_condition(
        run_dir,
        manifest,
        condition="B0",
        started_at="2026-10-21T12:01:00Z",
    )
    record_condition_progress(
        run_dir,
        condition="B0",
        processed_samples=1,
        estimated_spend_usd=0.0,
    )
    result_path, summary_path = _write_condition_outputs(
        run_dir,
        manifest,
        "B0",
        complete_capture=False,
    )
    summary_path.unlink()
    paused = pause_condition(
        run_dir,
        condition="B0",
        result_path=result_path,
        observed_model_ids=["openai:response-model-1"],
    )

    assert paused["status"] == "paused"
    assert paused["conditions"]["B0"]["pause_reason"] == "batch_limit"
    assert paused["conditions"]["B0"]["checkpoint_count"] == 1
    with pytest.raises(ProvenanceError, match="선행 조건"):
        start_condition(run_dir, manifest, condition="A")

    wrong_reason = json.loads(json.dumps(paused))
    wrong_reason["conditions"]["B0"]["pause_reason"] = "interrupt"
    _write_resigned_manifest(run_dir, wrong_reason)
    with pytest.raises(ProvenanceError, match="pause_reason"):
        load_run_manifest(run_dir)

    missing_count = json.loads(json.dumps(paused))
    missing_count["conditions"]["B0"].pop("checkpoint_count")
    _write_resigned_manifest(run_dir, missing_count)
    with pytest.raises(ProvenanceError, match="checkpoint_count"):
        load_run_manifest(run_dir)

    _write_resigned_manifest(run_dir, paused)
    verified = verify_run_manifest(
        run_dir,
        require_complete=False,
        repo_root=root,
        verify_inputs=True,
    )
    assert verified["conditions"]["B0"]["status"] == "paused"
    with pytest.raises(ProvenanceError, match="완료"):
        verify_run_manifest(run_dir, require_complete=True)

    row = json.loads(result_path.read_text(encoding="utf-8"))
    row["id"] = "dev-2"
    result_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    _rebind_output_hash(
        run_dir,
        condition="B0",
        key="checkpoint_result",
        path=result_path,
    )
    with pytest.raises(ProvenanceError, match="prefix"):
        verify_run_manifest(
            run_dir,
            require_complete=False,
            repo_root=root,
            verify_inputs=True,
        )


def test_resume_condition_rejects_changed_immutable_manifest(
    tmp_path: Path,
    synthetic_repository: tuple[Path, dict],
) -> None:
    _, provenance = synthetic_repository
    run_dir, manifest = _create_run(tmp_path, provenance)
    start_condition(run_dir, manifest, condition="B0")
    record_condition_progress(
        run_dir,
        condition="B0",
        processed_samples=1,
        estimated_spend_usd=0.0,
    )
    result_path, summary_path = _write_condition_outputs(
        run_dir,
        manifest,
        "B0",
        complete_capture=False,
    )
    summary_path.unlink()
    pause_condition(
        run_dir,
        condition="B0",
        result_path=result_path,
        observed_model_ids=["openai:response-model-1"],
    )
    changed = _manifest(
        provenance,
        budget=14.0,
    )

    with pytest.raises(ProvenanceError, match="불변 설정"):
        resume_condition(run_dir, changed, condition="B0")


@pytest.mark.parametrize("changed_field", ["judge_model", "judge_num_predict"])
def test_resume_rejects_changed_ollama_judge_contract(
    changed_field: str,
    tmp_path: Path,
    synthetic_repository: tuple[Path, dict],
) -> None:
    _, provenance = synthetic_repository
    models = {
        "llm_fast": "qwen3:0.6b",
        "llm_judge": "qwen3:8b",
        "llm_quality": "qwen3:8b",
        "vlm_observer": "qwen2.5vl:3b",
        "embed": "nomic-embed-text",
    }
    settings = {
        "temperature": 0,
        "ollama_runtime": {"llm_judge_num_predict": "512"},
    }
    run_dir, manifest = _create_run(
        tmp_path,
        provenance,
        models=models,
        settings=settings,
    )
    start_condition(run_dir, manifest, condition="B0")
    record_condition_progress(
        run_dir,
        condition="B0",
        processed_samples=1,
        estimated_spend_usd=0.0,
    )
    result_path, summary_path = _write_condition_outputs(
        run_dir,
        manifest,
        "B0",
        complete_capture=False,
    )
    summary_path.unlink()
    pause_condition(
        run_dir,
        condition="B0",
        result_path=result_path,
        observed_model_ids=["openai:response-model-1"],
    )
    changed_models = dict(models)
    changed_settings = json.loads(json.dumps(settings))
    if changed_field == "judge_model":
        changed_models["llm_judge"] = "qwen3:14b"
    else:
        changed_settings["ollama_runtime"]["llm_judge_num_predict"] = "1024"
    changed = _manifest(
        provenance,
        models=changed_models,
        settings=changed_settings,
    )

    with pytest.raises(ProvenanceError, match="불변 설정"):
        resume_condition(run_dir, changed, condition="B0")


def test_suite_conditions_must_start_in_preregistered_order(
    tmp_path: Path,
    synthetic_repository: tuple[Path, dict],
) -> None:
    _, provenance = synthetic_repository
    run_dir, manifest = _create_run(
        tmp_path,
        provenance,
        planned_conditions=("B0", "B1", "A"),
    )

    with pytest.raises(ProvenanceError, match="순서"):
        start_condition(run_dir, manifest, condition="A")
    with pytest.raises(ProvenanceError, match="순서"):
        start_condition(run_dir, manifest, condition="B1")

    start_condition(run_dir, manifest, condition="B0")
    _finish_condition(run_dir, manifest, "B0")
    with pytest.raises(ProvenanceError, match="순서"):
        start_condition(run_dir, manifest, condition="A")

    start_condition(run_dir, manifest, condition="B1")
    _finish_condition(run_dir, manifest, "B1")
    start_condition(run_dir, manifest, condition="A")


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), "15", True])
def test_suite_budget_rejects_invalid_values(
    value: object,
    synthetic_repository: tuple[Path, dict],
) -> None:
    _, provenance = synthetic_repository
    with pytest.raises(ProvenanceError, match="suite_budget_estimated_usd"):
        _manifest(provenance, budget=value)


@pytest.mark.parametrize(
    ("processed", "spend"),
    [(-1, 0.0), (True, 0.0), (0, -1.0), (0, float("nan")), (0, float("inf")), (0, "1")],
)
def test_progress_rejects_invalid_values(
    processed: object,
    spend: object,
    tmp_path: Path,
    synthetic_repository: tuple[Path, dict],
) -> None:
    _, provenance = synthetic_repository
    run_dir, manifest = _create_run(tmp_path, provenance)
    start_condition(run_dir, manifest, condition="B0")

    with pytest.raises(ProvenanceError):
        record_condition_progress(
            run_dir,
            condition="B0",
            processed_samples=processed,
            estimated_spend_usd=spend,
        )


def test_dry_run_is_rejected_from_manifest_even_after_result_rename(
    tmp_path: Path,
    synthetic_repository: tuple[Path, dict],
) -> None:
    _, provenance = synthetic_repository
    run_dir, manifest = _create_run(
        tmp_path,
        provenance,
        dry_run=True,
        backends={"llm": "fake", "vlm": "fake", "embed": "fake"},
    )
    start_condition(run_dir, manifest, condition="B0")
    _finish_condition(run_dir, manifest, "B0", result_name="looks-real.jsonl")

    with pytest.raises(ProvenanceError, match="dry-run"):
        verify_run_manifest(run_dir)
    verified = verify_run_manifest(
        run_dir,
        reject_dry_run=False,
        reject_fake=False,
    )
    assert verified["dry_run"] is True


def test_result_manifest_and_unregistered_file_tampering_are_rejected(
    tmp_path: Path,
    synthetic_repository: tuple[Path, dict],
) -> None:
    _, provenance = synthetic_repository
    run_dir, manifest = _create_run(tmp_path, provenance)
    start_condition(run_dir, manifest, condition="B0")
    result_path, _, _ = _finish_condition(run_dir, manifest, "B0")

    original = result_path.read_text(encoding="utf-8")
    result_path.write_text(original + "{}\n", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="변조"):
        verify_run_manifest(run_dir)
    result_path.write_text(original, encoding="utf-8")
    (run_dir / "foreign.jsonl").write_text(original, encoding="utf-8")
    with pytest.raises(ProvenanceError, match="등록"):
        verify_run_manifest(run_dir)

    (run_dir / "foreign.jsonl").unlink()
    manifest_path = run_dir / "manifest.json"
    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["suite_estimated_spend_usd"] = 9.0
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ProvenanceError, match="변조"):
        verify_run_manifest(run_dir)


def test_legacy_v2_rows_remain_verifiable_but_reject_v3_fields(
    tmp_path: Path,
    synthetic_repository: tuple[Path, dict],
) -> None:
    """v2는 기존 42개 필드만 엄격히 허용하고 v3와 혼용하지 않는다."""
    _, provenance = synthetic_repository
    manifest = _legacy_v2_manifest(_manifest(provenance))
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    run_dir = results_dir / manifest["run_id"]
    create_run_manifest(run_dir, manifest)
    start_condition(run_dir, manifest, condition="B0")
    result_path, _, _ = _finish_condition(run_dir, manifest, "B0")

    # 실제 초기 v2 manifest에는 배치 checkpoint 필드가 없었다.
    completed = load_run_manifest(run_dir)
    completed["conditions"]["B0"].pop("checkpoint_count")
    _write_resigned_manifest(run_dir, completed)

    verified = verify_run_manifest(run_dir)
    assert verified["schema_version"] == 2
    rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    assert tuple(rows[0]) == LEGACY_RESULT_ROW_KEYS_V2

    rows[0]["observer_status"] = "O-"
    result_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    _rebind_output_hash(
        run_dir,
        condition="B0",
        key="result",
        path=result_path,
    )

    with pytest.raises(ProvenanceError, match="행 스키마"):
        verify_run_manifest(run_dir)


def test_v3_manifest_rejects_legacy_v2_row_schema(
    tmp_path: Path,
    synthetic_repository: tuple[Path, dict],
) -> None:
    _, provenance = synthetic_repository
    run_dir, manifest = _create_run(tmp_path, provenance)
    assert manifest["schema_version"] == 3
    start_condition(run_dir, manifest, condition="B0")
    result_path, _, _ = _finish_condition(run_dir, manifest, "B0")
    rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    legacy_rows = [
        {key: row[key] for key in LEGACY_RESULT_ROW_KEYS_V2}
        for row in rows
    ]
    result_path.write_text(
        "".join(json.dumps(row) + "\n" for row in legacy_rows),
        encoding="utf-8",
    )
    _rebind_output_hash(
        run_dir,
        condition="B0",
        key="result",
        path=result_path,
    )

    with pytest.raises(ProvenanceError, match="행 스키마"):
        verify_run_manifest(run_dir)


def test_rebound_result_rejects_invalid_diagnostic_fields(
    tmp_path: Path,
    synthetic_repository: tuple[Path, dict],
) -> None:
    _, provenance = synthetic_repository
    run_dir, manifest = _create_run(tmp_path, provenance)
    start_condition(run_dir, manifest, condition="B0")
    result_path, _, _ = _finish_condition(run_dir, manifest, "B0")
    rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    rows[0]["judgement_corrected"] = "false"
    result_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    _rebind_output_hash(
        run_dir,
        condition="B0",
        key="result",
        path=result_path,
    )

    with pytest.raises(ProvenanceError, match="judgement_corrected"):
        verify_run_manifest(run_dir)


@pytest.mark.parametrize(
    "case",
    [
        "observations_text_type",
        "observation_missing_field",
        "cross_missing_field",
        "cross_checks_conflict",
        "cross_unknown_with_valid_checks",
        "judgement_status",
        "missing_after_status",
    ],
)
def test_successful_after_row_requires_replayable_diagnostics(
    case: str,
    tmp_path: Path,
    synthetic_repository: tuple[Path, dict],
) -> None:
    _, provenance = synthetic_repository
    run_dir, manifest = _create_run(
        tmp_path,
        provenance,
        planned_conditions=("A",),
    )
    start_condition(run_dir, manifest, condition="A")
    result_path, _, _ = _finish_condition(run_dir, manifest, "A")
    rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    if case == "observations_text_type":
        rows[0]["observations_text"] = []
    elif case == "observation_missing_field":
        rows[0]["observations"][0].pop("features")
    elif case == "cross_missing_field":
        rows[0]["cross_check"].pop("summary")
    elif case == "cross_checks_conflict":
        rows[0]["cross_check"]["checks"]["scratch"] = True
    elif case == "cross_unknown_with_valid_checks":
        rows[0]["cross_status"] = "C?"
    elif case == "judgement_status":
        rows[0]["judgement"]["status"] = "unknown"
    else:
        rows[0]["observer_status"] = None
    result_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    _rebind_output_hash(
        run_dir,
        condition="A",
        key="result",
        path=result_path,
    )

    with pytest.raises(ProvenanceError):
        verify_run_manifest(run_dir)


def test_synthesized_cross_summary_keeps_checks_but_remains_unknown(
    tmp_path: Path,
    synthetic_repository: tuple[Path, dict],
) -> None:
    _, provenance = synthetic_repository
    run_dir, manifest = _create_run(
        tmp_path,
        provenance,
        planned_conditions=("A",),
    )
    start_condition(run_dir, manifest, condition="A")
    result_path, _, _ = _finish_condition(run_dir, manifest, "A")
    rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    rows[0]["cross_status"] = "C?"
    rows[0]["cross_check"]["summary_synthesized"] = True
    result_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    _rebind_output_hash(
        run_dir,
        condition="A",
        key="result",
        path=result_path,
    )

    verified = verify_run_manifest(run_dir)

    assert verified["conditions"]["A"]["status"] == "complete"


def test_error_row_requires_safe_empty_diagnostics(
    tmp_path: Path,
    synthetic_repository: tuple[Path, dict],
) -> None:
    _, provenance = synthetic_repository
    run_dir, manifest = _create_run(
        tmp_path,
        provenance,
        planned_conditions=("A",),
    )
    start_condition(run_dir, manifest, condition="A")
    result_path, _, _ = _finish_condition(run_dir, manifest, "A")
    rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    rows[0]["error"] = "RuntimeError: sample failure"
    result_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    _rebind_output_hash(
        run_dir,
        condition="A",
        key="result",
        path=result_path,
    )

    with pytest.raises(ProvenanceError, match="오류 행"):
        verify_run_manifest(run_dir)


def test_rebound_result_requires_replayable_spec_hit_fields(
    tmp_path: Path,
    synthetic_repository: tuple[Path, dict],
) -> None:
    _, provenance = synthetic_repository
    run_dir, manifest = _create_run(tmp_path, provenance)
    start_condition(run_dir, manifest, condition="B0")
    result_path, _, _ = _finish_condition(run_dir, manifest, "B0")
    rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    rows[0]["spec_hits"] = [{"clause_id": "SCREW-03"}]
    result_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    _rebind_output_hash(
        run_dir,
        condition="B0",
        key="result",
        path=result_path,
    )

    with pytest.raises(ProvenanceError, match="spec_hits.*필수 필드"):
        verify_run_manifest(run_dir)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("row_count", "처리 건수"),
        ("spend", "예상 비용"),
        ("models", "model ID"),
    ],
)
def test_rebound_result_cannot_hide_internal_accounting_mismatch(
    mutation: str,
    message: str,
    tmp_path: Path,
    synthetic_repository: tuple[Path, dict],
) -> None:
    """파일·manifest hash를 다시 맞춰도 내부 계수 불일치는 거부한다."""
    _, provenance = synthetic_repository
    run_dir, manifest = _create_run(tmp_path, provenance)
    start_condition(run_dir, manifest, condition="B0")
    result_path, _, _ = _finish_condition(run_dir, manifest, "B0")
    rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    if mutation == "row_count":
        rows.pop()
    elif mutation == "spend":
        rows[0]["estimated_usd"] = 1.0
    else:
        rows[0]["observed_model_ids"] = ["openai:forged-model"]
        rows[1]["observed_model_ids"] = ["openai:forged-model"]
    result_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    _rebind_output_hash(
        run_dir,
        condition="B0",
        key="result",
        path=result_path,
    )

    with pytest.raises(ProvenanceError, match=message):
        verify_run_manifest(run_dir)


@pytest.mark.parametrize(
    ("mutation", "message", "verify_inputs"),
    [
        ("duplicate_id", "중복 id", False),
        ("outside_id", "평가 roster", False),
        ("truth", "dataset과 다릅니다", True),
    ],
)
def test_rebound_result_cannot_escape_manifest_roster_or_dataset_truth(
    mutation: str,
    message: str,
    verify_inputs: bool,
    tmp_path: Path,
    synthetic_repository: tuple[Path, dict],
) -> None:
    """파일·manifest hash를 다시 맞춰도 평가 대상·정답 결속은 바꿀 수 없다."""
    root, provenance = synthetic_repository
    run_dir, manifest = _create_run(tmp_path, provenance)
    start_condition(run_dir, manifest, condition="B0")
    result_path, _, _ = _finish_condition(run_dir, manifest, "B0")
    rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    if mutation == "duplicate_id":
        rows[1]["id"] = rows[0]["id"]
    elif mutation == "outside_id":
        rows[1]["id"] = "dev-outside"
    else:
        rows[0]["item"] = "bottle"
        rows[0]["label"] = "contamination"
        rows[0]["is_defect"] = True
    result_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    _rebind_output_hash(
        run_dir,
        condition="B0",
        key="result",
        path=result_path,
    )

    kwargs = {"repo_root": root, "verify_inputs": True} if verify_inputs else {}
    with pytest.raises(ProvenanceError, match=message):
        verify_run_manifest(run_dir, **kwargs)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("processed_samples", 1, "처리 건수"),
        ("condition_estimated_usd", 1.0, "예상 비용"),
        ("suite_estimated_usd", 1.0, "suite"),
        ("complete_capture", False, "complete_capture"),
    ],
)
def test_rebound_summary_cannot_hide_manifest_mismatch(
    field: str,
    value: object,
    message: str,
    tmp_path: Path,
    synthetic_repository: tuple[Path, dict],
) -> None:
    _, provenance = synthetic_repository
    run_dir, manifest = _create_run(tmp_path, provenance)
    start_condition(run_dir, manifest, condition="B0")
    _, summary_path, _ = _finish_condition(run_dir, manifest, "B0")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary[field] = value
    atomic_write_json(summary_path, summary)
    _rebind_output_hash(
        run_dir,
        condition="B0",
        key="summary",
        path=summary_path,
    )

    with pytest.raises(ProvenanceError, match=message):
        verify_run_manifest(run_dir)


@pytest.mark.parametrize("invalid", [True, "0.1", -1, float("nan")])
def test_rebound_result_rejects_invalid_estimated_usd(
    invalid: object,
    tmp_path: Path,
    synthetic_repository: tuple[Path, dict],
) -> None:
    _, provenance = synthetic_repository
    run_dir, manifest = _create_run(tmp_path, provenance)
    start_condition(run_dir, manifest, condition="B0")
    result_path, _, _ = _finish_condition(run_dir, manifest, "B0")
    rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    rows[0]["estimated_usd"] = invalid
    result_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    _rebind_output_hash(
        run_dir,
        condition="B0",
        key="result",
        path=result_path,
    )

    with pytest.raises(ProvenanceError):
        verify_run_manifest(run_dir)


def test_incomplete_condition_keeps_verified_accounting_without_final_conclusion(
    tmp_path: Path,
    synthetic_repository: tuple[Path, dict],
) -> None:
    _, provenance = synthetic_repository
    run_dir, manifest = _create_run(tmp_path, provenance)
    start_condition(run_dir, manifest, condition="B0")
    record_condition_progress(
        run_dir,
        condition="B0",
        processed_samples=1,
        estimated_spend_usd=0.25,
    )
    _finish_condition(run_dir, manifest, "B0", complete_capture=False)

    verified = verify_run_manifest(run_dir, require_complete=False)
    assert verified["status"] == "incomplete"
    assert verified["conditions"]["B0"]["complete_capture"] is False
    with pytest.raises(ProvenanceError, match="완료"):
        verify_run_manifest(run_dir)


def test_input_change_and_joined_suite_mismatch_are_rejected(
    tmp_path: Path,
    synthetic_repository: tuple[Path, dict],
) -> None:
    root, provenance = synthetic_repository
    run_dir, manifest = _create_run(tmp_path, provenance)
    changed = _manifest(
        provenance,
        models={
            "llm_fast": "different-model",
            "llm_quality": "gpt-quality",
            "vlm_observer": "gpt-vision",
            "embed": "embedding-model",
        },
    )
    with pytest.raises(ProvenanceError, match="설정"):
        join_run_manifest(run_dir, changed, condition="B0")

    start_condition(run_dir, manifest, condition="B0")
    _finish_condition(run_dir, manifest, "B0")
    (root / "data" / "specs" / "screw.md").write_text("# 변경\n", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="변경"):
        verify_run_manifest(run_dir, repo_root=root, verify_inputs=True)


@pytest.mark.parametrize("target", ["lots", "evaluation_image"])
def test_bound_lots_and_evaluation_image_changes_are_rejected(
    target: str,
    tmp_path: Path,
    synthetic_repository: tuple[Path, dict],
) -> None:
    root, provenance = synthetic_repository
    run_dir, manifest = _create_run(tmp_path, provenance)
    start_condition(run_dir, manifest, condition="B0")
    _finish_condition(run_dir, manifest, "B0")

    if target == "lots":
        changed_path = root / "data" / "lots.json"
        changed_path.write_text('{"changed": true}\n', encoding="utf-8")
    else:
        changed_path = (
            root / "data" / "mvtec" / "screw" / "test" / "good" / "001.png"
        )
        changed_path.write_bytes(b"changed-evaluation-image")

    with pytest.raises(ProvenanceError, match="변경"):
        verify_run_manifest(run_dir, repo_root=root, verify_inputs=True)


def test_new_input_file_is_detected_after_the_run(
    tmp_path: Path,
    synthetic_repository: tuple[Path, dict],
) -> None:
    root, provenance = synthetic_repository
    run_dir, manifest = _create_run(tmp_path, provenance)
    start_condition(run_dir, manifest, condition="B0")
    _finish_condition(run_dir, manifest, "B0")
    (root / "data" / "specs" / "added.md").write_text("# 추가\n", encoding="utf-8")

    with pytest.raises(ProvenanceError, match="파일 목록"):
        verify_run_manifest(run_dir, repo_root=root, verify_inputs=True)


def test_failed_condition_never_becomes_complete(
    tmp_path: Path,
    synthetic_repository: tuple[Path, dict],
) -> None:
    _, provenance = synthetic_repository
    run_dir, manifest = _create_run(tmp_path, provenance)
    start_condition(run_dir, manifest, condition="B0")

    failed = fail_condition(
        run_dir,
        condition="B0",
        error="RuntimeError: synthetic failure",
        failed_at="2026-10-21T12:05:00Z",
    )

    assert failed["status"] == "failed"
    assert failed["conditions"]["B0"]["status"] == "failed"
    with pytest.raises(ProvenanceError, match="완료"):
        verify_run_manifest(run_dir)


def test_atomic_json_rejects_nonfinite_without_overwriting(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    atomic_write_json(path, {"value": 1})
    original = path.read_bytes()

    with pytest.raises(ProvenanceError):
        atomic_write_json(path, {"value": math.nan})

    assert path.read_bytes() == original


def test_create_and_join_reject_existing_or_absolute_artifact_paths(
    tmp_path: Path,
    synthetic_repository: tuple[Path, dict],
) -> None:
    _, provenance = synthetic_repository
    run_dir, manifest = _create_run(tmp_path, provenance)
    with pytest.raises(ProvenanceError):
        create_run_manifest(run_dir, manifest)

    altered = json.loads(json.dumps(provenance))
    altered["artifacts"]["dataset"]["path"] = "/tmp/dataset.json"
    with pytest.raises(ProvenanceError, match="상대경로"):
        _manifest(altered)


def test_progress_persists_manifest_hash_and_reloads(
    tmp_path: Path,
    synthetic_repository: tuple[Path, dict],
) -> None:
    _, provenance = synthetic_repository
    run_dir, manifest = _create_run(tmp_path, provenance)
    start_condition(run_dir, manifest, condition="B0")
    updated = record_condition_progress(
        run_dir,
        condition="B0",
        processed_samples=2,
        estimated_spend_usd=0.125,
    )

    loaded = load_run_manifest(run_dir)
    assert loaded["manifest_sha256"] == updated["manifest_sha256"]
    assert loaded["conditions"]["B0"]["processed_samples"] == 2
    assert loaded["suite_estimated_spend_usd"] == pytest.approx(0.125)
