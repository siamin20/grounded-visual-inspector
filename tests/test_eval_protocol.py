from __future__ import annotations

import hashlib
import json

import pytest

from eval.preflight import (
    EvaluationPreflightError,
    guard_run_mode,
    select_dev_demo_samples,
)


REAL_BACKENDS = {
    "LLM_BACKEND": "openai",
    "VLM_BACKEND": "openai",
    "EMBED_BACKEND": "openai",
}


def _write_dataset(tmp_path, *, heldout_count: int = 140):
    dataset_path = tmp_path / "dataset.json"
    samples = [
        {"id": f"heldout-{index:03d}", "split": "heldout"}
        for index in range(heldout_count)
    ]
    samples.extend([
        {"id": "dev-good", "split": "dev", "label": "good"},
        {"id": "dev-defect", "split": "dev", "label": "scratch"},
    ])
    dataset_path.write_text(
        json.dumps({"samples": samples}),
        encoding="utf-8",
    )
    return dataset_path


def _write_manifest(tmp_path, dataset_path, **overrides):
    payload = {
        "schema_version": 1,
        "status": "frozen",
        "split": "heldout",
        "expected_samples": 140,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
    }
    payload.update(overrides)
    manifest_path = tmp_path / "freeze_manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def _guard(dataset_path, **overrides):
    values = {
        "split": "heldout",
        "dry_run": False,
        "items": None,
        "limit": None,
        "final_run": True,
        "freeze_manifest": None,
        "backends": REAL_BACKENDS,
        "dataset_path": dataset_path,
    }
    values.update(overrides)
    return guard_run_mode(**values)


def test_demo_selector_returns_only_dev_good_and_defect() -> None:
    samples = [
        {"id": "heldout-good", "split": "heldout", "label": "good"},
        {"id": "heldout-defect", "split": "heldout", "label": "scratch"},
        {"id": "dev-good", "split": "dev", "label": "good"},
        {"id": "dev-defect", "split": "dev", "label": "scratch"},
    ]

    good, defect = select_dev_demo_samples(samples)

    assert good["id"] == "dev-good"
    assert defect["id"] == "dev-defect"
    assert good["split"] == defect["split"] == "dev"


def test_demo_selector_does_not_fall_back_to_heldout() -> None:
    with pytest.raises(EvaluationPreflightError, match="dev 분할"):
        select_dev_demo_samples([
            {"id": "heldout-good", "split": "heldout", "label": "good"},
            {"id": "dev-defect", "split": "dev", "label": "scratch"},
        ])


def test_dev_keeps_dry_run_and_partial_filters() -> None:
    assert guard_run_mode(
        split="dev",
        dry_run=True,
        items="screw",
        limit=5,
        final_run=False,
        freeze_manifest=None,
        backends={
            "LLM_BACKEND": "fake",
            "VLM_BACKEND": "fake",
            "EMBED_BACKEND": "fake",
        },
        dataset_path=pytest.fail,
    ) is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dry_run": True}, "--dry-run"),
        ({"items": "screw"}, "--items"),
        ({"items": ""}, "--items"),
        ({"limit": 1}, "--limit"),
        ({"final_run": False}, "--final-run"),
    ],
)
def test_heldout_rejects_unsafe_modes(tmp_path, overrides, message) -> None:
    dataset_path = _write_dataset(tmp_path)
    manifest_path = _write_manifest(tmp_path, dataset_path)

    with pytest.raises(EvaluationPreflightError, match=message):
        _guard(
            dataset_path,
            freeze_manifest=str(manifest_path),
            **overrides,
        )


@pytest.mark.parametrize("fake_name", tuple(REAL_BACKENDS))
def test_heldout_rejects_each_fake_backend(tmp_path, fake_name) -> None:
    dataset_path = _write_dataset(tmp_path)
    manifest_path = _write_manifest(tmp_path, dataset_path)
    backends = dict(REAL_BACKENDS)
    backends[fake_name] = "fake"

    with pytest.raises(EvaluationPreflightError, match="fake 백엔드"):
        _guard(
            dataset_path,
            backends=backends,
            freeze_manifest=str(manifest_path),
        )


def test_heldout_requires_freeze_manifest(tmp_path) -> None:
    dataset_path = _write_dataset(tmp_path)

    with pytest.raises(EvaluationPreflightError, match="--freeze-manifest"):
        _guard(dataset_path)

    with pytest.raises(EvaluationPreflightError, match="manifest가 없습니다"):
        _guard(dataset_path, freeze_manifest=str(tmp_path / "missing.json"))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema_version": 2}, "schema_version"),
        ({"status": "draft"}, "status"),
        ({"split": "dev"}, "split"),
        ({"expected_samples": 139}, "expected_samples"),
        ({"dataset_sha256": "0" * 64}, "dataset.json이 변경"),
    ],
)
def test_heldout_rejects_invalid_manifest(tmp_path, overrides, message) -> None:
    dataset_path = _write_dataset(tmp_path)
    manifest_path = _write_manifest(tmp_path, dataset_path, **overrides)

    with pytest.raises(EvaluationPreflightError, match=message):
        _guard(dataset_path, freeze_manifest=str(manifest_path))


def test_heldout_rejects_wrong_dataset_count(tmp_path) -> None:
    dataset_path = _write_dataset(tmp_path, heldout_count=139)
    manifest_path = _write_manifest(tmp_path, dataset_path)

    with pytest.raises(EvaluationPreflightError, match="140건"):
        _guard(dataset_path, freeze_manifest=str(manifest_path))


def test_heldout_rejects_malformed_manifest_json(tmp_path) -> None:
    dataset_path = _write_dataset(tmp_path)
    manifest_path = tmp_path / "freeze_manifest.json"
    manifest_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(EvaluationPreflightError, match="올바른 JSON이 아닙니다"):
        _guard(dataset_path, freeze_manifest=str(manifest_path))


def test_valid_heldout_manifest_passes(tmp_path) -> None:
    dataset_path = _write_dataset(tmp_path)
    manifest_path = _write_manifest(tmp_path, dataset_path)

    manifest = _guard(dataset_path, freeze_manifest=str(manifest_path))

    assert manifest is not None
    assert manifest["status"] == "frozen"
