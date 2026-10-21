"""heldout 평가의 우발 실행과 개발 데이터 오염을 막는 사전 검사다."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_HELDOUT_SAMPLES = 140
FREEZE_SCHEMA_VERSION = 1


class EvaluationPreflightError(ValueError):
    """평가 실행 조건이 동결 규칙을 충족하지 못했을 때 발생한다."""


def select_dev_demo_samples(
    samples: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """dev 분할에서 good과 결함 샘플을 하나씩 고른다."""
    dev_samples = [
        dict(sample)
        for sample in samples
        if isinstance(sample, Mapping) and sample.get("split") == "dev"
    ]
    good = next(
        (sample for sample in dev_samples if sample.get("label") == "good"),
        None,
    )
    defect = next(
        (sample for sample in dev_samples if sample.get("label") != "good"),
        None,
    )
    if good is None or defect is None:
        raise EvaluationPreflightError(
            "dev 분할에 good과 결함 데모 샘플이 모두 필요합니다."
        )
    return good, defect


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _heldout_count(dataset_path: Path) -> int:
    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationPreflightError(
            f"데이터셋 메타데이터를 읽을 수 없습니다: {dataset_path}"
        ) from exc
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list):
        raise EvaluationPreflightError("eval/dataset.json의 samples가 배열이 아닙니다.")
    return sum(
        isinstance(sample, dict) and sample.get("split") == "heldout"
        for sample in samples
    )


def validate_freeze_manifest(
    manifest_path: Path,
    *,
    dataset_path: Path,
) -> dict[str, Any]:
    """R0 최소 동결 계약과 현재 데이터셋 메타데이터를 대조한다."""
    if not manifest_path.is_file():
        raise EvaluationPreflightError(
            f"동결 manifest가 없습니다: {manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationPreflightError(
            f"동결 manifest가 올바른 JSON이 아닙니다: {manifest_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise EvaluationPreflightError("동결 manifest 최상위 값은 객체여야 합니다.")

    expected = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "status": "frozen",
        "split": "heldout",
        "expected_samples": EXPECTED_HELDOUT_SAMPLES,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise EvaluationPreflightError(
                f"동결 manifest의 {key}가 {value!r}이어야 합니다."
            )

    recorded_hash = payload.get("dataset_sha256")
    if not isinstance(recorded_hash, str) or len(recorded_hash) != 64:
        raise EvaluationPreflightError(
            "동결 manifest에 64자 dataset_sha256이 필요합니다."
        )
    actual_hash = _sha256(dataset_path)
    if recorded_hash.lower() != actual_hash:
        raise EvaluationPreflightError(
            "동결 후 eval/dataset.json이 변경되었습니다."
        )

    actual_count = _heldout_count(dataset_path)
    if actual_count != EXPECTED_HELDOUT_SAMPLES:
        raise EvaluationPreflightError(
            "heldout 샘플 수가 "
            f"{EXPECTED_HELDOUT_SAMPLES}건이 아닙니다: {actual_count}건"
        )
    return dict(payload)


def guard_run_mode(
    *,
    split: str,
    dry_run: bool,
    items: str | None,
    limit: int | None,
    final_run: bool,
    freeze_manifest: str | None,
    backends: Mapping[str, str],
    dataset_path: Path,
) -> dict[str, Any] | None:
    """heldout에만 최종 실행 규칙을 강제한다."""
    if split != "heldout":
        return None
    if dry_run:
        raise EvaluationPreflightError("heldout은 --dry-run으로 실행할 수 없습니다.")
    fake_backends = sorted(
        name for name, value in backends.items() if value == "fake"
    )
    if fake_backends:
        raise EvaluationPreflightError(
            "heldout은 fake 백엔드를 허용하지 않습니다: "
            + ", ".join(fake_backends)
        )
    if items is not None:
        raise EvaluationPreflightError("heldout은 --items 부분 실행을 허용하지 않습니다.")
    if limit is not None:
        raise EvaluationPreflightError("heldout은 --limit 부분 실행을 허용하지 않습니다.")
    if not final_run:
        raise EvaluationPreflightError("heldout은 --final-run 명시 승인이 필요합니다.")
    if not freeze_manifest:
        raise EvaluationPreflightError("heldout은 --freeze-manifest 경로가 필요합니다.")
    return validate_freeze_manifest(
        Path(freeze_manifest).expanduser().resolve(),
        dataset_path=dataset_path,
    )
