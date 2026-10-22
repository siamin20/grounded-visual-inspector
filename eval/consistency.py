"""R6 dev 20장×3회 verdict 일관성을 사전 고정·집계한다.

이 모듈에서는 시드 `20261020`으로 dev roster에서 20건을 품목×정상/결함여부로
비례 고정하고, 서로 다른 실험 결과를 결합한 후 고정 분모 20을 유지한다.
heldout은 이 도구와 `run_eval.py --sample-ids-file`에서 모두 거부한다.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.provenance import (  # noqa: E402
    ProvenanceError,
    canonical_sha256,
    read_json_object,
    sha256_file,
    verify_run_manifest,
)


DEFAULT_DATASET = ROOT / "eval" / "dataset.json"
DEFAULT_SELECTION = ROOT / "eval" / "consistency_dev_ids.json"
DEFAULT_SEED = 20261020
DEFAULT_SAMPLE_N = 20
REPEAT_COUNT = 3
VALID_VERDICTS = frozenset({"PASS", "REJECT", "ABSTAIN"})
SELECTION_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
SELECTION_PURPOSE = "dev_verdict_consistency"
SAMPLING_METHOD = "proportional_stratified_item_truth"


class ConsistencyError(ValueError):
    """일관성 측정의 재현성·완전성 계약이 깨진 오류."""


@dataclass(frozen=True)
class ConsistencySelection:
    """검증된 선정 파일과 manifest 기록."""

    path: Path
    relative_path: str
    sha256: str
    payload: dict[str, Any]

    @property
    def ids(self) -> list[str]:
        return list(self.payload["ids"])

    def manifest_record(self) -> dict[str, Any]:
        ids = self.ids
        return {
            "path": self.relative_path,
            "sha256": self.sha256,
            "purpose": SELECTION_PURPOSE,
            "split": "dev",
            "condition": "A",
            "seed": DEFAULT_SEED,
            "sample_n": DEFAULT_SAMPLE_N,
            "repeat_count": REPEAT_COUNT,
            "sampling": SAMPLING_METHOD,
            "ids": ids,
            "ids_sha256": canonical_sha256(ids),
        }


def _canonical_dev_roster(dataset_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        dataset = read_json_object(dataset_path)
    except (OSError, ProvenanceError) as exc:
        raise ConsistencyError(f"dataset을 읽을 수 없습니다: {dataset_path}") from exc
    if dataset.get("seed") != DEFAULT_SEED:
        raise ConsistencyError(f"dataset seed는 {DEFAULT_SEED}이어야 합니다.")
    raw_samples = dataset.get("samples")
    if not isinstance(raw_samples, list):
        raise ConsistencyError("dataset.samples가 배열이 아닙니다.")

    roster: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_samples):
        if not isinstance(raw, Mapping) or raw.get("split") != "dev":
            continue
        sample_id = raw.get("id")
        item = raw.get("item")
        is_defect = raw.get("is_defect")
        if (
            not isinstance(sample_id, str)
            or not sample_id.strip()
            or sample_id != sample_id.strip()
            or not isinstance(item, str)
            or not item.strip()
            or type(is_defect) is not bool
        ):
            raise ConsistencyError(f"dev sample 스키마가 올바르지 않습니다: index={index}")
        if sample_id in seen:
            raise ConsistencyError(f"dev sample id가 중복되었습니다: {sample_id}")
        seen.add(sample_id)
        roster.append(dict(raw))
    if len(roster) < DEFAULT_SAMPLE_N:
        raise ConsistencyError(
            f"dev 표본이 부족합니다: 필요 {DEFAULT_SAMPLE_N}, 실제 {len(roster)}"
        )
    return dataset, sorted(roster, key=lambda row: str(row["id"]))


def _select_stratified(
    roster: Sequence[Mapping[str, Any]],
    *,
    sample_n: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """largest-remainder로 품목×정상/결함 비율을 보존한다."""

    groups: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in roster:
        groups[(str(row["item"]), bool(row["is_defect"]))].append(dict(row))
    exact = {key: sample_n * len(values) / len(roster) for key, values in groups.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remaining = sample_n - sum(quotas.values())
    order = sorted(groups, key=lambda key: (-(exact[key] - quotas[key]), repr(key)))
    for key in order[:remaining]:
        quotas[key] += 1

    generator = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for key in sorted(groups, key=repr):
        values = sorted(groups[key], key=lambda row: str(row["id"]))
        quota = quotas[key]
        if quota > len(values):
            raise ConsistencyError(f"층화 표본이 부족합니다: {key}")
        selected.extend(generator.sample(values, quota))
    generator.shuffle(selected)
    counts = {
        f"{item}/{'defect' if is_defect else 'normal'}": quotas[(item, is_defect)]
        for item, is_defect in sorted(groups, key=repr)
    }
    return selected, counts


def build_selection(dataset_path: str | Path = DEFAULT_DATASET) -> dict[str, Any]:
    """결과를 보기 전에 고정된 dev 20 ID 약속을 만든다."""

    target = Path(dataset_path).expanduser().resolve(strict=True)
    _, roster = _canonical_dev_roster(target)
    selected, stratum_counts = _select_stratified(
        roster,
        sample_n=DEFAULT_SAMPLE_N,
        seed=DEFAULT_SEED,
    )
    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "purpose": SELECTION_PURPOSE,
        "split": "dev",
        "condition": "A",
        "seed": DEFAULT_SEED,
        "sample_n": DEFAULT_SAMPLE_N,
        "repeat_count": REPEAT_COUNT,
        "sampling": SAMPLING_METHOD,
        "dataset_sha256": sha256_file(target),
        "stratum_counts": stratum_counts,
        "ids": [str(row["id"]) for row in selected],
    }


def load_selection(
    path: str | Path = DEFAULT_SELECTION,
    *,
    dataset_path: str | Path = DEFAULT_DATASET,
    repo_root: str | Path = ROOT,
) -> ConsistencySelection:
    """선정 파일이 현재 dataset에서 재생성한 층화 결과인지 검증한다."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    raw_target = Path(path).expanduser()
    target = (
        (root / raw_target).resolve(strict=True)
        if not raw_target.is_absolute()
        else raw_target.resolve(strict=True)
    )
    try:
        relative = target.relative_to(root).as_posix()
    except ValueError as exc:
        raise ConsistencyError("선정 파일은 저장소 안에 있어야 합니다.") from exc
    try:
        payload = read_json_object(target)
    except (OSError, ProvenanceError) as exc:
        raise ConsistencyError(f"선정 파일을 읽을 수 없습니다: {target}") from exc
    expected = build_selection(dataset_path)
    if payload != expected:
        raise ConsistencyError(
            "선정 파일이 seed·층화·dataset hash·ID 계약과 다릅니다."
        )
    return ConsistencySelection(
        path=target,
        relative_path=relative,
        sha256=sha256_file(target),
        payload=payload,
    )


def build_execution_plan(selection: ConsistencySelection) -> dict[str, Any]:
    """실모델을 호출하지 않고 독립 3회 실행 계획만 만든다."""

    common = [
        ".venv/bin/python3",
        "eval/run_eval.py",
        "--condition",
        "A",
        "--split",
        "dev",
        "--sample-ids-file",
        selection.relative_path,
        "--suite-conditions",
        "A",
    ]
    repetitions = [
        {
            "repeat": repeat,
            "argv": [
                *common,
                "--consistency-repeat",
                str(repeat),
            ],
            "run_id_policy": "auto_unique",
        }
        for repeat in range(1, REPEAT_COUNT + 1)
    ]
    return {
        "schema_version": 1,
        "evidence_status": "PLAN_ONLY_NO_MODEL_CALLS",
        "split": "dev",
        "condition": "A",
        "temperature": 0.0,
        "sample_n": DEFAULT_SAMPLE_N,
        "repeat_count": REPEAT_COUNT,
        "selection": selection.manifest_record(),
        "repetitions": repetitions,
        "postprocess_argv": [
            ".venv/bin/python3",
            "-m",
            "eval.consistency",
            "report",
            "--selection",
            selection.relative_path,
            "--run",
            "<repeat-1-run-dir>",
            "--run",
            "<repeat-2-run-dir>",
            "--run",
            "<repeat-3-run-dir>",
        ],
        "claim_limit": "실행 계획일 뿐이며 성능 결과가 아님.",
    }


def analyze_repetition_rows(
    repetitions: Sequence[Sequence[Mapping[str, Any]]],
    *,
    selection: Mapping[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    """세 반복의 결과와 보수적 실패 처리로 일관성을 집계한다."""

    if len(repetitions) != REPEAT_COUNT:
        raise ConsistencyError(f"반복은 {REPEAT_COUNT}회여야 합니다.")
    ids = selection.get("ids")
    if (
        not isinstance(ids, list)
        or len(ids) != DEFAULT_SAMPLE_N
        or any(
            not isinstance(value, str)
            or not value.strip()
            or value != value.strip()
            for value in ids
        )
        or len(set(ids)) != DEFAULT_SAMPLE_N
    ):
        raise ConsistencyError("선정 ID는 중복 없는 20개 문자여야 합니다.")
    expected_ids = set(ids)
    rows_by_repeat: list[dict[str, Mapping[str, Any]]] = []
    for repeat_index, rows in enumerate(repetitions, start=1):
        by_id: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            sample_id = row.get("id") if isinstance(row, Mapping) else None
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ConsistencyError(f"{repeat_index}회 결과에 유효한 id가 없습니다.")
            if sample_id not in expected_ids:
                raise ConsistencyError(f"{repeat_index}회 결과에 선정 외 ID가 있습니다: {sample_id}")
            if sample_id in by_id:
                raise ConsistencyError(f"{repeat_index}회 결과에 중복 ID가 있습니다: {sample_id}")
            by_id[sample_id] = row
        rows_by_repeat.append(by_id)

    per_id: list[dict[str, Any]] = []
    affected = {"missing": set(), "error": set(), "invalid_verdict": set()}
    occurrence_counts = {"missing": 0, "error": 0, "invalid_verdict": 0}
    strict_unstable_count = 0
    valid_triplet_count = 0
    observed_change_count = 0
    for sample_id in ids:
        states: list[str] = []
        verdicts: list[str | None] = []
        for by_id in rows_by_repeat:
            row = by_id.get(sample_id)
            if row is None:
                state, verdict = "MISSING", None
                affected["missing"].add(sample_id)
                occurrence_counts["missing"] += 1
            elif row.get("error") not in (None, ""):
                state, verdict = "ERROR", None
                affected["error"].add(sample_id)
                occurrence_counts["error"] += 1
            elif row.get("verdict") not in VALID_VERDICTS:
                state, verdict = "INVALID_VERDICT", None
                affected["invalid_verdict"].add(sample_id)
                occurrence_counts["invalid_verdict"] += 1
            else:
                state, verdict = "VALID", str(row["verdict"])
            states.append(state)
            verdicts.append(verdict)
        all_valid = all(state == "VALID" for state in states)
        changed = all_valid and len(set(verdicts)) > 1
        strict_unstable = not all_valid or changed
        if all_valid:
            valid_triplet_count += 1
            observed_change_count += int(changed)
        strict_unstable_count += int(strict_unstable)
        per_id.append(
            {
                "id": sample_id,
                "states": states,
                "verdicts": verdicts,
                "all_repeats_valid": all_valid,
                "observed_verdict_changed": changed,
                "strict_unstable": strict_unstable,
            }
        )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evidence_status": (
            "DRYRUN_NOT_PERFORMANCE" if dry_run else "REAL_MODEL_DEV_DIAGNOSTIC"
        ),
        "split": "dev",
        "condition": "A",
        "seed": DEFAULT_SEED,
        "sample_n": DEFAULT_SAMPLE_N,
        "repeat_count": REPEAT_COUNT,
        "denominator_policy": "fixed_20_missing_error_invalid_as_unstable",
        "verdict_variation_count": strict_unstable_count,
        "verdict_variation_rate": strict_unstable_count / DEFAULT_SAMPLE_N,
        "preregistered_max_variation_rate": 0.1,
        "threshold_assessment": (
            "NOT_APPLICABLE_DRYRUN"
            if dry_run
            else (
                "REPRODUCIBILITY_INSUFFICIENT"
                if strict_unstable_count / DEFAULT_SAMPLE_N > 0.1
                else "WITHIN_PREREGISTERED_THRESHOLD"
            )
        ),
        "strict_unstable_count": strict_unstable_count,
        "strict_instability_rate": strict_unstable_count / DEFAULT_SAMPLE_N,
        "valid_triplet_count": valid_triplet_count,
        "observed_change_count": observed_change_count,
        "observed_change_rate": (
            observed_change_count / valid_triplet_count if valid_triplet_count else None
        ),
        "missing_occurrences": occurrence_counts["missing"],
        "error_occurrences": occurrence_counts["error"],
        "invalid_verdict_occurrences": occurrence_counts["invalid_verdict"],
        "affected_ids": {
            key: sorted(values) for key, values in affected.items()
        },
        "per_id": per_id,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConsistencyError(f"결과 JSONL을 읽을 수 없습니다: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConsistencyError(f"결과 JSONL이 손상되었습니다: {path.name}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ConsistencyError(f"결과 JSONL 행이 개체가 아닙니다: {path.name}:{line_number}")
        rows.append(value)
    return rows


def _registered_result_path(run_dir: Path, manifest: Mapping[str, Any]) -> Path:
    condition = manifest.get("conditions", {}).get("A")
    record = condition.get("result") if isinstance(condition, Mapping) else None
    relative = record.get("path") if isinstance(record, Mapping) else None
    if not isinstance(relative, str) or not relative:
        raise ConsistencyError(f"A 결과 파일이 manifest에 등록되어 있지 않습니다: {run_dir}")
    target = (run_dir / relative).resolve(strict=True)
    try:
        target.relative_to(run_dir.resolve(strict=True))
    except ValueError as exc:
        raise ConsistencyError("manifest 결과 경로가 run 디렉터리 밖으로 나갔습니다.") from exc
    return target


def _validate_run_contracts(
    manifests: Sequence[Mapping[str, Any]],
    *,
    selection: ConsistencySelection,
) -> list[dict[str, Any]]:
    if len(manifests) != REPEAT_COUNT:
        raise ConsistencyError(f"manifest는 {REPEAT_COUNT}개여야 합니다.")
    expected_selection = selection.manifest_record()
    repeats: dict[int, Mapping[str, Any]] = {}
    run_ids: set[str] = set()
    for manifest in manifests:
        run_id = manifest.get("run_id")
        settings = manifest.get("settings")
        conditions = manifest.get("conditions")
        if manifest.get("split") != "dev" or manifest.get("planned_conditions") != ["A"]:
            raise ConsistencyError("일관성 결과는 dev/A 단일 suite여야 합니다.")
        if manifest.get("expected_n") != DEFAULT_SAMPLE_N:
            raise ConsistencyError("일관성 결과의 expected_n은 20이어야 합니다.")
        if not isinstance(settings, Mapping) or settings.get("temperature") != 0.0:
            raise ConsistencyError("일관성 결과의 temperature는 0이어야 합니다.")
        if settings.get("sample_ids_file") != expected_selection:
            raise ConsistencyError("일관성 선정 파일 기록이 다릅니다.")
        repeat = settings.get("consistency_repeat")
        if type(repeat) is not int or repeat not in {1, 2, 3} or repeat in repeats:
            raise ConsistencyError("일관성 반복 번호는 1·2·3이 중복 없이 정확히 하나여야 합니다.")
        if not isinstance(run_id, str) or run_id in run_ids:
            raise ConsistencyError("일관성 반복은 서로 다른 run_id여야 합니다.")
        condition_status = (
            conditions.get("A", {}).get("status")
            if isinstance(conditions, Mapping)
            else None
        )
        if condition_status not in {"complete", "incomplete"}:
            raise ConsistencyError(
                "일관성 A 조건은 결과가 결속된 complete 또는 incomplete여야 합니다."
            )
        repeats[repeat] = manifest
        run_ids.add(run_id)
    if set(repeats) != {1, 2, 3}:
        raise ConsistencyError("일관성 반복 1·2·3이 모두 필요합니다.")

    ordered = [repeats[index] for index in (1, 2, 3)]
    first = ordered[0]
    first_settings = dict(first["settings"])
    first_settings.pop("consistency_repeat", None)
    equality_fields = ("backends", "models", "pricing", "repository", "artifacts", "runtime")
    first_observed = first["conditions"]["A"].get("observed_model_ids")
    for manifest in ordered[1:]:
        settings = dict(manifest["settings"])
        settings.pop("consistency_repeat", None)
        if settings != first_settings:
            raise ConsistencyError("반복 사이의 실험 설정이 다릅니다.")
        if any(manifest.get(field) != first.get(field) for field in equality_fields):
            raise ConsistencyError("반복 사이의 모델·backend·provenance가 다릅니다.")
        if manifest["conditions"]["A"].get("observed_model_ids") != first_observed:
            raise ConsistencyError("반복 사이의 응답 model ID가 다릅니다.")
    return [
        {
            "repeat": index,
            "run_id": manifest["run_id"],
            "immutable_sha256": manifest["immutable_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "status": manifest["conditions"]["A"]["status"],
            "observed_model_ids": manifest["conditions"]["A"]["observed_model_ids"],
            "estimated_usd": manifest["conditions"]["A"]["estimated_spend_usd"],
        }
        for index, manifest in enumerate(ordered, start=1)
    ]


def analyze_run_dirs(
    run_dirs: Sequence[str | Path],
    *,
    selection_path: str | Path = DEFAULT_SELECTION,
    repo_root: str | Path = ROOT,
) -> dict[str, Any]:
    """결과 manifest가 결속한 실모델 3회 결과를 일관성 지표로 집계한다."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    selection = load_selection(
        selection_path,
        repo_root=root,
        dataset_path=root / "eval" / "dataset.json",
    )
    if len(run_dirs) != REPEAT_COUNT:
        raise ConsistencyError(f"--run은 {REPEAT_COUNT}개 지정해야 합니다.")
    manifests: list[dict[str, Any]] = []
    rows: list[list[dict[str, Any]]] = []
    for raw in run_dirs:
        run_dir = Path(raw).expanduser().resolve(strict=True)
        try:
            manifest = verify_run_manifest(
                run_dir,
                reject_dry_run=True,
                reject_fake=True,
                require_complete=False,
                validate_rows=True,
                repo_root=root,
                verify_inputs=True,
            )
        except ProvenanceError as exc:
            raise ConsistencyError(f"일관성 run 검증이 실패했습니다: {run_dir}: {exc}") from exc
        manifests.append(manifest)
        rows.append(_load_jsonl(_registered_result_path(run_dir, manifest)))

    run_records = _validate_run_contracts(manifests, selection=selection)
    report = analyze_repetition_rows(rows, selection=selection.payload, dry_run=False)
    report["selection"] = selection.manifest_record()
    report["runs"] = run_records
    report["claim_limit"] = "dev 일관성 진단임. heldout 성능으로 해석하지 않음."
    return report


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
    except FileExistsError as exc:
        raise ConsistencyError(f"기존 파일을 덮어쓸 수 없습니다: {path}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select", help="고정된 dev 20 ID 파일 생성")
    select.add_argument("--dataset", default=str(DEFAULT_DATASET))
    select.add_argument("--output", default=str(DEFAULT_SELECTION))

    plan = subparsers.add_parser("plan", help="독립 3회 실행 계획 생성")
    plan.add_argument("--selection", default=str(DEFAULT_SELECTION))
    plan.add_argument("--output")

    report = subparsers.add_parser("report", help="3회 실모델 결과 후처리")
    report.add_argument("--selection", default=str(DEFAULT_SELECTION))
    report.add_argument("--run", action="append", required=True)
    report.add_argument("--output")
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "select":
            payload = build_selection(args.dataset)
            output = Path(args.output).expanduser().resolve()
            _write_json_exclusive(output, payload)
            print(f"선정 파일: {output}")
            print(f"ID: {len(payload['ids'])}개, seed={payload['seed']}")
            return
        if args.command == "plan":
            selection = load_selection(args.selection)
            payload = build_execution_plan(selection)
            rendered = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
            if args.output:
                output = Path(args.output).expanduser().resolve()
                _write_json_exclusive(output, payload)
                print(f"일관성 실행 계획: {output}")
            else:
                print(rendered)
            return
        report = analyze_run_dirs(args.run, selection_path=args.selection)
        rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            output = Path(args.output).expanduser().resolve()
            _write_json_exclusive(output, report)
            print(f"일관성 보고서: {output}")
        else:
            print(rendered)
    except (ConsistencyError, OSError, ValueError) as exc:
        raise SystemExit(f"일관성 거부 실패: {exc}") from exc


if __name__ == "__main__":
    main()
