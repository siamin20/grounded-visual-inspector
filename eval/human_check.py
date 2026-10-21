"""validate 전·후 보고서를 짝지어 블라인드 사람 채점 세트로 만든다."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.report import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_RESULTS_DIR,
    ResultDataError,
    load_expected_rows,
    load_result_rows,
)
from eval.evaluators import wilson_interval  # noqa: E402


DEFAULT_SEED = 20261020
DEFAULT_SAMPLE = 20
DEFAULT_BLIND_NAME = "human_check_blind.jsonl"
DEFAULT_MAPPING_NAME = "human_check_mapping.json"
PROTOCOL_VERSION = 2
STAGES = ("before_validate", "after_validate")
JUDGMENTS = frozenset(
    {
        "SUPPORTED",
        "PARAPHRASE_SUPPORTED",
        "HALLUCINATED",
        "NON_FACTUAL",
        "UNCERTAIN",
    }
)
STAGE_GUESSES = frozenset({"before", "after", "unknown"})
_HALLUCINATION_JUDGMENTS = frozenset({"HALLUCINATED", "UNCERTAIN"})
_ROW_KEYS = frozenset({"id", "reference_observations", "claims", "stage_guess"})
_CLAIM_KEYS = frozenset({"claim_id", "source", "text", "judgment"})


class HumanCheckError(ValueError):
    """블라인드 채점의 완전성·무결성을 위반한 오류."""


def _atomic_write(path: Path, text: str, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _observation_texts(value: Any) -> list[str]:
    """조건 단서가 있는 provenance를 버리고 관찰문 텍스트만 정규화한다."""

    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if not isinstance(value, list):
        return []
    texts: list[str] = []
    for observation in value:
        text = observation.get("text") if isinstance(observation, Mapping) else observation
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return texts


def _report_claims(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """두 stage에 같은 claim 스키마를 적용한다."""

    claims: list[dict[str, Any]] = []
    summary = report.get("summary")
    if isinstance(summary, str) and summary.strip():
        claims.append(
            {
                "claim_id": "C-0001",
                "source": "summary",
                "text": summary.strip(),
                "judgment": None,
            }
        )
    for text in _observation_texts(report.get("observations", [])):
        claims.append(
            {
                "claim_id": f"C-{len(claims) + 1:04d}",
                "source": "observation",
                "text": text,
                "judgment": None,
            }
        )
    return claims


def _immutable_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    claims = row.get("claims")
    immutable_claims: list[dict[str, Any]] = []
    if isinstance(claims, list):
        for claim in claims:
            if isinstance(claim, Mapping):
                immutable_claims.append(
                    {
                        "claim_id": claim.get("claim_id"),
                        "source": claim.get("source"),
                        "text": claim.get("text"),
                    }
                )
    return {
        "id": row.get("id"),
        "reference_observations": row.get("reference_observations"),
        "claims": immutable_claims,
    }


def _payload_sha256(row: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _immutable_payload(row),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _report_sha256(report: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HumanCheckError("pre_validation_snapshot.report를 정규 JSON으로 만들 수 없습니다.") from exc
    return hashlib.sha256(encoded).hexdigest()


def _pre_validation_report(source: Mapping[str, Any], *, sample_id: str) -> Mapping[str, Any]:
    snapshot = source.get("pre_validation_snapshot")
    if not isinstance(snapshot, Mapping):
        raise HumanCheckError(
            "사전 선택된 결과의 pre_validation_snapshot이 없거나 무효입니다"
            f"(재표집하지 않음): {sample_id}"
        )
    if snapshot.get("version") != 1 or snapshot.get("stage") != "before_validate":
        raise HumanCheckError(f"pre_validation_snapshot 계약이 잘못되었습니다: {sample_id}")
    report = snapshot.get("report")
    digest = snapshot.get("report_sha256")
    if not isinstance(report, Mapping) or not isinstance(digest, str):
        raise HumanCheckError(f"pre_validation_snapshot 보고서 또는 hash가 무효입니다: {sample_id}")
    if _report_sha256(report) != digest:
        raise HumanCheckError(f"pre_validation_snapshot 보고서 hash가 일치하지 않습니다: {sample_id}")
    cost = snapshot.get("cost_usd")
    tokens = snapshot.get("tokens")
    if (
        isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(float(cost))
        or float(cost) < 0
        or isinstance(tokens, bool)
        or not isinstance(tokens, int)
        or tokens < 0
    ):
        raise HumanCheckError(f"pre_validation_snapshot usage가 무효입니다: {sample_id}")
    return report


def _derived_expected_rows(
    rows: Sequence[Mapping[str, Any]], *, split: str
) -> list[dict[str, Any]]:
    """직접 함수 호출의 하위 호환용 roster다. CLI는 항상 dataset roster를 쓴다."""

    derived: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if row.get("condition") != "A" or row.get("split") != split:
            continue
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id or sample_id != sample_id.strip():
            raise HumanCheckError(f"A/{split} 결과에 유효한 문자열 id가 필요합니다.")
        if sample_id in seen:
            raise HumanCheckError(f"A/{split}에 중복 샘플 id가 있습니다: {sample_id}")
        seen.add(sample_id)
        derived.append(
            {
                "id": sample_id,
                "item": row.get("item"),
                "label": row.get("label"),
                "is_defect": row.get("is_defect"),
                "split": split,
            }
        )
    return sorted(derived, key=lambda value: str(value["id"]))


def _canonical_roster(
    expected_rows: Sequence[Mapping[str, Any]], *, split: str
) -> list[dict[str, Any]]:
    roster: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in expected_rows:
        if not isinstance(raw, Mapping) or raw.get("split") != split:
            continue
        row = dict(raw)
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id or sample_id != sample_id.strip():
            raise HumanCheckError(f"canonical {split} roster에 유효한 문자열 id가 필요합니다.")
        if sample_id in seen:
            raise HumanCheckError(f"canonical {split} roster에 중복 id가 있습니다: {sample_id}")
        seen.add(sample_id)
        roster.append(row)
    if not roster:
        raise HumanCheckError(f"canonical roster에 split={split} 표본이 없습니다.")
    return sorted(roster, key=lambda row: str(row["id"]))


def _select_stratified(
    roster: Sequence[Mapping[str, Any]],
    *,
    sample: int,
    generator: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """품목·정상/결함 비율을 유지하는 largest-remainder 표본을 고른다."""

    groups: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
    for raw in roster:
        item = raw.get("item")
        is_defect = raw.get("is_defect")
        if not isinstance(item, str) or not item.strip() or type(is_defect) is not bool:
            raise HumanCheckError("canonical roster에는 item 문자열과 bool is_defect가 필요합니다.")
        groups[(item, is_defect)].append(dict(raw))

    exact = {
        key: sample * len(values) / len(roster)
        for key, values in groups.items()
    }
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remaining = sample - sum(quotas.values())
    order = sorted(
        groups,
        key=lambda key: (-(exact[key] - quotas[key]), repr(key)),
    )
    for key in order[:remaining]:
        quotas[key] += 1

    selected: list[dict[str, Any]] = []
    for key in sorted(groups, key=repr):
        values = sorted(groups[key], key=lambda row: str(row["id"]))
        quota = quotas[key]
        if quota > len(values):
            raise HumanCheckError(f"stratum 표본이 부족합니다: {key}")
        selected.extend(generator.sample(values, quota))
    generator.shuffle(selected)
    counts = {
        f"{item}/{'defect' if is_defect else 'normal'}": quotas[(item, is_defect)]
        for item, is_defect in sorted(groups, key=repr)
    }
    return selected, counts


def _result_rows_by_id(
    rows: Sequence[Mapping[str, Any]],
    *,
    roster: Sequence[Mapping[str, Any]],
    split: str,
) -> dict[str, dict[str, Any]]:
    expected = {str(row["id"]): row for row in roster}
    by_id: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if raw.get("condition") != "A":
            continue
        sample_id = raw.get("id")
        if not isinstance(sample_id, str) or not sample_id or sample_id != sample_id.strip():
            raise HumanCheckError("A 결과에 유효한 문자열 id가 필요합니다.")
        if raw.get("split") != split:
            if sample_id in expected:
                raise HumanCheckError(
                    f"canonical ID의 split이 A 결과와 다릅니다: {sample_id}"
                )
            continue
        if sample_id not in expected:
            raise HumanCheckError(f"A/{split} 결과에 roster 외 id가 있습니다: {sample_id}")
        if sample_id in by_id:
            raise HumanCheckError(f"A/{split}에 중복 샘플 id가 있습니다: {sample_id}")
        canonical = expected[sample_id]
        for field in ("item", "label", "is_defect", "split"):
            if raw.get(field) != canonical.get(field):
                raise HumanCheckError(
                    f"A/{split} 결과의 {field}가 canonical roster와 다릅니다: {sample_id}"
                )
        by_id[sample_id] = dict(raw)
    return by_id


def create_blind_set(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_rows: Sequence[Mapping[str, Any]] | None = None,
    sample: int = DEFAULT_SAMPLE,
    seed: int = DEFAULT_SEED,
    split: str = "heldout",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """canonical ID를 한 번 뽑아 validate 전·후 보고서 2N개를 섞는다."""

    if isinstance(sample, bool) or not isinstance(sample, int) or sample <= 0:
        raise HumanCheckError("--sample은 1 이상 정수여야 합니다.")
    canonical_source = (
        expected_rows
        if expected_rows is not None
        else _derived_expected_rows(rows, split=split)
    )
    roster = _canonical_roster(canonical_source, split=split)
    if sample > len(roster):
        raise HumanCheckError(
            f"{split} canonical 표본이 부족합니다: 필요 {sample}, 가능 {len(roster)}"
        )

    # 결과 가용성을 보기 전에 canonical ID를 먼저 고정한다. 실패 표본은 대체하지 않는다.
    generator = random.Random(seed)
    selected_expected, stratum_counts = _select_stratified(
        roster,
        sample=sample,
        generator=generator,
    )
    selected_ids = [str(row["id"]) for row in selected_expected]
    result_by_id = _result_rows_by_id(rows, roster=roster, split=split)

    selected: list[dict[str, Any]] = []
    for pair_index, sample_id in enumerate(selected_ids, start=1):
        pair_id = f"PAIR-{pair_index:04d}"
        source = result_by_id.get(sample_id)
        if source is None:
            raise HumanCheckError(
                f"사전 선택된 A/{split} 결과가 누락되었습니다(재표집하지 않음): {sample_id}"
            )
        error = source.get("error")
        if error not in (None, ""):
            raise HumanCheckError(
                f"사전 선택된 A/{split} 결과에 오류가 있습니다(재표집하지 않음): "
                f"{sample_id}: {error}"
            )
        references = _observation_texts(source.get("observations_text"))
        if not references:
            raise HumanCheckError(
                f"사전 선택된 결과의 관찰 원문이 비어 있습니다: {sample_id}"
            )
        stage_reports = (
            ("before_validate", _pre_validation_report(source, sample_id=sample_id)),
            ("after_validate", source.get("report")),
        )
        for stage, report in stage_reports:
            if not isinstance(report, Mapping):
                raise HumanCheckError(
                    "사전 선택된 결과의 최종 report가 없거나 무효입니다"
                    f"(재표집하지 않음): {sample_id}"
                )
            claims = _report_claims(report)
            if not claims:
                raise HumanCheckError(
                    f"사전 선택된 결과의 {stage}에 채점할 claim이 없습니다: {sample_id}"
                )
            selected.append(
                {
                    "pair_id": pair_id,
                    "result_id": sample_id,
                    "stage": stage,
                    "reference_observations": list(references),
                    "claims": claims,
                }
            )

    generator.shuffle(selected)
    blind_rows: list[dict[str, Any]] = []
    entries: dict[str, dict[str, str]] = {}
    for index, selected_row in enumerate(selected, start=1):
        anonymous_id = f"HC-{index:04d}"
        blind_row = {
            "id": anonymous_id,
            "reference_observations": selected_row["reference_observations"],
            "claims": selected_row["claims"],
            "stage_guess": None,
        }
        blind_rows.append(blind_row)
        entries[anonymous_id] = {
            "pair_id": str(selected_row["pair_id"]),
            "result_id": str(selected_row["result_id"]),
            "stage": str(selected_row["stage"]),
            "payload_sha256": _payload_sha256(blind_row),
        }

    mapping: dict[str, Any] = {
        "version": PROTOCOL_VERSION,
        "protocol": "paired_validate_hallucination",
        "seed": seed,
        "split": split,
        "sample_pairs": sample,
        "sampling": "proportional_stratified_item_truth",
        "stratum_counts": stratum_counts,
        "stages": list(STAGES),
        "selected_result_ids": selected_ids,
        "entries": entries,
    }
    return blind_rows, mapping


def write_blind_set(
    results: str | Path,
    *,
    output: str | Path | None = None,
    mapping_path: str | Path | None = None,
    dataset_path: str | Path = DEFAULT_DATASET,
    sample: int = DEFAULT_SAMPLE,
    seed: int = DEFAULT_SEED,
    split: str = "heldout",
    force: bool = False,
) -> tuple[Path, Path, int]:
    if split == "heldout" and (sample != DEFAULT_SAMPLE or seed != DEFAULT_SEED):
        raise HumanCheckError(
            "heldout 블라인드 프로토콜은 --sample 20, --seed 20261020으로 고정됩니다."
        )
    results_path = Path(results)
    output_path = Path(output) if output is not None else results_path / DEFAULT_BLIND_NAME
    secret_path = Path(mapping_path) if mapping_path is not None else results_path / DEFAULT_MAPPING_NAME
    if output_path.resolve() == secret_path.resolve():
        raise HumanCheckError("블라인드 파일과 익명 매핑 파일은 다른 경로여야 합니다.")
    existing = [path for path in (output_path, secret_path) if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise HumanCheckError(f"기존 파일을 덮어쓰지 않습니다: {names} (--force로 덮어쓰기)")

    try:
        rows = load_result_rows(results_path)
        expected_rows = load_expected_rows(dataset_path, split=split)
    except ResultDataError as exc:
        raise HumanCheckError(str(exc)) from exc
    blind_rows, mapping = create_blind_set(
        rows,
        expected_rows=expected_rows,
        sample=sample,
        seed=seed,
        split=split,
    )
    mapping["blind_file"] = output_path.name

    blind_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in blind_rows)
    mapping_text = json.dumps(mapping, ensure_ascii=False, indent=2) + "\n"
    # pair·stage·원본 ID 매핑은 채점자에게 보이지 않게 계정 사용자만 읽게 한다.
    _atomic_write(secret_path, mapping_text, mode=0o600)
    _atomic_write(output_path, blind_text)
    return output_path, secret_path, len(blind_rows)


def _read_blind_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise HumanCheckError(f"블라인드 파일이 없습니다: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HumanCheckError(f"블라인드 JSONL 파싱 오류: {path}:{line_number}: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise HumanCheckError(f"블라인드 행이 개체가 아닙니다: {path}:{line_number}")
        rows.append(row)
    return rows


def _validate_mapping(mapping: Mapping[str, Any]) -> tuple[int, dict[str, Mapping[str, Any]]]:
    if mapping.get("version") != PROTOCOL_VERSION:
        raise HumanCheckError(f"지원하지 않는 블라인드 매핑 version입니다: {mapping.get('version')}")
    if mapping.get("protocol") != "paired_validate_hallucination":
        raise HumanCheckError("블라인드 매핑 protocol이 잘못되었습니다.")
    if mapping.get("sampling") != "proportional_stratified_item_truth":
        raise HumanCheckError("블라인드 매핑 sampling이 잘못되었습니다.")
    split = mapping.get("split")
    if split not in {"dev", "heldout"}:
        raise HumanCheckError("블라인드 매핑 split은 dev 또는 heldout이어야 합니다.")
    seed = mapping.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise HumanCheckError("블라인드 매핑 seed는 0 이상의 정수여야 합니다.")
    stratum_counts = mapping.get("stratum_counts")
    if (
        not isinstance(stratum_counts, Mapping)
        or any(
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for key, value in stratum_counts.items()
        )
    ):
        raise HumanCheckError("블라인드 매핑 stratum_counts가 잘못되었습니다.")
    if mapping.get("stages") != list(STAGES):
        raise HumanCheckError("블라인드 매핑 stages가 잘못되었습니다.")
    sample = mapping.get("sample_pairs")
    if isinstance(sample, bool) or not isinstance(sample, int) or sample <= 0:
        raise HumanCheckError("블라인드 매핑 sample_pairs가 잘못되었습니다.")
    if sum(stratum_counts.values()) != sample:
        raise HumanCheckError("블라인드 매핑 stratum_counts 합계가 sample_pairs와 다릅니다.")
    entries = mapping.get("entries")
    if not isinstance(entries, Mapping) or len(entries) != sample * 2:
        raise HumanCheckError("블라인드 매핑은 정확히 2N개 entries가 필요합니다.")
    selected_ids = mapping.get("selected_result_ids")
    if (
        not isinstance(selected_ids, list)
        or len(selected_ids) != sample
        or any(not isinstance(value, str) or not value for value in selected_ids)
        or len(set(selected_ids)) != sample
    ):
        raise HumanCheckError("블라인드 매핑 selected_result_ids가 잘못되었습니다.")

    pair_entries: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    result_to_pair: dict[str, str] = {}
    normalized: dict[str, Mapping[str, Any]] = {}
    for anonymous_id, raw_entry in entries.items():
        if not isinstance(anonymous_id, str) or not anonymous_id:
            raise HumanCheckError("매핑의 anonymous id는 비어 있지 않은 문자열이어야 합니다.")
        if not isinstance(raw_entry, Mapping):
            raise HumanCheckError(f"매핑 entry가 개체가 아닙니다: {anonymous_id}")
        pair_id = raw_entry.get("pair_id")
        result_id = raw_entry.get("result_id")
        stage = raw_entry.get("stage")
        digest = raw_entry.get("payload_sha256")
        if not isinstance(pair_id, str) or not pair_id:
            raise HumanCheckError(f"매핑 pair_id가 잘못되었습니다: {anonymous_id}")
        if not isinstance(result_id, str) or not result_id:
            raise HumanCheckError(f"매핑 result_id가 잘못되었습니다: {anonymous_id}")
        if stage not in STAGES:
            raise HumanCheckError(f"매핑 stage가 잘못되었습니다: {anonymous_id}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise HumanCheckError(f"매핑 payload SHA-256이 잘못되었습니다: {anonymous_id}")
        previous_pair = result_to_pair.setdefault(result_id, pair_id)
        if previous_pair != pair_id:
            raise HumanCheckError(f"하나의 result_id가 여러 pair에 연결되었습니다: {result_id}")
        pair_entries[pair_id].append(raw_entry)
        normalized[anonymous_id] = raw_entry

    if set(result_to_pair) != set(selected_ids) or len(pair_entries) != sample:
        raise HumanCheckError("매핑의 pair와 selected_result_ids가 일치하지 않습니다.")
    for pair_id, values in pair_entries.items():
        stages = [value.get("stage") for value in values]
        result_ids = {value.get("result_id") for value in values}
        if len(values) != 2 or set(stages) != set(STAGES) or len(result_ids) != 1:
            raise HumanCheckError(f"각 pair에는 동일 ID의 validate 전·후가 하나씩 필요합니다: {pair_id}")
    return sample, normalized


def _validate_scored_row(
    row: Mapping[str, Any],
    *,
    line_number: int,
    entry: Mapping[str, Any],
) -> tuple[bool, Counter[str], str]:
    if set(row) != _ROW_KEYS:
        raise HumanCheckError(
            f"블라인드 {line_number}번 행의 필드가 채점 스키마와 다릅니다."
        )
    anonymous_id = row.get("id")
    if not isinstance(anonymous_id, str) or not anonymous_id:
        raise HumanCheckError(f"블라인드 {line_number}번 행에 익명 id가 없습니다.")
    references = row.get("reference_observations")
    if (
        not isinstance(references, list)
        or not references
        or any(not isinstance(value, str) or not value.strip() for value in references)
    ):
        raise HumanCheckError(f"{anonymous_id}의 reference_observations가 잘못되었습니다.")
    claims = row.get("claims")
    if not isinstance(claims, list) or not claims:
        raise HumanCheckError(f"{anonymous_id}에는 1개 이상의 claim이 필요합니다.")
    claim_ids: set[str] = set()
    counts: Counter[str] = Counter()
    for claim in claims:
        if not isinstance(claim, Mapping) or set(claim) != _CLAIM_KEYS:
            raise HumanCheckError(f"{anonymous_id}의 claim 스키마가 잘못되었습니다.")
        claim_id = claim.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id or claim_id in claim_ids:
            raise HumanCheckError(f"{anonymous_id}에 비어 있거나 중복된 claim_id가 있습니다.")
        claim_ids.add(claim_id)
        if claim.get("source") not in {"summary", "observation"}:
            raise HumanCheckError(f"{anonymous_id}/{claim_id}의 source가 잘못되었습니다.")
        if not isinstance(claim.get("text"), str) or not str(claim["text"]).strip():
            raise HumanCheckError(f"{anonymous_id}/{claim_id}의 text가 비어 있습니다.")
        judgment = claim.get("judgment")
        if judgment not in JUDGMENTS:
            raise HumanCheckError(
                f"{anonymous_id}/{claim_id}의 judgment를 허용값으로 모두 채워야 합니다."
            )
        counts[str(judgment)] += 1
    stage_guess = row.get("stage_guess")
    if stage_guess not in STAGE_GUESSES:
        raise HumanCheckError(
            f"{anonymous_id}의 stage_guess를 before|after|unknown 중 하나로 채워야 합니다."
        )
    if _payload_sha256(row) != entry.get("payload_sha256"):
        raise HumanCheckError(f"{anonymous_id}의 블라인드 payload가 생성 후 변조되었습니다.")
    has_hallucination = any(counts[value] > 0 for value in _HALLUCINATION_JUDGMENTS)
    return has_hallucination, counts, str(stage_guess)


def _exact_mcnemar(before_only: int, after_only: int) -> float:
    discordant = before_only + after_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index) for index in range(min(before_only, after_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def unblind_scores(
    blind_rows: Sequence[Mapping[str, Any]],
    mapping: Mapping[str, Any],
) -> dict[str, Any]:
    """무결성과 완전성을 확인한 뒤 validate 전·후 paired 지표를 계산한다."""

    sample, entries = _validate_mapping(mapping)
    if len(blind_rows) != sample * 2:
        raise HumanCheckError("블라인드 파일은 정확히 2N개 행이어야 합니다.")
    seen: set[str] = set()
    scored_by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    stage_claims: dict[str, Counter[str]] = {stage: Counter() for stage in STAGES}
    guesses = Counter()

    for line_number, row in enumerate(blind_rows, start=1):
        if not isinstance(row, Mapping):
            raise HumanCheckError(f"블라인드 {line_number}번 행이 개체가 아닙니다.")
        anonymous_id = row.get("id")
        if not isinstance(anonymous_id, str) or not anonymous_id:
            raise HumanCheckError(f"블라인드 {line_number}번 행에 익명 id가 없습니다.")
        if anonymous_id in seen:
            raise HumanCheckError(f"중복 anonymous_id입니다: {anonymous_id}")
        seen.add(anonymous_id)
        entry = entries.get(anonymous_id)
        if entry is None:
            raise HumanCheckError(f"매핑에 없는 anonymous_id입니다: {anonymous_id}")
        has_hallucination, counts, stage_guess = _validate_scored_row(
            row,
            line_number=line_number,
            entry=entry,
        )
        pair_id = str(entry["pair_id"])
        stage = str(entry["stage"])
        if stage in scored_by_pair[pair_id]:
            raise HumanCheckError(f"pair stage가 중복되었습니다: {pair_id}/{stage}")
        scored_by_pair[pair_id][stage] = {
            "has_hallucination": has_hallucination,
            "counts": counts,
        }
        stage_claims[stage].update(counts)
        if stage_guess == "unknown":
            guesses["unknown"] += 1
        else:
            guessed_stage = "before_validate" if stage_guess == "before" else "after_validate"
            guesses["correct" if guessed_stage == stage else "incorrect"] += 1

    missing = sorted(set(entries) - seen)
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise HumanCheckError(f"블라인드 파일에 매핑 행이 누락되었습니다: {preview}{suffix}")
    if len(scored_by_pair) != sample or any(
        set(values) != set(STAGES) for values in scored_by_pair.values()
    ):
        raise HumanCheckError("모든 pair의 validate 전·후 채점이 필요합니다.")

    stage_summary: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        hallucinated_reports = sum(
            bool(values[stage]["has_hallucination"])
            for values in scored_by_pair.values()
        )
        low, high = wilson_interval(hallucinated_reports, sample)
        if low is None or high is None:  # sample_pairs는 앞에서 1 이상으로 검증한다.
            raise HumanCheckError("Wilson 신뢰구간을 계산할 표본이 없습니다.")
        counts = stage_claims[stage]
        factual_claims = sum(
            value for key, value in counts.items() if key != "NON_FACTUAL"
        )
        hallucinated_claims = sum(counts[key] for key in _HALLUCINATION_JUDGMENTS)
        stage_summary[stage] = {
            "reports": sample,
            "reports_with_hallucination": hallucinated_reports,
            "report_hallucination_rate": hallucinated_reports / sample,
            "wilson_95_low": low,
            "wilson_95_high": high,
            "claims": sum(counts.values()),
            "factual_claims": factual_claims,
            "hallucinated_claims": hallucinated_claims,
            "uncertain_claims": counts["UNCERTAIN"],
            "claim_hallucination_rate": (
                hallucinated_claims / factual_claims if factual_claims else 0.0
            ),
        }

    paired = Counter()
    for values in scored_by_pair.values():
        before = bool(values["before_validate"]["has_hallucination"])
        after = bool(values["after_validate"]["has_hallucination"])
        if before and after:
            paired["both"] += 1
        elif before:
            paired["before_only"] += 1
        elif after:
            paired["after_only"] += 1
        else:
            paired["neither"] += 1
    mcnemar_p = _exact_mcnemar(paired["before_only"], paired["after_only"])
    before_rate = float(stage_summary["before_validate"]["report_hallucination_rate"])
    after_rate = float(stage_summary["after_validate"]["report_hallucination_rate"])
    threshold_met = after_rate <= 0.05
    decrease_observed = after_rate < before_rate
    significant = mcnemar_p < 0.05
    guessed = guesses["correct"] + guesses["incorrect"]
    split = str(mapping["split"])
    seed = int(mapping["seed"])
    conclusion_allowed = (
        split == "heldout"
        and seed == DEFAULT_SEED
        and sample == DEFAULT_SAMPLE
    )
    descriptive_criteria_met = threshold_met and decrease_observed and significant
    conclusion_scope = (
        "사전 고정 heldout 20쌍의 완전 채점 관측 표본에 한함"
        if conclusion_allowed
        else (
            "결론 금지: split=heldout, seed=20261020, sample_pairs=20인 "
            "사전 고정 프로토콜이 아니므로 기술 통계로만 사용"
        )
    )

    return {
        "sample_pairs": sample,
        "split": split,
        "seed": seed,
        "stages": stage_summary,
        "paired": {
            "neither": paired["neither"],
            "before_only": paired["before_only"],
            "after_only": paired["after_only"],
            "both": paired["both"],
            "discordant": paired["before_only"] + paired["after_only"],
            "mcnemar_exact_p": mcnemar_p,
        },
        "blinding": {
            "correct_guesses": guesses["correct"],
            "incorrect_guesses": guesses["incorrect"],
            "unknown_guesses": guesses["unknown"],
            "guess_accuracy_known": guesses["correct"] / guessed if guessed else None,
        },
        "h3": {
            "observed_after_at_most_5pct": threshold_met,
            "observed_decrease": decrease_observed,
            "mcnemar_significant": significant,
            "descriptive_criteria_met": descriptive_criteria_met,
            "observed_support": conclusion_allowed and descriptive_criteria_met,
        },
        "conclusion_allowed": conclusion_allowed,
        "conclusion_scope": conclusion_scope,
    }


def render_unblind_summary(summary: Mapping[str, Any]) -> str:
    stages = summary["stages"]
    paired = summary["paired"]
    blinding = summary["blinding"]
    h3 = summary["h3"]
    lines = [
        "# 블라인드 validate 전·후 환각 채점 집계",
        "",
        f"> 결론 가능 여부: **{'가능' if summary['conclusion_allowed'] else '금지'}** "
        f"({summary['conclusion_scope']})",
        "",
        "| stage | 보고서 | 환각 포함 | 관측률 | Wilson 95% CI | "
        "환각 claim / 사실 claim | UNCERTAIN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = {"before_validate": "validate 전", "after_validate": "validate 후"}
    for stage in STAGES:
        values = stages[stage]
        lines.append(
            f"| {labels[stage]} | {values['reports']} | "
            f"{values['reports_with_hallucination']} | "
            f"{float(values['report_hallucination_rate']):.3f} | "
            f"[{float(values['wilson_95_low']):.3f}, "
            f"{float(values['wilson_95_high']):.3f}] | "
            f"{values['hallucinated_claims']} / {values['factual_claims']} | "
            f"{values['uncertain_claims']} |"
        )
    lines.extend(
        [
            "",
            "## 보고서 수준 짝지은 표",
            "",
            "| | validate 후 무환각 | validate 후 환각 |",
            "| --- | ---: | ---: |",
            f"| validate 전 무환각 | {paired['neither']} | {paired['after_only']} |",
            f"| validate 전 환각 | {paired['before_only']} | {paired['both']} |",
            "",
            f"- exact McNemar p: `{float(paired['mcnemar_exact_p']):.6f}`",
            f"- H3 관측 5% 이하: `{'충족' if h3['observed_after_at_most_5pct'] else '미충족'}`",
            f"- H3 관측 감소: `{'충족' if h3['observed_decrease'] else '미충족'}`",
            f"- H3 paired 유의성: `{'충족' if h3['mcnemar_significant'] else '미충족'}`",
            "- H3 기술 통계 기준: "
            f"`{'충족' if h3['descriptive_criteria_met'] else '미충족'}`",
            "- H3 최종 결론: "
            f"`{'지지' if h3['observed_support'] else '결론 금지' if not summary['conclusion_allowed'] else '미입증'}`",
            "",
            "## 블라인드 점검",
            "",
            f"- stage 추측: 정답 {blinding['correct_guesses']}, 오답 "
            f"{blinding['incorrect_guesses']}, 모름 {blinding['unknown_guesses']}",
            "",
            "> UNCERTAIN은 보수적으로 환각에 포함했습니다.",
            "> 1인 채점이며, 이 결과와 Wilson 구간은 선택된 관측 표본에 한합니다. "
            "모집단 환각률이나 채점자 간 일치도로 일반화하지 않습니다.",
            f"> {summary['sample_pairs']}쌍에서 관측률의 한 건 해상도는 "
            f"{100 / int(summary['sample_pairs']):.2f}%p입니다.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_unblind(blind_path: str | Path, mapping_path: str | Path) -> str:
    blind = _read_blind_rows(Path(blind_path))
    secret_path = Path(mapping_path)
    if not secret_path.is_file():
        raise HumanCheckError(f"매핑 파일이 없습니다: {secret_path}")
    try:
        mapping = json.loads(secret_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HumanCheckError(f"매핑 JSON 파싱 오류: {secret_path}: {exc.msg}") from exc
    if not isinstance(mapping, dict):
        raise HumanCheckError("매핑 JSON은 개체여야 합니다.")
    return render_unblind_summary(unblind_scores(blind, mapping))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="동일 A 실행의 validate 전·후 보고서를 블라인드 paired 채점합니다."
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", default="heldout")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--unblind", action="store_true")
    parser.add_argument("--force", action="store_true", help="기존 블라인드/매핑 파일을 덮어씁니다.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    blind_path = args.output or args.results / DEFAULT_BLIND_NAME
    mapping_path = args.mapping or args.results / DEFAULT_MAPPING_NAME
    try:
        if args.unblind:
            markdown = run_unblind(blind_path, mapping_path)
            print(markdown, end="")
        else:
            output, secret, count = write_blind_set(
                args.results,
                output=blind_path,
                mapping_path=mapping_path,
                dataset_path=args.dataset,
                sample=args.sample,
                seed=args.seed,
                split=args.split,
                force=args.force,
            )
            print(f"블라인드 {count}건({count // 2}쌍) 생성: {output}")
            print(f"pair·stage 매핑 보관(0600): {secret}")
            print(
                "각 claim judgment: SUPPORTED | PARAPHRASE_SUPPORTED | "
                "HALLUCINATED | NON_FACTUAL | UNCERTAIN"
            )
            print("각 행 stage_guess: before | after | unknown")
    except (HumanCheckError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
