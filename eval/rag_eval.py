"""R6 dev RAG top-k 검색 품질을 두 임베딩 모델로 비교한다.

이미지를 다시 열거나 VLM을 재호출하지 않는다. 완전한 dev A 실행에 기록된
``rag_query_text``만 production Spec-Judge와 동일한 검색 질의로 사용한다.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rag import FakeEmbedder, OllamaEmbedder, OpenAIEmbedder, SpecIndex
from eval.provenance import (
    ProvenanceError,
    atomic_write_json,
    canonical_sha256,
    collect_repository_provenance,
    new_run_id,
    sha256_file,
    validate_run_id,
    verify_run_manifest,
)


DATASET_PATH = ROOT / "eval" / "dataset.json"
TYPE_MAP_PATH = ROOT / "eval" / "type_map.json"
SPEC_DIR = ROOT / "data" / "specs"
DEFAULT_RESULTS_DIR = ROOT / "eval" / "results"
TOP_K = 3
SCHEMA_VERSION = 1
_USAGE_FIELDS = (
    "tokens",
    "estimated_usd",
    "compute_proxy_usd",
    "successful_provider_requests",
    "unknown_token_usage_successes",
    "observed_model_response_count",
    "unobserved_model_successes",
)


class RagEvaluationError(ValueError):
    """RAG 비교 입력이나 산출물 계약을 위반한 오류."""


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RagEvaluationError(f"JSON을 읽을 수 없습니다: {path}") from exc
    if not isinstance(value, dict):
        raise RagEvaluationError(f"JSON 최상위 값이 개체가 아닙니다: {path}")
    return value


def load_dev_roster(
    dataset_path: str | Path = DATASET_PATH,
) -> tuple[int, list[dict[str, Any]]]:
    """고정 dataset에서 dev roster만 읽고 heldout 행은 반환하지 않는다."""
    payload = _read_json_object(Path(dataset_path))
    seed = payload.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RagEvaluationError("dataset seed가 유효한 0 이상의 정수가 아닙니다.")
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise RagEvaluationError("dataset samples가 배열이 아닙니다.")
    roster = [dict(row) for row in samples if isinstance(row, dict) and row.get("split") == "dev"]
    if not roster:
        raise RagEvaluationError("dev 평가 roster가 비어 있습니다.")
    ids = [row.get("id") for row in roster]
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise RagEvaluationError("dev roster에 유효하지 않은 id가 있습니다.")
    if len(ids) != len(set(ids)):
        raise RagEvaluationError("dev roster에 중복 id가 있습니다.")
    return seed, roster


def canonical_defect_type(
    *, item: str, label: str, type_map: Mapping[str, Any]
) -> str:
    """dataset 라벨을 type_map의 단일 canonical 결함 유형으로 해석한다."""
    item_types = type_map.get(item)
    if not isinstance(item_types, Mapping):
        raise RagEvaluationError(f"type_map에 품목이 없습니다: {item}")
    matched: list[str] = []
    for canonical, aliases in item_types.items():
        if not isinstance(canonical, str) or not isinstance(aliases, list):
            raise RagEvaluationError(f"type_map 형식이 올바르지 않습니다: {item}")
        values = {canonical, *(value for value in aliases if isinstance(value, str))}
        if label in values:
            matched.append(canonical)
    if len(matched) != 1:
        raise RagEvaluationError(
            f"dataset 라벨을 단일 결함 유형으로 해석할 수 없습니다: {item}/{label}"
        )
    return matched[0]


def validate_source_rows(
    rows: Sequence[Mapping[str, Any]],
    roster: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """A 결과를 canonical dev roster에 결속하고 exact RAG query를 강제한다."""
    expected = {str(row["id"]): row for row in roster}
    seen: dict[str, dict[str, Any]] = {}
    for number, raw in enumerate(rows, start=1):
        if not isinstance(raw, Mapping):
            raise RagEvaluationError(f"A 결과 {number}행이 개체가 아닙니다.")
        row = dict(raw)
        row_id = row.get("id")
        if not isinstance(row_id, str) or row_id not in expected:
            raise RagEvaluationError(f"A 결과에 roster 밖 id가 있습니다: {row_id}")
        if row_id in seen:
            raise RagEvaluationError(f"A 결과에 중복 id가 있습니다: {row_id}")
        truth = expected[row_id]
        for key in ("item", "label", "is_defect", "split"):
            if row.get(key) != truth.get(key):
                raise RagEvaluationError(f"A 결과의 {key}가 dataset과 다릅니다: {row_id}")
        if row.get("condition") != "A":
            raise RagEvaluationError(f"RAG 비교에는 A 조건 결과만 사용할 수 있습니다: {row_id}")
        # observations_text에는 Cross-Checker 문장이 섞이므로 대체 입력으로 허용하지 않는다.
        if "rag_query_text" not in row:
            raise RagEvaluationError(
                f"production 검색 질의 rag_query_text가 없습니다: {row_id}"
            )
        query = row.get("rag_query_text")
        if query is not None and not isinstance(query, str):
            raise RagEvaluationError(f"rag_query_text가 문자열 또는 null이 아닙니다: {row_id}")
        seen[row_id] = row
    missing = sorted(set(expected) - set(seen))
    if missing:
        raise RagEvaluationError("A 결과가 완전한 dev roster가 아닙니다: " + ", ".join(missing))
    return [seen[str(sample["id"])] for sample in roster]


def load_source_run(
    run_dir: str | Path,
    *,
    allow_dry_run: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    """검증된 dev A suite와 exact-query 행을 불러온다."""
    directory = Path(run_dir).resolve()
    try:
        manifest = verify_run_manifest(
            directory,
            reject_dry_run=not allow_dry_run,
            reject_fake=not allow_dry_run,
            require_complete=True,
            validate_rows=True,
            repo_root=ROOT,
            verify_inputs=True,
        )
    except ProvenanceError as exc:
        raise RagEvaluationError(f"source run 검증 실패: {exc}") from exc
    if manifest.get("split") != "dev":
        raise RagEvaluationError("R6 RAG 평가는 dev suite만 허용합니다.")
    record = manifest.get("conditions", {}).get("A")
    if not isinstance(record, Mapping) or record.get("status") != "complete":
        raise RagEvaluationError("source run에 완전한 A 조건이 없습니다.")
    result_record = record.get("result")
    if not isinstance(result_record, Mapping) or not isinstance(result_record.get("path"), str):
        raise RagEvaluationError("source run의 A 결과 경로가 올바르지 않습니다.")
    result_path = directory / str(result_record["path"])
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(result_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RagEvaluationError(f"A 결과 {line_number}행이 개체가 아닙니다.")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise RagEvaluationError(f"A 결과 JSONL을 읽을 수 없습니다: {result_path}") from exc
    return manifest, rows, result_path


def parse_embedding_spec(value: str) -> tuple[str, str]:
    """CLI의 ``backend:model`` 값을 검증한다."""
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise RagEvaluationError("임베딩 지정은 앞뒤 공백이 없는 문자열이어야 합니다.")
    if ":" in value:
        backend, model = value.split(":", 1)
    else:
        backend, model = value, ""
    backend = backend.lower()
    if backend == "fake":
        model = model or "char-2gram-512"
    elif backend == "openai":
        model = model or "text-embedding-3-small"
    elif backend == "ollama":
        model = model or "nomic-embed-text"
    else:
        raise RagEvaluationError(f"지원하지 않는 임베딩 backend입니다: {backend}")
    if not model.strip() or any(char.isspace() for char in model):
        raise RagEvaluationError("임베딩 모델명이 비어 있거나 공백을 포함합니다.")
    return backend, model


def validate_embedding_specs(values: Sequence[str], *, dry_run: bool) -> list[tuple[str, str]]:
    """비교 모델 두 개를 고정하고 실측에서 fake를 차단한다."""
    if len(values) != 2:
        raise RagEvaluationError("RAG 비교에는 임베딩 모델을 정확히 2개 지정해야 합니다.")
    parsed = [parse_embedding_spec(value) for value in values]
    if len(set(parsed)) != 2:
        raise RagEvaluationError("서로 다른 임베딩 모델 2개를 지정해야 합니다.")
    if not dry_run and any(backend == "fake" for backend, _ in parsed):
        raise RagEvaluationError("실측 RAG 비교는 fake 임베딩을 허용하지 않습니다.")
    return parsed


def make_named_embedder(backend: str, model: str):
    """모델명이 환경변수에 묻히지 않도록 명시적으로 backend를 만든다."""
    if backend == "fake":
        embedder = FakeEmbedder()
        embedder.name = f"fake:{model}"
        return embedder
    if backend == "openai":
        return OpenAIEmbedder(model=model)
    if backend == "ollama":
        return OllamaEmbedder(model=model)
    raise RagEvaluationError(f"지원하지 않는 임베딩 backend입니다: {backend}")


def _zero_usage() -> dict[str, int | float | list[str]]:
    return {
        "tokens": 0,
        "estimated_usd": 0.0,
        "compute_proxy_usd": 0.0,
        "successful_provider_requests": 0,
        "unknown_token_usage_successes": 0,
        "observed_model_response_count": 0,
        "unobserved_model_successes": 0,
        "observed_model_ids": [],
    }


def _add_last_usage(total: dict[str, Any], embedder: Any) -> None:
    usage = getattr(embedder, "last_usage", None)
    if not isinstance(usage, Mapping):
        raise RagEvaluationError("임베딩 backend가 usage를 제공하지 않았습니다.")
    for key in _USAGE_FIELDS:
        value = usage.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RagEvaluationError(f"임베딩 usage {key}가 수치가 아닙니다.")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise RagEvaluationError(f"임베딩 usage {key}가 유한한 0 이상이 아닙니다.")
        total[key] = total[key] + value
    observed = usage.get("observed_model")
    if isinstance(observed, str) and observed.strip():
        models = total["observed_model_ids"]
        if observed not in models:
            models.append(observed)


def evaluate_model(
    *,
    rows: Sequence[Mapping[str, Any]],
    roster: Sequence[Mapping[str, Any]],
    type_map: Mapping[str, Any],
    spec_dir: str | Path,
    embedder: Any,
    backend: str,
    model: str,
    top_k: int = TOP_K,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """한 임베딩 모델의 결함 유형 hit@k와 행별 근거를 계산한다."""
    if top_k != TOP_K:
        raise RagEvaluationError(f"R6 RAG 비교의 top-k는 {TOP_K}으로 고정됩니다.")
    if len(rows) != len(roster):
        raise RagEvaluationError("A 결과와 dev roster의 행 수가 다릅니다.")
    index = SpecIndex.from_dir(spec_dir, embedder)
    usage = _zero_usage()
    _add_last_usage(usage, embedder)
    hit_count = 0
    query_count = 0
    source_error_count = 0
    defect_expected_n = 0
    defect_query_count = 0
    defect_hit_count = 0
    details: dict[str, dict[str, Any]] = {}
    for row, truth in zip(rows, roster, strict=True):
        row_id = str(truth["id"])
        expected_type = canonical_defect_type(
            item=str(truth["item"]),
            label=str(truth["label"]),
            type_map=type_map,
        )
        error = row.get("error")
        query = row.get("rag_query_text")
        usable = not error and isinstance(query, str) and bool(query.strip())
        is_defect = truth.get("is_defect") is True
        if is_defect:
            defect_expected_n += 1
        hits: list[dict[str, Any]] = []
        if usable:
            query_count += 1
            if is_defect:
                defect_query_count += 1
            hits = index.search(query.strip(), item=str(truth["item"]), k=top_k)
            _add_last_usage(usage, embedder)
        else:
            source_error_count += 1
        hit = any(value.get("defect_type") == expected_type for value in hits)
        hit_count += int(hit)
        if is_defect:
            defect_hit_count += int(hit)
        details[row_id] = {
            "expected_defect_type": expected_type,
            "is_defect": is_defect,
            "query_available": usable,
            "hit_at_3": hit,
            "hits": [
                {
                    "rank": rank,
                    "clause_id": value.get("clause_id"),
                    "defect_type": value.get("defect_type"),
                    "score": value.get("score"),
                }
                for rank, value in enumerate(hits, start=1)
            ],
        }
    expected_n = len(roster)
    if defect_expected_n <= 0:
        raise RagEvaluationError("dev roster에 결함 표본이 없습니다.")
    model_id = f"{backend}:{model}"
    summary = {
        "model_id": model_id,
        "backend": backend,
        "requested_model": model,
        "expected_n": expected_n,
        "query_count": query_count,
        "missing_or_error_query_count": source_error_count,
        "query_coverage": query_count / expected_n,
        "hit_count": hit_count,
        # dev 전체 통계는 정상(good) 검색을 포함한 보조 지표다.
        "end_to_end_hit_at_3": hit_count / expected_n,
        "selective_hit_at_3": hit_count / query_count if query_count else None,
        # 정상 표본을 성공으로 세어 결함 유형 검색 품질을 부풀리지 않는 주 지표다.
        # 결함 query 누락·오류도 고정 분모에 남겨 실패로 처리한다.
        "defect_expected_n": defect_expected_n,
        "defect_query_count": defect_query_count,
        "defect_missing_or_error_query_count": defect_expected_n - defect_query_count,
        "defect_query_coverage": defect_query_count / defect_expected_n,
        "defect_hit_count": defect_hit_count,
        "defect_end_to_end_hit_at_3": defect_hit_count / defect_expected_n,
        "defect_selective_hit_at_3": (
            defect_hit_count / defect_query_count if defect_query_count else None
        ),
        "usage": usage,
    }
    return summary, details


def build_payload(
    *,
    source_manifest: Mapping[str, Any],
    source_result_path: Path,
    rows: Sequence[Mapping[str, Any]],
    roster: Sequence[Mapping[str, Any]],
    seed: int,
    type_map: Mapping[str, Any],
    model_specs: Sequence[tuple[str, str]],
    embedders: Sequence[Any],
    provenance: Mapping[str, Any],
    dry_run: bool,
    run_id: str,
    dataset_path: Path = DATASET_PATH,
    type_map_path: Path = TYPE_MAP_PATH,
    spec_dir: Path = SPEC_DIR,
) -> dict[str, Any]:
    """두 모델 비교 payload를 동일 query 순서로 구성한다."""
    if set(provenance) != {"repository", "artifacts", "runtime"}:
        raise RagEvaluationError(
            "RAG 결과 provenance에 repository·artifacts·runtime이 필요합니다."
        )
    source_records = provenance.get("artifacts", {}).get("source")
    if not isinstance(source_records, list) or "eval/rag_eval.py" not in {
        record.get("path") for record in source_records if isinstance(record, Mapping)
    }:
        raise RagEvaluationError("RAG 평가 코드 hash가 provenance에 없습니다.")
    model_summaries: list[dict[str, Any]] = []
    details_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for (backend, model), embedder in zip(model_specs, embedders, strict=True):
        summary, details = evaluate_model(
            rows=rows,
            roster=roster,
            type_map=type_map,
            spec_dir=spec_dir,
            embedder=embedder,
            backend=backend,
            model=model,
        )
        model_summaries.append(summary)
        details_by_model[summary["model_id"]] = details
    output_rows: list[dict[str, Any]] = []
    for row, truth in zip(rows, roster, strict=True):
        row_id = str(truth["id"])
        query = row.get("rag_query_text")
        output_rows.append({
            "id": row_id,
            "item": truth["item"],
            "label": truth["label"],
            "is_defect": truth["is_defect"],
            "split": "dev",
            "source_error": row.get("error"),
            "rag_query_text": query,
            "rag_query_sha256": canonical_sha256(query),
            "models": {
                model_id: details[row_id]
                for model_id, details in details_by_model.items()
            },
        })
    specs = [
        {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}
        for path in sorted(spec_dir.glob("*.md"))
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "dry_run_not_real" if dry_run else "complete",
        "measurement": "rag_defect_type_end_to_end_hit_at_3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": "dev",
        "seed": seed,
        "top_k": TOP_K,
        "dry_run": dry_run,
        "warning": (
            "FAKE/DRYRUN 결과이며 실제 검색 성능으로 사용할 수 없습니다."
            if dry_run
            else None
        ),
        "source": {
            "run_id": source_manifest.get("run_id"),
            "manifest_sha256": source_manifest.get("manifest_sha256"),
            "manifest_file_sha256": sha256_file(source_result_path.parent / "manifest.json"),
            "result_path": source_result_path.name,
            "result_sha256": sha256_file(source_result_path),
            "condition": "A",
        },
        "inputs": {
            "dataset": {
                "path": dataset_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(dataset_path),
            },
            "type_map": {
                "path": type_map_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(type_map_path),
            },
            "specs": specs,
        },
        "provenance": dict(provenance),
        "provenance_sha256": canonical_sha256(provenance),
        "models": model_summaries,
        "rows": output_rows,
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    """발표 표에 옮길 수 있는 비교 요약을 만든다."""
    lines = ["# R6 RAG top-3 검색 품질", ""]
    if payload.get("dry_run"):
        lines.extend([
            "> **DRYRUN — 실제 검색 성능으로 사용할 수 없음**",
            "",
        ])
    lines.extend([
        f"- split: `{payload.get('split')}`",
        f"- source run: `{payload.get('source', {}).get('run_id')}`",
        f"- top-k: `{payload.get('top_k')}`",
        "- 주 지표는 결함 표본만을 고정 분모로 삼고 오류·빈 query를 실패로 남긴 `defect_end_to_end_hit_at_3`이다.",
        "- dev 전체 지표는 정상(good) 검색을 포함하므로 보조 통계로만 사용한다.",
        "",
        "| 임베딩 모델 | 결함 N | 결함 query N | 결함 coverage | 결함 hit N | 결함 end-to-end hit@3 | 결함 selective hit@3 | dev 전체 hit@3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for model in payload.get("models", []):
        selective = model.get("defect_selective_hit_at_3")
        selective_text = "N/A" if selective is None else f"{selective:.4f}"
        lines.append(
            f"| {model['model_id']} | {model['defect_expected_n']} | "
            f"{model['defect_query_count']} | {model['defect_query_coverage']:.4f} | "
            f"{model['defect_hit_count']} | {model['defect_end_to_end_hit_at_3']:.4f} | "
            f"{selective_text} | {model['end_to_end_hit_at_3']:.4f} |"
        )
    lines.extend([
        "",
        "두 모델 비교는 검색 임베딩 선택에 대한 dev 기술 통계다. heldout 가설 결과가 아니다.",
        "",
    ])
    return "\n".join(lines)


def write_outputs(output_dir: str | Path, payload: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    """새 디렉터리에만 JSON·Markdown·hash manifest를 기록한다."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    result_path = directory / "rag_eval.json"
    markdown_path = directory / "rag_eval.md"
    atomic_write_json(result_path, payload)
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": payload.get("run_id"),
        "status": payload.get("status"),
        "dry_run": payload.get("dry_run"),
        "result": {"path": result_path.name, "sha256": sha256_file(result_path)},
        "report": {"path": markdown_path.name, "sha256": sha256_file(markdown_path)},
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = directory / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    return result_path, markdown_path, manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="완전한 dev A 실행의 production RAG query로 임베딩 모델 2종을 비교한다."
    )
    parser.add_argument("--source-run", required=True, help="완료된 dev B0/A run 디렉터리")
    parser.add_argument(
        "--embedding",
        action="append",
        required=True,
        help="backend:model 형식. 정확히 두 번 지정",
    )
    parser.add_argument("--run-id", help="RAG 비교 고유 식별자")
    parser.add_argument("--out", default=str(DEFAULT_RESULTS_DIR), help="새 결과 디렉터리의 부모")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fake/source dry-run을 허용하되 산출물을 실측 불가로 표시",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = validate_run_id(args.run_id or f"rag-{new_run_id()}")
    model_specs = validate_embedding_specs(args.embedding, dry_run=bool(args.dry_run))
    seed, roster = load_dev_roster()
    source_manifest, raw_rows, source_result_path = load_source_run(
        args.source_run,
        allow_dry_run=bool(args.dry_run),
    )
    rows = validate_source_rows(raw_rows, roster)
    type_map = _read_json_object(TYPE_MAP_PATH)
    # type_map과 모든 정답을 provider 호출 전에 검증한다.
    for sample in roster:
        canonical_defect_type(
            item=str(sample["item"]),
            label=str(sample["label"]),
            type_map=type_map,
        )
    # 외부 호출 전에 Git·코드·기준서 hash 수집이 실패하면 중단한다.
    provenance = collect_repository_provenance(
        ROOT,
        dataset_path=DATASET_PATH,
        type_map_path=TYPE_MAP_PATH,
        specs_dir=SPEC_DIR,
        evaluation_images=(),
        reference_images=(),
    )
    embedders = [make_named_embedder(*spec) for spec in model_specs]
    payload = build_payload(
        source_manifest=source_manifest,
        source_result_path=source_result_path,
        rows=rows,
        roster=roster,
        seed=seed,
        type_map=type_map,
        model_specs=model_specs,
        embedders=embedders,
        provenance=provenance,
        dry_run=bool(args.dry_run),
        run_id=run_id,
    )
    output_dir = Path(args.out) / run_id
    paths = write_outputs(output_dir, payload)
    print(f"RAG 비교 완료: {output_dir}")
    for model in payload["models"]:
        print(
            f"  {model['model_id']}: "
            f"{model['defect_hit_count']}/{model['defect_expected_n']} "
            f"defect end-to-end hit@3={model['defect_end_to_end_hit_at_3']:.4f}"
        )
    print("  " + ", ".join(path.name for path in paths))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RagEvaluationError, ProvenanceError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
