"""관찰 근거와 보고서를 블라인드로 대조할 채점세트를 만든다."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.report import DEFAULT_RESULTS_DIR, ResultDataError, load_result_rows  # noqa: E402


DEFAULT_SEED = 20261020
DEFAULT_BLIND_NAME = "human_check_blind.jsonl"
DEFAULT_MAPPING_NAME = "human_check_mapping.json"


class HumanCheckError(ValueError):
    """블라인드 채점의 완전성을 위반한 오류."""


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


def _observation_texts(report: Mapping[str, Any]) -> list[str]:
    observations = report.get("observations", [])
    if not isinstance(observations, list):
        return []
    texts: list[str] = []
    for observation in observations:
        if isinstance(observation, Mapping):
            text = observation.get("text")
        else:
            text = observation
        if isinstance(text, str) and text.strip():
            texts.append(text)
    return texts


def _eligible_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    condition: str,
) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in rows:
        if source.get("split") != split or source.get("condition") != condition:
            continue
        if source.get("error"):
            continue
        sample_id = str(source.get("id", "")).strip()
        report = source.get("report")
        if not sample_id or not isinstance(report, Mapping):
            continue
        if sample_id in seen:
            raise HumanCheckError(f"{condition}/{split}에 중복 샘플 id가 있습니다: {sample_id}")
        seen.add(sample_id)
        eligible.append(dict(source))
    return sorted(eligible, key=lambda row: str(row["id"]))


def create_blind_set(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample: int = 15,
    seed: int = DEFAULT_SEED,
    split: str = "heldout",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """B0·A에서 각각 sample건을 뽑아 조건을 가린 채 섞는다."""

    if sample <= 0:
        raise HumanCheckError("--sample은 1 이상이어야 합니다.")

    generator = random.Random(seed)
    selected: list[tuple[str, dict[str, Any]]] = []
    for condition in ("B0", "A"):
        candidates = _eligible_rows(rows, split=split, condition=condition)
        if len(candidates) < sample:
            raise HumanCheckError(
                f"{condition}/{split} 채점 가능 결과가 부족합니다: "
                f"필요 {sample}, 가능 {len(candidates)}"
            )
        selected.extend((condition, row) for row in generator.sample(candidates, sample))
    generator.shuffle(selected)

    blind_rows: list[dict[str, Any]] = []
    entries: dict[str, dict[str, str]] = {}
    for index, (condition, row) in enumerate(selected, start=1):
        anonymous_id = f"HC-{index:04d}"
        report = row["report"]
        blind_rows.append(
            {
                "id": anonymous_id,
                "observations_text": row.get("observations_text", []),
                "report": {
                    "summary": report.get("summary", ""),
                    # source_call_id는 조건을 유추할 수 있어 텍스트만 준다.
                    "observations": _observation_texts(report),
                },
                "hallucinated_sentences": None,
            }
        )
        entries[anonymous_id] = {
            "condition": condition,
            "result_id": str(row["id"]),
        }

    mapping = {
        "version": 1,
        "seed": seed,
        "split": split,
        "sample_per_condition": sample,
        "conditions": ["B0", "A"],
        "entries": entries,
    }
    return blind_rows, mapping


def write_blind_set(
    results: str | Path,
    *,
    output: str | Path | None = None,
    mapping_path: str | Path | None = None,
    sample: int = 15,
    seed: int = DEFAULT_SEED,
    force: bool = False,
) -> tuple[Path, Path, int]:
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
    except ResultDataError as exc:
        raise HumanCheckError(str(exc)) from exc
    blind_rows, mapping = create_blind_set(rows, sample=sample, seed=seed)
    mapping["blind_file"] = output_path.name

    blind_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in blind_rows)
    mapping_text = json.dumps(mapping, ensure_ascii=False, indent=2) + "\n"
    # 조건 매핑은 격리해 두고 계정 사용자만 읽도록 제한한다.
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


def unblind_scores(
    blind_rows: Sequence[Mapping[str, Any]],
    mapping: Mapping[str, Any],
) -> dict[str, dict[str, float | int]]:
    """채점자가 넣은 수를 조건별로 합산한다."""

    entries = mapping.get("entries")
    if not isinstance(entries, Mapping) or not entries:
        raise HumanCheckError("매핑 파일에 entries가 없거나 잘못되었습니다.")

    scores: dict[str, list[int]] = defaultdict(list)
    seen: set[str] = set()
    for line_number, row in enumerate(blind_rows, start=1):
        # 초기 파일의 anonymous_id 필드도 하위 호환성을 위해 읽는다.
        anonymous_id = str(row.get("id", row.get("anonymous_id", ""))).strip()
        if not anonymous_id:
            raise HumanCheckError(f"블라인드 {line_number}번 행에 익명 id가 없습니다.")
        if anonymous_id in seen:
            raise HumanCheckError(f"중복 anonymous_id입니다: {anonymous_id}")
        seen.add(anonymous_id)
        if anonymous_id not in entries:
            raise HumanCheckError(f"매핑에 없는 anonymous_id입니다: {anonymous_id}")
        score = row.get("hallucinated_sentences")
        if isinstance(score, bool) or not isinstance(score, int) or score < 0:
            raise HumanCheckError(
                f"{anonymous_id}의 hallucinated_sentences를 0 이상 정수로 채워야 합니다."
            )
        entry = entries[anonymous_id]
        if not isinstance(entry, Mapping) or entry.get("condition") not in {"B0", "A"}:
            raise HumanCheckError(f"매핑의 조건 정보가 잘못되었습니다: {anonymous_id}")
        scores[str(entry["condition"])].append(score)

    missing = sorted(set(entries) - seen)
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise HumanCheckError(f"블라인드 파일에 매핑 행이 누락되었습니다: {preview}{suffix}")

    summary: dict[str, dict[str, float | int]] = {}
    for condition in ("B0", "A"):
        values = scores.get(condition, [])
        if not values:
            raise HumanCheckError(f"채점된 {condition} 행이 없습니다.")
        total = sum(values)
        with_hallucination = sum(value > 0 for value in values)
        summary[condition] = {
            "reports": len(values),
            "hallucinated_sentences": total,
            "mean_per_report": total / len(values),
            "reports_with_hallucination": with_hallucination,
            "report_hallucination_rate": with_hallucination / len(values),
        }
    return summary


def render_unblind_summary(summary: Mapping[str, Mapping[str, float | int]]) -> str:
    lines = [
        "# 블라인드 환각 채점 집계",
        "",
        "| 조건 | 채점 보고서 | 환각 문장 | 보고서당 평균 | 환각 포함 보고서 | 환각 포함률 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for condition in ("B0", "A"):
        values = summary[condition]
        lines.append(
            "| "
            + " | ".join(
                [
                    condition,
                    str(values["reports"]),
                    str(values["hallucinated_sentences"]),
                    f"{float(values['mean_per_report']):.3f}",
                    str(values["reports_with_hallucination"]),
                    f"{float(values['report_hallucination_rate']):.3f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "> 1인 채점입니다. 환각 포함률은 환각 문장이 1개 이상인 보고서 비율입니다."])
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
    parser = argparse.ArgumentParser(description="B0/A 결과를 사람이 블라인드 채점하도록 준비합니다.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--sample", type=int, default=15)
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
                sample=args.sample,
                seed=args.seed,
                force=args.force,
            )
            print(f"블라인드 {count}건 생성: {output}")
            print(f"조건 매핑 보관(0600): {secret}")
    except (HumanCheckError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
