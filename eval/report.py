"""평가 결과 JSONL을 모아 마크다운 표로 정리한다.

이 모듈은 각 행의 ``condition``과 ``split`` 필드로 조건을 선별하며,
dry-run 표시가 있는 파일은 디렉터리 실측 집계에서 제외한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.evaluators import (  # noqa: E402
    binary_metrics,
    citation_fidelity,
    confidence_flag_validity,
    cost_latency,
    mcnemar,
    type_metrics,
)


DEFAULT_RESULTS_DIR = ROOT / "eval" / "results"
DEFAULT_TYPE_MAP = ROOT / "eval" / "type_map.json"
_NON_RESULT_PREFIXES = ("human_check_",)
_DRY_RUN_MARKER = "_DRYRUN"


class ResultDataError(ValueError):
    """결과 파일로 보고서를 만들 수 없음을 나타낸다."""


def parse_conditions(value: str) -> list[str]:
    """쉼표로 구분한 조건 목록을 순서를 유지하며 파싱한다."""

    conditions: list[str] = []
    for part in value.split(","):
        condition = part.strip()
        if condition and condition not in conditions:
            conditions.append(condition)
    if not conditions:
        raise ResultDataError("--conditions에 하나 이상의 조건을 지정해야 합니다.")
    return conditions


def _result_files(results: Path) -> list[Path]:
    if not results.exists():
        raise ResultDataError(f"결과 경로가 없습니다: {results}")
    if results.is_file():
        if results.suffix != ".jsonl":
            raise ResultDataError(f"JSONL 파일이 아닙니다: {results}")
        return [results]
    if not results.is_dir():
        raise ResultDataError(f"결과 경로가 디렉터리가 아닙니다: {results}")

    files = [
        path
        for path in sorted(results.glob("*.jsonl"))
        if not path.name.startswith(_NON_RESULT_PREFIXES)
        and _DRY_RUN_MARKER not in path.stem
    ]
    if not files:
        raise ResultDataError(f"결과 JSONL 파일이 없습니다: {results}")
    return files


def load_result_rows(results: str | Path) -> list[dict[str, Any]]:
    """결과 디렉터리의 JSONL을 읽고 행 구조를 검증한다.

    파일명이 바뀌어도 행 내용으로 조건을 식별한다. 형식이 잘못되면
    해당 파일과 줄 번호를 포함한 오류로 집계를 중단한다.
    """

    rows: list[dict[str, Any]] = []
    for path in _result_files(Path(results)):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ResultDataError(f"결과 파일을 읽을 수 없습니다: {path}: {exc}") from exc

        for line_number, raw_line in enumerate(lines, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ResultDataError(
                    f"JSONL 파싱 오류: {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ResultDataError(
                    f"JSONL 행이 개체가 아닙니다: {path}:{line_number}"
                )
            missing = [key for key in ("condition", "split") if not row.get(key)]
            if missing:
                joined = ", ".join(missing)
                raise ResultDataError(
                    f"결과 행의 필드가 누락되었습니다: {path}:{line_number}: {joined}"
                )
            rows.append(row)

    if not rows:
        raise ResultDataError("결과 JSONL에 유효한 행이 없습니다.")
    return rows


def select_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    split: str,
    conditions: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    """지정한 split·조건만 골라 중복 ID 없이 결과를 모은다."""

    grouped = {condition: [] for condition in conditions}
    seen: dict[str, set[str]] = defaultdict(set)
    for raw_row in rows:
        condition = str(raw_row.get("condition", ""))
        if raw_row.get("split") != split or condition not in grouped:
            continue
        row = dict(raw_row)
        sample_id = str(row.get("id", "")).strip()
        if not sample_id:
            raise ResultDataError(
                f"{condition}/{split} 결과 행에 id가 없습니다."
            )
        if sample_id in seen[condition]:
            raise ResultDataError(
                f"{condition}/{split}에 중복 샘플 id가 있습니다: {sample_id}"
            )
        seen[condition].add(sample_id)
        grouped[condition].append(row)

    missing = [condition for condition, condition_rows in grouped.items() if not condition_rows]
    if missing:
        raise ResultDataError(
            f"split={split}인 결과가 없는 조건: {', '.join(missing)}"
        )
    return grouped


def load_type_map(path: str | Path = DEFAULT_TYPE_MAP) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ResultDataError(f"type_map 파일이 없습니다: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ResultDataError(f"type_map JSON 파싱 오류: {path}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ResultDataError(f"type_map은 JSON 개체여야 합니다: {path}")
    return value


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return _escape(value)


def _integer(value: Any) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "-"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    header = "| " + " | ".join(_escape(cell) for cell in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_escape(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _error_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("error"))


def render_report(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    split: str,
    conditions: Sequence[str],
    type_map: Mapping[str, Any],
) -> str:
    """조건별 지표, McNemar, 품목·비용 표를 하나의 마크다운으로 만든다."""

    metric_rows: list[list[str]] = []
    for condition in conditions:
        rows = list(grouped[condition])
        binary = binary_metrics(rows)
        typed = type_metrics(rows, type_map)
        confidence = confidence_flag_validity(rows)
        citation = citation_fidelity(rows)
        metric_rows.append(
            [
                condition,
                _integer(binary.get("n")),
                _integer(_error_count(rows)),
                _number(binary.get("accuracy")),
                _number(binary.get("balanced_accuracy")),
                _number(binary.get("precision")),
                _number(binary.get("recall")),
                _number(binary.get("f1")),
                _number(binary.get("fpr")),
                _number(binary.get("fnr")),
                _number(typed.get("macro_f1")),
                _number(citation.get("citation_fidelity")),
                f"{_integer(confidence.get('low_count'))} / {_number(confidence.get('low_error_rate'))}",
                _number(confidence.get("overall_error_rate")),
            ]
        )

    sections = [
        f"# 평가 결과 ({_escape(split)})",
        "",
        "## 조건별 지표",
        "",
        _table(
            [
                "조건",
                "N",
                "오류",
                "Accuracy",
                "Balanced accuracy",
                "Precision",
                "Recall",
                "F1",
                "FPR",
                "FNR",
                "유형 macro-F1",
                "citation_fidelity",
                "Low 건수 / 오답률",
                "전체 오답률",
            ],
            metric_rows,
        ),
    ]

    pair_rows: list[list[str]] = []
    for left, right in (("B0", "A"), ("B1", "A")):
        if left not in grouped or right not in grouped:
            continue
        result = mcnemar(list(grouped[left]), list(grouped[right]))
        for stratum, label in (("overall", "전체"), ("normal", "정상"), ("defect", "결함")):
            values = result.get(stratum, {})
            pair_rows.append(
                [
                    f"{left} vs {right}",
                    label,
                    _integer(values.get("n")),
                    _integer(values.get("b")),
                    _integer(values.get("c")),
                    _number(values.get("p_value"), 6),
                ]
            )

    sections.extend(["", "## McNemar 검정", ""])
    if pair_rows:
        sections.append(
            _table(
                ["비교", "층", "짝 N", "b (왼쪽만 정답)", "c (오른쪽만 정답)", "p-value"],
                pair_rows,
            )
        )
    else:
        sections.append("B0/B1과 A가 함께 있어야 McNemar 검정을 계산할 수 있습니다.")

    item_rows: list[list[str]] = []
    item_names = sorted(
        {
            str(row.get("item"))
            for condition in conditions
            for row in grouped[condition]
            if row.get("item")
        }
    )
    for condition in conditions:
        for item in item_names:
            rows = [row for row in grouped[condition] if row.get("item") == item]
            if not rows:
                continue
            binary = binary_metrics(rows)
            typed = type_metrics(rows, type_map)
            item_rows.append(
                [
                    condition,
                    item,
                    _integer(binary.get("n")),
                    _integer(_error_count(rows)),
                    _number(binary.get("accuracy")),
                    _number(binary.get("balanced_accuracy")),
                    _number(binary.get("f1")),
                    _number(binary.get("fpr")),
                    _number(binary.get("fnr")),
                    _number(typed.get("macro_f1")),
                ]
            )

    sections.extend(
        [
            "",
            "## 품목별 분해",
            "",
            _table(
                ["조건", "품목", "N", "오류", "Accuracy", "Balanced accuracy", "F1", "FPR", "FNR", "유형 macro-F1"],
                item_rows,
            ),
        ]
    )

    cost_rows: list[list[str]] = []
    for condition in conditions:
        rows = list(grouped[condition])
        cost = cost_latency(rows)
        cost_rows.append(
            [
                condition,
                _integer(cost.get("n")),
                _number(cost.get("latency_p50_s")),
                _number(cost.get("latency_p95_s")),
                _number(cost.get("mean_tokens"), 1),
                _number(cost.get("mean_usd"), 6),
                _number(cost.get("hitl_rate")),
                _number(cost.get("mean_validation_attempts"), 2),
            ]
        )
    sections.extend(
        [
            "",
            "## 비용·지연",
            "",
            _table(
                ["조건", "N", "지연 p50(s)", "지연 p95(s)", "평균 tokens", "평균 USD", "HITL 비율", "평균 validation 횟수"],
                cost_rows,
            ),
        ]
    )
    return "\n".join(sections).rstrip() + "\n"


def write_report(
    results: str | Path,
    *,
    split: str,
    conditions: Sequence[str],
    type_map_path: str | Path = DEFAULT_TYPE_MAP,
) -> tuple[str, Path]:
    """마크다운 정리를 만들고 ``summary_{split}.md``로 저장한다."""

    results_path = Path(results)
    rows = load_result_rows(results_path)
    grouped = select_rows(rows, split=split, conditions=conditions)
    markdown = render_report(
        grouped,
        split=split,
        conditions=conditions,
        type_map=load_type_map(type_map_path),
    )
    output_dir = results_path if results_path.is_dir() else results_path.parent
    output_path = output_dir / f"summary_{split}.md"
    output_path.write_text(markdown, encoding="utf-8")
    return markdown, output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="평가 JSONL을 마크다운 표로 집계합니다.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--split", choices=("dev", "heldout"), default="heldout")
    parser.add_argument("--conditions", default="B0,B1,A")
    parser.add_argument("--type-map", type=Path, default=DEFAULT_TYPE_MAP)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        conditions = parse_conditions(args.conditions)
        markdown, _ = write_report(
            args.results,
            split=args.split,
            conditions=conditions,
            type_map_path=args.type_map,
        )
    except ResultDataError as exc:
        parser.error(str(exc))
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
