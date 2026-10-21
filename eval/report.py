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
    EvaluationDataError,
    binary_metrics,
    citation_fidelity,
    confidence_flag_validity,
    cost_latency,
    mcnemar,
    paired_balanced_accuracy_test,
    paired_macro_f1_test,
    type_metrics,
)


DEFAULT_RESULTS_DIR = ROOT / "eval" / "results"
DEFAULT_TYPE_MAP = ROOT / "eval" / "type_map.json"
DEFAULT_DATASET = ROOT / "eval" / "dataset.json"
_NON_RESULT_PREFIXES = ("human_check_",)
_DRY_RUN_MARKER = "_DRYRUN"
H4_MIN_GROUP_N = 20


class ResultDataError(ValueError):
    """결과 파일로 보고서를 만들 수 없음을 나타낸다."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"비표준 JSON 수치를 허용하지 않습니다: {value}")


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
        if _DRY_RUN_MARKER in results.stem:
            raise ResultDataError(f"dry-run 결과는 실측 보고서에 사용할 수 없습니다: {results}")
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
                row = json.loads(raw_line, parse_constant=_reject_json_constant)
            except json.JSONDecodeError as exc:
                raise ResultDataError(
                    f"JSONL 파싱 오류: {path}:{line_number}: {exc.msg}"
                ) from exc
            except ValueError as exc:
                raise ResultDataError(
                    f"JSONL 값 오류: {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ResultDataError(
                    f"JSONL 행이 개체가 아닙니다: {path}:{line_number}"
                )
            missing = [
                key
                for key in ("condition", "split")
                if not isinstance(row.get(key), str) or not row[key].strip()
            ]
            if missing:
                joined = ", ".join(missing)
                raise ResultDataError(
                    f"결과 행의 필드가 누락되었습니다: {path}:{line_number}: {joined}"
                )
            padded = [
                key
                for key in ("condition", "split")
                if row[key] != row[key].strip()
            ]
            if padded:
                raise ResultDataError(
                    f"결과 행의 필드에 앞뒤 공백이 있습니다: "
                    f"{path}:{line_number}: {', '.join(padded)}"
                )
            rows.append(row)

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
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ResultDataError(
                f"{condition}/{split} 결과 행에 id가 없습니다."
            )
        if sample_id != sample_id.strip():
            raise ResultDataError(
                f"{condition}/{split} 결과 id에 앞뒤 공백을 둘 수 없습니다: {sample_id!r}"
            )
        if sample_id in seen[condition]:
            raise ResultDataError(
                f"{condition}/{split}에 중복 샘플 id가 있습니다: {sample_id}"
            )
        seen[condition].add(sample_id)
        grouped[condition].append(row)

    return grouped


def load_expected_rows(
    path: str | Path = DEFAULT_DATASET,
    *,
    split: str,
) -> list[dict[str, Any]]:
    """dataset.json에서 split의 사전 등록 universe와 정답을 읽는다."""

    dataset_path = Path(path)
    try:
        payload = json.loads(
            dataset_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except FileNotFoundError as exc:
        raise ResultDataError(f"dataset 파일이 없습니다: {dataset_path}") from exc
    except json.JSONDecodeError as exc:
        raise ResultDataError(
            f"dataset JSON 파싱 오류: {dataset_path}: {exc.msg}"
        ) from exc
    except ValueError as exc:
        raise ResultDataError(f"dataset JSON 값 오류: {dataset_path}: {exc}") from exc
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list):
        raise ResultDataError(f"dataset의 samples는 배열이어야 합니다: {dataset_path}")
    if any(not isinstance(sample, Mapping) for sample in samples):
        raise ResultDataError("dataset의 모든 sample은 JSON 개체여야 합니다.")
    canonical = [dict(sample) for sample in samples]
    try:
        # 필터링 전에 전체 roster를 검증해 split 오타나 전역 중복 ID를 숨기지 않는다.
        binary_metrics([], canonical)
    except EvaluationDataError as exc:
        raise ResultDataError(f"dataset 사전 등록 오류: {exc}") from exc
    expected = [sample for sample in canonical if sample.get("split") == split]
    if not expected:
        raise ResultDataError(f"dataset에 split={split}인 사전 등록 표본이 없습니다.")
    return expected


def _reject_wrong_split_for_expected_ids(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_rows: Sequence[Mapping[str, Any]],
    split: str,
    conditions: Sequence[str],
) -> None:
    """대상 roster ID가 다른 split으로 기록된 경우 조용히 누락 처리하지 않는다."""

    expected_ids = {row.get("id") for row in expected_rows}
    selected_conditions = set(conditions)
    for row in rows:
        condition = row.get("condition")
        if not isinstance(condition, str):
            raise ResultDataError("결과 행의 condition은 문자열이어야 합니다.")
        if condition not in selected_conditions:
            continue
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id or row_id != row_id.strip():
            raise ResultDataError(
                f"{condition} 결과에 유효한 문자열 id가 필요합니다."
            )
        if row_id in expected_ids and row.get("split") != split:
            raise ResultDataError(
                f"사전 등록 ID의 split이 결과와 다릅니다: "
                f"{row_id} (expected={split}, result={row.get('split')})"
            )


def load_type_map(path: str | Path = DEFAULT_TYPE_MAP) -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except FileNotFoundError as exc:
        raise ResultDataError(f"type_map 파일이 없습니다: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ResultDataError(f"type_map JSON 파싱 오류: {path}: {exc.msg}") from exc
    except ValueError as exc:
        raise ResultDataError(f"type_map JSON 값 오류: {path}: {exc}") from exc
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


def _interval(low: Any, high: Any, digits: int = 3) -> str:
    if low is None or high is None:
        return "-"
    return f"[{_number(low, digits)}, {_number(high, digits)}]"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    header = "| " + " | ".join(_escape(cell) for cell in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_escape(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def render_report(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    split: str,
    conditions: Sequence[str],
    type_map: Mapping[str, Any],
    expected_rows: Sequence[Mapping[str, Any]],
) -> str:
    """조건별 지표, McNemar, 품목·비용 표를 하나의 마크다운으로 만든다."""

    metric_rows: list[list[str]] = []
    incomplete_conditions: list[str] = []
    for condition in conditions:
        rows = list(grouped[condition])
        binary = binary_metrics(rows, expected_rows)
        typed = type_metrics(rows, type_map, expected_rows)
        confidence = confidence_flag_validity(rows, expected_rows)
        citation = citation_fidelity(rows)
        if binary["observed_n"] != binary["expected_n"]:
            incomplete_conditions.append(condition)
        metric_rows.append(
            [
                condition,
                _integer(binary.get("expected_n")),
                _integer(binary.get("observed_n")),
                _integer(binary.get("missing_n")),
                _integer(binary.get("error_n")),
                _integer(binary.get("invalid_n")),
                _integer(binary.get("abstain_n")),
                _integer(binary.get("selective_n")),
                "완료" if binary["observed_n"] == binary["expected_n"] else "부분",
                _number(binary.get("accuracy")),
                _number(binary.get("balanced_accuracy")),
                _number(binary.get("fpr")),
                _number(binary.get("fnr")),
                _number(binary.get("coverage")),
                _number(binary.get("abstain_rate")),
                _number(binary.get("selective_accuracy")),
                _number(binary.get("selective_precision")),
                _number(binary.get("selective_recall")),
                _number(binary.get("selective_f1")),
                _number(typed.get("macro_f1")),
                _number(citation.get("citation_fidelity")),
                f"{_integer(confidence.get('low_count'))} / {_number(confidence.get('low_error_rate'))}",
                f"{_integer(confidence.get('non_low_count'))} / {_number(confidence.get('non_low_error_rate'))}",
                _number(confidence.get("selected_error_rate")),
            ]
        )

    sections = [
        f"# 평가 결과 ({_escape(split)})",
        "",
        *(
            [
                "> **결론 금지:** 사전 등록 표본을 모두 처리하지 않은 조건이 있습니다. "
                f"({', '.join(incomplete_conditions)}) 누락은 보수적 오답으로 집계했지만, "
                "이 보고서로 성능 결론을 내리면 안 됩니다.",
                "",
            ]
            if incomplete_conditions
            else []
        ),
        "## 조건별 지표",
        "",
        _table(
            [
                "조건",
                "사전등록 N",
                "실제 행 N",
                "누락",
                "오류",
                "무효",
                "ABSTAIN",
                "자동판정 N",
                "캡처 상태",
                "Accuracy",
                "Balanced accuracy",
                "보수 FPR",
                "보수 FNR",
                "Coverage",
                "Abstain rate",
                "Selective accuracy",
                "Selective precision",
                "Selective recall",
                "Selective F1",
                "유형 macro-F1",
                "citation_fidelity",
                "Low 건수 / 오답률",
                "High 건수 / 오답률",
                "자동판정 오답률",
            ],
            metric_rows,
        ),
    ]

    # H1/H2의 주 검정은 목표 지표 자체를 같은 ID 안에서 교환한다.
    # McNemar는 전체 이진 정오에 대한 보조 분석으로 아래에 별도 남겨 둔다.
    hypothesis_rows: list[list[str]] = []
    if "B0" in grouped and "A" in grouped:
        b0_rows = list(grouped["B0"])
        a_rows = list(grouped["A"])
        b0_complete = binary_metrics(b0_rows, expected_rows)["observed_n"] == len(expected_rows)
        a_complete = binary_metrics(a_rows, expected_rows)["observed_n"] == len(expected_rows)
        paired_complete = bool(b0_complete and a_complete)
        h1 = paired_balanced_accuracy_test(b0_rows, a_rows, expected_rows)
        h2 = paired_macro_f1_test(b0_rows, a_rows, type_map, expected_rows)
        for hypothesis, values in (("H1", h1), ("H2", h2)):
            supported = (
                paired_complete
                and float(values["delta"]) > 0.0
                and float(values["p_value"]) < 0.05
            )
            decision = (
                "보류(dev 탐색)"
                if split != "heldout"
                else "보류(불완전 캡처)"
                if not paired_complete
                else "지지" if supported else "효과 미입증"
            )
            hypothesis_rows.append([
                hypothesis,
                str(values["metric"]),
                _integer(values["n"]),
                _number(values["a"]),
                _number(values["b"]),
                _number(values["delta"]),
                _interval(values["delta_ci95_low"], values["delta_ci95_high"]),
                _number(values["p_value"], 6),
                f"{values['permutation_method']} / {values['permutation_samples']}",
                decision,
            ])

    sections.extend(["", "## 사전 등록 H1·H2 주 검정", ""])
    if hypothesis_rows:
        sections.extend([
            _table(
                [
                    "가설", "주 지표", "짝 N", "B0", "A", "Δ(A−B0)",
                    "Δ 95% CI", "양측 p", "순열 방법 / 횟수", "판정",
                ],
                hypothesis_rows,
            ),
            "",
            "> H1은 정상/결함 층화 짝지은 순열 검정, H2는 진실 유형 층화 "
            "짝지은 macro-F1 순열 검정을 쓴다. 95% CI는 짝과 층을 "
            "유지한 percentile bootstrap이며 효과 크기 서술용이다. "
            "seed=20261020, Monte-Carlo 순열 상한=20,000회, bootstrap=10,000회다.",
        ])
    else:
        sections.append("B0·A가 모두 있어야 H1·H2 주 검정을 계산할 수 있습니다.")

    if "A" in grouped:
        a_rows = list(grouped["A"])
        a_binary = binary_metrics(a_rows, expected_rows)
        confidence = confidence_flag_validity(a_rows, expected_rows)
        low_rate = confidence.get("low_error_rate")
        high_rate = confidence.get("non_low_error_rate")
        groups_large_enough = (
            int(confidence["low_count"]) >= H4_MIN_GROUP_N
            and int(confidence["non_low_count"]) >= H4_MIN_GROUP_N
        )
        enriched = (
            isinstance(low_rate, (int, float))
            and isinstance(high_rate, (int, float))
            and float(low_rate) > float(high_rate)
            and float(low_rate) >= 2.0 * float(high_rate)
        )
        if split != "heldout":
            h4_decision = "보류(dev 탐색)"
        elif a_binary["observed_n"] != a_binary["expected_n"]:
            h4_decision = "보류(불완전 캡처)"
        elif not groups_large_enough:
            h4_decision = f"보류(그룹별 {H4_MIN_GROUP_N}건 미만)"
        else:
            h4_decision = "지지" if enriched else "효과 미입증"
        sections.extend([
            "",
            "## 사전 등록 H4 신뢰도 플래그",
            "",
            _table(
                ["그룹", "N", "오답", "오답률", "95% Wilson CI"],
                [
                    ["low 자동판정", _integer(confidence["low_count"]), _integer(confidence["low_errors"]), _number(low_rate), _interval(confidence["low_error_ci95_low"], confidence["low_error_ci95_high"])],
                    ["high 자동판정", _integer(confidence["non_low_count"]), _integer(confidence["non_low_errors"]), _number(high_rate), _interval(confidence["non_low_error_ci95_low"], confidence["non_low_error_ci95_high"])],
                    ["전체 자동판정", _integer(confidence["selected_count"]), _integer(confidence["selected_errors"]), _number(confidence["selected_error_rate"]), _interval(confidence["selected_error_ci95_low"], confidence["selected_error_ci95_high"])],
                ],
            ),
            "",
            f"- 판정: **{h4_decision}**",
            f"- 별도 상태: ABSTAIN {confidence['abstain_count']}, 실행 오류 {confidence['error_count']}, 누락 {confidence['missing_count']}, 무효 판정 {confidence['invalid_count']}, 무효 confidence {confidence['invalid_confidence_count']}",
            f"- 계약: low/high PASS·REJECT가 각각 {H4_MIN_GROUP_N}건 이상일 때만 low 오답률이 high의 2배 이상이고 엄격히 높은지 판정한다.",
        ])

    coverage_rows: list[list[str]] = []
    for condition in conditions:
        binary = binary_metrics(list(grouped[condition]), expected_rows)
        coverage_rows.extend(
            [
                [
                    condition,
                    "전체",
                    _integer(binary.get("expected_n")),
                    _integer(binary.get("observed_n")),
                    _integer(binary.get("missing_n")),
                    _integer(binary.get("error_n")),
                    _integer(binary.get("invalid_n")),
                    _integer(binary.get("abstain_n")),
                    _integer(binary.get("selective_n")),
                    _number(binary.get("coverage")),
                    _number(binary.get("abstain_rate")),
                    _number(1.0 - float(binary.get("accuracy", 0.0))),
                ],
                [
                    condition,
                    "정상",
                    _integer(binary.get("normal_n")),
                    _integer(binary.get("normal_observed_n")),
                    _integer(binary.get("normal_missing_n")),
                    _integer(binary.get("normal_error_n")),
                    _integer(binary.get("normal_invalid_n")),
                    _integer(binary.get("normal_abstain_n")),
                    _integer(binary.get("normal_selective_n")),
                    _number(binary.get("normal_coverage")),
                    _number(binary.get("normal_abstain_rate")),
                    _number(binary.get("normal_failure_rate")),
                ],
                [
                    condition,
                    "결함",
                    _integer(binary.get("defect_n")),
                    _integer(binary.get("defect_observed_n")),
                    _integer(binary.get("defect_missing_n")),
                    _integer(binary.get("defect_error_n")),
                    _integer(binary.get("defect_invalid_n")),
                    _integer(binary.get("defect_abstain_n")),
                    _integer(binary.get("defect_selective_n")),
                    _number(binary.get("defect_coverage")),
                    _number(binary.get("defect_abstain_rate")),
                    _number(binary.get("defect_failure_rate")),
                ],
            ]
        )
    sections.extend(
        [
            "",
            "## Coverage·보류 층별 분해",
            "",
            _table(
                [
                    "조건", "층", "사전등록 N", "실제 행 N", "누락", "오류", "무효",
                    "ABSTAIN", "자동판정 N", "Coverage", "Abstain rate", "보수 실패율",
                ],
                coverage_rows,
            ),
        ]
    )

    pair_rows: list[list[str]] = []
    for left, right in (("B0", "A"), ("B1", "A")):
        if left not in grouped or right not in grouped:
            continue
        result = mcnemar(
            list(grouped[left]),
            list(grouped[right]),
            expected_rows,
        )
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

    sections.extend(["", "## 보조 이진 정오 McNemar 검정", ""])
    if pair_rows:
        sections.extend([
            _table(
                ["비교", "층", "짝 N", "b (왼쪽만 정답)", "c (오른쪽만 정답)", "p-value"],
                pair_rows,
            ),
            "",
            "> B0/A는 전체 시스템 비교다. B1/A는 전문 삽입 단일 VLM과 전체 "
            "에이전트의 대안 비교이며 RAG 검색 단독 효과로 해석하지 않는다.",
        ])
    else:
        sections.append("B0/B1과 A가 함께 있어야 McNemar 검정을 계산할 수 있습니다.")

    item_rows: list[list[str]] = []
    item_names = sorted({str(row.get("item")) for row in expected_rows if row.get("item")})
    for condition in conditions:
        for item in item_names:
            rows = [row for row in grouped[condition] if row.get("item") == item]
            item_expected = [row for row in expected_rows if row.get("item") == item]
            binary = binary_metrics(rows, item_expected)
            typed = type_metrics(rows, type_map, item_expected)
            item_rows.append(
                [
                    condition,
                    item,
                    _integer(binary.get("expected_n")),
                    _integer(binary.get("observed_n")),
                    _integer(binary.get("missing_n")),
                    _integer(binary.get("error_n")),
                    _integer(binary.get("invalid_n")),
                    _integer(binary.get("abstain_n")),
                    _integer(binary.get("selective_n")),
                    "완료" if binary["observed_n"] == binary["expected_n"] else "부분",
                    _number(binary.get("accuracy")),
                    _number(binary.get("balanced_accuracy")),
                    _number(binary.get("fpr")),
                    _number(binary.get("fnr")),
                    _number(binary.get("coverage")),
                    _number(binary.get("abstain_rate")),
                    _number(binary.get("selective_f1")),
                    _number(typed.get("macro_f1")),
                ]
            )

    sections.extend(
        [
            "",
            "## 품목별 분해",
            "",
            _table(
                [
                    "조건", "품목", "사전등록 N", "실제 행 N", "누락", "오류",
                    "무효", "ABSTAIN", "자동판정 N", "캡처 상태",
                    "Accuracy", "Balanced accuracy", "보수 FPR", "보수 FNR",
                    "Coverage", "Abstain rate", "Selective F1", "유형 macro-F1",
                ],
                item_rows,
            ),
        ]
    )

    cost_rows: list[list[str]] = []
    for condition in conditions:
        rows = list(grouped[condition])
        binary = binary_metrics(rows, expected_rows)
        cost = cost_latency(rows)
        cost_rows.append(
            [
                condition,
                _integer(binary.get("expected_n")),
                _integer(binary.get("observed_n")),
                _integer(binary.get("missing_n")),
                _integer(binary.get("error_n")),
                _integer(binary.get("invalid_n")),
                _integer(binary.get("abstain_n")),
                _integer(binary.get("selective_n")),
                "완료" if binary["observed_n"] == binary["expected_n"] else "부분",
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
                [
                    "조건", "사전등록 N", "실제 행 N", "누락", "오류", "무효", "ABSTAIN",
                    "자동판정 N", "캡처 상태", "지연 p50(s)", "지연 p95(s)",
                    "평균 tokens", "평균 USD", "HITL 비율", "평균 validation 횟수",
                ],
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
    dataset_path: str | Path = DEFAULT_DATASET,
) -> tuple[str, Path]:
    """마크다운 정리를 만들고 ``summary_{split}.md``로 저장한다."""

    results_path = Path(results)
    rows = load_result_rows(results_path)
    expected_rows = load_expected_rows(dataset_path, split=split)
    _reject_wrong_split_for_expected_ids(
        rows,
        expected_rows=expected_rows,
        split=split,
        conditions=conditions,
    )
    grouped = select_rows(rows, split=split, conditions=conditions)
    try:
        markdown = render_report(
            grouped,
            split=split,
            conditions=conditions,
            type_map=load_type_map(type_map_path),
            expected_rows=expected_rows,
        )
    except EvaluationDataError as exc:
        raise ResultDataError(f"결과와 dataset의 무결성 검증 실패: {exc}") from exc
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
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
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
            dataset_path=args.dataset,
        )
    except ResultDataError as exc:
        parser.error(str(exc))
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
