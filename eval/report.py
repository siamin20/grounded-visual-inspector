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
from eval.provenance import ProvenanceError, verify_run_manifest  # noqa: E402


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
            if row.get("dry_run") is True:
                raise ResultDataError(
                    f"dry-run 결과는 실측 보고서에 사용할 수 없습니다: "
                    f"{path}:{line_number}"
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


def _verified_result_files(
    run_dir: Path,
    *,
    split: str,
    conditions: Sequence[str],
    dataset_path: str | Path,
    type_map_path: str | Path,
    repo_root: str | Path,
) -> tuple[list[Path], bool, dict[str, dict[str, Any]]]:
    """manifest가 결속한 결과와 최종 결론 금지 여부를 반환한다."""

    if not run_dir.is_dir():
        raise ResultDataError(
            "provenance 검증 보고서의 --results는 run 디렉터리여야 합니다."
        )
    requested = list(conditions)
    if not requested or any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in requested
    ):
        raise ResultDataError("보고서 조건은 빈 값 없는 문자열 목록이어야 합니다.")
    if len(requested) != len(set(requested)):
        raise ResultDataError("보고서 조건을 중복해 지정할 수 없습니다.")

    try:
        manifest = verify_run_manifest(
            run_dir,
            reject_dry_run=True,
            reject_fake=True,
            require_complete=False,
            validate_rows=True,
            repo_root=repo_root,
            verify_inputs=True,
        )
    except ProvenanceError as exc:
        raise ResultDataError(f"실행 provenance 검증 실패: {exc}") from exc

    if manifest.get("status") not in {"complete", "incomplete"}:
        raise ResultDataError(
            f"보고서를 만들 수 없는 suite 상태입니다: {manifest.get('status')}"
        )
    if manifest.get("split") != split:
        raise ResultDataError(
            f"manifest split과 보고서 split이 다릅니다: "
            f"{manifest.get('split')} != {split}"
        )

    root = Path(repo_root)
    artifacts = manifest["artifacts"]
    for label, supplied, record in (
        ("dataset", dataset_path, artifacts["dataset"]),
        ("type_map", type_map_path, artifacts["type_map"]),
    ):
        try:
            supplied_resolved = Path(supplied).resolve(strict=True)
            recorded_resolved = (root / record["path"]).resolve(strict=True)
        except OSError as exc:
            raise ResultDataError(f"{label} 경로를 확인할 수 없습니다: {supplied}") from exc
        if supplied_resolved != recorded_resolved:
            raise ResultDataError(
                f"보고서 {label}이 manifest에 결속된 파일과 다릅니다."
            )

    planned = manifest["planned_conditions"]
    records = manifest["conditions"]
    paths: list[Path] = []
    setup_usages: dict[str, dict[str, Any]] = {}
    provenance_incomplete = manifest.get("status") != "complete"
    for condition in requested:
        if condition not in planned:
            raise ResultDataError(f"manifest에 사전 계획되지 않은 조건입니다: {condition}")
        record = records.get(condition)
        if not isinstance(record, Mapping) or record.get("status") not in {
            "complete",
            "incomplete",
        }:
            raise ResultDataError(f"결과가 결속되지 않은 조건입니다: {condition}")
        if record.get("status") != "complete":
            provenance_incomplete = True
        result = record.get("result")
        if not isinstance(result, Mapping) or not isinstance(result.get("path"), str):
            raise ResultDataError(f"manifest에 결과 경로가 없습니다: {condition}")
        paths.append(run_dir / result["path"])
        setup = record.get("setup_usage")
        if not isinstance(setup, Mapping):
            raise ResultDataError(f"manifest에 초기화 usage가 없습니다: {condition}")
        setup_usages[condition] = dict(setup)
    return paths, provenance_incomplete, setup_usages


def _cost_with_setup(
    rows: Sequence[Mapping[str, Any]],
    setup: Mapping[str, Any] | None,
) -> dict[str, int | float | None]:
    """조건 초기화 usage를 조건 총계와 표본당 평균에 상각한다."""
    result = cost_latency(rows)
    values = dict(setup or {})
    count = len(rows)
    setup_tokens = int(values.get("tokens", 0))
    setup_cost = float(values.get("estimated_usd", 0.0))
    setup_proxy = float(values.get("compute_proxy_usd", 0.0))
    setup_requests = int(values.get("successful_provider_requests", 0))
    unknown_usage_n = int(result.get("unknown_usage_n") or 0)
    row_total_cost = float(result.get("known_estimated_usd_sum") or 0.0)
    row_total_proxy = float(result.get("mean_compute_proxy_usd") or 0.0) * count
    row_total_requests = (
        float(result.get("mean_successful_provider_requests") or 0.0) * count
    )
    result["setup_latency_s"] = float(values.get("latency_s", 0.0))
    result["setup_estimated_usd"] = setup_cost
    result["total_estimated_usd"] = (
        None if unknown_usage_n else row_total_cost + setup_cost
    )
    result["total_successful_provider_requests"] = int(
        round(row_total_requests + setup_requests)
    )
    if count:
        if not unknown_usage_n:
            result["mean_tokens"] = (
                float(result.get("mean_tokens") or 0.0) + setup_tokens / count
            )
            result["mean_usd"] = (row_total_cost + setup_cost) / count
            result["mean_compute_proxy_usd"] = (
                row_total_proxy + setup_proxy
            ) / count
        result["mean_successful_provider_requests"] = (
            row_total_requests + setup_requests
        ) / count
        setup_mode = values.get("token_accounting", "total_only")
        if setup_mode != "total_only":
            if result.get("mean_input_tokens") is not None:
                result["mean_input_tokens"] = float(
                    result["mean_input_tokens"]
                ) + int(values.get("input_tokens", 0)) / count
            if result.get("mean_output_tokens") is not None:
                result["mean_output_tokens"] = float(
                    result["mean_output_tokens"]
                ) + int(values.get("output_tokens", 0)) / count
        elif setup_tokens or setup_requests:
            result["mean_input_tokens"] = None
            result["mean_output_tokens"] = None
    return result


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
    provenance_incomplete: bool = False,
    setup_usages: Mapping[str, Mapping[str, Any]] | None = None,
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

    conclusion_forbidden = bool(incomplete_conditions or provenance_incomplete)
    incomplete_reasons: list[str] = []
    if incomplete_conditions:
        incomplete_reasons.append(f"행 누락: {', '.join(incomplete_conditions)}")
    if provenance_incomplete:
        incomplete_reasons.append("manifest 또는 조건 상태 incomplete")

    sections = [
        f"# 평가 결과 ({_escape(split)})",
        "",
        *(
            [
                "> **결론 금지:** 평가 캡처가 완결되지 않았습니다. "
                f"({'; '.join(incomplete_reasons)}) "
                "누락은 보수적 오답으로 집계했지만, "
                "이 보고서로 성능 결론을 내리면 안 됩니다.",
                "",
            ]
            if conclusion_forbidden
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
        "",
        "> `citation_fidelity`는 저장된 VLM 관찰 기록과 생성 보고서 관찰문 "
        "사이의 8자 n-gram 인용 일치도다. 실제 이미지·픽셀 진실성이나 "
        "VLM 환각률을 측정하지 않으며, H3는 별도 20쌍 사람 블라인드 "
        "채점으로 판정한다.",
    ]

    # H1/H2의 주 검정은 목표 지표 자체를 같은 ID 안에서 교환한다.
    # McNemar는 전체 이진 정오에 대한 보조 분석으로 아래에 별도 남겨 둔다.
    hypothesis_rows: list[list[str]] = []
    if "B0" in grouped and "A" in grouped:
        b0_rows = list(grouped["B0"])
        a_rows = list(grouped["A"])
        b0_complete = binary_metrics(b0_rows, expected_rows)["observed_n"] == len(expected_rows)
        a_complete = binary_metrics(a_rows, expected_rows)["observed_n"] == len(expected_rows)
        paired_rows_complete = bool(b0_complete and a_complete)
        h1 = paired_balanced_accuracy_test(b0_rows, a_rows, expected_rows)
        h2 = paired_macro_f1_test(b0_rows, a_rows, type_map, expected_rows)
        for hypothesis, values in (("H1", h1), ("H2", h2)):
            supported = (
                paired_rows_complete
                and not provenance_incomplete
                and float(values["delta"]) > 0.0
                and float(values["p_value"]) < 0.05
            )
            decision = (
                "보류(dev 탐색)"
                if split != "heldout"
                else "보류(불완전 provenance)"
                if provenance_incomplete
                else "보류(불완전 캡처)"
                if not paired_rows_complete
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
        elif provenance_incomplete:
            h4_decision = "보류(불완전 provenance)"
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
        cost = _cost_with_setup(
            rows,
            (setup_usages or {}).get(condition),
        )
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
                _number(cost.get("setup_latency_s")),
                _number(cost.get("mean_tokens"), 1),
                _number(cost.get("mean_input_tokens"), 1),
                _number(cost.get("mean_output_tokens"), 1),
                _number(cost.get("mean_usd"), 6),
                _number(cost.get("total_estimated_usd"), 6),
                _number(cost.get("mean_compute_proxy_usd"), 6),
                _integer(cost.get("unknown_usage_n")),
                _number(cost.get("hitl_rate")),
                _number(cost.get("mean_validation_attempts"), 2),
                _number(cost.get("mean_validation_rejections"), 2),
                _number(cost.get("mean_retry_count"), 2),
                _number(cost.get("mean_replan_rounds"), 2),
                _number(cost.get("mean_successful_provider_requests"), 2),
                _integer(cost.get("total_successful_provider_requests")),
                _number(cost.get("mean_successful_vlm_requests"), 2),
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
                    "조건 초기화 지연(s)",
                    "평균 tokens", "평균 input tokens", "평균 output tokens",
                    "평균 estimated USD", "조건 총 estimated USD",
                    "평균 compute proxy USD", "usage 미지 N", "HITL 비율",
                    "평균 validation 횟수", "평균 validation 반려",
                    "평균 retry", "평균 replan", "평균 성공 provider 응답",
                    "조건 총 성공 provider 응답",
                    "평균 성공 VLM 응답",
                ],
                cost_rows,
            ),
            "",
            "> 배치 평가는 `auto_approve`로 실행한다. HITL 비율은 승인 필요 사례의 "
            "발생률이며, 사람 판단으로 정확도가 개선됐다는 지표가 아니다.",
            "",
            "> `성공 provider 응답`은 usage를 반환한 성공 응답 수이며, "
            "실패·타임아웃을 포함한 총 API 시도 횟수가 아니다. 조건 초기화의 "
            "기준서 색인 usage는 manifest와 summary에 별도 결속하고, "
            "평균 비용·토큰·성공 응답에는 실제 처리 행 수로 상각한다.",
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
    verify_provenance: bool = True,
    repo_root: str | Path = ROOT,
) -> tuple[str, Path]:
    """마크다운 정리를 만들고 ``summary_{split}.md``로 저장한다.

    기본값은 run manifest·해시·행 결속·입력 해시를 모두 검증한다.
    ``verify_provenance=False``는 합성 fixture를 사용하는 저수준 단위 테스트용이다.
    """

    results_path = Path(results)
    provenance_incomplete = False
    setup_usages: dict[str, dict[str, Any]] = {}
    if verify_provenance:
        result_files, provenance_incomplete, setup_usages = _verified_result_files(
            results_path,
            split=split,
            conditions=conditions,
            dataset_path=dataset_path,
            type_map_path=type_map_path,
            repo_root=repo_root,
        )
        rows = [row for path in result_files for row in load_result_rows(path)]
    else:
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
            provenance_incomplete=provenance_incomplete,
            setup_usages=setup_usages,
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
            verify_provenance=True,
        )
    except ResultDataError as exc:
        parser.error(str(exc))
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
