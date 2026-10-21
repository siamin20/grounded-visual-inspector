"""평가 결과와 사전 등록 표본에만 의존하는 코드 기반 평가기다.

이 모듈은 외부 모델이나 네트워크를 호출하지 않는 재현 가능한 순수 함수로 구성된다.
주 지표에서 ``ABSTAIN``·오류·누락을 분모에서 제외하지 않는다. 분모가 0인
비율은 0.0으로 반환한다.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping
from statistics import NormalDist
from typing import Any

from agent.usage import (
    MISSING,
    UsageValueError,
    nonnegative_finite_float,
    nonnegative_int,
)


Row = Mapping[str, Any]


DEFAULT_RESAMPLING_SEED = 20261020
DEFAULT_PERMUTATION_SAMPLES = 20_000
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
EXACT_PERMUTATION_MAX_PAIRS = 16


class EvaluationDataError(ValueError):
    """사전 등록 표본이나 결과 행의 식별 무결성 위반을 나타낸다."""


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    """분모가 0이면 0.0을 반환한다."""

    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def wilson_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float | None, float | None]:
    """이항 비율의 Wilson score 신뢰구간을 반환한다.

    표본이 없으면 비율이 정의되지 않으므로 ``(None, None)``을
    반환한다. 잘못된 건수를 0으로 보정하지 않고 명시적으로
    거부한다.
    """

    if isinstance(successes, bool) or not isinstance(successes, int):
        raise ValueError("successes는 0 이상의 정수여야 합니다.")
    if isinstance(total, bool) or not isinstance(total, int):
        raise ValueError("total은 0 이상의 정수여야 합니다.")
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("0 <= successes <= total을 만족해야 합니다.")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 < float(confidence) < 1.0
    ):
        raise ValueError("confidence는 0과 1 사이의 유한한 수여야 합니다.")
    if total == 0:
        return None, None

    probability = successes / total
    z = NormalDist().inv_cdf(0.5 + float(confidence) / 2.0)
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = (probability + z_squared / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total
            + z_squared / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _has_error(row: Row) -> bool:
    error = row.get("error")
    return not (error is None or error == "")


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def _ground_truth(row: Row) -> bool | None:
    truth = _as_bool(row.get("is_defect"))
    if truth is not None:
        return truth

    label = row.get("label")
    if isinstance(label, str) and label.strip():
        return label.strip().casefold() != "good"
    return None


def _verdict(row: Row) -> str | None:
    value = row.get("verdict")
    return value if isinstance(value, str) and value in {"PASS", "REJECT", "ABSTAIN"} else None


def _sample_id(row: Row, *, source: str) -> str:
    value = row.get("id")
    if not isinstance(value, str) or not value.strip():
        raise EvaluationDataError(f"{source}에 비어 있지 않은 문자열 id가 필요합니다.")
    if value != value.strip():
        raise EvaluationDataError(f"{source}의 id에 앞뒤 공백을 둘 수 없습니다: {value!r}")
    return value


def _index_rows(rows: Iterable[Row], *, source: str) -> dict[str, Row]:
    indexed: dict[str, Row] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise EvaluationDataError(f"{source}의 행은 개체여야 합니다.")
        row_id = _sample_id(row, source=source)
        if row_id in indexed:
            raise EvaluationDataError(f"{source}에 중복 id가 있습니다: {row_id}")
        indexed[row_id] = row
    return indexed


def _expected_index(rows: Iterable[Row], *, source: str = "사전 등록 표본") -> dict[str, Row]:
    indexed = _index_rows(rows, source=source)
    for row_id, row in indexed.items():
        for key in ("item", "label", "split"):
            if not isinstance(row.get(key), str) or not str(row[key]).strip():
                raise EvaluationDataError(
                    f"{source}의 {row_id}에 비어 있지 않은 {key} 문자열이 필요합니다."
                )
            if row[key] != row[key].strip():
                raise EvaluationDataError(
                    f"{source}의 {row_id}/{key}에 앞뒤 공백을 둘 수 없습니다."
                )
        if row["split"] not in {"dev", "heldout"}:
            raise EvaluationDataError(
                f"{source}의 {row_id}에 알 수 없는 split이 있습니다: {row['split']}"
            )
        if type(row.get("is_defect")) is not bool:
            raise EvaluationDataError(
                f"{source}의 {row_id}에 bool is_defect가 필요합니다."
            )
        label_is_defect = row["label"] != "good"
        if label_is_defect != row["is_defect"]:
            raise EvaluationDataError(
                f"{source}의 {row_id}에서 label과 is_defect가 모순됩니다."
            )
    return indexed


def _records(
    rows: Iterable[Row],
    expected_rows: Iterable[Row] | None,
) -> list[tuple[str, Row, Row | None]]:
    materialized = list(rows)
    observed = _index_rows(materialized, source="평가 결과")
    expected = _expected_index(
        materialized if expected_rows is None else expected_rows
    )
    unexpected = sorted(observed.keys() - expected.keys())
    if unexpected:
        raise EvaluationDataError(
            "사전 등록에 없는 결과 id가 있습니다: " + ", ".join(unexpected)
        )
    for row_id, observed_row in observed.items():
        _validate_observed_metadata(expected[row_id], observed_row, row_id=row_id)
    return [(row_id, expected[row_id], observed.get(row_id)) for row_id in expected]


def _validate_observed_metadata(expected: Row, observed: Row, *, row_id: str) -> None:
    for key in ("item", "label", "split"):
        if key not in observed or observed.get(key) != expected.get(key):
            raise EvaluationDataError(
                f"결과 {row_id}의 {key}가 사전 등록과 다릅니다."
            )
    if type(observed.get("is_defect")) is not bool:
        raise EvaluationDataError(f"결과 {row_id}의 is_defect는 bool이어야 합니다.")
    if observed["is_defect"] is not expected["is_defect"]:
        raise EvaluationDataError(
            f"결과 {row_id}의 is_defect가 사전 등록과 다릅니다."
        )


def _usable_verdict(expected: Row, observed: Row | None) -> str | None:
    if observed is None or _has_error(observed):
        return None
    return _verdict(observed)


def _is_correct(expected: Row, observed: Row | None) -> bool:
    verdict = _usable_verdict(expected, observed)
    truth = _ground_truth(expected)
    return (truth is True and verdict == "REJECT") or (
        truth is False and verdict == "PASS"
    )


def binary_metrics(
    rows: Iterable[Row],
    expected_rows: Iterable[Row] | None = None,
) -> dict[str, int | float]:
    """결함 여부를 기준으로 이진 분류 지표를 계산한다.

    ``REJECT``를 양성(결함)으로 간주한다. 사전 등록 표본이 주어지면 그
    전체가 분모다. 정상은 PASS, 결함은 REJECT일 때만 정답이며
    ``ABSTAIN``·오류·누락·잘못된 verdict는 해당 층의 실패다.
    """

    records = _records(rows, expected_rows)
    tp = tn = fp = fn = 0
    selective_tp = selective_tn = selective_fp = selective_fn = 0
    observed_n = missing_n = error_n = abstain_n = invalid_n = 0
    normal_n = defect_n = normal_selected = defect_selected = 0
    normal_abstain = defect_abstain = 0
    normal_observed = defect_observed = 0
    normal_missing = defect_missing = 0
    normal_error = defect_error = 0
    normal_invalid = defect_invalid = 0

    for _, expected, observed in records:
        truth = bool(_ground_truth(expected))
        defect_n += int(truth)
        normal_n += int(not truth)
        if observed is None:
            missing_n += 1
            defect_missing += int(truth)
            normal_missing += int(not truth)
            verdict = None
        else:
            observed_n += 1
            defect_observed += int(truth)
            normal_observed += int(not truth)
            verdict = _usable_verdict(expected, observed)
            if _has_error(observed):
                error_n += 1
                defect_error += int(truth)
                normal_error += int(not truth)
            elif verdict is None:
                invalid_n += 1
                defect_invalid += int(truth)
                normal_invalid += int(not truth)
            elif verdict == "ABSTAIN":
                abstain_n += 1
                defect_abstain += int(truth)
                normal_abstain += int(not truth)

        if truth:
            if verdict == "REJECT":
                tp += 1
            else:
                fn += 1
        elif verdict == "PASS":
            tn += 1
        else:
            fp += 1

        if verdict not in {"PASS", "REJECT"}:
            continue
        prediction = verdict == "REJECT"
        defect_selected += int(truth)
        normal_selected += int(not truth)
        if truth and prediction:
            selective_tp += 1
        elif not truth and not prediction:
            selective_tn += 1
        elif not truth and prediction:
            selective_fp += 1
        else:
            selective_fn += 1

    total = len(records)
    recall = _safe_div(tp, defect_n)
    specificity = _safe_div(tn, normal_n)
    selective_n = selective_tp + selective_tn + selective_fp + selective_fn
    selective_precision = _safe_div(selective_tp, selective_tp + selective_fp)
    selective_recall = _safe_div(selective_tp, selective_tp + selective_fn)

    return {
        "n": total,
        "expected_n": total,
        "observed_n": observed_n,
        "result_n": observed_n,
        "missing_n": missing_n,
        "error_n": error_n,
        "abstain_n": abstain_n,
        "invalid_n": invalid_n,
        "normal_n": normal_n,
        "defect_n": defect_n,
        "normal_observed_n": normal_observed,
        "defect_observed_n": defect_observed,
        "normal_missing_n": normal_missing,
        "defect_missing_n": defect_missing,
        "normal_error_n": normal_error,
        "defect_error_n": defect_error,
        "normal_invalid_n": normal_invalid,
        "defect_invalid_n": defect_invalid,
        "normal_abstain_n": normal_abstain,
        "defect_abstain_n": defect_abstain,
        "normal_selective_n": normal_selected,
        "defect_selective_n": defect_selected,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": _safe_div(tp + tn, total),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "fpr": _safe_div(fp, normal_n),
        "fnr": _safe_div(fn, defect_n),
        "normal_failure_rate": _safe_div(fp, normal_n),
        "defect_failure_rate": _safe_div(fn, defect_n),
        "coverage": _safe_div(selective_n, total),
        "normal_coverage": _safe_div(normal_selected, normal_n),
        "defect_coverage": _safe_div(defect_selected, defect_n),
        "abstain_rate": _safe_div(abstain_n, total),
        "normal_abstain_rate": _safe_div(normal_abstain, normal_n),
        "defect_abstain_rate": _safe_div(defect_abstain, defect_n),
        "selective_n": selective_n,
        "selective_tp": selective_tp,
        "selective_tn": selective_tn,
        "selective_fp": selective_fp,
        "selective_fn": selective_fn,
        "selective_accuracy": _safe_div(selective_tp + selective_tn, selective_n),
        "selective_precision": selective_precision,
        "selective_recall": selective_recall,
        "selective_f1": _safe_div(
            2.0 * selective_precision * selective_recall,
            selective_precision + selective_recall,
        ),
    }


def _alias_index(type_map: Mapping[str, Any], item: str) -> dict[str, str]:
    item_types = type_map.get(item, {})
    if not isinstance(item_types, Mapping):
        return {}

    aliases: dict[str, str] = {}

    def register(value: str, canonical_name: str) -> None:
        key = value.strip().casefold()
        existing = aliases.get(key)
        if existing is not None and existing != canonical_name:
            raise EvaluationDataError(
                f"type_map 동의어가 충돌합니다: {item}/{value.strip()} "
                f"({existing}, {canonical_name})"
            )
        aliases[key] = canonical_name

    for canonical, values in item_types.items():
        if not isinstance(canonical, str):
            continue
        canonical_name = canonical.strip()
        if not canonical_name:
            continue
        register(canonical_name, canonical_name)
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, Iterable) or isinstance(values, Mapping):
            continue
        for value in values:
            if isinstance(value, str) and value.strip():
                register(value, canonical_name)
    return aliases


def _normalize_type(value: Any, item: str, type_map: Mapping[str, Any]) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    canonical = _alias_index(type_map, item).get(raw.casefold())
    return canonical if canonical is not None else raw


def type_metrics(
    rows: Iterable[Row],
    type_map: Mapping[str, Any],
    expected_rows: Iterable[Row] | None = None,
) -> dict[str, Any]:
    """결함 샘플만 유형 분류의 macro-F1을 계산한다.

    예측 유형은 ``type_map``의 동의어로 정규화한다. 사전 등록된 결함 표본
    전체를 support로 고정하고, 오류·누락·ABSTAIN과 clean REJECT가 아닌 결과는
    예측 유형이 없는 것으로 채점한다.
    """

    pairs: list[tuple[str, str | None]] = []
    labels: set[str] = set()
    observed_n = missing_n = error_n = abstain_n = invalid_n = 0

    for _, expected, observed in _records(rows, expected_rows):
        if _ground_truth(expected) is not True:
            continue
        item = expected.get("item")
        label = expected.get("label")
        if not isinstance(item, str) or not isinstance(label, str):
            raise EvaluationDataError("결함 표본에 item과 label 문자열이 필요합니다.")
        truth = _alias_index(type_map, item).get(label.strip().casefold())
        if truth is None or truth.casefold() == "good":
            raise EvaluationDataError(
                f"결함 표본의 label이 type_map에 없습니다: {item}/{label}"
            )

        prediction: str | None = None
        if observed is None:
            missing_n += 1
        else:
            observed_n += 1
            verdict = _usable_verdict(expected, observed)
            if _has_error(observed):
                error_n += 1
            elif verdict is None:
                invalid_n += 1
            elif verdict == "ABSTAIN":
                abstain_n += 1
            elif verdict == "REJECT":
                prediction = _normalize_type(observed.get("defect_type"), item, type_map)
        pairs.append((truth, prediction))
        labels.add(truth)

    per_type: dict[str, dict[str, int | float]] = {}
    f1_values: list[float] = []
    for label in sorted(labels):
        tp = sum(truth == label and prediction == label for truth, prediction in pairs)
        fp = sum(truth != label and prediction == label for truth, prediction in pairs)
        fn = sum(truth == label and prediction != label for truth, prediction in pairs)
        support = sum(truth == label for truth, _ in pairs)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        f1_values.append(f1)
        per_type[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "n": len(pairs),
        "expected_n": len(pairs),
        "observed_n": observed_n,
        "result_n": observed_n,
        "missing_n": missing_n,
        "error_n": error_n,
        "abstain_n": abstain_n,
        "invalid_n": invalid_n,
        "macro_f1": _safe_div(sum(f1_values), len(f1_values)),
        "per_type": per_type,
    }


def _validate_resampling_options(
    *,
    seed: int,
    permutation_samples: int,
    bootstrap_samples: int,
) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed는 정수여야 합니다.")
    for name, value in (
        ("permutation_samples", permutation_samples),
        ("bootstrap_samples", bootstrap_samples),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name}는 1 이상의 정수여야 합니다.")


def _percentile(values: list[float], probability: float) -> float:
    """정렬된 표본 위치 사이를 선형 보간한다."""

    if not values:
        raise ValueError("백분위수를 계산할 값이 없습니다.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def _paired_records(
    rows_a: Iterable[Row],
    rows_b: Iterable[Row],
    expected_rows: Iterable[Row] | None,
    *,
    source: str,
) -> list[tuple[str, Row, Row | None, Row | None]]:
    """사전 등록 ID 순서로 두 조건을 짝짓고 메타데이터를 검증한다."""

    indexed_a = _index_rows(rows_a, source=f"{source} A 결과")
    indexed_b = _index_rows(rows_b, source=f"{source} B 결과")
    expected = (
        _paired_expected_index(indexed_a, indexed_b)
        if expected_rows is None
        else _expected_index(expected_rows)
    )
    unexpected = sorted((indexed_a.keys() | indexed_b.keys()) - expected.keys())
    if unexpected:
        raise EvaluationDataError(
            f"사전 등록에 없는 {source} 결과 id가 있습니다: "
            + ", ".join(unexpected)
        )
    for indexed in (indexed_a, indexed_b):
        for row_id, observed in indexed.items():
            _validate_observed_metadata(expected[row_id], observed, row_id=row_id)

    # 난수를 쓰는 검정이 입력 행 순서에 의존하지 않게 한다.
    return [
        (row_id, expected[row_id], indexed_a.get(row_id), indexed_b.get(row_id))
        for row_id in sorted(expected)
    ]


def _balanced_accuracy_from_correctness(
    truths: list[bool],
    correctness: list[bool],
) -> float:
    normal = [value for truth, value in zip(truths, correctness) if not truth]
    defect = [value for truth, value in zip(truths, correctness) if truth]
    if not normal or not defect:
        raise EvaluationDataError(
            "balanced accuracy 검정에는 정상과 결함 표본이 모두 필요합니다."
        )
    return (sum(normal) / len(normal) + sum(defect) / len(defect)) / 2.0


def _macro_f1_from_predictions(
    truths: list[str],
    predictions: list[str | None],
    labels: tuple[str, ...],
) -> float:
    scores: list[float] = []
    for label in labels:
        tp = sum(
            truth == label and prediction == label
            for truth, prediction in zip(truths, predictions)
        )
        fp = sum(
            truth != label and prediction == label
            for truth, prediction in zip(truths, predictions)
        )
        fn = sum(
            truth == label and prediction != label
            for truth, prediction in zip(truths, predictions)
        )
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        scores.append(_safe_div(2.0 * precision * recall, precision + recall))
    return _safe_div(sum(scores), len(scores))


def _paired_permutation(
    truths: list[Any],
    values_a: list[Any],
    values_b: list[Any],
    metric: Any,
    *,
    seed: int,
    samples: int,
) -> tuple[float, str, int]:
    """짝 내 A/B 교환으로 양측 순열 검정을 수행한다."""

    metric_a = metric(truths, values_a)
    metric_b = metric(truths, values_b)
    observed_delta = metric_b - metric_a
    swappable = [
        index for index, (left, right) in enumerate(zip(values_a, values_b))
        if left != right
    ]
    tolerance = 1e-12

    def permuted_delta(mask: int | None, generator: random.Random | None) -> float:
        permuted_a = list(values_a)
        permuted_b = list(values_b)
        for offset, index in enumerate(swappable):
            should_swap = (
                bool(mask & (1 << offset))
                if mask is not None
                else bool(generator and generator.getrandbits(1))
            )
            if should_swap:
                permuted_a[index], permuted_b[index] = (
                    permuted_b[index],
                    permuted_a[index],
                )
        return metric(truths, permuted_b) - metric(truths, permuted_a)

    if len(swappable) <= EXACT_PERMUTATION_MAX_PAIRS:
        evaluated = 1 << len(swappable)
        extreme = sum(
            abs(permuted_delta(mask, None))
            >= abs(observed_delta) - tolerance
            for mask in range(evaluated)
        )
        return extreme / evaluated, "exact", evaluated

    generator = random.Random(seed)
    extreme = sum(
        abs(permuted_delta(None, generator))
        >= abs(observed_delta) - tolerance
        for _ in range(samples)
    )
    # Monte-Carlo p-value가 0이 되지 않도록 관측 배치를 포함한다.
    return (extreme + 1) / (samples + 1), "monte_carlo", samples


def _paired_stratified_bootstrap(
    truths: list[Any],
    values_a: list[Any],
    values_b: list[Any],
    strata: list[Any],
    metric: Any,
    *,
    seed: int,
    samples: int,
) -> tuple[float, float]:
    """각 층의 크기와 A/B 짝을 유지한 백분위 부트스트랩 CI."""

    grouped: dict[Any, list[int]] = {}
    for index, stratum in enumerate(strata):
        grouped.setdefault(stratum, []).append(index)
    ordered_groups = [grouped[key] for key in sorted(grouped, key=repr)]
    generator = random.Random(seed)
    deltas: list[float] = []
    for _ in range(samples):
        sampled_indices = [
            generator.choice(indices)
            for indices in ordered_groups
            for _ in range(len(indices))
        ]
        sampled_truths = [truths[index] for index in sampled_indices]
        sampled_a = [values_a[index] for index in sampled_indices]
        sampled_b = [values_b[index] for index in sampled_indices]
        deltas.append(
            metric(sampled_truths, sampled_b) - metric(sampled_truths, sampled_a)
        )
    return _percentile(deltas, 0.025), _percentile(deltas, 0.975)


def paired_balanced_accuracy_test(
    rows_a: Iterable[Row],
    rows_b: Iterable[Row],
    expected_rows: Iterable[Row],
    *,
    seed: int = DEFAULT_RESAMPLING_SEED,
    permutation_samples: int = DEFAULT_PERMUTATION_SAMPLES,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    """H1용 짝지은 balanced accuracy 차이를 검정한다.

    두 조건의 동일 ID를 짝으로 유지하고, 정상/결함 층을
    고정한다. ``delta``는 B-A이며 누락·오류·ABSTAIN은 오답이다.
    """

    _validate_resampling_options(
        seed=seed,
        permutation_samples=permutation_samples,
        bootstrap_samples=bootstrap_samples,
    )
    records = _paired_records(
        rows_a, rows_b, expected_rows, source="balanced accuracy 비교"
    )
    truths = [bool(_ground_truth(expected)) for _, expected, _, _ in records]
    correct_a = [
        _is_correct(expected, observed_a)
        for _, expected, observed_a, _ in records
    ]
    correct_b = [
        _is_correct(expected, observed_b)
        for _, expected, _, observed_b in records
    ]
    metric_a = _balanced_accuracy_from_correctness(truths, correct_a)
    metric_b = _balanced_accuracy_from_correctness(truths, correct_b)
    p_value, permutation_method, evaluated = _paired_permutation(
        truths,
        correct_a,
        correct_b,
        _balanced_accuracy_from_correctness,
        seed=seed,
        samples=permutation_samples,
    )
    ci_low, ci_high = _paired_stratified_bootstrap(
        truths,
        correct_a,
        correct_b,
        truths,
        _balanced_accuracy_from_correctness,
        seed=seed,
        samples=bootstrap_samples,
    )
    normal_n = sum(not truth for truth in truths)
    defect_n = sum(truths)
    return {
        "n": len(records),
        "normal_n": normal_n,
        "defect_n": defect_n,
        "metric": "balanced_accuracy",
        "a": metric_a,
        "b": metric_b,
        "delta": metric_b - metric_a,
        "delta_ci95_low": ci_low,
        "delta_ci95_high": ci_high,
        "confidence_level": 0.95,
        "p_value": p_value,
        "alternative": "two-sided",
        "permutation_method": permutation_method,
        "permutation_samples": evaluated,
        "requested_permutation_samples": permutation_samples,
        "swappable_pairs": sum(
            left != right for left, right in zip(correct_a, correct_b)
        ),
        "exact_permutation_max_pairs": EXACT_PERMUTATION_MAX_PAIRS,
        "bootstrap_method": "paired_stratified_percentile",
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "strata": {"normal": normal_n, "defect": defect_n},
    }


def _type_truth_and_prediction(
    expected: Row,
    observed: Row | None,
    type_map: Mapping[str, Any],
) -> tuple[str, str, str | None]:
    item = expected.get("item")
    label = expected.get("label")
    if not isinstance(item, str) or not isinstance(label, str):
        raise EvaluationDataError("결함 표본에 item과 label 문자열이 필요합니다.")
    truth = _alias_index(type_map, item).get(label.strip().casefold())
    if truth is None or truth.casefold() == "good":
        raise EvaluationDataError(
            f"결함 표본의 label이 type_map에 없습니다: {item}/{label}"
        )
    prediction = None
    if _usable_verdict(expected, observed) == "REJECT" and observed is not None:
        prediction = _normalize_type(observed.get("defect_type"), item, type_map)
    return item, truth, prediction


def paired_macro_f1_test(
    rows_a: Iterable[Row],
    rows_b: Iterable[Row],
    type_map: Mapping[str, Any],
    expected_rows: Iterable[Row],
    *,
    seed: int = DEFAULT_RESAMPLING_SEED,
    permutation_samples: int = DEFAULT_PERMUTATION_SAMPLES,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    """H2용 짝지은 결함 유형 macro-F1 차이를 검정한다.

    이진 정오만 비교하는 McNemar를 사용하지 않고, 각 결함
    ID의 A/B 유형 예측을 교환한 순열 검정을 쓴다. 부트스트랩은
    품목/진실 유형 층을 유지한다. ``delta``는 B-A이다.
    """

    _validate_resampling_options(
        seed=seed,
        permutation_samples=permutation_samples,
        bootstrap_samples=bootstrap_samples,
    )
    records = _paired_records(rows_a, rows_b, expected_rows, source="macro-F1 비교")
    truths: list[str] = []
    predictions_a: list[str | None] = []
    predictions_b: list[str | None] = []
    strata: list[tuple[str, str]] = []
    for _, expected, observed_a, observed_b in records:
        if _ground_truth(expected) is not True:
            continue
        item, truth, prediction_a = _type_truth_and_prediction(
            expected, observed_a, type_map
        )
        _, repeated_truth, prediction_b = _type_truth_and_prediction(
            expected, observed_b, type_map
        )
        if repeated_truth != truth:
            raise EvaluationDataError("동일 결함 ID의 진실 유형이 일치하지 않습니다.")
        truths.append(truth)
        predictions_a.append(prediction_a)
        predictions_b.append(prediction_b)
        strata.append((item, truth))
    if not truths:
        raise EvaluationDataError("macro-F1 검정에는 결함 표본이 필요합니다.")

    labels = tuple(sorted(set(truths)))

    def metric(sample_truths: list[str], predictions: list[str | None]) -> float:
        return _macro_f1_from_predictions(sample_truths, predictions, labels)

    metric_a = metric(truths, predictions_a)
    metric_b = metric(truths, predictions_b)
    p_value, permutation_method, evaluated = _paired_permutation(
        truths,
        predictions_a,
        predictions_b,
        metric,
        seed=seed,
        samples=permutation_samples,
    )
    ci_low, ci_high = _paired_stratified_bootstrap(
        truths,
        predictions_a,
        predictions_b,
        strata,
        metric,
        seed=seed,
        samples=bootstrap_samples,
    )
    stratum_counts: dict[str, int] = {}
    for item, truth in strata:
        name = f"{item}/{truth}"
        stratum_counts[name] = stratum_counts.get(name, 0) + 1
    return {
        "n": len(truths),
        "type_n": len(labels),
        "metric": "macro_f1",
        "a": metric_a,
        "b": metric_b,
        "delta": metric_b - metric_a,
        "delta_ci95_low": ci_low,
        "delta_ci95_high": ci_high,
        "confidence_level": 0.95,
        "p_value": p_value,
        "alternative": "two-sided",
        "permutation_method": permutation_method,
        "permutation_samples": evaluated,
        "requested_permutation_samples": permutation_samples,
        "swappable_pairs": sum(
            left != right for left, right in zip(predictions_a, predictions_b)
        ),
        "exact_permutation_max_pairs": EXACT_PERMUTATION_MAX_PAIRS,
        "bootstrap_method": "paired_stratified_percentile",
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "strata": stratum_counts,
    }


def _exact_mcnemar_p_value(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = min(b, c)
    numerator = sum(math.comb(discordant, k) for k in range(tail + 1))
    return min(1.0, 2.0 * numerator / (1 << discordant))


def _mcnemar_table(pairs: Iterable[tuple[bool, bool]]) -> dict[str, int | float]:
    both_correct = both_wrong = b = c = 0
    for a_correct, b_correct in pairs:
        if a_correct and b_correct:
            both_correct += 1
        elif not a_correct and not b_correct:
            both_wrong += 1
        elif a_correct:
            b += 1
        else:
            c += 1

    return {
        "n": both_correct + both_wrong + b + c,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "b": b,
        "c": c,
        "p_value": _exact_mcnemar_p_value(b, c),
    }


def _paired_expected_index(
    indexed_a: Mapping[str, Row],
    indexed_b: Mapping[str, Row],
) -> dict[str, Row]:
    expected: dict[str, Row] = {}
    for row_id in indexed_a.keys() | indexed_b.keys():
        candidates = [
            indexed[row_id]
            for indexed in (indexed_a, indexed_b)
            if row_id in indexed
        ]
        canonical = _expected_index(
            [candidates[0]], source="McNemar 결과 메타데이터"
        )[row_id]
        for candidate in candidates[1:]:
            _validate_observed_metadata(canonical, candidate, row_id=row_id)
        expected[row_id] = canonical
    return expected


def mcnemar(
    rows_a: Iterable[Row],
    rows_b: Iterable[Row],
    expected_rows: Iterable[Row] | None = None,
) -> dict[str, dict[str, int | float]]:
    """같은 id를 짝지어 양측 McNemar 정확 검정을 수행한다.

    ``b``는 A만 정답, ``c``는 B만 정답인 불일치 표의 의미다.
    사전 등록 ID 전체를 짝지으며 한쪽의 누락·오류·ABSTAIN도 false로 남긴다.
    """

    records = _paired_records(
        rows_a, rows_b, expected_rows, source="McNemar"
    )
    strata: dict[str, list[tuple[bool, bool]]] = {
        "overall": [],
        "normal": [],
        "defect": [],
    }

    for _, expected_row, row_a, row_b in records:
        pair = (
            _is_correct(expected_row, row_a),
            _is_correct(expected_row, row_b),
        )
        strata["overall"].append(pair)
        strata["defect" if _ground_truth(expected_row) else "normal"].append(pair)

    return {name: _mcnemar_table(pairs) for name, pairs in strata.items()}


def confidence_flag_validity(
    rows: Iterable[Row],
    expected_rows: Iterable[Row] | None = None,
) -> dict[str, int | float | bool | None]:
    """low 플래그가 자동 판정 오답을 농축하는지 계산한다.

    주 비교는 오류 없는 PASS/REJECT 중 ``low``와 ``high``다.
    ABSTAIN·실행 오류·누락·무효 판정은 종단 오답률에는 남지만
    플래그 농축도의 분자나 분모에 합치지 않는다.
    """

    total_count = total_errors = 0
    selected_count = selected_errors = 0
    low_count = low_errors = 0
    non_low_count = non_low_errors = 0
    abstain_count = error_count = missing_count = invalid_count = 0
    invalid_confidence_count = invalid_confidence_errors = 0
    for _, expected, observed in _records(rows, expected_rows):
        total_count += 1
        is_error = not _is_correct(expected, observed)
        total_errors += int(is_error)
        if observed is None:
            missing_count += 1
            continue
        if _has_error(observed):
            error_count += 1
            continue
        verdict = _usable_verdict(expected, observed)
        if verdict == "ABSTAIN":
            abstain_count += 1
            continue
        if verdict not in {"PASS", "REJECT"}:
            invalid_count += 1
            continue

        selected_count += 1
        selected_errors += int(is_error)
        confidence = observed.get("confidence")
        normalized_confidence = (
            confidence.strip().casefold()
            if isinstance(confidence, str)
            else None
        )
        if normalized_confidence == "low":
            low_count += 1
            low_errors += int(is_error)
        elif normalized_confidence == "high":
            non_low_count += 1
            non_low_errors += int(is_error)
        else:
            invalid_confidence_count += 1
            invalid_confidence_errors += int(is_error)

    overall_error_rate = _safe_div(total_errors, total_count)
    selected_error_rate = (
        _safe_div(selected_errors, selected_count) if selected_count else None
    )
    low_error_rate = _safe_div(low_errors, low_count) if low_count else None
    non_low_error_rate = (
        _safe_div(non_low_errors, non_low_count) if non_low_count else None
    )
    enrichment_ratio = (
        low_error_rate / non_low_error_rate
        if low_error_rate is not None
        and non_low_error_rate is not None
        and non_low_error_rate > 0.0
        else None
    )
    legacy_ratio = (
        low_error_rate / overall_error_rate
        if low_error_rate is not None and overall_error_rate > 0.0
        else None
    )
    low_ci = wilson_interval(low_errors, low_count)
    non_low_ci = wilson_interval(non_low_errors, non_low_count)
    selected_ci = wilson_interval(selected_errors, selected_count)
    return {
        "total_count": total_count,
        "total_errors": total_errors,
        "overall_error_rate": overall_error_rate,
        "selected_count": selected_count,
        "selected_errors": selected_errors,
        "selected_error_rate": selected_error_rate,
        "selected_error_ci95_low": selected_ci[0],
        "selected_error_ci95_high": selected_ci[1],
        "low_count": low_count,
        "low_errors": low_errors,
        "low_error_rate": low_error_rate,
        "low_error_ci95_low": low_ci[0],
        "low_error_ci95_high": low_ci[1],
        "non_low_count": non_low_count,
        "non_low_errors": non_low_errors,
        "non_low_error_rate": non_low_error_rate,
        "non_low_error_ci95_low": non_low_ci[0],
        "non_low_error_ci95_high": non_low_ci[1],
        "enrichment_ratio": enrichment_ratio,
        "enrichment_available": enrichment_ratio is not None,
        # R3 보고서 호환용: low/종단 전체 오답률은 보조 값으로만 남겨 둔다.
        "error_rate_ratio": legacy_ratio,
        "abstain_count": abstain_count,
        "abstain_rate": _safe_div(abstain_count, total_count),
        "error_count": error_count,
        "error_rate": _safe_div(error_count, total_count),
        "missing_count": missing_count,
        "missing_rate": _safe_div(missing_count, total_count),
        "invalid_count": invalid_count,
        "invalid_rate": _safe_div(invalid_count, total_count),
        "invalid_confidence_count": invalid_confidence_count,
        "invalid_confidence_errors": invalid_confidence_errors,
    }


def _source_texts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []

    texts: list[str] = []
    for entry in value:
        if isinstance(entry, str):
            texts.append(entry)
        elif isinstance(entry, Mapping) and isinstance(entry.get("text"), str):
            texts.append(entry["text"])
    return texts


def _has_exact_ngram(text: str, sources: list[str], size: int = 8) -> bool:
    if len(text) < size:
        return False
    return any(text[index : index + size] in source for index in range(len(text) - size + 1) for source in sources)


def citation_fidelity(rows: Iterable[Row]) -> dict[str, int | float]:
    """보고서 관찰문의 8자 연속 n-gram 원문 인용 충실도를 계산한다.

    대소문자나 공백을 정규화하지 않고 각 관찰문을 원문과 비교한다.
    """

    total = matched = 0
    for row in rows:
        if _has_error(row):
            continue
        report = row.get("report")
        if not isinstance(report, Mapping):
            continue
        observations = report.get("observations")
        if not isinstance(observations, list):
            continue
        sources = _source_texts(row.get("observations_text"))
        for observation in observations:
            if not isinstance(observation, Mapping) or not isinstance(observation.get("text"), str):
                continue
            total += 1
            matched += int(_has_exact_ngram(observation["text"], sources))

    return {
        "total_citations": total,
        "matched_citations": matched,
        "citation_fidelity": _safe_div(matched, total),
    }


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def cost_latency(rows: Iterable[Row]) -> dict[str, int | float | None]:
    """조건별 실행 지연·비용·HITL 지표를 집계한다.

    오류 행도 실행 비용이 발생할 수 있으므로 제외하지 않는다. 누락된 수치는
    명시적 기본값으로 처리하지만, 존재하는 잘못된 값은 집계를 중단한다.
    """

    materialized = [row for row in rows if isinstance(row, Mapping)]
    count = len(materialized)
    latencies: list[float] = []
    tokens: list[int] = []
    input_tokens: list[int] = []
    output_tokens: list[int] = []
    split_tokens_complete = True
    costs: list[float] = []
    compute_proxy_costs: list[float] = []
    attempts: list[int] = []
    validation_rejections: list[int] = []
    retries: list[int] = []
    replans: list[int] = []
    provider_requests: list[int] = []
    vlm_requests: list[int] = []
    unknown_usage_n = 0
    for index, row in enumerate(materialized, start=1):
        prefix = f"평가 결과 {index}"
        try:
            latencies.append(nonnegative_finite_float(
                row.get("latency_s", MISSING),
                name=f"{prefix}.latency_s",
                default=0.0,
            ))
            tokens.append(nonnegative_int(
                row.get("tokens", MISSING),
                name=f"{prefix}.tokens",
                default=0,
            ))
            accounting = row.get("token_accounting", "total_only")
            if accounting in {"provider_split", "estimated_split", "mixed_split"}:
                row_input = nonnegative_int(
                    row.get("input_tokens", MISSING),
                    name=f"{prefix}.input_tokens",
                )
                row_output = nonnegative_int(
                    row.get("output_tokens", MISSING),
                    name=f"{prefix}.output_tokens",
                )
                if row_input + row_output != tokens[-1]:
                    raise UsageValueError(
                        f"{prefix} 입력·출력 토큰 합이 전체 토큰과 다릅니다."
                    )
                input_tokens.append(row_input)
                output_tokens.append(row_output)
            elif accounting == "total_only":
                if row.get("input_tokens") is not None or row.get("output_tokens") is not None:
                    raise UsageValueError(
                        f"{prefix} total_only 토큰에 분리값이 있습니다."
                    )
                split_tokens_complete = False
            else:
                raise UsageValueError(f"{prefix}.token_accounting 값이 올바르지 않습니다.")
            if "estimated_usd" in row:
                raw_cost = row["estimated_usd"]
                cost_name = f"{prefix}.estimated_usd"
            else:
                raw_cost = row.get("usd", MISSING)
                cost_name = f"{prefix}.usd"
            costs.append(nonnegative_finite_float(
                raw_cost,
                name=cost_name,
                default=0.0,
            ))
            compute_proxy_costs.append(nonnegative_finite_float(
                row.get("compute_proxy_usd", MISSING),
                name=f"{prefix}.compute_proxy_usd",
                default=0.0,
            ))
            attempts.append(nonnegative_int(
                row.get("validation_attempts", MISSING),
                name=f"{prefix}.validation_attempts",
                default=0,
            ))
            validation_rejections.append(nonnegative_int(
                row.get("validation_rejections", MISSING),
                name=f"{prefix}.validation_rejections",
                default=0,
            ))
            retries.append(nonnegative_int(
                row.get("retry_count", MISSING),
                name=f"{prefix}.retry_count",
                default=0,
            ))
            replans.append(nonnegative_int(
                row.get("replan_rounds", MISSING),
                name=f"{prefix}.replan_rounds",
                default=0,
            ))
            provider_requests.append(nonnegative_int(
                row.get("successful_provider_requests", MISSING),
                name=f"{prefix}.successful_provider_requests",
                default=0,
            ))
            vlm_requests.append(nonnegative_int(
                row.get("successful_vlm_requests", MISSING),
                name=f"{prefix}.successful_vlm_requests",
                default=0,
            ))
            if row.get("usage_status") in {
                "unknown_after_error",
                "unknown_tokens_after_success",
            }:
                unknown_usage_n += 1
        except UsageValueError as exc:
            raise EvaluationDataError(str(exc)) from exc
    approvals = [
        _as_bool(row.get("approval_required")) is True
        for row in materialized
    ]

    result: dict[str, int | float] = {
        "n": count,
        "latency_p50_s": _percentile(latencies, 0.50),
        "latency_p95_s": _percentile(latencies, 0.95),
        "mean_tokens": (
            None if unknown_usage_n else _safe_div(sum(tokens), count)
        ),
        "mean_input_tokens": (
            _safe_div(sum(input_tokens), count)
            if split_tokens_complete and count and not unknown_usage_n else None
        ),
        "mean_output_tokens": (
            _safe_div(sum(output_tokens), count)
            if split_tokens_complete and count and not unknown_usage_n else None
        ),
        "mean_usd": (
            None if unknown_usage_n else _safe_div(sum(costs), count)
        ),
        "mean_compute_proxy_usd": (
            None if unknown_usage_n else _safe_div(sum(compute_proxy_costs), count)
        ),
        "known_estimated_usd_sum": sum(
            cost
            for cost, row in zip(costs, materialized)
            if row.get("usage_status") not in {
                "unknown_after_error",
                "unknown_tokens_after_success",
            }
        ),
        "known_cost_n": count - unknown_usage_n,
        "unknown_usage_n": unknown_usage_n,
        "hitl_rate": _safe_div(sum(approvals), count),
        "mean_validation_attempts": _safe_div(sum(attempts), count),
    }
    if count:
        result["mean_validation_rejections"] = _safe_div(
            sum(validation_rejections), count
        )
        result["mean_retry_count"] = _safe_div(sum(retries), count)
        result["mean_replan_rounds"] = _safe_div(sum(replans), count)
        result["mean_successful_provider_requests"] = _safe_div(
            sum(provider_requests), count
        )
        result["mean_successful_vlm_requests"] = _safe_div(
            sum(vlm_requests), count
        )
    return result
