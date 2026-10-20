"""평가 결과에만 의존하는 코드 기반 평가기다.

이 모듈은 외부 모델이나 네트워크를 호출하지 않는 재현 가능한 순수 함수로 구성된다.
``error``가 기록된 행은 채점에서 제외하고,
분모가 0인 비율은 0.0으로 반환한다.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


Row = Mapping[str, Any]


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    """분모가 0이면 0.0을 반환한다."""

    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _has_error(row: Row) -> bool:
    error = row.get("error")
    return error not in (None, "", False)


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


def _prediction(row: Row) -> bool | None:
    verdict = row.get("verdict")
    if not isinstance(verdict, str):
        return None
    normalized = verdict.strip().upper()
    if normalized == "REJECT":
        return True
    if normalized == "PASS":
        return False
    return None


def _outcome(row: Row) -> tuple[bool, bool] | None:
    if _has_error(row):
        return None
    truth = _ground_truth(row)
    prediction = _prediction(row)
    if truth is None or prediction is None:
        return None
    return truth, prediction


def binary_metrics(rows: Iterable[Row]) -> dict[str, int | float]:
    """결함 여부를 기준으로 이진 분류 지표를 계산한다.

    ``REJECT``를 양성(결함)으로 간주한다. 오류 행이나 정답이 없는 행은
    분모에서 제외한다.
    """

    tp = tn = fp = fn = 0
    for row in rows:
        outcome = _outcome(row)
        if outcome is None:
            continue
        truth, prediction = outcome
        if truth and prediction:
            tp += 1
        elif not truth and not prediction:
            tn += 1
        elif not truth and prediction:
            fp += 1
        else:
            fn += 1

    total = tp + tn + fp + fn
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)

    return {
        "n": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": _safe_div(tp + tn, total),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2.0 * precision * recall, precision + recall),
        "fpr": _safe_div(fp, fp + tn),
        "fnr": _safe_div(fn, fn + tp),
    }


def _alias_index(type_map: Mapping[str, Any], item: str) -> dict[str, str]:
    item_types = type_map.get(item, {})
    if not isinstance(item_types, Mapping):
        return {}

    aliases: dict[str, str] = {}
    for canonical, values in item_types.items():
        if not isinstance(canonical, str):
            continue
        canonical_name = canonical.strip()
        if not canonical_name:
            continue
        aliases.setdefault(canonical_name.casefold(), canonical_name)
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, Iterable) or isinstance(values, Mapping):
            continue
        for value in values:
            if isinstance(value, str) and value.strip():
                aliases.setdefault(value.strip().casefold(), canonical_name)
    return aliases


def _normalize_type(value: Any, item: str, type_map: Mapping[str, Any]) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    canonical = _alias_index(type_map, item).get(raw.casefold())
    return canonical if canonical is not None else raw


def type_metrics(rows: Iterable[Row], type_map: Mapping[str, Any]) -> dict[str, Any]:
    """결함 샘플만 유형 분류의 macro-F1을 계산한다.

    예측 유형은 ``type_map``의 동의어로 정규화한다. macro 평균은 해당 분할에
    실제로 나타난 정답 유형만 포함해, 자료에 없던 유형이 평균에 영향을
    주지 않게 한다.
    """

    pairs: list[tuple[str, str | None]] = []
    labels: set[str] = set()

    for row in rows:
        if _has_error(row) or _ground_truth(row) is not True:
            continue
        item = row.get("item")
        label = row.get("label")
        if not isinstance(item, str) or not isinstance(label, str):
            continue
        truth = _normalize_type(label, item, type_map)
        if truth in (None, "good"):
            continue
        prediction = _normalize_type(row.get("defect_type"), item, type_map)
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
        "macro_f1": _safe_div(sum(f1_values), len(f1_values)),
        "per_type": per_type,
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


def _rows_by_id(rows: Iterable[Row]) -> dict[Any, Row]:
    indexed: dict[Any, Row] = {}
    for row in rows:
        if _has_error(row) or "id" not in row:
            continue
        row_id = row.get("id")
        try:
            hash(row_id)
        except TypeError:
            continue
        if row_id is not None:
            indexed.setdefault(row_id, row)
    return indexed


def mcnemar(rows_a: Iterable[Row], rows_b: Iterable[Row]) -> dict[str, dict[str, int | float]]:
    """같은 id를 짝지어 양측 McNemar 정확 검정을 수행한다.

    ``b``는 A만 정답, ``c``는 B만 정답인 불일치 표의 의미다. 두 행의 라벨이
    다르거나 정답이 없는 짝은 제외한다.
    """

    indexed_a = _rows_by_id(rows_a)
    indexed_b = _rows_by_id(rows_b)
    strata: dict[str, list[tuple[bool, bool]]] = {
        "overall": [],
        "normal": [],
        "defect": [],
    }

    for row_id in indexed_a.keys() & indexed_b.keys():
        outcome_a = _outcome(indexed_a[row_id])
        outcome_b = _outcome(indexed_b[row_id])
        if outcome_a is None or outcome_b is None:
            continue
        truth_a, prediction_a = outcome_a
        truth_b, prediction_b = outcome_b
        if truth_a != truth_b:
            continue
        pair = (prediction_a == truth_a, prediction_b == truth_b)
        strata["overall"].append(pair)
        strata["defect" if truth_a else "normal"].append(pair)

    return {name: _mcnemar_table(pairs) for name, pairs in strata.items()}


def confidence_flag_validity(rows: Iterable[Row]) -> dict[str, int | float]:
    """low 플래그가 실제 오답을 집중시키는지 비율과 건수를 반환한다."""

    total_count = total_errors = low_count = low_errors = 0
    for row in rows:
        outcome = _outcome(row)
        if outcome is None:
            continue
        truth, prediction = outcome
        is_error = truth != prediction
        is_low = isinstance(row.get("confidence"), str) and row["confidence"].strip().casefold() == "low"
        total_count += 1
        total_errors += int(is_error)
        if is_low:
            low_count += 1
            low_errors += int(is_error)

    overall_error_rate = _safe_div(total_errors, total_count)
    low_error_rate = _safe_div(low_errors, low_count)
    return {
        "total_count": total_count,
        "total_errors": total_errors,
        "overall_error_rate": overall_error_rate,
        "low_count": low_count,
        "low_errors": low_errors,
        "low_error_rate": low_error_rate,
        "error_rate_ratio": _safe_div(low_error_rate, overall_error_rate),
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


def cost_latency(rows: Iterable[Row]) -> dict[str, int | float]:
    """조건별 실행 지연·비용·HITL 지표를 집계한다.

    오류 행도 실행 비용이 발생할 수 있으므로 제외하지 않는다. 수치가 아닌 값은
    0으로 처리하며, 오류 행에 기록된 비용도 포함한다.
    """

    materialized = [row for row in rows if isinstance(row, Mapping)]
    count = len(materialized)
    latencies = [max(0.0, _number(row.get("latency_s"))) for row in materialized]
    tokens = [max(0.0, _number(row.get("tokens"))) for row in materialized]
    costs = [max(0.0, _number(row.get("usd"))) for row in materialized]
    attempts = [max(0.0, _number(row.get("validation_attempts"))) for row in materialized]
    approvals = [
        _as_bool(row.get("approval_required")) is True
        for row in materialized
    ]

    return {
        "n": count,
        "latency_p50_s": _percentile(latencies, 0.50),
        "latency_p95_s": _percentile(latencies, 0.95),
        "mean_tokens": _safe_div(sum(tokens), count),
        "mean_usd": _safe_div(sum(costs), count),
        "hitl_rate": _safe_div(sum(approvals), count),
        "mean_validation_attempts": _safe_div(sum(attempts), count),
    }
