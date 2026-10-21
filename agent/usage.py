"""비용 성능과 평가에서 공통으로 사용하는 수치 값 검증."""

from __future__ import annotations

import math
from typing import Any


MISSING = object()


class UsageValueError(ValueError):
    """사용량 또는 비용 값이 불변 조건을 만족하지 못할 때 발생한다."""


def nonnegative_finite_float(
    value: Any = MISSING,
    *,
    name: str = "value",
    default: Any = MISSING,
) -> float:
    """bool이나 문자열 변환 없이 0 이상의 유한한 실수를 반환한다.

    ``default``는 값이 아예 전달되지 않았을 때만 적용한다. 명시적인
    ``None``이나 잘못된 값은 기본값으로 바꾸지 않는다.
    """

    if value is MISSING:
        if default is MISSING:
            raise UsageValueError(f"{name}이(가) 누락되었습니다.")
        value = default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UsageValueError(f"{name}은 0 이상의 유한한 수여야 합니다.")
    try:
        number = float(value)
    except OverflowError as exc:
        raise UsageValueError(
            f"{name}은 0 이상의 유한한 수여야 합니다."
        ) from exc
    if not math.isfinite(number) or number < 0:
        raise UsageValueError(f"{name}은 0 이상의 유한한 수여야 합니다.")
    return number


def nonnegative_int(
    value: Any = MISSING,
    *,
    name: str = "value",
    default: Any = MISSING,
) -> int:
    """bool이나 문자열 변환 없이 0 이상의 정수를 반환한다."""

    if value is MISSING:
        if default is MISSING:
            raise UsageValueError(f"{name}이(가) 누락되었습니다.")
        value = default
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UsageValueError(f"{name}은 0 이상의 정수여야 합니다.")
    return value
