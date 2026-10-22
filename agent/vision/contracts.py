"""Vision 구성요소 간 공통 계약.

Cross-Checker와 MCP 경계에서 사용하는 점검 항목이
서로 달라지지 않도록 이 모듈에서 한 번만 고정한다.
"""

from __future__ import annotations

from typing import Any


CHECKLIST_KEYS = (
    "scratch",
    "break_or_crack",
    "contamination",
    "discoloration",
    "bend_or_deformation",
    "thread_damage",
)


def normalized_checklist(value: Any) -> dict[str, bool] | None:
    """exact-key·strict-bool 체크리스트를 정규화하고 아니면 ``None``.

    ``bool``은 ``int``의 하위 클래스이므로, 0/1과 문자열 ``null``을
    모두 계약 위반으로 거부한다.
    """
    if not isinstance(value, dict) or set(value) != set(CHECKLIST_KEYS):
        return None
    if any(type(value[key]) is not bool for key in CHECKLIST_KEYS):
        return None
    return {key: value[key] for key in CHECKLIST_KEYS}
