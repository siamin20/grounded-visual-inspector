"""시각 관찰의 판독 가능성을 공유하는 규칙."""

from __future__ import annotations

from typing import Any
import unicodedata


_READING_ACTIONS = (
    "판독",
    "식별",
    "확인",
    "판별",
    "판단",
    "판정",
    "구분",
    "구별",
    "확정",
    "가리",
    "확신",
    "결론",
    "단정",
)
_INABILITY_MARKERS = (
    "불가",
    "할수없",
    "어렵",
    "힘들",
    "곤란",
    "되지않",
    "애매",
    "보류",
    "유보",
    "부족",
    "수없",
    "제한",
)
_DIRECT_UNREADABLE_MARKERS = (
    "알수없",
    "불확실",
    "모호",
    "너무흐리",
    "너무흐려",
    "흐릿",
    "분명하지않",
    "명확하지않",
    "불분명",
    "확신이없",
    "신뢰할수없",
    "미확정",
    "불명확",
)
_ENGLISH_READING_ACTIONS = ("determine", "assess", "identify", "verify", "tell")
_ENGLISH_INABILITY_MARKERS = (
    "unable",
    "cannot",
    "cant",
    "difficult",
    "hardto",
    "impossible",
    "unclear",
)


def is_unreadable_text(text: Any) -> bool:
    """표현이 달라도 결론을 확정할 수 없으면 판독 불가로 본다.

    표현 자체로 옥판을 피하기 아니라 ``확인·판별·판단`` 등의 행외
    ``불가·할 수 없음·어려움``이 같이 있을 때만 보수적으로 보류한다.
    """
    if not isinstance(text, str):
        return True
    compact = unicodedata.normalize("NFC", "".join(text.lower().split()))
    if not compact:
        return True
    ascii_compact = compact.replace("'", "").replace("’", "")
    if any(
        marker in ascii_compact
        for marker in ("unreadable", "cannotread", "tooblurry", "notenoughdetail")
    ):
        return True
    if any(
        unicodedata.normalize("NFC", marker) in compact
        for marker in _DIRECT_UNREADABLE_MARKERS
    ):
        return True
    if (
        any(action in ascii_compact for action in _ENGLISH_READING_ACTIONS)
        and any(
            marker in ascii_compact for marker in _ENGLISH_INABILITY_MARKERS
        )
    ):
        return True
    return (
        any(
            unicodedata.normalize("NFC", action) in compact
            for action in _READING_ACTIONS
        )
        and any(
            unicodedata.normalize("NFC", marker) in compact
            for marker in _INABILITY_MARKERS
        )
    )
