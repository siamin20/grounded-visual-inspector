"""결함 조치 가이드를 baseline과 agent가 같은 규칙으로 적용하게 한다."""

from pathlib import Path
import re


REJECT_ACTIONS = {"재작업", "폐기", "라인정지"}
_ACTION_RISK_ORDER = ("라인정지", "폐기", "재작업")
_GUIDE_DIR = Path(__file__).resolve().parents[1] / "data" / "guides"


def parse_guide_actions(guide_text: str) -> list[str]:
    """가이드의 조치 옵션을 등장 순서대로 파싱한다."""
    if not isinstance(guide_text, str):
        return []
    match = re.search(r"조치\s*옵션\s*:\s*([^\n]+)", guide_text)
    if match is None:
        return []
    return list(
        dict.fromkeys(
            action.strip()
            for action in match.group(1).split("|")
            if action.strip() in REJECT_ACTIONS
        )
    )


def load_guide_actions(defect_type: object) -> list[str]:
    """결함 유형의 고정된 조치 가이드를 읽어 허용 집합을 돌려준다."""
    if not isinstance(defect_type, str) or re.fullmatch(
        r"[a-z0-9_]+", defect_type
    ) is None:
        return []
    path = _GUIDE_DIR / f"{defect_type}.md"
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return []
    return parse_guide_actions(content)


def resolve_reject_actions(
    raw_actions: object,
    allowed_actions: list[str],
) -> list[str]:
    """허용 조치만 남기고, 하나도 없으면 가장 보수적인 허용 조치를 쓴다."""
    selected: list[str] = []
    if isinstance(raw_actions, list):
        selected = list(
            dict.fromkeys(
                action
                for action in raw_actions
                if isinstance(action, str) and action in allowed_actions
            )
        )
    if selected:
        return selected
    fallback = next(
        (action for action in _ACTION_RISK_ORDER if action in allowed_actions),
        None,
    )
    return [fallback] if fallback else []
