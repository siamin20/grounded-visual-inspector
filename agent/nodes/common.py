"""노드가 공유하는 데이터 로더와 상태 헬퍼."""

import json
from pathlib import Path
from typing import Any

from agent.state import AgentState
from agent.typemap import load_type_map


ROOT = Path(__file__).resolve().parents[2]
LOTS_PATH = ROOT / "data" / "lots.json"
KNOWN_ITEMS = frozenset(load_type_map())


def load_lot(image_path: str | None) -> dict:
    """이미지 경로의 `<item>/test/<type>/<stem>`을 lots 키로 변환한다."""
    if not image_path:
        return {}
    try:
        lots = json.loads(LOTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    path = Path(image_path)
    parts = path.parts
    try:
        test_at = parts.index("test")
        item = parts[test_at - 1]
        defect_type = parts[test_at + 1]
        sample_id = f"{item}-{defect_type}-{path.stem}"
    except (ValueError, IndexError):
        return {}
    value = lots.get(sample_id, {}) if isinstance(lots, dict) else {}
    return dict(value) if isinstance(value, dict) else {}


def event(state: AgentState, event_type: str | None = None, **fields: Any) -> list[dict]:
    """현재 이벤트 목록에 한 건을 붙인 새 목록을 반환한다."""
    if event_type is None:
        event_type = str(fields.pop("type"))
    return [*state.get("events", []), {"type": event_type, **fields}]


def _cost(state: AgentState, usage: dict | None) -> float:
    usd = float((usage or {}).get("usd", 0.0))
    return round(float(state.get("cost_usd", 0.0)) + usd, 6)
