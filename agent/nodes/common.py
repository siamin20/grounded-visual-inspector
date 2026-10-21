"""노드가 공유하는 데이터 로더와 상태 헬퍼."""

import json
from pathlib import Path
from typing import Any

from agent.readability import is_unreadable_text
from agent.state import AgentState
from agent.typemap import load_type_map


ROOT = Path(__file__).resolve().parents[2]
LOTS_PATH = ROOT / "data" / "lots.json"
KNOWN_ITEMS = frozenset(load_type_map())
_QUADRANTS = frozenset({"tl", "tr", "bl", "br"})
_REGIONS = frozenset({*_QUADRANTS, "center"})


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


def _tokens(state: AgentState, usage: dict | None) -> int:
    """모델·임베딩 사용량의 실측 토큰 수를 상태에 누적한다."""
    return int(state.get("tokens", 0)) + int((usage or {}).get("tokens", 0))


def is_readable_observation(observation: Any) -> bool:
    """Observer 결과가 정상/이상을 구분할 수 있는 구조인지 판정한다.

    빈 관찰문은 `anomaly_seen=false`만으로 정상 판정을 만들 수 있으므로 판독
    가능으로 인정하지 않는다. `bool`은 정수 0/1과 구분해 명시적 출력만 받는다.
    """
    if not isinstance(observation, dict):
        return False
    text = observation.get("text")
    return (
        type(observation.get("anomaly_seen")) is bool
        and isinstance(text, str)
        and not is_unreadable_text(text)
        and "판독 불가" not in text
    )


def observer_status(observations: Any) -> str:
    """Observer 관찰들을 설계 계약의 `O+|O-|O?`로 집계한다."""
    if not isinstance(observations, list):
        return "O?"
    readable = [
        observation
        for observation in observations
        if is_readable_observation(observation)
    ]
    if any(observation["anomaly_seen"] is True for observation in readable):
        return "O+"
    if any(
        observation.get("region") == "full"
        and observation["anomaly_seen"] is False
        for observation in readable
    ):
        return "O-"
    negative_quadrants = {
        str(observation.get("region"))
        for observation in readable
        if observation["anomaly_seen"] is False
        and observation.get("region") in _QUADRANTS
    }
    return "O-" if len(negative_quadrants) >= 3 else "O?"


def positive_observer_regions(observations: Any) -> set[str]:
    """판독 가능하며 이상을 본 Observer 결과의 영역만 반환한다."""
    if not isinstance(observations, list):
        return set()
    return {
        str(region)
        for observation in observations
        if is_readable_observation(observation)
        and observation["anomaly_seen"] is True
        and isinstance(observation.get("regions"), list)
        for region in observation["regions"]
        if region in _REGIONS
    }


def cross_status(cross_check: Any, *, skipped: bool = False) -> str:
    """Cross-Checker 결과를 `C+|C-|C?|C-skip`으로 정규화한다."""
    if skipped:
        return "C-skip"
    if not isinstance(cross_check, dict):
        return "C?"
    summary = cross_check.get("summary")
    regions = cross_check.get("regions")
    if (
        type(cross_check.get("anomaly_seen")) is not bool
        or not isinstance(summary, str)
        or is_unreadable_text(summary)
        or "판독 불가" in summary
        or not isinstance(regions, list)
        or any(region not in _REGIONS for region in regions)
    ):
        return "C?"
    return "C+" if cross_check["anomaly_seen"] is True else "C-"


def normalized_judgement_status(judgement: Any) -> str:
    """검증된 Spec-Judge 결과를 `V+|V0|V?`로 정규화한다."""
    if not isinstance(judgement, dict):
        return "V?"
    status = judgement.get("status")
    if status is not None:
        violations = judgement.get("violations")
        defect_type = judgement.get("defect_type")
        grounded_shape = (
            isinstance(violations, list)
            and bool(violations)
            and all(isinstance(value, str) and value for value in violations)
            and isinstance(defect_type, str)
            and bool(defect_type)
            and defect_type != "good"
        )
        none_shape = violations == [] and defect_type is None
        if status in {"grounded", "V+"} and grounded_shape:
            return "V+"
        if status in {"none", "V0"} and none_shape:
            return "V0"
        return "V?"

    # 이전 상태나 단위 테스트처럼 status가 없는 입력도 보수적으로 해석한다.
    violations = judgement.get("violations")
    defect_type = judgement.get("defect_type")
    if isinstance(violations, list) and violations and isinstance(defect_type, str):
        return "V+"
    if violations == [] and defect_type is None:
        return "V0"
    return "V?"
