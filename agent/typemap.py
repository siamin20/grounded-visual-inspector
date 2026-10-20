"""품목·결함 유형 매핑의 단일 로더.

`eval/type_map.json`이 품목과 결함 유형 목록의 SSOT이다. 노드와
vision 백엔드가 이 모듈을 공유해 순환 import 없이 같은 목록을 쓴다.
"""

from functools import lru_cache
import json
from pathlib import Path
from typing import Any


TYPE_MAP_PATH = Path(__file__).resolve().parent.parent / "eval" / "type_map.json"


@lru_cache(maxsize=1)
def load_type_map() -> dict[str, dict[str, list[str]]]:
    """type map을 읽고 최소 스키마를 검증한다.

    폴백 목록을 두면 SSOT가 둘이 되므로, 파일이 없거나 손상됐을 때는
    즉시 실패하여 설정 문제가 드러나게 한다.
    """
    try:
        raw: Any = json.loads(TYPE_MAP_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"품목 유형 매핑을 읽을 수 없습니다: {TYPE_MAP_PATH}") from exc

    if not isinstance(raw, dict) or not raw:
        raise RuntimeError("type_map.json은 비어 있지 않은 JSON 객체여야 합니다.")

    normalized: dict[str, dict[str, list[str]]] = {}
    for item, defect_map in raw.items():
        if not isinstance(item, str) or not isinstance(defect_map, dict) or "good" not in defect_map:
            raise RuntimeError("type_map.json의 각 품목은 good을 포함한 객체여야 합니다.")
        normalized[item] = {}
        for defect_type, aliases in defect_map.items():
            if not isinstance(defect_type, str) or not isinstance(aliases, list):
                raise RuntimeError("type_map.json의 결함 유형 값은 별칭 배열이어야 합니다.")
            normalized[item][defect_type] = [str(alias) for alias in aliases]
    return normalized
