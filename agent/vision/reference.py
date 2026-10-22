"""정상 참조 이미지 경로를."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def normal_reference_path(item: str) -> str | None:
    """품목의 고정한 참조룰 돌려주고 없으면 ``None``을 돌려준다."""
    path = ROOT / "data" / "mvtec" / item / "train" / "good" / "000.png"
    return str(path) if path.exists() else None
