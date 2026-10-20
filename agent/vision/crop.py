"""VLM 입력 이미지의 영역 자르기와 좌우 좌표 변환 유틸리티."""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image


REGIONS = {"full", "tl", "tr", "bl", "br", "center"}


def crop_region(img: Image.Image, region: str) -> Image.Image:
    """이미지를 지정 영역으로 자른 새 이미지를 반환한다.

    사분면은 홀수 크기에서 오른쪽·아래쪽이 남는 픽셀을 포함한다. ``center``는
    각 축의 1/4부터 3/4까지인 중앙 50% 영역이다.
    """
    if region not in REGIONS:
        raise ValueError(f"지원하지 않는 이미지 영역입니다: {region}")

    width, height = img.size
    if width < 1 or height < 1:
        raise ValueError("크기가 0인 이미지는 자를 수 없습니다.")
    if region == "full":
        return img.copy()

    mid_x, mid_y = width // 2, height // 2
    boxes = {
        "tl": (0, 0, mid_x, mid_y),
        "tr": (mid_x, 0, width, mid_y),
        "bl": (0, mid_y, mid_x, height),
        "br": (mid_x, mid_y, width, height),
        "center": (width // 4, height // 4, width * 3 // 4, height * 3 // 4),
    }
    box = boxes[region]
    if box[0] == box[2] or box[1] == box[3]:
        raise ValueError(f"이미지가 {region} 영역을 자르기에 너무 작습니다: {img.size}")
    return img.crop(box)


def flip_h(img: Image.Image) -> Image.Image:
    """원본을 변경하지 않고 좌우 반전한 이미지를 반환한다."""
    return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def unflip_regions(regions: Sequence[str]) -> list[str]:
    """좌우 반전 이미지의 영역 좌표를 원본 좌표로 되돌린다."""
    swapped = {"tl": "tr", "tr": "tl", "bl": "br", "br": "bl"}
    return [swapped.get(region, region) for region in regions]
