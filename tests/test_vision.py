"""Phase 2 비전 백엔드와 MCP Tool의 공개 계약 테스트."""

from pathlib import Path

import pytest
from PIL import Image

from agent.mcp_client import MCPClient
from agent.vision.backends import FakeVLM, make_vlm
from agent.vision.crop import crop_region, flip_h, unflip_regions


BACKEND_KEYS = {
    "text",
    "anomaly_seen",
    "regions",
    "features",
    "model",
    "tokens",
    "usd",
}
INSPECT_KEYS = BACKEND_KEYS | {"region", "flip"}
VALID_REGIONS = {"tl", "tr", "bl", "br", "center"}


def _coordinate_image() -> Image.Image:
    """자른 영역의 시작·끝 좌표를 픽셀값으로 확인할 수 있는 8×8 이미지."""
    image = Image.new("RGB", (8, 8))
    for y in range(8):
        for x in range(8):
            image.putpixel((x, y), (x, y, x + y))
    return image


@pytest.mark.parametrize(
    ("region", "box"),
    [
        ("full", (0, 0, 8, 8)),
        ("tl", (0, 0, 4, 4)),
        ("tr", (4, 0, 8, 4)),
        ("bl", (0, 4, 4, 8)),
        ("br", (4, 4, 8, 8)),
        ("center", (2, 2, 6, 6)),
    ],
)
def test_crop_region_uses_expected_pixel_box(region: str, box: tuple[int, ...]) -> None:
    source = _coordinate_image()

    cropped = crop_region(source, region)

    expected = source.crop(box)
    assert cropped.size == expected.size
    assert cropped.tobytes() == expected.tobytes()


def test_flip_h_reverses_pixels_without_mutating_source() -> None:
    source = Image.new("RGB", (3, 1))
    source.putdata([(10, 0, 0), (20, 0, 0), (30, 0, 0)])
    original = source.tobytes()

    flipped = flip_h(source)

    assert [flipped.getpixel((x, 0)) for x in range(3)] == [
        (30, 0, 0),
        (20, 0, 0),
        (10, 0, 0),
    ]
    assert source.tobytes() == original


def test_unflip_regions_swaps_left_and_right_without_mutating_input() -> None:
    regions = ["tl", "tr", "bl", "br", "center"]

    restored = unflip_regions(regions)

    assert restored == ["tr", "tl", "br", "bl", "center"]
    assert regions == ["tl", "tr", "bl", "br", "center"]


async def _observe(vlm: FakeVLM, image_path: str, *, flip: bool = False) -> dict:
    return await vlm.observe(
        image_path,
        item="screw",
        region="full",
        style="observe",
        flip=flip,
        reference_path=None,
    )


@pytest.mark.asyncio
async def test_fake_vlm_handles_good_defect_and_non_mvtec_paths() -> None:
    vlm = FakeVLM()

    good = await _observe(vlm, "/dataset/screw/test/good/000.png")
    defect = await _observe(vlm, "/dataset/screw/test/scratch_neck/000.png")
    unreadable = await _observe(vlm, "/dataset/screw/train/good/000.png")

    assert set(good) == BACKEND_KEYS
    assert good["anomaly_seen"] is False
    assert good["regions"] == []

    assert set(defect) == BACKEND_KEYS
    assert defect["anomaly_seen"] is True
    assert defect["text"] != "판독 불가"
    assert 1 <= len(defect["regions"]) <= 2
    assert set(defect["regions"]) <= VALID_REGIONS

    assert set(unreadable) == BACKEND_KEYS
    assert unreadable["anomaly_seen"] is None
    assert unreadable["text"] == "판독 불가"
    assert unreadable["regions"] == []


@pytest.mark.asyncio
async def test_fake_vlm_is_deterministic_and_flips_reported_regions() -> None:
    vlm = FakeVLM()
    path = "/dataset/screw/test/scratch_neck/007.png"

    first = await _observe(vlm, path)
    second = await _observe(vlm, path)
    flipped = await _observe(vlm, path, flip=True)

    assert first == second
    assert unflip_regions(flipped["regions"]) == first["regions"]


def test_make_vlm_selects_fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLM_BACKEND", "fake")

    assert isinstance(make_vlm("observer"), FakeVLM)
    assert isinstance(make_vlm("cross"), FakeVLM)


@pytest.mark.asyncio
async def test_fake_inspect_image_over_real_mcp_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """stdio MCP 연결과 JSON 직렬화까지 포함해 Tool 계약을 확인한다."""
    monkeypatch.setenv("VLM_BACKEND", "fake")
    image_path = tmp_path / "screw" / "test" / "good" / "000.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(image_path)

    client = MCPClient(flaky_first_tool_call=False)
    await client.start()
    try:
        server = await client.find_tool("inspect_image")
        result = await client.call_tool(
            "inspect_image",
            {
                "image_path": str(image_path),
                "item": "screw",
                "region": "full",
                "style": "observe",
                "flip": False,
                "model_role": "observer",
            },
            server=server,
        )
    finally:
        await client.stop()

    assert server == "vision"
    assert set(result) == INSPECT_KEYS
    assert result["anomaly_seen"] is False
    assert result["region"] == "full"
    assert result["flip"] is False
