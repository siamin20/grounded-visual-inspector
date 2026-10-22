"""Phase 2 비전 백엔드와 MCP Tool의 공개 계약 테스트."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from agent.vision import backends as vision_backends
from agent.mcp_client import MCPClient
from agent.vision.backends import (
    FakeVLM,
    OllamaVLM,
    OpenAIVLM,
    VLMBackend,
    _normalize_result,
    make_vlm,
)
from agent.vision.contracts import CHECKLIST_KEYS
from agent.vision.crop import crop_region, flip_h, unflip_regions


BACKEND_KEYS = {
    "text",
    "anomaly_seen",
    "regions",
    "features",
    "model",
    "tokens",
    "usd",
    "input_tokens",
    "output_tokens",
    "token_accounting",
    "estimated_usd",
    "compute_proxy_usd",
    "requested_model",
    "observed_model",
    "successful_provider_requests",
    "unknown_token_usage_successes",
    "observed_model_response_count",
    "unobserved_model_successes",
    "observed_model_ids",
}
INSPECT_KEYS = BACKEND_KEYS | {"region", "flip"}
VALID_REGIONS = {"tl", "tr", "bl", "br", "center"}


def _checks(*true_keys: str) -> dict[str, bool]:
    return {key: key in true_keys for key in CHECKLIST_KEYS}


class _ConcurrentSession:
    def __init__(self) -> None:
        self.entered = 0
        self.release = asyncio.Event()

    async def call_tool(self, name: str, arguments: dict):
        del name, arguments
        self.entered += 1
        if self.entered == 2:
            self.release.set()
        await asyncio.wait_for(self.release.wait(), timeout=0.2)
        return SimpleNamespace(content=[SimpleNamespace(text="{}")])


class _StubVLM(VLMBackend):
    name = "openai:requested"

    def __init__(self, responses: list[tuple[str, dict]]) -> None:
        super().__init__("requested")
        self.responses = list(responses)

    async def _request(self, prompt: str, images: list[str]) -> tuple[str, dict]:
        del prompt, images
        return self.responses.pop(0)


class _AsyncHTTPResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


def _async_client_for(payload: dict):
    class AsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **_kwargs):
            return _AsyncHTTPResponse(payload)

    return AsyncClient


def _coordinate_image() -> Image.Image:
    """자른 영역의 시작·끝 좌표를 픽셀값으로 확인할 수 있는 8×8 이미지."""
    image = Image.new("RGB", (8, 8))
    for y in range(8):
        for x in range(8):
            image.putpixel((x, y), (x, y, x + y))
    return image


@pytest.mark.asyncio
async def test_mcp_tool_calls_can_overlap_for_parallel_observation() -> None:
    client = MCPClient(flaky_first_tool_call=False)
    session = _ConcurrentSession()
    client.sessions["vision"] = session
    await asyncio.gather(
        client.call_tool("inspect_image", {"region": "tl"}),
        client.call_tool("inspect_image", {"region": "tr"}),
    )
    assert session.entered == 2


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

    good = await _observe(vlm, "/synthetic/screw/test/good/900.png")
    defect = await _observe(vlm, "/synthetic/screw/test/scratch_neck/900.png")
    unreadable = await _observe(vlm, "/synthetic/screw/train/good/900.png")

    assert set(good) == BACKEND_KEYS
    assert good["anomaly_seen"] is False
    assert good["regions"] == []
    assert good["token_accounting"] == "estimated_split"
    assert good["input_tokens"] + good["output_tokens"] == good["tokens"]
    assert good["successful_provider_requests"] == 0

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
async def test_vlm_korean_retry_accumulates_both_successful_requests(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    base_usage = {
        "model": "openai:resolved-1",
        "tokens": 6,
        "usd": 0.001,
        "input_tokens": 4,
        "output_tokens": 2,
        "token_accounting": "provider_split",
        "estimated_usd": 0.001,
        "compute_proxy_usd": 0.0,
        "requested_model": "openai:requested",
        "observed_model": "openai:resolved-1",
        "successful_provider_requests": 1,
    }
    retry_usage = {
        **base_usage,
        "model": "openai:resolved-2",
        "tokens": 9,
        "usd": 0.002,
        "input_tokens": 6,
        "output_tokens": 3,
        "estimated_usd": 0.002,
        "observed_model": "openai:resolved-2",
    }
    vlm = _StubVLM([
        ('{"text":"scratch","anomaly_seen":true}', base_usage),
        (
            '{"text":"표면에 선형 흔적이 보입니다.",'
            '"anomaly_seen":true,"regions":["tl"],"features":["선형 흔적"]}',
            retry_usage,
        ),
    ])

    result = await vlm.observe(
        str(image_path),
        item="screw",
        region="full",
        style="observe",
        flip=False,
        reference_path=None,
    )

    assert result["tokens"] == 15
    assert result["input_tokens"] == 10
    assert result["output_tokens"] == 5
    assert result["estimated_usd"] == 0.003
    assert result["successful_provider_requests"] == 2
    assert result["requested_model"] == "openai:requested"
    assert result["observed_model"] == "openai:resolved-2"
    assert result["model"] == "openai:resolved-2"


@pytest.mark.asyncio
async def test_vlm_observe_enforces_checklist_contract_end_to_end(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    raw = {
        "text": "모든 점검 항목에서 이상이 보이지 않습니다.",
        "anomaly_seen": False,
        "regions": [],
        "features": [],
        "checks": _checks(),
    }
    usage = {
        "model": "openai:resolved",
        "tokens": 7,
        "usd": 0.001,
        "input_tokens": 5,
        "output_tokens": 2,
        "token_accounting": "provider_split",
        "estimated_usd": 0.001,
        "compute_proxy_usd": 0.0,
        "requested_model": "openai:requested",
        "observed_model": "openai:resolved",
        "successful_provider_requests": 1,
    }
    vlm = _StubVLM([(json.dumps(raw, ensure_ascii=False), usage)])

    result = await vlm.observe(
        str(image_path),
        item="screw",
        region="full",
        style="checklist",
        flip=False,
        reference_path=None,
    )

    assert result["anomaly_seen"] is False
    assert result["checks"] == _checks()


@pytest.mark.asyncio
async def test_vlm_rejects_unknown_item_before_opening_image() -> None:
    vlm = _StubVLM([])

    with pytest.raises(ValueError, match="지원하지 않는 검사 품목"):
        await vlm.observe(
            "/does/not/exist.png",
            item="unknown-item",
            region="full",
            style="observe",
            flip=False,
            reference_path=None,
        )


@pytest.mark.asyncio
async def test_ollama_vlm_does_not_invent_missing_response_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "message": {"content": '{"text":"이상이 보이지 않습니다."}'},
        "prompt_eval_count": 8,
        "eval_count": 2,
    }
    monkeypatch.setattr(
        vision_backends.httpx,
        "AsyncClient",
        lambda **_kwargs: _async_client_for(payload)(),
    )

    _, usage = await OllamaVLM("requested")._request("prompt", [])

    assert usage["model"] is None
    assert usage["observed_model"] is None
    assert usage["requested_model"] == "ollama:requested"
    assert usage["input_tokens"] == 8
    assert usage["output_tokens"] == 2
    assert usage["successful_provider_requests"] == 1


@pytest.mark.asyncio
async def test_openai_vlm_uses_response_model_and_provider_token_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "model": "resolved-2026-10-21",
        "choices": [{
            "message": {"content": '{"text":"이상이 보이지 않습니다."}'},
        }],
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": 2,
            "total_tokens": 10,
        },
    }
    monkeypatch.setattr(
        vision_backends.httpx,
        "AsyncClient",
        lambda **_kwargs: _async_client_for(payload)(),
    )
    vlm = OpenAIVLM("requested")
    vlm.api_key = "test-key"

    _, usage = await vlm._request("prompt", [])

    assert usage["model"] == "openai:resolved-2026-10-21"
    assert usage["observed_model"] == "openai:resolved-2026-10-21"
    assert usage["requested_model"] == "openai:requested"
    assert usage["input_tokens"] == 8
    assert usage["output_tokens"] == 2
    assert usage["tokens"] == 10
    assert usage["successful_provider_requests"] == 1


@pytest.mark.asyncio
async def test_fake_baseline_uses_abstain_for_unreadable_path() -> None:
    result = await FakeVLM().observe(
        "/synthetic/screw/train/good/900.png",
        item="screw",
        region="full",
        style="baseline",
        flip=False,
        reference_path=None,
    )

    assert result["verdict"] == "ABSTAIN"
    assert result["defect_type"] is None
    assert result["anomaly_seen"] is None
    assert result["observations"] == ["판독 불가"]
    assert result["abstain_reason"] == "insufficient_visual_evidence"


@pytest.mark.parametrize(
    ("raw", "expected_reason"),
    [
        (
            {
                "verdict": "PASS",
                "observations": ["표면과 욘곽에 이상이 보이지 않습니다."],
            },
            "validation_failure",
        ),
        *[
            (
                {
                    "verdict": "PASS",
                    "defect_type": alias,
                    "observations": ["표면과 욘곽에 이상이 보이지 않습니다."],
                },
                "validation_failure",
            )
            for alias in ("good", "none", "null", "\uc815\uc0c1")
        ],
        (
            {
                "verdict": "MAYBE",
                "defect_type": "scratch_neck",
                "observations": ["목 부위에 선형 흔적이 보입니다."],
            },
            "validation_failure",
        ),
        (
            {
                "verdict": "PASS",
                "defect_type": "scratch_neck",
                "observations": ["목 부위에 선형 흔적이 보입니다."],
            },
            "validation_failure",
        ),
        ({}, "validation_failure"),
        (
            {
                "verdict": "REJECT",
                "defect_type": "not_in_type_map",
                "observations": ["표면에 이상 흔적이 보입니다."],
            },
            "validation_failure",
        ),
        (
            {"verdict": "PASS", "defect_type": None, "observations": []},
            "insufficient_visual_evidence",
        ),
        (
            {
                "verdict": "PASS",
                "defect_type": None,
                "observations": [],
                "text": "표면과 윤곽에 이상이 보이지 않습니다.",
            },
            "insufficient_visual_evidence",
        ),
        (
            {
                "verdict": "PASS",
                "defect_type": ["scratch_neck"],
                "observations": ["표면과 윤곽에 이상이 보이지 않습니다."],
            },
            "validation_failure",
        ),
    ],
)
def test_baseline_normalizer_fails_closed(
    raw: dict, expected_reason: str
) -> None:
    result = _normalize_result(
        raw,
        style="baseline",
        model="test",
        tokens=0,
        usd=0.0,
        item="screw",
    )

    assert result["verdict"] == "ABSTAIN"
    assert result["defect_type"] is None
    assert result["anomaly_seen"] is None
    assert result["abstain_reason"] == expected_reason


@pytest.mark.parametrize("style", ["observe", "checklist"])
@pytest.mark.parametrize(
    "text",
    [
        "이미지를 판독할 수 없습니다.",
        "형태를 식별할 수 없습니다.",
        "결함 여부를 확인할 수 없습니다.",
        "이미지가 흐려 판별할 수 없습니다.",
        "화질이 낮아 판단하기 어렵습니다.",
        "unreadable image",
    ],
)
def test_observation_normalizer_canonicalizes_unreadable_text(
    style: str, text: str
) -> None:
    result = _normalize_result(
        {
            "text": text,
            "anomaly_seen": False,
            "regions": ["tl"],
            "features": ["정상처럼 보임"],
        },
        style=style,
        model="test",
        tokens=3,
        usd=0.0,
        item="screw",
    )

    assert result["text"] == "판독 불가"
    assert result["anomaly_seen"] is None
    assert result["regions"] == []
    assert result["features"] == []


@pytest.mark.parametrize(
    ("regions", "features"),
    [([], ["짧은 선형 흔적"]), (["tl"], []), ([], [])],
)
@pytest.mark.parametrize("style", ["observe", "checklist"])
def test_positive_observation_requires_location_and_feature_evidence(
    style: str, regions: list[str], features: list[str]
) -> None:
    result = _normalize_result(
        {
            "text": "머리 왼쪽에 짧은 선형 흔적이 보입니다.",
            "anomaly_seen": True,
            "regions": regions,
            "features": features,
        },
        style=style,
        model="test",
        tokens=3,
        usd=0.0,
        item="screw",
    )

    assert result["text"] == "판독 불가"
    assert result["anomaly_seen"] is None
    assert result["regions"] == []
    assert result["features"] == []


@pytest.mark.parametrize(
    ("raw", "expected_anomaly"),
    [
        (
            {
                "text": "여섯 점검 항목 모두 이상이 보이지 않습니다.",
                "anomaly_seen": False,
                "regions": [],
                "features": [],
                "checks": _checks(),
            },
            False,
        ),
        (
            {
                "text": "좌상단에 짧은 선형 흔적이 보입니다.",
                "anomaly_seen": True,
                "regions": ["tl"],
                "features": ["짧은 선형 흔적"],
                "checks": _checks("scratch"),
            },
            True,
        ),
    ],
)
def test_checklist_normalizer_accepts_only_complete_consistent_contract(
    raw: dict, expected_anomaly: bool
) -> None:
    result = _normalize_result(
        raw,
        style="checklist",
        model="test",
        tokens=0,
        usd=0.0,
        item="screw",
    )

    assert result["anomaly_seen"] is expected_anomaly
    assert result["checks"] == raw["checks"]


@pytest.mark.parametrize(
    ("raw", "expected_text"),
    [
        (
            {
                "anomaly_seen": False,
                "regions": [],
                "features": [],
                "checks": _checks(),
            },
            "체크리스트의 여섯 항목에서 이상 신호가 보고되지 않았습니다.",
        ),
        (
            {
                "anomaly_seen": True,
                "regions": ["tl"],
                "features": ["짧은 선형 흔적"],
                "checks": _checks("scratch"),
            },
            "체크리스트의 여섯 항목에서 이상 신호가 보고되었습니다: 짧은 선형 흔적",
        ),
    ],
)
def test_checklist_missing_text_uses_structured_korean_summary(
    raw: dict, expected_text: str
) -> None:
    result = _normalize_result(
        raw,
        style="checklist",
        model="test",
        tokens=0,
        usd=0.0,
        item="screw",
    )

    assert result["text"] == expected_text
    assert result["anomaly_seen"] is raw["anomaly_seen"]
    assert result["checks"] == raw["checks"]
    assert result["summary_synthesized"] is True


@pytest.mark.parametrize(
    "raw",
    [
        {
            "text": "제품 외관이 보입니다.",
            "anomaly_seen": True,
            "regions": ["tl"],
            "features": ["외관"],
        },
        {
            "text": "긁힘이 보입니다.",
            "anomaly_seen": True,
            "regions": ["tl"],
            "features": ["선형 흔적"],
            "checks": {key: False for key in CHECKLIST_KEYS[:-1]},
        },
        {
            "text": "긁힘이 보입니다.",
            "anomaly_seen": True,
            "regions": ["tl"],
            "features": ["선형 흔적"],
            "checks": {**_checks("scratch"), "other": False},
        },
        {
            "text": "긁힘이 보입니다.",
            "anomaly_seen": True,
            "regions": ["tl"],
            "features": ["선형 흔적"],
            "checks": {**_checks(), "scratch": 1},
        },
        {
            "text": "긁힘이 보입니다.",
            "anomaly_seen": False,
            "regions": ["tl"],
            "features": ["선형 흔적"],
            "checks": _checks("scratch"),
        },
        {
            "text": "긁힘이 보입니다.",
            "anomaly_seen": True,
            "regions": ["tl", "outside"],
            "features": ["선형 흔적"],
            "checks": _checks("scratch"),
        },
        {
            "text": "긁힘이 보입니다.",
            "anomaly_seen": True,
            "regions": ["tl"],
            "features": ["선형 흔적", 1],
            "checks": _checks("scratch"),
        },
        {
            "text": "모든 항목이 정상입니다.",
            "anomaly_seen": False,
            "regions": [],
            "features": ["균일한 표면"],
            "checks": _checks(),
        },
    ],
)
def test_checklist_contract_failures_are_canonical_unreadable(raw: dict) -> None:
    result = _normalize_result(
        raw,
        style="checklist",
        model="test",
        tokens=0,
        usd=0.0,
        item="screw",
    )

    assert result["text"] == "판독 불가"
    assert result["anomaly_seen"] is None
    assert result["regions"] == []
    assert result["features"] == []
    assert "checks" not in result


def test_cropped_observer_requires_source_region_for_positive_evidence() -> None:
    raw = {
        "text": "크롭 안에 짧은 선형 흔적이 보입니다.",
        "anomaly_seen": True,
        "regions": ["tl"],
        "features": ["짧은 선형 흔적"],
    }

    mismatched = _normalize_result(
        raw,
        style="observe",
        model="test",
        tokens=0,
        usd=0.0,
        item="screw",
        region="br",
    )
    matched = _normalize_result(
        {**raw, "regions": ["br"]},
        style="observe",
        model="test",
        tokens=0,
        usd=0.0,
        item="screw",
        region="br",
    )

    assert mismatched["anomaly_seen"] is None
    assert mismatched["regions"] == []
    assert matched["anomaly_seen"] is True
    assert matched["regions"] == ["br"]


def test_cropped_observer_clears_negative_region_labels() -> None:
    result = _normalize_result(
        {
            "text": "크롭 안에 구체적 이상이 보이지 않습니다.",
            "anomaly_seen": False,
            "regions": ["br"],
            "features": ["균일한 표면"],
        },
        style="observe",
        model="test",
        tokens=0,
        usd=0.0,
        item="screw",
        region="br",
    )

    assert result["anomaly_seen"] is False
    assert result["regions"] == []


@pytest.mark.parametrize(
    ("style", "raw"),
    [
        (
            "baseline",
            {
                "verdict": "PASS",
                "defect_type": None,
                "observations": ["눈으로 확인한 한국어 관찰"],
            },
        ),
        (
            "baseline",
            {
                "verdict": "PASS",
                "defect_type": None,
                "observations": [],
                "text": "한국어 관찰",
            },
        ),
        (
            "observe",
            {
                "text": "한국어 관찰",
                "anomaly_seen": False,
                "regions": ["tl"],
                "features": ["균일한 표면"],
            },
        ),
        (
            "checklist",
            {
                "text": "표면과 윤곽에서 이상이 보이지 않습니다.",
                "anomaly_seen": False,
                "regions": [],
                "features": ["실제로 확인한 색·형태·상대 크기만 적으라"],
            },
        ),
        (
            "observe",
            {
                "text": "다른 사물명으로 바꾸어 추측하지 말라.",
                "anomaly_seen": True,
                "regions": ["br"],
                "features": ["검은 점"],
            },
        ),
        (
            "observe",
            {
                "text": "주 대상을 검사 대상으로 지칭하고 사물 종류를 다시 추측하지 말라.",
                "anomaly_seen": False,
                "regions": [],
                "features": [],
            },
        ),
        (
            "checklist",
            {
                "text": "checks의 여섯 항목을 모두 빠짐없이 boolean으로 적으라.",
                "anomaly_seen": False,
                "regions": [],
                "features": [],
                "checks": _checks(),
            },
        ),
        (
            "checklist",
            {
                "text": "점검 결과를 기록했습니다.",
                "anomaly_seen": True,
                "regions": ["tl"],
                "features": ["하나라도 true이면 anomaly_seen=true"],
                "checks": _checks("scratch"),
            },
        ),
    ],
)
def test_prompt_meta_text_is_never_visual_evidence(
    style: str, raw: dict
) -> None:
    result = _normalize_result(
        raw,
        style=style,
        model="test",
        tokens=0,
        usd=0.0,
        item="screw",
    )

    assert result["text"] == "판독 불가"
    assert result["anomaly_seen"] is None
    assert result["regions"] == []
    assert result["features"] == []
    if style == "baseline":
        assert result["verdict"] == "ABSTAIN"
        assert result["observations"] == []
        assert result["abstain_reason"] == "validation_failure"


@pytest.mark.parametrize(
    "prompt_copy",
    [
        "점검 항목의 이름을 나열하는 것은 시각 근거가 아니다.",
        (
            "정상 참조와 다른 구체적 부위·형태·질감·"
            "경계·패턴이 직접 보일 때만 해당 항목을 true로 적으라."
        ),
        "모든 항목을 일괄 true로 적지 마라.",
    ],
)
def test_new_checklist_prompt_copy_is_never_visual_evidence(
    prompt_copy: str,
) -> None:
    raw = {
        "text": "점검 결과를 기록했습니다.",
        "anomaly_seen": True,
        "regions": ["tl"],
        "features": [prompt_copy],
        "checks": _checks("scratch"),
    }

    assert vision_backends._is_prompt_meta_text(prompt_copy) is True
    result = _normalize_result(
        raw,
        style="checklist",
        model="test",
        tokens=0,
        usd=0.0,
        item="screw",
    )

    assert result["text"] == "판독 불가"
    assert result["anomaly_seen"] is None
    assert result["regions"] == []
    assert result["features"] == []


@pytest.mark.parametrize("style", ["observe", "checklist", "baseline"])
def test_specific_korean_observation_survives_meta_guard(style: str) -> None:
    text = "목 부위에 짧은 선형 흔적이 하나 보입니다."
    raw = (
        {"verdict": "REJECT", "defect_type": "scratch_neck", "observations": [text]}
        if style == "baseline"
        else {
            "text": text,
            "anomaly_seen": True,
            "regions": ["tl"],
            "features": ["짧은 선형 흔적"],
        }
    )
    if style == "checklist":
        raw["checks"] = _checks("scratch")

    result = _normalize_result(
        raw,
        style=style,
        model="test",
        tokens=0,
        usd=0.0,
        item="screw",
    )

    assert result["text"] == text
    assert result["anomaly_seen"] is True
    if style == "baseline":
        assert result["verdict"] == "REJECT"
        assert result["observations"] == [text]


@pytest.mark.parametrize("style", ["observe", "checklist", "baseline"])
def test_vlm_prompt_has_typed_contract_without_value_placeholders(style: str) -> None:
    prompt = vision_backends._prompt(
        style,
        item="screw",
        region="full",
        has_reference=False,
        extra_context=None,
    )

    assert "한국어 관찰" not in prompt
    assert '"색·형태·상대 크기 특징"' not in prompt
    assert "JSON 객체" in prompt
    if style in {"observe", "checklist"}:
        assert "검사 품목은 screw이며 이미 확정" in prompt
        assert "다른 사물명" in prompt
    if style == "observe":
        assert "구체적 차이가 없으면 anomaly_seen=false" in prompt
        assert "직접 판독할 수 없을 때만 anomaly_seen=null" in prompt
        assert all(word not in prompt for word in ("호박", "콩", "브러시"))
    if style == "checklist":
        assert "checks의 여섯 항목" in prompt
        assert all(key in prompt for key in CHECKLIST_KEYS)


@pytest.mark.parametrize(
    "text",
    [
        "결함 여부를 확인할 수 없습니다.",
        "이미지가 흐려 판별할 수 없습니다.",
        "화질이 낮아 판단하기 어렵습니다.",
    ],
)
def test_baseline_uncertain_wording_abstains(text: str) -> None:
    result = _normalize_result(
        {"verdict": "PASS", "defect_type": None, "observations": [text]},
        style="baseline",
        model="test",
        tokens=0,
        usd=0.0,
        item="screw",
    )

    assert result["verdict"] == "ABSTAIN"
    assert result["abstain_reason"] == "insufficient_visual_evidence"


@pytest.mark.parametrize(
    "text",
    [
        "Unable to determine whether a defect is present.",
        "Cannot determine whether the surface is normal.",
        "The image is too blurry to assess.",
        "The surface appears normal.",
    ],
)
def test_non_korean_observation_is_never_negative_evidence(text: str) -> None:
    result = _normalize_result(
        {
            "text": text,
            "anomaly_seen": False,
            "regions": [],
            "features": [],
        },
        style="observe",
        model="test",
        tokens=0,
        usd=0.0,
        item="screw",
    )

    assert result["text"] == "\ud310\ub3c5 \ubd88\uac00"
    assert result["anomaly_seen"] is None


@pytest.mark.parametrize(
    "text",
    [
        "이미지가 흐려 결함 여부렬 알 수 없습니다.",
        "결함 여부가 불확실합니다.",
        "해상도가 낮아 결과가 모호합니다.",
        "이미지가 너무 흐립니다.",
        "결함 여부렬 구분할 수 없습니다.",
        "결함 여부렬 확정할 수 없습니다.",
        "이미지가 흐릿해 결함 여부 확인이 곤란합니다.",
        "결함 여부가 분명하지 않습니다.",
        "욱안으로 구별되지 않습니다.",
        "판정하기 애매합니다.",
        "판정을 보류합니다.",
        "판단을 유보합니다.",
        "판독 결과에 확신이 없습니다.",
    ],
)
def test_korean_uncertainty_is_never_negative_evidence(text: str) -> None:
    result = _normalize_result(
        {
            "text": text,
            "anomaly_seen": False,
            "regions": [],
            "features": [],
        },
        style="observe",
        model="test",
        tokens=0,
        usd=0.0,
        item="screw",
    )

    assert result["text"] == "\ud310\ub3c5 \ubd88\uac00"
    assert result["anomaly_seen"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("label", "expected_anomaly"),
    [("good", False), ("scratch_neck", True)],
)
async def test_fake_vlm_recovers_dataset_label_for_uploaded_copy(
    tmp_path: Path,
    label: str,
    expected_anomaly: bool,
) -> None:
    """dataset 등록 파일과 바이트가 같은 업로드 복사본은 같은 라벨 결과를 낸다."""
    source = tmp_path / "data" / "mvtec" / "screw" / "test" / label / "000.png"
    source.parent.mkdir(parents=True)
    image_bytes = f"fake-png-bytes:{label}".encode()
    source.write_bytes(image_bytes)

    upload = tmp_path / "data" / "uploads" / f"{label}.png"
    upload.parent.mkdir(parents=True)
    upload.write_bytes(image_bytes)

    dataset_path = tmp_path / "eval" / "dataset.json"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text(
        json.dumps({
            "samples": [{
                "path": str(source.relative_to(tmp_path)),
                "item": "screw",
                "label": label,
            }],
        }),
        encoding="utf-8",
    )
    vlm = FakeVLM(dataset_path=dataset_path)

    source_result = await _observe(vlm, str(source))
    upload_result = await _observe(vlm, str(upload))

    assert upload_result["anomaly_seen"] is expected_anomaly
    assert upload_result["text"] == source_result["text"]
    assert upload_result["features"] == source_result["features"]


@pytest.mark.asyncio
async def test_fake_vlm_keeps_unregistered_upload_unreadable(tmp_path: Path) -> None:
    dataset_path = tmp_path / "eval" / "dataset.json"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text('{"samples": []}', encoding="utf-8")
    upload = tmp_path / "data" / "uploads" / "unknown.png"
    upload.parent.mkdir(parents=True)
    upload.write_bytes(b"unregistered-image-bytes")

    result = await _observe(FakeVLM(dataset_path=dataset_path), str(upload))

    assert set(result) == BACKEND_KEYS
    assert result["anomaly_seen"] is None
    assert result["text"] == "판독 불가"
    assert result["regions"] == []


@pytest.mark.asyncio
async def test_fake_vlm_is_deterministic_and_flips_reported_regions() -> None:
    vlm = FakeVLM()
    path = "/synthetic/screw/test/scratch_neck/907.png"

    first = await _observe(vlm, path)
    second = await _observe(vlm, path)
    flipped = await _observe(vlm, path, flip=True)

    assert first == second
    assert unflip_regions(flipped["regions"]) == first["regions"]


@pytest.mark.parametrize(
    "text",
    [
        "\ud310\ub2e8\ud558\uae30\uc5d0\ub294 \uc815\ubcf4\uac00 \ubd80\uc871",
        "\uc774 \uc774\ubbf8\uc9c0\ub9cc\uc73c\ub85c\ub294 \uacb0\ub860\uc744 \ub0b4\ub9b4 \uc218 \uc5c6\uc74c",
        "\ud604\uc7ac \uc790\ub8cc\ub85c \ud310\uc815\uc774 \uc81c\ud55c",
        "\ud45c\uba74 \uc0c1\ud0dc\ub97c \ub2e8\uc815\ud560 \uc218 \uc5c6\uc74c",
        "\ubbf8\ud655\uc815",
        "\ubd88\uba85\ud655",
    ],
)
@pytest.mark.parametrize("style", ["observe", "checklist", "baseline"])
def test_uncertainty_variants_fail_closed_in_every_vision_style(
    text: str,
    style: str,
) -> None:
    raw = (
        {"verdict": "PASS", "defect_type": None, "observations": [text]}
        if style == "baseline"
        else {"text": text, "anomaly_seen": False}
    )

    result = _normalize_result(
        raw,
        style=style,
        model="test",
        tokens=0,
        usd=0.0,
        item="screw",
    )

    assert result["anomaly_seen"] is None
    if style == "baseline":
        assert result["verdict"] == "ABSTAIN"
        assert result["abstain_reason"] == "insufficient_visual_evidence"
    else:
        assert result["text"] == "\ud310\ub3c5 \ubd88\uac00"


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
