"""VLM 백엔드 구현.

===============================================================================
주의: FakeVLM은 라벨(폴더명)을 들여다보므로 평가에 절대 쓰지 않는다.
eval 은 fake 를 거부한다. FakeVLM은 테스트와 데모에만 사용한다.
===============================================================================
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Literal

import httpx
from langsmith import traceable
from PIL import Image

from agent.typemap import load_type_map

from .crop import REGIONS, crop_region, flip_h, unflip_regions


Role = Literal["observer", "cross"]

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"

_FAKE_OBSERVATIONS = {
    "scratch_head": ("머리 표면에 선형 긁힘이 보입니다.", ["선형 긁힘", "머리 표면"]),
    "scratch_neck": ("목 부위에 선형 긁힘이 보입니다.", ["선형 긁힘", "목 부위"]),
    "thread_side": ("측면 나사산의 일부가 찌그러지고 연속성이 끊겨 보입니다.", ["나사산 변형", "연속성 단절"]),
    "thread_top": ("상단 나사산의 일부가 마모되어 윤곽이 불규칙해 보입니다.", ["나사산 마모", "불규칙한 윤곽"]),
    "broken_large": ("입구 테두리의 넓은 부분에 윤곽 단절과 열린 틈이 보입니다.", ["넓은 윤곽 단절", "열린 틈"]),
    "broken_small": ("입구 테두리의 국소 부위에 작은 조각 결손이 보입니다.", ["국소 결손", "테두리"]),
    "contamination": ("표면에 주변과 색과 질감이 다른 이물 흔적이 보입니다.", ["이물", "색과 질감 차이"]),
    "bent": ("외곽과 중앙 구멍의 간격이 한쪽으로 치우쳐 보입니다.", ["비대칭", "구멍 편향"]),
    "color": ("표면 일부에 주변과 구별되는 색상 변화가 보입니다.", ["색상 변화", "표면"]),
    "scratch": ("표면에 주변 질감을 가로지르는 선형 흔적이 보입니다.", ["선형 흔적", "질감 단절"]),
    "crack": ("껍질 표면에 가늘고 긴 균열이 보입니다.", ["균열", "가늘고 긴 선"]),
    "cut": ("껍질 경계에 평탄하고 긴 절단면이 보입니다.", ["절단면", "평탄한 경계"]),
    "hole": ("껍질 표면에 둘러싼 경계가 분명한 구멍이 보입니다.", ["구멍", "분명한 경계"]),
    "print": ("껍질 표면에 주변 무늬와 다른 인쇄형 흔적이 보입니다.", ["인쇄형 흔적", "표면 무늬 차이"]),
}


def _encode_png(path: str, region: str, flip: bool) -> str:
    """이미지를 자르고 필요하면 반전해 base64 PNG로 변환한다."""
    with Image.open(path) as source:
        source.load()
        prepared = crop_region(source, region)
        if flip:
            prepared = flip_h(prepared)
        output = io.BytesIO()
        prepared.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def _strip_json_fence(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped


def _parse_json(content: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    try:
        parsed = json.loads(_strip_json_fence(content))
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "예"}:
            return True
        if lowered in {"false", "no", "아니오"}:
            return False
    return None


def _as_string_list(value: Any, *, allowed: set[str] | None = None) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for entry in value:
        if not isinstance(entry, str):
            continue
        clean = entry.strip()
        if clean and (allowed is None or clean in allowed):
            result.append(clean)
    return result


def _has_korean(text: str) -> bool:
    return bool(re.search(r"[ㄱ-ㆎ가-힣]", text))


def _language_text(raw: dict[str, Any], style: str) -> str:
    if style == "baseline":
        observations = raw.get("observations", [])
        if isinstance(observations, list):
            return " ".join(str(item) for item in observations)
    return str(raw.get("text", ""))


def _normalize_result(raw: dict[str, Any], *, style: str, model: str,
                      tokens: int, usd: float) -> dict[str, Any]:
    if style == "baseline":
        verdict = str(raw.get("verdict", "")).upper()
        defect_type = raw.get("defect_type")
        if (
            not isinstance(defect_type, str)
            or not defect_type.strip()
            or defect_type.strip().lower() in {"null", "none", "good"}
            or defect_type.strip() == "정상"
        ):
            defect_type = None
        else:
            defect_type = defect_type.strip()
        observations = _as_string_list(raw.get("observations"))
        if not observations and isinstance(raw.get("text"), str):
            observations = [raw["text"].strip()] if raw["text"].strip() else []
        if verdict not in {"PASS", "REJECT"}:
            verdict = "REJECT" if defect_type else "PASS"
        text = "\n".join(observations) or "판독 불가"
        features = _as_string_list(raw.get("features"))
        if not features and defect_type:
            features = [defect_type]
        return {
            "text": text,
            "anomaly_seen": verdict == "REJECT",
            "regions": _as_string_list(raw.get("regions"), allowed=REGIONS - {"full"}),
            "features": features,
            "model": model,
            "tokens": tokens,
            "usd": round(float(usd), 6),
            "verdict": verdict,
            "defect_type": defect_type,
            "observations": observations,
        }

    text = raw.get("text")
    return {
        "text": text.strip() if isinstance(text, str) and text.strip() else "판독 불가",
        "anomaly_seen": _as_bool(raw.get("anomaly_seen")),
        "regions": _as_string_list(raw.get("regions"), allowed=REGIONS - {"full"}),
        "features": _as_string_list(raw.get("features")),
        "model": model,
        "tokens": tokens,
        "usd": round(float(usd), 6),
    }


def _prompt(style: str, *, item: str, region: str, has_reference: bool,
            extra_context: str | None) -> str:
    image_note = (
        "첫 번째 이미지는 정상 참조이고 두 번째 이미지가 검사 대상이다."
        if has_reference else "이미지는 검사 대상이다."
    )
    context = f"\n\n## 추가 기준\n{extra_context}" if extra_context else ""
    if style == "baseline":
        defect_types = ", ".join(
            defect_type for defect_type in load_type_map().get(item, {})
            if defect_type != "good"
        ) or "(정의되지 않음)"
        return (
            "당신은 제품 외관 검사기다. 한국어로만 응답하라. "
            f"{image_note} 품목은 {item}이고 가능한 결함 유형은 "
            f"[{defect_types}]이다. 이미지를 한 번에 판정하라. "
            "정상이면 verdict=PASS, defect_type=null로 하고, 결함이면 "
            "verdict=REJECT와 목록 중 하나의 defect_type을 쓰라. "
            "JSON 객체로만 응답하라: "
            '{"verdict":"PASS|REJECT","defect_type":"결함 유형|null",'
            '"observations":["눈으로 확인한 한국어 관찰"]}' + context
        )

    location_note = (
        f"현재 입력은 전체 이미지의 {region} 영역이므로 regions에는 "
        f"관찰 위치를 {region}으로 적으라. "
        if region != "full" else
        "위치는 tl, tr, bl, br, center 중 해당하는 값으로 적으라. "
    )
    schema = (
        'JSON 객체로만 응답하라: {"text":"한국어 관찰",'
        '"anomaly_seen":true|false|null,"regions":["tl|tr|bl|br|center"],'
        '"features":["색·형태·상대 크기 특징"]}. '
    )
    if style == "observe":
        instruction = (
            "합격·불합격이나 결함 유형을 판정하지 말고 눈에 보이는 것만 기술하라. "
            "색, 형태, 그리고 부품 대비 상대 크기를 적으라. "
        )
    elif style == "checklist":
        instruction = (
            "text에 다음 항목을 각각 예 또는 아니오로 확인하라: 긁힘, "
            "깨짐·균열, 이물·오염, 변색, 휨·변형, 나사산 손상. "
            "합격·불합격은 판정하지 말라. "
        )
    else:
        raise ValueError(f"지원하지 않는 VLM style입니다: {style}")
    return (
        f"당신은 {item} 제품의 시각 관찰기다. 한국어로만 응답하라. "
        f"{image_note} {instruction}{location_note}{schema}{context}"
    )


class VLMBackend(ABC):
    """Ollama·OpenAI VLM이 공유하는 이미지 처리와 응답 정규화."""

    name: str

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    async def _request(self, prompt: str, images: list[str]) -> tuple[str, int, float]:
        """백엔드에 프롬프트와 base64 PNG 목록을 보낸다."""

    @traceable(run_type="llm", name="vlm.observe")
    async def observe(self, image_path: str, *, item: str, region: str, style: str,
                      flip: bool, reference_path: str | None,
                      extra_context: str | None = None) -> dict[str, Any]:
        if region not in REGIONS:
            raise ValueError(f"지원하지 않는 이미지 영역입니다: {region}")
        if style not in {"observe", "checklist", "baseline"}:
            raise ValueError(f"지원하지 않는 VLM style입니다: {style}")

        images = []
        if reference_path:
            images.append(_encode_png(reference_path, region, flip))
        images.append(_encode_png(image_path, region, flip))
        prompt = _prompt(style, item=item, region=region,
                         has_reference=bool(reference_path), extra_context=extra_context)

        content, tokens, usd = await self._request(prompt, images)
        raw = _parse_json(content)
        if not _has_korean(_language_text(raw, style)):
            retry_prompt = (
                prompt
                + "\n\n이전 응답의 관찰문이 한국어가 아니었다. "
                  "필드명과 결함 유형 코드를 제외한 모든 설명을 반드시 한국어로 다시 작성하라."
            )
            content, retry_tokens, retry_usd = await self._request(retry_prompt, images)
            tokens += retry_tokens
            usd += retry_usd
            raw = _parse_json(content)

        return _normalize_result(raw, style=style, model=self.name,
                                 tokens=tokens, usd=usd)


class FakeVLM(VLMBackend):
    """폴더 라벨로 결과를 만드는 외부 호출 없는 결정론적 백엔드."""

    name = "fake-vlm"

    def __init__(self):
        super().__init__("fake")

    async def _request(self, prompt: str, images: list[str]) -> tuple[str, int, float]:
        raise NotImplementedError("FakeVLM은 이미지를 외부 모델에 보내지 않습니다.")

    @staticmethod
    def _regions(image_path: str, region: str) -> list[str]:
        if region != "full":
            return [region]
        digest = hashlib.sha256(image_path.encode("utf-8")).digest()
        candidates = ["tl", "tr", "bl", "br", "center"]
        first = digest[0] % len(candidates)
        count = 1 + digest[1] % 2
        if count == 1:
            return [candidates[first]]
        second = (first + 1 + digest[2] % (len(candidates) - 1)) % len(candidates)
        return [candidates[first], candidates[second]]

    @traceable(run_type="llm", name="fake_vlm.observe")
    async def observe(self, image_path: str, *, item: str, region: str, style: str,
                      flip: bool, reference_path: str | None,
                      extra_context: str | None = None) -> dict[str, Any]:
        del reference_path, extra_context
        if region not in REGIONS:
            raise ValueError(f"지원하지 않는 이미지 영역입니다: {region}")
        if style not in {"observe", "checklist", "baseline"}:
            raise ValueError(f"지원하지 않는 VLM style입니다: {style}")

        normalized_path = image_path.replace("\\", "/")
        match = re.search(r"(?:^|/)test/([^/]+)(?:/|$)", normalized_path)
        if not match:
            return {
                "text": "판독 불가",
                "anomaly_seen": None,
                "regions": [],
                "features": [],
                "model": self.name,
                "tokens": 0,
                "usd": 0.0,
            }

        defect_type = match.group(1)
        if defect_type not in load_type_map().get(item, {}):
            return {
                "text": "판독 불가",
                "anomaly_seen": None,
                "regions": [],
                "features": [],
                "model": self.name,
                "tokens": 0,
                "usd": 0.0,
            }
        if defect_type == "good":
            text = (
                "체크리스트의 모든 항목이 아니오이며 눈에 띄는 이상이 보이지 않습니다."
                if style == "checklist" else
                "표면과 윤곽에서 눈에 띄는 이상이 보이지 않습니다."
            )
            anomaly_seen = False
            regions: list[str] = []
            features: list[str] = ["균일한 윤곽", "일관된 표면"]
        else:
            observation, features = _FAKE_OBSERVATIONS.get(
                defect_type,
                ("표면에 주변과 구별되는 형태 차이가 보입니다.", ["형태 차이"]),
            )
            text = (
                f"체크리스트에서 해당 이상 항목은 예입니다. {observation}"
                if style == "checklist" else observation
            )
            anomaly_seen = True
            regions = self._regions(normalized_path, region)
            if flip:
                regions = unflip_regions(regions)

        tokens = (len(normalized_path) + len(text)) // 4 + 1
        result: dict[str, Any] = {
            "text": text,
            "anomaly_seen": anomaly_seen,
            "regions": regions,
            "features": features,
            "model": self.name,
            "tokens": tokens,
            "usd": 0.0,
        }
        if style == "baseline":
            result.update({
                "verdict": "REJECT" if anomaly_seen else "PASS",
                "defect_type": defect_type if anomaly_seen else None,
                "observations": [text],
            })
        return result


class OllamaVLM(VLMBackend):
    """로컬 Ollama의 vision chat API를 사용한다."""

    def __init__(self, model: str = "qwen2.5vl:3b"):
        super().__init__(model)
        self.name = f"ollama:{model}"
        self.url = os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL).rstrip("/") + "/api/chat"

    async def _request(self, prompt: str, images: list[str]) -> tuple[str, int, float]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt, "images": images}],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(self.url, json=payload)
        except httpx.TimeoutException as exc:
            raise asyncio.TimeoutError("Ollama VLM 응답 시간이 초과됐습니다.") from exc
        response.raise_for_status()
        data = response.json()
        content = data.get("message", {}).get("content", "")
        tokens = (data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0)
        return str(content), int(tokens), 0.0


class OpenAIVLM(VLMBackend):
    """OpenAI Chat Completions의 멀티모달 입력을 사용한다."""

    def __init__(self, model: str):
        super().__init__(model)
        self.name = f"openai:{model}"
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.usd_per_million = 0.15 if model.startswith("gpt-4o-mini") else 2.5

    async def _request(self, prompt: str, images: list[str]) -> tuple[str, int, float]:
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY가 설정되지 않았습니다 — .env 또는 환경변수로 지정하세요."
            )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image}"},
        } for image in images)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(OPENAI_URL, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise asyncio.TimeoutError("OpenAI VLM 응답 시간이 초과됐습니다.") from exc
        response.raise_for_status()
        data = response.json()
        answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        tokens = int(data.get("usage", {}).get("total_tokens") or 0)
        usd = tokens / 1_000_000 * self.usd_per_million
        return str(answer), tokens, usd


def make_vlm(role: Role) -> VLMBackend:
    """환경변수와 역할에 맞는 VLM 백엔드를 생성한다."""
    if role not in {"observer", "cross"}:
        raise ValueError(f"지원하지 않는 VLM 역할입니다: {role}")

    backend = os.environ.get("VLM_BACKEND") or os.environ.get("LLM_BACKEND", "fake")
    backend = backend.strip().lower()
    if backend == "fake":
        return FakeVLM()

    env_name = "VLM_OBSERVER_MODEL" if role == "observer" else "VLM_CROSS_MODEL"
    if backend == "ollama":
        model = os.environ.get(env_name, "qwen2.5vl:3b")
        return OllamaVLM(model)
    if backend == "openai":
        default_model = "gpt-4o-mini" if role == "observer" else "gpt-4o"
        model = os.environ.get(env_name, default_model)
        return OpenAIVLM(model)
    raise ValueError(f"지원하지 않는 VLM_BACKEND입니다: {backend}")
