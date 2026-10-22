"""VLM 백엔드 구현.

===============================================================================
주의: FakeVLM은 경로와 파일 해시로 라벨을 들여다본다 — 평가 금지.
eval/dataset.json 등록 파일은 SHA-256으로 라벨을 역조회해 업로드 복사본에도 쓴다.
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
import unicodedata
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import httpx
from langsmith import traceable
from PIL import Image

from agent.readability import is_unreadable_text
from agent.typemap import load_type_map
from agent.usage import MISSING, nonnegative_finite_float, nonnegative_int

from .contracts import CHECKLIST_KEYS, normalized_checklist
from .crop import REGIONS, crop_region, flip_h, unflip_regions


Role = Literal["observer", "cross"]

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[2] / "eval" / "dataset.json"

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


def openai_vlm_usd_per_million(model: str) -> float:
    """실제 OpenAI VLM 후보 모델에 적용할 평가용 블렌디드 단가다."""
    return 0.15 if model.startswith("gpt-4o-mini") else 2.5


def _file_sha256(path: Path) -> str:
    """파일을 이미지로 해석하지 않고 바이트 해시만 계산한다."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def _load_fake_label_hashes(dataset_path: Path) -> dict[str, str]:
    """dataset.json에 등록되고 현재 존재하는 파일의 해시-라벨 맵을 만든다."""
    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list):
        return {}

    root = dataset_path.resolve().parent.parent
    labels_by_hash: dict[str, str] = {}
    ambiguous_hashes: set[str] = set()
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        raw_path = sample.get("path")
        label = sample.get("label")
        if not isinstance(raw_path, str) or not isinstance(label, str):
            continue
        image_path = Path(raw_path)
        if not image_path.is_absolute():
            image_path = root / image_path
        try:
            file_hash = _file_sha256(image_path)
        except OSError:
            continue
        previous = labels_by_hash.get(file_hash)
        if previous is not None and previous != label:
            ambiguous_hashes.add(file_hash)
            continue
        labels_by_hash[file_hash] = label

    for file_hash in ambiguous_hashes:
        labels_by_hash.pop(file_hash, None)
    return labels_by_hash


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


_PROMPT_META_MARKERS = (
    "한국어관찰",
    "색형태상대크기특징",
    "이미지에서직접보이는구체적사실",
    "이미지에서직접확인한대상위치색형태상대크기",
    "실제로확인한색형태상대크기만",
    "다른사물명으로바꾸",
    "그차이를말할수없으면",
    "예로확인한항목",
    "주대상을검사대상으로지칭",
    "구체적차이가없으면",
    "직접판독할수없을때만",
    "checks의여섯항목",
    "하나라도true이면",
    "모두false이면",
    "true항목이있으면",
    "추가키는금지",
    "한국어완결문",
    "필드설명",
    "예시문구",
    "지시문",
)

_PROMPT_META_MARKERS += (
    "점검항목의이름을나열하는것은시각근거가아니다",
    "정상참조와다른구체적부위형태질감경계패턴",
    "모든항목을일괄true로적지마라",
)


def _is_prompt_meta_text(value: Any) -> bool:
    """프롬프트의 예시·설명을 실제 시각 근거와 구분한다."""
    if not isinstance(value, str):
        return False
    compact = re.sub(
        r"[\W_]+",
        "",
        unicodedata.normalize("NFC", value).lower(),
    )
    return any(marker in compact for marker in _PROMPT_META_MARKERS)


def _has_prompt_meta_evidence(raw: dict[str, Any], style: str) -> bool:
    """관찰 필드 중 하나라도 메타 문구면 응답 전체를 보류한다."""
    values: list[Any] = []
    if style == "baseline":
        observations = raw.get("observations")
        if isinstance(observations, list):
            values.extend(observations)
        if isinstance(raw.get("text"), str):
            values.append(raw["text"])
    else:
        values.append(raw.get("text"))
    features = raw.get("features")
    if isinstance(features, list):
        values.extend(features)
    return any(_is_prompt_meta_text(value) for value in values)


def _validated_checklist_fields(
    raw: dict[str, Any],
) -> tuple[dict[str, bool], bool, list[str], list[str]] | None:
    """Cross의 핵심 구조 필드만 교약을 검증한다.

    ``text``는 서술용 요약이며 설계에서 결정로 만들 수 있다.
    """
    checks = normalized_checklist(raw.get("checks"))
    raw_regions = raw.get("regions")
    raw_features = raw.get("features")
    strict_regions = (
        isinstance(raw_regions, list)
        and all(
            isinstance(value, str)
            and value.strip() == value
            and value in REGIONS - {"full"}
            for value in raw_regions
        )
    )
    strict_features = (
        isinstance(raw_features, list)
        and all(
            isinstance(value, str)
            and bool(value.strip())
            and value.strip() == value
            for value in raw_features
        )
    )
    expected_anomaly = any(checks.values()) if checks is not None else None
    valid = (
        checks is not None
        and type(raw.get("anomaly_seen")) is bool
        and raw["anomaly_seen"] is expected_anomaly
        and strict_regions
        and strict_features
        and (
            (expected_anomaly and bool(raw_regions) and bool(raw_features))
            or (
                expected_anomaly is False
                and raw_regions == []
                and raw_features == []
            )
        )
    )
    if not valid:
        return None
    return checks, bool(expected_anomaly), list(raw_regions), list(raw_features)


def _language_text(raw: dict[str, Any], style: str) -> str:
    if style == "baseline":
        observations = raw.get("observations", [])
        if isinstance(observations, list):
            return " ".join(str(item) for item in observations)
    return str(raw.get("text", ""))


def _usage_count(data: dict[str, Any], key: str, *, provider: str) -> int | None:
    """Provider usage의 누락과 잘못된 값을 구분한다."""
    if key not in data:
        return None
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{provider} {key}이 0 이상 정수가 아닙니다.")
    return value


def _observed_model(provider: str, value: Any) -> str | None:
    """응답에 실제 모델 ID가 있을 때만 backend 접두어를 붙인다."""
    if not isinstance(value, str) or not value.strip():
        return None
    return f"{provider}:{value.strip()}"


def _merge_request_usage(*usages: dict[str, Any]) -> dict[str, Any]:
    """한 관찰 안의 한국어 재요청까지 실제 사용량으로 합산한다."""
    if not usages:
        raise ValueError("합산할 VLM usage가 없습니다.")
    usages = tuple(
        _result_usage(
            model=usage.get("observed_model"),
            tokens=0,
            usd=0.0,
            usage=usage,
        )
        for usage in usages
    )
    split = all(
        usage.get("token_accounting") == "provider_split"
        and usage.get("input_tokens") is not None
        and usage.get("output_tokens") is not None
        for usage in usages
    )
    observed_model = next(
        (
            usage.get("observed_model")
            for usage in reversed(usages)
            if usage.get("observed_model") is not None
        ),
        None,
    )
    observed_model_ids = list(dict.fromkeys(
        model
        for usage in usages
        for model in (
            usage.get("observed_model_ids")
            if isinstance(usage.get("observed_model_ids"), list)
            else [usage.get("observed_model")]
        )
        if isinstance(model, str) and model.strip()
    ))
    estimated_usd = round(
        sum(float(usage.get("estimated_usd", 0.0)) for usage in usages), 6
    )
    compute_proxy_usd = round(
        sum(float(usage.get("compute_proxy_usd", 0.0)) for usage in usages), 6
    )
    return {
        "model": observed_model,
        "tokens": sum(int(usage.get("tokens", 0)) for usage in usages),
        "usd": round(sum(float(usage.get("usd", 0.0)) for usage in usages), 6),
        "input_tokens": (
            sum(int(usage["input_tokens"]) for usage in usages) if split else None
        ),
        "output_tokens": (
            sum(int(usage["output_tokens"]) for usage in usages) if split else None
        ),
        "token_accounting": "provider_split" if split else "total_only",
        "estimated_usd": estimated_usd,
        "compute_proxy_usd": compute_proxy_usd,
        "requested_model": usages[0].get("requested_model"),
        "observed_model": observed_model,
        "observed_model_ids": observed_model_ids,
        "successful_provider_requests": sum(
            int(usage.get("successful_provider_requests", 0)) for usage in usages
        ),
        "unknown_token_usage_successes": sum(
            int(usage.get("unknown_token_usage_successes", 0)) for usage in usages
        ),
        "observed_model_response_count": sum(
            int(usage.get("observed_model_response_count", 0)) for usage in usages
        ),
        "unobserved_model_successes": sum(
            int(usage.get("unobserved_model_successes", 0)) for usage in usages
        ),
    }


def _result_usage(
    *,
    model: str | None,
    tokens: int,
    usd: float,
    usage: dict[str, Any] | None,
) -> dict[str, Any]:
    """정규화 결과에 공통 usage 계약을 붙인다."""
    values = usage or {}
    observed_model = values.get("observed_model", model)
    if observed_model is not None and (
        not isinstance(observed_model, str) or not observed_model.strip()
    ):
        raise ValueError("VLM observed_model은 null 또는 빈 값이 아닌 문자열이어야 합니다.")
    if isinstance(observed_model, str):
        observed_model = observed_model.strip()
    tokens_value = nonnegative_int(
        values.get("tokens", MISSING), name="vlm.tokens", default=tokens
    )
    usd_value = nonnegative_finite_float(
        values.get("usd", MISSING), name="vlm.usd", default=usd
    )
    estimated_usd = nonnegative_finite_float(
        values.get("estimated_usd", MISSING),
        name="vlm.estimated_usd",
        default=usd_value,
    )
    compute_proxy_usd = nonnegative_finite_float(
        values.get("compute_proxy_usd", MISSING),
        name="vlm.compute_proxy_usd",
        default=0.0,
    )
    successful_requests = nonnegative_int(
        values.get("successful_provider_requests", MISSING),
        name="vlm.successful_provider_requests",
        default=0,
    )
    accounting = values.get("token_accounting", "total_only")
    if accounting not in {
        "provider_split", "estimated_split", "mixed_split", "total_only"
    }:
        raise ValueError("VLM token_accounting 값이 올바르지 않습니다.")
    raw_input = values.get("input_tokens")
    raw_output = values.get("output_tokens")
    if accounting == "total_only":
        if raw_input is not None or raw_output is not None:
            raise ValueError("VLM total_only usage에 분리 토큰을 둘 수 없습니다.")
        input_tokens = output_tokens = None
    else:
        input_tokens = nonnegative_int(raw_input, name="vlm.input_tokens")
        output_tokens = nonnegative_int(raw_output, name="vlm.output_tokens")
        if input_tokens + output_tokens != tokens_value:
            raise ValueError("VLM 입력·출력 토큰 합이 전체와 다릅니다.")
    observed_ids = values.get("observed_model_ids")
    if observed_ids is None:
        observed_ids = [observed_model] if observed_model else []
    if not isinstance(observed_ids, list) or any(
        not isinstance(value, str) or not value.strip() for value in observed_ids
    ):
        raise ValueError("VLM observed_model_ids가 올바르지 않습니다.")
    unknown_tokens = nonnegative_int(
        values.get("unknown_token_usage_successes", MISSING),
        name="vlm.unknown_token_usage_successes",
        default=(successful_requests if accounting == "total_only" and tokens_value == 0 else 0),
    )
    observed_count = nonnegative_int(
        values.get("observed_model_response_count", MISSING),
        name="vlm.observed_model_response_count",
        default=(1 if successful_requests and observed_model else 0),
    )
    unobserved_count = nonnegative_int(
        values.get("unobserved_model_successes", MISSING),
        name="vlm.unobserved_model_successes",
        default=(0 if observed_model else successful_requests),
    )
    if unknown_tokens > successful_requests:
        raise ValueError("VLM token usage 미지 수가 provider 성공 수보다 큽니다.")
    if observed_count + unobserved_count != successful_requests:
        raise ValueError("VLM model ID 관측 카운터가 provider 성공 수와 다릅니다.")
    if observed_count and not observed_ids:
        raise ValueError("VLM model ID 관측 응답이 있지만 observed_model_ids가 비었습니다.")
    if observed_model and observed_model not in observed_ids:
        raise ValueError("VLM observed_model이 observed_model_ids에 없습니다.")
    if (
        successful_requests
        and accounting == "total_only"
        and tokens_value == 0
        and not unknown_tokens
    ):
        raise ValueError("VLM token usage 누락을 미지 성공으로 기록해야 합니다.")
    return {
        # 예전 model 필드도 요청값이 아닌 실제 응답 ID를 의미한다.
        "model": observed_model,
        "tokens": tokens_value,
        "usd": round(usd_value, 6),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "token_accounting": accounting,
        "estimated_usd": round(estimated_usd, 6),
        "compute_proxy_usd": round(compute_proxy_usd, 6),
        "requested_model": values.get("requested_model", model),
        "observed_model": observed_model,
        "observed_model_ids": list(dict.fromkeys(observed_ids)),
        "successful_provider_requests": successful_requests,
        "unknown_token_usage_successes": unknown_tokens,
        "observed_model_response_count": observed_count,
        "unobserved_model_successes": unobserved_count,
    }




def _normalize_result(raw: dict[str, Any], *, style: str, model: str | None,
                      tokens: int, usd: float,
                      item: str | None = None,
                      usage: dict[str, Any] | None = None,
                      region: str | None = None) -> dict[str, Any]:
    result_usage = _result_usage(
        model=model,
        tokens=tokens,
        usd=usd,
        usage=usage,
    )
    if style == "baseline":
        verdict = str(raw.get("verdict", "")).upper()
        raw_defect_type = raw.get("defect_type")
        invalid_defect_shape = (
            "defect_type" not in raw
            or (raw_defect_type is not None and not isinstance(raw_defect_type, str))
            or (
                isinstance(raw_defect_type, str)
                and (
                    not raw_defect_type.strip()
                    or raw_defect_type.strip().lower() in {"null", "none", "good"}
                    or raw_defect_type.strip() == "\uc815\uc0c1"
                )
            )
        )
        defect_type = raw_defect_type
        if (
            invalid_defect_shape
            or not isinstance(defect_type, str)
            or not defect_type.strip()
            or defect_type.strip().lower() in {"null", "none", "good"}
            or defect_type.strip() == "정상"
        ):
            defect_type = None
        else:
            defect_type = defect_type.strip()
        raw_observations = raw.get("observations")
        invalid_observations_shape = not isinstance(raw_observations, list) or any(
            not isinstance(value, str) for value in raw_observations
        )
        observations = _as_string_list(raw_observations)
        if not observations and isinstance(raw.get("text"), str):
            observations = [raw["text"].strip()] if raw["text"].strip() else []
        has_prompt_meta = _has_prompt_meta_evidence(raw, style)
        text = "판독 불가" if has_prompt_meta else (
            "\n".join(observations) or "판독 불가"
        )
        allowed_types = {
            name for name in load_type_map().get(item or "", {}) if name != "good"
        }
        abstain_reason: str | None = None
        if not raw or invalid_defect_shape or invalid_observations_shape:
            verdict = "ABSTAIN"
            abstain_reason = "validation_failure"
        elif has_prompt_meta:
            verdict = "ABSTAIN"
            abstain_reason = "validation_failure"
        elif not _as_string_list(raw_observations):
            verdict = "ABSTAIN"
            abstain_reason = "insufficient_visual_evidence"
        elif is_unreadable_text(text):
            verdict = "ABSTAIN"
            abstain_reason = "insufficient_visual_evidence"
        elif not _has_korean(text):
            verdict = "ABSTAIN"
            abstain_reason = "validation_failure"
        elif verdict == "ABSTAIN" and defect_type is None:
            abstain_reason = "insufficient_visual_evidence"
        elif verdict not in {"PASS", "REJECT"}:
            verdict = "ABSTAIN"
            abstain_reason = "validation_failure"
        elif verdict == "PASS" and defect_type is not None:
            verdict = "ABSTAIN"
            abstain_reason = "validation_failure"
        elif verdict == "REJECT" and defect_type not in allowed_types:
            verdict = "ABSTAIN"
            abstain_reason = "validation_failure"
        if verdict == "ABSTAIN":
            defect_type = None
        if has_prompt_meta:
            observations = []
        features = [] if has_prompt_meta else _as_string_list(raw.get("features"))
        if not features and defect_type:
            features = [defect_type]
        return {
            "text": text,
            "anomaly_seen": (
                None if verdict == "ABSTAIN" else verdict == "REJECT"
            ),
            "regions": (
                []
                if has_prompt_meta
                else _as_string_list(raw.get("regions"), allowed=REGIONS - {"full"})
            ),
            "features": features,
            **result_usage,
            "verdict": verdict,
            "defect_type": defect_type,
            "observations": observations,
            "abstain_reason": abstain_reason,
        }

    text = raw.get("text")
    has_text = isinstance(text, str) and bool(text.strip())
    normalized_text = text.strip() if has_text else "판독 불가"
    if _has_prompt_meta_evidence(raw, style):
        return {
            "text": "판독 불가",
            "anomaly_seen": None,
            "regions": [],
            "features": [],
            **result_usage,
        }
    checks: dict[str, bool] | None = None
    summary_synthesized = False
    if style == "checklist":
        checklist_fields = _validated_checklist_fields(raw)
        if checklist_fields is None:
            return {
                "text": "판독 불가",
                "anomaly_seen": None,
                "regions": [],
                "features": [],
                **result_usage,
            }
        checks, anomaly_seen, regions, features = checklist_fields
        if not has_text:
            summary_synthesized = True
            normalized_text = (
                "체크리스트의 여섯 항목에서 이상 신호가 보고되었습니다: "
                + ", ".join(features)
                if anomaly_seen else
                "체크리스트의 여섯 항목에서 이상 신호가 보고되지 않았습니다."
            )
        elif is_unreadable_text(normalized_text) or not _has_korean(normalized_text):
            return {
                "text": "판독 불가",
                "anomaly_seen": None,
                "regions": [],
                "features": [],
                **result_usage,
            }
    else:
        if is_unreadable_text(normalized_text) or not _has_korean(normalized_text):
            return {
                "text": "판독 불가",
                "anomaly_seen": None,
                "regions": [],
                "features": [],
                **result_usage,
            }
        anomaly_seen = _as_bool(raw.get("anomaly_seen"))
        regions = _as_string_list(raw.get("regions"), allowed=REGIONS - {"full"})
        features = _as_string_list(raw.get("features"))
    if style == "observe" and region not in {None, "full"}:
        if anomaly_seen is True and regions != [region]:
            return {
                "text": "판독 불가",
                "anomaly_seen": None,
                "regions": [],
                "features": [],
                **result_usage,
            }
        if anomaly_seen is False:
            regions = []
    # 소형 VLM이 구체적 근거 없이 true만 출력하면 검색·판정에
    # 가짜 양성이 전파된다. 양성은 위치와 시각 특징을 둘 다 제공한
    # 경우만 판독 가능한 관찰로 채택한다.
    if anomaly_seen is True and (not regions or not features):
        return {
            "text": "판독 불가",
            "anomaly_seen": None,
            "regions": [],
            "features": [],
            **result_usage,
        }
    result = {
        "text": normalized_text,
        "anomaly_seen": anomaly_seen,
        "regions": regions,
        "features": features,
        **result_usage,
    }
    if checks is not None:
        result["checks"] = checks
        result["summary_synthesized"] = summary_synthesized
    return result


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
            "이미지를 판독할 수 없으면 verdict=ABSTAIN, defect_type=null로 하라. "
            "JSON 객체로만 응답하라. 응답 계약은 "
            "verdict: enum(PASS, REJECT, ABSTAIN), defect_type: string|null, "
            "observations: array<string>(최소 1개)이다. "
            "observations에는 이미지에서 직접 확인한 대상·위치·색·형태·"
            "상대 크기의 구체적 사실만 한국어 완결문으로 적으라. "
            "예시, 필드 설명, 지시문을 값으로 복사하지 마라." + context
        )

    location_note = (
        f"현재 입력은 전체 이미지의 {region} 영역이므로 양성일 때만 regions에 "
        f"관찰 위치를 {region}으로 하나만 적으라. 음성이면 빈 배열로 적으라. "
        if region != "full" else
        "위치는 tl, tr, bl, br, center 중 해당하는 값으로 적으라. "
    )
    schema = (
        "JSON 객체로만 응답하라. 응답 계약은 text: string, "
        "anomaly_seen: boolean|null, regions: array<enum(tl, tr, bl, br, center)>, "
        "features: array<string>이다. text에는 이미지에서 직접 보이는 "
        "구체적 사실을 한국어 완결문으로 적고, features에는 실제로 "
        "확인한 색·형태·상대 크기만 적으라. 예시, 필드 설명, "
        "지시문을 값으로 복사하지 마라. "
    )
    if style == "observe":
        instruction = (
            f"검사 품목은 {item}이며 이미 확정됐으므로 다른 사물명으로 바꾸어 추측하지 말라. "
            "주 대상을 '검사 대상'으로 지칭하고 사물 종류를 다시 추측하지 말라. "
            "합격·불합격이나 결함 유형을 판정하지 말고 눈에 보이는 것만 기술하라. "
            "색, 형태, 그리고 부품 대비 상대 크기를 적으라. "
            "정상 참조와 비교해 구체적 차이가 없으면 anomaly_seen=false로 적으라. "
            "anomaly_seen=true라면 text와 features에 부위·형태·주변과의 "
            "차이를 최소 하나 구체적으로 적으라. 대상이 없거나 흐림·가림 등으로 "
            "직접 판독할 수 없을 때만 anomaly_seen=null로 적으라. "
        )
    elif style == "checklist":
        checklist_keys = ", ".join(CHECKLIST_KEYS)
        instruction = (
            f"검사 품목은 {item}이며 이미 확정됐으므로 다른 사물명으로 바꾸지 말라. "
            "긁힘, 깨짐·균열, 이물·오염, 변색, 휨·변형, 나사산 손상을 각각 확인하라. "
            "점검 항목의 이름을 나열하는 것은 시각 근거가 아니다. 정상 참조와 다른 "
            "구체적 부위·형태·질감·경계·패턴이 직접 보일 때만 해당 항목을 true로 "
            "적고, 근거가 없으면 false로 적으라. 모든 항목을 일괄 true로 적지 마라. "
            "checks의 여섯 항목을 모두 빠짐없이 boolean으로 적고, 하나라도 true이면 "
            "anomaly_seen=true, 모두 false이면 anomaly_seen=false로 적으라. "
            "true 항목이 있으면 그 부위와 시각 특징을 regions와 features에 적고, "
            "모두 false이면 regions와 features를 빈 배열로 적으라. "
            "판독 불가이면 anomaly_seen=null로 적고 checks를 넣지 말라. "
            "합격·불합격은 판정하지 말라. "
        )
        schema = (
            "JSON 객체로만 응답하라. 응답 계약은 text: string, "
            "anomaly_seen: boolean|null, regions: array<enum(tl, tr, bl, br, center)>, "
            "features: array<string>, checks: object이다. "
            f"checks의 키는 정확히 {checklist_keys}이고 추가 키는 금지한다. "
            "text에는 각 확인 결과와 이미지에서 직접 보이는 구체적 사실만 한국어 "
            "완결문으로 적으라. 예시, 필드 설명, 지시문을 값으로 복사하지 마라. "
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
    async def _request(
        self, prompt: str, images: list[str]
    ) -> tuple[str, dict[str, Any]]:
        """백엔드에 프롬프트와 base64 PNG 목록을 보낸다."""

    @traceable(run_type="llm", name="vlm.observe")
    async def observe(self, image_path: str, *, item: str, region: str, style: str,
                      flip: bool, reference_path: str | None,
                      extra_context: str | None = None) -> dict[str, Any]:
        if item not in load_type_map():
            raise ValueError(f"지원하지 않는 검사 품목입니다: {item}")
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

        content, usage = await self._request(prompt, images)
        raw = _parse_json(content)
        checklist_without_text = (
            style == "checklist"
            and _validated_checklist_fields(raw) is not None
            and not (
                isinstance(raw.get("text"), str)
                and bool(raw["text"].strip())
            )
        )
        if not checklist_without_text and not _has_korean(_language_text(raw, style)):
            retry_prompt = (
                prompt
                + "\n\n이전 응답의 관찰문이 한국어가 아니었다. "
                  "필드명과 결함 유형 코드를 제외한 모든 설명을 반드시 한국어로 다시 작성하라."
            )
            content, retry_usage = await self._request(retry_prompt, images)
            usage = _merge_request_usage(usage, retry_usage)
            raw = _parse_json(content)

        return _normalize_result(
            raw,
            style=style,
            model=usage.get("observed_model"),
            tokens=usage.get("tokens", 0),
            usd=usage.get("usd", 0.0),
            item=item,
            usage=usage,
            region=region,
        )


class FakeVLM(VLMBackend):
    """폴더 라벨과 등록 파일 해시로 결과를 만드는 결정론적 백엔드."""

    name = "fake-vlm"

    def __init__(self, dataset_path: str | Path | None = None):
        super().__init__("fake")
        resolved_path = DEFAULT_DATASET_PATH if dataset_path is None else Path(dataset_path)
        self._labels_by_hash = _load_fake_label_hashes(resolved_path.resolve())

    def _label_from_hash(self, image_path: str) -> str | None:
        try:
            file_hash = _file_sha256(Path(image_path))
        except OSError:
            return None
        return self._labels_by_hash.get(file_hash)

    async def _request(
        self, prompt: str, images: list[str]
    ) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError("FakeVLM은 이미지를 외부 모델에 보내지 않습니다.")

    def _usage(self, normalized_path: str, text: str, tokens: int) -> dict[str, Any]:
        input_tokens = min(len(normalized_path) // 4, tokens)
        return {
            "model": self.name,
            "tokens": tokens,
            "usd": 0.0,
            "input_tokens": input_tokens,
            "output_tokens": tokens - input_tokens,
            "token_accounting": "estimated_split",
            "estimated_usd": 0.0,
            "compute_proxy_usd": 0.0,
            "requested_model": self.name,
            "observed_model": self.name,
            "successful_provider_requests": 0,
            "observed_model_ids": [self.name],
            "unknown_token_usage_successes": 0,
            "observed_model_response_count": 0,
            "unobserved_model_successes": 0,
        }

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
        if item not in load_type_map():
            raise ValueError(f"지원하지 않는 검사 품목입니다: {item}")
        if region not in REGIONS:
            raise ValueError(f"지원하지 않는 이미지 영역입니다: {region}")
        if style not in {"observe", "checklist", "baseline"}:
            raise ValueError(f"지원하지 않는 VLM style입니다: {style}")

        normalized_path = image_path.replace("\\", "/")
        match = re.search(r"(?:^|/)test/([^/]+)(?:/|$)", normalized_path)
        defect_type = match.group(1) if match else self._label_from_hash(image_path)
        if defect_type is None:
            usage = self._usage(normalized_path, "판독 불가", 0)
            result: dict[str, Any] = {
                "text": "판독 불가",
                "anomaly_seen": None,
                "regions": [],
                "features": [],
                **usage,
            }
            if style == "baseline":
                result.update({
                    "verdict": "ABSTAIN",
                    "defect_type": None,
                    "observations": ["판독 불가"],
                    "abstain_reason": "insufficient_visual_evidence",
                })
            return result

        if defect_type not in load_type_map().get(item, {}):
            usage = self._usage(normalized_path, "판독 불가", 0)
            result = {
                "text": "판독 불가",
                "anomaly_seen": None,
                "regions": [],
                "features": [],
                **usage,
            }
            if style == "baseline":
                result.update({
                    "verdict": "ABSTAIN",
                    "defect_type": None,
                    "observations": ["판독 불가"],
                    "abstain_reason": "insufficient_visual_evidence",
                })
            return result
        if defect_type == "good":
            text = (
                "체크리스트의 모든 항목이 아니오이며 눈에 띄는 이상이 보이지 않습니다."
                if style == "checklist" else
                "표면과 윤곽에서 눈에 띄는 이상이 보이지 않습니다."
            )
            anomaly_seen = False
            regions: list[str] = []
            features: list[str] = [] if style == "checklist" else [
                "균일한 윤곽", "일관된 표면",
            ]
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
        usage = self._usage(normalized_path, text, tokens)
        result: dict[str, Any] = {
            "text": text,
            "anomaly_seen": anomaly_seen,
            "regions": regions,
            "features": features,
            **usage,
        }
        if style == "checklist":
            checks = {key: False for key in CHECKLIST_KEYS}
            if anomaly_seen:
                if defect_type.startswith("scratch") or defect_type == "scratch":
                    checks["scratch"] = True
                elif defect_type.startswith("broken") or defect_type in {
                    "crack", "cut", "hole",
                }:
                    checks["break_or_crack"] = True
                elif defect_type == "contamination":
                    checks["contamination"] = True
                elif defect_type in {"color", "print"}:
                    checks["discoloration"] = True
                elif defect_type == "bent":
                    checks["bend_or_deformation"] = True
                elif defect_type.startswith("thread"):
                    checks["thread_damage"] = True
            result["checks"] = checks
        if style == "baseline":
            result.update({
                "verdict": "REJECT" if anomaly_seen else "PASS",
                "defect_type": defect_type if anomaly_seen else None,
                "observations": [text],
                "abstain_reason": None,
            })
        return result


class OllamaVLM(VLMBackend):
    """로컬 Ollama의 vision chat API를 사용한다."""

    def __init__(
        self,
        model: str = "qwen2.5vl:3b",
        compute_proxy_usd_per_1k: float = 0.005,
    ):
        super().__init__(model)
        self.name = f"ollama:{model}"
        self.compute_proxy_usd_per_1k = compute_proxy_usd_per_1k
        self.url = os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL).rstrip("/") + "/api/chat"

    async def _request(
        self, prompt: str, images: list[str]
    ) -> tuple[str, dict[str, Any]]:
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
        input_tokens = _usage_count(data, "prompt_eval_count", provider="Ollama")
        output_tokens = _usage_count(data, "eval_count", provider="Ollama")
        has_split = input_tokens is not None and output_tokens is not None
        tokens = input_tokens + output_tokens if has_split else 0
        observed_model = _observed_model("ollama", data.get("model"))
        compute_proxy_usd = round(
            tokens / 1000 * self.compute_proxy_usd_per_1k, 6
        )
        usage = {
            "model": observed_model,
            "tokens": tokens,
            "usd": 0.0,
            "input_tokens": input_tokens if has_split else None,
            "output_tokens": output_tokens if has_split else None,
            "token_accounting": "provider_split" if has_split else "total_only",
            "estimated_usd": 0.0,
            "compute_proxy_usd": compute_proxy_usd,
            "requested_model": self.name,
            "observed_model": observed_model,
            "observed_model_ids": [observed_model] if observed_model else [],
            "successful_provider_requests": 1,
            "unknown_token_usage_successes": 0 if has_split else 1,
            "observed_model_response_count": 1 if observed_model else 0,
            "unobserved_model_successes": 0 if observed_model else 1,
        }
        return str(content), usage


class OpenAIVLM(VLMBackend):
    """OpenAI Chat Completions의 멀티모달 입력을 사용한다."""

    def __init__(self, model: str):
        super().__init__(model)
        self.name = f"openai:{model}"
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.usd_per_million = openai_vlm_usd_per_million(model)

    async def _request(
        self, prompt: str, images: list[str]
    ) -> tuple[str, dict[str, Any]]:
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
        raw_usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        input_tokens = _usage_count(raw_usage, "prompt_tokens", provider="OpenAI")
        output_tokens = _usage_count(raw_usage, "completion_tokens", provider="OpenAI")
        total_tokens = _usage_count(raw_usage, "total_tokens", provider="OpenAI")
        has_split = input_tokens is not None and output_tokens is not None
        token_usage_known = has_split or total_tokens is not None
        if has_split:
            total_tokens = input_tokens + output_tokens
        elif total_tokens is None:
            total_tokens = 0
        estimated_usd = round(
            total_tokens / 1_000_000 * self.usd_per_million, 6
        )
        observed_model = _observed_model("openai", data.get("model"))
        usage = {
            "model": observed_model,
            "tokens": total_tokens,
            "usd": estimated_usd,
            "input_tokens": input_tokens if has_split else None,
            "output_tokens": output_tokens if has_split else None,
            "token_accounting": "provider_split" if has_split else "total_only",
            "estimated_usd": estimated_usd,
            "compute_proxy_usd": 0.0,
            "requested_model": self.name,
            "observed_model": observed_model,
            "observed_model_ids": [observed_model] if observed_model else [],
            "successful_provider_requests": 1,
            "unknown_token_usage_successes": 0 if token_usage_known else 1,
            "observed_model_response_count": 1 if observed_model else 0,
            "unobserved_model_successes": 0 if observed_model else 1,
        }
        return str(answer), usage


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
        proxy_rate = 0.00015 if role == "observer" else 0.005
        return OllamaVLM(model, compute_proxy_usd_per_1k=proxy_rate)
    if backend == "openai":
        default_model = "gpt-4o-mini" if role == "observer" else "gpt-4o"
        model = os.environ.get(env_name, default_model)
        return OpenAIVLM(model)
    raise ValueError(f"지원하지 않는 VLM_BACKEND입니다: {backend}")
