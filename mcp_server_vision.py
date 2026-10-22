"""제품 외관 검사용 MCP vision 서버.

VLM 호출은 Tool로 격리하고, 기준서와 조치 가이드는 Resource로,
표준 검사 순서는 Prompt로 노출한다. 모델 백엔드는 역할별로 프로세스
생존 기간 동안 한 번만 생성한다.
"""
from pathlib import Path

from fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator
from typing import Literal

from agent.vision.backends import VLMBackend, make_vlm

mcp = FastMCP(name="grounded-visual-inspector-vision")

DATA_DIR = Path(__file__).parent / "data"
SPEC_DIR = DATA_DIR / "specs"
GUIDE_DIR = DATA_DIR / "guides"

_BACKENDS: dict[str, VLMBackend] = {}


class InspectResult(BaseModel):
    model_config = ConfigDict(strict=True)

    text: str = Field(description="이미지에서 관찰한 사실")
    anomaly_seen: bool | None = Field(description="이상 징후 관찰 여부")
    regions: list[str] = Field(default_factory=list, description="이상이 관찰된 영역")
    features: list[str] = Field(default_factory=list, description="관찰된 시각 특징")
    checks: dict[str, bool] | None = Field(
        default=None,
        description="checklist 응답의 6개 구조화 점검 결과",
    )
    model: str | None = Field(
        default=None,
        description="provider 응답에서 관찰된 실제 모델 ID",
    )
    tokens: int = Field(description="사용 토큰 수", ge=0)
    usd: float = Field(description="기존 호환 비용 값(실제 청구액 아님)", ge=0, allow_inf_nan=False)
    input_tokens: int | None = Field(default=None, ge=0, description="입력 토큰 수")
    output_tokens: int | None = Field(default=None, ge=0, description="출력 토큰 수")
    token_accounting: Literal[
        "provider_split", "estimated_split", "mixed_split", "total_only"
    ] = Field(description="토큰 분해 근거")
    estimated_usd: float = Field(description="OpenAI 추정 비용", ge=0, allow_inf_nan=False)
    compute_proxy_usd: float = Field(
        description="Ollama 계산 자원 프록시",
        ge=0,
        allow_inf_nan=False,
    )
    requested_model: str = Field(description="요청한 backend:model ID")
    observed_model: str | None = Field(
        default=None,
        description="provider 응답의 실제 backend:model ID",
    )
    successful_provider_requests: int = Field(
        description="성공 응답을 받은 provider 요청 수",
        ge=0,
    )
    unknown_token_usage_successes: int = Field(
        description="token usage가 누락된 provider 성공 수", ge=0
    )
    observed_model_response_count: int = Field(
        description="실제 모델 ID를 반환한 provider 성공 수", ge=0
    )
    unobserved_model_successes: int = Field(
        description="모델 ID가 누락된 provider 성공 수", ge=0
    )
    observed_model_ids: list[str] = Field(
        description="provider 응답에서 관측한 모델 ID 합집합"
    )
    region: str = Field(description="요청한 검사 영역")
    flip: bool = Field(description="좌우 반전 적용 여부")
    verdict: str | None = Field(default=None, description="baseline 판정")
    defect_type: str | None = Field(default=None, description="baseline 결함 유형")
    observations: list[str] | None = Field(default=None, description="baseline 관찰 목록")
    abstain_reason: str | None = Field(default=None, description="baseline 판정 보류 사유")

    @model_validator(mode="after")
    def _validate_usage_contract(self):
        if self.token_accounting == "total_only":
            if self.input_tokens is not None or self.output_tokens is not None:
                raise ValueError("total_only usage에 분리 토큰을 둘 수 없습니다.")
        else:
            if self.input_tokens is None or self.output_tokens is None:
                raise ValueError("분리 token accounting에 입력·출력 토큰이 필요합니다.")
            if self.input_tokens + self.output_tokens != self.tokens:
                raise ValueError("입력·출력 토큰 합이 전체와 다릅니다.")
        if (
            self.observed_model_response_count + self.unobserved_model_successes
            != self.successful_provider_requests
        ):
            raise ValueError("모델 ID 관측 카운트가 provider 성공 수와 다릅니다.")
        if self.unknown_token_usage_successes > self.successful_provider_requests:
            raise ValueError("토큰 미지 성공 수가 provider 성공 수보다 큽니다.")
        if any(not value.strip() or value != value.strip() for value in self.observed_model_ids):
            raise ValueError("관측 모델 ID는 빈 문자열일 수 없습니다.")
        if len(self.observed_model_ids) != len(set(self.observed_model_ids)):
            raise ValueError("관측 모델 ID는 중복될 수 없습니다.")
        if self.observed_model_response_count and not self.observed_model_ids:
            raise ValueError("관측 model ID 응답이 있지만 ID 목록이 비었습니다.")
        if self.observed_model is not None:
            if not self.observed_model.strip():
                raise ValueError("observed_model은 빈 문자열일 수 없습니다.")
            if self.observed_model not in self.observed_model_ids:
                raise ValueError("observed_model이 observed_model_ids에 없습니다.")
        if self.model is not None and not self.model.strip():
            raise ValueError("model은 빈 문자열일 수 없습니다.")
        if self.model != self.observed_model:
            raise ValueError("model은 observed_model의 기존 호환 별칭여야 합니다.")
        if not self.requested_model.strip():
            raise ValueError("requested_model은 빈 문자열일 수 없습니다.")
        if (
            self.successful_provider_requests
            and self.token_accounting == "total_only"
            and self.tokens == 0
            and not self.unknown_token_usage_successes
        ):
            raise ValueError("token usage 누락을 미지 성공으로 기록해야 합니다.")
        return self

    @model_serializer(mode="wrap")
    def _serialize_without_none(self, handler):
        """baseline 전용 필드는 값이 있을 때만 직렬화한다."""
        data = handler(self)
        baseline_keys = ("verdict", "defect_type", "observations", "abstain_reason")
        if not any(data[key] is not None for key in baseline_keys):
            for key in baseline_keys:
                del data[key]
        if data["checks"] is None:
            del data["checks"]
        return data


def _get_backend(model_role: str) -> VLMBackend:
    """역할별 VLM 백엔드를 한 번만 생성해 재사용한다."""
    if model_role not in _BACKENDS:
        _BACKENDS[model_role] = make_vlm(model_role)
    return _BACKENDS[model_role]


@mcp.tool()
async def inspect_image(
    image_path: str,
    item: str,
    region: str = "full",
    style: str = "observe",
    flip: bool = False,
    reference_path: str | None = None,
    model_role: str = "observer",
    extra_context: str | None = None,
) -> InspectResult:
    """이미지를 VLM으로 관찰하고 구조화된 결과를 반환한다."""
    backend = _get_backend(model_role)
    result = await backend.observe(
        image_path,
        item=item,
        region=region,
        style=style,
        flip=flip,
        reference_path=reference_path,
        extra_context=extra_context,
    )
    return InspectResult(
        text=result["text"],
        anomaly_seen=result.get("anomaly_seen"),
        regions=result.get("regions", []),
        features=result.get("features", []),
        checks=result.get("checks"),
        model=result.get("model"),
        tokens=result["tokens"],
        usd=result["usd"],
        input_tokens=result.get("input_tokens"),
        output_tokens=result.get("output_tokens"),
        token_accounting=result["token_accounting"],
        estimated_usd=result["estimated_usd"],
        compute_proxy_usd=result["compute_proxy_usd"],
        requested_model=result["requested_model"],
        observed_model=result.get("observed_model"),
        successful_provider_requests=result["successful_provider_requests"],
        unknown_token_usage_successes=result["unknown_token_usage_successes"],
        observed_model_response_count=result["observed_model_response_count"],
        unobserved_model_successes=result["unobserved_model_successes"],
        observed_model_ids=result["observed_model_ids"],
        region=region,
        flip=flip,
        verdict=result.get("verdict"),
        defect_type=result.get("defect_type"),
        observations=result.get("observations"),
        abstain_reason=result.get("abstain_reason"),
    )


@mcp.resource("spec://{item}")
def get_spec(item: str) -> str:
    """품목별 외관 검사 기준서를 반환한다."""
    spec_path = SPEC_DIR / f"{item}.md"
    if not spec_path.exists():
        return "# Spec Not Found"
    return spec_path.read_text(encoding="utf-8")


@mcp.resource("guide://{defect_type}")
def get_guide(defect_type: str) -> str:
    """결함 유형별 조치 가이드를 반환한다."""
    guide_path = GUIDE_DIR / f"{defect_type}.md"
    if not guide_path.exists():
        return "# Guide Not Found"
    return guide_path.read_text(encoding="utf-8")


@mcp.prompt()
def inspection_prompt(item: str, lot: str) -> str:
    """관찰→기준 대조→보고의 표준 검사 절차를 반환한다."""
    return f"""다음 제품을 표준 절차에 따라 검사하세요.

## 검사 대상
- 품목: {item}
- 로트: {lot}

## 검사 절차
1. 관찰: inspect_image Tool로 이미지의 위치·색·형태·상대 크기를 관찰하세요.
2. 기준 대조: spec://{item} 기준서와 관찰 결과를 대조하세요.
3. 보고: 판정과 근거, 필요한 조치를 명확히 보고하세요.
"""


if __name__ == "__main__":
    try:
        mcp.run(show_banner=False)
    except TypeError:  # 구버전 fastmcp 호환
        mcp.run()
