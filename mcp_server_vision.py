"""제품 외관 검사용 MCP vision 서버.

VLM 호출은 Tool로 격리하고, 기준서와 조치 가이드는 Resource로,
표준 검사 순서는 Prompt로 노출한다. 모델 백엔드는 역할별로 프로세스
생존 기간 동안 한 번만 생성한다.
"""
from pathlib import Path

from fastmcp import FastMCP
from pydantic import BaseModel, Field, model_serializer

from agent.vision.backends import VLMBackend, make_vlm

mcp = FastMCP(name="grounded-visual-inspector-vision")

DATA_DIR = Path(__file__).parent / "data"
SPEC_DIR = DATA_DIR / "specs"
GUIDE_DIR = DATA_DIR / "guides"

_BACKENDS: dict[str, VLMBackend] = {}


class InspectResult(BaseModel):
    text: str = Field(description="이미지에서 관찰한 사실")
    anomaly_seen: bool | None = Field(description="이상 징후 관찰 여부")
    regions: list[str] = Field(default_factory=list, description="이상이 관찰된 영역")
    features: list[str] = Field(default_factory=list, description="관찰된 시각 특징")
    model: str = Field(description="관찰에 사용한 모델")
    tokens: int = Field(default=0, description="사용 토큰 수")
    usd: float = Field(default=0.0, description="추정 비용(USD)")
    region: str = Field(description="요청한 검사 영역")
    flip: bool = Field(description="좌우 반전 적용 여부")
    verdict: str | None = Field(default=None, description="baseline 판정")
    defect_type: str | None = Field(default=None, description="baseline 결함 유형")
    observations: list[str] | None = Field(default=None, description="baseline 관찰 목록")

    @model_serializer(mode="wrap")
    def _serialize_without_none(self, handler):
        """baseline 전용 필드는 값이 있을 때만 직렬화한다."""
        data = handler(self)
        baseline_keys = ("verdict", "defect_type", "observations")
        if not any(data[key] is not None for key in baseline_keys):
            for key in baseline_keys:
                del data[key]
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
        model=result["model"],
        tokens=result.get("tokens", 0),
        usd=result.get("usd", 0.0),
        region=region,
        flip=flip,
        verdict=result.get("verdict"),
        defect_type=result.get("defect_type"),
        observations=result.get("observations"),
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
