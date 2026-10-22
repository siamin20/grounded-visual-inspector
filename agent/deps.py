"""Wave 13 — 의존성 주입.

노드 함수가 closure의 지역 변수에 암묵적으로 기대지 않도록, supervisor가 쓰는 모든
의존성을 SupervisorDeps 하나에 명시적으로 담아 functools.partial로 각 노드에 주입한다.
파일을 어디로 옮겨도, 단위 테스트에서 mock deps로 갈아 끼워도 동작이 같다.
"""
from dataclasses import dataclass
from typing import Any


SUPERVISOR_VERSIONS = frozenset({"v1", "v2"})
HITL_MODES = frozenset({"interactive", "auto_approve", "auto_reject"})
ABLATION_MODES = frozenset({"", "no_cross", "no_crops", "no_validate"})


def validate_supervisor_version(value: Any) -> str:
    """Supervisor 선택을 표준화하고 알 수 없는 값은 즉시 거부한다."""
    normalized = str(value).strip().lower()
    if normalized not in SUPERVISOR_VERSIONS:
        allowed = "|".join(sorted(SUPERVISOR_VERSIONS))
        raise ValueError(
            f"지원하지 않는 SUPERVISOR_VERSION입니다: {value!r} "
            f"(허용: {allowed})"
        )
    return normalized


def validate_hitl_mode(value: Any) -> str:
    """HITL 모드를 표준화하고 fail-open이 될 오타를 거부한다."""
    normalized = str(value).strip().lower()
    if normalized not in HITL_MODES:
        allowed = "|".join(sorted(HITL_MODES))
        raise ValueError(
            f"지원하지 않는 HITL_MODE입니다: {value!r} "
            f"(허용: {allowed})"
        )
    return normalized


def validate_ablation(value: Any) -> str:
    """Ablation 오타가 전체 조건으로 묵시적 실행되지 않게 한다."""

    normalized = str(value).strip().lower()
    if normalized not in ABLATION_MODES:
        allowed = "|".join(mode or "(empty)" for mode in sorted(ABLATION_MODES))
        raise ValueError(
            f"지원하지 않는 ABLATION입니다: {value!r} "
            f"(허용: {allowed})"
        )
    return normalized


@dataclass
class SupervisorDeps:
    fast_llm: Any        # Wave 10a — 분류/계획 등 짧은 단계용 저비용 모델
    quality_llm: Any     # Wave 10a — 최종 답변 작성용 고품질 모델
    mcp: Any             # Wave 2  — MCP stdio 클라이언트 (Tool/Resource/Prompt)
    session_store: Any   # Wave 12 — SQLite 세션 저장소
    version: str         # Wave 14 — SUPERVISOR_VERSION (v1/v2)
    judge_llm: Any = None  # Spec-Judge 전담. 생략 시 fast_llm을 재사용.
    vector_store: Any = None
    checkpointer: Any = None
    hitl_mode: str = "interactive"
    ablation: str = ""

    def __post_init__(self) -> None:
        """잘못된 실행 모드가 그래프까지 전파되기 전에 중단한다."""
        self.version = validate_supervisor_version(self.version)
        self.hitl_mode = validate_hitl_mode(self.hitl_mode)
        self.ablation = validate_ablation(self.ablation)
        if self.judge_llm is None:
            # _build_llms의 3-tuple 공통 인터페이스를 유지하면서
            # Ollama에서만 fast 번들에 붙인 전담 Judge를 승격한다.
            self.judge_llm = getattr(self.fast_llm, "judge_llm", self.fast_llm)
