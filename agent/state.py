"""Supervisor 그래프와 research 서브그래프가 공유하는 상태."""

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    query: str
    thread_id: str
    history: list[dict]
    route: str
    needs_analysis: bool
    tasks: list[dict]
    gate_removed: int
    collected: list[dict]
    replan_rounds: int
    research_sufficient: bool
    retry_count: int
    parallel_elapsed: float
    draft: str
    validation_attempts: int
    validation_feedback: str | None
    validation_passed: bool
    final: str
    last_prompt: str
    cost_usd: float
    events: list[dict]

    image_path: str
    item: str
    lot: dict
    observations: list[dict]
    cross_check: dict
    spec_hits: list[dict]
    judgement: dict
    verdict: str
    defect_type: str | None
    confidence: str
    report: dict
    approval: dict
    validation_errors: list[str]
    ablation: str
    hitl_rejections: int

    # 병렬 서브그래프 분기는 공통 필드를 동시에 쓰지 않고 임시 결과로 넘긴다.
    _observer_result: dict[str, Any]
    _cross_result: dict[str, Any]
