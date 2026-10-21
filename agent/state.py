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
    validation_terminal: bool
    final: str
    last_prompt: str
    cost_usd: float
    tokens: int
    input_tokens: int | None
    output_tokens: int | None
    token_accounting: str | None
    estimated_usd: float
    compute_proxy_usd: float
    observed_model_ids: list[str]
    successful_provider_requests: int
    successful_vlm_requests: int
    unknown_token_usage_successes: int
    unobserved_model_successes: int
    observed_model_response_count: int
    events: list[dict]

    image_path: str
    item: str
    lot: dict
    observations: list[dict]
    cross_check: dict
    spec_hits: list[dict]
    judgement: dict
    judgement_status: str
    verdict: str
    defect_type: str | None
    confidence: str
    abstain_reason: str | None
    observer_status: str
    cross_status: str
    report: dict
    approval: dict
    reapproval_required: bool
    validation_errors: list[str]
    pre_validation_snapshot: dict[str, Any]
    ablation: str
    hitl_rejections: int

    # 병렬 서브그래프 분기는 공통 필드를 동시에 쓰지 않고 임시 결과로 넘긴다.
    _observer_result: dict[str, Any]
    _cross_result: dict[str, Any]
