"""Spec-Judge 단계: 관찰문을 기준서 조항과 대조하고 결과를 검증한다."""

from __future__ import annotations

from typing import Any

from langsmith import traceable

from agent.nodes.common import (
    _cost,
    _tokens,
    is_readable_observation,
    load_type_map,
)
from agent.state import AgentState


def _mark_tasks_done(tasks: list[dict], indices: set[int]) -> list[dict]:
    updated = [dict(task) for task in tasks]
    for index, task in enumerate(updated):
        if index in indices or (
            task.get("kind") == "spec_judge"
            and task.get("status", "pending") == "pending"
        ):
            task["status"] = "done"
    return updated


def _validated_judgement(raw: Any, spec_hits: list[dict], item: str) -> tuple[dict, bool]:
    """LLM이 검색 결과 밖의 조항·유형을 인용하지 못하게 교정한다."""
    source = raw if isinstance(raw, dict) else {}
    original_violations = source.get("violations")
    violations_are_list = isinstance(original_violations, list)
    violations_are_strings = violations_are_list and all(
        isinstance(value, str) for value in original_violations
    )
    raw_violations = original_violations if violations_are_strings else []
    raw_defect_type = source.get("defect_type")
    rationale = source.get("rationale")
    shape_valid = (
        isinstance(raw, dict)
        and {"violations", "defect_type", "rationale"} <= set(source)
        and violations_are_strings
        and (raw_defect_type is None or isinstance(raw_defect_type, str))
        and isinstance(rationale, str)
        and bool(rationale.strip())
    )
    clause_types = {
        str(hit["clause_id"]): str(hit["defect_type"])
        for hit in spec_hits
        if hit.get("clause_id") is not None
        and hit.get("item") == item
        and isinstance(hit.get("defect_type"), str)
        and hit.get("defect_type") != "good"
    }
    violations = list(dict.fromkeys(
        clause_id for clause_id in raw_violations if clause_id in clause_types
    ))
    all_claims_supported = bool(raw_violations) and all(
        clause_id in clause_types for clause_id in raw_violations
    )

    type_map = load_type_map()
    allowed_types = set(type_map.get(item, {})) - {"good"}
    matching_types = {clause_types[clause_id] for clause_id in violations}
    defect_type = (
        raw_defect_type
        if all_claims_supported
        and violations
        and isinstance(raw_defect_type, str)
        and raw_defect_type in allowed_types
        and matching_types == {raw_defect_type}
        else None
    )
    no_claim = (
        shape_valid
        and raw_violations == []
        and raw_defect_type is None
    )
    if (
        shape_valid
        and all_claims_supported
        and violations
        and defect_type is not None
    ):
        status = "grounded"
    elif no_claim:
        status = "none"
    else:
        status = "invalid"
    corrected = (
        not shape_valid
        or violations != raw_violations
        or defect_type != raw_defect_type
        or status == "invalid"
    )
    judgement = {
        "violations": violations,
        "defect_type": defect_type,
        "rationale": rationale.strip() if isinstance(rationale, str) else "",
        "status": status,
    }
    return judgement, corrected


@traceable(name="spec_judge_node")
async def spec_judge_node(state: AgentState, deps: Any) -> dict:
    """두 병렬 분기 결과를 합친 뒤 기준서 검색과 조항 판정을 순차 실행한다."""
    observer_result = state.get("_observer_result") or {}
    cross_result = state.get("_cross_result") or {}
    observations = [
        *state.get("observations", []),
        *observer_result.get("observations", []),
    ]
    readable = [
        observation
        for observation in observations
        if is_readable_observation(observation)
    ]
    observations_text = "\n".join(
        str(observation.get("text", "")).strip()
        for observation in readable
        if str(observation.get("text", "")).strip()
    )

    usage: dict = {"tokens": 0, "usd": 0.0}
    embedding_usage: dict = {"tokens": 0, "usd": 0.0}
    if observations_text:
        spec_hits = list(deps.vector_store.search(
            observations_text,
            item=state["item"],
            k=3,
        ))[:3]
        embedder = getattr(deps.vector_store, "embedder", None)
        last_usage = getattr(embedder, "last_usage", None)
        if isinstance(last_usage, dict):
            embedding_usage = last_usage
        raw_judgement, usage = deps.fast_llm.judge(
            observations_text,
            spec_hits,
            state["item"],
        )
        judgement, corrected = _validated_judgement(
            raw_judgement, spec_hits, state["item"]
        )
    else:
        spec_hits = []
        judgement = {
            "violations": [],
            "defect_type": None,
            "rationale": "관찰 없음",
            "status": "none",
        }
        corrected = False

    replace_cross = bool(cross_result.get("replace", False))
    cross_check = (
        cross_result.get("cross_check", {})
        if replace_cross
        else state.get("cross_check", {})
    )
    done_indices = {
        *observer_result.get("done_task_indices", []),
        *cross_result.get("done_task_indices", []),
    }
    tasks = _mark_tasks_done(state.get("tasks", []), done_indices)

    observer_elapsed = float(observer_result.get("elapsed", 0.0))
    cross_elapsed = float(cross_result.get("elapsed", 0.0))
    parallel_elapsed = round(
        float(state.get("parallel_elapsed", 0.0))
        + max(observer_elapsed, cross_elapsed),
        6,
    )
    retry_count = (
        int(state.get("retry_count", 0))
        + int(observer_result.get("retries", 0))
        + int(cross_result.get("retries", 0))
    )
    vlm_usd = (
        float(observer_result.get("usd", 0.0))
        + float(cross_result.get("usd", 0.0))
    )
    vlm_tokens = (
        int(observer_result.get("tokens", 0))
        + int(cross_result.get("tokens", 0))
    )
    total_usage = dict(usage or {})
    total_usage["tokens"] = (
        int(total_usage.get("tokens", 0))
        + int(embedding_usage.get("tokens", 0))
        + vlm_tokens
    )
    total_usage["usd"] = (
        float(total_usage.get("usd", 0.0))
        + float(embedding_usage.get("usd", 0.0))
        + vlm_usd
    )

    clause_ids = [
        str(hit["clause_id"])
        for hit in spec_hits
        if hit.get("clause_id") is not None
    ]
    judge_event = {
        "type": "SpecJudged",
        "clause_ids": clause_ids,
        "violations": judgement["violations"],
        "defect_type": judgement["defect_type"],
        "judgement_status": judgement["status"],
        "corrected": corrected,
    }
    events = [
        *state.get("events", []),
        *observer_result.get("events", []),
        *cross_result.get("events", []),
        judge_event,
    ]
    return {
        "observations": observations,
        "cross_check": cross_check,
        "spec_hits": spec_hits,
        "judgement": judgement,
        "judgement_status": judgement["status"],
        "tasks": tasks,
        "retry_count": retry_count,
        "parallel_elapsed": parallel_elapsed,
        "cost_usd": _cost(state, total_usage),
        "tokens": _tokens(state, total_usage),
        "events": events,
    }
