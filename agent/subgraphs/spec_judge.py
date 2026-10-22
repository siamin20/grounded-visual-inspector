"""Spec-Judge 단계: 관찰문을 기준서 조항과 대조하고 결과를 검증한다."""

from __future__ import annotations

from typing import Any

from langsmith import traceable

from agent.nodes.common import (
    _merge_usages,
    _usage_updates,
    cross_status,
    is_readable_observation,
    load_type_map,
)
from agent.state import AgentState
from agent.usage import MISSING, nonnegative_int


_MAX_JUDGE_VIOLATIONS = 3
_MAX_JUDGE_RATIONALE_CHARS = 240


def _mark_tasks_done(tasks: list[dict], indices: set[int]) -> list[dict]:
    updated = [dict(task) for task in tasks]
    for index, task in enumerate(updated):
        if index in indices or (
            task.get("kind") == "spec_judge"
            and task.get("status", "pending") == "pending"
        ):
            task["status"] = "done"
    return updated


def _string_list(value: Any) -> list[str]:
    """모델 응답에서 빈 문자열을 빼고 순서를 유지해 중복을 제거한다."""
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(
        entry.strip()
        for entry in value
        if isinstance(entry, str) and entry.strip()
    ))


def _build_evidence_text(observations: Any, cross_check: Any) -> str:
    """검색·Judge가 공유하는 시각 근거를 결정론적으로 구성한다.

    정답 라벨·파일 경로가 아니라 Observer와 Cross-Checker가 반환한
    관찰 필드만 사용한다. 양성 Observer가 하나라도 있으면 검색 잡음을
    줄이기 위해 양성 관찰만 선택한다.
    """
    readable = [
        observation
        for observation in (observations if isinstance(observations, list) else [])
        if is_readable_observation(observation)
    ]
    positive = [
        observation for observation in readable
        if observation.get("anomaly_seen") is True
    ]
    cross_positive = (
        isinstance(cross_check, dict)
        and cross_status(cross_check) == "C+"
    )
    # Observer 양성이 없고 Cross만 양성이면 일반 사물 설명인
    # Observer 문장을 섞지 않는다. 둘 다 양성이 아닐 때만
    # 정상/판독 근거로 readable Observer를 유지한다.
    selected = positive if positive else ([] if cross_positive else readable)

    lines: list[str] = []
    observer_features: list[str] = []
    for observation in selected:
        text = str(observation.get("text", "")).strip()
        if text:
            region = str(observation.get("region", "unknown")).strip() or "unknown"
            lines.append(f"Observer({region}): {text}")
        observer_features.extend(_string_list(observation.get("features")))
    observer_features = list(dict.fromkeys(observer_features))
    if observer_features:
        lines.append("Observer 특징: " + ", ".join(observer_features))

    if (
        cross_positive
        and is_readable_observation({
            "text": cross_check.get("summary"),
            "anomaly_seen": True,
        })
    ):
        lines.append(f"Cross-Checker: {str(cross_check['summary']).strip()}")
        cross_features = _string_list(cross_check.get("features"))
        if cross_features:
            lines.append("Cross-Checker 특징: " + ", ".join(cross_features))

    return "\n".join(dict.fromkeys(lines))


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
    rationale_text = rationale.strip() if isinstance(rationale, str) else ""
    shape_valid = (
        isinstance(raw, dict)
        and {"violations", "defect_type", "rationale"} <= set(source)
        and violations_are_strings
        and len(raw_violations) <= _MAX_JUDGE_VIOLATIONS
        and len(set(raw_violations)) == len(raw_violations)
        and (raw_defect_type is None or isinstance(raw_defect_type, str))
        and isinstance(rationale, str)
        and bool(rationale_text)
        and len(rationale_text) <= _MAX_JUDGE_RATIONALE_CHARS
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
        "rationale": rationale_text,
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
    replace_cross = bool(cross_result.get("replace", False))
    cross_check = (
        cross_result.get("cross_check", {})
        if replace_cross
        else state.get("cross_check", {})
    )
    evidence_text = _build_evidence_text(observations, cross_check)

    usage: dict | None = None
    embedding_usage: dict | None = None
    raw_judgement: Any = None
    if evidence_text:
        spec_hits = list(deps.vector_store.search(
            evidence_text,
            item=state["item"],
            k=3,
        ))[:3]
        embedder = getattr(deps.vector_store, "embedder", None)
        last_usage = getattr(embedder, "last_usage", None)
        if isinstance(last_usage, dict):
            embedding_usage = last_usage
        judge_llm = getattr(deps, "judge_llm", None) or deps.fast_llm
        raw_judgement, usage = judge_llm.judge(
            evidence_text,
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
    observer_usage = observer_result.get("usage")
    cross_usage = cross_result.get("usage")
    # 기존 합성 테스트 fixture는 branch usage 객체 대신 합계만 줄 수 있다.
    # 0건 branch를 total_only로 오인하지 않고, 실제 값이 있을 때만 호환한다.
    if not isinstance(observer_usage, dict) and (
        observer_result.get("tokens", 0) or observer_result.get("usd", 0.0)
    ):
        observer_usage = {
            "tokens": observer_result.get("tokens", 0),
            "usd": observer_result.get("usd", 0.0),
        }
    if not isinstance(cross_usage, dict) and (
        cross_result.get("tokens", 0) or cross_result.get("usd", 0.0)
    ):
        cross_usage = {
            "tokens": cross_result.get("tokens", 0),
            "usd": cross_result.get("usd", 0.0),
        }
    total_usage = _merge_usages(
        usage,
        embedding_usage,
        observer_usage if isinstance(observer_usage, dict) else None,
        cross_usage if isinstance(cross_usage, dict) else None,
    )
    successful_vlm_requests = nonnegative_int(
        state.get("successful_vlm_requests", MISSING),
        name="state.successful_vlm_requests",
        default=0,
    ) + nonnegative_int(
        observer_result.get("successful_provider_requests", MISSING),
        name="observer.successful_provider_requests",
        default=0,
    ) + nonnegative_int(
        cross_result.get("successful_provider_requests", MISSING),
        name="cross.successful_provider_requests",
        default=0,
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
        # 후속 RAG 품질 평가가 동일한 Observer·Cross 근거를
        # 재사용할 수 있도록 실제 검색 문자열을 보존한다.
        "rag_query_text": evidence_text,
        "spec_hits": spec_hits,
        "raw_judgement": raw_judgement,
        "judgement": judgement,
        "judgement_corrected": corrected,
        "judgement_status": judgement["status"],
        "tasks": tasks,
        "retry_count": retry_count,
        "parallel_elapsed": parallel_elapsed,
        **_usage_updates(state, total_usage),
        "successful_vlm_requests": successful_vlm_requests,
        "events": events,
    }
