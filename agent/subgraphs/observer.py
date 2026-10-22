"""Observer 분기: 전체·사분면 이미지를 병렬로 관찰한다."""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

from langsmith import traceable

from agent.retry import with_retry
from agent.state import AgentState
from agent.nodes.common import _merge_usages, _vlm_result_usage
from agent.vision.reference import normal_reference_path


_DEFAULT_REGIONS = ("full", "tl", "tr", "bl", "br")
_DECISION_WORDS = re.compile(
    r"불합격|불량|합격|\bdefect\b|\breject\b|\bpass\b",
    flags=re.IGNORECASE,
)


def _pending_task_indices(state: AgentState, kind: str) -> list[int]:
    return [
        index
        for index, task in enumerate(state.get("tasks", []))
        if task.get("kind") == kind and task.get("status", "pending") == "pending"
    ]


def _regions_to_observe(state: AgentState, deps: Any) -> tuple[list[str], list[int]]:
    """현재 pending observe task만 실행할 영역과 해당 task index를 돌려준다."""
    pending = _pending_task_indices(state, "observe")
    center_tasks = [
        index for index in pending
        if state.get("tasks", [])[index].get("region") == "center"
    ]
    if center_tasks:
        return ["center"], center_tasks

    initial_tasks = [
        index for index in pending
        if not state.get("tasks", [])[index].get("region")
    ]
    if not initial_tasks:
        return [], []
    ablation = state.get("ablation") or getattr(deps, "ablation", "")
    regions = ["full"] if ablation == "no_crops" else list(
        _DEFAULT_REGIONS
    )
    return regions, initial_tasks


def _filter_decision_words(text: str) -> tuple[str, list[str]]:
    """관찰문에서 판정에 해당하는 단어만 제거한다."""
    filtered_words = list(dict.fromkeys(
        match.group(0).lower() for match in _DECISION_WORDS.finditer(text)
    ))
    filtered = _DECISION_WORDS.sub("", text)
    filtered = re.sub(r"[ \t]{2,}", " ", filtered)
    filtered = re.sub(r"\s+([,.;:!?])", r"\1", filtered).strip()
    return filtered, filtered_words


@traceable(name="observer_node")
async def observer_node(state: AgentState, deps: Any) -> dict:
    """pending observe task를 병렬 실행하고 branch 전용 임시 결과를 남긴다."""
    regions, task_indices = _regions_to_observe(state, deps)
    if not regions:
        return {
            "_observer_result": {
                "observations": [],
                "events": [],
                "retries": 0,
                "elapsed": 0.0,
                "tokens": 0,
                "usd": 0.0,
                "usage": None,
                "successful_provider_requests": 0,
                "done_task_indices": [],
            }
        }

    server = await deps.mcp.find_tool("inspect_image")
    reference_path = normal_reference_path(state["item"])

    async def observe_one(region: str) -> dict:
        result, retries = await with_retry(
            deps.mcp.call_tool,
            "inspect_image",
            {
                "image_path": state["image_path"],
                "item": state["item"],
                "region": region,
                "style": "observe",
                "flip": False,
                "reference_path": reference_path,
                "model_role": "observer",
            },
            server=server,
        )
        return {"region": region, "result": result, "retries": retries}

    started = time.perf_counter()
    calls = await asyncio.gather(*(observe_one(region) for region in regions))
    elapsed = time.perf_counter() - started

    observations: list[dict] = []
    events: list[dict] = []
    retry_count = 0
    for call in calls:
        region = call["region"]
        result = call["result"]
        text, filtered_words = _filter_decision_words(str(result.get("text", "")))
        observation = {
            "call_id": f"obs-{region}",
            "region": region,
            "text": text,
            "anomaly_seen": result.get("anomaly_seen"),
            "regions": list(result.get("regions") or []),
            "features": list(result.get("features") or []),
            "model": result.get("model") if isinstance(result.get("model"), str) else None,
        }
        observations.append(observation)
        ready_event = {
            "type": "ObservationReady",
            "region": region,
            "text": observation["text"],
            "anomaly_seen": observation["anomaly_seen"],
            "regions": observation["regions"],
        }
        if filtered_words:
            ready_event["filtered_words"] = filtered_words
        events.append(ready_event)
        retry_count += int(call["retries"])
    merged_usage = _merge_usages(*(
        _vlm_result_usage(call["result"]) for call in calls
    ))

    return {
        "_observer_result": {
            "observations": observations,
            "events": events,
            "retries": retry_count,
            "elapsed": elapsed,
            "tokens": int(merged_usage.get("tokens", 0)),
            "usd": float(merged_usage.get("usd", 0.0)),
            "usage": merged_usage,
            "successful_provider_requests": int(
                merged_usage.get("successful_provider_requests", 0)
            ),
            "done_task_indices": task_indices,
        }
    }
