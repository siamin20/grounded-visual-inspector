"""Cross-Checker 분기: 좌우 반전된 전체 이미지를 체크리스트로 확인한다."""

from __future__ import annotations

import time
from typing import Any

from langsmith import traceable

from agent.retry import with_retry
from agent.state import AgentState
from agent.vision.crop import unflip_regions


def _cross_task_indices(state: AgentState, *, pending_only: bool) -> list[int]:
    return [
        index
        for index, task in enumerate(state.get("tasks", []))
        if task.get("kind") == "cross_check"
        and (not pending_only or task.get("status", "pending") == "pending")
    ]


def _empty_result(*, replace: bool = False, skipped: bool = False,
                  done_task_indices: list[int] | None = None) -> dict:
    events = [{"type": "CrossCheckDone", "skipped": True}] if skipped else []
    return {
        "_cross_result": {
            "cross_check": {} if replace else None,
            "replace": replace,
            "events": events,
            "retries": 0,
            "elapsed": 0.0,
            "tokens": 0,
            "usd": 0.0,
            "done_task_indices": done_task_indices or [],
        }
    }


@traceable(name="cross_checker_node")
async def cross_checker_node(state: AgentState, deps: Any) -> dict:
    """pending cross_check task 1건을 실행하고 영역을 원본 좌표로 복원한다."""
    all_cross_tasks = _cross_task_indices(state, pending_only=False)
    pending = _cross_task_indices(state, pending_only=True)
    if (state.get("ablation") or getattr(deps, "ablation", "")) == "no_cross":
        return _empty_result(
            replace=True,
            skipped=True,
            done_task_indices=pending,
        )
    if not all_cross_tasks:
        return _empty_result(replace=True, skipped=True)
    if not pending:
        # center 재계획 진입 시 이미 완료한 교차 검증은 재호출하지 않는다.
        return _empty_result()

    server = await deps.mcp.find_tool("inspect_image")
    started = time.perf_counter()
    result, retries = await with_retry(
        deps.mcp.call_tool,
        "inspect_image",
        {
            "image_path": state["image_path"],
            "item": state["item"],
            "region": "full",
            "style": "checklist",
            "flip": True,
            "reference_path": None,
            "model_role": "cross",
        },
        server=server,
    )
    elapsed = time.perf_counter() - started
    restored_regions = unflip_regions(result.get("regions") or [])
    cross_check = {
        "call_id": "cc-full",
        "anomaly_seen": result.get("anomaly_seen"),
        "regions": restored_regions,
        "summary": str(result.get("text", "")),
        "model": str(result.get("model", "")),
    }
    done_event = {
        "type": "CrossCheckDone",
        "skipped": False,
        "summary": cross_check["summary"],
        "anomaly_seen": cross_check["anomaly_seen"],
        "regions": restored_regions,
    }
    return {
        "_cross_result": {
            "cross_check": cross_check,
            "replace": True,
            "events": [done_event],
            "retries": int(retries),
            "elapsed": elapsed,
            "tokens": int(result.get("tokens", 0)),
            "usd": float(result.get("usd", 0.0)),
            "done_task_indices": pending,
        }
    }
