"""판정 근거와 조치 가이드를 보고서 JSON으로 작성한다."""

import json
import re

from langgraph.config import get_stream_writer
from langsmith import traceable

from agent.deps import SupervisorDeps
from agent.state import AgentState

from .common import _cost, event


def _parse_report(draft: str) -> dict:
    content = draft.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content)
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("보고서는 JSON 객체여야 합니다.")
    return parsed


def _merge_report_observations(report: dict) -> dict:
    """관찰문이 같은 항목을 합치고 호출 ID는 등장 순서대로 보존한다."""
    observations = report.get("observations")
    if not isinstance(observations, list):
        return report

    merged: list = []
    positions: dict[str, int] = {}
    for observation in observations:
        if not isinstance(observation, dict) or not isinstance(
            observation.get("text"), str
        ):
            merged.append(observation)
            continue

        text = observation["text"]
        raw_ids = observation.get("source_call_id")
        source_ids = list(raw_ids) if isinstance(raw_ids, list) else [raw_ids]
        if text not in positions:
            normalized = dict(observation)
            normalized["source_call_id"] = source_ids
            positions[text] = len(merged)
            merged.append(normalized)
            continue

        existing = merged[positions[text]].get("source_call_id", [])
        for source_id in source_ids:
            if source_id not in existing:
                existing.append(source_id)

    normalized_report = dict(report)
    normalized_report["observations"] = merged
    return normalized_report


@traceable(name="write_node")
async def write_node(state: AgentState, deps: SupervisorDeps) -> dict:
    guide_text = ""
    if state.get("verdict") == "REJECT" and state.get("defect_type"):
        server, uri_template = await deps.mcp.find_resource_template("get_guide")
        guide_text = await deps.mcp.read_resource(
            uri_template.format(defect_type=state["defect_type"]), server=server
        )

    attempt = int(state.get("validation_attempts", 0)) + 1
    context = {
        "query": state.get("query", ""),
        "item": state.get("item"),
        "lot": state.get("lot", {}),
        "observations": state.get("observations", []),
        "cross_check": state.get("cross_check", {}),
        "spec_hits": state.get("spec_hits", []),
        "judgement": state.get("judgement", {}),
        "verdict": state.get("verdict"),
        "defect_type": state.get("defect_type"),
        "confidence": state.get("confidence"),
        "guide_text": guide_text,
    }
    writer = get_stream_writer()
    draft, usage = await deps.quality_llm.write_report(
        context,
        attempt,
        state.get("validation_feedback"),
        on_token=lambda piece: writer(
            {"type": "AnswerToken", "delta": piece, "attempt": attempt}
        ),
    )
    feedback = None
    try:
        report = _merge_report_observations(_parse_report(draft))
    except (json.JSONDecodeError, TypeError, ValueError):
        report = {}
        feedback = "JSON 파싱 실패"

    return {
        "draft": draft,
        "report": report,
        "validation_attempts": attempt,
        "validation_feedback": feedback,
        "last_prompt": usage.get("prompt"),
        "cost_usd": _cost(state, usage),
        "events": event(state, "DraftWritten", attempt=attempt),
    }
