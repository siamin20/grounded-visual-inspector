"""구조화 보고서를 사람이 읽는 마크다운으로 렌더한다."""

import json

from langsmith import traceable

from agent.deps import SupervisorDeps
from agent.state import AgentState

from .common import event


def _source_label(source_call_id) -> str:
    return f"`{source_call_id}`" if isinstance(source_call_id, str) else ""


@traceable(name="finalize_node")
async def finalize_node(state: AgentState, deps: SupervisorDeps) -> dict:
    del deps
    report = state.get("report") or {}
    observations = report.get("observations") or []
    observation_lines = [
        f"- {value.get('text', '')} ({_source_label(value.get('source_call_id'))})"
        for value in observations
        if isinstance(value, dict)
    ] or ["- 관찰 기록 없음"]
    clauses = ", ".join(report.get("spec_clauses") or []) or "없음"
    actions = ", ".join(report.get("actions") or []) or "없음"
    lot = json.dumps(report.get("lot") or {}, ensure_ascii=False)
    verdict = report.get("verdict", "UNKNOWN")
    verdict_label = (
        "판정 보류 — 재검사 필요 (ABSTAIN)"
        if verdict == "ABSTAIN"
        else str(verdict)
    )
    reason_line = (
        f"- 보류 사유: {report.get('abstain_reason')}"
        if verdict == "ABSTAIN"
        else "- 보류 사유: 없음"
    )
    final = "\n".join(
        [
            "# 제품 검사 보고서",
            "",
            f"- 판정: **{verdict_label}**",
            f"- 결함 유형: {report.get('defect_type') or '없음'}",
            f"- 확신도: {report.get('confidence', 'low')}",
            reason_line,
            f"- 로트: `{lot}`",
            "",
            "## 관찰 근거",
            *observation_lines,
            "",
            f"- 위반 조항: {clauses}",
            f"- 조치: {actions}",
            "",
            str(report.get("summary", "")),
        ]
    )
    return {
        "final": final,
        "events": event(state, "Finalize", pipeline="inspection-v2"),
    }
