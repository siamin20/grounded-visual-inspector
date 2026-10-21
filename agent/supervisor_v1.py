"""Before 기준선: VLM 한 번으로 판정하는 선형 supervisor."""

import json
import os
from functools import partial
import re
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from .deps import SupervisorDeps, validate_hitl_mode
from .guide_actions import load_guide_actions, resolve_reject_actions
from .nodes.common import event, load_lot
from .nodes.hitl import hitl_node
from .nodes.report_contract import deterministic_summary
from .readability import is_unreadable_text
from .retry import with_retry
from .state import AgentState
from .typemap import load_type_map


ROOT = Path(__file__).resolve().parent.parent
ABSTAIN_REASONS = {
    "insufficient_visual_evidence",
    "ungrounded_anomaly",
    "evidence_conflict",
    "validation_failure",
}
BASELINE_MODES = frozenset({"B0", "B1"})


def validate_baseline_mode(value: Any) -> str:
    """Before 조건 오타를 B0로 묵시적 보정하지 않는다."""
    normalized = str(value).strip().upper()
    if normalized not in BASELINE_MODES:
        allowed = "|".join(sorted(BASELINE_MODES))
        raise ValueError(
            f"지원하지 않는 BASELINE_MODE입니다: {value!r} "
            f"(허용: {allowed})"
        )
    return normalized


def _normalize_baseline_result(
    result: dict[str, Any], item: str, type_map: dict[str, Any]
) -> tuple[str, str | None, str | None, list[str]]:
    """B0/B1 VLM 출력을 fail-closed 공통 판정 계약으로 바꾼다."""
    raw_verdict = result.get("verdict")
    verdict = raw_verdict.strip().upper() if isinstance(raw_verdict, str) else ""
    raw_defect = result.get("defect_type")
    invalid_defect_shape = (
        "defect_type" not in result
        or (raw_defect is not None and not isinstance(raw_defect, str))
        or (
            isinstance(raw_defect, str)
            and (
                not raw_defect.strip()
                or raw_defect.strip().lower() in {"null", "none", "good"}
                or raw_defect.strip() == "\uc815\uc0c1"
            )
        )
    )
    defect_type = raw_defect.strip() if isinstance(raw_defect, str) else None
    if (
        invalid_defect_shape
        or not defect_type
        or defect_type.lower() in {"null", "none", "good"}
        or defect_type == "정상"
    ):
        defect_type = None

    raw_observations = result.get("observations")
    invalid_observations_shape = not isinstance(raw_observations, list) or any(
        not isinstance(value, str) for value in raw_observations
    )
    observations = [
        value.strip()
        for value in raw_observations
        if isinstance(value, str) and value.strip()
    ] if isinstance(raw_observations, list) else []
    if not observations and isinstance(result.get("text"), str):
        text = result["text"].strip()
        if text:
            observations = [text]

    requested_reason = result.get("abstain_reason")
    reason = requested_reason if requested_reason in ABSTAIN_REASONS else None
    if invalid_defect_shape or invalid_observations_shape:
        return "ABSTAIN", None, "validation_failure", observations or ["판독 불가"]
    if verdict == "ABSTAIN" and reason is not None and defect_type is None:
        return "ABSTAIN", None, reason, observations or ["판독 불가"]
    if not [
        value for value in raw_observations
        if isinstance(value, str) and value.strip()
    ]:
        return (
            "ABSTAIN",
            None,
            "insufficient_visual_evidence",
            observations or ["판독 불가"],
        )
    if not observations or any(is_unreadable_text(text) for text in observations):
        return (
            "ABSTAIN",
            None,
            "insufficient_visual_evidence",
            observations or ["판독 불가"],
        )
    if not all(re.search(r"[ㄱ-ㆎ가-힣]", text) for text in observations):
        return "ABSTAIN", None, "validation_failure", observations
    if verdict == "ABSTAIN":
        if defect_type is not None or (
            requested_reason is not None and requested_reason not in ABSTAIN_REASONS
        ):
            return "ABSTAIN", None, "validation_failure", observations
        return "ABSTAIN", None, reason or "insufficient_visual_evidence", observations
    if verdict not in {"PASS", "REJECT"}:
        return "ABSTAIN", None, "validation_failure", observations

    anomaly_seen = result.get("anomaly_seen")
    if anomaly_seen is not None and not isinstance(anomaly_seen, bool):
        return "ABSTAIN", None, "validation_failure", observations
    if verdict == "PASS":
        if defect_type is not None or anomaly_seen is True:
            return "ABSTAIN", None, "validation_failure", observations
        return "PASS", None, None, observations

    allowed_types = {name for name in type_map.get(item, {}) if name != "good"}
    if defect_type not in allowed_types or anomaly_seen is False:
        return "ABSTAIN", None, "validation_failure", observations
    return "REJECT", defect_type, None, observations


def _render_report(report: dict) -> str:
    observations = "\n".join(
        f"- {value['text']} (`{value['source_call_id']}`)"
        for value in report["observations"]
    ) or "- 관찰 기록 없음"
    return (
        "# 제품 검사 보고서\n\n"
        f"- 판정: **{report['verdict']}**\n"
        f"- 결함 유형: {report['defect_type'] or '없음'}\n"
        f"- 확신도: {report['confidence']}\n"
        f"- 판정 보류 사유: {report['abstain_reason'] or '해당 없음'}\n\n"
        f"## 관찰 근거\n{observations}\n\n"
        f"- 조치: {', '.join(report['actions'])}\n"
        f"- 로트: `{json.dumps(report['lot'], ensure_ascii=False)}`\n\n"
        f"{report['summary']}"
    )


async def _prepare_v1(
    deps: SupervisorDeps,
    message: str,
    thread_id: str,
    history: list[dict],
    image_path: str | None = None,
    item: str | None = None,
) -> dict:
    del thread_id
    baseline_mode = validate_baseline_mode(
        os.environ.get("BASELINE_MODE", "B0")
    )
    type_map = load_type_map()
    if not image_path:
        reply, usage = await deps.fast_llm.chat(message, history)
        return {
            "final": reply,
            "route": "chat",
            "events": [
                {"type": "RouteDecided", "route": "chat"},
                {"type": "Finalize", "pipeline": "baseline-v1"},
            ],
            "cost_usd": usage.get("usd", 0.0),
            "tokens": int(usage.get("tokens", 0)),
            "validation_attempts": 0,
            "replan_rounds": 0,
            "gate_removed": 0,
            "retry_count": 0,
        }
    if item not in type_map:
        reply = f"품목을 선택해 주세요({'/'.join(type_map)})"
        return {
            "final": reply,
            "route": "needs_item",
            "events": [
                {"type": "RouteDecided", "route": "needs_item"},
                {"type": "Finalize", "pipeline": "baseline-v1"},
            ],
            "cost_usd": 0.0,
            "tokens": 0,
            "validation_attempts": 0,
            "replan_rounds": 0,
            "gate_removed": 0,
            "retry_count": 0,
        }

    defect_types = [name for name in type_map[item] if name != "good"]
    context = f"가능한 결함 유형: {', '.join(defect_types)}"
    if baseline_mode == "B1":
        resource_server, uri_template = await deps.mcp.find_resource_template("get_spec")
        spec_text = await deps.mcp.read_resource(
            uri_template.format(item=item), server=resource_server
        )
        context = f"{context}\n\n{spec_text}"

    reference = ROOT / "data" / "mvtec" / item / "train" / "good" / "000.png"
    tool_server = await deps.mcp.find_tool("inspect_image")
    result, retries = await with_retry(
        deps.mcp.call_tool,
        "inspect_image",
        {
            "image_path": image_path,
            "item": item,
            "region": "full",
            "style": "baseline",
            "flip": False,
            "reference_path": str(reference) if reference.exists() else None,
            "model_role": "observer",
            "extra_context": context,
        },
        server=tool_server,
    )
    verdict, defect_type, abstain_reason, observations = _normalize_baseline_result(
        result, item, type_map
    )
    reject_actions: list[str] = []
    if verdict == "REJECT":
        reject_actions = resolve_reject_actions(
            [], load_guide_actions(defect_type)
        )
        if not reject_actions:
            # 시각 판정은 있어도 검증된 후속 조치가 없으면 안전하게
            # 집행할 수 없으므로 결함 확정 대신 재검사를 요청한다.
            verdict = "ABSTAIN"
            defect_type = None
            abstain_reason = "validation_failure"
    confidence = "low" if verdict == "ABSTAIN" else "high"
    actions = (
        ["통과"] if verdict == "PASS"
        else reject_actions if verdict == "REJECT"
        else ["재검사"]
    )
    summary = (
        "단일 VLM이 판독하지 못해 판정을 보류했습니다. 재검사가 필요합니다."
        if verdict == "ABSTAIN"
        else "단일 VLM 판독을 Before 기준선 보고서로 정리했습니다."
    )
    report = {
        "verdict": verdict,
        "defect_type": defect_type,
        "confidence": confidence,
        "abstain_reason": abstain_reason,
        "observations": [
            {"text": text, "source_call_id": "baseline-full"}
            for text in observations
        ],
        "spec_clauses": [],
        "actions": actions,
        "lot": load_lot(image_path),
        "summary": summary,
    }
    events = [
        {"type": "RouteDecided", "route": "inspection"},
        {
            "type": "ObservationReady",
            "region": "full",
            "call_id": "baseline-full",
            "text": observations[0] if observations else "판독 불가",
            "anomaly_seen": result.get("anomaly_seen"),
        },
        {
            "type": "AnalysisDecided",
            "verdict": verdict,
            "confidence": confidence,
            "abstain_reason": abstain_reason,
        },
        {"type": "DraftWritten", "attempt": 1},
    ]
    return {
        "final": "",
        "route": "inspection",
        "events": events,
        "report": report,
        "draft": json.dumps(report, ensure_ascii=False),
        "verdict": verdict,
        "defect_type": defect_type,
        "confidence": confidence,
        "abstain_reason": abstain_reason,
        "observations": report["observations"],
        "approval": {},
        "cost_usd": float(result.get("usd", 0.0)),
        "tokens": int(result.get("tokens", 0)),
        "validation_attempts": 0,
        "replan_rounds": 0,
        "gate_removed": 0,
        "retry_count": retries,
    }


@traceable(name="supervisor_v1_baseline_node")
async def _baseline_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """v1 단일 VLM 결과를 HITL 직전의 초안 상태로 만든다."""
    validate_hitl_mode(getattr(deps, "hitl_mode", "interactive"))
    return await _prepare_v1(
        deps,
        str(state.get("query", "")),
        str(state.get("thread_id", "")),
        list(state.get("history", [])),
        image_path=state.get("image_path"),
        item=state.get("item"),
    )


@traceable(name="supervisor_v1_finalize")
async def _finalize_v1_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """HITL 결정을 보존한 채 Before 보고서를 최종 렌더링한다."""
    del deps
    report = state.get("report") or {}
    return {
        "final": _render_report(report),
        "events": event(state, "Finalize", pipeline="baseline-v1"),
    }


def _after_baseline(state: AgentState) -> str:
    return "hitl" if state.get("route") == "inspection" else "done"


def _report_requires_hitl(report: dict) -> bool:
    raw_actions = report.get("actions")
    actions = {
        action for action in raw_actions if isinstance(action, str)
    } if isinstance(raw_actions, list) else set()
    return (
        report.get("verdict") == "ABSTAIN"
        or (
            report.get("verdict") == "REJECT"
            and bool(actions & {"폐기", "라인정지"})
        )
        or report.get("confidence") == "low"
    )


def _rejected_v1_updates(state: dict) -> dict:
    """반려된 Before 보고서를 모델 재호출 없이 안전한 보류로 바꾼다."""
    report = dict(state.get("report") or {})
    report.update(
        verdict="ABSTAIN",
        defect_type=None,
        confidence="low",
        abstain_reason="validation_failure",
        spec_clauses=[],
        actions=["재검사"],
    )
    report["summary"] = deterministic_summary(report)
    events = event(
        state,
        "AnalysisDecided",
        verdict="ABSTAIN",
        confidence="low",
        abstain_reason="validation_failure",
        reason="hitl_rejected",
    )
    attempt = 1 + sum(
        value.get("type") == "DraftWritten"
        for value in state.get("events", [])
        if isinstance(value, dict)
    )
    events.append({
        "type": "DraftWritten",
        "attempt": attempt,
        "deterministic": True,
    })
    return {
        "final": "",
        "draft": json.dumps(report, ensure_ascii=False),
        "report": report,
        "verdict": "ABSTAIN",
        "defect_type": None,
        "confidence": "low",
        "abstain_reason": "validation_failure",
        "approval": {"required": False, "decision": None, "note": ""},
        "hitl_rejections": max(1, int(state.get("hitl_rejections", 0))),
        "events": events,
    }


@traceable(name="supervisor_v1_reject_fallback")
async def _reject_v1_node(state: AgentState, deps: SupervisorDeps) -> dict:
    del deps
    return _rejected_v1_updates(state)


def _after_v1_hitl(state: AgentState) -> str:
    approval = state.get("approval") or {}
    return "reject_fallback" if approval.get("decision") == "reject" else "finalize"


def build_v1_graph(deps: SupervisorDeps, checkpointer=None):
    """Before 기준선에도 동일한 SQLite interrupt/resume 규약을 적용한다."""
    validate_hitl_mode(getattr(deps, "hitl_mode", "interactive"))
    validate_baseline_mode(os.environ.get("BASELINE_MODE", "B0"))
    graph = StateGraph(AgentState)
    graph.add_node("baseline", partial(_baseline_node, deps=deps))
    graph.add_node("hitl", partial(hitl_node, deps=deps))
    graph.add_node("reject_fallback", partial(_reject_v1_node, deps=deps))
    graph.add_node("finalize", partial(_finalize_v1_node, deps=deps))
    graph.add_edge(START, "baseline")
    graph.add_conditional_edges(
        "baseline", _after_baseline, {"hitl": "hitl", "done": END}
    )
    graph.add_conditional_edges(
        "hitl",
        _after_v1_hitl,
        {"reject_fallback": "reject_fallback", "finalize": "finalize"},
    )
    # Before에는 보고서 작성 LLM이 없다. 반려 시 원본을 완료하지
    # 않고 결정론적 ABSTAIN 초안으로 바꿔 새 승인을 받는다.
    graph.add_edge("reject_fallback", "hitl")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


@traceable(name="supervisor_v1_baseline")
async def run_v1(
    deps: SupervisorDeps,
    message: str,
    thread_id: str,
    history: list[dict],
    image_path: str | None = None,
    item: str | None = None,
) -> dict:
    """평가용 직접 실행. interactive ABSTAIN은 최종화하지 않는다.

    서버의 interactive 경로는 ``build_v1_graph``의 interrupt/checkpointer를
    사용한다. 이 함수를 직접 쓰는 평가는 auto 모드만 완료한다.
    """
    hitl_mode = validate_hitl_mode(
        getattr(deps, "hitl_mode", "interactive")
    )
    result = await _prepare_v1(
        deps,
        message,
        thread_id,
        history,
        image_path=image_path,
        item=item,
    )
    if result.get("route") != "inspection":
        return result

    required = _report_requires_hitl(result.get("report") or {})
    if required and hitl_mode == "interactive":
        result["approval"] = {"required": True, "decision": None, "note": ""}
        result["events"] = [
            *result.get("events", []),
            {
                "type": "ApprovalRequired",
                "required": True,
                "report": result.get("report") or {},
                "pipeline": "baseline-v1",
            },
        ]
        result["paused"] = True
        return result

    decision = None
    if required:
        decision = "reject" if hitl_mode == "auto_reject" else "approve"
        result["events"] = [
            *result.get("events", []),
            {
                "type": "ApprovalRequired",
                "required": True,
                "report": result.get("report") or {},
                "pipeline": "baseline-v1",
            },
            {
                "type": "ApprovalResolved",
                "decision": decision,
                "pipeline": "baseline-v1",
            },
        ]
        if decision == "reject":
            result.update(_rejected_v1_updates(result))
            result["approval"] = {
                "required": True,
                "decision": "approve",
                "note": "",
            }
            result["events"] = [
                *result.get("events", []),
                {
                    "type": "ApprovalRequired",
                    "required": True,
                    "report": result.get("report") or {},
                    "pipeline": "baseline-v1",
                },
                {
                    "type": "ApprovalResolved",
                    "decision": "approve",
                    "pipeline": "baseline-v1",
                },
            ]
            decision = "approve"
    result["approval"] = {
        "required": required,
        "decision": decision,
        "note": "",
    }
    finalized = await _finalize_v1_node(result, deps)
    result.update(finalized)
    return result


__all__ = ["build_v1_graph", "run_v1", "validate_baseline_mode"]
