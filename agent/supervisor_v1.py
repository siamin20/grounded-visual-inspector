"""Before 기준선: VLM 한 번으로 판정하는 선형 supervisor."""

import json
import os
from pathlib import Path

from langsmith import traceable

from .deps import SupervisorDeps
from .nodes.common import load_lot
from .retry import with_retry
from .typemap import load_type_map


ROOT = Path(__file__).resolve().parent.parent


def _render_report(report: dict) -> str:
    observations = "\n".join(
        f"- {value['text']} (`{value['source_call_id']}`)"
        for value in report["observations"]
    ) or "- 관찰 기록 없음"
    return (
        "# 제품 검사 보고서\n\n"
        f"- 판정: **{report['verdict']}**\n"
        f"- 결함 유형: {report['defect_type'] or '없음'}\n"
        f"- 확신도: {report['confidence']}\n\n"
        f"## 관찰 근거\n{observations}\n\n"
        f"- 조치: {', '.join(report['actions'])}\n"
        f"- 로트: `{json.dumps(report['lot'], ensure_ascii=False)}`\n\n"
        f"{report['summary']}"
    )


@traceable(name="supervisor_v1_baseline")
async def run_v1(
    deps: SupervisorDeps,
    message: str,
    thread_id: str,
    history: list[dict],
    image_path: str | None = None,
    item: str | None = None,
) -> dict:
    del thread_id
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
    if os.environ.get("BASELINE_MODE", "B0").upper() == "B1":
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
    verdict = result.get("verdict") or (
        "REJECT" if result.get("anomaly_seen") else "PASS"
    )
    defect_type = result.get("defect_type") if verdict == "REJECT" else None
    observations = result.get("observations") or [result.get("text", "판독 불가")]
    report = {
        "verdict": verdict,
        "defect_type": defect_type,
        "confidence": "high",
        "observations": [
            {"text": text, "source_call_id": "baseline-full"}
            for text in observations
        ],
        "spec_clauses": [],
        "actions": ["통과"] if verdict == "PASS" else ["재작업"],
        "lot": load_lot(image_path),
        "summary": "단일 VLM 판독을 Before 기준선 보고서로 정리했습니다.",
    }
    events = [
        {"type": "RouteDecided", "route": "inspection"},
        {"type": "ObservationReady", "region": "full", "call_id": "baseline-full"},
        {"type": "AnalysisDecided", "verdict": verdict, "confidence": "high"},
        {"type": "DraftWritten", "attempt": 1},
        {"type": "Finalize", "pipeline": "baseline-v1"},
    ]
    return {
        "final": _render_report(report),
        "route": "inspection",
        "events": events,
        "report": report,
        "cost_usd": float(result.get("usd", 0.0)),
        "tokens": int(result.get("tokens", 0)),
        "validation_attempts": 0,
        "replan_rounds": 0,
        "gate_removed": 0,
        "retry_count": retries,
    }
