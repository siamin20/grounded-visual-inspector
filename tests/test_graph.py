"""Phase 3 그래프의 HTTP·SSE 완료 기준 통합 테스트."""

from __future__ import annotations

import json
from pathlib import Path
import uuid

from fastapi.testclient import TestClient

from server import app


ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT / "data" / "mvtec" / "screw" / "test" / "scratch_neck" / "000.png"


def _event_types(events: list[dict]) -> list[str]:
    return [event["type"] for event in events]


def test_fake_post_chat_returns_grounded_report(monkeypatch) -> None:
    monkeypatch.setenv("LLM_BACKEND", "fake")
    monkeypatch.setenv("VLM_BACKEND", "fake")
    monkeypatch.setenv("EMBED_BACKEND", "fake")
    monkeypatch.setenv("HITL_MODE", "interactive")
    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "message": "검사해줘",
                "thread_id": f"phase3-post-{uuid.uuid4()}",
                "image_path": str(IMAGE_PATH),
                "item": "screw",
            },
        )

    assert response.status_code == 200
    body = response.json()
    report = body["report"]
    assert report["verdict"] == "REJECT"
    assert report["defect_type"] == "scratch_neck"
    assert report["confidence"] == "high"
    assert "SCREW-04" in report["spec_clauses"]
    assert report["actions"]
    assert all(
        "hallucinated-observation" not in observation["source_call_id"]
        for observation in report["observations"]
    )
    assert body["metadata"]["validation_attempts"] == 2
    expected_metadata = {
        "verdict": "REJECT",
        "defect_type": "scratch_neck",
        "confidence": "high",
        "approval_required": False,
        "hitl_mode": "interactive",
        "vlm_calls": 6,
        "ablation": "",
    }
    assert {
        key: body["metadata"][key] for key in expected_metadata
    } == expected_metadata
    assert _event_types(body["events"]) == [
        "RouteDecided",
        "PlanCreated",
        "GateApplied",
        "ObservationReady",
        "ObservationReady",
        "ObservationReady",
        "ObservationReady",
        "ObservationReady",
        "CrossCheckDone",
        "SpecJudged",
        "ReplanCheck",
        "AnalysisDecided",
        "DraftWritten",
        "Validation",
        "DraftWritten",
        "Validation",
        "Finalize",
    ]


def test_fake_sse_preserves_event_order_and_format(monkeypatch) -> None:
    monkeypatch.setenv("LLM_BACKEND", "fake")
    monkeypatch.setenv("VLM_BACKEND", "fake")
    monkeypatch.setenv("EMBED_BACKEND", "fake")
    with TestClient(app) as client:
        response = client.get(
            "/chat/stream",
            params={
                "message": "검사해줘",
                "thread_id": f"phase3-sse-{uuid.uuid4()}",
                "image_path": str(IMAGE_PATH),
                "item": "screw",
            },
        )

    assert response.status_code == 200
    data_lines = [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert data_lines[-1] == "[DONE]"
    payloads = [json.loads(line) for line in data_lines[:-1]]
    types = [payload["type"] for payload in payloads]
    progress_types = [
        value for value in types if value not in {"AnswerToken", "FinalAnswer"}
    ]
    assert progress_types == [
        "RouteDecided",
        "PlanCreated",
        "GateApplied",
        "ObservationReady",
        "ObservationReady",
        "ObservationReady",
        "ObservationReady",
        "ObservationReady",
        "CrossCheckDone",
        "SpecJudged",
        "ReplanCheck",
        "AnalysisDecided",
        "DraftWritten",
        "Validation",
        "DraftWritten",
        "Validation",
        "Finalize",
    ]
    assert types[-1] == "FinalAnswer"
    final = next(payload for payload in reversed(payloads) if payload["type"] == "FinalAnswer")
    assert final["report"] is not None
    assert final["reply"].startswith("# 제품 검사 보고서")


def test_v1_baseline_does_not_initialize_embedding_backend(monkeypatch) -> None:
    monkeypatch.setenv("SUPERVISOR_VERSION", "v1")
    monkeypatch.setenv("BASELINE_MODE", "B1")
    monkeypatch.setenv("LLM_BACKEND", "fake")
    monkeypatch.setenv("VLM_BACKEND", "fake")
    monkeypatch.setenv("EMBED_BACKEND", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "message": "검사해줘",
                "thread_id": f"phase3-v1-{uuid.uuid4()}",
                "image_path": str(IMAGE_PATH),
                "item": "screw",
            },
        )
    assert response.status_code == 200
    assert response.json()["report"]["defect_type"] == "scratch_neck"
    assert "Validation" not in _event_types(response.json()["events"])
