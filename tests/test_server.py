"""비동기 ASGI 경계에서 서버의 검사·SSE·HITL 계약을 검증한다."""

from __future__ import annotations

import asyncio
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
from PIL import Image
import pytest

import server


ROOT = Path(__file__).resolve().parents[1]


def _configure_fake(monkeypatch, checkpoint_path: Path, hitl_mode: str) -> None:
    monkeypatch.setenv("SUPERVISOR_VERSION", "v2")
    monkeypatch.setenv("LLM_BACKEND", "fake")
    monkeypatch.setenv("VLM_BACKEND", "fake")
    monkeypatch.setenv("EMBED_BACKEND", "fake")
    monkeypatch.setenv("HITL_MODE", hitl_mode)
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(checkpoint_path))


def _sse_lines(response: httpx.Response) -> list[str]:
    return [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def _sse_payloads(lines: list[str]) -> list[dict]:
    return [
        json.loads(line)
        for line in lines
        if line not in {"[PAUSED]", "[DONE]"}
    ]


def _approval_id(response) -> str:
    approvals = [
        payload
        for payload in _sse_payloads(_sse_lines(response))
        if payload["type"] == "ApprovalRequired"
    ]
    return approvals[-1]["approval_id"]


def _unique_png() -> bytes:
    """평가 데이터 해시와 겹치지 않는 작은 업로드 이미지를 메모리에서 만든다."""
    output = BytesIO()
    image = Image.new("RGB", (17, 19), (23, 71, 149))
    image.putpixel((3, 5), (251, 17, 89))
    image.save(output, format="PNG")
    return output.getvalue()


@pytest.mark.asyncio
async def test_async_server_good_and_auto_reject_contract(
    monkeypatch,
    tmp_path,
    synthetic_mvtec_paths,
) -> None:
    """자동 반려 뒤 변경 보고서는 같은 앱 수명에서 새 승인을 기다린다."""
    _configure_fake(
        monkeypatch,
        tmp_path / "auto-reject-checkpoints.db",
        "auto_reject",
    )
    transport = httpx.ASGITransport(app=server.app)

    async with server.lifespan(server.app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            good = await client.post(
                "/chat",
                json={
                    "message": "검사해줘",
                    "thread_id": f"async-good-{uuid4()}",
                    "image_path": str(synthetic_mvtec_paths["good"]),
                    "item": "screw",
                },
            )
            defect = await client.get(
                "/chat/stream",
                params={
                    "message": "검사해줘",
                    "thread_id": f"async-auto-reject-{uuid4()}",
                    "image_path": str(
                        synthetic_mvtec_paths["bottle_broken_small"]
                    ),
                    "item": "bottle",
                },
            )
            health = await client.get("/health")

    assert good.status_code == 200
    good_body = good.json()
    assert good_body["paused"] is False
    assert good_body["report"]["verdict"] == "PASS"
    assert good_body["report"]["defect_type"] is None
    assert good_body["metadata"]["hitl_mode"] == "auto_reject"
    assert good_body["metadata"]["abstain_reason"] is None

    assert defect.status_code == 200
    assert defect.headers["content-type"].startswith("text/event-stream")
    lines = _sse_lines(defect)
    payloads = _sse_payloads(lines)
    event_types = [payload["type"] for payload in payloads]
    approvals = [
        payload for payload in payloads if payload["type"] == "ApprovalResolved"
    ]
    assert lines[-1] == "[PAUSED]"
    assert event_types.count("ApprovalRequired") == 2
    assert event_types.count("ApprovalResolved") == 1
    assert [approval["decision"] for approval in approvals] == ["reject"]
    assert event_types.index("ApprovalResolved") < event_types.index(
        "DraftWritten",
        event_types.index("ApprovalResolved"),
    )
    assert payloads[-1]["type"] == "ApprovalRequired"
    assert payloads[-1]["report"]["verdict"] == "REJECT"
    assert payloads[-1].get("approval_id")
    assert health.status_code == 200
    assert health.json()["hitl_mode"] == "auto_reject"


@pytest.mark.asyncio
async def test_async_inspect_pause_and_resume_contract(
    monkeypatch,
    tmp_path,
) -> None:
    """업로드 검사는 interactive 중단 뒤 같은 thread에서 승인 재개된다."""
    _configure_fake(
        monkeypatch,
        tmp_path / "interactive-checkpoints.db",
        "interactive",
    )
    monkeypatch.setattr(server, "UPLOAD_ROOT", tmp_path / "uploads")
    thread_id = f"async-upload-{uuid4()}"
    transport = httpx.ASGITransport(app=server.app)

    async with server.lifespan(server.app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            inspected = await client.post(
                "/inspect",
                data={"item": "screw", "thread_id": thread_id},
                files={"file": ("input.png", _unique_png(), "image/png")},
            )
            saved_before_conflict = list(
                (tmp_path / "uploads" / thread_id).iterdir()
            )
            blocked_chat = await client.post(
                "/chat",
                json={"message": "새로 검사", "thread_id": thread_id},
            )
            blocked_stream = await client.get(
                "/chat/stream",
                params={"message": "새로 검사", "thread_id": thread_id},
            )
            blocked_inspect = await client.post(
                "/inspect",
                data={"item": "screw", "thread_id": thread_id},
                files={"file": ("second.png", _unique_png(), "image/png")},
            )
            first_resume = await client.post(
                "/chat/resume",
                json={
                    "thread_id": thread_id,
                    "approval_id": _approval_id(inspected),
                    "decision": "approve",
                    "note": "검토 완료",
                },
            )
            resumed = await client.post(
                "/chat/resume",
                json={
                    "thread_id": thread_id,
                    "approval_id": _approval_id(first_resume),
                    "decision": "approve",
                    "note": "재작성본 검토 완료",
                },
            )

    inspected_lines = _sse_lines(inspected)
    inspected_payloads = _sse_payloads(inspected_lines)
    first_resume_lines = _sse_lines(first_resume)
    first_resume_payloads = _sse_payloads(first_resume_lines)
    resumed_lines = _sse_lines(resumed)
    resumed_payloads = _sse_payloads(resumed_lines)

    assert inspected.status_code == 200
    assert inspected_lines[-1] == "[PAUSED]"
    assert [payload["type"] for payload in inspected_payloads].count(
        "ApprovalRequired"
    ) == 1
    assert blocked_chat.status_code == 409
    assert blocked_stream.status_code == 409
    assert blocked_inspect.status_code == 409
    assert blocked_chat.json()["detail"].startswith("사람 승인")
    assert list((tmp_path / "uploads" / thread_id).iterdir()) == (
        saved_before_conflict
    )
    assert first_resume_lines[-1] == "[PAUSED]"
    assert [payload["type"] for payload in first_resume_payloads].count(
        "ApprovalRequired"
    ) == 1
    assert resumed.status_code == 200
    assert resumed_lines[-1] == "[DONE]"
    assert "[PAUSED]" not in resumed_lines
    assert [payload["type"] for payload in resumed_payloads].count(
        "ApprovalResolved"
    ) == 1
    assert resumed_payloads[-1]["type"] == "FinalAnswer"
    assert resumed_payloads[-1]["report"]["confidence"] == "low"


@pytest.mark.asyncio
@pytest.mark.parametrize("thread_id", ["", "bad:id", ".", "..", "../escape"])
async def test_chat_rejects_invalid_thread_id_before_graph(
    monkeypatch,
    thread_id: str,
) -> None:
    """잘못된 ID로 재개할 수 없는 checkpoint를 만들지 않는다."""
    async def unexpected_handle(*args, **kwargs):
        raise AssertionError("thread_id 검증 전에 그래프를 실행함")

    monkeypatch.setattr(server, "_handle_chat", unexpected_handle)
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/chat",
            json={"message": "검사해줘", "thread_id": thread_id},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "thread_id 형식이 올바르지 않습니다."


@pytest.mark.asyncio
async def test_inspect_requires_explicit_thread_or_legacy_session_id() -> None:
    """ID 없는 업로드를 공용 default checkpoint에 저장하지 않는다."""
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/inspect",
            data={"item": "screw"},
            files={"file": ("input.png", _unique_png(), "image/png")},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "thread_id 또는 session_id가 필요합니다."
    assert server._validated_inspect_thread_id(None, "legacy-session") == (
        "legacy-session"
    )


@pytest.mark.asyncio
async def test_concurrent_new_stream_for_same_thread_is_single_flight(
    monkeypatch,
) -> None:
    """SSE가 끝나기 전 같은 thread의 두 번째 신규 검사를 거부한다."""
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def no_pending(thread_id: str) -> None:
        return None

    async def slow_stream(*args, **kwargs):
        entered.set()
        await finish.wait()
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(server, "_ensure_thread_is_not_paused", no_pending)
    monkeypatch.setattr(server, "_stream_request", slow_stream)
    monkeypatch.setattr(
        server.app.state, "checkpoint_namespace", "v2", raising=False
    )
    monkeypatch.setattr(
        server.app.state, "active_thread_requests", set(), raising=False
    )
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        first_task = asyncio.create_task(client.get(
            "/chat/stream",
            params={"message": "첫 검사", "thread_id": "same-thread"},
        ))
        await asyncio.wait_for(entered.wait(), timeout=1)
        second = await client.get(
            "/chat/stream",
            params={"message": "두 번째 검사", "thread_id": "same-thread"},
        )
        finish.set()
        first = await asyncio.wait_for(first_task, timeout=1)

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json()["detail"] == "같은 thread의 요청이 이미 실행 중입니다."
    assert server.app.state.active_thread_requests == set()


@pytest.mark.asyncio
async def test_concurrent_resume_reuses_approval_id_only_once(monkeypatch) -> None:
    """동일 approval ID의 동시 재개 중 하나만 checkpoint를 갱신한다."""
    entered = asyncio.Event()
    finish = asyncio.Event()
    approval_id = "approval-once"

    async def get_state(config):
        return SimpleNamespace(
            values={"events": [], "query": "검사", "image_path": None},
            tasks=[SimpleNamespace(
                interrupts=(SimpleNamespace(id=approval_id),),
            )],
        )

    async def slow_resume(*args, **kwargs):
        entered.set()
        await finish.wait()
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(
        server.app.state,
        "graph",
        SimpleNamespace(aget_state=get_state),
        raising=False,
    )
    monkeypatch.setattr(server, "_stream_graph", slow_resume)
    monkeypatch.setattr(
        server.app.state, "checkpoint_namespace", "v2", raising=False
    )
    monkeypatch.setattr(
        server.app.state, "active_thread_requests", set(), raising=False
    )
    body = {
        "thread_id": "resume-once",
        "approval_id": approval_id,
        "decision": "approve",
    }
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        first_task = asyncio.create_task(client.post("/chat/resume", json=body))
        await asyncio.wait_for(entered.wait(), timeout=1)
        second = await client.post("/chat/resume", json=body)
        finish.set()
        first = await asyncio.wait_for(first_task, timeout=1)

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json()["detail"] == "같은 thread의 요청이 이미 실행 중입니다."
    assert server.app.state.active_thread_requests == set()


def test_metadata_propagates_abstain_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        server.app.state,
        "deps",
        SimpleNamespace(version="v2", hitl_mode="interactive", ablation=""),
        raising=False,
    )
    monkeypatch.setattr(server.app.state, "llm_backend", "fake", raising=False)
    monkeypatch.setattr(server.app.state, "instance_id", 1, raising=False)
    metadata = server._metadata(
        {
            "route": "inspection",
            "report": {
                "verdict": "ABSTAIN",
                "defect_type": None,
                "confidence": "low",
                "abstain_reason": "insufficient_visual_evidence",
            },
        },
        None,
    )

    assert metadata["verdict"] == "ABSTAIN"
    assert metadata["abstain_reason"] == "insufficient_visual_evidence"


def test_web_ui_renders_abstain_as_neutral_reinspection() -> None:
    html = (ROOT / "web_chat.html").read_text(encoding="utf-8")

    assert ".badge.abstain" in html
    assert "if (verdict === 'ABSTAIN') return 'abstain'" in html
    assert "판정 보류" in html
    assert "재검사 필요" in html
    assert "insufficient_visual_evidence: '시각 근거 부족'" in html
    assert "event.nonblocking" not in html
    assert "ctx.observerFlags" not in html
    assert "ctx.observerRegions" not in html
    assert "crossCard.agree = typeof event.agree === 'boolean'" in html
    assert "Number.isInteger(event.attempt) ? 'draft' : 'agent'" in html
    assert "ctx.streamItem.kind === 'draft'" in html
    assert "approvalId: event.approval_id" in html
    assert "approval_id: card.approvalId" in html
