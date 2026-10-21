"""Phase 4 업로드·HITL 재개·SQLite 복원 통합 테스트."""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from PIL import Image
import pytest

import server
from agent.vision.backends import FakeVLM


def _configure_fake(monkeypatch, checkpoint_path: Path, hitl_mode: str) -> None:
    monkeypatch.setenv("SUPERVISOR_VERSION", "v2")
    monkeypatch.setenv("LLM_BACKEND", "fake")
    monkeypatch.setenv("VLM_BACKEND", "fake")
    monkeypatch.setenv("EMBED_BACKEND", "fake")
    monkeypatch.setenv("HITL_MODE", hitl_mode)
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(checkpoint_path))


def _data_lines(response) -> list[str]:
    return [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def _payloads(lines: list[str]) -> list[dict]:
    return [json.loads(line) for line in lines if line not in {"[PAUSED]", "[DONE]"}]


def _approval_id(response) -> str:
    approvals = [
        payload
        for payload in _payloads(_data_lines(response))
        if payload["type"] == "ApprovalRequired"
    ]
    return approvals[-1]["approval_id"]


def _png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (8, 8), (230, 230, 230)).save(output, format="PNG")
    return output.getvalue()


def _jpeg_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (8, 8), (230, 230, 230)).save(output, format="JPEG")
    return output.getvalue()


def test_inspect_validates_and_saves_uuid_png(monkeypatch, tmp_path) -> None:
    _configure_fake(monkeypatch, tmp_path / "inspect-checkpoints.db", "interactive")
    monkeypatch.setattr(server, "UPLOAD_ROOT", tmp_path / "uploads")
    thread_id = f"upload-{uuid4()}"
    png_content = _png_bytes()

    with TestClient(server.app) as client:
        response = client.post(
            "/inspect",
            data={"item": "screw", "thread_id": thread_id},
            files={"file": ("../../unsafe.jpg", png_content, "image/jpeg")},
        )
        saved = list((tmp_path / "uploads" / thread_id).iterdir())
        first_resume = client.post(
            "/chat/resume",
            json={
                "thread_id": thread_id,
                "approval_id": _approval_id(response),
                "decision": "approve",
            },
        )
        resumed = client.post(
            "/chat/resume",
            json={
                "thread_id": thread_id,
                "approval_id": _approval_id(first_resume),
                "decision": "approve",
            },
        )
        upload_history = client.app.state.deps.session_store.load(thread_id)
        invalid = client.post(
            "/inspect",
            data={"item": "screw", "thread_id": "../escape"},
            files={"file": ("input.png", _png_bytes(), "image/png")},
        )
        non_image = client.post(
            "/inspect",
            data={"item": "screw", "thread_id": f"bad-{uuid4()}"},
            files={"file": ("input.png", b"not an image", "image/png")},
        )
        too_large = client.post(
            "/inspect",
            data={"item": "screw", "thread_id": f"large-{uuid4()}"},
            files={"file": ("input.png", b"x" * (server.MAX_UPLOAD_BYTES + 1), "image/png")},
        )

    assert response.status_code == 200
    assert _data_lines(response)[-1] == "[PAUSED]"
    assert len(saved) == 1
    assert saved[0].suffix == ".png"
    UUID(saved[0].stem)
    assert saved[0].read_bytes() == png_content
    assert _data_lines(first_resume)[-1] == "[PAUSED]"
    assert _data_lines(resumed)[-1] == "[DONE]"
    assert upload_history[0] == {
        "role": "user",
        "content": f"검사해줘 [image: {saved[0].name}]",
    }
    assert invalid.status_code == 422
    assert non_image.status_code == 422
    assert too_large.status_code == 413


@pytest.mark.asyncio
async def test_dataset_png_upload_keeps_hash_for_fake_vlm_label(
    monkeypatch,
    tmp_path,
) -> None:
    """Pillow가 PNG로 확인한 업로드는 바이트 해시와 fake 라벨을 유지한다."""
    png_content = _png_bytes()
    source = (
        tmp_path / "data" / "mvtec" / "screw" / "test" / "scratch_neck" / "000.png"
    )
    source.parent.mkdir(parents=True)
    source.write_bytes(png_content)
    dataset_path = tmp_path / "eval" / "dataset.json"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text(
        json.dumps({
            "samples": [{
                "path": str(source.relative_to(tmp_path)),
                "item": "screw",
                "label": "scratch_neck",
            }],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "UPLOAD_ROOT", tmp_path / "uploads")
    upload = server.UploadFile(
        BytesIO(png_content),
        filename="misleading.jpg",
    )

    saved = await server._save_uploaded_png(upload, "hash-fallback")
    result = await FakeVLM(dataset_path=dataset_path).observe(
        str(saved),
        item="screw",
        region="full",
        style="observe",
        flip=False,
        reference_path=None,
    )

    assert hashlib.sha256(saved.read_bytes()).digest() == hashlib.sha256(
        source.read_bytes()
    ).digest()
    assert result["anomaly_seen"] is True
    assert result["text"] != "판독 불가"


@pytest.mark.asyncio
async def test_non_png_upload_is_reencoded_as_png(monkeypatch, tmp_path) -> None:
    jpeg_content = _jpeg_bytes()
    monkeypatch.setattr(server, "UPLOAD_ROOT", tmp_path / "uploads")
    upload = server.UploadFile(
        BytesIO(jpeg_content),
        filename="misleading.png",
    )

    saved = await server._save_uploaded_png(upload, "jpeg-conversion")

    saved_content = saved.read_bytes()
    assert saved_content.startswith(b"\x89PNG\r\n\x1a\n")
    assert saved_content != jpeg_content


def test_resume_survives_server_restart_without_replaying_events(
    monkeypatch, tmp_path, synthetic_mvtec_paths
) -> None:
    _configure_fake(monkeypatch, tmp_path / "restart-checkpoints.db", "interactive")
    thread_id = f"restart-{uuid4()}"

    with TestClient(server.app) as first_client:
        initial = first_client.get(
            "/chat/stream",
            params={
                "message": "검사해줘",
                "thread_id": thread_id,
                "image_path": str(synthetic_mvtec_paths["bottle_broken_small"]),
                "item": "bottle",
            },
        )
        assert first_client.app.state.deps.session_store.load(thread_id) == []

    initial_lines = _data_lines(initial)
    initial_payloads = _payloads(initial_lines)
    assert initial_lines[-1] == "[PAUSED]"
    assert [value["type"] for value in initial_payloads].count("ApprovalRequired") == 1
    assert initial_payloads[-1]["report"]["defect_type"] == "broken_small"

    # 다른 lifespan이 같은 SQLite checkpoint를 열어 재개해야 한다.
    with TestClient(server.app) as second_client:
        first_resume = second_client.post(
            "/chat/resume",
            json={
                "thread_id": thread_id,
                "approval_id": _approval_id(initial),
                "decision": "approve",
                "note": "확인",
            },
        )
        resumed = second_client.post(
            "/chat/resume",
            json={
                "thread_id": thread_id,
                "approval_id": _approval_id(first_resume),
                "decision": "approve",
                "note": "재작성본 확인",
            },
        )
        history = second_client.app.state.deps.session_store.load(thread_id)
        repeated = second_client.get(
            "/chat/stream",
            params={
                "message": "다시 검사해줘",
                "thread_id": thread_id,
                "image_path": str(synthetic_mvtec_paths["bottle_broken_small"]),
                "item": "bottle",
            },
        )
        history_after_repeat = second_client.app.state.deps.session_store.load(
            thread_id
        )

    first_resume_lines = _data_lines(first_resume)
    assert first_resume_lines[-1] == "[PAUSED]"
    assert [
        value["type"] for value in _payloads(first_resume_lines)
    ].count("ApprovalRequired") == 1

    resumed_lines = _data_lines(resumed)
    resumed_payloads = _payloads(resumed_lines)
    resumed_types = [value["type"] for value in resumed_payloads]
    assert resumed_lines[-1] == "[DONE]"
    assert "RouteDecided" not in resumed_types
    assert "ObservationReady" not in resumed_types
    assert resumed_types.count("ApprovalResolved") == 1
    assert resumed_types[-1] == "FinalAnswer"
    assert resumed_payloads[-1]["report"]["defect_type"] == "broken_small"
    assert history[0] == {"role": "user", "content": "검사해줘 [image: 000.png]"}
    assert history[1]["role"] == "assistant"
    # 같은 thread의 새 검사는 이전 approval을 재사용하지 않는다.
    repeated_lines = _data_lines(repeated)
    assert repeated_lines[-1] == "[PAUSED]"
    assert [
        value["type"] for value in _payloads(repeated_lines)
    ].count("ApprovalRequired") == 1
    assert history_after_repeat == history


def test_resume_rejects_missing_or_stale_approval_id(
    monkeypatch, tmp_path, synthetic_mvtec_paths
) -> None:
    """재개는 현재 checkpoint의 승인 ID와 정확히 결합된다."""
    _configure_fake(monkeypatch, tmp_path / "stale-checkpoints.db", "interactive")
    thread_id = f"stale-{uuid4()}"

    with TestClient(server.app) as client:
        initial = client.post(
            "/chat",
            json={
                "message": "검사해줘",
                "thread_id": thread_id,
                "image_path": str(synthetic_mvtec_paths["bottle_broken_small"]),
                "item": "bottle",
            },
        )
        initial_body = initial.json()
        missing = client.post(
            "/chat/resume",
            json={"thread_id": thread_id, "decision": "approve"},
        )
        stale = client.post(
            "/chat/resume",
            json={
                "thread_id": thread_id,
                "approval_id": "stale-approval-id",
                "decision": "approve",
            },
        )
        first_resume = client.post(
            "/chat/resume",
            json={
                "thread_id": thread_id,
                "approval_id": initial_body["approval_id"],
                "decision": "approve",
            },
        )
        next_approval_id = _approval_id(first_resume)
        replayed = client.post(
            "/chat/resume",
            json={
                "thread_id": thread_id,
                "approval_id": initial_body["approval_id"],
                "decision": "approve",
            },
        )
        completed = client.post(
            "/chat/resume",
            json={
                "thread_id": thread_id,
                "approval_id": next_approval_id,
                "decision": "approve",
            },
        )

    assert initial.status_code == 200
    assert initial_body["paused"] is True
    assert isinstance(initial_body["approval_id"], str)
    assert initial_body["approval_id"]
    assert missing.status_code == 422
    assert stale.status_code == 409
    assert _data_lines(first_resume)[-1] == "[PAUSED]"
    assert next_approval_id != initial_body["approval_id"]
    assert replayed.status_code == 409
    assert _data_lines(completed)[-1] == "[DONE]"


def test_auto_approve_finishes_without_pause(
    monkeypatch, tmp_path, synthetic_mvtec_paths
) -> None:
    _configure_fake(monkeypatch, tmp_path / "auto-checkpoints.db", "auto_approve")
    thread_id = f"auto-{uuid4()}"
    with TestClient(server.app) as client:
        response = client.get(
            "/chat/stream",
            params={
                "message": "검사해줘",
                "thread_id": thread_id,
                "image_path": str(synthetic_mvtec_paths["bottle_broken_small"]),
                "item": "bottle",
            },
        )
        health = client.get("/health").json()

    lines = _data_lines(response)
    types = [value["type"] for value in _payloads(lines)]
    assert lines[-1] == "[DONE]"
    assert "[PAUSED]" not in lines
    assert types.count("ApprovalRequired") == 2
    assert types.count("ApprovalResolved") == 2
    assert types[-1] == "FinalAnswer"
    assert health["vlm_backend"] == "fake"
    assert health["embed_backend"] == "fake"
    assert health["hitl_mode"] == "auto_approve"
    assert health["ablation"] == ""


def test_interactive_reject_requires_fresh_approval_for_rewrite(
    monkeypatch, tmp_path, synthetic_mvtec_paths
) -> None:
    _configure_fake(monkeypatch, tmp_path / "reject-checkpoints.db", "interactive")
    thread_id = f"reject-{uuid4()}"
    with TestClient(server.app) as client:
        initial = client.get(
            "/chat/stream",
            params={
                "message": "검사해줘",
                "thread_id": thread_id,
                "image_path": str(synthetic_mvtec_paths["bottle_broken_small"]),
                "item": "bottle",
            },
        )
        rejected = client.post(
            "/chat/resume",
            json={
                "thread_id": thread_id,
                "approval_id": _approval_id(initial),
                "decision": "reject",
                "note": "조치를 다시 작성해 주세요.",
            },
        )
        resumed = client.post(
            "/chat/resume",
            json={
                "thread_id": thread_id,
                "approval_id": _approval_id(rejected),
                "decision": "approve",
                "note": "재작성본 승인",
            },
        )

    assert _data_lines(initial)[-1] == "[PAUSED]"
    rejected_lines = _data_lines(rejected)
    rejected_types = [value["type"] for value in _payloads(rejected_lines)]
    assert rejected_lines[-1] == "[PAUSED]"
    assert rejected_types.count("ApprovalRequired") == 1
    assert rejected_types.count("ApprovalResolved") == 1
    assert "DraftWritten" in rejected_types
    resumed_lines = _data_lines(resumed)
    resumed_types = [value["type"] for value in _payloads(resumed_lines)]
    assert resumed_lines[-1] == "[DONE]"
    assert "[PAUSED]" not in resumed_lines
    assert resumed_types.count("ApprovalRequired") == 0
    assert resumed_types.count("ApprovalResolved") == 1
    assert resumed_types[-1] == "FinalAnswer"
