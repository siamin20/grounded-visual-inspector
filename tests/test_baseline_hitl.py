"""Before 기준선의 ABSTAIN 중단·재개 계약."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient
from PIL import Image
import pytest

import server
from agent.deps import SupervisorDeps
from agent.nodes.hitl import hitl_node
from agent.supervisor_v1 import build_v1_graph, run_v1


class _UnreadableMCP:
    def __init__(self, text: str = "판독 불가") -> None:
        self.text = text
        self.calls = 0

    async def find_tool(self, name: str) -> str:
        assert name == "inspect_image"
        return "vision"

    async def call_tool(self, name: str, arguments: dict, *, server: str) -> dict:
        assert name == "inspect_image"
        assert server == "vision"
        self.calls += 1
        return {
            "verdict": "ABSTAIN",
            "defect_type": None,
            "observations": [self.text],
            "anomaly_seen": None,
            "text": self.text,
        }


class _RejectMCP(_UnreadableMCP):
    async def call_tool(self, name: str, arguments: dict, *, server: str) -> dict:
        assert name == "inspect_image"
        assert server == "vision"
        self.calls += 1
        return {
            "verdict": "REJECT",
            "defect_type": "broken_large",
            "observations": ["입구 테두리의 넓은 부분이 끊겨 보입니다."],
            "anomaly_seen": True,
            "text": "입구 테두리의 넓은 부분이 끊겨 보입니다.",
        }


def _data_lines(response) -> list[str]:
    return [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def _payloads(lines: list[str]) -> list[dict]:
    return [
        json.loads(line)
        for line in lines
        if line not in {"[PAUSED]", "[DONE]"}
    ]


def _approval_id(response) -> str:
    approvals = [
        payload
        for payload in _payloads(_data_lines(response))
        if payload["type"] == "ApprovalRequired"
    ]
    return approvals[-1]["approval_id"]


@pytest.mark.asyncio
async def test_direct_v1_interactive_abstain_fails_closed(tmp_path: Path) -> None:
    """checkpointer 없이 직접 호출해도 승인 전 ABSTAIN을 완료하지 않는다."""
    image_path = tmp_path / "unreadable.png"
    Image.new("RGB", (7, 11), (19, 37, 73)).save(image_path, format="PNG")
    result = await run_v1(
        SimpleNamespace(mcp=_UnreadableMCP(), hitl_mode="interactive"),
        "검사해줘",
        "direct-interactive",
        [],
        image_path=str(image_path),
        item="screw",
    )

    assert result["final"] == ""
    assert result["paused"] is True
    assert result["approval"] == {
        "required": True,
        "decision": None,
        "note": "",
    }
    event_types = [value["type"] for value in result["events"]]
    assert event_types[-1] == "ApprovalRequired"
    assert "Finalize" not in event_types


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "이미지에서 결함 여부 확인이 어렵습니다.",
        "표면 상태를 판별할 수 없습니다.",
        "현재 해상도로는 정상 여부 판단이 불가능합니다.",
    ],
)
async def test_direct_v1_unreadable_wording_abstains(
    tmp_path: Path, text: str
) -> None:
    """판독 불가의 표현이 달라도 공통 규칙으로 보류한다."""
    image_path = tmp_path / "wording.png"
    Image.new("RGB", (5, 9), (31, 41, 59)).save(image_path, format="PNG")
    result = await run_v1(
        SimpleNamespace(mcp=_UnreadableMCP(text), hitl_mode="auto_approve"),
        "검사해줘",
        "unreadable-wording",
        [],
        image_path=str(image_path),
        item="screw",
    )

    assert result["verdict"] == "ABSTAIN"
    assert result["abstain_reason"] == "insufficient_visual_evidence"
    assert result["report"]["actions"] == ["재검사"]


@pytest.mark.parametrize(
    ("field", "value", "error_name"),
    [
        ("version", "v3", "SUPERVISOR_VERSION"),
        ("hitl_mode", "manual", "HITL_MODE"),
        ("ablation", "no-crop", "ABLATION"),
    ],
)
def test_supervisor_deps_rejects_invalid_modes(
    field: str, value: str, error_name: str
) -> None:
    values = {
        "fast_llm": None,
        "quality_llm": None,
        "mcp": None,
        "session_store": None,
        "version": "v2",
        "hitl_mode": "interactive",
        "ablation": "",
    }
    values[field] = value
    with pytest.raises(ValueError, match=error_name):
        SupervisorDeps(**values)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("environment", "error_name"),
    [
        ({"SUPERVISOR_VERSION": "v3"}, "SUPERVISOR_VERSION"),
        ({"HITL_MODE": "manual"}, "HITL_MODE"),
        ({"ABLATION": "no-crop"}, "ABLATION"),
        (
            {"SUPERVISOR_VERSION": "v1", "BASELINE_MODE": "B2"},
            "BASELINE_MODE",
        ),
    ],
)
async def test_server_rejects_invalid_modes_before_mcp_start(
    monkeypatch, environment: dict[str, str], error_name: str
) -> None:
    monkeypatch.setenv("SUPERVISOR_VERSION", "v2")
    monkeypatch.setenv("HITL_MODE", "interactive")
    monkeypatch.setenv("BASELINE_MODE", "B0")
    monkeypatch.setenv("ABLATION", "")
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    class _MustNotStart:
        def __init__(self) -> None:
            raise AssertionError("invalid 모드에서 MCP를 시작함")

    monkeypatch.setattr(server, "MCPClient", _MustNotStart)
    with pytest.raises(ValueError, match=error_name):
        async with server.lifespan(server.app):
            raise AssertionError("invalid 모드로 lifespan을 시작함")


@pytest.mark.asyncio
async def test_v1_direct_and_graph_reject_invalid_hitl_mode(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BASELINE_MODE", "B0")
    mcp = _UnreadableMCP()
    deps = SimpleNamespace(mcp=mcp, hitl_mode="manual")
    with pytest.raises(ValueError, match="HITL_MODE"):
        build_v1_graph(deps)
    with pytest.raises(ValueError, match="HITL_MODE"):
        await run_v1(
            deps,
            "검사해줘",
            "invalid-hitl",
            [],
            image_path=str(tmp_path / "not-read.png"),
            item="screw",
        )
    with pytest.raises(ValueError, match="HITL_MODE"):
        await hitl_node(
            {"report": {"verdict": "ABSTAIN", "confidence": "low"}},
            deps,
        )
    assert mcp.calls == 0


@pytest.mark.asyncio
async def test_v1_direct_rejects_invalid_baseline_mode(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BASELINE_MODE", "B2")
    mcp = _UnreadableMCP()
    with pytest.raises(ValueError, match="BASELINE_MODE"):
        await run_v1(
            SimpleNamespace(mcp=mcp, hitl_mode="auto_approve"),
            "검사해줘",
            "invalid-baseline",
            [],
            image_path=str(tmp_path / "not-read.png"),
            item="screw",
        )
    assert mcp.calls == 0


@pytest.mark.asyncio
async def test_v1_missing_action_guide_fails_closed(
    monkeypatch, tmp_path: Path
) -> None:
    """결함은 보였어도 신뢰할 조치 가이드가 없으면 재검사로 보류한다."""
    monkeypatch.setenv("BASELINE_MODE", "B0")
    monkeypatch.setattr(
        "agent.supervisor_v1.load_guide_actions",
        lambda defect_type: [],
    )
    result = await run_v1(
        SimpleNamespace(mcp=_RejectMCP(), hitl_mode="auto_approve"),
        "검사해줘",
        "missing-guide",
        [],
        image_path=str(tmp_path / "bottle" / "test" / "broken_large" / "000.png"),
        item="bottle",
    )

    assert result["verdict"] == "ABSTAIN"
    assert result["defect_type"] is None
    assert result["abstain_reason"] == "validation_failure"
    assert result["report"]["actions"] == ["재검사"]


@pytest.mark.parametrize("hitl_mode", ["interactive", "auto_approve"])
def test_v1_broken_large_guide_action_triggers_hitl(
    monkeypatch, tmp_path: Path, hitl_mode: str
) -> None:
    """Before REJECT도 가이드의 보수 조치를 쓰고 위험 조치는 HITL을 거친다."""
    image_path = (
        tmp_path
        / "mvtec"
        / "bottle"
        / "test"
        / "broken_large"
        / "000.png"
    )
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (19, 23), (101, 67, 43)).save(image_path, format="PNG")
    thread_id = f"v1-broken-large-{hitl_mode}-{uuid4()}"
    monkeypatch.setenv("SUPERVISOR_VERSION", "v1")
    monkeypatch.setenv("BASELINE_MODE", "B0")
    monkeypatch.setenv("LLM_BACKEND", "fake")
    monkeypatch.setenv("VLM_BACKEND", "fake")
    monkeypatch.setenv("EMBED_BACKEND", "fake")
    monkeypatch.setenv("HITL_MODE", hitl_mode)
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / f"{hitl_mode}.db"))

    with TestClient(server.app) as client:
        response = client.get(
            "/chat/stream",
            params={
                "message": "검사해줘",
                "thread_id": thread_id,
                "image_path": str(image_path),
                "item": "bottle",
            },
        )
        resumed = None
        if hitl_mode == "interactive":
            resumed = client.post(
                "/chat/resume",
                json={
                    "thread_id": thread_id,
                    "approval_id": _approval_id(response),
                    "decision": "approve",
                    "note": "위험 조치 확인",
                },
            )

    lines = _data_lines(response)
    payloads = _payloads(lines)
    event_types = [payload["type"] for payload in payloads]
    approval = next(
        payload for payload in payloads if payload["type"] == "ApprovalRequired"
    )
    assert approval["report"]["verdict"] == "REJECT"
    assert approval["report"]["defect_type"] == "broken_large"
    assert approval["report"]["actions"] == ["라인정지"]

    if hitl_mode == "interactive":
        assert lines[-1] == "[PAUSED]"
        assert "ApprovalResolved" not in event_types
        assert resumed is not None
        resumed_lines = _data_lines(resumed)
        resumed_payloads = _payloads(resumed_lines)
        assert resumed_lines[-1] == "[DONE]"
        assert resumed_payloads[-1]["report"]["actions"] == ["라인정지"]
    else:
        assert lines[-1] == "[DONE]"
        assert event_types.count("ApprovalRequired") == 1
        assert event_types.count("ApprovalResolved") == 1
        assert payloads[-1]["report"]["actions"] == ["라인정지"]


@pytest.mark.asyncio
async def test_direct_v1_auto_reject_replaces_original_report_without_vlm_recall(
    monkeypatch, tmp_path: Path
) -> None:
    """자동 반려된 위험 판정은 원본 대신 재검사 보류로만 완료한다."""
    monkeypatch.setenv("BASELINE_MODE", "B0")
    mcp = _RejectMCP()
    result = await run_v1(
        SimpleNamespace(mcp=mcp, hitl_mode="auto_reject"),
        "검사해줘",
        "direct-auto-reject",
        [],
        image_path=str(tmp_path / "bottle" / "test" / "broken_large" / "000.png"),
        item="bottle",
    )

    resolved = [
        value["decision"]
        for value in result["events"]
        if value["type"] == "ApprovalResolved"
    ]
    assert mcp.calls == 1
    assert resolved == ["reject", "approve"]
    assert [
        value["type"] for value in result["events"]
    ].count("ApprovalRequired") == 2
    assert result["final"]
    assert result["verdict"] == "ABSTAIN"
    assert result["report"]["verdict"] == "ABSTAIN"
    assert result["report"]["confidence"] == "low"
    assert result["report"]["abstain_reason"] == "validation_failure"
    assert result["report"]["actions"] == ["재검사"]


def test_v1_interactive_reject_requires_fresh_abstain_approval(
    monkeypatch, tmp_path: Path
) -> None:
    """사람 반려는 ABSTAIN 재검사 초안으로 바꾸고 두 번째 승인을 요구한다."""
    image_path = (
        tmp_path / "mvtec" / "bottle" / "test" / "broken_large" / "000.png"
    )
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (29, 31), (109, 79, 53)).save(image_path, format="PNG")
    thread_id = f"v1-interactive-reject-{uuid4()}"
    monkeypatch.setenv("SUPERVISOR_VERSION", "v1")
    monkeypatch.setenv("BASELINE_MODE", "B0")
    monkeypatch.setenv("LLM_BACKEND", "fake")
    monkeypatch.setenv("VLM_BACKEND", "fake")
    monkeypatch.setenv("EMBED_BACKEND", "fake")
    monkeypatch.setenv("HITL_MODE", "interactive")
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "interactive-reject.db"))

    with TestClient(server.app) as client:
        initial = client.get(
            "/chat/stream",
            params={
                "message": "검사해줘",
                "thread_id": thread_id,
                "image_path": str(image_path),
                "item": "bottle",
            },
        )
        rejected = client.post(
            "/chat/resume",
            json={
                "thread_id": thread_id,
                "approval_id": _approval_id(initial),
                "decision": "reject",
                "note": "위험 조치 반려",
            },
        )
        approved = client.post(
            "/chat/resume",
            json={
                "thread_id": thread_id,
                "approval_id": _approval_id(rejected),
                "decision": "approve",
                "note": "재검사 요청 승인",
            },
        )

    assert _data_lines(initial)[-1] == "[PAUSED]"
    rejected_lines = _data_lines(rejected)
    rejected_payloads = _payloads(rejected_lines)
    rejected_types = [value["type"] for value in rejected_payloads]
    second_approval = next(
        value
        for value in rejected_payloads
        if value["type"] == "ApprovalRequired"
    )
    assert rejected_lines[-1] == "[PAUSED]"
    assert rejected_types.count("ApprovalResolved") == 1
    assert rejected_types.count("DraftWritten") == 1
    assert second_approval["report"]["verdict"] == "ABSTAIN"
    assert second_approval["report"]["abstain_reason"] == "validation_failure"
    assert second_approval["report"]["actions"] == ["재검사"]

    approved_lines = _data_lines(approved)
    approved_payloads = _payloads(approved_lines)
    assert approved_lines[-1] == "[DONE]"
    assert approved_payloads[-1]["report"]["verdict"] == "ABSTAIN"
    assert approved_payloads[-1]["report"]["actions"] == ["재검사"]


def test_v1_auto_reject_rewrites_then_auto_approves_fresh_abstain(
    monkeypatch, tmp_path: Path
) -> None:
    """auto_reject도 반려된 REJECT를 강제 완료하지 않는다."""
    image_path = (
        tmp_path / "mvtec" / "bottle" / "test" / "broken_large" / "000.png"
    )
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (37, 41), (113, 83, 61)).save(image_path, format="PNG")
    thread_id = f"v1-auto-reject-{uuid4()}"
    monkeypatch.setenv("SUPERVISOR_VERSION", "v1")
    monkeypatch.setenv("BASELINE_MODE", "B0")
    monkeypatch.setenv("LLM_BACKEND", "fake")
    monkeypatch.setenv("VLM_BACKEND", "fake")
    monkeypatch.setenv("EMBED_BACKEND", "fake")
    monkeypatch.setenv("HITL_MODE", "auto_reject")
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "auto-reject.db"))

    with TestClient(server.app) as client:
        response = client.get(
            "/chat/stream",
            params={
                "message": "검사해줘",
                "thread_id": thread_id,
                "image_path": str(image_path),
                "item": "bottle",
            },
        )

    lines = _data_lines(response)
    payloads = _payloads(lines)
    event_types = [value["type"] for value in payloads]
    decisions = [
        value["decision"]
        for value in payloads
        if value["type"] == "ApprovalResolved"
    ]
    assert lines[-1] == "[DONE]"
    assert event_types.count("ApprovalRequired") == 2
    assert decisions == ["reject", "approve"]
    assert payloads[-1]["report"]["verdict"] == "ABSTAIN"
    assert payloads[-1]["report"]["abstain_reason"] == "validation_failure"
    assert payloads[-1]["report"]["actions"] == ["재검사"]


def test_v1_abstain_resumes_after_server_restart(monkeypatch, tmp_path: Path) -> None:
    """라벨을 알 수 없는 합성 입력은 최종화 전에 멈추고 재시작 후 재개된다."""
    image_path = tmp_path / "unregistered.png"
    Image.new("RGB", (13, 17), (47, 83, 131)).save(image_path, format="PNG")
    checkpoint_path = tmp_path / "v1-checkpoints.db"
    thread_id = f"v1-abstain-{uuid4()}"
    monkeypatch.setenv("SUPERVISOR_VERSION", "v1")
    monkeypatch.setenv("BASELINE_MODE", "B0")
    monkeypatch.setenv("LLM_BACKEND", "fake")
    monkeypatch.setenv("VLM_BACKEND", "fake")
    # v1은 RAG를 사용하지 않으므로 이 설정의 키를 요구하면 안 된다.
    monkeypatch.setenv("EMBED_BACKEND", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("HITL_MODE", "interactive")
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(checkpoint_path))

    with TestClient(server.app) as first_client:
        initial = first_client.get(
            "/chat/stream",
            params={
                "message": "검사해줘",
                "thread_id": thread_id,
                "image_path": str(image_path),
                "item": "screw",
            },
        )
        assert first_client.app.state.deps.session_store.load(thread_id) == []

    initial_lines = _data_lines(initial)
    initial_payloads = _payloads(initial_lines)
    initial_types = [payload["type"] for payload in initial_payloads]
    assert initial_lines[-1] == "[PAUSED]"
    assert initial_types.count("ApprovalRequired") == 1
    assert initial_types.index("DraftWritten") < initial_types.index(
        "ApprovalRequired"
    )
    approval = next(
        payload
        for payload in initial_payloads
        if payload["type"] == "ApprovalRequired"
    )
    assert approval["report"]["verdict"] == "ABSTAIN"
    assert approval["report"]["actions"] == ["재검사"]

    with TestClient(server.app) as second_client:
        resumed = second_client.post(
            "/chat/resume",
            json={
                "thread_id": thread_id,
                "approval_id": _approval_id(initial),
                "decision": "approve",
                "note": "재검사 요청 확인",
            },
        )
        history = second_client.app.state.deps.session_store.load(thread_id)

    resumed_lines = _data_lines(resumed)
    resumed_payloads = _payloads(resumed_lines)
    resumed_types = [payload["type"] for payload in resumed_payloads]
    assert resumed.status_code == 200
    assert resumed_lines[-1] == "[DONE]"
    assert "[PAUSED]" not in resumed_lines
    assert resumed_types.count("ApprovalResolved") == 1
    assert resumed_payloads[-1]["type"] == "FinalAnswer"
    assert resumed_payloads[-1]["report"]["verdict"] == "ABSTAIN"
    assert len(history) == 2


def test_restart_cannot_resume_across_pipeline_namespaces(
    monkeypatch, tmp_path: Path
) -> None:
    """v1:B0, v1:B1, v2는 같은 DB와 thread ID를 공유해도 재개하지 않는다."""
    image_path = tmp_path / "namespace-unregistered.png"
    Image.new("RGB", (11, 13), (37, 73, 109)).save(image_path, format="PNG")
    checkpoint_path = tmp_path / "pipeline-namespaces.db"
    thread_id = f"pipeline-namespace-{uuid4()}"
    monkeypatch.setenv("LLM_BACKEND", "fake")
    monkeypatch.setenv("VLM_BACKEND", "fake")
    monkeypatch.setenv("EMBED_BACKEND", "fake")
    monkeypatch.setenv("HITL_MODE", "interactive")
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(checkpoint_path))

    monkeypatch.setenv("SUPERVISOR_VERSION", "v1")
    monkeypatch.setenv("BASELINE_MODE", "B0")
    with TestClient(server.app) as b0_client:
        initial = b0_client.get(
            "/chat/stream",
            params={
                "message": "검사해줘",
                "thread_id": thread_id,
                "image_path": str(image_path),
                "item": "screw",
            },
        )
        assert b0_client.app.state.checkpoint_namespace == "v1:B0"
    approval_id = _approval_id(initial)

    monkeypatch.setenv("BASELINE_MODE", "B1")
    with TestClient(server.app) as b1_client:
        cross_b1 = b1_client.post(
            "/chat/resume",
            json={
                "thread_id": thread_id,
                "approval_id": approval_id,
                "decision": "approve",
            },
        )
        assert b1_client.app.state.checkpoint_namespace == "v1:B1"

    monkeypatch.setenv("SUPERVISOR_VERSION", "v2")
    with TestClient(server.app) as v2_client:
        cross_v2 = v2_client.post(
            "/chat/resume",
            json={
                "thread_id": thread_id,
                "approval_id": approval_id,
                "decision": "approve",
            },
        )
        assert v2_client.app.state.checkpoint_namespace == "v2"

    monkeypatch.setenv("SUPERVISOR_VERSION", "v1")
    monkeypatch.setenv("BASELINE_MODE", "B0")
    with TestClient(server.app) as restored_client:
        restored = restored_client.post(
            "/chat/resume",
            json={
                "thread_id": thread_id,
                "approval_id": approval_id,
                "decision": "approve",
            },
        )

    assert _data_lines(initial)[-1] == "[PAUSED]"
    assert cross_b1.status_code == 409
    assert cross_v2.status_code == 409
    assert restored.status_code == 200
    assert _data_lines(restored)[-1] == "[DONE]"
