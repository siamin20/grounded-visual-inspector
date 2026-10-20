"""LLM을 쓰지 않는 Phase 3 규칙 노드 테스트."""

import json
import uuid
from functools import partial
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command

from agent.fake_llm import FastLLM
from agent.graph import _after_validate
from agent.nodes.analysis import analysis_node
from agent.nodes.finalize import finalize_node
from agent.nodes.hitl import hitl_node
from agent.nodes.route import route_node
from agent.nodes.validate import validate_node
from agent.nodes.write import write_node
from agent.state import AgentState


class ClauseStore:
    def has(self, clause_id: str, item: str) -> bool:
        return clause_id == "SCREW-04" and item == "screw"


class DuplicateObservationLLM:
    async def write_report(self, _context, _attempt, _feedback, on_token=None):
        report = {
            "verdict": "PASS",
            "defect_type": None,
            "confidence": "high",
            "observations": [
                {"text": "같은 관찰", "source_call_id": "obs-tl"},
                {"text": "같은 관찰", "source_call_id": ["obs-tr", "obs-tl"]},
                {"text": "같은 관찰 ", "source_call_id": "obs-space"},
                {"text": "다른 관찰", "source_call_id": "obs-full"},
            ],
            "spec_clauses": [],
            "actions": ["통과"],
            "lot": {},
            "summary": "정상입니다.",
        }
        draft = json.dumps(report, ensure_ascii=False)
        if on_token is not None:
            on_token(draft)
        return draft, {"usd": 0.0, "prompt": "test"}


def _valid_state() -> dict:
    text = "목 부위에 선형 긁힘이 보입니다."
    return {
        "item": "screw",
        "observations": [{
            "call_id": "obs-full",
            "text": text,
            "anomaly_seen": True,
            "regions": ["tl"],
            "features": ["선형 긁힘"],
        }],
        "cross_check": {
            "call_id": "cc-full",
            "anomaly_seen": True,
            "regions": ["tl"],
            "summary": text,
        },
        "judgement": {
            "violations": ["SCREW-04"],
            "defect_type": "scratch_neck",
        },
        "verdict": "REJECT",
        "defect_type": "scratch_neck",
        "confidence": "high",
        "report": {
            "verdict": "REJECT",
            "defect_type": "scratch_neck",
            "confidence": "high",
            "observations": [{"text": text, "source_call_id": "obs-full"}],
            "spec_clauses": ["SCREW-04"],
            "actions": ["재작업"],
            "lot": {},
            "summary": "목 부위 긁힘으로 재작업이 필요합니다.",
        },
        "events": [],
        "validation_attempts": 1,
        "ablation": "",
    }


def test_forced_validation_routes_through_hitl_before_finalize() -> None:
    assert _after_validate({"validation_passed": False}) == "write"
    assert _after_validate({
        "validation_passed": True,
        "approval": {"required": True, "decision": None},
    }) == "hitl"
    assert _after_validate({
        "validation_passed": True,
        "approval": {"required": True, "decision": "approve"},
    }) == "finalize"


@pytest.mark.asyncio
async def test_route_forces_non_image_request_to_chat_and_requires_known_item() -> None:
    deps = SimpleNamespace(fast_llm=FastLLM("fake", 0.0))
    chat = await route_node(
        {"query": "장애 진단해줘", "events": [], "cost_usd": 0.0}, deps
    )
    needs_item = await route_node(
        {"query": "검사", "image_path": "/tmp/input.png", "events": []}, deps
    )
    assert chat["route"] == "chat"
    assert needs_item["route"] == "needs_item"


@pytest.mark.asyncio
async def test_analysis_treats_none_cross_result_as_disagreement() -> None:
    state = _valid_state()
    state["observations"][0]["anomaly_seen"] = False
    state["observations"][0]["regions"] = []
    state["cross_check"]["anomaly_seen"] = None
    state["cross_check"]["regions"] = []
    state["judgement"] = {"violations": [], "defect_type": None}
    result = await analysis_node(state, None)
    assert result["verdict"] == "PASS"
    assert result["confidence"] == "low"

    state["cross_check"] = {}
    result = await analysis_node(state, None)
    assert result["confidence"] == "high"
    assert result["events"][-1]["cross_check_skipped"] is True


@pytest.mark.asyncio
async def test_validate_rejects_embellished_observation_and_forces_after_three() -> None:
    state = _valid_state()
    state["report"]["observations"][0]["text"] += " 우측에 균열도 있습니다."
    deps = SimpleNamespace(vector_store=ClauseStore(), ablation="")

    rejected = await validate_node(state, deps)
    assert rejected["validation_passed"] is False
    assert any("근거 없는 관찰" in error for error in rejected["validation_errors"])

    state["validation_attempts"] = 3
    forced = await validate_node(state, deps)
    assert forced["validation_passed"] is True
    assert forced["report"]["confidence"] == "low"
    assert forced["approval"]["required"] is True
    assert forced["validation_feedback"] is None


@pytest.mark.asyncio
async def test_no_validate_records_would_fail_and_auto_reject_loops_once() -> None:
    state = _valid_state()
    state["report"]["spec_clauses"] = ["MADE-UP-99"]
    deps = SimpleNamespace(vector_store=ClauseStore(), ablation="no_validate")
    result = await validate_node(state, deps)
    assert result["validation_passed"] is True
    assert result["events"][-1]["would_fail"]

    state = _valid_state()
    state["report"]["actions"] = ["폐기"]
    hitl_deps = SimpleNamespace(hitl_mode="auto_reject")
    first = await hitl_node(state, hitl_deps)
    assert first["approval"]["decision"] == "reject"
    assert first["hitl_rejections"] == 1

    state.update(first)
    second = await hitl_node(state, hitl_deps)
    assert second["approval"]["decision"] == "approve"
    assert "hitl_rejections" not in second


@pytest.mark.asyncio
async def test_writer_cannot_override_code_decision_or_crash_hitl() -> None:
    state = _valid_state()
    state["report"] = {
        **state["report"],
        "verdict": "PASS",
        "defect_type": None,
        "confidence": "low",
        "actions": [{"action": "폐기"}],
    }
    hitl = await hitl_node(state, SimpleNamespace(hitl_mode="auto_approve"))
    assert hitl["approval"]["required"] is True  # low 확신도로 HITL

    result = await validate_node(
        state, SimpleNamespace(vector_store=ClauseStore(), ablation="")
    )
    assert result["validation_passed"] is False
    assert any("코드 판정과 보고서" in error for error in result["validation_errors"])
    assert any("actions" in error for error in result["validation_errors"])


@pytest.mark.asyncio
async def test_write_merges_exact_observations_and_preserves_all_call_ids(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "agent.nodes.write.get_stream_writer", lambda: lambda _event: None
    )
    state = {
        "query": "검사해줘",
        "item": "screw",
        "lot": {},
        "observations": [],
        "cross_check": {},
        "spec_hits": [],
        "judgement": {},
        "verdict": "PASS",
        "defect_type": None,
        "confidence": "high",
        "events": [],
        "validation_attempts": 0,
        "cost_usd": 0.0,
    }
    result = await write_node(
        state,
        SimpleNamespace(quality_llm=DuplicateObservationLLM()),
    )

    assert result["report"]["observations"] == [
        {
            "text": "같은 관찰",
            "source_call_id": ["obs-tl", "obs-tr"],
        },
        {
            "text": "같은 관찰 ",
            "source_call_id": ["obs-space"],
        },
        {
            "text": "다른 관찰",
            "source_call_id": ["obs-full"],
        },
    ]


@pytest.mark.asyncio
async def test_validate_and_finalize_accept_merged_and_legacy_source_ids() -> None:
    state = _valid_state()
    state["report"]["observations"] = [
        {
            "text": state["observations"][0]["text"],
            "source_call_id": ["obs-tl", "obs-tr"],
        }
    ]
    validated = await validate_node(
        state, SimpleNamespace(vector_store=ClauseStore(), ablation="")
    )
    assert validated["validation_passed"] is True

    state["report"]["observations"].append(
        {
            "text": state["observations"][0]["text"],
            "source_call_id": "obs-full",
        }
    )
    finalized = await finalize_node(state, None)
    assert "(`obs-tl`, `obs-tr`)" in finalized["final"]
    assert "(`obs-full`)" in finalized["final"]


@pytest.mark.asyncio
async def test_hitl_interrupts_once_and_resume_only_resolves_approval() -> None:
    builder = StateGraph(AgentState)
    builder.add_node(
        "hitl",
        partial(
            hitl_node,
            deps=SimpleNamespace(hitl_mode="interactive"),
        ),
    )
    builder.add_edge(START, "hitl")
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": f"hitl-{uuid.uuid4()}"}}
    state = _valid_state()
    state["report"]["actions"] = ["폐기"]

    paused = await graph.ainvoke(state, config=config)
    assert len(paused["__interrupt__"]) == 1
    assert paused["__interrupt__"][0].value == {
        "type": "ApprovalRequired",
        "report": state["report"],
    }
    assert not any(
        event["type"] == "ApprovalRequired" for event in paused.get("events", [])
    )

    resumed = await graph.ainvoke(
        Command(resume={"decision": "approve", "note": "확인"}),
        config=config,
    )
    assert resumed["approval"] == {
        "required": True,
        "decision": "approve",
        "note": "확인",
    }
    assert [event["type"] for event in resumed["events"]] == ["ApprovalResolved"]

    reject_config = {"configurable": {"thread_id": f"hitl-{uuid.uuid4()}"}}
    rejected_pause = await graph.ainvoke(state, config=reject_config)
    assert len(rejected_pause["__interrupt__"]) == 1
    rejected = await graph.ainvoke(
        Command(resume={"decision": "reject", "note": "조치 변경"}),
        config=reject_config,
    )
    assert rejected["hitl_rejections"] == 1
    after_rewrite = await hitl_node(
        rejected, SimpleNamespace(hitl_mode="interactive")
    )
    assert after_rewrite["approval"]["decision"] == "approve"
    assert "events" not in after_rewrite


@pytest.mark.asyncio
async def test_hitl_omits_unneeded_event_and_deduplicates_reentry() -> None:
    state = _valid_state()
    skipped = await hitl_node(state, SimpleNamespace(hitl_mode="interactive"))
    assert skipped["approval"]["required"] is False
    assert "events" not in skipped

    state["report"]["actions"] = ["폐기"]
    first = await hitl_node(state, SimpleNamespace(hitl_mode="auto_approve"))
    state.update(first)
    second = await hitl_node(state, SimpleNamespace(hitl_mode="auto_approve"))
    assert [event["type"] for event in state["events"]].count("ApprovalRequired") == 1
    assert "events" not in second
