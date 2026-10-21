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
from agent.nodes.report_contract import deterministic_summary
from agent.nodes.route import route_node
from agent.nodes.validate import validate_node
from agent.nodes.write import (
    _enforce_decision_contract,
    _parse_report,
    _split_report_observations,
    write_node,
)
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
            "abstain_reason": None,
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


class InvalidActionLLM:
    async def write_report(self, context, _attempt, _feedback, on_token=None):
        observation = context["observations"][0]
        report = {
            "verdict": "REJECT",
            "defect_type": context["defect_type"],
            "confidence": "high",
            "abstain_reason": None,
            "observations": [{
                "text": observation["text"],
                "source_call_id": observation["call_id"],
            }],
            "spec_clauses": list(context["judgement"]["violations"]),
            "actions": ["재작업"],
            "lot": {"lot_id": "FORGED"},
            "summary": "관찰에 없는 사실",
        }
        draft = json.dumps(report, ensure_ascii=False)
        if on_token is not None:
            on_token(draft)
        return draft, {"usd": 0.0, "prompt": "test"}


class StaticGuideMCP:
    async def find_resource_template(self, _name):
        return "vision", "guide://{defect_type}"

    async def read_resource(self, _uri, server):
        assert server == "vision"
        return "# broken_large 조치 가이드\n- 조치 옵션: 폐기 | 라인정지\n"


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_report_parser_rejects_nonstandard_json_numbers(constant: str) -> None:
    with pytest.raises(ValueError):
        _parse_report('{"verdict":"PASS","model_score":' + constant + "}")


def test_report_canonicalizer_strips_unknown_keys() -> None:
    raw = {
        "verdict": "PASS",
        "defect_type": None,
        "confidence": "high",
        "abstain_reason": None,
        "observations": [{
            "text": "surface appears normal",
            "source_call_id": "obs-full",
            "score": float("nan"),
        }],
        "spec_clauses": [],
        "actions": ["pass"],
        "lot": {},
        "summary": "normal",
        "model_score": float("inf"),
    }

    canonical = _enforce_decision_contract(
        _split_report_observations(raw),
        {
            "verdict": "PASS",
            "confidence": "high",
            "lot": {},
        },
    )

    assert set(canonical) == {
        "verdict",
        "defect_type",
        "confidence",
        "abstain_reason",
        "observations",
        "spec_clauses",
        "actions",
        "lot",
        "summary",
    }
    assert set(canonical["observations"][0]) == {"text", "source_call_id"}


def _valid_state() -> dict:
    text = "목 부위에 선형 긁힘이 보입니다."
    return {
        "item": "screw",
        "observations": [{
            "call_id": "obs-full",
            "region": "full",
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
        "abstain_reason": None,
        "report": {
            "verdict": "REJECT",
            "defect_type": "scratch_neck",
            "confidence": "high",
            "abstain_reason": None,
            "observations": [{"text": text, "source_call_id": "obs-full"}],
            "spec_clauses": ["SCREW-04"],
            "actions": ["재작업"],
            "lot": {},
            "summary": (
                "관찰 근거(목 부위에 선형 긁힘이 보입니다.)와 위반 조항(SCREW-04)"
                "을 바탕으로 scratch_neck 결함을 REJECT로 판정하고 "
                "재작업 조치를 요청합니다."
            ),
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
    state["ablation"] = "no_cross"
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
    abstained = await validate_node(state, deps)
    assert abstained["validation_passed"] is True
    assert abstained["report"]["verdict"] == "ABSTAIN"
    assert abstained["report"]["actions"] == ["재검사"]
    assert abstained["approval"]["required"] is True
    assert abstained["validation_feedback"] is None


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
    assert [event["type"] for event in second["events"][-2:]] == [
        "ApprovalRequired",
        "ApprovalResolved",
    ]


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
async def test_write_keeps_one_source_id_per_observation(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "agent.nodes.write.get_stream_writer", lambda: lambda _event: None
    )
    state = {
        "query": "검사해줘",
        "item": "screw",
        "lot": {"lot_id": "LOT-001", "meta": {"line": "L1"}},
        "observations": [],
        "cross_check": {},
        "spec_hits": [],
        "judgement": {},
        "verdict": "PASS",
        "defect_type": None,
        "confidence": "high",
        "abstain_reason": None,
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
            "source_call_id": "obs-tl",
        },
        {
            "text": "같은 관찰",
            "source_call_id": "obs-tr",
        },
        {
            "text": "같은 관찰 ",
            "source_call_id": "obs-space",
        },
        {
            "text": "다른 관찰",
            "source_call_id": "obs-full",
        },
    ]
    assert result["report"]["lot"] == {
        "lot_id": "LOT-001",
        "meta": {"line": "L1"},
    }
    assert result["report"]["summary"] == (
        "관찰 근거(같은 관찰 / 다른 관찰)를 바탕으로 PASS로 판정하고 "
        "통과를 요청합니다."
    )
    assert result["draft"] == json.dumps(result["report"], ensure_ascii=False)


@pytest.mark.asyncio
async def test_write_freezes_only_the_first_pre_validation_report(
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
        "abstain_reason": None,
        "events": [],
        "validation_attempts": 0,
        "cost_usd": 0.0,
        "tokens": 0,
        "pre_validation_snapshot": {},
    }

    first = await write_node(
        state,
        SimpleNamespace(quality_llm=DuplicateObservationLLM()),
    )
    snapshot = first["pre_validation_snapshot"]
    assert snapshot["stage"] == "before_validate"
    assert snapshot["report"] == first["report"]
    assert len(snapshot["report_sha256"]) == 64

    # 보고서 객체를 바꿔도 불변 스냅샷은 따라가지 않는다.
    first["report"]["summary"] = "재작성된 요약"
    assert snapshot["report"]["summary"] != first["report"]["summary"]

    rewritten_state = {**state, **first, "pre_validation_snapshot": snapshot}
    second = await write_node(
        rewritten_state,
        SimpleNamespace(quality_llm=DuplicateObservationLLM()),
    )
    assert "pre_validation_snapshot" not in second


@pytest.mark.asyncio
async def test_write_uses_conservative_guide_action_and_canonical_draft(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "agent.nodes.write.get_stream_writer", lambda: lambda _event: None
    )
    state = _valid_state()
    state.update(
        item="bottle",
        lot={"lot_id": "LOT-BOTTLE-001"},
        defect_type="broken_large",
        judgement={
            "violations": ["BOTTLE-03"],
            "defect_type": "broken_large",
        },
    )

    result = await write_node(
        state,
        SimpleNamespace(quality_llm=InvalidActionLLM(), mcp=StaticGuideMCP()),
    )

    assert result["report"]["actions"] == ["라인정지"]
    assert result["report"]["lot"] == {"lot_id": "LOT-BOTTLE-001"}
    assert result["report"]["summary"] == deterministic_summary(result["report"])
    assert result["draft"] == json.dumps(result["report"], ensure_ascii=False)


@pytest.mark.asyncio
async def test_write_enforces_abstain_and_clears_previous_approval(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "agent.nodes.write.get_stream_writer", lambda: lambda _event: None
    )
    state = _valid_state()
    state.update(
        verdict="ABSTAIN",
        defect_type=None,
        confidence="low",
        abstain_reason="insufficient_visual_evidence",
        approval={"required": True, "decision": "approve", "note": "이전 초안"},
    )
    result = await write_node(
        state,
        SimpleNamespace(quality_llm=DuplicateObservationLLM()),
    )

    assert result["report"]["verdict"] == "ABSTAIN"
    assert result["report"]["defect_type"] is None
    assert result["report"]["confidence"] == "low"
    assert result["report"]["abstain_reason"] == "insufficient_visual_evidence"
    assert result["report"]["spec_clauses"] == []
    assert result["report"]["actions"] == ["재검사"]
    assert result["approval"] == {
        "required": False,
        "decision": None,
        "note": "",
    }
    assert result["reapproval_required"] is True


@pytest.mark.asyncio
async def test_finalize_labels_abstain_as_pending_reinspection() -> None:
    state = _valid_state()
    state["report"].update(
        verdict="ABSTAIN",
        defect_type=None,
        confidence="low",
        abstain_reason="validation_failure",
        spec_clauses=[],
        actions=["재검사"],
    )

    result = await finalize_node(state, None)

    assert "판정 보류 — 재검사 필요 (ABSTAIN)" in result["final"]
    assert "보류 사유: validation_failure" in result["final"]


@pytest.mark.asyncio
async def test_validate_and_finalize_keep_same_text_sources_separate() -> None:
    state = _valid_state()
    source_text = state["observations"][0]["text"]
    state["observations"] = [
        {
            **state["observations"][0],
            "call_id": source_id,
            "region": region,
        }
        for source_id, region in (("obs-tl", "tl"), ("obs-tr", "tr"))
    ]
    state["report"]["observations"] = [
        {
            "text": source_text,
            "source_call_id": "obs-tl",
        },
        {
            "text": source_text,
            "source_call_id": "obs-tr",
        },
    ]
    state["report"]["summary"] = (
        "관찰 근거(목 부위에 선형 긁힘이 보입니다.)와 위반 조항(SCREW-04)"
        "을 바탕으로 scratch_neck 결함을 REJECT로 판정하고 "
        "재작업 조치를 요청합니다."
    )
    validated = await validate_node(
        state, SimpleNamespace(vector_store=ClauseStore(), ablation="")
    )
    assert validated["validation_passed"] is True

    state["report"]["observations"].append(
        {
            "text": state["cross_check"]["summary"],
            "source_call_id": "cc-full",
        }
    )
    finalized = await finalize_node(state, None)
    assert "(`obs-tl`)" in finalized["final"]
    assert "(`obs-tr`)" in finalized["final"]
    assert "(`cc-full`)" in finalized["final"]


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
    # write가 새 초안을 만들면 approval을 초기화하므로 재작성본은 다시 중단한다.
    rejected["approval"] = {"required": False, "decision": None, "note": ""}
    rewrite_config = {"configurable": {"thread_id": f"hitl-{uuid.uuid4()}"}}
    rewritten_pause = await graph.ainvoke(rejected, config=rewrite_config)
    assert len(rewritten_pause["__interrupt__"]) == 1


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
