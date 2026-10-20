"""LLM을 쓰지 않는 Phase 3 규칙 노드 테스트."""

from types import SimpleNamespace

import pytest

from agent.fake_llm import FastLLM
from agent.nodes.analysis import analysis_node
from agent.nodes.hitl import hitl_node
from agent.nodes.route import route_node
from agent.nodes.validate import validate_node


class ClauseStore:
    def has(self, clause_id: str, item: str) -> bool:
        return clause_id == "SCREW-04" and item == "screw"


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
    hitl = await hitl_node(state, SimpleNamespace(hitl_mode="interactive"))
    assert hitl["approval"]["required"] is True  # low 확신도로 HITL

    result = await validate_node(
        state, SimpleNamespace(vector_store=ClauseStore(), ablation="")
    )
    assert result["validation_passed"] is False
    assert any("코드 판정과 보고서" in error for error in result["validation_errors"])
    assert any("actions" in error for error in result["validation_errors"])
