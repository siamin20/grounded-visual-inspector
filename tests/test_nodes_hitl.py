"""HITL 트리거 조건과 세 실행 모드의 공개 계약 테스트."""

from copy import deepcopy
from functools import partial
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command

from agent.nodes.hitl import hitl_node
from agent.state import AgentState


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("verdict", "actions", "confidence", "required"),
    [
        ("REJECT", ["폐기"], "high", True),
        ("REJECT", ["라인정지"], "high", True),
        ("REJECT", ["재작업"], "high", False),
        ("PASS", ["폐기"], "high", False),
        ("PASS", ["통과"], "low", True),
        ("ABSTAIN", ["재검사"], "high", True),
    ],
)
async def test_hitl_required_truth_table(
    inspection_state: dict,
    verdict: str,
    actions: list[str],
    confidence: str,
    required: bool,
) -> None:
    state = deepcopy(inspection_state)
    state["report"].update(
        verdict=verdict,
        actions=actions,
        confidence=confidence,
    )

    result = await hitl_node(state, SimpleNamespace(hitl_mode="auto_approve"))

    assert result["approval"]["required"] is required
    if required:
        assert result["approval"]["decision"] == "approve"
        assert [event["type"] for event in result["events"]] == [
            "ApprovalRequired",
            "ApprovalResolved",
        ]
    else:
        assert result["approval"]["decision"] is None
        assert "events" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("previous_decision", ["approve", "reject"])
async def test_rewritten_safe_draft_still_requires_fresh_approval(
    inspection_state: dict,
    previous_decision: str,
) -> None:
    state = deepcopy(inspection_state)
    state["report"].update(
        verdict="REJECT",
        actions=["재작업"],
        confidence="high",
    )
    state["approval"] = {
        "required": True,
        "decision": previous_decision,
        "note": "previous draft",
    }
    state["reapproval_required"] = True

    result = await hitl_node(state, SimpleNamespace(hitl_mode="auto_approve"))

    assert result["approval"] == {
        "required": True,
        "decision": "approve",
        "note": "",
    }
    assert result["reapproval_required"] is False
    assert [value["type"] for value in result["events"][-2:]] == [
        "ApprovalRequired",
        "ApprovalResolved",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "decision"),
    [("auto_approve", "approve"), ("auto_reject", "reject")],
)
async def test_hitl_automatic_modes_resolve_without_interrupt(
    inspection_state: dict,
    mode: str,
    decision: str,
) -> None:
    state = deepcopy(inspection_state)
    state["report"]["actions"] = ["폐기"]

    result = await hitl_node(state, SimpleNamespace(hitl_mode=mode))

    assert result["approval"]["decision"] == decision
    assert result["events"][-1] == {
        "type": "ApprovalResolved",
        "decision": decision,
    }
    if mode == "auto_reject":
        assert result["hitl_rejections"] == 1
        assert "사람이 반려" in result["validation_feedback"]
        state.update(result)
        rewritten = await hitl_node(state, SimpleNamespace(hitl_mode=mode))
        assert rewritten["approval"]["decision"] == "approve"
        assert [event["type"] for event in rewritten["events"][-2:]] == [
            "ApprovalRequired",
            "ApprovalResolved",
        ]
    else:
        assert "hitl_rejections" not in result


@pytest.mark.asyncio
async def test_hitl_interactive_mode_interrupts_and_resumes(
    inspection_state: dict,
) -> None:
    state = deepcopy(inspection_state)
    state["report"]["actions"] = ["폐기"]
    builder = StateGraph(AgentState)
    builder.add_node(
        "hitl",
        partial(hitl_node, deps=SimpleNamespace(hitl_mode="interactive")),
    )
    builder.add_edge(START, "hitl")
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": f"hitl-{uuid4()}"}}

    paused = await graph.ainvoke(state, config=config)

    assert len(paused["__interrupt__"]) == 1
    assert paused["__interrupt__"][0].value == {
        "type": "ApprovalRequired",
        "report": state["report"],
    }

    resumed = await graph.ainvoke(
        Command(resume={"decision": "approve", "note": "검토 완료"}),
        config=config,
    )
    assert resumed["approval"] == {
        "required": True,
        "decision": "approve",
        "note": "검토 완료",
    }
    assert [event["type"] for event in resumed["events"]] == [
        "ApprovalResolved"
    ]
