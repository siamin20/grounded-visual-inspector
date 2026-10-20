"""Phase 3 research 서브그래프의 입·출력 계약 테스트."""

from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace

import pytest

from agent.retry import with_retry as real_with_retry
from agent.subgraphs import observer as observer_module
from agent.subgraphs.research import build_research_subgraph
from agent.subgraphs.spec_judge import _validated_judgement


class FakeMCP:
    def __init__(self, *, fail_once_region: str | None = None,
                 unreadable: bool = False) -> None:
        self.calls: list[dict] = []
        self.find_calls: list[str] = []
        self.fail_once_region = fail_once_region
        self.unreadable = unreadable
        self._failed = False

    async def find_tool(self, name: str) -> str:
        self.find_calls.append(name)
        return "vision"

    async def call_tool(self, name: str, arguments: dict, *, server: str) -> dict:
        assert name == "inspect_image"
        assert server == "vision"
        request = dict(arguments)
        self.calls.append(request)
        if (
            request["region"] == self.fail_once_region
            and not self._failed
        ):
            self._failed = True
            raise asyncio.TimeoutError("test transient")

        if request["style"] == "checklist":
            return {
                "text": "체크리스트에서 긁힘 항목은 예입니다.",
                "anomaly_seen": True,
                "regions": ["tl", "br"],
                "features": ["선형 흔적"],
                "model": "fake-cross",
                "tokens": 1,
                "usd": 0.001,
                "region": "full",
                "flip": True,
            }
        if self.unreadable:
            return {
                "text": "판독 불가",
                "anomaly_seen": None,
                "regions": [],
                "features": [],
                "model": "fake-observer",
                "tokens": 1,
                "usd": 0.001,
                "region": request["region"],
                "flip": False,
            }
        region = request["region"]
        return {
            "text": f"불량 PASS {region} 영역에 선형 긁힘이 보입니다.",
            "anomaly_seen": True,
            "regions": ["tl" if region == "full" else region],
            "features": ["선형 흔적"],
            "model": "fake-observer",
            "tokens": 1,
            "usd": 0.001,
            "region": region,
            "flip": False,
        }


class FakeSpecIndex:
    def __init__(self) -> None:
        self.search_calls: list[dict] = []

    def search(self, query: str, *, item: str, k: int) -> list[dict]:
        self.search_calls.append({"query": query, "item": item, "k": k})
        return [
            {
                "clause_id": "SCREW-03",
                "item": "screw",
                "defect_type": "scratch_neck",
                "text": "목 부위에 선형 긁힘이 보이면 위반이다.",
                "score": 0.9,
            },
            {
                "clause_id": "SCREW-02",
                "item": "screw",
                "defect_type": "scratch_head",
                "text": "머리 부위 긁힘을 확인한다.",
                "score": 0.8,
            },
            {
                "clause_id": "SCREW-06",
                "item": "screw",
                "defect_type": "thread_top",
                "text": "상단 나사산 연속성을 확인한다.",
                "score": 0.7,
            },
            {
                "clause_id": "SCREW-99",
                "item": "screw",
                "defect_type": "thread_side",
                "text": "k=3 밖의 조항",
                "score": 0.6,
            },
        ]


class FakeJudge:
    def __init__(self, *, invalid: bool = False) -> None:
        self.invalid = invalid
        self.calls: list[dict] = []

    def judge(self, observations_text: str, clauses: list[dict], item: str):
        self.calls.append({
            "observations_text": observations_text,
            "clauses": clauses,
            "item": item,
        })
        if self.invalid:
            result = {
                "violations": ["SCREW-03", "MADE-UP-99"],
                "defect_type": "invented_type",
                "rationale": "교정 테스트",
            }
        else:
            result = {
                "violations": ["SCREW-03"],
                "defect_type": "scratch_neck",
                "rationale": "관찰문의 선형 흔적과 조항이 일치합니다.",
            }
        return result, {"model": "fake-fast", "tokens": 1, "usd": 0.02}


def _deps(*, ablation: str = "", fail_once_region: str | None = None,
          invalid_judgement: bool = False, unreadable: bool = False):
    return SimpleNamespace(
        mcp=FakeMCP(fail_once_region=fail_once_region, unreadable=unreadable),
        vector_store=FakeSpecIndex(),
        fast_llm=FakeJudge(invalid=invalid_judgement),
        ablation=ablation,
    )


def _tasks() -> list[dict]:
    return [
        {"id": "task-observe", "agent": "research", "kind": "observe",
         "status": "pending"},
        {"id": "task-cross", "agent": "research", "kind": "cross_check",
         "status": "pending"},
        {"id": "task-judge", "agent": "research", "kind": "spec_judge",
         "status": "pending"},
    ]


def _state() -> dict:
    return {
        "query": "검사해줘",
        "image_path": "/synthetic/screw/test/scratch_neck/000.png",
        "item": "screw",
        "tasks": _tasks(),
        "observations": [],
        "cross_check": {},
        "events": [{"type": "GateApplied"}],
        "retry_count": 2,
        "parallel_elapsed": 0.25,
        "cost_usd": 0.5,
    }


def test_spec_judge_never_accepts_good_clause_as_violation() -> None:
    judgement, corrected = _validated_judgement(
        {
            "violations": ["SCREW-01"],
            "defect_type": "scratch_neck",
            "rationale": "잘못된 판정",
        },
        [{
            "clause_id": "SCREW-01",
            "item": "screw",
            "defect_type": "good",
            "text": "합격 조건",
            "score": 1.0,
        }],
        "screw",
    )
    assert judgement["violations"] == []
    assert judgement["defect_type"] is None
    assert corrected is True


@pytest.mark.asyncio
async def test_research_runs_observer_and_cross_before_spec_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deps = _deps(fail_once_region="tl")

    async def without_retry_delay(fn, *args, **kwargs):
        return await real_with_retry(fn, *args, base_delay=0, **kwargs)

    monkeypatch.setattr(observer_module, "with_retry", without_retry_delay)
    result = await build_research_subgraph(deps).ainvoke(_state())

    assert [observation["region"] for observation in result["observations"]] == [
        "full", "tl", "tr", "bl", "br",
    ]
    assert all(
        not re.search(r"불량|합격|불합격|\bdefect\b|\breject\b|\bpass\b",
                      observation["text"], flags=re.IGNORECASE)
        for observation in result["observations"]
    )
    assert result["cross_check"] == {
        "call_id": "cc-full",
        "anomaly_seen": True,
        "regions": ["tr", "bl"],
        "summary": "체크리스트에서 긁힘 항목은 예입니다.",
        "model": "fake-cross",
    }
    assert len(result["spec_hits"]) == 3
    assert result["judgement"]["violations"] == ["SCREW-03"]
    assert result["judgement"]["defect_type"] == "scratch_neck"
    assert all(task["status"] == "done" for task in result["tasks"])
    assert result["retry_count"] == 3
    assert result["parallel_elapsed"] >= 0.25
    assert result["cost_usd"] == pytest.approx(0.526)

    calls = deps.mcp.calls
    assert len(calls) == 7  # observer 5 + tl 재시도 1 + cross-check 1
    assert sum(call["region"] == "tl" and call["style"] == "observe"
               for call in calls) == 2
    cross_call = next(call for call in calls if call["style"] == "checklist")
    assert cross_call["flip"] is True
    assert cross_call["model_role"] == "cross"
    assert all(call["flip"] is False for call in calls if call["style"] == "observe")

    event_types = [event["type"] for event in result["events"]]
    assert event_types == [
        "GateApplied",
        "ObservationReady", "ObservationReady", "ObservationReady",
        "ObservationReady", "ObservationReady",
        "CrossCheckDone",
        "SpecJudged",
    ]
    assert all(
        event.get("filtered_words") == ["불량", "pass"]
        for event in result["events"]
        if event["type"] == "ObservationReady"
    )
    observation_events = [
        event for event in result["events"]
        if event["type"] == "ObservationReady"
    ]
    assert all(event["text"] for event in observation_events)
    cross_event = next(
        event for event in result["events"]
        if event["type"] == "CrossCheckDone"
    )
    assert cross_event["summary"] == result["cross_check"]["summary"]


@pytest.mark.asyncio
async def test_no_crops_calls_only_full_observation() -> None:
    deps = _deps(ablation="no_crops")

    result = await build_research_subgraph(deps).ainvoke(_state())

    assert [observation["region"] for observation in result["observations"]] == ["full"]
    assert [call["style"] for call in deps.mcp.calls].count("observe") == 1
    assert result["cross_check"]["call_id"] == "cc-full"


@pytest.mark.asyncio
async def test_no_cross_skips_call_and_returns_empty_cross_check() -> None:
    deps = _deps(ablation="no_cross")

    result = await build_research_subgraph(deps).ainvoke(_state())

    assert len(result["observations"]) == 5
    assert result["cross_check"] == {}
    assert all(call["style"] == "observe" for call in deps.mcp.calls)
    skipped = [event for event in result["events"] if event["type"] == "CrossCheckDone"]
    assert skipped == [{"type": "CrossCheckDone", "skipped": True}]
    assert result["tasks"][1]["status"] == "done"


@pytest.mark.asyncio
async def test_replan_calls_only_pending_center_observation() -> None:
    deps = _deps()
    state = _state()
    state["tasks"] = [
        *[{**task, "status": "done"} for task in _tasks()],
        {"id": "task-observe-center", "agent": "research", "kind": "observe",
         "region": "center", "status": "pending"},
    ]
    state["observations"] = [{
        "call_id": "obs-full",
        "region": "full",
        "text": "전체 윤곽을 관찰했습니다.",
        "anomaly_seen": False,
        "regions": [],
        "features": ["전체 윤곽"],
        "model": "fake-observer",
    }]
    state["cross_check"] = {
        "call_id": "cc-full",
        "anomaly_seen": False,
        "regions": [],
        "summary": "이상이 보이지 않습니다.",
        "model": "fake-cross",
    }

    result = await build_research_subgraph(deps).ainvoke(state)

    assert [(call["region"], call["style"]) for call in deps.mcp.calls] == [
        ("center", "observe")
    ]
    assert [observation["call_id"] for observation in result["observations"]] == [
        "obs-full", "obs-center",
    ]
    assert result["cross_check"] == state["cross_check"]
    assert result["tasks"][-1]["status"] == "done"


@pytest.mark.asyncio
async def test_spec_judge_corrects_ids_and_defect_type_outside_allowed_sets() -> None:
    deps = _deps(invalid_judgement=True)

    result = await build_research_subgraph(deps).ainvoke(_state())

    assert result["judgement"] == {
        "violations": ["SCREW-03"],
        "defect_type": None,
        "rationale": "교정 테스트",
    }
    judged = next(event for event in result["events"] if event["type"] == "SpecJudged")
    assert judged["corrected"] is True
    assert "MADE-UP-99" not in judged["violations"]


@pytest.mark.asyncio
async def test_all_unreadable_observations_skip_search_and_judgement() -> None:
    deps = _deps(unreadable=True)

    result = await build_research_subgraph(deps).ainvoke(_state())

    assert len(result["observations"]) == 5
    assert result["spec_hits"] == []
    assert result["judgement"] == {
        "violations": [],
        "defect_type": None,
        "rationale": "관찰 없음",
    }
    assert deps.vector_store.search_calls == []
    assert deps.fast_llm.calls == []
