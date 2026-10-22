"""Phase 3 research 서브그래프의 입·출력 계약 테스트."""

from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace

import pytest

from agent.retry import with_retry as real_with_retry
from agent.subgraphs import cross_checker as cross_checker_module
from agent.subgraphs import observer as observer_module
from agent.subgraphs.research import build_research_subgraph
from agent.subgraphs.spec_judge import (
    _build_evidence_text,
    _validated_judgement,
    spec_judge_node,
)
from agent.vision.contracts import CHECKLIST_KEYS


def _checks(*true_keys: str) -> dict[str, bool]:
    return {key: key in true_keys for key in CHECKLIST_KEYS}


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

        if self.unreadable:
            return {
                "text": "판독 불가",
                "anomaly_seen": None,
                "regions": [],
                "features": [],
                "model": (
                    "fake-cross"
                    if request["style"] == "checklist"
                    else "fake-observer"
                ),
                "tokens": 1,
                "usd": 0.001,
                "region": request["region"],
                "flip": request["flip"],
            }
        if request["style"] == "checklist":
            return {
                "text": "체크리스트에서 긁힘 항목은 예입니다.",
                "anomaly_seen": True,
                "regions": ["tl", "br"],
                "features": ["선형 흔적"],
                "checks": _checks("scratch"),
                "model": "fake-cross",
                "tokens": 1,
                "usd": 0.001,
                "region": "full",
                "flip": True,
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
    assert judgement["status"] == "invalid"
    assert corrected is True


@pytest.mark.parametrize(
    ("hit", "defect_type"),
    [
        ({
            "clause_id": "SCREW-03",
            "item": "bottle",
            "defect_type": "scratch_neck",
        }, "scratch_neck"),
        ({
            "clause_id": "SCREW-03",
            "item": "screw",
            "defect_type": "scratch_neck",
        }, "scratch_head"),
    ],
)
def test_spec_judge_rejects_item_or_clause_type_mismatch(
    hit: dict,
    defect_type: str,
) -> None:
    judgement, corrected = _validated_judgement(
        {
            "violations": ["SCREW-03"],
            "defect_type": defect_type,
            "rationale": "모순 테스트",
        },
        [hit],
        "screw",
    )

    assert judgement["status"] == "invalid"
    assert judgement["defect_type"] is None
    assert corrected is True


def test_spec_judge_rejects_mixed_supported_and_unknown_claim() -> None:
    judgement, corrected = _validated_judgement(
        {
            "violations": ["SCREW-03", "MADE-UP-99"],
            "defect_type": "scratch_neck",
            "rationale": "유효·무효 조항이 섞임",
        },
        [{
            "clause_id": "SCREW-03",
            "item": "screw",
            "defect_type": "scratch_neck",
        }],
        "screw",
    )

    assert judgement["violations"] == ["SCREW-03"]
    assert judgement["defect_type"] is None
    assert judgement["status"] == "invalid"
    assert corrected is True


@pytest.mark.parametrize(
    "raw",
    [
        None,
        [],
        {},
        {"violations": [], "defect_type": None},
        {
            "violations": "SCREW-03",
            "defect_type": "scratch_neck",
            "rationale": "x",
        },
        {"violations": [{}], "defect_type": "scratch_neck", "rationale": "x"},
        {"violations": [], "defect_type": "scratch_neck", "rationale": "x"},
        {"violations": ["SCREW-03"], "defect_type": None, "rationale": "x"},
        {
            "violations": ["SCREW-03"],
            "defect_type": "scratch_head",
            "rationale": "x",
        },
        {
            "violations": ["MADE-UP-99"],
            "defect_type": "scratch_neck",
            "rationale": "x",
        },
        {
            "violations": ["SCREW-03", "SCREW-03"],
            "defect_type": "scratch_neck",
            "rationale": "x",
        },
        {
            "violations": ["SCREW-03", "SCREW-03", "SCREW-03", "SCREW-03"],
            "defect_type": "scratch_neck",
            "rationale": "x",
        },
        {
            "violations": ["SCREW-03"],
            "defect_type": "scratch_neck",
            "rationale": "x" * 241,
        },
    ],
)
def test_spec_judge_malformed_or_contradictory_claim_is_invalid(raw) -> None:
    judgement, corrected = _validated_judgement(
        raw,
        [{
            "clause_id": "SCREW-03",
            "item": "screw",
            "defect_type": "scratch_neck",
        }],
        "screw",
    )

    assert judgement["status"] == "invalid"
    assert corrected is True


def test_spec_judge_accepts_only_explicit_well_formed_no_claim() -> None:
    judgement, corrected = _validated_judgement(
        {
            "violations": [],
            "defect_type": None,
            "rationale": "no matching violation",
        },
        [],
        "screw",
    )

    assert judgement["status"] == "none"
    assert corrected is False


def test_evidence_builder_uses_positive_observations_and_never_state_labels() -> None:
    evidence = _build_evidence_text(
        [
            {
                "region": "full",
                "text": "전체 윤곽은 균일합니다.",
                "anomaly_seen": False,
                "features": ["균일한 윤곽"],
                "label": "scratch_neck",
                "image_path": "/secret/test/scratch_neck/001.png",
            },
            {
                "region": "tr",
                "text": "목 부위에 선형 흔적이 보입니다.",
                "anomaly_seen": True,
                "features": ["선형 흔적", "목 부위", "선형 흔적"],
                "label": "scratch_neck",
                "image_path": "/secret/test/scratch_neck/001.png",
            },
        ],
        {
            "summary": "교차 확인에서 같은 선형 흔적이 보입니다.",
            "anomaly_seen": True,
            "regions": ["tr"],
            "features": ["선형 흔적"],
            "checks": _checks("scratch"),
            "label": "scratch_neck",
            "image_path": "/secret/test/scratch_neck/001.png",
        },
    )

    assert "목 부위에 선형 흔적" in evidence
    assert "교차 확인" in evidence
    assert "전체 윤곽은 균일" not in evidence
    assert evidence.count("선형 흔적") >= 2
    assert "scratch_neck" not in evidence
    assert "/secret/" not in evidence


def test_evidence_builder_excludes_unvalidated_generic_cross_positive() -> None:
    evidence = _build_evidence_text(
        [{
            "region": "full",
            "text": "전체 윤곽에서 구체적 이상이 보이지 않습니다.",
            "anomaly_seen": False,
            "features": ["균일한 윤곽"],
        }],
        {
            "summary": "제품 외관이 보입니다.",
            "anomaly_seen": True,
            "regions": ["tl"],
            "features": ["외관"],
        },
    )

    assert "Cross-Checker" not in evidence
    assert "제품 외관" not in evidence
    assert "전체 윤곽" in evidence


@pytest.mark.asyncio
async def test_cross_only_positive_evidence_reaches_search_and_judge() -> None:
    deps = _deps()
    state = _state()
    state["_observer_result"] = {
        "observations": [{
            "region": "full",
            "text": "전체 윤곽은 균일합니다.",
            "anomaly_seen": False,
            "features": ["균일한 윤곽"],
        }],
        "events": [],
        "done_task_indices": [0],
    }
    state["_cross_result"] = {
        "cross_check": {
            "call_id": "cc-full",
            "anomaly_seen": True,
            "regions": ["tr"],
            "summary": "교차 검사에서 목 부위 선형 흔적이 보입니다.",
            "features": ["선형 흔적", "목 부위"],
            "checks": _checks("scratch"),
            "model": "fake-cross",
        },
        "replace": True,
        "events": [],
        "done_task_indices": [1],
    }

    result = await spec_judge_node(state, deps)

    assert result["rag_query_text"] == (
        "Cross-Checker: 교차 검사에서 목 부위 선형 흔적이 보입니다.\n"
        "Cross-Checker 특징: 선형 흔적, 목 부위"
    )
    assert "전체 윤곽은 균일" not in result["rag_query_text"]
    assert deps.vector_store.search_calls[0]["query"] == result["rag_query_text"]
    assert deps.fast_llm.calls[0]["observations_text"] == result["rag_query_text"]
    assert result["judgement"]["status"] == "grounded"
    assert result["raw_judgement"]["violations"] == ["SCREW-03"]
    assert result["judgement_corrected"] is False


@pytest.mark.asyncio
async def test_research_runs_observer_and_cross_before_spec_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deps = _deps(fail_once_region="tl")
    reference_path = "/synthetic/screw/train/good/000.png"
    monkeypatch.setattr(
        observer_module, "normal_reference_path", lambda _item: reference_path
    )
    monkeypatch.setattr(
        cross_checker_module, "normal_reference_path", lambda _item: reference_path
    )

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
        "features": ["선형 흔적"],
        "checks": _checks("scratch"),
        "model": "fake-cross",
    }
    assert len(result["spec_hits"]) == 3
    assert result["rag_query_text"] == _build_evidence_text(
        result["observations"], result["cross_check"]
    )
    assert result["cross_check"]["summary"] in result["rag_query_text"]
    assert result["judgement"]["violations"] == ["SCREW-03"]
    assert result["judgement"]["defect_type"] == "scratch_neck"
    assert result["judgement"]["status"] == "grounded"
    assert result["raw_judgement"] == {
        "violations": ["SCREW-03"],
        "defect_type": "scratch_neck",
        "rationale": "관찰문의 선형 흔적과 조항이 일치합니다.",
    }
    assert result["judgement_corrected"] is False
    assert result["judgement_status"] == "grounded"
    assert all(task["status"] == "done" for task in result["tasks"])
    assert result["retry_count"] == 3
    assert result["successful_vlm_requests"] == 0
    assert result["observed_model_ids"] == ["fake-observer", "fake-cross"]
    assert result["parallel_elapsed"] >= 0.25
    assert result["cost_usd"] == pytest.approx(0.526)

    calls = deps.mcp.calls
    assert len(calls) == 7  # observer 5 + tl 재시도 1 + cross-check 1
    assert sum(call["region"] == "tl" and call["style"] == "observe"
               for call in calls) == 2
    cross_call = next(call for call in calls if call["style"] == "checklist")
    assert cross_call["flip"] is True
    assert cross_call["model_role"] == "cross"
    assert cross_call["reference_path"] == reference_path
    assert all(
        call["reference_path"] == reference_path
        for call in calls if call["style"] == "observe"
    )
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
    assert cross_event["checks"] == _checks("scratch")


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
        "status": "invalid",
    }
    assert result["raw_judgement"] == {
        "violations": ["SCREW-03", "MADE-UP-99"],
        "defect_type": "invented_type",
        "rationale": "교정 테스트",
    }
    assert result["judgement_corrected"] is True
    judged = next(event for event in result["events"] if event["type"] == "SpecJudged")
    assert judged["corrected"] is True
    assert judged["judgement_status"] == "invalid"
    assert "MADE-UP-99" not in judged["violations"]


@pytest.mark.asyncio
async def test_all_unreadable_observations_skip_search_and_judgement() -> None:
    deps = _deps(unreadable=True)

    result = await build_research_subgraph(deps).ainvoke(_state())

    assert len(result["observations"]) == 5
    assert result["rag_query_text"] == ""
    assert result["spec_hits"] == []
    assert result["judgement"] == {
        "violations": [],
        "defect_type": None,
        "rationale": "관찰 없음",
        "status": "none",
    }
    assert result["judgement_status"] == "none"
    assert result["raw_judgement"] is None
    assert result["judgement_corrected"] is False
    assert deps.vector_store.search_calls == []
    assert deps.fast_llm.calls == []
