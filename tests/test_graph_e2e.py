"""fake 의존성으로 검사 그래프 전체 계약을 검증한다."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.deps import SupervisorDeps
from agent.fake_llm import FastLLM, QualityLLM
from agent.graph import build_graph
from agent.rag.embedder import FakeEmbedder
from agent.rag.index import SpecIndex


class GraphMCP:
    """외부 서버 없이 MCP 공개 계약만 대신하는 스텁."""

    async def find_prompt(self, name: str) -> str:
        assert name == "inspection_prompt"
        return "vision"

    async def get_prompt(
        self, name: str, arguments: dict, *, server: str
    ) -> str:
        assert name == "inspection_prompt"
        assert arguments["item"] == "screw"
        assert server == "vision"
        return "전체와 사분면을 관찰하고 기준서와 대조한다."

    async def find_tool(self, name: str) -> str:
        assert name == "inspect_image"
        return "vision"

    async def call_tool(
        self, name: str, arguments: dict, *, server: str
    ) -> dict:
        assert name == "inspect_image"
        assert server == "vision"
        defective = "/scratch_neck/" in str(arguments["image_path"])
        if defective:
            text = "목 부위에 선형 긁힘이 보입니다."
            # cross는 반전 좌표를 줄 때 tr, observer는 원본 tl을 준다.
            regions = ["tr"] if arguments["flip"] else ["tl"]
            features = ["선형 긁힘"]
        else:
            text = "표면과 윤곽에서 눈에 띄는 이상이 보이지 않습니다."
            regions = []
            features = ["정상 윤곽"]
        return {
            "text": text,
            "anomaly_seen": defective,
            "regions": regions,
            "features": features,
            "model": "fake-vlm",
            "tokens": 1,
            "usd": 0.0,
            "region": arguments["region"],
            "flip": arguments["flip"],
        }

    async def find_resource_template(self, name: str) -> tuple[str, str]:
        assert name == "get_guide"
        return "vision", "guide://{defect_type}"

    async def read_resource(self, uri: str, *, server: str) -> str:
        assert uri == "guide://scratch_neck"
        assert server == "vision"
        return "# scratch_neck 조치 가이드\n\n조치 옵션: 재작업"


def _deps(specs_dir: Path) -> SupervisorDeps:
    return SupervisorDeps(
        fast_llm=FastLLM("fake-fast", 0.0),
        quality_llm=QualityLLM("fake-quality", 0.0),
        mcp=GraphMCP(),
        session_store=None,
        version="v2",
        vector_store=SpecIndex.from_dir(specs_dir, FakeEmbedder()),
        hitl_mode="auto_approve",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("label", "expected_verdict", "expected_type"),
    [
        ("good", "PASS", None),
        ("scratch_neck", "REJECT", "scratch_neck"),
    ],
)
async def test_graph_finishes_good_and_defect_inspections(
    synthetic_mvtec_paths: dict[str, Path],
    minimal_specs_dir: Path,
    label: str,
    expected_verdict: str,
    expected_type: str | None,
) -> None:
    state = await build_graph(_deps(minimal_specs_dir)).ainvoke({
        "query": "검사해줘",
        "image_path": str(synthetic_mvtec_paths[label]),
        "item": "screw",
        "events": [],
        "observations": [],
        "cross_check": {},
        "cost_usd": 0.0,
        "tokens": 0,
        "validation_attempts": 0,
        "replan_rounds": 0,
        "retry_count": 0,
        "parallel_elapsed": 0.0,
        "ablation": "",
    })

    assert state["report"]["verdict"] == expected_verdict
    assert state["report"]["defect_type"] == expected_type
    assert f"**{expected_verdict}**" in state["final"]
    assert state["validation_attempts"] == 2
    assert state["validation_passed"] is True
    assert state["approval"]["required"] is False
    assert [event["type"] for event in state["events"]] == [
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

