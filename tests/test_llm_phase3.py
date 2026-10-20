"""Phase 3 제품 검사용 텍스트 LLM 인터페이스 검증."""

import json
from pathlib import Path

import pytest

from agent.fake_llm import FastLLM, QualityLLM
from agent.ollama_llm import OllamaFastLLM
from agent.openai_llm import OpenAIFastLLM, OpenAIQualityLLM
from agent.rag import FakeEmbedder, SpecIndex
from agent.typemap import load_type_map
from agent.vision.backends import FakeVLM


CLAUSES = [
    {
        "clause_id": "SCREW-03",
        "item": "screw",
        "defect_type": "scratch_head",
        "text": "머리 윗면에 주변 표면과 구별되는 선형 긁힘이 보이면 불합격한다.",
    },
    {
        "clause_id": "SCREW-04",
        "item": "screw",
        "defect_type": "scratch_neck",
        "text": "목 부위에서 나사산 방향을 가로지르는 긁힘이 보이면 불합격한다.",
    },
    {
        "clause_id": "SCREW-01",
        "item": "screw",
        "defect_type": "good",
        "text": "전체 윤곽이 일정하고 긁힘이 없으면 합격한다.",
    },
]


def test_fake_inspection_plan_and_judge_are_deterministic():
    llm = FastLLM("fake-fast", 0.0)

    plan, usage = llm.plan_inspection("screw", "표준 절차")
    assert [task["kind"] for task in plan["tasks"]] == [
        "observe", "cross_check", "spec_judge",
    ]
    assert usage["model"] == "fake-fast"

    result, _ = llm.judge("목 부위에 선형 긁힘이 보입니다.", CLAUSES, "screw")
    assert result["violations"] == ["SCREW-04"]
    assert result["defect_type"] == "scratch_neck"

    normal, _ = llm.judge(
        "표면과 윤곽에서 눈에 띄는 이상이 보이지 않습니다.", CLAUSES, "screw"
    )
    assert normal["violations"] == []
    assert normal["defect_type"] is None


def test_fake_judge_does_not_confuse_color_with_scratch_common_words():
    llm = FastLLM("fake-fast", 0.0)
    clauses = [
        {
            "clause_id": "METAL_NUT-06",
            "item": "metal_nut",
            "defect_type": "scratch",
            "text": "표면에 주변의 정상 질감과 구별되는 연속된 선형 긁힘이 뚜렷하게 보이면 불합격한다.",
        },
        {
            "clause_id": "METAL_NUT-05",
            "item": "metal_nut",
            "defect_type": "color",
            "text": "표면에 주변 금속색과 명확히 다른 변색 영역이 보이면 불합격한다.",
        },
    ]
    result, _ = llm.judge(
        "표면 일부에 주변과 구별되는 색상 변화가 보입니다.",
        clauses,
        "metal_nut",
    )
    assert result["violations"] == ["METAL_NUT-05"]
    assert result["defect_type"] == "color"


@pytest.mark.asyncio
async def test_fake_observation_and_rag_cover_every_type_map_entry():
    root = Path(__file__).resolve().parents[1]
    index = SpecIndex.from_dir(root / "data" / "specs", FakeEmbedder())
    llm = FastLLM("fake-fast", 0.0)
    vlm = FakeVLM()

    for item, defect_types in load_type_map().items():
        for defect_type in defect_types:
            observation = await vlm.observe(
                f"/synthetic/{item}/test/{defect_type}/000.png",
                item=item,
                region="full",
                style="observe",
                flip=False,
                reference_path=None,
            )
            hits = index.search(observation["text"], item=item, k=3)
            judgement, _ = llm.judge(observation["text"], hits, item)
            if defect_type == "good":
                assert judgement["violations"] == [], item
                assert judgement["defect_type"] is None, item
            else:
                assert judgement["violations"], (item, defect_type, hits)
                assert judgement["defect_type"] == defect_type, (
                    item,
                    defect_type,
                    hits,
                )


@pytest.mark.asyncio
async def test_fake_report_first_attempt_is_rejected_then_grounded():
    ctx = {
        "query": "검사해줘",
        "item": "screw",
        "lot": {"lot_id": "LOT-001"},
        "observations": [{
            "call_id": "obs-full",
            "text": "목 부위에 선형 긁힘이 보입니다.",
        }],
        "judgement": {"violations": ["SCREW-04"]},
        "verdict": "REJECT",
        "defect_type": "scratch_neck",
        "confidence": "high",
        "guide_text": "- 조치 옵션: 재작업 | 폐기",
    }
    llm = QualityLLM("fake-quality", 0.0)

    first = json.loads((await llm.write_report(ctx, 1, None))[0])
    second = json.loads((await llm.write_report(ctx, 2, "근거 없는 관찰"))[0])

    assert any(row["source_call_id"] == "hallucinated-observation"
               for row in first["observations"])
    assert second["observations"] == [{
        "text": "목 부위에 선형 긁힘이 보입니다.",
        "source_call_id": "obs-full",
    }]
    assert second["actions"] == ["재작업"]


@pytest.mark.parametrize("factory", [
    lambda: OllamaFastLLM("qwen", 0.0),
    lambda: OpenAIFastLLM("gpt", 0.0),
])
def test_real_plan_repairs_missing_and_invalid_tasks(monkeypatch, factory):
    llm = factory()

    def fake_chat(_system, _user, **_kwargs):
        return '{"tasks":[{"kind":"observe"},{"kind":"invalid"}]}', {
            "model": llm.name, "tokens": 1, "usd": 0.0,
        }

    monkeypatch.setattr(llm, "_chat", fake_chat)
    result, _ = llm.plan_inspection("screw", "표준 절차")
    assert [task["kind"] for task in result["tasks"]] == [
        "observe", "cross_check", "spec_judge",
    ]


def test_real_judge_drops_unknown_clause_and_uses_clause_type(monkeypatch):
    llm = OpenAIFastLLM("gpt", 0.0)

    def fake_chat(_system, _user, **_kwargs):
        return json.dumps({
            "violations": ["SCREW-04", "MADE-UP-99"],
            "defect_type": "thread_top",
            "rationale": "목 부위 긁힘",
        }), {"model": llm.name, "tokens": 1, "usd": 0.0}

    monkeypatch.setattr(llm, "_chat", fake_chat)
    result, _ = llm.judge("목 부위 긁힘", CLAUSES, "screw")
    assert result == {
        "violations": ["SCREW-04"],
        "defect_type": "scratch_neck",
        "rationale": "목 부위 긁힘",
    }


@pytest.mark.asyncio
async def test_openai_report_keeps_json_and_token_callback(monkeypatch):
    llm = OpenAIQualityLLM("gpt", 0.0)
    expected = '{"verdict":"PASS"}'

    def fake_chat(_system, _user, **_kwargs):
        return expected, {"model": llm.name, "tokens": 1, "usd": 0.0}

    tokens = []
    monkeypatch.setattr(llm, "_chat", fake_chat)
    content, usage = await llm.write_report({}, 1, None, on_token=tokens.append)
    assert content == expected
    assert tokens == [expected]
    assert usage["tokens"] == 1
