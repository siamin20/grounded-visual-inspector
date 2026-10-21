"""Phase 3 제품 검사용 텍스트 LLM 인터페이스 검증."""

import json
from pathlib import Path

import pytest

from agent import ollama_llm as ollama_backend
from agent import openai_llm as openai_backend
from agent.fake_llm import FastLLM, QualityLLM
from agent.ollama_llm import (
    OllamaFastLLM,
    _normalize_judgement as normalize_ollama_judgement,
)
from agent.openai_llm import (
    OpenAIFastLLM,
    OpenAIQualityLLM,
    _normalize_judgement as normalize_openai_judgement,
)
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


class _HTTPResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


def test_fake_inspection_plan_and_judge_are_deterministic():
    llm = FastLLM("fake-fast", 0.0)

    plan, usage = llm.plan_inspection("screw", "표준 절차")
    assert [task["kind"] for task in plan["tasks"]] == [
        "observe", "cross_check", "spec_judge",
    ]
    assert usage["model"] == "fake-fast"
    assert usage["input_tokens"] + usage["output_tokens"] == usage["tokens"]
    assert usage["token_accounting"] == "estimated_split"
    assert usage["estimated_usd"] == 0.0
    assert usage["compute_proxy_usd"] == 0.0
    assert usage["requested_model"] == "fake-fast"
    assert usage["observed_model"] == "fake-fast"
    assert usage["successful_provider_requests"] == 0

    result, _ = llm.judge("목 부위에 선형 긁힘이 보입니다.", CLAUSES, "screw")
    assert result["violations"] == ["SCREW-04"]
    assert result["defect_type"] == "scratch_neck"

    normal, _ = llm.judge(
        "표면과 윤곽에서 눈에 띄는 이상이 보이지 않습니다.", CLAUSES, "screw"
    )
    assert normal["violations"] == []
    assert normal["defect_type"] is None


def test_openai_usage_keeps_provider_split_and_observed_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    llm = OpenAIFastLLM("requested", 0.1)
    payload = {
        "model": "resolved-2026-10-21",
        "choices": [{"message": {"content": "{}"}}],
        "usage": {
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "total_tokens": 10,
        },
    }
    monkeypatch.setattr(
        openai_backend.httpx,
        "post",
        lambda *args, **kwargs: _HTTPResponse(payload),
    )

    _, usage = llm._chat("system", "user")

    assert usage == {
        "model": "openai:requested",
        "tokens": 10,
        "usd": 0.001,
        "input_tokens": 7,
        "output_tokens": 3,
        "token_accounting": "provider_split",
        "estimated_usd": 0.001,
        "compute_proxy_usd": 0.0,
        "requested_model": "openai:requested",
        "observed_model": "openai:resolved-2026-10-21",
        "successful_provider_requests": 1,
            "unknown_token_usage_successes": 0,
            "observed_model_response_count": 1,
            "unobserved_model_successes": 0,
    }


def test_openai_usage_does_not_invent_missing_observed_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    llm = OpenAIFastLLM("requested", 0.1)
    monkeypatch.setattr(
        openai_backend.httpx,
        "post",
        lambda *args, **kwargs: _HTTPResponse({
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"total_tokens": 5},
        }),
    )

    _, usage = llm._chat("system", "user")

    assert usage["observed_model"] is None
    assert usage["input_tokens"] is None
    assert usage["output_tokens"] is None
    assert usage["token_accounting"] == "total_only"
    assert usage["tokens"] == 5
    assert usage["successful_provider_requests"] == 1


def test_ollama_usage_separates_compute_proxy_from_estimated_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = OllamaFastLLM("requested", 0.2)
    monkeypatch.setattr(
        ollama_backend.httpx,
        "post",
        lambda *args, **kwargs: _HTTPResponse({
            "model": "resolved:latest",
            "message": {"content": "{}"},
            "prompt_eval_count": 4,
            "eval_count": 6,
        }),
    )

    _, usage = llm._chat("system", "user")

    assert usage == {
        "model": "ollama:requested",
        "tokens": 10,
        "usd": 0.002,
        "input_tokens": 4,
        "output_tokens": 6,
        "token_accounting": "provider_split",
        "estimated_usd": 0.0,
        "compute_proxy_usd": 0.002,
        "requested_model": "ollama:requested",
        "observed_model": "ollama:resolved:latest",
        "successful_provider_requests": 1,
            "unknown_token_usage_successes": 0,
            "observed_model_response_count": 1,
            "unobserved_model_successes": 0,
    }


@pytest.mark.asyncio
async def test_ollama_stream_usage_uses_final_counts_and_response_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = [
        json.dumps({
            "model": "resolved:latest",
            "message": {"content": "안"},
            "done": False,
        }),
        json.dumps({
            "model": "resolved:latest",
            "message": {"content": "녕"},
            "done": True,
            "prompt_eval_count": 5,
            "eval_count": 2,
        }),
    ]

    class StreamResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def raise_for_status(self) -> None:
            return None

        async def aiter_lines(self):
            for line in lines:
                yield line

    class AsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, *_args, **_kwargs):
            return StreamResponse()

    monkeypatch.setattr(
        ollama_backend.httpx,
        "AsyncClient",
        lambda **_kwargs: AsyncClient(),
    )
    llm = OllamaFastLLM("requested", 0.2)

    content, usage = await llm._chat_stream("system", "user")

    assert content == "안녕"
    assert usage["input_tokens"] == 5
    assert usage["output_tokens"] == 2
    assert usage["tokens"] == 7
    assert usage["observed_model"] == "ollama:resolved:latest"
    assert usage["successful_provider_requests"] == 1


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
    assert second["abstain_reason"] is None


@pytest.mark.asyncio
async def test_fake_report_emits_abstain_contract() -> None:
    llm = QualityLLM("fake-quality", 0.0)
    ctx = {
        "item": "screw",
        "lot": {},
        "observations": [],
        "judgement": {"violations": ["SCREW-04"]},
        "verdict": "ABSTAIN",
        "defect_type": "scratch_neck",
        "confidence": "high",
        "abstain_reason": "insufficient_visual_evidence",
        "guide_text": "- 조치 옵션: 폐기",
    }

    report = json.loads((await llm.write_report(ctx, 2, None))[0])

    assert report["verdict"] == "ABSTAIN"
    assert report["defect_type"] is None
    assert report["confidence"] == "low"
    assert report["abstain_reason"] == "insufficient_visual_evidence"
    assert report["spec_clauses"] == []
    assert report["actions"] == ["재검사"]


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


def test_real_judge_preserves_claim_for_shared_validation(monkeypatch):
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
        "violations": ["SCREW-04", "MADE-UP-99"],
        "defect_type": "thread_top",
        "rationale": "목 부위 긁힘",
    }


@pytest.mark.parametrize(
    "normalizer",
    [normalize_openai_judgement, normalize_ollama_judgement],
)
@pytest.mark.parametrize("raw", [None, [], {"violations": [{}]}])
def test_real_judge_normalizer_never_crashes_or_invents_claims(
    normalizer,
    raw,
) -> None:
    result = normalizer(raw, CLAUSES)

    assert isinstance(result, dict)
    if isinstance(raw, dict):
        assert result == raw
    else:
        assert result.get("_invalid_shape")


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
