from __future__ import annotations

import numpy as np
import pytest

from agent.rag import (
    FakeEmbedder,
    OllamaEmbedder,
    OpenAIEmbedder,
    SpecIndex,
    make_embedder,
)


def _write_specs(tmp_path):
    (tmp_path / "screw.md").write_text(
        """# screw 거ᄗ사 기준서

## 결함유형: good
- [SCREW-01] 모둔 외관이 고라 합격한다.
## 결함유형: scratch_neck
- [SCREW-04] 목 부위에 선형 긁힘이 보이면 불합격한다.
""",
        encoding="utf-8",
    )
    (tmp_path / "bottle.md").write_text(
        """# bottle 거ᄗ사 기준서
## 결함유형: contamination
- [BOTTLE-06] 표면에 이물이 보이면 불합겨ᅨ한다.
""",
        encoding="utf-8",
    )


def test_fake_embedder_is_deterministic_and_normalised():
    embedder = FakeEmbedder()
    first = embedder.embed(["목 부위 긁힘", ""])
    second = FakeEmbedder().embed(["목 부위 긁힘", ""])

    assert first.shape == (2, 512)
    assert np.array_equal(first, second)
    assert np.linalg.norm(first[0]) == pytest.approx(1.0)
    assert np.count_nonzero(first[1]) == 0
    assert embedder.last_usage["tokens"] > 0
    assert embedder.last_usage["usd"] == 0.0
    assert embedder.last_usage["input_tokens"] == embedder.last_usage["tokens"]
    assert embedder.last_usage["output_tokens"] == 0
    assert embedder.last_usage["token_accounting"] == "estimated_split"
    assert embedder.last_usage["successful_provider_requests"] == 0


def test_spec_index_parses_filters_and_searches(tmp_path):
    _write_specs(tmp_path)
    index = SpecIndex.from_dir(tmp_path, FakeEmbedder())

    found = index.search("목에 긁힘이 보임", item="screw", k=2)

    assert [row["clause_id"] for row in found] == ["SCREW-04", "SCREW-01"]
    assert all(row["item"] == "screw" for row in found)
    assert found[0]["score"] > found[1]["score"]
    assert index.has("SCREW-04", "screw")
    assert index.has_clause("SCREW-04", "screw")
    assert not index.has("SCREW-04", "bottle")
    assert index.clause("BOTTLE-06")["defect_type"] == "contamination"
    assert "SCREW-04" in index.full_text("screw")
    assert index.search("anything", item="unknown") == []
    assert index.embedder.last_usage["tokens"] == 0
    assert index.embedder.last_usage["successful_provider_requests"] == 0
    assert index.search("anything", item="screw", k=0) == []
    assert index.embedder.last_usage["tokens"] == 0
    assert index.embedder.last_usage["successful_provider_requests"] == 0


def test_spec_index_rejects_duplicate_clause_ids(tmp_path):
    _write_specs(tmp_path)
    (tmp_path / "other.md").write_text(
        "## 결함유형: good\n- [SCREW-01] 중복 조항.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="clause_id"):
        SpecIndex.from_dir(tmp_path, FakeEmbedder())


def test_make_embedder_uses_backend_environment(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "fake")
    monkeypatch.delenv("EMBED_BACKEND", raising=False)
    assert isinstance(make_embedder(), FakeEmbedder)

    monkeypatch.setenv("EMBED_BACKEND", "ollama")
    assert isinstance(make_embedder(), OllamaEmbedder)

    with pytest.raises(ValueError, match="EMBED_BACKEND"):
        make_embedder("missing")


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_ollama_embedder_records_usage(monkeypatch):
    called = {}

    def fake_post(url, **kwargs):
        called.update(url=url, **kwargs)
        return _Response({
            "embeddings": [[1.0, 0.0], [0.0, 1.0]],
            "prompt_eval_count": 7,
            "model": "embed-test-resolved",
        })

    monkeypatch.setattr("agent.rag.embedder.httpx.post", fake_post)
    embedder = OllamaEmbedder(model="embed-test", base_url="http://ollama.local")
    matrix = embedder.embed(["a", "b"])

    assert called["url"] == "http://ollama.local/api/embed"
    assert called["json"] == {"model": "embed-test", "input": ["a", "b"]}
    assert matrix.shape == (2, 2)
    assert embedder.last_usage == {
        "model": "ollama:embed-test",
        "tokens": 7,
        "usd": 0.0,
        "input_tokens": 7,
        "output_tokens": 0,
        "token_accounting": "provider_split",
        "estimated_usd": 0.0,
        "compute_proxy_usd": 0.0,
        "requested_model": "ollama:embed-test",
        "observed_model": "ollama:embed-test-resolved",
        "successful_provider_requests": 1,
            "unknown_token_usage_successes": 0,
            "observed_model_response_count": 1,
            "unobserved_model_successes": 0,
    }


def test_openai_embedder_orders_rows_and_records_usage(monkeypatch):
    called = {}

    def fake_post(url, **kwargs):
        called.update(url=url, **kwargs)
        return _Response({
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ],
            "usage": {"prompt_tokens": 50, "total_tokens": 50},
            "model": "embed-test-2026",
        })

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("agent.rag.embedder.httpx.post", fake_post)
    embedder = OpenAIEmbedder(model="embed-test", url="https://openai.test/v1/embeddings")
    matrix = embedder.embed(["a", "b"])

    assert matrix.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert called["headers"]["Authorization"] == "Bearer test-key"
    assert embedder.last_usage == {
        "model": "openai:embed-test",
        "tokens": 50,
        "usd": 1e-06,
        "input_tokens": 50,
        "output_tokens": 0,
        "token_accounting": "provider_split",
        "estimated_usd": 1e-06,
        "compute_proxy_usd": 0.0,
        "requested_model": "openai:embed-test",
        "observed_model": "openai:embed-test-2026",
        "successful_provider_requests": 1,
            "unknown_token_usage_successes": 0,
            "observed_model_response_count": 1,
            "unobserved_model_successes": 0,
    }


@pytest.mark.parametrize("backend", ["ollama", "openai"])
def test_embedder_preserves_provider_usage_before_vector_shape_validation(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if backend == "ollama":
        payload = {
            "embeddings": [[1.0, 0.0]],
            "prompt_eval_count": 7,
            "model": "resolved-embed",
        }
        embedder = OllamaEmbedder(
            model="embed-test", base_url="http://ollama.local"
        )
    else:
        payload = {
            "data": [{"index": 0, "embedding": [1.0, 0.0]}],
            "usage": {"prompt_tokens": 7},
            "model": "resolved-embed",
        }
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        embedder = OpenAIEmbedder(
            model="embed-test", url="https://openai.test/v1/embeddings"
        )
    monkeypatch.setattr(
        "agent.rag.embedder.httpx.post",
        lambda *args, **kwargs: _Response(payload),
    )

    with pytest.raises(RuntimeError):
        embedder.embed(["first", "second"])

    assert embedder.last_usage["successful_provider_requests"] == 1
    assert embedder.last_usage["tokens"] == 7
    assert embedder.last_usage["observed_model"] in {
        "ollama:resolved-embed", "openai:resolved-embed"
    }


def test_ollama_embedder_does_not_invent_observed_model(monkeypatch):
    monkeypatch.setattr(
        "agent.rag.embedder.httpx.post",
        lambda *args, **kwargs: _Response({
            "embeddings": [[1.0, 0.0]],
            "prompt_eval_count": 3,
        }),
    )
    embedder = OllamaEmbedder(model="requested", base_url="http://ollama.local")

    embedder.embed(["a"])

    assert embedder.last_usage["requested_model"] == "ollama:requested"
    assert embedder.last_usage["observed_model"] is None
    assert embedder.last_usage["successful_provider_requests"] == 1


@pytest.mark.parametrize("bad_count", [-1, True, "7"])
def test_ollama_embedder_rejects_invalid_usage_counts(monkeypatch, bad_count):
    monkeypatch.setattr(
        "agent.rag.embedder.httpx.post",
        lambda *args, **kwargs: _Response({
            "embeddings": [[1.0, 0.0]],
            "prompt_eval_count": bad_count,
        }),
    )
    embedder = OllamaEmbedder(model="embed-test", base_url="http://ollama.local")

    with pytest.raises(ValueError, match="prompt_eval_count"):
        embedder.embed(["a"])


def test_real_embedder_empty_input_records_no_provider_request(monkeypatch):
    embedder = OpenAIEmbedder(model="embed-test")

    matrix = embedder.embed([])

    assert matrix.shape == (0, 0)
    assert embedder.last_usage["requested_model"] == "openai:embed-test"
    assert embedder.last_usage["observed_model"] is None
    assert embedder.last_usage["successful_provider_requests"] == 0
