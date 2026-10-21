"""거ᄗ사 RAG에서 사용하는 임베딩 백엔드.

fake 백엔드는 외부 호출 없이 결과와 무관하게 문자 2-gram을 고정 해쉬 버킷에
누적한다. 실모델 백엔드는 같은 ``embed`` 메서드와 ``last_usage``
필드로 사용해 교체할 수 있게 한다.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol

import httpx
import numpy as np

from agent.usage import MISSING, nonnegative_int


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OPENAI_URL = "https://api.openai.com/v1/embeddings"


class Embedder(Protocol):
    """임베딩 백엔드가 지켜야 할 최소 계약."""

    last_usage: dict

    def embed(self, texts: list[str]) -> np.ndarray:
        """입력 순서대로 벡터를 반환다."""


def _empty_usage(model: str) -> dict:
    """외부 호출이 없음을 usage에 명시한다."""
    is_fake = model.startswith("fake:")
    return {
        "model": model,
        "tokens": 0,
        "usd": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "token_accounting": "estimated_split" if is_fake else "provider_split",
        "estimated_usd": 0.0,
        "compute_proxy_usd": 0.0,
        "requested_model": model,
        "observed_model": model if is_fake else None,
        "successful_provider_requests": 0,
        "unknown_token_usage_successes": 0,
        "observed_model_response_count": 0,
        "unobserved_model_successes": 0,
    }


def _usage_count(container: object, key: str, *, default: object = MISSING) -> int:
    """누락값과 잘못된 값을 구분하고 잘못된 수치는 허용하지 않는다."""
    if not isinstance(container, dict) or key not in container:
        return nonnegative_int(default, name=key)
    return nonnegative_int(container[key], name=key)


def _observed_model(prefix: str, value: object) -> str | None:
    """관측 모델명은 실제 옹답 필드에서만 가져온다."""
    if not isinstance(value, str) or not value.strip():
        return None
    return f"{prefix}:{value.strip()}"


def _as_matrix(values: object, expected_rows: int) -> np.ndarray:
    """API 영베딩을 정규화고 행 수라는 형태로 검증한다."""
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows:
        raise RuntimeError(
            f"임베딩 옹답의 형태가 올바르지 않습니다: {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise RuntimeError("임베딩 옹답에 유한 값이 있습니다.")
    return matrix


class FakeEmbedder:
    """무강화 재현한 512차원 문자 2-gram 임베딩."""

    dimension = 512

    def __init__(self) -> None:
        self.name = "fake:char-2gram-512"
        self.last_usage = _empty_usage(self.name)

    @staticmethod
    def _normalise(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @classmethod
    def _grams(cls, text: str) -> list[str]:
        value = cls._normalise(text)
        if not value:
            return []
        if len(value) == 1:
            return [value]
        return [value[index:index + 2] for index in range(len(value) - 1)]

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError("임베딩 입력은 문자여야 합니다.")
            for gram in self._grams(text):
                digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest, "big") % self.dimension
                matrix[row, bucket] += 1.0
            norm = float(np.linalg.norm(matrix[row]))
            if norm:
                matrix[row] /= norm
        tokens = sum(len(self._grams(text)) for text in texts)
        self.last_usage = {
            "model": self.name,
            "tokens": tokens,
            "usd": 0.0,
            "input_tokens": tokens,
            "output_tokens": 0,
            "token_accounting": "estimated_split",
            "estimated_usd": 0.0,
            "compute_proxy_usd": 0.0,
            "requested_model": self.name,
            "observed_model": self.name,
            "successful_provider_requests": 0,
            "unknown_token_usage_successes": 0,
            "observed_model_response_count": 0,
            "unobserved_model_successes": 0,
        }
        return matrix


class OllamaEmbedder:
    """Ollama ``/api/embed`` 엔드포인틀 사용하는 임베딩."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        compute_proxy_usd_per_1k: float = 0.00002,
    ) -> None:
        self.model = model or os.environ.get("EMBED_MODEL", "nomic-embed-text")
        self.name = f"ollama:{self.model}"
        root = base_url or os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)
        self.url = root.rstrip("/") + "/api/embed"
        self.compute_proxy_usd_per_1k = compute_proxy_usd_per_1k
        self.last_usage = _empty_usage(self.name)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            self.last_usage = _empty_usage(self.name)
            return np.empty((0, 0), dtype=np.float32)
        response = httpx.post(
            self.url,
            json={"model": self.model, "input": texts},
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()
        token_usage_known = False
        if isinstance(data, dict) and "prompt_eval_count" in data:
            tokens = _usage_count(data, "prompt_eval_count")
            token_usage_known = True
        else:
            raw_usage = data.get("usage") if isinstance(data, dict) else None
            if isinstance(raw_usage, dict) and "total_tokens" in raw_usage:
                tokens = _usage_count(raw_usage, "total_tokens")
                token_usage_known = True
            else:
                tokens = 0
        observed_model = _observed_model("ollama", data.get("model"))
        compute_proxy_usd = round(
            tokens / 1000 * self.compute_proxy_usd_per_1k, 6
        )
        self.last_usage = {
            "model": self.name,
            "tokens": tokens,
            "usd": 0.0,
            "input_tokens": tokens if token_usage_known else None,
            "output_tokens": 0 if token_usage_known else None,
            "token_accounting": "provider_split" if token_usage_known else "total_only",
            "estimated_usd": 0.0,
            "compute_proxy_usd": compute_proxy_usd,
            "requested_model": self.name,
            "observed_model": observed_model,
            "successful_provider_requests": 1,
            "unknown_token_usage_successes": 0 if token_usage_known else 1,
            "observed_model_response_count": 1 if observed_model else 0,
            "unobserved_model_successes": 0 if observed_model else 1,
        }
        # provider 성공 응답의 usage는 vector shape 검증이 실패해도
        # 눈에서 사라지지 않도록 먼저 고정한다.
        matrix = _as_matrix(data.get("embeddings"), len(texts))
        return matrix


class OpenAIEmbedder:
    """OpenAI Embeddings API를 ``httpx``로 지겁 호출하는 백엔드."""

    def __init__(
        self,
        model: str | None = None,
        url: str | None = None,
        usd_per_1k_tokens: float = 0.00002,
    ) -> None:
        self.model = model or os.environ.get("EMBED_MODEL", "text-embedding-3-small")
        self.name = f"openai:{self.model}"
        self.url = url or os.environ.get("OPENAI_EMBED_URL", DEFAULT_OPENAI_URL)
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.usd_per_1k = usd_per_1k_tokens
        self.last_usage = _empty_usage(self.name)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            self.last_usage = _empty_usage(self.name)
            return np.empty((0, 0), dtype=np.float32)
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY가 설정되지 않아 OpenAI 임베딩을 사용할 수 없습니다."
            )
        response = httpx.post(
            self.url,
            json={"model": self.model, "input": texts},
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()
        raw_usage = data.get("usage") if isinstance(data, dict) else None
        token_usage_known = False
        if isinstance(raw_usage, dict) and "prompt_tokens" in raw_usage:
            tokens = _usage_count(raw_usage, "prompt_tokens")
            token_usage_known = True
        elif isinstance(raw_usage, dict) and "total_tokens" in raw_usage:
            tokens = _usage_count(raw_usage, "total_tokens")
            token_usage_known = True
        else:
            tokens = 0
        estimated_usd = round(tokens / 1000 * self.usd_per_1k, 6)
        observed_model = _observed_model("openai", data.get("model"))
        self.last_usage = {
            "model": self.name,
            "tokens": tokens,
            "usd": estimated_usd,
            "input_tokens": tokens if token_usage_known else None,
            "output_tokens": 0 if token_usage_known else None,
            "token_accounting": "provider_split" if token_usage_known else "total_only",
            "estimated_usd": estimated_usd,
            "compute_proxy_usd": 0.0,
            "requested_model": self.name,
            "observed_model": observed_model,
            "successful_provider_requests": 1,
            "unknown_token_usage_successes": 0 if token_usage_known else 1,
            "observed_model_response_count": 1 if observed_model else 0,
            "unobserved_model_successes": 0 if observed_model else 1,
        }
        # usage 영수증을 남긴 뒤 벡터 크기를 검증한다.
        ordered = sorted(data.get("data", []), key=lambda row: row.get("index", 0))
        matrix = _as_matrix([row.get("embedding") for row in ordered], len(texts))
        return matrix


def make_embedder(backend: str | None = None) -> Embedder:
    """``EMBED_BACKEND``(미지정 시 ``LLM_BACKEND``)에 맞춰 백엔드를 만든다."""
    selected = (backend or os.environ.get("EMBED_BACKEND")
                or os.environ.get("LLM_BACKEND", "fake")).strip().lower()
    if selected == "fake":
        return FakeEmbedder()
    if selected == "ollama":
        return OllamaEmbedder()
    if selected == "openai":
        return OpenAIEmbedder()
    raise ValueError(f"지원하지 않은 EMBED_BACKEND입니다: {selected}")
