"""거ᄗ사 거색 조항을 색인화고 검색하는 가벼운 RAG 공개 API."""

from .embedder import (
    FakeEmbedder,
    OllamaEmbedder,
    OpenAIEmbedder,
    make_embedder,
)
from .index import SpecIndex

__all__ = [
    "FakeEmbedder",
    "OllamaEmbedder",
    "OpenAIEmbedder",
    "SpecIndex",
    "make_embedder",
]
