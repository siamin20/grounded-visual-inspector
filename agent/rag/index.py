"""거ᄗ사서의 조항을 파싱하고 검색하는 인메모리 벡터 스토어."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from langsmith import traceable

from .embedder import Embedder


# macOS 파일시스템의 한글명이 NFC/NFD 어느 쪽인지와 무관하게
# 결함유형 문자열을 지정하지 않고 제목 구조로 파싱한다.
SECTION_RE = re.compile(r"^##\s*[^:]+:\s*(\S+)\s*$")
CLAUSE_RE = re.compile(r"^\s*-\s*\[([^\]]+)\]\s*(.+?)\s*$")


class SpecIndex:
    """외부 DB 없이 NumPy만으로 작동하는 조항 단위 색인."""

    def __init__(
        self,
        clauses: list[dict[str, str]],
        matrix: np.ndarray,
        embedder: Embedder,
        full_texts: dict[str, str],
    ) -> None:
        if matrix.ndim != 2 or matrix.shape[0] != len(clauses):
            raise ValueError("조항 수와 임베딩 행렬 수가 일치하지 않습니다.")
        self.clauses = [dict(clause) for clause in clauses]
        self.embedder = embedder
        self._matrix = np.asarray(matrix, dtype=np.float32)
        self._full_texts = dict(full_texts)
        self._by_id = {clause["clause_id"]: clause for clause in self.clauses}
        if len(self._by_id) != len(self.clauses):
            raise ValueError("거ᄗ사서에 중복된 clause_id가 있습니다.")

    @classmethod
    def from_dir(cls, path: str | Path, embedder: Embedder) -> "SpecIndex":
        root = Path(path)
        if not root.is_dir():
            raise FileNotFoundError(f"거ᄗ사서 디렉터리가 없습니다: {root}")

        clauses: list[dict[str, str]] = []
        full_texts: dict[str, str] = {}
        for file_path in sorted(root.glob("*.md")):
            item = file_path.stem
            text = file_path.read_text(encoding="utf-8")
            full_texts[item] = text
            defect_type: str | None = None
            for line_number, line in enumerate(text.splitlines(), start=1):
                section = SECTION_RE.match(line)
                if section:
                    defect_type = section.group(1)
                    continue
                clause = CLAUSE_RE.match(line)
                if not clause:
                    continue
                if defect_type is None:
                    raise ValueError(
                        f"결함유형 섹션 없이 조항이 나타났습니다: {file_path}:{line_number}"
                    )
                clauses.append({
                    "clause_id": clause.group(1).strip(),
                    "item": item,
                    "defect_type": defect_type,
                    "text": clause.group(2).strip(),
                })

        if not clauses:
            raise ValueError(f"색인할 거ᄗ사서 조항이 없습니다: {root}")
        matrix = np.asarray(embedder.embed([clause["text"] for clause in clauses]), dtype=np.float32)
        return cls(clauses, matrix, embedder, full_texts)

    @traceable(run_type="retriever", name="spec_search")
    def search(self, query: str, item: str, k: int = 3) -> list[dict[str, Any]]:
        """품목을 먼저 필터한 다음 cosine 상위 ``k``개 조항을 반환다."""
        if k <= 0:
            return []
        indices = [index for index, clause in enumerate(self.clauses)
                   if clause["item"] == item]
        if not indices:
            return []

        query_matrix = np.asarray(self.embedder.embed([query]), dtype=np.float32)
        if query_matrix.ndim != 2 or query_matrix.shape[0] != 1:
            raise RuntimeError("질의 임베딩 형태가 올바르지 않습니다.")
        if query_matrix.shape[1] != self._matrix.shape[1]:
            raise RuntimeError("질의와 색인 임베딩 차원이 답릅니다.")

        selected = self._matrix[indices]
        query_vector = query_matrix[0]
        numerator = selected @ query_vector
        denominator = np.linalg.norm(selected, axis=1) * np.linalg.norm(query_vector)
        scores = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float32),
            where=denominator > 0,
        )
        order = sorted(range(len(indices)), key=lambda pos: (-float(scores[pos]), pos))[:k]
        results: list[dict[str, Any]] = []
        for pos in order:
            row: dict[str, Any] = dict(self.clauses[indices[pos]])
            row["score"] = float(scores[pos])
            results.append(row)
        return results

    def has(self, clause_id: str, item: str) -> bool:
        clause = self._by_id.get(clause_id)
        return clause is not None and clause["item"] == item

    def has_clause(self, clause_id: str, item: str) -> bool:
        """``has``의 명시적 별칭. 평가와 노드에서 모두 의미로 공개한다."""
        return self.has(clause_id, item)

    def clause(self, clause_id: str) -> dict[str, str]:
        try:
            return dict(self._by_id[clause_id])
        except KeyError as exc:
            raise KeyError(f"알 수 없는 거ᄗ사 조항입니다: {clause_id}") from exc

    def full_text(self, item: str) -> str:
        try:
            return self._full_texts[item]
        except KeyError as exc:
            raise KeyError(f"알 수 없는 품목입니다: {item}") from exc
