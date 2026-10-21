"""평가 전에 고정한 실제 기준서 4개의 공개 계약 테스트."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from agent.rag import FakeEmbedder, SpecIndex


ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = ROOT / "data" / "specs"
EXPECTED_CLAUSES_BY_ITEM = {
    "screw": 6,
    "bottle": 6,
    "metal_nut": 6,
    "hazelnut": 7,
}


def test_real_specs_keep_25_clause_item_filter_contract() -> None:
    index = SpecIndex.from_dir(SPEC_DIR, FakeEmbedder())

    assert len(index.clauses) == 25
    assert Counter(clause["item"] for clause in index.clauses) == Counter(
        EXPECTED_CLAUSES_BY_ITEM
    )

    for item, expected_count in EXPECTED_CLAUSES_BY_ITEM.items():
        found = index.search("외관 결함 검사", item=item, k=25)
        assert len(found) == expected_count
        assert {row["item"] for row in found} == {item}
    assert index.search("외관 결함 검사", item="unknown", k=25) == []


def test_real_specs_expose_has_and_full_text_contract() -> None:
    index = SpecIndex.from_dir(SPEC_DIR, FakeEmbedder())

    assert index.has("SCREW-01", "screw")
    assert index.has_clause("SCREW-06", "screw")
    assert not index.has("SCREW-01", "bottle")
    assert not index.has("UNKNOWN-00", "screw")

    for item in EXPECTED_CLAUSES_BY_ITEM:
        expected = (SPEC_DIR / f"{item}.md").read_text(encoding="utf-8")
        assert index.full_text(item) == expected
    with pytest.raises(KeyError):
        index.full_text("unknown")
