from __future__ import annotations

import json

import pytest

from agent.rag import FakeEmbedder
from eval.rag_eval import (
    RagEvaluationError,
    evaluate_model,
    load_dev_roster,
    render_markdown,
    validate_embedding_specs,
    validate_source_rows,
    write_outputs,
)


class RecordingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]):
        self.calls.append(list(texts))
        return super().embed(texts)


def _write_spec(directory) -> None:
    (directory / "screw.md").write_text(
        """# screw 검사 기준서

## 결함유형: good
- [SCREW-01] 전체 윤곽에 긁힘이 없고 정상 상태이면 합격한다.

## 결함유형: scratch_neck
- [SCREW-04] 목 부위에 선형 긁힘이 보이면 불합격한다.
""",
        encoding="utf-8",
    )


def _roster():
    return [
        {
            "id": "screw-scratch_neck-000",
            "item": "screw",
            "label": "scratch_neck",
            "is_defect": True,
            "split": "dev",
        },
        {
            "id": "screw-good-000",
            "item": "screw",
            "label": "good",
            "is_defect": False,
            "split": "dev",
        },
        {
            "id": "screw-scratch_neck-001",
            "item": "screw",
            "label": "scratch_neck",
            "is_defect": True,
            "split": "dev",
        },
    ]


def _rows():
    return [
        {
            **_roster()[0],
            "condition": "A",
            "rag_query_text": "목 부위에 선형 긁힘이 보인다.",
            "observations_text": "정상으로 보인다. 교차 검사도 정상이다.",
            "error": None,
        },
        {
            **_roster()[1],
            "condition": "A",
            "rag_query_text": "전체 윤곽에 긁힘이 없고 정상 상태다.",
            "observations_text": "목 긁힘이라는 잘못된 합성 문자열",
            "error": None,
        },
        {
            **_roster()[2],
            "condition": "A",
            "rag_query_text": "",
            "observations_text": "목 부위에 긁힘",
            "error": "RuntimeError: VLM timeout",
        },
    ]


def _type_map():
    return {
        "screw": {
            "good": ["good", "정상"],
            "scratch_neck": ["scratch_neck", "목 긁힘"],
        }
    }


def test_load_dev_roster_never_returns_heldout(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "seed": 20261020,
                "samples": [
                    _roster()[0],
                    {**_roster()[1], "split": "heldout"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    seed, rows = load_dev_roster(dataset)

    assert seed == 20261020
    assert [row["id"] for row in rows] == ["screw-scratch_neck-000"]
    assert all(row["split"] == "dev" for row in rows)


def test_source_rows_require_exact_production_query_and_complete_roster():
    ordered = validate_source_rows(_rows(), _roster())
    assert [row["id"] for row in ordered] == [row["id"] for row in _roster()]

    without_query = _rows()
    without_query[0].pop("rag_query_text")
    with pytest.raises(RagEvaluationError, match="rag_query_text"):
        validate_source_rows(without_query, _roster())

    with pytest.raises(RagEvaluationError, match="완전한 dev roster"):
        validate_source_rows(_rows()[:-1], _roster())


def test_rag_hit_rate_keeps_missing_query_in_denominator_and_uses_exact_query(tmp_path):
    _write_spec(tmp_path)
    embedder = RecordingEmbedder()

    summary, details = evaluate_model(
        rows=validate_source_rows(_rows(), _roster()),
        roster=_roster(),
        type_map=_type_map(),
        spec_dir=tmp_path,
        embedder=embedder,
        backend="fake",
        model="recording",
    )

    # 첫 호출은 조항 색인이고 이후 두 호출만 production query다.
    assert embedder.calls[1:] == [
        ["목 부위에 선형 긁힘이 보인다."],
        ["전체 윤곽에 긁힘이 없고 정상 상태다."],
    ]
    assert summary["expected_n"] == 3
    assert summary["query_count"] == 2
    assert summary["missing_or_error_query_count"] == 1
    assert summary["query_coverage"] == pytest.approx(2 / 3)
    assert summary["hit_count"] == 2
    assert summary["end_to_end_hit_at_3"] == pytest.approx(2 / 3)
    assert summary["selective_hit_at_3"] == 1.0
    assert summary["defect_expected_n"] == 2
    assert summary["defect_query_count"] == 1
    assert summary["defect_missing_or_error_query_count"] == 1
    assert summary["defect_query_coverage"] == 0.5
    assert summary["defect_hit_count"] == 1
    assert summary["defect_end_to_end_hit_at_3"] == 0.5
    assert summary["defect_selective_hit_at_3"] == 1.0
    assert details["screw-good-000"]["is_defect"] is False
    assert details["screw-scratch_neck-001"]["hit_at_3"] is False
    assert details["screw-scratch_neck-001"]["hits"] == []


def test_embedding_comparison_requires_two_distinct_models_and_blocks_fake_real_run():
    assert validate_embedding_specs(
        ["fake:one", "fake:two"], dry_run=True
    ) == [("fake", "one"), ("fake", "two")]

    with pytest.raises(RagEvaluationError, match="정확히 2개"):
        validate_embedding_specs(["openai:text-embedding-3-small"], dry_run=False)
    with pytest.raises(RagEvaluationError, match="서로 다른"):
        validate_embedding_specs(
            ["openai:text-embedding-3-small"] * 2, dry_run=False
        )
    with pytest.raises(RagEvaluationError, match="fake"):
        validate_embedding_specs(
            ["fake:one", "openai:text-embedding-3-small"], dry_run=False
        )


def test_dry_run_outputs_are_visibly_non_real_and_never_overwritten(tmp_path):
    payload = {
        "run_id": "rag-test",
        "status": "dry_run_not_real",
        "dry_run": True,
        "split": "dev",
        "top_k": 3,
        "source": {"run_id": "source-test"},
        "models": [
            {
                "model_id": "fake:one",
                "expected_n": 2,
                "query_count": 2,
                "query_coverage": 1.0,
                "hit_count": 2,
                "end_to_end_hit_at_3": 1.0,
                "selective_hit_at_3": 1.0,
                "defect_expected_n": 1,
                "defect_query_count": 1,
                "defect_query_coverage": 1.0,
                "defect_hit_count": 1,
                "defect_end_to_end_hit_at_3": 1.0,
                "defect_selective_hit_at_3": 1.0,
            }
        ],
    }

    text = render_markdown(payload)
    assert "DRYRUN" in text
    assert "실제 검색 성능으로 사용할 수 없음" in text
    assert "defect_end_to_end_hit_at_3" in text
    assert "정상(good) 검색을 포함하므로 보조 통계" in text

    directory = tmp_path / "rag-test"
    result_path, markdown_path, manifest_path = write_outputs(directory, payload)
    assert result_path.is_file()
    assert markdown_path.is_file()
    assert manifest_path.is_file()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["dry_run"] is True
    with pytest.raises(FileExistsError):
        write_outputs(directory, payload)
