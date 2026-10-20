from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from eval.human_check import create_blind_set, unblind_scores, write_blind_set
from eval.report import ResultDataError, load_result_rows, select_rows, write_report


TYPE_MAP = {
    "screw": {
        "good": ["good", "정상"],
        "scratch_neck": ["scratch_neck", "목 긁힘"],
    }
}


def _row(condition: str, sample_id: str, *, defect: bool) -> dict[str, object]:
    observation = "목 표면에 연속적인 긁힘이 보입니다."
    return {
        "id": sample_id,
        "item": "screw",
        "label": "scratch_neck" if defect else "good",
        "is_defect": defect,
        "split": "heldout",
        "condition": condition,
        "verdict": "REJECT" if defect else "PASS",
        "defect_type": "scratch_neck" if defect else None,
        "confidence": "high",
        "approval_required": False,
        "validation_attempts": 1,
        "observations_text": [observation],
        "report": {
            "summary": "검사 결과입니다.",
            "observations": [{"text": observation, "source_call_id": "obs-full"}],
        },
        "latency_s": 1.0,
        "tokens": 10,
        "usd": 0.01,
        "error": "",
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_report_groups_rows_by_fields_not_filenames(tmp_path: Path) -> None:
    rows = [
        _row(condition, sample_id, defect=defect)
        for condition in ("B0", "B1", "A")
        for sample_id, defect in (("normal", False), ("defect", True))
    ]
    _write_jsonl(tmp_path / "arbitrary-first.jsonl", rows[::2])
    _write_jsonl(tmp_path / "arbitrary-second.jsonl", rows[1::2])
    # 디렉터리 집계에서는 dry-run이 실측 결과를 오염시키지 않는다.
    _write_jsonl(tmp_path / "A_heldout_DRYRUN.jsonl", [_row("A", "normal", defect=False)])
    type_map_path = tmp_path / "type_map.json"
    type_map_path.write_text(json.dumps(TYPE_MAP, ensure_ascii=False), encoding="utf-8")

    markdown, output_path = write_report(
        tmp_path,
        split="heldout",
        conditions=["B0", "B1", "A"],
        type_map_path=type_map_path,
    )

    assert output_path == tmp_path / "summary_heldout.md"
    assert output_path.read_text(encoding="utf-8") == markdown
    assert "B0 vs A" in markdown
    assert "B1 vs A" in markdown
    assert "citation_fidelity" in markdown
    assert "screw" in markdown


def test_report_identifies_malformed_jsonl_and_missing_condition(tmp_path: Path) -> None:
    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"condition": "A", "split": "heldout"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ResultDataError, match=r"broken\.jsonl:2"):
        load_result_rows(tmp_path)

    with pytest.raises(ResultDataError, match="B1"):
        select_rows(
            [_row("A", "one", defect=False)],
            split="heldout",
            conditions=["A", "B1"],
        )


def test_human_check_is_deterministic_blind_and_unblinds() -> None:
    rows = [
        _row(condition, f"sample-{index}", defect=bool(index % 2))
        for condition in ("B0", "A")
        for index in range(4)
    ]

    first, first_mapping = create_blind_set(rows, sample=3, seed=7)
    second, second_mapping = create_blind_set(rows, sample=3, seed=7)

    assert first == second
    assert first_mapping == second_mapping
    assert len(first) == 6
    assert all("condition" not in row for row in first)
    assert all("source_call_id" not in json.dumps(row, ensure_ascii=False) for row in first)
    for index, row in enumerate(first):
        row["hallucinated_sentences"] = index % 2

    summary = unblind_scores(first, first_mapping)
    assert summary["B0"]["reports"] == 3
    assert summary["A"]["reports"] == 3
    assert sum(int(values["hallucinated_sentences"]) for values in summary.values()) == 3


def test_human_check_writes_private_mapping(tmp_path: Path) -> None:
    rows = [
        _row(condition, f"sample-{index}", defect=bool(index % 2))
        for condition in ("B0", "A")
        for index in range(2)
    ]
    _write_jsonl(tmp_path / "results.jsonl", rows)

    blind_path, mapping_path, count = write_blind_set(tmp_path, sample=2, seed=3)

    assert count == 4
    assert stat.S_IMODE(mapping_path.stat().st_mode) == 0o600
    assert "condition" not in blind_path.read_text(encoding="utf-8")
    assert "condition" in mapping_path.read_text(encoding="utf-8")
