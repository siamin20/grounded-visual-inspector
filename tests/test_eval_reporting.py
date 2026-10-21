from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest

from eval.human_check import (
    DEFAULT_SEED,
    HumanCheckError,
    create_blind_set,
    main as human_check_main,
    unblind_scores,
    write_blind_set,
)
from eval.report import (
    ResultDataError,
    load_expected_rows,
    load_result_rows,
    select_rows,
    write_report,
)


TYPE_MAP = {
    "screw": {
        "good": ["good", "정상"],
        "scratch_neck": ["scratch_neck", "목 긁힘"],
    }
}


def _row(condition: str, sample_id: str, *, defect: bool) -> dict[str, object]:
    observation = "목 표면에 연속적인 긁힘이 보입니다."
    before_report = {
        "summary": "표면 상태를 확인한 결과입니다.",
        "observations": [{"text": observation, "source_call_id": "obs-full"}],
    }
    before_sha256 = hashlib.sha256(
        json.dumps(
            before_report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
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
        "pre_validation_snapshot": {
            "version": 1,
            "stage": "before_validate",
            "report": before_report,
            "report_sha256": before_sha256,
            "cost_usd": 0.005,
            "tokens": 5,
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


def _write_dataset(path: Path, *, split: str = "heldout") -> Path:
    samples = [
        {
            "id": sample_id,
            "item": "screw",
            "label": "scratch_neck" if defect else "good",
            "is_defect": defect,
            "split": split,
            "path": f"unused/{sample_id}.png",
        }
        for sample_id, defect in (("normal", False), ("defect", True))
    ]
    path.write_text(
        json.dumps({"samples": samples}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


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
    dataset_path = _write_dataset(tmp_path / "dataset.json")

    markdown, output_path = write_report(
        tmp_path,
        split="heldout",
        conditions=["B0", "B1", "A"],
        type_map_path=type_map_path,
        dataset_path=dataset_path,
    )

    assert output_path == tmp_path / "summary_heldout.md"
    assert output_path.read_text(encoding="utf-8") == markdown
    assert "B0 vs A" in markdown
    assert "B1 vs A" in markdown
    assert "사전 등록 H1·H2 주 검정" in markdown
    assert "balanced_accuracy" in markdown
    assert "macro_f1" in markdown
    assert "Δ 95% CI" in markdown
    assert "사전 등록 H4 신뢰도 플래그" in markdown
    assert "보류(그룹별 20건 미만)" in markdown
    assert "citation_fidelity" in markdown
    assert "screw" in markdown
    assert "사전등록 N" in markdown
    assert "Selective accuracy" in markdown
    assert "결론 금지" not in markdown


def test_dev_report_keeps_all_preregistered_hypotheses_descriptive(
    tmp_path: Path,
) -> None:
    rows = [
        _row(condition, sample_id, defect=defect)
        for condition in ("B0", "A")
        for sample_id, defect in (("normal", False), ("defect", True))
    ]
    for row in rows:
        row["split"] = "dev"
    _write_jsonl(tmp_path / "results.jsonl", rows)
    type_map_path = tmp_path / "type_map.json"
    type_map_path.write_text(json.dumps(TYPE_MAP, ensure_ascii=False), encoding="utf-8")

    markdown, _ = write_report(
        tmp_path,
        split="dev",
        conditions=["B0", "A"],
        type_map_path=type_map_path,
        dataset_path=_write_dataset(tmp_path / "dataset.json", split="dev"),
    )

    # H1·H2·H4 모두 dev에서는 수치만 보고하고 확정 판정하지 않는다.
    assert markdown.count("보류(dev 탐색)") == 3
    assert "효과 미입증" not in markdown


def test_report_identifies_malformed_jsonl_and_missing_condition(tmp_path: Path) -> None:
    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"condition": "A", "split": "heldout"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ResultDataError, match=r"broken\.jsonl:2"):
        load_result_rows(tmp_path)

    grouped = select_rows(
        [_row("A", "one", defect=False)],
        split="heldout",
        conditions=["A", "B1"],
    )
    assert grouped["B1"] == []


def test_report_rejects_nonstandard_json_numbers(tmp_path: Path) -> None:
    path = tmp_path / "nan.jsonl"
    path.write_text(
        '{"condition":"A","split":"heldout","usd":NaN}\n',
        encoding="utf-8",
    )

    with pytest.raises(ResultDataError, match="NaN"):
        load_result_rows(path)


def test_dataset_roster_is_validated_before_split_filter(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path / "dataset.json")
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    duplicate = dict(payload["samples"][0])
    duplicate["split"] = "dev"
    payload["samples"].append(duplicate)
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResultDataError, match="중복"):
        load_expected_rows(dataset_path, split="heldout")


def test_report_counts_partial_run_as_failure_and_forbids_conclusion(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "results.jsonl", [_row("A", "normal", defect=False)])
    type_map_path = tmp_path / "type_map.json"
    type_map_path.write_text(json.dumps(TYPE_MAP, ensure_ascii=False), encoding="utf-8")

    markdown, _ = write_report(
        tmp_path,
        split="heldout",
        conditions=["A"],
        type_map_path=type_map_path,
        dataset_path=_write_dataset(tmp_path / "dataset.json"),
    )

    assert "**결론 금지:**" in markdown
    assert "| A | 2 | 1 | 1 | 0 | 0 | 0 | 1 | 부분 | 0.500" in markdown


def test_report_rejects_direct_dry_run_file_duplicate_and_metadata_drift(tmp_path: Path) -> None:
    dry_run = tmp_path / "A_heldout_DRYRUN.jsonl"
    _write_jsonl(dry_run, [_row("A", "normal", defect=False)])
    with pytest.raises(ResultDataError, match="dry-run"):
        load_result_rows(dry_run)

    duplicate = [_row("A", "normal", defect=False)] * 2
    with pytest.raises(ResultDataError, match="중복"):
        select_rows(duplicate, split="heldout", conditions=["A"])

    drift = _row("A", "normal", defect=False)
    drift["item"] = "bottle"
    _write_jsonl(tmp_path / "actual.jsonl", [drift])
    type_map_path = tmp_path / "type_map.json"
    type_map_path.write_text(json.dumps(TYPE_MAP, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ResultDataError, match="item"):
        write_report(
            tmp_path / "actual.jsonl",
            split="heldout",
            conditions=["A"],
            type_map_path=type_map_path,
            dataset_path=_write_dataset(tmp_path / "dataset.json"),
        )


def test_report_handles_zero_rows_as_all_missing(tmp_path: Path) -> None:
    result_path = tmp_path / "empty.jsonl"
    result_path.write_text("", encoding="utf-8")
    type_map_path = tmp_path / "type_map.json"
    type_map_path.write_text(json.dumps(TYPE_MAP, ensure_ascii=False), encoding="utf-8")

    markdown, _ = write_report(
        result_path,
        split="heldout",
        conditions=["A"],
        type_map_path=type_map_path,
        dataset_path=_write_dataset(tmp_path / "dataset.json"),
    )

    assert "**결론 금지:**" in markdown
    assert "| A | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 부분 | 0.000" in markdown


def test_report_rejects_target_id_with_wrong_split_before_selection(tmp_path: Path) -> None:
    wrong_split = _row("A", "normal", defect=False)
    wrong_split["split"] = "dev"
    result_path = tmp_path / "wrong-split.jsonl"
    _write_jsonl(result_path, [wrong_split])
    type_map_path = tmp_path / "type_map.json"
    type_map_path.write_text(json.dumps(TYPE_MAP, ensure_ascii=False), encoding="utf-8")
    dataset_path = _write_dataset(tmp_path / "dataset.json")

    with pytest.raises(ResultDataError, match="split"):
        write_report(
            result_path,
            split="heldout",
            conditions=["A"],
            type_map_path=type_map_path,
            dataset_path=dataset_path,
        )

    unrelated = _row("A", "dev-only", defect=False)
    unrelated["split"] = "dev"
    _write_jsonl(result_path, [unrelated, _row("A", "normal", defect=False)])
    markdown, _ = write_report(
        result_path,
        split="heldout",
        conditions=["A"],
        type_map_path=type_map_path,
        dataset_path=dataset_path,
    )
    assert "**결론 금지:**" in markdown


def test_report_rejects_non_string_id_without_type_error(tmp_path: Path) -> None:
    malformed = _row("A", "normal", defect=False)
    malformed["id"] = []
    result_path = tmp_path / "malformed-id.jsonl"
    _write_jsonl(result_path, [malformed])
    type_map_path = tmp_path / "type_map.json"
    type_map_path.write_text(json.dumps(TYPE_MAP, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ResultDataError, match="문자열 id"):
        write_report(
            result_path,
            split="heldout",
            conditions=["A"],
            type_map_path=type_map_path,
            dataset_path=_write_dataset(tmp_path / "dataset.json"),
        )


def _expected_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "id": row["id"],
            "item": row["item"],
            "label": row["label"],
            "is_defect": row["is_defect"],
            "split": row["split"],
        }
        for row in rows
    ]


def _human_rows(count: int, *, split: str = "heldout") -> list[dict[str, object]]:
    rows = [
        _row("A", f"sample-{index:02d}", defect=bool(index % 2))
        for index in range(count)
    ]
    for row in rows:
        row["split"] = split
    return rows


def _score_blind_rows(
    blind_rows: list[dict[str, object]],
    mapping: dict[str, object],
    *,
    hallucinated_before: set[str] | None = None,
    hallucinated_after: set[str] | None = None,
    uncertain_after: set[str] | None = None,
) -> None:
    before = hallucinated_before or set()
    after = hallucinated_after or set()
    uncertain = uncertain_after or set()
    entries = mapping["entries"]
    assert isinstance(entries, dict)
    for row in blind_rows:
        entry = entries[row["id"]]
        result_id = entry["result_id"]
        stage = entry["stage"]
        claims = row["claims"]
        assert isinstance(claims, list)
        target = before if stage == "before_validate" else after
        for index, claim in enumerate(claims):
            if stage == "after_validate" and result_id in uncertain and index == 0:
                claim["judgment"] = "UNCERTAIN"
            elif result_id in target and index == 0:
                claim["judgment"] = "HALLUCINATED"
            else:
                claim["judgment"] = "SUPPORTED"
        row["stage_guess"] = "unknown"


def test_human_check_is_deterministic_paired_and_blind() -> None:
    rows = _human_rows(6)
    expected = _expected_rows(rows)

    first, first_mapping = create_blind_set(
        rows, expected_rows=expected, sample=4, seed=7
    )
    second, second_mapping = create_blind_set(
        rows, expected_rows=expected, sample=4, seed=7
    )

    assert first == second
    assert first_mapping == second_mapping
    assert len(first) == 8
    assert first_mapping["sampling"] == "proportional_stratified_item_truth"
    assert sum(first_mapping["stratum_counts"].values()) == 4
    entries = first_mapping["entries"]
    assert isinstance(entries, dict)
    by_result: dict[str, set[str]] = {}
    for entry in entries.values():
        by_result.setdefault(entry["result_id"], set()).add(entry["stage"])
    assert len(by_result) == 4
    assert all(stages == {"before_validate", "after_validate"} for stages in by_result.values())
    blind_text = json.dumps(first, ensure_ascii=False)
    assert "condition" not in blind_text
    assert "source_call_id" not in blind_text
    assert "pair_id" not in blind_text
    assert "result_id" not in blind_text
    assert "before_validate" not in blind_text
    assert "after_validate" not in blind_text


def test_human_check_sampling_preserves_item_truth_strata() -> None:
    rows: list[dict[str, object]] = []
    for item in ("screw", "bottle", "metal_nut", "hazelnut"):
        for is_defect in (False, True):
            for index in range(3):
                row = _row(
                    "A",
                    f"{item}-{'defect' if is_defect else 'normal'}-{index}",
                    defect=is_defect,
                )
                row["item"] = item
                rows.append(row)

    _, mapping = create_blind_set(
        rows,
        expected_rows=_expected_rows(rows),
        sample=8,
        seed=11,
    )

    assert mapping["stratum_counts"] == {
        "bottle/normal": 1,
        "bottle/defect": 1,
        "hazelnut/normal": 1,
        "hazelnut/defect": 1,
        "metal_nut/normal": 1,
        "metal_nut/defect": 1,
        "screw/normal": 1,
        "screw/defect": 1,
    }


def test_human_check_includes_abstain_and_unblinds_paired_scores() -> None:
    rows = _human_rows(2)
    rows[0]["verdict"] = "ABSTAIN"
    rows[0]["report"]["verdict"] = "ABSTAIN"
    blind, mapping = create_blind_set(
        rows, expected_rows=_expected_rows(rows), sample=2, seed=3
    )
    selected = mapping["selected_result_ids"]
    _score_blind_rows(
        blind,
        mapping,
        hallucinated_before={selected[0]},
        uncertain_after={selected[1]},
    )

    summary = unblind_scores(blind, mapping)

    assert summary["sample_pairs"] == 2
    assert summary["stages"]["before_validate"]["reports_with_hallucination"] == 1
    # UNCERTAIN도 보수적으로 환각으로 센다.
    assert summary["stages"]["after_validate"]["reports_with_hallucination"] == 1
    assert summary["paired"] == {
        "neither": 0,
        "before_only": 1,
        "after_only": 1,
        "both": 0,
        "discordant": 2,
        "mcnemar_exact_p": 1.0,
    }
    assert summary["conclusion_allowed"] is False
    assert summary["h3"]["observed_support"] is False


@pytest.mark.parametrize(
    "failure", ["missing", "error", "invalid_before", "invalid_after", "snapshot_tamper"]
)
def test_human_check_does_not_resample_unavailable_selected_rows(failure: str) -> None:
    rows = _human_rows(3)
    expected = _expected_rows(rows)
    if failure == "missing":
        rows.pop()
    elif failure == "error":
        rows[-1]["error"] = "RuntimeError: failed"
    elif failure == "invalid_before":
        rows[-1]["pre_validation_snapshot"] = None
    elif failure == "snapshot_tamper":
        rows[-1]["pre_validation_snapshot"]["report"]["summary"] = "hash 이후 변조"
    else:
        rows[-1]["report"] = None

    expected_message = "hash가 일치" if failure == "snapshot_tamper" else "재표집하지 않음"
    with pytest.raises(HumanCheckError, match=expected_message):
        create_blind_set(rows, expected_rows=expected, sample=3, seed=11)


def test_human_check_detects_payload_tampering() -> None:
    rows = _human_rows(1)
    blind, mapping = create_blind_set(
        rows, expected_rows=_expected_rows(rows), sample=1, seed=2
    )
    _score_blind_rows(blind, mapping)
    blind[0]["claims"][0]["text"] = "생성 뒤 바꾼 문장"

    with pytest.raises(HumanCheckError, match="변조"):
        unblind_scores(blind, mapping)


def test_human_check_rejects_duplicate_result_id_and_invalid_stage_guess() -> None:
    rows = _human_rows(2)
    duplicate = [rows[0], dict(rows[0])]
    with pytest.raises(HumanCheckError, match="중복 샘플 id"):
        create_blind_set(
            duplicate,
            expected_rows=_expected_rows(rows),
            sample=2,
            seed=2,
        )

    blind, mapping = create_blind_set(
        rows, expected_rows=_expected_rows(rows), sample=2, seed=2
    )
    _score_blind_rows(blind, mapping)
    blind[0]["stage_guess"] = "A"
    with pytest.raises(HumanCheckError, match="stage_guess"):
        unblind_scores(blind, mapping)


@pytest.mark.parametrize("invalid", [None, "MAYBE"])
def test_human_check_requires_every_claim_judgment(invalid: object) -> None:
    rows = _human_rows(1)
    blind, mapping = create_blind_set(
        rows, expected_rows=_expected_rows(rows), sample=1, seed=2
    )
    _score_blind_rows(blind, mapping)
    blind[0]["claims"][0]["judgment"] = invalid

    with pytest.raises(HumanCheckError, match="judgment"):
        unblind_scores(blind, mapping)


def test_human_check_rejects_incomplete_duplicate_and_invalid_pair_mapping() -> None:
    rows = _human_rows(2)
    blind, mapping = create_blind_set(
        rows, expected_rows=_expected_rows(rows), sample=2, seed=2
    )
    _score_blind_rows(blind, mapping)

    with pytest.raises(HumanCheckError, match="정확히 2N"):
        unblind_scores(blind[:-1], mapping)

    duplicate = [dict(row) for row in blind]
    duplicate[-1] = duplicate[0]
    with pytest.raises(HumanCheckError, match="중복 anonymous_id"):
        unblind_scores(duplicate, mapping)

    broken_mapping = json.loads(json.dumps(mapping))
    first_pair = next(iter(broken_mapping["entries"].values()))["pair_id"]
    same_pair = [
        value
        for value in broken_mapping["entries"].values()
        if value["pair_id"] == first_pair
    ]
    same_pair[1]["stage"] = same_pair[0]["stage"]
    with pytest.raises(HumanCheckError, match="validate 전·후"):
        unblind_scores(blind, broken_mapping)


@pytest.mark.parametrize(
    ("after_count", "expected_rate", "threshold_met", "expected_p"),
    [(0, 0.0, True, 0.015625), (1, 0.05, True, 0.03125), (2, 0.10, False, 0.0625)],
)
def test_human_check_five_percent_boundary(
    after_count: int,
    expected_rate: float,
    threshold_met: bool,
    expected_p: float,
) -> None:
    rows = _human_rows(20)
    blind, mapping = create_blind_set(
        rows, expected_rows=_expected_rows(rows), sample=20, seed=DEFAULT_SEED
    )
    selected = mapping["selected_result_ids"]
    _score_blind_rows(
        blind,
        mapping,
        hallucinated_before=set(selected[:7]),
        hallucinated_after=set(selected[:after_count]),
    )

    summary = unblind_scores(blind, mapping)
    after = summary["stages"]["after_validate"]

    assert after["report_hallucination_rate"] == expected_rate
    assert summary["h3"]["observed_after_at_most_5pct"] is threshold_met
    assert summary["conclusion_allowed"] is True
    assert 0.0 <= after["wilson_95_low"] <= after["wilson_95_high"] <= 1.0
    assert summary["paired"]["before_only"] >= 5
    assert summary["paired"]["mcnemar_exact_p"] == pytest.approx(expected_p)


def test_human_check_conclusion_requires_exact_preregistered_protocol() -> None:
    rows = _human_rows(20)
    blind, mapping = create_blind_set(
        rows,
        expected_rows=_expected_rows(rows),
        sample=20,
        seed=DEFAULT_SEED,
    )
    _score_blind_rows(blind, mapping)
    assert unblind_scores(blind, mapping)["conclusion_allowed"] is True

    dev_mapping = json.loads(json.dumps(mapping))
    dev_mapping["split"] = "dev"
    dev_summary = unblind_scores(blind, dev_mapping)
    assert dev_summary["conclusion_allowed"] is False
    assert "기술 통계" in dev_summary["conclusion_scope"]

    arbitrary_seed = json.loads(json.dumps(mapping))
    arbitrary_seed["seed"] = DEFAULT_SEED + 1
    assert unblind_scores(blind, arbitrary_seed)["conclusion_allowed"] is False

    blind_19, mapping_19 = create_blind_set(
        rows,
        expected_rows=_expected_rows(rows),
        sample=19,
        seed=DEFAULT_SEED,
    )
    _score_blind_rows(blind_19, mapping_19)
    assert unblind_scores(blind_19, mapping_19)["conclusion_allowed"] is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("split", "train", "split"), ("seed", "20261020", "seed")],
)
def test_human_check_validates_mapping_protocol_fields(
    field: str, value: object, message: str
) -> None:
    rows = _human_rows(1)
    blind, mapping = create_blind_set(
        rows, expected_rows=_expected_rows(rows), sample=1, seed=2
    )
    _score_blind_rows(blind, mapping)
    mapping[field] = value

    with pytest.raises(HumanCheckError, match=message):
        unblind_scores(blind, mapping)


@pytest.mark.parametrize("override", [["--sample", "19"], ["--seed", "7"]])
def test_human_check_cli_rejects_heldout_protocol_override(
    override: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        human_check_main(["--results", str(tmp_path), "--split", "heldout", *override])

    assert exc_info.value.code == 2
    assert "heldout 블라인드 프로토콜" in capsys.readouterr().err


def test_human_check_writes_private_paired_mapping(tmp_path: Path) -> None:
    rows = [_row("A", sample_id, defect=defect) for sample_id, defect in (("normal", False), ("defect", True))]
    for row in rows:
        row["split"] = "dev"
    _write_jsonl(tmp_path / "results.jsonl", rows)
    dataset_path = _write_dataset(tmp_path / "dataset.json", split="dev")

    blind_path, mapping_path, count = write_blind_set(
        tmp_path, dataset_path=dataset_path, sample=2, seed=3, split="dev"
    )

    assert count == 4
    assert stat.S_IMODE(mapping_path.stat().st_mode) == 0o600
    blind_text = blind_path.read_text(encoding="utf-8")
    mapping_text = mapping_path.read_text(encoding="utf-8")
    assert "condition" not in blind_text
    assert "pair_id" not in blind_text
    assert "before_validate" not in blind_text
    assert "pair_id" in mapping_text
    assert "before_validate" in mapping_text
