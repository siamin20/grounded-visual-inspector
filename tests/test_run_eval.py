from __future__ import annotations

from argparse import Namespace
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace

import pytest

from eval import run_eval


def test_fake_backends_require_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BACKEND", "fake")
    monkeypatch.setenv("VLM_BACKEND", "openai")
    monkeypatch.setenv("EMBED_BACKEND", "openai")

    with pytest.raises(SystemExit, match="fake LLM/VLM"):
        run_eval._guard_backends(dry_run=False)

    run_eval._guard_backends(dry_run=True)

    monkeypatch.setenv("LLM_BACKEND", "typo")
    with pytest.raises(SystemExit, match="지원하지 않는 평가 백엔드"):
        run_eval._guard_backends(dry_run=True)


def test_dry_run_output_path_is_separate(tmp_path) -> None:
    result, summary = run_eval._output_paths(
        tmp_path, "A", "dev", dry_run=True
    )

    assert result.name == "A_dev_DRYRUN.jsonl"
    assert summary.name == "summary_A_dev_DRYRUN.json"


def test_llm_backend_name_is_normalized_and_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_BACKEND", " FAKE ")
    backend, _, _ = run_eval._build_llms()
    assert backend == "fake"

    monkeypatch.setenv("LLM_BACKEND", "typo")
    with pytest.raises(ValueError, match="지원하지 않는 LLM_BACKEND"):
        run_eval._build_llms()


@pytest.mark.asyncio
async def test_run_records_sample_error_and_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    samples = [
        {
            "id": "screw-good-001",
            "path": "data/mvtec/screw/test/good/004.png",
            "item": "screw",
            "label": "good",
            "is_defect": False,
            "split": "dev",
        },
        {
            "id": "screw-bad-001",
            "path": "data/mvtec/screw/test/scratch_head/004.png",
            "item": "screw",
            "label": "scratch_head",
            "is_defect": True,
            "split": "dev",
        },
    ]
    monkeypatch.setenv("LLM_BACKEND", "fake")
    monkeypatch.setenv("VLM_BACKEND", "fake")
    monkeypatch.setenv("EVAL_BUDGET_USD", "15")
    monkeypatch.setattr(run_eval, "_load_samples", lambda **_kwargs: samples)

    @asynccontextmanager
    async def fake_runtime(_condition: str):
        yield SimpleNamespace(version="v2", ablation=""), object()

    calls = 0

    async def fake_evaluate(sample, _condition, _deps, _graph):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("sample failed")
        return (
            {
                "verdict": "ABSTAIN",
                "defect_type": None,
                "confidence": "low",
                "abstain_reason": "insufficient_visual_evidence",
                "observations": [
                    {"text": "머리 표면에 긴 긁힘이 관찰됩니다."}
                ],
                "report": {
                    "verdict": "ABSTAIN",
                    "defect_type": None,
                    "confidence": "low",
                    "abstain_reason": "insufficient_visual_evidence",
                    "observations": [],
                },
                "pre_validation_snapshot": {
                    "version": 1,
                    "stage": "before_validate",
                    "report": {"summary": "첫 초안"},
                    "report_sha256": "a" * 64,
                    "cost_usd": 0.005,
                    "tokens": 21,
                },
                "validation_attempts": 1,
                "tokens": 42,
                "cost_usd": 0.01,
                "approval": {"required": True},
                "events": [{"type": "ApprovalRequired", "required": True}],
            },
            "run-2",
        )

    monkeypatch.setattr(run_eval, "_runtime", fake_runtime)
    monkeypatch.setattr(run_eval, "_evaluate_sample", fake_evaluate)
    args = Namespace(
        condition="A",
        split="dev",
        items=None,
        limit=2,
        out=str(tmp_path),
        dry_run=True,
        push_dataset=False,
        final_run=False,
        freeze_manifest=None,
    )

    result_path, summary_path, summary = await run_eval._run(args)

    rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    assert len(rows) == 2
    assert list(rows[0]) == list(run_eval.REQUIRED_ROW_KEYS)
    assert rows[0]["error"] == "RuntimeError: sample failed"
    assert rows[1]["error"] is None
    assert rows[1]["tokens"] == 42
    assert rows[1]["verdict"] == "ABSTAIN"
    assert rows[1]["abstain_reason"] == "insufficient_visual_evidence"
    assert rows[1]["approval_required"] is True
    assert rows[1]["pre_validation_snapshot"]["report"]["summary"] == "첫 초안"
    assert rows[1]["langsmith_run_id"] == "run-2"
    assert summary["processed_samples"] == 2
    assert summary["missing_samples"] == 0
    assert summary["complete_capture"] is True
    assert summary["error_count"] == 1
    assert summary["binary_metrics"]["n"] == 2
    assert summary["binary_metrics"]["accuracy"] == 0.0
    assert summary_path.is_file()


@pytest.mark.asyncio
async def test_budget_stop_keeps_unprocessed_samples_in_summary_denominator(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    samples = [
        {
            "id": "normal",
            "path": "unused/normal.png",
            "item": "screw",
            "label": "good",
            "is_defect": False,
            "split": "dev",
        },
        {
            "id": "defect",
            "path": "unused/defect.png",
            "item": "screw",
            "label": "scratch_neck",
            "is_defect": True,
            "split": "dev",
        },
    ]
    monkeypatch.setenv("LLM_BACKEND", "fake")
    monkeypatch.setenv("VLM_BACKEND", "fake")
    monkeypatch.setenv("EVAL_BUDGET_USD", "0")
    monkeypatch.setattr(run_eval, "_load_samples", lambda **_kwargs: samples)

    @asynccontextmanager
    async def fake_runtime(_condition: str):
        yield SimpleNamespace(version="v2", ablation=""), object()

    async def fake_evaluate(_sample, _condition, _deps, _graph):
        return (
            {
                "verdict": "PASS",
                "defect_type": None,
                "confidence": "high",
                "report": {},
                "tokens": 1,
                "cost_usd": 0.01,
                "events": [],
            },
            "run-1",
        )

    monkeypatch.setattr(run_eval, "_runtime", fake_runtime)
    monkeypatch.setattr(run_eval, "_evaluate_sample", fake_evaluate)
    args = Namespace(
        condition="A",
        split="dev",
        items=None,
        limit=None,
        out=str(tmp_path),
        dry_run=True,
        push_dataset=False,
        final_run=False,
        freeze_manifest=None,
    )

    _, _, summary = await run_eval._run(args)

    assert summary["processed_samples"] == 1
    assert summary["missing_samples"] == 1
    assert summary["complete_capture"] is False
    assert summary["budget_exceeded"] is True
    assert summary["binary_metrics"]["expected_n"] == 2
    assert summary["binary_metrics"]["observed_n"] == 1
    assert summary["binary_metrics"]["accuracy"] == 0.5


@pytest.mark.asyncio
async def test_invalid_heldout_fails_before_side_effects(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("LLM_BACKEND", "openai")
    monkeypatch.setenv("VLM_BACKEND", "openai")
    monkeypatch.setenv("EMBED_BACKEND", "openai")

    def unexpected_load(**_kwargs):
        raise AssertionError("preflight 실패 후 샘플을 읽으면 안 됩니다.")

    monkeypatch.setattr(run_eval, "_load_samples", unexpected_load)
    args = Namespace(
        condition="A",
        split="heldout",
        items=None,
        limit=None,
        out=str(tmp_path / "results"),
        dry_run=False,
        push_dataset=False,
        final_run=False,
        freeze_manifest=None,
    )

    with pytest.raises(SystemExit, match="--final-run"):
        await run_eval._run(args)

    assert not (tmp_path / "results").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("condition", ["A-X", "A-C", "A-V"])
async def test_heldout_rejects_exploratory_ablations_before_side_effects(
    condition: str, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("LLM_BACKEND", "openai")
    monkeypatch.setenv("VLM_BACKEND", "openai")
    monkeypatch.setenv("EMBED_BACKEND", "openai")

    def unexpected_load(**_kwargs):
        raise AssertionError("heldout ablation 거부 뒤 샘플을 읽으면 안 됩니다.")

    monkeypatch.setattr(run_eval, "_load_samples", unexpected_load)
    args = Namespace(
        condition=condition,
        split="heldout",
        items=None,
        limit=None,
        out=str(tmp_path / "results"),
        dry_run=False,
        push_dataset=False,
        final_run=True,
        freeze_manifest="unused-freeze-manifest.json",
    )

    with pytest.raises(SystemExit, match="ablation\uc740 dev \uc804\uc6a9"):
        await run_eval._run(args)

    assert not (tmp_path / "results").exists()


def test_parser_exposes_heldout_confirmation_flags() -> None:
    args = run_eval._parser().parse_args([
        "--condition",
        "A",
        "--split",
        "heldout",
        "--final-run",
        "--freeze-manifest",
        "eval/freeze_manifest.json",
    ])

    assert args.final_run is True
    assert args.freeze_manifest == "eval/freeze_manifest.json"
