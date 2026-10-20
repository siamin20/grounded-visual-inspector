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
            "path": "data/mvtec/screw/test/good/001.png",
            "item": "screw",
            "label": "good",
            "is_defect": False,
            "split": "dev",
        },
        {
            "id": "screw-bad-001",
            "path": "data/mvtec/screw/test/scratch_head/001.png",
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
                "verdict": "REJECT",
                "defect_type": "scratch_head",
                "confidence": "high",
                "observations": [
                    {"text": "머리 표면에 긴 긁힘이 관찰됩니다."}
                ],
                "report": {
                    "verdict": "REJECT",
                    "defect_type": "scratch_head",
                    "confidence": "high",
                    "observations": [],
                },
                "validation_attempts": 1,
                "tokens": 42,
                "cost_usd": 0.01,
                "events": [],
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
    )

    result_path, summary_path, summary = await run_eval._run(args)

    rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    assert len(rows) == 2
    assert list(rows[0]) == list(run_eval.REQUIRED_ROW_KEYS)
    assert rows[0]["error"] == "RuntimeError: sample failed"
    assert rows[1]["error"] is None
    assert rows[1]["tokens"] == 42
    assert rows[1]["langsmith_run_id"] == "run-2"
    assert summary["processed_samples"] == 2
    assert summary["error_count"] == 1
    assert summary_path.is_file()
