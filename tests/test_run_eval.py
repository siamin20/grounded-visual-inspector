from __future__ import annotations

from argparse import Namespace
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace

import pytest

from eval import run_eval


def _sample(sample_id: str, *, defect: bool = False) -> dict:
    return {
        "id": sample_id,
        "path": f"unused/{sample_id}.png",
        "item": "screw",
        "label": "scratch_neck" if defect else "good",
        "is_defect": defect,
        "split": "dev",
    }


def _args(tmp_path, *, run_id: str, condition: str = "A") -> Namespace:
    return Namespace(
        condition=condition,
        split="dev",
        items=None,
        limit=None,
        out=str(tmp_path / "results"),
        dry_run=True,
        push_dataset=False,
        final_run=False,
        freeze_manifest=None,
        run_id=run_id,
        suite_conditions=condition,
        _allow_test_output=True,
    )


def _known_setup_usage(*, estimated_usd: float = 0.002) -> dict:
    return {
        "tokens": 10,
        "input_tokens": 10,
        "output_tokens": 0,
        "token_accounting": "provider_split",
        "estimated_usd": estimated_usd,
        "compute_proxy_usd": 0.0,
        "successful_provider_requests": 1,
        "successful_vlm_requests": 0,
        "unknown_token_usage_successes": 0,
        "unobserved_model_successes": 0,
        "observed_model_response_count": 1,
        "observed_model_ids": ["openai:embed-resolved"],
        "latency_s": 0.01,
        "usage_status": "known_split",
    }


def _real_success_result(*, estimated_usd: float = 0.003) -> dict:
    return {
        "route": "inspection",
        "verdict": "PASS",
        "defect_type": None,
        "confidence": "high",
        "observations": [{"text": "이상 없음", "model": "openai:observer-v1"}],
        "cross_check": {},
        "report": {"verdict": "PASS"},
        "validation_attempts": 1,
        "retry_count": 0,
        "replan_rounds": 0,
        "tokens": 10,
        "input_tokens": 7,
        "output_tokens": 3,
        "token_accounting": "provider_split",
        "estimated_usd": estimated_usd,
        "compute_proxy_usd": 0.0,
        "observed_model_ids": ["openai:observer-v1"],
        "successful_provider_requests": 1,
        "successful_vlm_requests": 1,
        "unknown_token_usage_successes": 0,
        "unobserved_model_successes": 0,
        "observed_model_response_count": 1,
        "cost_usd": estimated_usd,
        "events": [],
    }


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


def test_real_success_without_token_usage_is_not_reported_as_zero_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_BACKEND", "openai")
    monkeypatch.setenv("VLM_BACKEND", "openai")
    monkeypatch.setenv("EMBED_BACKEND", "openai")

    usage = run_eval._usage_fields({
        "tokens": 0,
        "input_tokens": None,
        "output_tokens": None,
        "token_accounting": "total_only",
        "estimated_usd": 0.0,
        "compute_proxy_usd": 0.0,
        "successful_provider_requests": 1,
        "successful_vlm_requests": 1,
    })

    assert usage["usage_status"] == "unknown_tokens_after_success"


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
        yield (
            SimpleNamespace(version="v2", ablation=""),
            object(),
            run_eval._no_setup_usage(),
        )

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
                    {
                        "text": "머리 표면에 긴 긁힘이 관찰됩니다.",
                        "model": "fake-observer",
                    }
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
                "input_tokens": 21,
                "output_tokens": 21,
                "token_accounting": "estimated_split",
                "estimated_usd": 0.0,
                "compute_proxy_usd": 0.0,
                "observed_model_ids": ["fake-observer"],
                "successful_provider_requests": 0,
                "successful_vlm_requests": 0,
                "cost_usd": 0.0,
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
        run_id="test-error-continue",
        suite_conditions="A",
        _allow_test_output=True,
    )

    result_path, summary_path, summary = await run_eval._run(args)

    rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    assert len(rows) == 2
    assert list(rows[0]) == list(run_eval.REQUIRED_ROW_KEYS)
    assert rows[0]["error"] == "RuntimeError: sample failed"
    assert rows[1]["error"] is None
    assert rows[1]["tokens"] == 42
    assert rows[1]["observed_model_ids"] == ["fake-observer"]
    assert rows[1]["estimated_usd"] == 0.0
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
async def test_zero_budget_stops_before_any_side_effect(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("LLM_BACKEND", "fake")
    monkeypatch.setenv("VLM_BACKEND", "fake")
    monkeypatch.setenv("EMBED_BACKEND", "fake")
    monkeypatch.setenv("EVAL_SUITE_BUDGET_USD", "0")

    def unexpected_load(**_kwargs):
        raise AssertionError("0 USD guard 후에 데이터를 읽으면 안 됩니다.")

    monkeypatch.setattr(run_eval, "_load_samples", unexpected_load)
    output_root = tmp_path / "results"
    args = Namespace(
        condition="A",
        split="dev",
        items=None,
        limit=None,
        out=str(output_root),
        dry_run=True,
        push_dataset=False,
        final_run=False,
        freeze_manifest=None,
        run_id="zero-budget",
        suite_conditions="A",
        _allow_test_output=True,
    )

    with pytest.raises(ValueError, match="0 USD"):
        await run_eval._run(args)

    assert not output_root.exists()


@pytest.mark.asyncio
async def test_after_suite_rejects_fake_embed_before_side_effect(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("LLM_BACKEND", "openai")
    monkeypatch.setenv("VLM_BACKEND", "openai")
    monkeypatch.setenv("EMBED_BACKEND", "fake")

    def unexpected_load(**_kwargs):
        raise AssertionError("backend guard 후에 데이터를 읽으면 안 됩니다.")

    monkeypatch.setattr(run_eval, "_load_samples", unexpected_load)
    output_root = tmp_path / "results"
    args = Namespace(
        condition="B0",
        split="dev",
        items=None,
        limit=None,
        out=str(output_root),
        dry_run=False,
        push_dataset=False,
        final_run=False,
        freeze_manifest=None,
        run_id="fake-embed-suite",
        suite_conditions="B0,A",
        _allow_test_output=True,
    )

    with pytest.raises(SystemExit, match="fake embedding"):
        await run_eval._run(args)

    assert not output_root.exists()


@pytest.mark.parametrize("value", ["-1", "NaN", "Infinity", "-Infinity", "text"])
def test_suite_budget_rejects_invalid_values(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVAL_SUITE_BUDGET_USD", value)

    with pytest.raises(ValueError, match="0 이상|0이상"):
        run_eval._suite_budget()


@pytest.mark.asyncio
async def test_soft_budget_stop_keeps_missing_sample_in_denominator(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    samples = [_sample("normal"), _sample("defect", defect=True)]
    monkeypatch.setenv("LLM_BACKEND", "openai")
    monkeypatch.setenv("VLM_BACKEND", "openai")
    monkeypatch.setenv("EMBED_BACKEND", "openai")
    monkeypatch.setenv("EVAL_SUITE_BUDGET_USD", "0.005")
    monkeypatch.setattr(run_eval, "_load_samples", lambda **_kwargs: samples)

    @asynccontextmanager
    async def fake_runtime(_condition: str):
        yield (
            SimpleNamespace(version="v2", ablation=""),
            object(),
            run_eval._no_setup_usage(),
        )

    async def fake_evaluate(_sample_value, _condition, _deps, _graph):
        return (
            {
                "route": "inspection",
                "verdict": "PASS",
                "defect_type": None,
                "confidence": "high",
                "observations": [
                    {"text": "이상 없음", "model": "openai:observer-v1"}
                ],
                "cross_check": {},
                "report": {"verdict": "PASS"},
                "validation_attempts": 1,
                "retry_count": 1,
                "replan_rounds": 0,
                "tokens": 10,
                "input_tokens": 7,
                "output_tokens": 3,
                "token_accounting": "provider_split",
                "estimated_usd": 0.01,
                "compute_proxy_usd": 0.0,
                "observed_model_ids": ["openai:observer-v1"],
                "successful_provider_requests": 1,
                "successful_vlm_requests": 1,
                "cost_usd": 0.01,
                "events": [
                    {"type": "Validation", "passed": False, "errors": ["x"]}
                ],
            },
            "trace-1",
        )

    monkeypatch.setattr(run_eval, "_runtime", fake_runtime)
    monkeypatch.setattr(run_eval, "_evaluate_sample", fake_evaluate)

    result_path, _, summary = await run_eval._run(
        _args(tmp_path, run_id="soft-budget")
    )

    rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["validation_rejections"] == 1
    assert rows[0]["successful_vlm_requests"] == 1
    assert rows[0]["retry_count"] == 1
    assert summary["processed_samples"] == 1
    assert summary["missing_samples"] == 1
    assert summary["complete_capture"] is False
    assert summary["budget_exceeded"] is True
    assert summary["suite_estimated_usd"] == 0.01
    assert summary["binary_metrics"]["expected_n"] == 2
    assert summary["binary_metrics"]["observed_n"] == 1
    assert summary["binary_metrics"]["accuracy"] == 0.5
    manifest = run_eval.load_run_manifest(result_path.parent)
    assert manifest["status"] == "incomplete"
    assert manifest["conditions"]["A"]["status"] == "incomplete"


@pytest.mark.asyncio
async def test_setup_usage_is_added_once_to_condition_and_suite_totals(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("LLM_BACKEND", "openai")
    monkeypatch.setenv("VLM_BACKEND", "openai")
    monkeypatch.setenv("EMBED_BACKEND", "openai")
    monkeypatch.setenv("EVAL_SUITE_BUDGET_USD", "1")
    monkeypatch.setattr(run_eval, "_load_samples", lambda **_kwargs: [_sample("one")])

    @asynccontextmanager
    async def fake_runtime(_condition: str):
        yield SimpleNamespace(version="v2", ablation=""), object(), _known_setup_usage()

    async def fake_evaluate(*_args):
        return _real_success_result(), "trace-one"

    monkeypatch.setattr(run_eval, "_runtime", fake_runtime)
    monkeypatch.setattr(run_eval, "_evaluate_sample", fake_evaluate)

    result_path, _, summary = await run_eval._run(
        _args(tmp_path, run_id="setup-accounting")
    )

    assert summary["condition_estimated_usd"] == pytest.approx(0.005)
    assert summary["suite_estimated_usd"] == pytest.approx(0.005)
    manifest = run_eval.load_run_manifest(result_path.parent)
    record = manifest["conditions"]["A"]
    assert record["setup_usage"]["estimated_usd"] == pytest.approx(0.002)
    assert record["estimated_spend_usd"] == pytest.approx(0.005)


@pytest.mark.asyncio
async def test_setup_budget_limit_stops_before_first_sample(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("LLM_BACKEND", "openai")
    monkeypatch.setenv("VLM_BACKEND", "openai")
    monkeypatch.setenv("EMBED_BACKEND", "openai")
    monkeypatch.setenv("EVAL_SUITE_BUDGET_USD", "0.002")
    monkeypatch.setattr(run_eval, "_load_samples", lambda **_kwargs: [_sample("one")])

    @asynccontextmanager
    async def fake_runtime(_condition: str):
        yield SimpleNamespace(version="v2", ablation=""), object(), _known_setup_usage()

    async def unexpected_evaluate(*_args):
        raise AssertionError("setup이 예산을 소진한 뒤 샘플을 호출하면 안 됩니다.")

    monkeypatch.setattr(run_eval, "_runtime", fake_runtime)
    monkeypatch.setattr(run_eval, "_evaluate_sample", unexpected_evaluate)

    result_path, _, summary = await run_eval._run(
        _args(tmp_path, run_id="setup-budget-stop")
    )

    assert result_path.read_text(encoding="utf-8") == ""
    assert summary["processed_samples"] == 0
    assert summary["missing_samples"] == 1
    assert summary["budget_exceeded"] is True
    assert summary["complete_capture"] is False
    assert summary["condition_estimated_usd"] == pytest.approx(0.002)


@pytest.mark.asyncio
@pytest.mark.parametrize("known_usage", [True, False])
async def test_setup_failure_records_usage_before_failing_condition(
    known_usage: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("LLM_BACKEND", "openai")
    monkeypatch.setenv("VLM_BACKEND", "openai")
    monkeypatch.setenv("EMBED_BACKEND", "openai")
    monkeypatch.setenv("EVAL_SUITE_BUDGET_USD", "1")
    monkeypatch.setattr(run_eval, "_load_samples", lambda **_kwargs: [_sample("one")])
    setup = (
        _known_setup_usage()
        if known_usage
        else run_eval._unknown_setup_error(latency_s=0.01)
    )

    @asynccontextmanager
    async def failing_runtime(_condition: str):
        raise run_eval.SetupUsageError("index failed", setup_usage=setup)
        yield  # pragma: no cover

    monkeypatch.setattr(run_eval, "_runtime", failing_runtime)

    with pytest.raises(run_eval.SetupUsageError, match="index failed"):
        await run_eval._run(_args(tmp_path, run_id=f"setup-failed-{known_usage}"))

    manifest = run_eval.load_run_manifest(
        tmp_path / "results" / f"setup-failed-{known_usage}"
    )
    assert manifest["status"] == "failed"
    record = manifest["conditions"]["A"]
    assert record["status"] == "failed"
    assert record["setup_usage"]["usage_status"] == (
        "known_split" if known_usage else "unknown_after_error"
    )
    assert record["estimated_spend_usd"] == pytest.approx(
        0.002 if known_usage else 0.0
    )


@pytest.mark.asyncio
async def test_last_sample_budget_overshoot_is_disclosed_without_hiding_capture(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("LLM_BACKEND", "openai")
    monkeypatch.setenv("VLM_BACKEND", "openai")
    monkeypatch.setenv("EMBED_BACKEND", "openai")
    monkeypatch.setenv("EVAL_SUITE_BUDGET_USD", "0.005")
    monkeypatch.setattr(run_eval, "_load_samples", lambda **_kwargs: [_sample("one")])

    @asynccontextmanager
    async def fake_runtime(_condition: str):
        yield SimpleNamespace(version="v2", ablation=""), object(), run_eval._no_setup_usage()

    async def fake_evaluate(*_args):
        return _real_success_result(estimated_usd=0.01), "trace-one"

    monkeypatch.setattr(run_eval, "_runtime", fake_runtime)
    monkeypatch.setattr(run_eval, "_evaluate_sample", fake_evaluate)

    result_path, _, summary = await run_eval._run(
        _args(tmp_path, run_id="last-sample-overshoot")
    )

    assert summary["processed_samples"] == 1
    assert summary["complete_capture"] is True
    assert summary["budget_exceeded"] is True
    assert run_eval.load_run_manifest(result_path.parent)["status"] == "complete"


@pytest.mark.asyncio
async def test_same_run_condition_cannot_be_overwritten(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("LLM_BACKEND", "fake")
    monkeypatch.setenv("VLM_BACKEND", "fake")
    monkeypatch.setenv("EMBED_BACKEND", "fake")
    monkeypatch.setenv("EVAL_SUITE_BUDGET_USD", "15")
    monkeypatch.setattr(
        run_eval, "_load_samples", lambda **_kwargs: [_sample("only")]
    )

    @asynccontextmanager
    async def fake_runtime(_condition: str):
        yield (
            SimpleNamespace(version="v2", ablation=""),
            object(),
            run_eval._no_setup_usage(),
        )

    calls = 0

    async def fake_evaluate(_sample_value, _condition, _deps, _graph):
        nonlocal calls
        calls += 1
        return (
            {
                "route": "inspection",
                "verdict": "PASS",
                "confidence": "high",
                "observations": [
                    {"text": "이상 없음", "model": "fake-observer"}
                ],
                "report": {"verdict": "PASS"},
                "tokens": 1,
                "input_tokens": 1,
                "output_tokens": 0,
                "token_accounting": "estimated_split",
                "estimated_usd": 0.0,
                "compute_proxy_usd": 0.0,
                "observed_model_ids": ["fake-observer"],
                "successful_provider_requests": 0,
                "successful_vlm_requests": 0,
                "cost_usd": 0.0,
                "events": [],
            },
            "trace-only",
        )

    monkeypatch.setattr(run_eval, "_runtime", fake_runtime)
    monkeypatch.setattr(run_eval, "_evaluate_sample", fake_evaluate)
    args = _args(tmp_path, run_id="no-overwrite")

    await run_eval._run(args)
    with pytest.raises(run_eval.ProvenanceError, match="이미 시작|완료"):
        await run_eval._run(args)

    assert calls == 1


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
        _allow_test_output=True,
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
        _allow_test_output=True,
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


def test_trace_inputs_do_not_expose_ground_truth_or_labelled_paths() -> None:
    traced = run_eval._trace_inputs({
        "sample": {
            "id": "screw-good-000",
            "item": "screw",
            "split": "heldout",
            "path": "data/mvtec/screw/test/good/000.png",
            "label": "good",
            "is_defect": False,
        },
        "condition": "B0",
        "deps": object(),
        "graph": object(),
    })

    assert traced == {
        "sample": {"item": "screw", "split": "heldout"},
        "condition": "B0",
    }
    rendered = json.dumps(traced, ensure_ascii=False)
    assert "screw-good-000" not in rendered
    assert "000.png" not in rendered
    assert '"label"' not in rendered
    assert "is_defect" not in rendered
