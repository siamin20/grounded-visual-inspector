from __future__ import annotations

import pytest

from agent.nodes.common import (
    _cost,
    _merge_usages,
    _tokens,
    _usage_updates,
    _vlm_result_usage,
)
from agent.usage import (
    UsageValueError,
    nonnegative_finite_float,
    nonnegative_int,
)
from eval.evaluators import EvaluationDataError, cost_latency


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        "1",
        None,
        -1,
        10**1000,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_nonnegative_finite_float_rejects_invalid_values(value: object) -> None:
    with pytest.raises(UsageValueError):
        nonnegative_finite_float(value, name="test")


@pytest.mark.parametrize("value", [True, False, "1", None, -1, 1.0])
def test_nonnegative_int_rejects_invalid_values(value: object) -> None:
    with pytest.raises(UsageValueError):
        nonnegative_int(value, name="test")


def test_numeric_parsers_only_default_missing_values() -> None:
    assert nonnegative_finite_float(default=0.25) == 0.25
    assert nonnegative_int(default=3) == 3
    with pytest.raises(UsageValueError, match="누락"):
        nonnegative_finite_float()
    with pytest.raises(UsageValueError):
        nonnegative_finite_float(None, default=0.0)


def test_cost_and_tokens_validate_state_and_increment() -> None:
    state = {"cost_usd": 0.1, "tokens": 10}
    assert _cost(state, {"estimated_usd": 0.2, "usd": 99.0}) == 0.3
    assert _cost(state, {"usd": 0.2}) == 0.3
    assert _tokens(state, {"tokens": 5}) == 15
    assert _cost({}, None) == 0.0
    assert _tokens({}, None) == 0


def test_usage_updates_preserve_split_cost_basis_and_observed_models() -> None:
    first = _usage_updates({"successful_vlm_requests": 2}, {
        "tokens": 13,
        "input_tokens": 10,
        "output_tokens": 3,
        "token_accounting": "provider_split",
        "estimated_usd": 0.1,
        "compute_proxy_usd": 0.0,
        "requested_model": "requested-a",
        "observed_model": "response-a",
        "successful_provider_requests": 1,
    })
    second = _usage_updates(first, {
        "tokens": 3,
        "input_tokens": 2,
        "output_tokens": 1,
        "token_accounting": "estimated_split",
        "estimated_usd": 0.0,
        "compute_proxy_usd": 0.02,
        "requested_model": "requested-b",
        "observed_model": "response-b",
        "successful_provider_requests": 1,
    })

    assert second["tokens"] == 16
    assert second["input_tokens"] == 12
    assert second["output_tokens"] == 4
    assert second["token_accounting"] == "mixed_split"
    assert second["estimated_usd"] == 0.1
    assert second["compute_proxy_usd"] == 0.02
    assert second["observed_model_ids"] == ["response-a", "response-b"]
    assert "requested-a" not in second["observed_model_ids"]
    assert second["successful_provider_requests"] == 2
    assert second["successful_vlm_requests"] == 2


def test_unknown_token_split_does_not_invent_breakdown() -> None:
    state = _usage_updates({}, {
        "tokens": 4,
        "input_tokens": 3,
        "output_tokens": 1,
        "token_accounting": "provider_split",
    })
    result = _usage_updates(state, {"tokens": 2, "usd": 0.0})

    assert result["tokens"] == 6
    assert result["input_tokens"] is None
    assert result["output_tokens"] is None
    assert result["token_accounting"] == "total_only"


def test_merge_usages_keeps_component_costs_separate() -> None:
    merged = _merge_usages(
        {
            "tokens": 2,
            "input_tokens": 1,
            "output_tokens": 1,
            "token_accounting": "provider_split",
            "estimated_usd": 0.01,
            "compute_proxy_usd": 0.0,
            "observed_model": "openai:response",
            "successful_provider_requests": 1,
        },
        {
            "tokens": 5,
            "input_tokens": 4,
            "output_tokens": 1,
            "token_accounting": "provider_split",
            "estimated_usd": 0.0,
            "compute_proxy_usd": 0.2,
            "observed_model": "ollama:response",
            "successful_provider_requests": 1,
        },
    )

    assert merged["estimated_usd"] == 0.01
    assert merged["compute_proxy_usd"] == 0.2
    assert merged["observed_model_ids"] == [
        "openai:response",
        "ollama:response",
    ]
    assert merged["successful_provider_requests"] == 2


def test_vlm_legacy_model_is_observed_but_explicit_unknown_stays_unknown() -> None:
    legacy = _vlm_result_usage({
        "model": "response-vlm",
        "requested_model": "requested-vlm",
    })
    unknown = _vlm_result_usage({
        "model": "requested-vlm",
        "requested_model": "requested-vlm",
        "observed_model": None,
    })

    assert _usage_updates({}, legacy)["observed_model_ids"] == ["response-vlm"]
    assert _usage_updates({}, unknown)["observed_model_ids"] == []


def test_usage_increment_cannot_borrow_model_id_from_previous_call() -> None:
    state = _usage_updates({}, {
        "successful_provider_requests": 1,
        "observed_model_response_count": 1,
        "unobserved_model_successes": 0,
        "observed_model": "openai:first",
        "observed_model_ids": ["openai:first"],
    })

    with pytest.raises(UsageValueError, match="현재 응답 ID"):
        _usage_updates(state, {
            "successful_provider_requests": 1,
            "observed_model_response_count": 1,
            "unobserved_model_successes": 0,
        })


def test_usage_observed_model_must_match_current_id_list() -> None:
    with pytest.raises(UsageValueError, match="observed_model_ids"):
        _usage_updates({}, {
            "successful_provider_requests": 1,
            "observed_model_response_count": 1,
            "unobserved_model_successes": 0,
            "observed_model": "openai:actual",
            "observed_model_ids": ["openai:other"],
        })


def test_merge_allows_only_zero_cost_legacy_usage_with_classified_usage() -> None:
    merged = _merge_usages(
        {
            "tokens": 1,
            "input_tokens": 1,
            "output_tokens": 0,
            "token_accounting": "provider_split",
            "estimated_usd": 0.01,
            "compute_proxy_usd": 0.0,
        },
        {"tokens": 1, "usd": 0.0},
    )
    assert merged["tokens"] == 2
    assert merged["token_accounting"] == "total_only"
    assert merged["estimated_usd"] == 0.01

    with pytest.raises(UsageValueError, match="미분류"):
        _merge_usages(
            {
                "tokens": 1,
                "input_tokens": 1,
                "output_tokens": 0,
                "token_accounting": "provider_split",
                "estimated_usd": 0.01,
                "compute_proxy_usd": 0.0,
            },
            {"tokens": 1, "usd": 0.01},
        )


@pytest.mark.parametrize(
    ("state", "usage", "function"),
    [
        ({"cost_usd": -1.0}, {"usd": 0.0}, _cost),
        ({"cost_usd": 0.0}, {"usd": float("nan")}, _cost),
        ({"cost_usd": 0.0}, {"estimated_usd": "0.1"}, _cost),
        ({"tokens": -1}, {"tokens": 0}, _tokens),
        ({"tokens": 0}, {"tokens": True}, _tokens),
        ({}, "invalid", _cost),
        ({}, {"tokens": 0, "usd": float("nan")}, _usage_updates),
        ({}, {"tokens": 0, "estimated_usd": 0.0}, _usage_updates),
        (
            {
                "tokens": 1,
                "input_tokens": -1,
                "output_tokens": 2,
                "token_accounting": "provider_split",
            },
            {"tokens": 0},
            _usage_updates,
        ),
    ],
)
def test_cost_and_tokens_fail_closed(
    state: dict, usage: object, function: object
) -> None:
    with pytest.raises(UsageValueError):
        function(state, usage)  # type: ignore[operator]


def test_cost_latency_prefers_estimate_and_reports_operational_means() -> None:
    result = cost_latency([
        {
            "latency_s": 1.0,
            "tokens": 10,
            "estimated_usd": 0.2,
            "usd": 99.0,
            "approval_required": True,
            "validation_attempts": 1,
            "validation_rejections": 1,
            "retry_count": 2,
            "replan_rounds": 1,
            "successful_provider_requests": 8,
            "successful_vlm_requests": 6,
        },
        {
            "latency_s": 3.0,
            "tokens": 20,
            "usd": 0.4,
            "approval_required": False,
            "validation_attempts": 3,
            "validation_rejections": 0,
            "retry_count": 0,
            "replan_rounds": 0,
            "successful_provider_requests": 2,
            "successful_vlm_requests": 1,
        },
    ])

    assert result["mean_usd"] == pytest.approx(0.3)
    assert result["mean_retry_count"] == 1.0
    assert result["mean_replan_rounds"] == 0.5
    assert result["mean_validation_rejections"] == 0.5
    assert result["mean_successful_provider_requests"] == 5.0
    assert result["mean_successful_vlm_requests"] == 3.5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("latency_s", "1"),
        ("latency_s", float("nan")),
        ("tokens", -1),
        ("tokens", 1.5),
        ("estimated_usd", float("inf")),
        ("usd", -0.01),
        ("validation_attempts", -1),
        ("validation_rejections", -1),
        ("retry_count", True),
        ("replan_rounds", "0"),
        ("successful_vlm_requests", -1),
    ],
)
def test_cost_latency_rejects_malformed_numbers(field: str, value: object) -> None:
    row = {
        "latency_s": 1.0,
        "tokens": 1,
        "usd": 0.1,
        "validation_attempts": 0,
        "validation_rejections": 0,
        "retry_count": 0,
        "replan_rounds": 0,
        "successful_provider_requests": 0,
        "successful_vlm_requests": 0,
    }
    row[field] = value
    with pytest.raises(EvaluationDataError):
        cost_latency([row])
