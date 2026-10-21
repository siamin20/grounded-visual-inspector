"""노드가 공유하는 데이터 로더와 상태 헬퍼."""

import json
from pathlib import Path
from typing import Any

from agent.readability import is_unreadable_text
from agent.state import AgentState
from agent.typemap import load_type_map
from agent.usage import (
    MISSING,
    UsageValueError,
    nonnegative_finite_float,
    nonnegative_int,
)


ROOT = Path(__file__).resolve().parents[2]
LOTS_PATH = ROOT / "data" / "lots.json"
KNOWN_ITEMS = frozenset(load_type_map())
_QUADRANTS = frozenset({"tl", "tr", "bl", "br"})
_REGIONS = frozenset({*_QUADRANTS, "center"})


def load_lot(image_path: str | None) -> dict:
    """이미지 경로의 `<item>/test/<type>/<stem>`을 lots 키로 변환한다."""
    if not image_path:
        return {}
    try:
        lots = json.loads(LOTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    path = Path(image_path)
    parts = path.parts
    try:
        test_at = parts.index("test")
        item = parts[test_at - 1]
        defect_type = parts[test_at + 1]
        sample_id = f"{item}-{defect_type}-{path.stem}"
    except (ValueError, IndexError):
        return {}
    value = lots.get(sample_id, {}) if isinstance(lots, dict) else {}
    return dict(value) if isinstance(value, dict) else {}


def event(state: AgentState, event_type: str | None = None, **fields: Any) -> list[dict]:
    """현재 이벤트 목록에 한 건을 붙인 새 목록을 반환한다."""
    if event_type is None:
        event_type = str(fields.pop("type"))
    return [*state.get("events", []), {"type": event_type, **fields}]


def _cost(state: AgentState, usage: dict | None) -> float:
    if usage is not None and not isinstance(usage, dict):
        raise UsageValueError("usage는 JSON 개체나 null이어야 합니다.")
    values = usage or {}
    current = nonnegative_finite_float(
        state.get("cost_usd", MISSING),
        name="state.cost_usd",
        default=0.0,
    )
    if "estimated_usd" in values:
        increment = nonnegative_finite_float(
            values["estimated_usd"], name="usage.estimated_usd"
        )
    else:
        increment = nonnegative_finite_float(
            values.get("usd", MISSING), name="usage.usd", default=0.0
        )
    return round(
        nonnegative_finite_float(
            current + increment, name="state.cost_usd + usage"
        ),
        6,
    )


def _tokens(state: AgentState, usage: dict | None) -> int:
    """모델·임베딩 사용량의 실측 토큰 수를 상태에 누적한다."""
    if usage is not None and not isinstance(usage, dict):
        raise UsageValueError("usage는 JSON 개체나 null이어야 합니다.")
    values = usage or {}
    current = nonnegative_int(
        state.get("tokens", MISSING), name="state.tokens", default=0
    )
    increment = nonnegative_int(
        values.get("tokens", MISSING), name="usage.tokens", default=0
    )
    return nonnegative_int(current + increment, name="state.tokens + usage")


_SPLIT_ACCOUNTING = frozenset({"provider_split", "estimated_split", "mixed_split"})


def _usage_models(state: AgentState, usage: dict) -> list[str]:
    existing = state.get("observed_model_ids", [])
    if not isinstance(existing, list) or any(
        not isinstance(value, str) or not value.strip() for value in existing
    ):
        raise UsageValueError("state.observed_model_ids는 빈 문자열이 없는 목록이어야 합니다.")
    values = list(existing)
    many = usage.get("observed_model_ids")
    if many is not None:
        if not isinstance(many, list) or any(
            not isinstance(value, str) or not value.strip() for value in many
        ):
            raise UsageValueError("usage.observed_model_ids는 빈 문자열이 없는 목록이어야 합니다.")
        values.extend(value.strip() for value in many)
    observed = usage.get("observed_model")
    if observed is not None:
        if not isinstance(observed, str) or not observed.strip():
            raise UsageValueError("usage.observed_model은 null 또는 빈 값이 아닌 문자열이어야 합니다.")
        values.append(observed.strip())
    return list(dict.fromkeys(values))


def _vlm_result_usage(result: dict) -> dict[str, Any]:
    """VLM 결과의 구 ``model``을 실제 응답 ID로만 호환한다."""
    if not isinstance(result, dict):
        raise UsageValueError("VLM 결과는 JSON 개체여야 합니다.")
    usage = dict(result)
    # 새 계약이 observed_model=null을 명시하면 응답 ID 미확인이므로
    # requested/model 값으로 채우지 않는다. 키 자체가 없는 구 VLM 결과만
    # model이 실제 응답 ID라는 기존 공개 계약에 따라 호환한다.
    if "observed_model" not in usage:
        model = usage.get("model")
        if isinstance(model, str) and model.strip():
            usage["observed_model"] = model.strip()
    return usage


def _token_breakdown(
    state: AgentState,
    usage: dict,
) -> tuple[int | None, int | None, str]:
    """provider가 나눈 토큰만 누적하고 한 번이라도 미지수면 합계만 남긴다."""
    current_total = nonnegative_int(
        state.get("tokens", MISSING), name="state.tokens", default=0
    )
    current_mode = state.get("token_accounting")
    if (
        current_mode is not None
        and current_mode not in {*_SPLIT_ACCOUNTING, "total_only"}
    ):
        raise UsageValueError("state.token_accounting 값이 올바르지 않습니다.")

    state_has_input = "input_tokens" in state and state.get("input_tokens") is not None
    state_has_output = "output_tokens" in state and state.get("output_tokens") is not None
    if state_has_input != state_has_output:
        raise UsageValueError("state input/output token은 둘 다 있거나 둘 다 없어야 합니다.")
    if current_mode in _SPLIT_ACCOUNTING:
        if not state_has_input:
            raise UsageValueError("state 분리 토큰이 누락되었습니다.")
        nonnegative_int(state["input_tokens"], name="state.input_tokens")
        nonnegative_int(state["output_tokens"], name="state.output_tokens")
    elif state_has_input:
        raise UsageValueError("state 분리 토큰에 split token_accounting이 필요합니다.")

    has_input = "input_tokens" in usage and usage.get("input_tokens") is not None
    has_output = "output_tokens" in usage and usage.get("output_tokens") is not None
    usage_mode = usage.get("token_accounting")
    usage_is_split = has_input and has_output and usage_mode in _SPLIT_ACCOUNTING
    if has_input != has_output:
        raise UsageValueError("usage input/output token은 둘 다 있거나 둘 다 없어야 합니다.")
    if usage_mode is not None and usage_mode not in {*_SPLIT_ACCOUNTING, "total_only"}:
        raise UsageValueError("usage.token_accounting 값이 올바르지 않습니다.")
    if (has_input or has_output) and not usage_is_split:
        raise UsageValueError("분리 토큰에는 split token_accounting이 필요합니다.")

    previous_is_split = current_mode in _SPLIT_ACCOUNTING
    # 기존 토큰이 있는데 accounting이 없으면 과거 분해를 지어내지 않는다.
    previous_unknown = current_mode == "total_only" or (
        current_mode is None and current_total > 0
    )
    if previous_unknown or not usage_is_split:
        return None, None, "total_only"

    current_input = nonnegative_int(
        state.get("input_tokens", MISSING),
        name="state.input_tokens",
        default=0,
    ) if previous_is_split else 0
    current_output = nonnegative_int(
        state.get("output_tokens", MISSING),
        name="state.output_tokens",
        default=0,
    ) if previous_is_split else 0
    usage_input = nonnegative_int(usage["input_tokens"], name="usage.input_tokens")
    usage_output = nonnegative_int(usage["output_tokens"], name="usage.output_tokens")
    modes = {mode for mode in (current_mode, usage_mode) if mode in _SPLIT_ACCOUNTING}
    mode = next(iter(modes)) if len(modes) == 1 else "mixed_split"
    return current_input + usage_input, current_output + usage_output, mode


def _usage_updates(state: AgentState, usage: dict | None) -> dict[str, Any]:
    """비용·토큰·응답 모델·성공 요청 수를 손실 없이 상태에 누적한다."""
    if usage is not None and not isinstance(usage, dict):
        raise UsageValueError("usage는 JSON 개체나 null이어야 합니다.")
    values = dict(usage or {})
    if ("estimated_usd" in values) != ("compute_proxy_usd" in values):
        raise UsageValueError(
            "usage의 estimated_usd와 compute_proxy_usd는 함께 있어야 합니다."
        )
    if "usd" in values:
        nonnegative_finite_float(values["usd"], name="usage.usd")
    increment_requests = nonnegative_int(
        values.get("successful_provider_requests", MISSING),
        name="usage.successful_provider_requests",
        default=0,
    )
    default_observed_count = (
        1
        if increment_requests
        and (
            isinstance(values.get("observed_model"), str)
            and values["observed_model"].strip()
        )
        else 0
    )
    increment_observed_count = nonnegative_int(
        values.get("observed_model_response_count", MISSING),
        name="usage.observed_model_response_count",
        default=default_observed_count,
    )
    current_ids = values.get("observed_model_ids")
    if current_ids is not None:
        if not isinstance(current_ids, list) or any(
            not isinstance(value, str) or not value.strip()
            for value in current_ids
        ):
            raise UsageValueError(
                "usage.observed_model_ids는 빈 문자열이 없는 목록이어야 합니다."
            )
        current_ids = [value.strip() for value in current_ids]
    observed_model = values.get("observed_model")
    if observed_model is not None:
        if not isinstance(observed_model, str) or not observed_model.strip():
            raise UsageValueError(
                "usage.observed_model은 null 또는 빈 값이 아닌 문자열이어야 합니다."
            )
        observed_model = observed_model.strip()
        if current_ids is not None and observed_model not in current_ids:
            raise UsageValueError(
                "usage.observed_model이 현재 usage의 observed_model_ids에 없습니다."
            )
    current_evidence = list(current_ids or [])
    if observed_model is not None:
        current_evidence.append(observed_model)
    if increment_observed_count and not current_evidence:
        raise UsageValueError(
            "usage에 model ID 관측 응답이 있지만 현재 응답 ID가 없습니다."
        )
    increment_unobserved = nonnegative_int(
        values.get("unobserved_model_successes", MISSING),
        name="usage.unobserved_model_successes",
        default=max(0, increment_requests - increment_observed_count),
    )
    default_unknown_tokens = (
        increment_requests
        if values.get("token_accounting") == "total_only"
        and nonnegative_int(
            values.get("tokens", MISSING), name="usage.tokens", default=0
        ) == 0
        else 0
    )
    increment_unknown_tokens = nonnegative_int(
        values.get("unknown_token_usage_successes", MISSING),
        name="usage.unknown_token_usage_successes",
        default=default_unknown_tokens,
    )
    if increment_observed_count + increment_unobserved != increment_requests:
        raise UsageValueError(
            "usage 모델 ID 관측·미관측 성공 수가 provider 성공 수와 다릅니다."
        )
    if increment_unknown_tokens > increment_requests:
        raise UsageValueError(
            "usage 토큰 미지 성공 수가 provider 성공 수보다 큽니다."
        )
    if not values:
        tokens = _tokens(state, None)
        mode = state.get("token_accounting")
        if mode is None:
            mode = "total_only"
        if mode not in {*_SPLIT_ACCOUNTING, "total_only"}:
            raise UsageValueError("state.token_accounting 값이 올바르지 않습니다.")
        if mode in _SPLIT_ACCOUNTING:
            input_tokens = nonnegative_int(
                state.get("input_tokens", MISSING), name="state.input_tokens"
            )
            output_tokens = nonnegative_int(
                state.get("output_tokens", MISSING), name="state.output_tokens"
            )
        else:
            if (
                state.get("input_tokens") is not None
                or state.get("output_tokens") is not None
            ):
                raise UsageValueError(
                    "state total_only 토큰에는 input/output 합계를 남길 수 없습니다."
                )
            input_tokens = output_tokens = None
        return {
            "cost_usd": _cost(state, None),
            "tokens": tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "token_accounting": mode,
            "estimated_usd": nonnegative_finite_float(
                state.get("estimated_usd", MISSING),
                name="state.estimated_usd",
                default=0.0,
            ),
            "compute_proxy_usd": nonnegative_finite_float(
                state.get("compute_proxy_usd", MISSING),
                name="state.compute_proxy_usd",
                default=0.0,
            ),
            "observed_model_ids": _usage_models(state, values),
            "successful_provider_requests": nonnegative_int(
                state.get("successful_provider_requests", MISSING),
                name="state.successful_provider_requests",
                default=0,
            ),
            "successful_vlm_requests": nonnegative_int(
                state.get("successful_vlm_requests", MISSING),
                name="state.successful_vlm_requests",
                default=0,
            ),
            "unknown_token_usage_successes": nonnegative_int(
                state.get("unknown_token_usage_successes", MISSING),
                name="state.unknown_token_usage_successes",
                default=0,
            ),
            "unobserved_model_successes": nonnegative_int(
                state.get("unobserved_model_successes", MISSING),
                name="state.unobserved_model_successes",
                default=0,
            ),
            "observed_model_response_count": nonnegative_int(
                state.get("observed_model_response_count", MISSING),
                name="state.observed_model_response_count",
                default=0,
            ),
        }
    input_tokens, output_tokens, token_accounting = _token_breakdown(state, values)
    current_estimated = nonnegative_finite_float(
        state.get("estimated_usd", MISSING),
        name="state.estimated_usd",
        default=0.0,
    )
    increment_estimated = nonnegative_finite_float(
        values.get("estimated_usd", MISSING),
        name="usage.estimated_usd",
        default=0.0,
    )
    current_proxy = nonnegative_finite_float(
        state.get("compute_proxy_usd", MISSING),
        name="state.compute_proxy_usd",
        default=0.0,
    )
    increment_proxy = nonnegative_finite_float(
        values.get("compute_proxy_usd", MISSING),
        name="usage.compute_proxy_usd",
        default=0.0,
    )
    current_requests = nonnegative_int(
        state.get("successful_provider_requests", MISSING),
        name="state.successful_provider_requests",
        default=0,
    )
    current_unknown_tokens = nonnegative_int(
        state.get("unknown_token_usage_successes", MISSING),
        name="state.unknown_token_usage_successes",
        default=0,
    )
    current_unobserved = nonnegative_int(
        state.get("unobserved_model_successes", MISSING),
        name="state.unobserved_model_successes",
        default=0,
    )
    current_observed_count = nonnegative_int(
        state.get("observed_model_response_count", MISSING),
        name="state.observed_model_response_count",
        default=0,
    )
    return {
        "cost_usd": _cost(state, values),
        "tokens": _tokens(state, values),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "token_accounting": token_accounting,
        "estimated_usd": round(current_estimated + increment_estimated, 6),
        "compute_proxy_usd": round(current_proxy + increment_proxy, 6),
        "observed_model_ids": _usage_models(state, values),
        "successful_provider_requests": current_requests + increment_requests,
        "successful_vlm_requests": nonnegative_int(
            state.get("successful_vlm_requests", MISSING),
            name="state.successful_vlm_requests",
            default=0,
        ),
        "unknown_token_usage_successes": (
            current_unknown_tokens + increment_unknown_tokens
        ),
        "unobserved_model_successes": current_unobserved + increment_unobserved,
        "observed_model_response_count": (
            current_observed_count + increment_observed_count
        ),
    }


def _merge_usages(*usages: dict | None) -> dict[str, Any]:
    """서로 다른 provider usage를 하나의 누적 usage로 합친다."""
    merged_state: AgentState = {}
    saw_usage = False
    classifications: list[bool] = []
    unclassified_costs: list[float] = []
    for usage in usages:
        if usage is None:
            continue
        if not isinstance(usage, dict):
            raise UsageValueError("usage는 JSON 개체나 null이어야 합니다.")
        if not usage:
            continue
        saw_usage = True
        classified = "estimated_usd" in usage and "compute_proxy_usd" in usage
        if ("estimated_usd" in usage) != ("compute_proxy_usd" in usage):
            raise UsageValueError(
                "usage의 estimated_usd와 compute_proxy_usd는 함께 있어야 합니다."
            )
        classifications.append(classified)
        if not classified:
            unclassified_costs.append(nonnegative_finite_float(
                usage.get("usd", MISSING), name="usage.usd", default=0.0
            ))
        merged_state.update(_usage_updates(merged_state, usage))
    if not saw_usage:
        return {}
    if any(classifications) and not all(classifications) and any(unclassified_costs):
        raise UsageValueError(
            "비용이 있는 미분류 usage는 분류된 usage와 섞을 수 없습니다."
        )
    result = {
        "tokens": merged_state["tokens"],
        "input_tokens": merged_state["input_tokens"],
        "output_tokens": merged_state["output_tokens"],
        "token_accounting": merged_state["token_accounting"],
        "usd": merged_state["cost_usd"],
        "observed_model_ids": merged_state["observed_model_ids"],
        "successful_provider_requests": merged_state["successful_provider_requests"],
        "unknown_token_usage_successes": merged_state[
            "unknown_token_usage_successes"
        ],
        "unobserved_model_successes": merged_state[
            "unobserved_model_successes"
        ],
        "observed_model_response_count": merged_state[
            "observed_model_response_count"
        ],
    }
    # 기존 스텁의 0달러 usage는 사용량 호환을 위해 허용하되,
    # 비용 분류값은 실제로 근거를 준 provider usage에서만 남긴다.
    if any(classifications):
        result["estimated_usd"] = merged_state["estimated_usd"]
        result["compute_proxy_usd"] = merged_state["compute_proxy_usd"]
    return result


def is_readable_observation(observation: Any) -> bool:
    """Observer 결과가 정상/이상을 구분할 수 있는 구조인지 판정한다.

    빈 관찰문은 `anomaly_seen=false`만으로 정상 판정을 만들 수 있으므로 판독
    가능으로 인정하지 않는다. `bool`은 정수 0/1과 구분해 명시적 출력만 받는다.
    """
    if not isinstance(observation, dict):
        return False
    text = observation.get("text")
    return (
        type(observation.get("anomaly_seen")) is bool
        and isinstance(text, str)
        and not is_unreadable_text(text)
        and "판독 불가" not in text
    )


def observer_status(observations: Any) -> str:
    """Observer 관찰들을 설계 계약의 `O+|O-|O?`로 집계한다."""
    if not isinstance(observations, list):
        return "O?"
    readable = [
        observation
        for observation in observations
        if is_readable_observation(observation)
    ]
    if any(observation["anomaly_seen"] is True for observation in readable):
        return "O+"
    if any(
        observation.get("region") == "full"
        and observation["anomaly_seen"] is False
        for observation in readable
    ):
        return "O-"
    negative_quadrants = {
        str(observation.get("region"))
        for observation in readable
        if observation["anomaly_seen"] is False
        and observation.get("region") in _QUADRANTS
    }
    return "O-" if len(negative_quadrants) >= 3 else "O?"


def positive_observer_regions(observations: Any) -> set[str]:
    """판독 가능하며 이상을 본 Observer 결과의 영역만 반환한다."""
    if not isinstance(observations, list):
        return set()
    return {
        str(region)
        for observation in observations
        if is_readable_observation(observation)
        and observation["anomaly_seen"] is True
        and isinstance(observation.get("regions"), list)
        for region in observation["regions"]
        if region in _REGIONS
    }


def cross_status(cross_check: Any, *, skipped: bool = False) -> str:
    """Cross-Checker 결과를 `C+|C-|C?|C-skip`으로 정규화한다."""
    if skipped:
        return "C-skip"
    if not isinstance(cross_check, dict):
        return "C?"
    summary = cross_check.get("summary")
    regions = cross_check.get("regions")
    if (
        type(cross_check.get("anomaly_seen")) is not bool
        or not isinstance(summary, str)
        or is_unreadable_text(summary)
        or "판독 불가" in summary
        or not isinstance(regions, list)
        or any(region not in _REGIONS for region in regions)
    ):
        return "C?"
    return "C+" if cross_check["anomaly_seen"] is True else "C-"


def normalized_judgement_status(judgement: Any) -> str:
    """검증된 Spec-Judge 결과를 `V+|V0|V?`로 정규화한다."""
    if not isinstance(judgement, dict):
        return "V?"
    status = judgement.get("status")
    if status is not None:
        violations = judgement.get("violations")
        defect_type = judgement.get("defect_type")
        grounded_shape = (
            isinstance(violations, list)
            and bool(violations)
            and all(isinstance(value, str) and value for value in violations)
            and isinstance(defect_type, str)
            and bool(defect_type)
            and defect_type != "good"
        )
        none_shape = violations == [] and defect_type is None
        if status in {"grounded", "V+"} and grounded_shape:
            return "V+"
        if status in {"none", "V0"} and none_shape:
            return "V0"
        return "V?"

    # 이전 상태나 단위 테스트처럼 status가 없는 입력도 보수적으로 해석한다.
    violations = judgement.get("violations")
    defect_type = judgement.get("defect_type")
    if isinstance(violations, list) and violations and isinstance(defect_type, str):
        return "V+"
    if violations == [] and defect_type is None:
        return "V0"
    return "V?"
