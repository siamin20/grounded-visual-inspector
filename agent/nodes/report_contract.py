"""보고서에 노출되는 요약과 로트를 코드 규칙으로 고정한다."""

from copy import deepcopy
import hashlib
import json
from typing import Any, Mapping

from agent.usage import MISSING, nonnegative_finite_float, nonnegative_int


def state_lot(state: dict) -> dict:
    """로트는 상태의 조회값만 사용하고 작성 모델의 값은 버린다."""
    lot = state.get("lot", {})
    return deepcopy(lot) if isinstance(lot, dict) else {}


def make_pre_validation_snapshot(
    report: dict,
    *,
    usage: Mapping[str, Any],
) -> dict:
    """validate 직전 첫 보고서를 재작성과 분리해 고정한다."""

    frozen_report = deepcopy(report)
    canonical = json.dumps(
        frozen_report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    tokens = nonnegative_int(usage.get("tokens", MISSING), name="snapshot.tokens")
    accounting = usage.get("token_accounting")
    if accounting not in {
        "provider_split", "estimated_split", "mixed_split", "total_only"
    }:
        raise ValueError("snapshot.token_accounting 값이 올바르지 않습니다.")
    raw_input = usage.get("input_tokens")
    raw_output = usage.get("output_tokens")
    if accounting == "total_only":
        if raw_input is not None or raw_output is not None:
            raise ValueError("snapshot total_only usage에 분리 토큰을 둘 수 없습니다.")
        input_tokens = output_tokens = None
    else:
        input_tokens = nonnegative_int(raw_input, name="snapshot.input_tokens")
        output_tokens = nonnegative_int(raw_output, name="snapshot.output_tokens")
        if input_tokens + output_tokens != tokens:
            raise ValueError("snapshot 입력·출력 토큰 합이 전체와 다릅니다.")
    observed_ids = usage.get("observed_model_ids")
    if not isinstance(observed_ids, list) or any(
        not isinstance(value, str) or not value.strip() or value != value.strip()
        for value in observed_ids
    ):
        raise ValueError("snapshot.observed_model_ids가 올바르지 않습니다.")
    successful = nonnegative_int(
        usage.get("successful_provider_requests", MISSING),
        name="snapshot.successful_provider_requests",
    )
    observed_count = nonnegative_int(
        usage.get("observed_model_response_count", MISSING),
        name="snapshot.observed_model_response_count",
    )
    unobserved = nonnegative_int(
        usage.get("unobserved_model_successes", MISSING),
        name="snapshot.unobserved_model_successes",
    )
    unknown_tokens = nonnegative_int(
        usage.get("unknown_token_usage_successes", MISSING),
        name="snapshot.unknown_token_usage_successes",
    )
    if observed_count + unobserved != successful or unknown_tokens > successful:
        raise ValueError("snapshot provider usage 카운터가 일관되지 않습니다.")
    if observed_count and not observed_ids:
        raise ValueError("snapshot의 관측 model ID가 누락됐습니다.")
    return {
        "version": 2,
        "stage": "before_validate",
        "report": frozen_report,
        "report_sha256": hashlib.sha256(canonical).hexdigest(),
        "cost_usd": nonnegative_finite_float(
            usage.get("cost_usd", MISSING), name="snapshot.cost_usd"
        ),
        "tokens": tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "token_accounting": accounting,
        "estimated_usd": nonnegative_finite_float(
            usage.get("estimated_usd", MISSING), name="snapshot.estimated_usd"
        ),
        "compute_proxy_usd": nonnegative_finite_float(
            usage.get("compute_proxy_usd", MISSING),
            name="snapshot.compute_proxy_usd",
        ),
        "observed_model_ids": list(dict.fromkeys(observed_ids)),
        "successful_provider_requests": successful,
        "successful_vlm_requests": nonnegative_int(
            usage.get("successful_vlm_requests", MISSING),
            name="snapshot.successful_vlm_requests",
        ),
        "unknown_token_usage_successes": unknown_tokens,
        "unobserved_model_successes": unobserved,
        "observed_model_response_count": observed_count,
    }


def deterministic_summary(report: dict) -> str:
    """인용한 관찰과 코드 판정 필드만으로 결정론적 요약을 만든다."""
    observations = report.get("observations")
    texts: list[str] = []
    if isinstance(observations, list):
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            text = observation.get("text")
            if (
                isinstance(text, str)
                and text.strip()
                and text.strip() not in texts
            ):
                texts.append(text.strip())
    evidence = " / ".join(texts) if texts else "유효한 관찰 근거 없음"

    verdict = report.get("verdict")
    if verdict == "PASS":
        return f"관찰 근거({evidence})를 바탕으로 PASS로 판정하고 통과를 요청합니다."
    if verdict == "REJECT":
        defect_type = report.get("defect_type")
        defect = (
            defect_type
            if isinstance(defect_type, str) and defect_type
            else "유형 미상"
        )
        clauses = report.get("spec_clauses")
        clause_values = clauses if isinstance(clauses, list) else []
        clause_text = ", ".join(
            clause_id for clause_id in clause_values if isinstance(clause_id, str)
        ) or "검증된 조항 없음"
        actions = report.get("actions")
        action_values = actions if isinstance(actions, list) else []
        action_text = ", ".join(
            action for action in action_values if isinstance(action, str)
        ) or "조치 미정"
        return (
            f"관찰 근거({evidence})와 위반 조항({clause_text})을 바탕으로 "
            f"{defect} 결함을 REJECT로 판정하고 {action_text} 조치를 요청합니다."
        )
    if verdict == "ABSTAIN":
        reason = report.get("abstain_reason")
        reason_text = (
            reason if isinstance(reason, str) and reason else "validation_failure"
        )
        return (
            f"관찰 근거({evidence})만으로 확정할 수 없어 {reason_text} 사유로 "
            "ABSTAIN하고 재검사를 요청합니다."
        )
    return f"관찰 근거({evidence})에 대한 판정이 유효하지 않습니다."
