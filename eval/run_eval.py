"""서버 없이 조건별 이미지 검사 평가를 실행한다."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, AsyncIterator, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langsmith import Client, traceable, tracing_context
from langsmith.run_helpers import get_current_run_tree
from langsmith.utils import LangSmithNotFoundError

from agent.deps import SupervisorDeps
from agent.graph import build_graph
from agent.mcp_client import MCPClient
from agent.nodes.validate import _validation_threshold
from agent.rag import SpecIndex, make_embedder
from agent.supervisor_v1 import run_v1
from agent.usage import (
    MISSING,
    UsageValueError,
    nonnegative_finite_float,
    nonnegative_int,
)
from agent.vision.backends import openai_vlm_usd_per_million
from eval.evaluators import binary_metrics, cost_latency
from eval.preflight import EvaluationPreflightError, guard_run_mode
from eval.provenance import (
    ProvenanceError,
    REQUIRED_RESULT_ROW_KEYS,
    atomic_write_json,
    build_run_manifest,
    canonical_sha256,
    collect_repository_provenance,
    complete_condition,
    create_run_manifest,
    fail_condition,
    load_run_manifest,
    new_run_id,
    record_condition_progress,
    record_condition_setup_usage,
    sha256_file,
    start_condition,
    validate_run_id,
)
from server import _build_llms, _initial_state


DATASET_PATH = ROOT / "eval" / "dataset.json"
DEFAULT_RESULTS_DIR = ROOT / "eval" / "results"
DEFAULT_SUITE_BUDGET_USD = 15.0
CONDITIONS = ("B0", "B1", "A", "A-X", "A-C", "A-V")
HELDOUT_CONDITIONS = frozenset({"B0", "B1", "A"})
ABLATIONS = {
    "A": "",
    "A-X": "no_cross",
    "A-C": "no_crops",
    "A-V": "no_validate",
}
REQUIRED_ROW_KEYS = REQUIRED_RESULT_ROW_KEYS
class _NullSessionStore:
    """평가에서는 대화 DB를 만들지 않지만 deps 인터페이스는 유지한다."""

    def load(self, _thread_id: str) -> list[dict]:
        return []

    def append(self, _thread_id: str, _role: str, _content: str) -> None:
        return None

    def close(self) -> None:
        return None


class SetupUsageError(RuntimeError):
    """색인 초기화 실패와 그 전까지 확보한 usage를 함께 전달한다."""

    def __init__(self, message: str, *, setup_usage: Mapping[str, Any]):
        super().__init__(message)
        self.setup_usage = dict(setup_usage)


def _selected_backends() -> tuple[str, str, str]:
    llm_backend = (os.environ.get("LLM_BACKEND") or "fake").strip().lower()
    vlm_backend = (
        os.environ.get("VLM_BACKEND") or llm_backend
    ).strip().lower()
    embed_backend = (
        os.environ.get("EMBED_BACKEND") or llm_backend
    ).strip().lower()
    return llm_backend, vlm_backend, embed_backend


def _guard_backends(*, dry_run: bool, condition: str | None = None) -> None:
    llm_backend, vlm_backend, embed_backend = _selected_backends()
    supported = {"fake", "ollama", "openai"}
    unknown = [
        f"{name}={value}"
        for name, value in (
            ("LLM_BACKEND", llm_backend),
            ("VLM_BACKEND", vlm_backend),
            ("EMBED_BACKEND", embed_backend),
        )
        if value not in supported
    ]
    if unknown:
        raise SystemExit(f"지원하지 않는 평가 백엔드: {', '.join(unknown)}")
    if not dry_run and (llm_backend == "fake" or vlm_backend == "fake"):
        raise SystemExit(
            "평가 결과 조작을 막기 위해 fake LLM/VLM은 거부합니다. "
            "로컬 검증은 --dry-run을 사용하세요."
        )
    if (
        not dry_run
        and condition not in {None, "B0", "B1"}
        and embed_backend == "fake"
    ):
        raise SystemExit(
            "After 실측 평가는 fake embedding을 허용하지 않습니다."
        )


def _guard_run_request(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.split == "heldout" and args.condition not in HELDOUT_CONDITIONS:
        raise SystemExit(
            "heldout은 사전 등록 주 조건 B0|B1|A만 실행할 수 있습니다. "
            "A-X|A-C|A-V ablation은 dev 전용입니다."
        )
    llm_backend, vlm_backend, embed_backend = _selected_backends()
    try:
        return guard_run_mode(
            split=args.split,
            dry_run=args.dry_run,
            items=args.items,
            limit=args.limit,
            final_run=args.final_run,
            freeze_manifest=args.freeze_manifest,
            backends={
                "LLM_BACKEND": llm_backend,
                "VLM_BACKEND": vlm_backend,
                "EMBED_BACKEND": embed_backend,
            },
            dataset_path=DATASET_PATH,
        )
    except EvaluationPreflightError as exc:
        raise SystemExit(f"평가 preflight 실패: {exc}") from exc


def _load_samples(
    *, split: str, items: set[str] | None, limit: int | None
) -> list[dict[str, Any]]:
    payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    raw_samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(raw_samples, list):
        raise ValueError("eval/dataset.json의 samples가 배열이 아닙니다.")

    known_items = {
        str(sample.get("item"))
        for sample in raw_samples
        if isinstance(sample, dict) and sample.get("item")
    }
    if items:
        unknown = sorted(items - known_items)
        if unknown:
            raise ValueError(f"알 수 없는 품목: {', '.join(unknown)}")

    selected = [
        dict(sample)
        for sample in raw_samples
        if isinstance(sample, dict)
        and sample.get("split") == split
        and (not items or sample.get("item") in items)
    ]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit은 1 이상이어야 합니다.")
        selected = selected[:limit]
    if not selected:
        raise ValueError("선택된 평가 샘플이 없습니다.")
    return selected


def _output_paths(
    output_dir: Path, condition: str, split: str, *, dry_run: bool
) -> tuple[Path, Path]:
    suffix = "_DRYRUN" if dry_run else ""
    stem = f"{condition}_{split}{suffix}"
    return output_dir / f"{stem}.jsonl", output_dir / f"summary_{stem}.json"


def _parse_suite_conditions(value: str | None, *, current: str, split: str) -> list[str]:
    """동일 run에서 실행할 조건을 한 번 고정한다."""
    raw_values = [current] if value is None else value.split(",")
    conditions = [entry.strip() for entry in raw_values]
    if not conditions or any(not entry for entry in conditions):
        raise ValueError("--suite-conditions에 빈 조건을 둘 수 없습니다.")
    unknown = [entry for entry in conditions if entry not in CONDITIONS]
    if unknown:
        raise ValueError(
            "알 수 없는 suite 조건: " + ", ".join(dict.fromkeys(unknown))
        )
    if len(set(conditions)) != len(conditions):
        raise ValueError("--suite-conditions에 중복 조건을 둘 수 없습니다.")
    if current not in conditions:
        raise ValueError("현재 --condition이 --suite-conditions에 포함되어야 합니다.")
    if split == "heldout" and tuple(conditions) != ("B0", "B1", "A"):
        raise ValueError("heldout suite 조건은 B0,B1,A 순서로 고정됩니다.")
    return conditions


def _suite_budget() -> float:
    """suite 공유 예산을 환경변수에서 읽고 유한한 수치만 허용한다."""
    raw = os.environ.get(
        "EVAL_SUITE_BUDGET_USD",
        os.environ.get("EVAL_BUDGET_USD", str(DEFAULT_SUITE_BUDGET_USD)),
    )
    try:
        parsed = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "EVAL_SUITE_BUDGET_USD는 유한한 0 이상 수치여야 합니다."
        ) from exc
    try:
        return nonnegative_finite_float(
            parsed, name="EVAL_SUITE_BUDGET_USD"
        )
    except UsageValueError as exc:
        raise ValueError(str(exc)) from exc


def _dataset_seed() -> int:
    payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("eval/dataset.json 최상위 값은 개체여야 합니다.")
    try:
        return nonnegative_int(payload.get("seed", MISSING), name="dataset.seed")
    except UsageValueError as exc:
        raise ValueError(str(exc)) from exc


def _requested_execution(
    planned_conditions: list[str],
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    """비밀값 없이 요청 backend·model·가격 산정 근거를 고정한다."""
    llm_backend, vlm_backend, embed_backend = _selected_backends()
    uses_after = any(condition not in {"B0", "B1"} for condition in planned_conditions)
    effective_embed_backend = embed_backend if uses_after else "not_applicable"

    if llm_backend == "openai":
        fast_model = os.environ.get("OPENAI_FAST_MODEL", "gpt-4o-mini")
        quality_model = os.environ.get("OPENAI_QUALITY_MODEL", "gpt-4o")
    elif llm_backend == "ollama":
        fast_model = os.environ.get("OLLAMA_FAST_MODEL", "qwen3:0.6b")
        quality_model = os.environ.get("OLLAMA_QUALITY_MODEL", "qwen3:8b")
    else:
        fast_model = "fast-model-stub"
        quality_model = "quality-model-stub"

    if vlm_backend == "openai":
        observer_model = os.environ.get("VLM_OBSERVER_MODEL", "gpt-4o-mini")
        cross_model = os.environ.get("VLM_CROSS_MODEL", "gpt-4o")
    elif vlm_backend == "ollama":
        observer_model = os.environ.get("VLM_OBSERVER_MODEL", "qwen2.5vl:3b")
        cross_model = os.environ.get("VLM_CROSS_MODEL", "qwen2.5vl:3b")
    else:
        observer_model = cross_model = "fake-vlm"

    if not uses_after:
        embed_model = "not_applicable"
    elif embed_backend == "openai":
        embed_model = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
    elif embed_backend == "ollama":
        embed_model = os.environ.get("EMBED_MODEL", "nomic-embed-text")
    else:
        embed_model = "fake-hash-embedder"

    pricing: dict[str, Any] = {
        "version": "provider-effective-v2",
        "kind": "estimated_not_billed",
        "llm_fast_usd_per_1k_tokens": 0.00015,
        "llm_quality_usd_per_1k_tokens": 0.005,
        "openai_embed_usd_per_1k_tokens": (
            0.00002 if effective_embed_backend == "openai" else None
        ),
        "ollama_embed_compute_proxy_usd_per_1k_tokens": (
            0.00002 if effective_embed_backend == "ollama" else None
        ),
        "openai_vlm_observer_usd_per_million_tokens": (
            openai_vlm_usd_per_million(observer_model)
            if vlm_backend == "openai" else None
        ),
        "openai_vlm_cross_usd_per_million_tokens": (
            openai_vlm_usd_per_million(cross_model)
            if vlm_backend == "openai" else None
        ),
        "ollama_vlm_observer_compute_proxy_usd_per_1k_tokens": (
            0.00015 if vlm_backend == "ollama" else None
        ),
        "ollama_vlm_cross_compute_proxy_usd_per_1k_tokens": (
            0.005 if vlm_backend == "ollama" else None
        ),
    }
    return (
        {
            "llm": llm_backend,
            "vlm": vlm_backend,
            # 사용 여부와 무관하게 실행 환경에서 선택한 backend를 남겨
            # dry-run usage 분류와 verifier가 같은 구성을 보게 한다.
            "embed": embed_backend,
        },
        {
            "llm_fast": fast_model,
            "llm_quality": quality_model,
            "vlm_observer": observer_model,
            "vlm_cross": cross_model,
            "embed": embed_model,
        },
        pricing,
    )


def _manifest_settings(args: argparse.Namespace, *, items: set[str] | None) -> dict[str, Any]:
    freeze_record: dict[str, str] | None = None
    freeze_value = getattr(args, "freeze_manifest", None)
    if freeze_value:
        freeze_path = Path(freeze_value).expanduser().resolve(strict=True)
        try:
            relative = freeze_path.relative_to(ROOT.resolve(strict=True)).as_posix()
        except ValueError as exc:
            raise ValueError("동결 manifest는 저장소 안에 있어야 합니다.") from exc
        freeze_record = {"path": relative, "sha256": sha256_file(freeze_path)}
    return {
        "items": sorted(items) if items else None,
        "limit": args.limit,
        "temperature": 0.0,
        "validation_similarity_threshold": _validation_threshold(),
        "budget_guard": "suite_soft_post_sample_v2",
        "budget_guard_note": (
            "조건 초기화 provider 요청 후와 샘플 종료 후 usage로 확인하므로 "
            "최대 한 초기화 요청 또는 한 샘플의 다중 provider 요청 비용을 "
            "초과할 수 있음"
        ),
        "freeze_manifest": freeze_record,
    }


def _image_provenance_inputs(
    samples: list[dict[str, Any]],
    *,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """실측에 실제로 입력되는 평가·참조 이미지 목록을 고정한다.

    dry-run은 성능 근거로 쓰지 않으므로 이미지 바이트 해시를 생성하지
    않는다. 실측 run만 manifest 생성 전에 모든 입력을 해시한다.
    """
    if dry_run:
        return [], []
    evaluation = [
        {"id": str(sample["id"]), "path": ROOT / str(sample["path"])}
        for sample in samples
    ]
    references = [
        {
            "item": item,
            "path": ROOT / "data" / "mvtec" / item / "train" / "good" / "000.png",
        }
        for item in sorted({str(sample["item"]) for sample in samples})
    ]
    return evaluation, references


def _push_dataset_once(samples: list[dict[str, Any]], *, split: str) -> None:
    """명시적 플래그가 있을 때만 LangSmith Dataset을 한 번 생성한다."""
    name = f"vision-inspection-{split}"
    examples = [
        {
            "inputs": {
                "id": sample["id"],
                "image_path": sample["path"],
                "item": sample["item"],
            },
            "outputs": {
                "label": sample["label"],
                "is_defect": sample["is_defect"],
                "split": sample["split"],
            },
        }
        for sample in samples
    ]
    roster_sha256 = canonical_sha256(examples)
    marker = f"roster_sha256={roster_sha256}; n={len(examples)}"
    client = Client()
    try:
        existing = client.read_dataset(dataset_name=name)
    except LangSmithNotFoundError:
        dataset = client.create_dataset(
            name,
            description=(
                "MVTec AD 이미지 경로와 폴더 라벨 기반 검사 평가셋; "
                + marker
            ),
        )
        client.create_examples(
            dataset_id=dataset.id,
            examples=examples,
        )
        print(f"LangSmith Dataset 생성: {name} ({len(samples)}건)")
    else:
        description = getattr(existing, "description", None)
        if not isinstance(description, str) or marker not in description:
            raise RuntimeError(
                f"기존 LangSmith Dataset roster가 현재 실행과 다릅니다: {name}"
            )
        print(f"LangSmith Dataset이 이미 있어 업로드를 건너뜁니다: {name}")


def _no_setup_usage() -> dict[str, Any]:
    """provider 초기화 호출이 없는 조건의 명시적 0 usage."""
    return {
        "tokens": 0,
        "input_tokens": None,
        "output_tokens": None,
        "token_accounting": "total_only",
        "estimated_usd": 0.0,
        "compute_proxy_usd": 0.0,
        "successful_provider_requests": 0,
        "successful_vlm_requests": 0,
        "unknown_token_usage_successes": 0,
        "unobserved_model_successes": 0,
        "observed_model_response_count": 0,
        "observed_model_ids": [],
        "latency_s": 0.0,
        "usage_status": "not_applicable",
    }


def _unknown_setup_error(*, latency_s: float) -> dict[str, Any]:
    """provider 응답·과금 여부를 확정할 수 없는 setup 실패다."""
    usage = _no_setup_usage()
    usage["latency_s"] = nonnegative_finite_float(
        latency_s, name="setup_usage.latency_s"
    )
    usage["usage_status"] = "unknown_after_error"
    return usage


def _setup_usage(raw: Mapping[str, Any], *, latency_s: float) -> dict[str, Any]:
    """기준서 색인 usage를 샘플 행과 독립된 초기화 usage로 고정한다."""
    payload = dict(raw)
    payload["successful_vlm_requests"] = 0
    normalized = _usage_fields(payload)
    observed: list[str] = []
    many = payload.get("observed_model_ids")
    if isinstance(many, list):
        observed.extend(
            value.strip()
            for value in many
            if isinstance(value, str) and value.strip()
        )
    one = payload.get("observed_model")
    if isinstance(one, str) and one.strip():
        observed.append(one.strip())
    return {
        "tokens": normalized["tokens"],
        "input_tokens": normalized["input_tokens"],
        "output_tokens": normalized["output_tokens"],
        "token_accounting": normalized["token_accounting"],
        "estimated_usd": normalized["estimated_usd"],
        "compute_proxy_usd": normalized["compute_proxy_usd"],
        "successful_provider_requests": normalized[
            "successful_provider_requests"
        ],
        "successful_vlm_requests": 0,
        "unknown_token_usage_successes": normalized[
            "unknown_token_usage_successes"
        ],
        "unobserved_model_successes": normalized[
            "unobserved_model_successes"
        ],
        "observed_model_response_count": normalized[
            "observed_model_response_count"
        ],
        "observed_model_ids": list(dict.fromkeys(observed)),
        "latency_s": nonnegative_finite_float(
            latency_s, name="setup_usage.latency_s"
        ),
        "usage_status": normalized["usage_status"],
    }


@asynccontextmanager
async def _runtime(
    condition: str,
) -> AsyncIterator[tuple[SupervisorDeps, Any, dict[str, Any]]]:
    """MCP와 모델 의존성을 평가 프로세스 수명 동안 한 번만 연다."""
    version = "v1" if condition in {"B0", "B1"} else "v2"
    ablation = ABLATIONS.get(condition, "")
    previous_baseline = os.environ.get("BASELINE_MODE")
    if version == "v1":
        os.environ["BASELINE_MODE"] = condition

    mcp = MCPClient(flaky_first_tool_call=False)
    try:
        await mcp.start()
        _, fast_llm, quality_llm = _build_llms()
        vector_store = None
        setup_usage = _no_setup_usage()
        if version == "v2":
            embedder = make_embedder()
            setup_started = time.perf_counter()
            try:
                vector_store = SpecIndex.from_dir(ROOT / "data" / "specs", embedder)
            except Exception as exc:
                latency_s = round(time.perf_counter() - setup_started, 6)
                raw_usage = getattr(embedder, "last_usage", {})
                try:
                    successful = nonnegative_int(
                        raw_usage.get("successful_provider_requests", MISSING),
                        name="setup_usage.successful_provider_requests",
                        default=0,
                    ) if isinstance(raw_usage, Mapping) else 0
                    failed_usage = (
                        _setup_usage(raw_usage, latency_s=latency_s)
                        if successful
                        else _unknown_setup_error(latency_s=latency_s)
                    )
                except (UsageValueError, ValueError, TypeError):
                    failed_usage = _unknown_setup_error(latency_s=latency_s)
                raise SetupUsageError(
                    f"기준서 색인 초기화 실패: {type(exc).__name__}: {exc}",
                    setup_usage=failed_usage,
                ) from exc
            setup_usage = _setup_usage(
                getattr(embedder, "last_usage", {}),
                latency_s=round(time.perf_counter() - setup_started, 6),
            )
        deps = SupervisorDeps(
            fast_llm=fast_llm,
            quality_llm=quality_llm,
            mcp=mcp,
            session_store=_NullSessionStore(),
            version=version,
            vector_store=vector_store,
            hitl_mode="auto_approve",
            ablation=ablation,
        )
        graph = build_graph(deps) if version == "v2" else None
        yield deps, graph, setup_usage
    finally:
        try:
            await mcp.stop()
        finally:
            if previous_baseline is None:
                os.environ.pop("BASELINE_MODE", None)
            else:
                os.environ["BASELINE_MODE"] = previous_baseline


def _validation_details(events: object) -> tuple[list[str], list[str]]:
    first_errors: list[str] = []
    would_fail: list[str] = []
    if not isinstance(events, list):
        return first_errors, would_fail
    for value in events:
        if not isinstance(value, dict) or value.get("type") != "Validation":
            continue
        errors = value.get("errors")
        if not first_errors and isinstance(errors, list) and errors:
            first_errors = [str(error) for error in errors]
        potential = value.get("would_fail")
        if not would_fail and isinstance(potential, list) and potential:
            would_fail = [str(error) for error in potential]
    return first_errors, would_fail


def _validation_rejections(events: object) -> int:
    """validate가 실제로 반려한 횟수를 세고 no_validate 진단은 뺀다."""
    if not isinstance(events, list):
        return 0
    return sum(
        1
        for value in events
        if isinstance(value, dict)
        and value.get("type") == "Validation"
        and value.get("passed") is False
        and not value.get("skipped")
    )


def _observed_model_ids(result: dict[str, Any]) -> list[str]:
    """모델이 실제 반환한 결과에서만 ID를 수집한다."""
    explicit = result.get("observed_model_ids", MISSING)
    if not isinstance(explicit, list) or any(
        not isinstance(value, str) or not value.strip() for value in explicit
    ):
        raise UsageValueError(
            "result.observed_model_ids는 빈 문자열이 없는 목록이어야 합니다."
        )
    return list(dict.fromkeys(value.strip() for value in explicit))


def _usage_fields(result: dict[str, Any]) -> dict[str, Any]:
    """provider가 누적한 실측 usage를 손실 없이 평가 행으로 옮긴다."""
    tokens = nonnegative_int(
        result.get("tokens", MISSING), name="result.tokens"
    )
    token_accounting = result.get("token_accounting")
    if token_accounting not in {
        "provider_split",
        "estimated_split",
        "mixed_split",
        "total_only",
    }:
        raise UsageValueError("result.token_accounting 값이 올바르지 않습니다.")
    raw_input = result.get("input_tokens")
    raw_output = result.get("output_tokens")
    if token_accounting == "total_only":
        if raw_input is not None or raw_output is not None:
            raise UsageValueError(
                "total_only usage에는 input/output token을 둘 수 없습니다."
            )
        input_tokens = output_tokens = None
    else:
        input_tokens = nonnegative_int(
            raw_input, name="result.input_tokens"
        )
        output_tokens = nonnegative_int(
            raw_output, name="result.output_tokens"
        )
        if input_tokens + output_tokens != tokens:
            raise UsageValueError(
                "result 입력·출력 토큰 합이 전체 토큰과 다릅니다."
            )
    estimated_usd = nonnegative_finite_float(
        result.get("estimated_usd", MISSING), name="result.estimated_usd"
    )
    compute_proxy_usd = nonnegative_finite_float(
        result.get("compute_proxy_usd", MISSING),
        name="result.compute_proxy_usd",
    )
    successful_provider_requests = nonnegative_int(
        result.get("successful_provider_requests", MISSING),
        name="result.successful_provider_requests",
    )
    successful_vlm_requests = nonnegative_int(
        result.get("successful_vlm_requests", MISSING),
        name="result.successful_vlm_requests",
    )
    unknown_token_usage_successes = nonnegative_int(
        result.get("unknown_token_usage_successes", MISSING),
        name="result.unknown_token_usage_successes",
        default=(
            successful_provider_requests
            if token_accounting == "total_only" and tokens == 0
            else 0
        ),
    )
    unobserved_model_successes = nonnegative_int(
        result.get("unobserved_model_successes", MISSING),
        name="result.unobserved_model_successes",
        default=0,
    )
    observed_model_response_count = nonnegative_int(
        result.get("observed_model_response_count", MISSING),
        name="result.observed_model_response_count",
        default=max(0, successful_provider_requests - unobserved_model_successes),
    )
    if observed_model_response_count + unobserved_model_successes != successful_provider_requests:
        raise UsageValueError(
            "result 모델 ID 관측 카운트가 provider 성공 수와 다릅니다."
        )
    if unknown_token_usage_successes > successful_provider_requests:
        raise UsageValueError(
            "result 토큰 미지 성공 수가 provider 성공 수보다 큽니다."
        )
    llm_backend, vlm_backend, embed_backend = _selected_backends()
    selected = {llm_backend, vlm_backend, embed_backend}
    if selected == {"fake"}:
        basis = "fake_no_billing"
    elif "openai" in selected and "ollama" in selected:
        basis = "mixed_provider_estimate_and_compute_proxy_v1"
    elif "openai" in selected:
        basis = "provider_token_estimate_v1"
    else:
        basis = "ollama_api_zero_compute_proxy_v1"
    if selected == {"fake"}:
        usage_status = "fake_estimate_no_billing"
    elif unknown_token_usage_successes or unobserved_model_successes:
        # 하나의 성공 응답이라도 usage나 model ID가 비면 0비용·
        # 완전한 provenance로 가장하지 않는다.
        usage_status = "unknown_tokens_after_success"
    elif token_accounting == "total_only":
        usage_status = "known_total_only"
    else:
        usage_status = "known_split"
    return {
        "tokens": tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "token_accounting": token_accounting,
        "estimated_usd": round(estimated_usd, 6),
        "billed_usd": None,
        "compute_proxy_usd": round(compute_proxy_usd, 6),
        "cost_basis": basis,
        "successful_provider_requests": successful_provider_requests,
        "successful_vlm_requests": successful_vlm_requests,
        "unknown_token_usage_successes": unknown_token_usage_successes,
        "unobserved_model_successes": unobserved_model_successes,
        "observed_model_response_count": observed_model_response_count,
        "usage_status": usage_status,
    }


def _observations_text(result: dict[str, Any], report: dict[str, Any]) -> str:
    parts: list[str] = []
    observations = result.get("observations")
    if not isinstance(observations, list):
        observations = report.get("observations", [])
    for value in observations if isinstance(observations, list) else []:
        if isinstance(value, dict) and str(value.get("text", "")).strip():
            parts.append(str(value["text"]).strip())
    cross_check = result.get("cross_check")
    if isinstance(cross_check, dict) and str(cross_check.get("summary", "")).strip():
        parts.append(str(cross_check["summary"]).strip())
    return "\n".join(parts)


async def _invoke_sample(
    sample: dict[str, Any],
    condition: str,
    deps: SupervisorDeps,
    graph: Any,
) -> dict[str, Any]:
    image_path = str(ROOT / str(sample["path"]))
    thread_id = f"eval-{condition}-{sample['id']}"
    if deps.version == "v1":
        return await run_v1(
            deps,
            "검사해줘",
            thread_id,
            [],
            image_path=image_path,
            item=str(sample["item"]),
        )
    state = _initial_state(
        "검사해줘",
        thread_id,
        [],
        image_path,
        str(sample["item"]),
        deps.ablation,
    )
    return await graph.ainvoke(state)


def _trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """추적에는 정답을 드러내지 않는 최소 문맥만 남긴다."""
    sample = inputs.get("sample")
    public_sample = (
        {
            "item": sample.get("item"),
            "split": sample.get("split"),
        }
        if isinstance(sample, dict)
        else None
    )
    return {
        # dataset id와 path에도 결함 label이 들어 있으므로
        # label·is_defect와 함께 trace에서 제외한다.
        "sample": public_sample,
        "condition": inputs.get("condition"),
    }


@traceable(name="evaluation_sample", process_inputs=_trace_inputs)
async def _evaluate_sample(
    sample: dict[str, Any],
    condition: str,
    deps: SupervisorDeps,
    graph: Any,
) -> tuple[dict[str, Any], str | None]:
    result = await _invoke_sample(sample, condition, deps, graph)
    run_tree = get_current_run_tree()
    return result, str(run_tree.id) if run_tree is not None else None


def _empty_row(
    sample: dict[str, Any],
    condition: str,
    *,
    run_id: str,
    provenance_sha256: str,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "provenance_sha256": provenance_sha256,
        "dry_run": dry_run,
        "id": sample.get("id"),
        "item": sample.get("item"),
        "label": sample.get("label"),
        "is_defect": bool(sample.get("is_defect")),
        "split": sample.get("split"),
        "condition": condition,
        "verdict": None,
        "defect_type": None,
        "confidence": None,
        "abstain_reason": None,
        "approval_required": False,
        "validation_attempts": 0,
        "validation_rejections": 0,
        "retry_count": 0,
        "replan_rounds": 0,
        "successful_provider_requests": 0,
        "successful_vlm_requests": 0,
        "unknown_token_usage_successes": 0,
        "unobserved_model_successes": 0,
        "observed_model_response_count": 0,
        "validation_errors_first": [],
        "would_fail": [],
        "observations_text": "",
        "pre_validation_snapshot": {},
        "report": {},
        "latency_s": 0.0,
        "tokens": 0,
        "input_tokens": None,
        "output_tokens": None,
        "token_accounting": "total_only",
        "estimated_usd": 0.0,
        "billed_usd": None,
        "compute_proxy_usd": 0.0,
        "cost_basis": "unknown",
        "observed_model_ids": [],
        "langsmith_run_id": None,
        "usage_status": "not_started",
        "error": None,
    }


async def _run(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    _guard_backends(dry_run=args.dry_run, condition=args.condition)
    _guard_run_request(args)
    planned_conditions = _parse_suite_conditions(
        getattr(args, "suite_conditions", None),
        current=args.condition,
        split=args.split,
    )
    if (
        not args.dry_run
        and any(condition not in {"B0", "B1"} for condition in planned_conditions)
        and _selected_backends()[2] == "fake"
    ):
        raise SystemExit(
            "After가 포함된 실측 suite는 fake embedding을 허용하지 않습니다."
        )
    output_root = Path(args.out).expanduser().resolve()
    if (
        output_root != DEFAULT_RESULTS_DIR.resolve()
        and not getattr(args, "_allow_test_output", False)
    ):
        raise ValueError(
            "평가 산출물은 eval/results/<run_id>에만 저장할 수 있습니다."
        )
    run_id = validate_run_id(getattr(args, "run_id", None) or new_run_id())
    budget = _suite_budget()
    if budget == 0:
        raise ValueError(
            "suite 예산이 0 USD이므로 모델·MCP·파일을 시작하지 않습니다."
        )
    items = {
        value.strip() for value in (args.items or "").split(",") if value.strip()
    } or None
    samples = _load_samples(split=args.split, items=items, limit=args.limit)
    seed = _dataset_seed()
    backends, models, pricing = _requested_execution(planned_conditions)
    evaluation_images, reference_images = _image_provenance_inputs(
        samples, dry_run=bool(args.dry_run)
    )
    provenance = collect_repository_provenance(
        ROOT,
        evaluation_images=evaluation_images,
        reference_images=reference_images,
    )
    expected_manifest = build_run_manifest(
        run_id=run_id,
        split=args.split,
        planned_conditions=planned_conditions,
        seed=seed,
        expected_n=len(samples),
        dry_run=bool(args.dry_run),
        suite_budget_estimated_usd=budget,
        provenance=provenance,
        backends=backends,
        models=models,
        settings=_manifest_settings(args, items=items),
        pricing=pricing,
    )

    run_dir = output_root / run_id
    if run_dir.exists():
        if not run_dir.is_dir():
            raise ProvenanceError(f"run 경로가 디렉터리가 아닙니다: {run_dir}")
        current_manifest = load_run_manifest(run_dir)
        if current_manifest["immutable_sha256"] != expected_manifest["immutable_sha256"]:
            raise ProvenanceError("기존 suite의 불변 설정과 새 실행 설정이 다릅니다.")
        unfinished = sorted(
            name
            for name, record in current_manifest["conditions"].items()
            if record.get("status") != "complete"
        )
        if unfinished:
            raise ProvenanceError(
                "완료되지 않은 선행 조건이 있어 suite를 계속할 수 없습니다: "
                + ", ".join(unfinished)
            )
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        create_run_manifest(run_dir, expected_manifest)
        current_manifest = load_run_manifest(run_dir)

    prior_suite_spend = nonnegative_finite_float(
        current_manifest["suite_estimated_spend_usd"],
        name="manifest.suite_estimated_spend_usd",
    )
    if prior_suite_spend >= budget:
        raise ValueError(
            "suite 공유 예산이 이미 소진되어 새 조건을 시작하지 않습니다."
        )

    result_path, summary_path = _output_paths(
        run_dir, args.condition, args.split, dry_run=args.dry_run
    )
    provenance_sha256 = expected_manifest["immutable_sha256"]

    rows: list[dict[str, Any]] = []
    condition_spend = 0.0
    budget_exceeded = False
    usage_unknown = False
    observed_models: list[str] = []
    condition_started = False
    os.environ["LANGCHAIN_PROJECT"] = f"vision-insp-{run_id}-{args.condition}"
    try:
        start_condition(
            run_dir,
            expected_manifest,
            condition=args.condition,
        )
        condition_started = True
        if args.push_dataset:
            _push_dataset_once(samples, split=args.split)

        async with AsyncExitStack() as stack:
            stack.enter_context(
                tracing_context(
                    project_name=f"vision-insp-{run_id}-{args.condition}",
                    enabled=False if args.dry_run else None,
                )
            )
            try:
                deps, graph, setup_usage = await stack.enter_async_context(
                    _runtime(args.condition)
                )
            except SetupUsageError as exc:
                record_condition_setup_usage(
                    run_dir,
                    condition=args.condition,
                    setup_usage=exc.setup_usage,
                )
                raise
            setup_progress = record_condition_setup_usage(
                run_dir,
                condition=args.condition,
                setup_usage=setup_usage,
            )
            condition_spend = nonnegative_finite_float(
                setup_usage["estimated_usd"],
                name="setup_usage.estimated_usd",
            )
            observed_models.extend(setup_usage["observed_model_ids"])
            setup_requests = nonnegative_int(
                setup_usage["successful_provider_requests"],
                name="setup_usage.successful_provider_requests",
            )
            if setup_requests and (
                setup_usage["usage_status"] == "unknown_tokens_after_success"
                or not setup_usage["observed_model_ids"]
            ):
                raise UsageValueError(
                    "조건 초기화 provider 응답의 token usage 또는 "
                    "관측 model ID를 확인할 수 없습니다."
                )
            if not setup_progress["conditions"][args.condition].get("setup_usage"):
                raise RuntimeError("조건 초기화 usage가 manifest에 기록되지 않았습니다.")
            with result_path.open("x", encoding="utf-8") as output:
                for index, sample in enumerate(samples, start=1):
                    suite_spend_before_call = prior_suite_spend + condition_spend
                    if suite_spend_before_call >= budget:
                        budget_exceeded = True
                        print(
                            "비용 가드 중단: "
                            f"suite estimated_usd ${suite_spend_before_call:.6f} "
                            f">= ${budget:.6f}",
                            file=sys.stderr,
                        )
                        break

                    row = _empty_row(
                        sample,
                        args.condition,
                        run_id=run_id,
                        provenance_sha256=provenance_sha256,
                        dry_run=bool(args.dry_run),
                    )
                    started = time.perf_counter()
                    try:
                        result, langsmith_run_id = await _evaluate_sample(
                            sample, args.condition, deps, graph
                        )
                        report = result.get("report")
                        report = report if isinstance(report, dict) else {}
                        events = result.get("events")
                        events = events if isinstance(events, list) else []
                        first_errors, would_fail = _validation_details(events)
                        approval = result.get("approval")
                        approval_required = bool(
                            approval.get("required")
                            if isinstance(approval, dict) else False
                        ) or any(
                            isinstance(value, dict)
                            and value.get("type") == "ApprovalRequired"
                            for value in events
                        )
                        usage = _usage_fields(result)
                        if usage["usage_status"] == "unknown_tokens_after_success":
                            usage_unknown = True
                        model_ids = _observed_model_ids(result)
                        observed_models.extend(model_ids)
                        validation_attempts = nonnegative_int(
                            result.get("validation_attempts", MISSING),
                            name="result.validation_attempts",
                            default=0,
                        )
                        retry_count = nonnegative_int(
                            result.get("retry_count", MISSING),
                            name="result.retry_count",
                            default=0,
                        )
                        replan_rounds = nonnegative_int(
                            result.get("replan_rounds", MISSING),
                            name="result.replan_rounds",
                            default=0,
                        )
                        row.update({
                            "verdict": result.get(
                                "verdict", report.get("verdict")
                            ),
                            "defect_type": result.get(
                                "defect_type", report.get("defect_type")
                            ),
                            "confidence": result.get(
                                "confidence", report.get("confidence")
                            ),
                            "abstain_reason": result.get(
                                "abstain_reason", report.get("abstain_reason")
                            ),
                            "approval_required": approval_required,
                            "validation_attempts": validation_attempts,
                            "validation_rejections": _validation_rejections(events),
                            "retry_count": retry_count,
                            "replan_rounds": replan_rounds,
                            "validation_errors_first": first_errors,
                            "would_fail": would_fail,
                            "observations_text": _observations_text(result, report),
                            "pre_validation_snapshot": result.get(
                                "pre_validation_snapshot", {}
                            ),
                            "report": report,
                            **usage,
                            "observed_model_ids": model_ids,
                            "langsmith_run_id": langsmith_run_id,
                        })
                    except Exception as exc:  # 오류도 분모에 남기되 유료 사용량은 지어내지 않는다.
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        if args.dry_run and _selected_backends() == (
                            "fake", "fake", "fake"
                        ):
                            row["usage_status"] = "fake_error_no_billing"
                            row["cost_basis"] = "fake_no_billing"
                        else:
                            row["usage_status"] = "unknown_after_error"
                            usage_unknown = True
                    row["latency_s"] = nonnegative_finite_float(
                        round(time.perf_counter() - started, 6),
                        name="row.latency_s",
                    )
                    if tuple(row) != REQUIRED_RESULT_ROW_KEYS:
                        raise RuntimeError("평가 행 스키마 순서가 예상과 다릅니다.")
                    output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    output.flush()
                    rows.append(row)
                    condition_spend = round(
                        condition_spend + float(row["estimated_usd"]), 6
                    )
                    suite_spend = round(prior_suite_spend + condition_spend, 6)
                    record_condition_progress(
                        run_dir,
                        condition=args.condition,
                        processed_samples=len(rows),
                        estimated_spend_usd=condition_spend,
                    )
                    print(
                        f"[{index}/{len(samples)}] {row['id']} "
                        f"verdict={row['verdict']} "
                        f"estimated_usd={row['estimated_usd']:.6f}"
                    )
                    if usage_unknown:
                        print(
                            "사용량을 확인할 수 없는 실측 오류가 발생해 suite를 중단합니다.",
                            file=sys.stderr,
                        )
                        break
                    if suite_spend >= budget:
                        budget_exceeded = True
                        if index < len(samples):
                            print(
                                "비용 가드 중단: "
                                f"suite estimated_usd ${suite_spend:.6f} "
                                f">= ${budget:.6f}",
                                file=sys.stderr,
                            )
                            break

        complete_capture = len(rows) == len(samples) and not usage_unknown
        suite_estimated_spend = round(prior_suite_spend + condition_spend, 6)
        summary = {
            "run_id": run_id,
            "provenance_sha256": provenance_sha256,
            "condition": args.condition,
            "split": args.split,
            "dry_run": bool(args.dry_run),
            "requested_samples": len(samples),
            "processed_samples": len(rows),
            "missing_samples": len(samples) - len(rows),
            "complete_capture": complete_capture,
            "error_count": sum(bool(row.get("error")) for row in rows),
            "usage_unknown_count": sum(
                row.get("usage_status") in {
                    "unknown_after_error",
                    "unknown_tokens_after_success",
                }
                for row in rows
            ),
            "condition_estimated_usd": condition_spend,
            "setup_usage": setup_usage,
            "suite_estimated_usd": suite_estimated_spend,
            "suite_budget_estimated_usd": budget,
            "budget_guard": "suite_soft_post_sample_v2",
            "budget_exceeded": budget_exceeded,
            # 비용 가드로 중단된 표본도 사전 등록 분모에서 빼지 않는다.
            "binary_metrics": binary_metrics(rows, samples),
            "cost_latency": cost_latency(rows),
            "result_path": result_path.name,
        }
        atomic_write_json(summary_path, summary)
        if not observed_models:
            raise RuntimeError("실제 응답에서 관측된 model ID가 없습니다.")
        complete_condition(
            run_dir,
            condition=args.condition,
            result_path=result_path,
            summary_path=summary_path,
            complete_capture=complete_capture,
            observed_model_ids=list(dict.fromkeys(observed_models)),
        )
        return result_path, summary_path, summary
    except Exception as exc:
        if condition_started:
            try:
                current = load_run_manifest(run_dir)
                record = current.get("conditions", {}).get(args.condition, {})
                if record.get("status") == "running":
                    fail_condition(
                        run_dir,
                        condition=args.condition,
                        error=f"{type(exc).__name__}: {exc}",
                    )
            except Exception:
                # 원래 실패를 보존한다. manifest 오류는 다음 검증에서 fail-closed된다.
                pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(out=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--split", choices=("dev", "heldout"), required=True)
    parser.add_argument("--items", help="쉼표로 구분한 품목 필터")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--run-id",
        help="같은 평가 suite의 조건이 공유할 고유 식별자",
    )
    parser.add_argument(
        "--suite-conditions",
        help="현재 run에 사전 고정할 조건 목록(예: B0,B1,A)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="fake 백엔드 로컬 검증 허용"
    )
    parser.add_argument(
        "--push-dataset", action="store_true", help="LangSmith Dataset 1회 생성"
    )
    parser.add_argument(
        "--final-run",
        action="store_true",
        help="동결된 heldout 최종 실행임을 명시",
    )
    parser.add_argument(
        "--freeze-manifest",
        help="heldout 실행에 사용할 동결 manifest JSON 경로",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result_path, summary_path, summary = asyncio.run(_run(args))
    print(
        f"완료: {summary['processed_samples']}건, 오류 {summary['error_count']}건, "
        f"suite estimated_usd ${summary['suite_estimated_usd']:.6f}"
    )
    print(f"결과: {result_path}")
    print(f"요약: {summary_path}")


if __name__ == "__main__":
    main()
