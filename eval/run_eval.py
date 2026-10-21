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
from typing import Any, AsyncIterator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langsmith import Client, traceable, tracing_context
from langsmith.run_helpers import get_current_run_tree
from langsmith.utils import LangSmithNotFoundError

from agent.deps import SupervisorDeps
from agent.graph import build_graph
from agent.mcp_client import MCPClient
from agent.rag import SpecIndex, make_embedder
from agent.supervisor_v1 import run_v1
from eval.evaluators import binary_metrics, cost_latency
from eval.preflight import EvaluationPreflightError, guard_run_mode
from server import _build_llms, _initial_state


DATASET_PATH = ROOT / "eval" / "dataset.json"
DEFAULT_RESULTS_DIR = ROOT / "eval" / "results"
CONDITIONS = ("B0", "B1", "A", "A-X", "A-C", "A-V")
HELDOUT_CONDITIONS = frozenset({"B0", "B1", "A"})
ABLATIONS = {
    "A": "",
    "A-X": "no_cross",
    "A-C": "no_crops",
    "A-V": "no_validate",
}
REQUIRED_ROW_KEYS = (
    "id",
    "item",
    "label",
    "is_defect",
    "split",
    "condition",
    "verdict",
    "defect_type",
    "confidence",
    "abstain_reason",
    "approval_required",
    "validation_attempts",
    "validation_errors_first",
    "would_fail",
    "observations_text",
    "pre_validation_snapshot",
    "report",
    "latency_s",
    "tokens",
    "usd",
    "langsmith_run_id",
    "error",
)


class _NullSessionStore:
    """평가에서는 대화 DB를 만들지 않지만 deps 인터페이스는 유지한다."""

    def load(self, _thread_id: str) -> list[dict]:
        return []

    def append(self, _thread_id: str, _role: str, _content: str) -> None:
        return None

    def close(self) -> None:
        return None


def _selected_backends() -> tuple[str, str, str]:
    llm_backend = (os.environ.get("LLM_BACKEND") or "fake").strip().lower()
    vlm_backend = (
        os.environ.get("VLM_BACKEND") or llm_backend
    ).strip().lower()
    embed_backend = (
        os.environ.get("EMBED_BACKEND") or llm_backend
    ).strip().lower()
    return llm_backend, vlm_backend, embed_backend


def _guard_backends(*, dry_run: bool) -> None:
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


def _push_dataset_once(samples: list[dict[str, Any]], *, split: str) -> None:
    """명시적 플래그가 있을 때만 LangSmith Dataset을 한 번 생성한다."""
    name = f"vision-inspection-{split}"
    client = Client()
    try:
        client.read_dataset(dataset_name=name)
    except LangSmithNotFoundError:
        dataset = client.create_dataset(
            name,
            description="MVTec AD 이미지 경로와 폴더 라벨 기반 검사 평가셋",
        )
        client.create_examples(
            dataset_id=dataset.id,
            examples=[
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
            ],
        )
        print(f"LangSmith Dataset 생성: {name} ({len(samples)}건)")
    else:
        print(f"LangSmith Dataset이 이미 있어 업로드를 건너뜁니다: {name}")


@asynccontextmanager
async def _runtime(condition: str) -> AsyncIterator[tuple[SupervisorDeps, Any]]:
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
        if version == "v2":
            vector_store = SpecIndex.from_dir(
                ROOT / "data" / "specs", make_embedder()
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
        yield deps, graph
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
    """추적에는 평가 식별자만 남기고 실행 객체는 제외한다."""
    return {
        "sample": inputs.get("sample"),
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


def _empty_row(sample: dict[str, Any], condition: str) -> dict[str, Any]:
    return {
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
        "validation_errors_first": [],
        "would_fail": [],
        "observations_text": "",
        "pre_validation_snapshot": {},
        "report": {},
        "latency_s": 0.0,
        "tokens": 0,
        "usd": 0.0,
        "langsmith_run_id": None,
        "error": None,
    }


async def _run(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    _guard_backends(dry_run=args.dry_run)
    _guard_run_request(args)
    items = {
        value.strip() for value in (args.items or "").split(",") if value.strip()
    } or None
    samples = _load_samples(split=args.split, items=items, limit=args.limit)
    if args.push_dataset:
        _push_dataset_once(samples, split=args.split)

    output_dir = Path(args.out).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path, summary_path = _output_paths(
        output_dir, args.condition, args.split, dry_run=args.dry_run
    )
    budget = float(os.environ.get("EVAL_BUDGET_USD", "15"))
    if budget < 0:
        raise ValueError("EVAL_BUDGET_USD는 0 이상이어야 합니다.")

    rows: list[dict[str, Any]] = []
    cumulative_usd = 0.0
    budget_exceeded = False
    os.environ["LANGCHAIN_PROJECT"] = f"vision-insp-{args.condition}"
    async with AsyncExitStack() as stack:
        stack.enter_context(
            tracing_context(
                project_name=f"vision-insp-{args.condition}",
                enabled=False if args.dry_run else None,
            )
        )
        deps, graph = await stack.enter_async_context(_runtime(args.condition))
        with result_path.open("w", encoding="utf-8") as output:
            for index, sample in enumerate(samples, start=1):
                row = _empty_row(sample, args.condition)
                started = time.perf_counter()
                try:
                    result, run_id = await _evaluate_sample(
                        sample, args.condition, deps, graph
                    )
                    report = result.get("report")
                    report = report if isinstance(report, dict) else {}
                    first_errors, would_fail = _validation_details(
                        result.get("events")
                    )
                    approval = result.get("approval")
                    approval_required = bool(
                        approval.get("required")
                        if isinstance(approval, dict) else False
                    ) or any(
                        isinstance(value, dict)
                        and value.get("type") == "ApprovalRequired"
                        for value in result.get("events", [])
                    )
                    row.update({
                        "verdict": result.get("verdict", report.get("verdict")),
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
                        "validation_attempts": int(
                            result.get("validation_attempts", 0)
                        ),
                        "validation_errors_first": first_errors,
                        "would_fail": would_fail,
                        "observations_text": _observations_text(result, report),
                        "pre_validation_snapshot": result.get(
                            "pre_validation_snapshot", {}
                        ),
                        "report": report,
                        "tokens": int(result.get("tokens", 0)),
                        "usd": float(result.get("cost_usd", 0.0)),
                        "langsmith_run_id": run_id,
                    })
                except Exception as exc:  # 샘플 오류는 다음 평가를 막지 않는다.
                    row["error"] = f"{type(exc).__name__}: {exc}"
                row["latency_s"] = round(time.perf_counter() - started, 6)
                if tuple(row) != REQUIRED_ROW_KEYS:
                    raise RuntimeError("평가 행 스키마 순서가 예상과 다릅니다.")
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                rows.append(row)
                cumulative_usd = round(cumulative_usd + float(row["usd"]), 6)
                print(
                    f"[{index}/{len(samples)}] {row['id']} "
                    f"verdict={row['verdict']} usd={row['usd']:.6f}"
                )
                if cumulative_usd > budget:
                    budget_exceeded = True
                    print(
                        f"비용 가드 중단: ${cumulative_usd:.6f} > ${budget:.6f}",
                        file=sys.stderr,
                    )
                    break

    summary = {
        "condition": args.condition,
        "split": args.split,
        "dry_run": bool(args.dry_run),
        "requested_samples": len(samples),
        "processed_samples": len(rows),
        "missing_samples": len(samples) - len(rows),
        "complete_capture": len(rows) == len(samples),
        "error_count": sum(bool(row.get("error")) for row in rows),
        "cumulative_usd": cumulative_usd,
        "budget_usd": budget,
        "budget_exceeded": budget_exceeded,
        # 비용 가드로 중단된 표본도 사전 등록 분모에서 빼지 않는다.
        "binary_metrics": binary_metrics(rows, samples),
        "cost_latency": cost_latency(rows),
        "result_path": str(result_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result_path, summary_path, summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--split", choices=("dev", "heldout"), required=True)
    parser.add_argument("--items", help="쉼표로 구분한 품목 필터")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", default=str(DEFAULT_RESULTS_DIR))
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
        f"누적 ${summary['cumulative_usd']:.6f}"
    )
    print(f"결과: {result_path}")
    print(f"요약: {summary_path}")


if __name__ == "__main__":
    main()
