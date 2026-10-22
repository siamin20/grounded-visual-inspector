"""all_total 통합 실습 — FastAPI 백엔드 (Wave 1·3·12·14 축 + 전체 조립).

- Wave 1  : lifespan에서 Agent 번들(MCP 연결·LLM·그래프)을 프로세스당 정확히 1회 생성,
            app.state로 모든 요청이 공유 (creation_count/instance_id로 증명)
- Wave 2  : vision·docs MCP 서버를 subprocess(stdio)로 기동해 연결 유지
- Wave 3  : GET /chat/stream — 그래프 실행 과정을 SSE 이벤트로 스트리밍
- Wave 12 : 대화 이력을 SQLite(session.db)에 저장 — 재시작해도 유지
- Wave 14 : SUPERVISOR_VERSION(v1/v2) 환경변수로 Before/에이전트 기준선 전환
- LangSmith: 요청당 chat_request 루트 run 아래로 노드·LLM·MCP 호출이 계층 기록됨
"""
import asyncio
import io
import json
import os
import re
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Literal
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()   # LANGCHAIN_TRACING_V2 / API_KEY / PROJECT — import보다 먼저

# fake 조합은 네트워크 없이 재현 가능해야 하므로 로컬 .env의 추적 설정도 끈다.
_llm_backend = os.environ.get("LLM_BACKEND", "fake").strip().lower()
_vlm_backend = os.environ.get("VLM_BACKEND", _llm_backend).strip().lower()
_embed_backend = os.environ.get("EMBED_BACKEND", _llm_backend).strip().lower()
if {_llm_backend, _vlm_backend, _embed_backend} == {"fake"}:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from agent.deps import (
    SupervisorDeps,
    validate_ablation,
    validate_hitl_mode,
    validate_supervisor_version,
)
from agent.fake_llm import FastLLM, QualityLLM
from agent.graph import build_graph
from agent.mcp_client import MCPClient
from agent.rag import SpecIndex, make_embedder
from agent.session_store import SqliteSessionStore
from agent.supervisor_v1 import build_v1_graph, validate_baseline_mode
from agent.typemap import load_type_map

CREATION_COUNT = 0   # Wave 1 — 프로세스 내 Agent 번들 생성 횟수 (항상 1이어야 정상)
ROOT = Path(__file__).resolve().parent
CHECKPOINT_DB_PATH = ROOT / "checkpoints.db"
UPLOAD_ROOT = ROOT / "data" / "uploads"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
_THREAD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class ChatRequest(BaseModel):   # Wave 1 — Pydantic 요청 검증
    message: str
    thread_id: str = "default"
    image_path: str | None = None
    item: str | None = None


class ResumeRequest(BaseModel):
    thread_id: str
    approval_id: str
    decision: Literal["approve", "reject"]
    note: str = ""


def _build_llms():
    """LLM_BACKEND=fake(기본, 결정론) | ollama(로컬 실모델) | openai(실제 OpenAI API)."""
    backend = os.environ.get("LLM_BACKEND", "fake").strip().lower()
    if backend == "ollama":
        from agent.ollama_llm import (
            OllamaFastLLM,
            OllamaJudgeLLM,
            OllamaQualityLLM,
        )
        fast = OllamaFastLLM(os.environ.get("OLLAMA_FAST_MODEL", "qwen3:0.6b"),
                             usd_per_1k_tokens=0.00015)
        judge = OllamaJudgeLLM(
            os.environ.get("OLLAMA_JUDGE_MODEL", "qwen3:8b"),
            usd_per_1k_tokens=0.005,
        )
        quality = OllamaQualityLLM(os.environ.get("OLLAMA_QUALITY_MODEL", "qwen3:8b"),
                                   usd_per_1k_tokens=0.005)
        # eval/run_eval.py도 사용하는 기존 3-tuple을 유지한다.
        # SupervisorDeps.__post_init__이 이 전담 의존성을 명시 필드로 옮긴다.
        fast.judge_llm = judge
        return backend, fast, quality
    if backend == "openai":
        from agent.openai_llm import OpenAIFastLLM, OpenAIQualityLLM
        fast = OpenAIFastLLM(os.environ.get("OPENAI_FAST_MODEL", "gpt-4o-mini"),
                             usd_per_1k_tokens=0.00015)
        quality = OpenAIQualityLLM(os.environ.get("OPENAI_QUALITY_MODEL", "gpt-4o"),
                                   usd_per_1k_tokens=0.005)
        return backend, fast, quality
    if backend == "fake":
        fast = FastLLM("fast-model-stub", usd_per_1k_tokens=0.00015)
        quality = QualityLLM("quality-model-stub", usd_per_1k_tokens=0.005)
        return backend, fast, quality
    raise ValueError(f"지원하지 않는 LLM_BACKEND입니다: {backend}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global CREATION_COUNT
    version = validate_supervisor_version(
        os.environ.get("SUPERVISOR_VERSION", "v2")
    )
    hitl_mode = validate_hitl_mode(os.environ.get("HITL_MODE", "interactive"))
    ablation = validate_ablation(os.environ.get("ABLATION", ""))
    baseline_mode = None
    if version == "v1":
        baseline_mode = validate_baseline_mode(
            os.environ.get("BASELINE_MODE", "B0")
        )
    CREATION_COUNT += 1
    mcp = MCPClient()
    await mcp.start()   # Wave 2 — 각 MCP subprocess 연결은 프로세스 수명 동안 1회 생성
    deps = None
    checkpointer_context = None
    try:
        backend, fast_llm, quality_llm = _build_llms()
        app.state.llm_backend = backend
        app.state.vlm_backend = (
            os.environ.get("VLM_BACKEND") or backend
        ).strip().lower()
        app.state.embed_backend = (
            os.environ.get("EMBED_BACKEND") or backend
        ).strip().lower()
        # Before(B0/B1)는 RAG를 사용하지 않지만 ABSTAIN HITL은
        # 재시작 후에도 재개해야 하므로 모든 버전이 checkpointer를 공유한다.
        vector_store = None
        if version == "v2":
            embedder = make_embedder()
            vector_store = SpecIndex.from_dir(ROOT / "data" / "specs", embedder)
        checkpoint_path = Path(
            os.environ.get("CHECKPOINT_DB_PATH", str(CHECKPOINT_DB_PATH))
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        # 3.1.2의 from_conn_string은 async context manager이며, 그래프와
        # SQLite 연결의 수명을 FastAPI lifespan과 같게 맞춘다.
        checkpointer_context = AsyncSqliteSaver.from_conn_string(
            str(checkpoint_path)
        )
        checkpointer = await checkpointer_context.__aenter__()
        deps = SupervisorDeps(
            fast_llm=fast_llm,
            quality_llm=quality_llm,
            mcp=mcp,
            session_store=SqliteSessionStore(),
            version=version,
            vector_store=vector_store,
            checkpointer=checkpointer,
            hitl_mode=hitl_mode,
            ablation=ablation,
        )
        app.state.deps = deps
        app.state.checkpointer = checkpointer
        app.state.checkpoint_namespace = (
            version if baseline_mode is None else f"{version}:{baseline_mode}"
        )
        # 같은 실행 조건·thread의 checkpoint를 동시에 갱신하지 않는다.
        # SSE 응답은 endpoint 반환 뒤 실행되므로 스트림 종료까지 예약을 유지한다.
        app.state.active_thread_requests = set()
        app.state.graph = (
            build_graph(deps, checkpointer=checkpointer)
            if deps.version == "v2"
            else build_v1_graph(deps, checkpointer=checkpointer)
        )
        app.state.instance_id = id(deps)
        yield
    finally:
        await mcp.stop()
        if deps is not None:
            deps.session_store.close()
        if checkpointer_context is not None:
            await checkpointer_context.__aexit__(None, None, None)
        try:   # 종료 전 LangSmith 배치 전송 플러시 — kill 시 run이 pending으로 남는 것 방지
            from langchain_core.tracers.langchain import wait_for_all_tracers
            wait_for_all_tracers()
        except Exception:
            pass


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "instance_id": app.state.instance_id,
            "creation_count": CREATION_COUNT,
            "supervisor_version": app.state.deps.version,
            "llm_backend": app.state.llm_backend,
            "vlm_backend": app.state.vlm_backend,
            "embed_backend": app.state.embed_backend,
            "hitl_mode": app.state.deps.hitl_mode,
            "ablation": app.state.deps.ablation,
            "fast_llm": app.state.deps.fast_llm.name,
            "judge_llm": app.state.deps.judge_llm.name,
            "quality_llm": app.state.deps.quality_llm.name}


@app.get("/version")
async def version():
    return {"supervisor_version": app.state.deps.version}


def _metadata(result: dict, run_id: str | None) -> dict:
    report = result.get("report") if isinstance(result.get("report"), dict) else {}
    approval = result.get("approval")
    return {
        "route": result.get("route"),
        "task_count": len(result.get("tasks", [])),
        "gate_removed": result.get("gate_removed", 0),
        "replan_rounds": result.get("replan_rounds", 0),
        "retry_count": result.get("retry_count", 0),
        "parallel_elapsed": result.get("parallel_elapsed"),
        "validation_attempts": result.get("validation_attempts", 0),
        "cost_usd": result.get("cost_usd", 0.0),
        "tokens": result.get("tokens", 0),
        "verdict": result.get("verdict", report.get("verdict")),
        "defect_type": result.get("defect_type", report.get("defect_type")),
        "confidence": result.get("confidence", report.get("confidence")),
        "abstain_reason": result.get(
            "abstain_reason", report.get("abstain_reason")
        ),
        "approval_required": bool(
            approval.get("required") if isinstance(approval, dict) else False
        ) or bool(result.get("__interrupt__")),
        "hitl_mode": app.state.deps.hitl_mode,
        "successful_provider_requests": result.get(
            "successful_provider_requests", 0
        ),
        "successful_vlm_requests": result.get("successful_vlm_requests", 0),
        "unknown_token_usage_successes": result.get(
            "unknown_token_usage_successes", 0
        ),
        "unobserved_model_successes": result.get(
            "unobserved_model_successes", 0
        ),
        "observed_model_response_count": result.get(
            "observed_model_response_count", 0
        ),
        "ablation": result.get("ablation", app.state.deps.ablation),
        "llm_backend": app.state.llm_backend,
        "supervisor_version": app.state.deps.version,
        "instance_id": app.state.instance_id,
        "langsmith_run_id": run_id,
    }


def _graph_config(thread_id: str) -> dict:
    """pipeline 버전별 저장용 thread ID를 구성한다."""
    return {
        "configurable": {
            "thread_id": f"{app.state.checkpoint_namespace}:{thread_id}",
        }
    }


async def _has_pending_interrupt(thread_id: str) -> bool:
    """동일 thread에 사람 승인 대기 중인 요청이 있는지 확인한다."""
    snapshot = await app.state.graph.aget_state(_graph_config(thread_id))
    return any(getattr(task, "interrupts", ()) for task in snapshot.tasks)


async def _ensure_thread_is_not_paused(thread_id: str) -> None:
    """사람 승인을 새 입력으로 덮어쓰는 것을 막는다."""
    if await _has_pending_interrupt(thread_id):
        raise HTTPException(
            status_code=409,
            detail="사람 승인 대기 중인 thread입니다. /chat/resume을 사용하세요.",
        )


def _reserve_thread(thread_id: str) -> str:
    """namespace·thread별 요청을 하나만 실행하도록 원자적으로 예약한다."""
    key = f"{app.state.checkpoint_namespace}:{thread_id}"
    active = app.state.active_thread_requests
    if key in active:
        raise HTTPException(
            status_code=409,
            detail="같은 thread의 요청이 이미 실행 중입니다.",
        )
    # 이벤트 루프는 await 지점에서만 다른 task로 전환하므로 검사와 추가가 한 단위다.
    active.add(key)
    return key


def _release_thread(key: str) -> None:
    """완료·예외·클라이언트 취소 모두에서 single-flight 예약을 해제한다."""
    app.state.active_thread_requests.discard(key)


async def _reserved_stream(source: AsyncIterator[str], key: str) -> AsyncIterator[str]:
    """SSE iterator가 자연 종료되거나 취소될 때까지 thread 예약을 유지한다."""
    try:
        async for chunk in source:
            yield chunk
    finally:
        _release_thread(key)


def _initial_state(
    message: str,
    thread_id: str,
    history: list[dict],
    image_path: str | None,
    item: str | None,
    ablation: str,
) -> dict:
    # SQLite checkpointer는 같은 thread_id의 이전 상태와 새 입력을 병합한다.
    # 요청 단위 필드를 명시적으로 비워 이전 승인·보고서가 재사용되지 않게 한다.
    return {
        "query": message,
        "thread_id": thread_id,
        "history": history,
        "route": "",
        "needs_analysis": False,
        "tasks": [],
        "gate_removed": 0,
        "collected": [],
        "replan_rounds": 0,
        "research_sufficient": False,
        "retry_count": 0,
        "parallel_elapsed": 0.0,
        "draft": "",
        "validation_attempts": 0,
        "validation_feedback": None,
        "validation_passed": False,
        "validation_terminal": False,
        "final": "",
        "last_prompt": "",
        "image_path": image_path,
        "item": item,
        "lot": {},
        "observations": [],
        "cross_check": {},
        "spec_hits": [],
        "judgement": {},
        "verdict": "",
        "defect_type": None,
        "confidence": "",
        "abstain_reason": None,
        "report": {},
        "approval": {},
        "reapproval_required": False,
        "validation_errors": [],
        "pre_validation_snapshot": {},
        "events": [],
        "cost_usd": 0.0,
        "tokens": 0,
        "input_tokens": None,
        "output_tokens": None,
        "token_accounting": None,
        "estimated_usd": 0.0,
        "compute_proxy_usd": 0.0,
        "observed_model_ids": [],
        "successful_provider_requests": 0,
        "successful_vlm_requests": 0,
        "unknown_token_usage_successes": 0,
        "unobserved_model_successes": 0,
        "observed_model_response_count": 0,
        "ablation": ablation,
        "hitl_rejections": 0,
        "_observer_result": {},
        "_cross_result": {},
    }


def _session_message(message: str, image_path: str | None) -> str:
    if image_path:
        return f"{message} [image: {Path(image_path).name}]"
    return message


def _save_session(
    deps: SupervisorDeps,
    thread_id: str,
    message: str,
    image_path: str | None,
    final: str,
) -> None:
    deps.session_store.append(
        thread_id, "user", _session_message(message, image_path)
    )
    deps.session_store.append(thread_id, "assistant", final)


@traceable(name="chat_request")
async def _handle_chat(
    message: str,
    thread_id: str,
    image_path: str | None = None,
    item: str | None = None,
) -> tuple[dict, str | None]:
    deps = app.state.deps
    await _ensure_thread_is_not_paused(thread_id)
    history = deps.session_store.load(thread_id)   # Wave 12
    result = await app.state.graph.ainvoke(
        _initial_state(
            message, thread_id, history, image_path, item, deps.ablation
        ),
        config=_graph_config(thread_id),
        durability="sync",
    )
    # interrupt 중인 요청은 최종 답이 없으므로 resume 완료 시점에 저장한다.
    if result.get("final"):
        _save_session(deps, thread_id, message, image_path, result["final"])
    rt = get_current_run_tree()
    return result, (str(rt.id) if rt else None)


@app.post("/chat")
async def chat(req: ChatRequest):
    thread_id = _validated_thread_id(req.thread_id)
    reservation = _reserve_thread(thread_id)
    try:
        result, run_id = await _handle_chat(
            req.message, thread_id, req.image_path, req.item
        )
    finally:
        _release_thread(reservation)
    raw_interrupt = result.get("__interrupt__")
    approval_id = (
        _interrupt_event(raw_interrupt, result.get("report")).get("approval_id")
        if raw_interrupt else None
    )
    return {"reply": result.get("final", ""), "events": result.get("events", []),
            "report": result.get("report"),
            "paused": bool(raw_interrupt),
            "approval_id": approval_id,
            "metadata": _metadata(result, run_id)}


def _sse(payload: Any) -> str:
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _interrupt_event(raw: Any, report: dict | None) -> dict:
    values = raw if isinstance(raw, (list, tuple)) else (raw,)
    first = values[0] if values else None
    value = getattr(first, "value", first)
    event = dict(value) if isinstance(value, dict) else {}
    event["type"] = "ApprovalRequired"
    event["required"] = True
    interrupt_id = getattr(first, "id", None)
    if isinstance(interrupt_id, str) and interrupt_id:
        event["approval_id"] = interrupt_id
    if not isinstance(event.get("report"), dict):
        event["report"] = report or {}
    return event


def _pending_interrupt_ids(snapshot: Any) -> tuple[str, ...]:
    """현재 checkpoint에서 재개 가능한 interrupt ID만 추출한다."""
    interrupt_ids: list[str] = []
    for task in snapshot.tasks:
        for pending in getattr(task, "interrupts", ()):
            interrupt_id = getattr(pending, "id", None)
            if isinstance(interrupt_id, str) and interrupt_id:
                interrupt_ids.append(interrupt_id)
    return tuple(interrupt_ids)


async def _stream_graph(
    graph_input: dict | Command,
    *,
    thread_id: str,
    session_context: dict,
    seen: int = 0,
) -> AsyncIterator[str]:
    """v1/v2 신규 요청과 resume가 같은 SSE 규칙을 쓰게 한다."""
    deps = app.state.deps
    final = str(session_context.get("final") or "")
    report = session_context.get("report")
    report = report if isinstance(report, dict) else None
    last_prompt = session_context.get("last_prompt")
    paused = False
    async for mode, payload in app.state.graph.astream(
        graph_input,
        config=_graph_config(thread_id),
        stream_mode=["updates", "custom"],
        durability="sync",
    ):
        if mode == "custom":
            yield _sse(payload)
            continue
        if not isinstance(payload, dict):
            continue
        if "__interrupt__" in payload:
            yield _sse(_interrupt_event(payload["__interrupt__"], report))
            paused = True
            continue
        for node_name, delta in payload.items():
            if not isinstance(delta, dict):
                continue
            events = delta.get("events")
            if isinstance(events, list):
                for value in events[seen:]:
                    if isinstance(value, dict):
                        yield _sse({**value, "node": node_name})
                seen = max(seen, len(events))
            if delta.get("final"):
                final = str(delta["final"])
            if isinstance(delta.get("report"), dict):
                report = delta["report"]
            if delta.get("last_prompt"):
                last_prompt = delta["last_prompt"]

    if paused:
        # interrupt chunk 직후 반복기를 닫으면 async SQLite 쓰기가
        # 취소될 수 있다. astream을 자연 종료시킨 뒤 멈춤을 알린다.
        yield _sse("[PAUSED]")
        return

    yield _sse(
        {
            "type": "FinalAnswer",
            "reply": final,
            "report": report,
            "prompt": last_prompt,
        }
    )
    _save_session(
        deps,
        thread_id,
        str(session_context.get("query") or ""),
        session_context.get("image_path"),
        final,
    )
    yield _sse("[DONE]")


async def _stream_request(
    message: str,
    thread_id: str,
    image_path: str | None,
    item: str | None,
) -> AsyncIterator[str]:
    deps = app.state.deps
    history = deps.session_store.load(thread_id)
    initial = _initial_state(
        message, thread_id, history, image_path, item, deps.ablation
    )
    async for chunk in _stream_graph(
        initial, thread_id=thread_id, session_context=initial
    ):
        yield chunk


def _validated_thread_id(thread_id: str | None, session_id: str | None = None) -> str:
    if thread_id == "" or session_id == "":
        raise HTTPException(status_code=422, detail="thread_id 형식이 올바르지 않습니다.")
    if thread_id and session_id and thread_id != session_id:
        raise HTTPException(status_code=422, detail="thread_id와 session_id가 다릅니다.")
    value = thread_id or session_id or "default"
    if not _THREAD_ID.fullmatch(value) or value in {".", ".."}:
        raise HTTPException(status_code=422, detail="thread_id 형식이 올바르지 않습니다.")
    return value


def _validated_inspect_thread_id(
    thread_id: str | None,
    session_id: str | None,
) -> str:
    """업로드는 명시적 thread ID 또는 호환용 session ID를 요구한다."""
    if thread_id is None and session_id is None:
        raise HTTPException(
            status_code=422,
            detail="thread_id 또는 session_id가 필요합니다.",
        )
    return _validated_thread_id(thread_id, session_id)


def _convert_to_png(content: bytes, destination: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(content)) as source:
            source.verify()
        with Image.open(io.BytesIO(content)) as source:
            source.load()
            detected_format = source.format
            if detected_format == "PNG":
                destination.write_bytes(content)
                return
            has_alpha = source.mode in {"RGBA", "LA"} or "transparency" in source.info
            prepared = source.convert("RGBA" if has_alpha else "RGB")
            prepared.save(destination, format="PNG")


async def _save_uploaded_png(file: UploadFile, thread_id: str) -> Path:
    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="이미지는 최대 10MB까지 업로드할 수 있습니다.")
    if not content:
        raise HTTPException(status_code=422, detail="빈 파일은 업로드할 수 없습니다.")

    directory = UPLOAD_ROOT / thread_id
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{uuid4()}.png"
    try:
        await asyncio.to_thread(_convert_to_png, content, destination)
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="유효한 이미지 파일이 아닙니다.") from exc
    return destination


@app.get("/chat/stream")
async def chat_stream(
    message: str,
    thread_id: str = "default",
    image_path: str | None = None,
    item: str | None = None,
):
    """Wave 3 — 그래프 진행 상황을 SSE 규약으로 내보낸다."""
    thread_id = _validated_thread_id(thread_id)
    reservation = _reserve_thread(thread_id)
    try:
        await _ensure_thread_is_not_paused(thread_id)
    except BaseException:
        _release_thread(reservation)
        raise
    return StreamingResponse(
        _reserved_stream(
            _stream_request(message, thread_id, image_path, item),
            reservation,
        ),
        media_type="text/event-stream",
    )


@app.post("/inspect")
async def inspect_image(
    file: UploadFile = File(...),
    item: str = Form(...),
    thread_id: str | None = Form(None),
    session_id: str | None = Form(None),
):
    """업로드를 검증해 UUID PNG로 저장한 뒤 검사를 스트리밍한다."""
    resolved_thread_id = _validated_inspect_thread_id(thread_id, session_id)
    if item not in load_type_map():
        raise HTTPException(status_code=422, detail="지원하지 않는 품목입니다.")
    reservation = _reserve_thread(resolved_thread_id)
    try:
        await _ensure_thread_is_not_paused(resolved_thread_id)
        image_path = await _save_uploaded_png(file, resolved_thread_id)
    except BaseException:
        _release_thread(reservation)
        raise
    return StreamingResponse(
        _reserved_stream(
            _stream_request("검사해줘", resolved_thread_id, str(image_path), item),
            reservation,
        ),
        media_type="text/event-stream",
    )


@app.post("/chat/resume")
async def chat_resume(req: ResumeRequest):
    """저장된 interrupt를 동일 thread_id에서 이어서 실행한다."""
    thread_id = _validated_thread_id(req.thread_id)
    reservation = _reserve_thread(thread_id)
    try:
        config = _graph_config(thread_id)
        snapshot = await app.state.graph.aget_state(config)
        pending_interrupt_ids = _pending_interrupt_ids(snapshot)
        if not pending_interrupt_ids:
            raise HTTPException(status_code=409, detail="재개할 중단 상태가 없습니다.")
        if pending_interrupt_ids != (req.approval_id,):
            raise HTTPException(
                status_code=409,
                detail="approval_id가 현재 재개 대기 상태와 일치하지 않습니다.",
            )
        context = dict(snapshot.values)
        events = context.get("events")
        seen = len(events) if isinstance(events, list) else 0
        command = Command(resume={
            req.approval_id: {"decision": req.decision, "note": req.note}
        })
    except BaseException:
        _release_thread(reservation)
        raise
    return StreamingResponse(
        _reserved_stream(
            _stream_graph(
                command,
                thread_id=thread_id,
                session_context=context,
                seen=seen,
            ),
            reservation,
        ),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8100, log_level="warning")
