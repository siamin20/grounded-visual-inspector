"""Phase 6 계약 테스트가 공유하는 최소 합성 fixture."""

from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from PIL import Image

from agent.deps import SupervisorDeps
from agent.fake_llm import FastLLM, QualityLLM
from agent.mcp_client import MCPClient
from agent.rag import FakeEmbedder, SpecIndex
from agent.session_store import SqliteSessionStore


@pytest.fixture
def inspection_state() -> dict:
    """노드 규칙 테스트용으로 서로 일관된 정상 보고서 상태를 만든다."""
    text = "목 부위에 선형 긁힘이 보입니다."
    return {
        "item": "screw",
        "observations": [
            {
                "call_id": "obs-full",
                "region": "full",
                "text": text,
                "anomaly_seen": True,
                "regions": ["tl"],
                "features": ["선형 긁힘"],
                "model": "fake-observer",
            }
        ],
        "cross_check": {
            "call_id": "cc-full",
            "anomaly_seen": True,
            "regions": ["tl"],
            "summary": text,
            "model": "fake-cross",
        },
        "judgement": {
            "violations": ["SCREW-04"],
            "defect_type": "scratch_neck",
        },
        "verdict": "REJECT",
        "defect_type": "scratch_neck",
        "confidence": "high",
        "abstain_reason": None,
        "report": {
            "verdict": "REJECT",
            "defect_type": "scratch_neck",
            "confidence": "high",
            "abstain_reason": None,
            "observations": [
                {"text": text, "source_call_id": "obs-full"}
            ],
            "spec_clauses": ["SCREW-04"],
            "actions": ["재작업"],
            "lot": {},
            "summary": (
                "관찰 근거(목 부위에 선형 긁힘이 보입니다.)와 위반 조항(SCREW-04)"
                "을 바탕으로 scratch_neck 결함을 REJECT로 판정하고 "
                "재작업 조치를 요청합니다."
            ),
        },
        "events": [],
        "validation_attempts": 1,
        "ablation": "",
    }


@pytest.fixture
def synthetic_mvtec_paths(tmp_path: Path) -> dict[str, Path]:
    """시각 확인 없이 생성한 정상·결함 PNG 경로를 반환한다."""
    paths: dict[str, Path] = {}
    samples = (
        ("good", "screw", "good", "white"),
        ("scratch_neck", "screw", "scratch_neck", "gray"),
        ("bottle_broken_small", "bottle", "broken_small", "silver"),
    )
    for key, item, label, color in samples:
        path = tmp_path / "mvtec" / item / "test" / label / "000.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), color).save(path, format="PNG")
        paths[key] = path
    return paths


@pytest.fixture
def minimal_specs_dir(tmp_path: Path) -> Path:
    """graph E2E에 필요한 screw 조항만 가진 임시 기준서 디렉터리."""
    path = tmp_path / "specs"
    path.mkdir()
    (path / "screw.md").write_text(
        """# screw 검사 기준서

## 결함유형: good
- [SCREW-01] 전체 윤곽과 표면에 이상이 보이지 않으면 합격한다.

## 결함유형: scratch_neck
- [SCREW-04] 목 부위에 선형 긁힘이 보이면 불합격한다.
""",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def minimal_guides_dir(tmp_path: Path) -> Path:
    """임시 결함 조치 가이드 디렉터리를 만든다."""
    path = tmp_path / "guides"
    path.mkdir()
    (path / "scratch_neck.md").write_text(
        "# scratch_neck 조치 가이드\n\n조치 옵션: 재작업\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def supervisor_deps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    minimal_specs_dir: Path,
    minimal_guides_dir: Path,
) -> object:
    """같은 async task에서 실제 MCP 의존성을 여닫는 factory를 반환한다."""
    monkeypatch.setenv("LLM_BACKEND", "fake")
    monkeypatch.setenv("VLM_BACKEND", "fake")
    monkeypatch.setenv("EMBED_BACKEND", "fake")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")

    @asynccontextmanager
    async def create():
        mcp = MCPClient(flaky_first_tool_call=False)
        session_store = SqliteSessionStore(tmp_path / "session.db")
        await mcp.start()
        deps = SupervisorDeps(
            fast_llm=FastLLM("fake-fast", 0.0),
            quality_llm=QualityLLM("fake-quality", 0.0),
            mcp=mcp,
            session_store=session_store,
            version="v2",
            vector_store=SpecIndex.from_dir(minimal_specs_dir, FakeEmbedder()),
            hitl_mode="auto_approve",
        )
        try:
            yield deps
        finally:
            await mcp.stop()
            session_store.close()

    return create
