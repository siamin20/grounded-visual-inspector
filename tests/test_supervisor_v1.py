"""Before(B0/B1) 기준선의 단일 VLM 호출 계약 테스트."""

from types import SimpleNamespace

import pytest

from agent.supervisor_v1 import run_v1


class RecordingMCP:
    def __init__(self) -> None:
        self.tool_calls: list[dict] = []
        self.resource_reads: list[str] = []

    async def find_tool(self, name: str) -> str:
        assert name == "inspect_image"
        return "vision"

    async def call_tool(self, name: str, arguments: dict, *, server: str) -> dict:
        assert name == "inspect_image"
        assert server == "vision"
        self.tool_calls.append(dict(arguments))
        return {
            "verdict": "REJECT",
            "defect_type": "scratch_neck",
            "observations": ["목 부위에 선형 긁힘이 보입니다."],
            "anomaly_seen": True,
            "text": "목 부위에 선형 긁힘이 보입니다.",
            "usd": 0.0,
        }

    async def find_resource_template(self, name: str) -> tuple[str, str]:
        assert name == "get_spec"
        return "vision", "spec://{item}"

    async def read_resource(self, uri: str, *, server: str) -> str:
        assert server == "vision"
        self.resource_reads.append(uri)
        return "# screw 검사 기준서\n- [SCREW-04] 목 부위 긁힘"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode, expected_reads", [("B0", []), ("B1", ["spec://screw"])])
async def test_baseline_calls_vlm_once_and_b1_adds_full_spec(
    monkeypatch, mode: str, expected_reads: list[str]
) -> None:
    monkeypatch.setenv("BASELINE_MODE", mode)
    mcp = RecordingMCP()
    deps = SimpleNamespace(mcp=mcp, hitl_mode="auto_approve")
    result = await run_v1(
        deps,
        "검사해줘",
        "baseline-test",
        [],
        image_path="/synthetic/screw/test/scratch_neck/000.png",
        item="screw",
    )
    assert len(mcp.tool_calls) == 1
    assert mcp.tool_calls[0]["style"] == "baseline"
    assert "scratch_neck" in mcp.tool_calls[0]["extra_context"]
    assert mcp.resource_reads == expected_reads
    assert result["report"]["defect_type"] == "scratch_neck"
    assert "Validation" not in [event["type"] for event in result["events"]]
