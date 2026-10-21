"""실제 stdio subprocess를 통한 MCP vision 공개 계약 테스트."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.mcp_client import MCPClient
from agent.retry import with_retry
from mcp_server_vision import InspectResult


def _configure_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BACKEND", "fake")
    monkeypatch.setenv("VLM_BACKEND", "fake")
    monkeypatch.setenv("EMBED_BACKEND", "fake")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")


def _inspect_result_kwargs() -> dict[str, object]:
    return {
        "text": "이상이 보이지 않습니다.",
        "anomaly_seen": False,
        "model": "openai:observed",
        "tokens": 3,
        "usd": 0.0,
        "input_tokens": 2,
        "output_tokens": 1,
        "token_accounting": "provider_split",
        "estimated_usd": 0.0,
        "compute_proxy_usd": 0.0,
        "requested_model": "openai:requested",
        "observed_model": "openai:observed",
        "successful_provider_requests": 1,
        "unknown_token_usage_successes": 0,
        "observed_model_response_count": 1,
        "unobserved_model_successes": 0,
        "observed_model_ids": ["openai:observed"],
        "region": "full",
        "flip": False,
    }


@pytest.mark.parametrize(
    "override",
    [
        {"tokens": -1},
        {"tokens": True},
        {"usd": float("nan")},
        {"estimated_usd": float("inf")},
        {"compute_proxy_usd": -0.1},
        {"successful_provider_requests": 2},
        {"unknown_token_usage_successes": 2},
        {"observed_model_response_count": 1, "observed_model_ids": []},
        {"observed_model": "openai:other"},
        {"model": "openai:other"},
        {
            "token_accounting": "total_only",
            "tokens": 0,
            "input_tokens": None,
            "output_tokens": None,
        },
    ],
)
def test_inspect_result_rejects_invalid_usage_contract(
    override: dict[str, object],
) -> None:
    values = _inspect_result_kwargs()
    values.update(override)

    with pytest.raises(ValueError):
        InspectResult(**values)


def test_inspect_result_preserves_unknown_observed_model() -> None:
    """응답 model이 비었을 때 요청 model로 대체하지 않는다."""
    serialized = InspectResult(
        text="이상이 보이지 않습니다.",
        anomaly_seen=False,
        model=None,
        tokens=0,
        usd=0.0,
        input_tokens=0,
        output_tokens=0,
        token_accounting="provider_split",
        estimated_usd=0.0,
        compute_proxy_usd=0.0,
        requested_model="openai:requested",
        observed_model=None,
        successful_provider_requests=1,
        unknown_token_usage_successes=0,
        observed_model_response_count=0,
        unobserved_model_successes=1,
        observed_model_ids=[],
        region="full",
        flip=False,
    ).model_dump()

    assert serialized["model"] is None
    assert serialized["observed_model"] is None
    assert serialized["requested_model"] == "openai:requested"
    assert serialized["successful_provider_requests"] == 1


@pytest.mark.asyncio
async def test_vision_server_exposes_tool_resources_and_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    supervisor_deps,
) -> None:
    _configure_fake(monkeypatch)
    image_path = tmp_path / "screw" / "test" / "good" / "000.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fake-image-bytes")
    async with supervisor_deps() as deps:
        client = deps.mcp

        assert list(client.sessions) == ["vision", "docs"]

        tool_server = await client.find_tool("inspect_image")
        tool_result = await client.call_tool(
            "inspect_image",
            {
                "image_path": str(image_path),
                "item": "screw",
                "region": "full",
                "style": "observe",
                "flip": False,
                "model_role": "observer",
            },
            server=tool_server,
        )
        baseline_result = await client.call_tool(
            "inspect_image",
            {
                "image_path": str(tmp_path / "screw" / "train" / "good" / "000.png"),
                "item": "screw",
                "region": "full",
                "style": "baseline",
                "flip": False,
                "model_role": "observer",
            },
            server=tool_server,
        )

        spec_server, spec_template = await client.find_resource_template("get_spec")
        spec = await client.read_resource("spec://screw", server=spec_server)
        guide_server, guide_template = await client.find_resource_template("get_guide")
        guide = await client.read_resource(
            "guide://scratch_neck",
            server=guide_server,
        )

        prompt_server = await client.find_prompt("inspection_prompt")
        prompt = await client.get_prompt(
            "inspection_prompt",
            {"item": "screw", "lot": "LOT-TEST"},
            server=prompt_server,
        )

    assert tool_server == "vision"
    assert tool_result["anomaly_seen"] is False
    assert tool_result["region"] == "full"
    assert tool_result["flip"] is False
    assert tool_result["model"] == "fake-vlm"
    assert tool_result["requested_model"] == "fake-vlm"
    assert tool_result["observed_model"] == "fake-vlm"
    assert tool_result["token_accounting"] == "estimated_split"
    assert (
        tool_result["input_tokens"] + tool_result["output_tokens"]
        == tool_result["tokens"]
    )
    assert tool_result["estimated_usd"] == 0.0
    assert tool_result["compute_proxy_usd"] == 0.0
    assert tool_result["successful_provider_requests"] == 0
    assert "abstain_reason" not in tool_result
    assert baseline_result["verdict"] == "ABSTAIN"
    assert baseline_result["defect_type"] is None
    assert baseline_result["abstain_reason"] == "insufficient_visual_evidence"

    assert (spec_server, spec_template) == ("vision", "spec://{item}")
    assert "SCREW-01" in spec
    assert (guide_server, guide_template) == (
        "vision",
        "guide://{defect_type}",
    )
    assert "scratch_neck" in guide

    assert prompt_server == "vision"
    assert "품목: screw" in prompt
    assert "로트: LOT-TEST" in prompt
    assert "관찰" in prompt and "기준 대조" in prompt and "보고" in prompt


@pytest.mark.asyncio
async def test_first_vision_timeout_recovers_through_with_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_fake(monkeypatch)
    image_path = tmp_path / "screw" / "test" / "scratch_neck" / "000.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fake-image-bytes")
    client = MCPClient(flaky_first_tool_call=True)

    try:
        await client.start()
        tool_server = await client.find_tool("inspect_image")
        result, retry_count = await with_retry(
            client.call_tool,
            "inspect_image",
            {
                "image_path": str(image_path),
                "item": "screw",
                "region": "full",
                "style": "observe",
                "flip": False,
                "model_role": "observer",
            },
            server=tool_server,
            retries=2,
            base_delay=0,
        )
    finally:
        await client.stop()

    assert retry_count == 1
    assert result["anomaly_seen"] is True
    assert result["text"] != "판독 불가"
