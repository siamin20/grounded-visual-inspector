"""Before(B0/B1) 기준선의 보고서와 기준서 전문 전달 계약."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.supervisor_v1 import _normalize_baseline_result, run_v1
from agent.vision.backends import _normalize_result


FULL_SPEC = """# screw 검사 기준서

## 결함유형: scratch_neck
- [SCREW-04] 목 부위에 선형 긁힘이 보이면 불합격한다.
"""


class BaselineMCP:
    def __init__(self, result: dict | None = None) -> None:
        self.tool_calls: list[dict] = []
        self.resource_reads: list[str] = []
        self.result = result if result is not None else {
            "verdict": "REJECT",
            "defect_type": "scratch_neck",
            "observations": ["목 부위에 선형 긁힘이 보입니다."],
            "anomaly_seen": True,
            "text": "목 부위에 선형 긁힘이 보입니다.",
            "tokens": 7,
            "usd": 0.0,
        }

    async def find_tool(self, name: str) -> str:
        assert name == "inspect_image"
        return "vision"

    async def call_tool(
        self, name: str, arguments: dict, *, server: str
    ) -> dict:
        assert name == "inspect_image"
        assert server == "vision"
        self.tool_calls.append(dict(arguments))
        return dict(self.result)

    async def find_resource_template(self, name: str) -> tuple[str, str]:
        assert name == "get_spec"
        return "vision", "spec://{item}"

    async def read_resource(self, uri: str, *, server: str) -> str:
        assert server == "vision"
        self.resource_reads.append(uri)
        return FULL_SPEC


@pytest.mark.parametrize("alias", [None, "good", "none", "null", "\uc815\uc0c1"])
def test_baseline_direct_normalizer_requires_explicit_json_null(alias) -> None:
    raw = {
        "verdict": "PASS",
        "observations": ["\ud45c\uba74\uacfc \uc724\uacfd\uc5d0 \uc774\uc0c1\uc774 \ubcf4\uc774\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4."],
    }
    if alias is not None:
        raw["defect_type"] = alias

    verdict, defect_type, reason, _ = _normalize_baseline_result(
        raw,
        "screw",
        {"screw": {"good": {}, "scratch_neck": {}}},
    )

    assert verdict == "ABSTAIN"
    assert defect_type is None
    assert reason == "validation_failure"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["B0", "B1"])
async def test_baseline_report_schema_and_b1_full_spec_context(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_mvtec_paths: dict[str, Path],
    mode: str,
) -> None:
    monkeypatch.setenv("BASELINE_MODE", mode)
    mcp = BaselineMCP()

    result = await run_v1(
        SimpleNamespace(mcp=mcp, hitl_mode="auto_approve"),
        "검사해줘",
        f"baseline-{mode}",
        [],
        image_path=str(synthetic_mvtec_paths["scratch_neck"]),
        item="screw",
    )

    assert len(mcp.tool_calls) == 1
    call = mcp.tool_calls[0]
    assert call["style"] == "baseline"
    assert call["model_role"] == "observer"
    assert "scratch_neck" in call["extra_context"]
    if mode == "B1":
        assert mcp.resource_reads == ["spec://screw"]
        assert FULL_SPEC in call["extra_context"]
        assert "[SCREW-04]" in call["extra_context"]
    else:
        assert mcp.resource_reads == []
        assert FULL_SPEC not in call["extra_context"]
        assert "[SCREW-04]" not in call["extra_context"]

    report = result["report"]
    assert set(report) == {
        "verdict",
        "defect_type",
        "confidence",
        "abstain_reason",
        "observations",
        "spec_clauses",
        "actions",
        "lot",
        "summary",
    }
    assert report["verdict"] == "REJECT"
    assert report["defect_type"] == "scratch_neck"
    assert report["confidence"] in {"high", "low"}
    assert report["abstain_reason"] is None
    assert report["observations"] == [{
        "text": "목 부위에 선형 긁힘이 보입니다.",
        "source_call_id": "baseline-full",
    }]
    assert isinstance(report["spec_clauses"], list)
    assert isinstance(report["actions"], list) and report["actions"]
    assert isinstance(report["lot"], dict)
    assert isinstance(report["summary"], str) and report["summary"]
    assert "**REJECT**" in result["final"]
    assert "Validation" not in [event["type"] for event in result["events"]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("vlm_result", "reason"),
    [
        (
            {
                "verdict": "ABSTAIN",
                "defect_type": None,
                "observations": ["판독 불가"],
                "anomaly_seen": None,
                "text": "판독 불가",
            },
            "insufficient_visual_evidence",
        ),
        (
            {
                "verdict": "MAYBE",
                "defect_type": "scratch_neck",
                "observations": ["목 부위에 선형 흔적이 보입니다."],
                "anomaly_seen": True,
                "text": "목 부위에 선형 흔적이 보입니다.",
            },
            "validation_failure",
        ),
        (
            {
                "verdict": "REJECT",
                "defect_type": "unknown_type",
                "observations": ["표면에 이상 흔적이 보입니다."],
                "anomaly_seen": True,
                "text": "표면에 이상 흔적이 보입니다.",
            },
            "validation_failure",
        ),
        (
            {
                "verdict": "PASS",
                "defect_type": ["scratch_neck"],
                "observations": ["표면과 윤곽에 이상이 보이지 않습니다."],
                "anomaly_seen": False,
                "text": "표면과 윤곽에 이상이 보이지 않습니다.",
            },
            "validation_failure",
        ),
        ({}, "validation_failure"),
        (
            {
                "verdict": "PASS",
                "defect_type": None,
                "observations": [],
                "anomaly_seen": False,
                "text": "표면과 윤곽에 이상이 보이지 않습니다.",
            },
            "insufficient_visual_evidence",
        ),
    ],
)
async def test_baseline_unreadable_or_invalid_output_abstains(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_mvtec_paths: dict[str, Path],
    vlm_result: dict,
    reason: str,
) -> None:
    monkeypatch.setenv("BASELINE_MODE", "B0")
    result = await run_v1(
        SimpleNamespace(mcp=BaselineMCP(vlm_result), hitl_mode="auto_approve"),
        "검사해줘",
        "baseline-abstain",
        [],
        image_path=str(synthetic_mvtec_paths["scratch_neck"]),
        item="screw",
    )

    report = result["report"]
    assert report["verdict"] == "ABSTAIN"
    assert report["defect_type"] is None
    assert report["confidence"] == "low"
    assert report["abstain_reason"] == reason
    assert report["spec_clauses"] == []
    assert report["actions"] == ["재검사"]
    assert result["abstain_reason"] == reason
    assert result["approval"] == {
        "required": True,
        "decision": "approve",
        "note": "",
    }
    event_types = [event["type"] for event in result["events"]]
    assert event_types.count("ApprovalRequired") == 1
    assert event_types.count("ApprovalResolved") == 1
    analysis = next(
        event for event in result["events"] if event["type"] == "AnalysisDecided"
    )
    assert analysis["verdict"] == "ABSTAIN"
    assert analysis["abstain_reason"] == reason
    assert result["report"]["verdict"] == "ABSTAIN"


@pytest.mark.asyncio
async def test_baseline_valid_pass_keeps_empty_spec_exception(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_mvtec_paths: dict[str, Path],
) -> None:
    monkeypatch.setenv("BASELINE_MODE", "B0")
    mcp = BaselineMCP({
        "verdict": "PASS",
        "defect_type": None,
        "observations": ["표면과 윤곽에 눈에 띄는 이상이 보이지 않습니다."],
        "anomaly_seen": False,
        "text": "표면과 윤곽에 눈에 띄는 이상이 보이지 않습니다.",
    })

    result = await run_v1(
        SimpleNamespace(mcp=mcp, hitl_mode="auto_approve"),
        "검사해줘",
        "baseline-pass",
        [],
        image_path=str(synthetic_mvtec_paths["good"]),
        item="screw",
    )

    assert result["report"]["verdict"] == "PASS"
    assert result["report"]["confidence"] == "high"
    assert result["report"]["abstain_reason"] is None
    assert result["report"]["spec_clauses"] == []
    assert result["report"]["actions"] == ["통과"]


@pytest.mark.asyncio
async def test_backend_validation_failure_reason_survives_supervisor(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_mvtec_paths: dict[str, Path],
) -> None:
    monkeypatch.setenv("BASELINE_MODE", "B0")
    backend_result = _normalize_result(
        {},
        style="baseline",
        model="test",
        tokens=0,
        usd=0.0,
        item="screw",
    )

    result = await run_v1(
        SimpleNamespace(
            mcp=BaselineMCP(backend_result),
            hitl_mode="auto_approve",
        ),
        "검사해줘",
        "baseline-validation-chain",
        [],
        image_path=str(synthetic_mvtec_paths["scratch_neck"]),
        item="screw",
    )

    assert backend_result["abstain_reason"] == "validation_failure"
    assert result["report"]["verdict"] == "ABSTAIN"
    assert result["report"]["abstain_reason"] == "validation_failure"
    assert result["report"]["actions"] == ["재검사"]
