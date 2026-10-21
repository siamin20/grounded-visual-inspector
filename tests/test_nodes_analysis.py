"""R1 판독 가능성 및 analysis 결정표 계약 테스트."""

from __future__ import annotations

from types import SimpleNamespace
import unicodedata

import pytest

from agent.nodes.analysis import analysis_node
from agent.nodes.common import (
    cross_status,
    is_readable_observation,
    normalized_judgement_status,
    observer_status,
)
from agent.nodes.replan_check import replan_check_node


def _observer(status: str, *, positive_region: str = "tl") -> list[dict]:
    if status == "O+":
        return [{
            "call_id": "obs-full",
            "region": "full",
            "text": "좌상단에 선형 흔적이 보입니다.",
            "anomaly_seen": True,
            "regions": [positive_region],
            "features": ["선형 흔적"],
        }]
    if status == "O-":
        return [{
            "call_id": "obs-full",
            "region": "full",
            "text": "전체 표면에 특이 소견이 없습니다.",
            "anomaly_seen": False,
            "regions": [],
            "features": [],
        }]
    return [{
        "call_id": "obs-full",
        "region": "full",
        "text": "판독 불가",
        "anomaly_seen": None,
        "regions": [],
        "features": [],
    }]


def _cross(status: str, *, positive_region: str = "tl") -> dict:
    if status == "C+":
        return {
            "call_id": "cc-full",
            "summary": "좌상단 이상 항목은 예입니다.",
            "anomaly_seen": True,
            "regions": [positive_region],
        }
    if status == "C-":
        return {
            "call_id": "cc-full",
            "summary": "모든 이상 항목은 아니오입니다.",
            "anomaly_seen": False,
            "regions": [],
        }
    return {
        "call_id": "cc-full",
        "summary": "판독 불가",
        "anomaly_seen": None,
        "regions": [],
    }


def _judgement(status: str) -> dict:
    if status == "V+":
        return {
            "violations": ["SCREW-04"],
            "defect_type": "scratch_neck",
            "rationale": "관찰과 기준서가 일치합니다.",
            "status": "grounded",
        }
    if status == "V0":
        return {
            "violations": [],
            "defect_type": None,
            "rationale": "위반 조항이 없습니다.",
            "status": "none",
        }
    return {
        "violations": ["MADE-UP-99"],
        "defect_type": None,
        "rationale": "지원하지 않는 조항입니다.",
        "status": "invalid",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "observer", "cross", "judged", "overlap", "ablation",
        "verdict", "confidence", "reason", "agree",
    ),
    [
        # design.md §4.1 결정표 순서와 같은 순서다.
        ("O+", "C+", "V?", True, "", "ABSTAIN", "low", "validation_failure", True),
        ("O+", "C+", "V+", True, "", "REJECT", "high", None, True),
        ("O+", "C+", "V+", False, "", "REJECT", "low", None, False),
        ("O+", "C-", "V+", True, "", "REJECT", "low", None, False),
        ("O+", "C?", "V+", True, "", "REJECT", "low", None, False),
        ("O?", "C+", "V+", True, "", "REJECT", "low", None, False),
        ("O-", "C+", "V+", True, "", "REJECT", "low", None, False),
        ("O+", "C+", "V0", True, "", "ABSTAIN", "low", "ungrounded_anomaly", True),
        ("O+", "C-", "V0", True, "", "ABSTAIN", "low", "ungrounded_anomaly", False),
        ("O-", "C+", "V0", True, "", "ABSTAIN", "low", "ungrounded_anomaly", False),
        ("O-", "C-", "V+", True, "", "ABSTAIN", "low", "evidence_conflict", True),
        ("O-", "C-", "V0", True, "", "PASS", "high", None, True),
        ("O-", "C?", "V0", True, "", "PASS", "low", None, False),
        ("O?", "C-", "V0", True, "", "ABSTAIN", "low", "insufficient_visual_evidence", False),
        ("O?", "C?", "V0", True, "", "ABSTAIN", "low", "insufficient_visual_evidence", False),
        ("O+", "C?", "V+", True, "no_cross", "REJECT", "high", None, True),
        ("O-", "C?", "V0", True, "no_cross", "PASS", "high", None, True),
        ("O+", "C?", "V0", True, "no_cross", "ABSTAIN", "low", "ungrounded_anomaly", False),
        ("O?", "C?", "V+", True, "no_cross", "ABSTAIN", "low", "evidence_conflict", False),
    ],
)
async def test_analysis_decision_table(
    observer: str,
    cross: str,
    judged: str,
    overlap: bool,
    ablation: str,
    verdict: str,
    confidence: str,
    reason: str | None,
    agree: bool,
) -> None:
    state = {
        "observations": _observer(observer),
        "cross_check": _cross(cross, positive_region="tl" if overlap else "br"),
        "judgement": _judgement(judged),
        "ablation": ablation,
        "events": [],
    }

    result = await analysis_node(state, SimpleNamespace(ablation=""))

    assert result["verdict"] == verdict
    assert result["confidence"] == confidence
    assert result["abstain_reason"] == reason
    assert result["defect_type"] == (
        "scratch_neck" if verdict == "REJECT" else None
    )
    event = result["events"][-1]
    assert event["observer_status"] == observer
    assert event["cross_status"] == (
        "C-skip" if ablation == "no_cross" else cross
    )
    assert event["judgement_status"] == judged
    assert event["agree"] is agree
    assert event["abstain_reason"] == reason


@pytest.mark.asyncio
async def test_positive_region_union_excludes_negative_observations() -> None:
    state = {
        "observations": [
            *_observer("O+", positive_region="tl"),
            {
                "call_id": "obs-br",
                "region": "br",
                "text": "우하단 특이 소견이 없습니다.",
                "anomaly_seen": False,
                "regions": ["br"],
                "features": [],
            },
        ],
        "cross_check": _cross("C+", positive_region="br"),
        "judgement": _judgement("V+"),
        "events": [],
    }

    result = await analysis_node(state, SimpleNamespace(ablation=""))
    assert result["verdict"] == "REJECT"
    assert result["confidence"] == "low"
    assert result["events"][-1]["agree"] is False


def test_readability_and_observer_coverage_boundaries() -> None:
    assert is_readable_observation(_observer("O-")[0]) is True
    assert is_readable_observation(_observer("O?")[0]) is False
    assert is_readable_observation({"text": "", "anomaly_seen": False}) is False
    assert is_readable_observation({"text": "정상", "anomaly_seen": 0}) is False

    quadrants = [
        {
            "region": region,
            "text": f"{region} 특이 소견 없음",
            "anomaly_seen": False,
            "regions": [],
        }
        for region in ("tl", "tr", "bl")
    ]
    assert observer_status(quadrants[:2]) == "O?"
    assert observer_status(quadrants) == "O-"
    assert observer_status([
        *quadrants[:2],
        {"region": "center", "text": "중앙 정상", "anomaly_seen": False},
    ]) == "O?"


@pytest.mark.parametrize(
    ("observations", "expected"),
    [
        pytest.param(
            [
                {
                    "region": region,
                    "text": "판독 불가",
                    "anomaly_seen": None,
                    "regions": [],
                }
                for region in ("full", "tl", "tr", "bl", "br")
            ],
            "O?",
            id="all-five-unreadable",
        ),
        pytest.param(
            [
                {
                    "region": "full",
                    "text": "전체 특이 소견 없음",
                    "anomaly_seen": False,
                    "regions": [],
                },
                {
                    "region": "tl",
                    "text": "판독 불가",
                    "anomaly_seen": None,
                    "regions": [],
                },
            ],
            "O-",
            id="readable-full-negative-one-crop-unreadable",
        ),
        pytest.param(
            [
                {
                    "region": "full",
                    "text": "판독 불가",
                    "anomaly_seen": None,
                    "regions": [],
                },
                *[
                    {
                        "region": region,
                        "text": f"{region} 특이 소견 없음",
                        "anomaly_seen": False,
                        "regions": [],
                    }
                    for region in ("tl", "tr")
                ],
            ],
            "O?",
            id="unreadable-full-two-negative-quadrants",
        ),
        pytest.param(
            [
                {
                    "region": "full",
                    "text": "판독 불가",
                    "anomaly_seen": None,
                    "regions": [],
                },
                *[
                    {
                        "region": region,
                        "text": f"{region} 특이 소견 없음",
                        "anomaly_seen": False,
                        "regions": [],
                    }
                    for region in ("tl", "tr", "bl")
                ],
            ],
            "O-",
            id="unreadable-full-three-negative-quadrants",
        ),
        pytest.param(
            [{
                "region": "center",
                "text": "중앙 특이 소견 없음",
                "anomaly_seen": False,
                "regions": [],
            }],
            "O?",
            id="center-negative-does-not-prove-coverage",
        ),
        pytest.param(
            [{
                "region": "center",
                "text": "중앙에 선형 흔적이 보임",
                "anomaly_seen": True,
                "regions": ["center"],
            }],
            "O+",
            id="center-positive-is-positive-evidence",
        ),
    ],
)
def test_observer_status_named_contract_boundaries(
    observations: list[dict],
    expected: str,
) -> None:
    assert observer_status(observations) == expected


@pytest.mark.parametrize(
    "text",
    [
        "이미지를 판독할 수 없습니다.",
        "대상을 식별할 수 없습니다.",
        "unreadable image",
    ],
)
def test_unreadable_wording_variants_never_count_as_negative_evidence(
    text: str,
) -> None:
    text = unicodedata.normalize("NFC", text)
    assert observer_status([{
        "region": "full",
        "text": text,
        "anomaly_seen": False,
        "regions": [],
    }]) == "O?"
    assert cross_status({
        "summary": text,
        "anomaly_seen": False,
        "regions": [],
    }) == "C?"


@pytest.mark.parametrize(
    "text",
    [
        "결함 여부를 확인할 수 없습니다.",
        "이미지가 흐려 판별할 수 없습니다.",
        "화질이 나자 판단하기 어렵습니다.",
    ],
)
def test_uncertain_wording_never_counts_as_negative_evidence(text: str) -> None:
    assert observer_status([{
        "region": "full",
        "text": text,
        "anomaly_seen": False,
        "regions": [],
    }]) == "O?"
    assert cross_status({
        "summary": text,
        "anomaly_seen": False,
        "regions": [],
    }) == "C?"


@pytest.mark.parametrize(
    "text",
    [
        "Unable to determine whether a defect is present.",
        "Cannot determine whether the surface is normal.",
        "The image is too blurry to assess.",
    ],
)
def test_english_uncertainty_never_counts_as_negative_evidence(text: str) -> None:
    assert observer_status([{
        "region": "full",
        "text": text,
        "anomaly_seen": False,
        "regions": [],
    }]) == "O?"
    assert cross_status({
        "summary": text,
        "anomaly_seen": False,
        "regions": [],
    }) == "C?"


def test_malformed_cross_and_grounded_judgement_fail_closed() -> None:
    assert cross_status({
        "summary": "이상이 없습니다.",
        "anomaly_seen": False,
        "regions": "tl",
    }) == "C?"
    assert cross_status({
        "summary": "이상이 없습니다.",
        "anomaly_seen": False,
        "regions": ["outside"],
    }) == "C?"
    assert normalized_judgement_status({
        "status": "grounded",
        "violations": ["SCREW-04"],
        "defect_type": None,
    }) == "V?"
    assert normalized_judgement_status({
        "status": "none",
        "violations": ["SCREW-04"],
        "defect_type": "scratch_neck",
    }) == "V?"


@pytest.mark.asyncio
async def test_replan_only_for_o_unknown_and_only_once() -> None:
    base = {"tasks": [], "events": [], "replan_rounds": 0}

    full_negative = await replan_check_node({
        **base,
        "observations": [
            *_observer("O-"),
            {
                "region": "tl",
                "text": "판독 불가",
                "anomaly_seen": None,
                "regions": [],
            },
        ],
    }, None)
    assert full_negative["research_sufficient"] is True
    assert "tasks" not in full_negative

    unknown = await replan_check_node(
        {**base, "observations": _observer("O?")}, None
    )
    assert unknown["research_sufficient"] is False
    assert unknown["replan_rounds"] == 1
    assert unknown["tasks"][-1]["region"] == "center"
    assert unknown["events"][-1]["observer_status"] == "O?"

    after_center = await replan_check_node(
        {
            **base,
            "observations": _observer("O?"),
            "replan_rounds": 1,
        },
        None,
    )
    assert after_center["research_sufficient"] is True
    assert "tasks" not in after_center


@pytest.mark.asyncio
@pytest.mark.parametrize("ablation_source", ["state", "deps"])
async def test_no_crops_skips_center_replan_for_o_unknown(
    ablation_source: str,
) -> None:
    state = {
        "tasks": [],
        "events": [],
        "replan_rounds": 0,
        "observations": _observer("O?"),
        "ablation": "no_crops" if ablation_source == "state" else "",
    }
    deps = SimpleNamespace(
        ablation="no_crops" if ablation_source == "deps" else ""
    )

    result = await replan_check_node(state, deps)

    assert result["research_sufficient"] is True
    assert "tasks" not in result
    assert "replan_rounds" not in result
    assert result["events"][-1] == {
        "type": "ReplanCheck",
        "round": 0,
        "is_sufficient": True,
        "observer_status": "O?",
        "replan_skipped": True,
        "skip_reason": "no_crops_full_only",
        "ablation": "no_crops",
    }
