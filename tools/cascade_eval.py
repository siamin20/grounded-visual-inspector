#!/usr/bin/env python3
"""완료된 dev B0/A 결과로 탐색적 A+ cascade를 계산한다.

이 분석은 이미 생성된 결과만 사용하며 이미지를 다시 열지 않는다.
정책은 B0의 REJECT를 우선하고, 그 외에서 A의 PASS만 수용한다.
dev 결과를 본 뒤 정의한 post-hoc 분석이므로 primary 가설이나
held-out 성능 근거로 사용할 수 없다.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.evaluators import (  # noqa: E402
    EvaluationDataError,
    binary_metrics,
    mcnemar,
    paired_balanced_accuracy_test,
)
from eval.provenance import (  # noqa: E402
    ProvenanceError,
    sha256_file,
    verify_run_manifest,
)
from eval.report import (  # noqa: E402
    ResultDataError,
    load_expected_rows,
    load_result_rows,
)


POLICY_ID = "b0_reject_then_a_pass_v1"
ANALYSIS_TYPE = "post_hoc_dev_cascade"
CLAIM_STATUS = "exploratory_only"
WARNING = (
    "dev 결과 확인 후 정의한 탐색적 정책이며 primary 가설이나 "
    "held-out 성능 근거로 사용할 수 없습니다."
)
MARKDOWN_WARNING = (
    "**탐색적 dev 후속 분석 — 1차 가설 및 held-out 성능 주장에 사용 금지**"
)


class CascadeDataError(ValueError):
    """A+ cascade를 안전하게 계산할 수 없을 때 발생한다."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _usable_verdict(row: Mapping[str, Any]) -> str | None:
    error = row.get("error")
    if error is not None and error != "":
        return None
    verdict = row.get("verdict")
    return verdict if verdict in {"PASS", "REJECT", "ABSTAIN"} else None


def cascade_verdict(
    b0_row: Mapping[str, Any],
    a_row: Mapping[str, Any],
) -> tuple[str, str | None]:
    """정답과 무관한 고정 A+ 정책으로 verdict와 출처를 반환한다."""
    b0_verdict = _usable_verdict(b0_row)
    a_verdict = _usable_verdict(a_row)
    if b0_verdict is None:
        return "ABSTAIN", None
    if b0_verdict == "REJECT":
        return "REJECT", "B0"
    if a_verdict is None:
        return "ABSTAIN", None
    if a_verdict == "PASS":
        return "PASS", "A"
    return "ABSTAIN", None


def _cascade_error(
    b0_row: Mapping[str, Any],
    a_row: Mapping[str, Any],
) -> str | None:
    """정책 판단에 필요한 upstream 오류를 숨기지 않는다."""
    if _usable_verdict(b0_row) is None:
        return "upstream_b0_unusable"
    if _usable_verdict(b0_row) != "REJECT" and _usable_verdict(a_row) is None:
        return "upstream_a_unusable"
    return None


def _index_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    name: str,
    condition: str,
    split: str,
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise CascadeDataError(f"{name} {position}번 행이 객체가 아닙니다.")
        if row.get("condition") != condition or row.get("split") != split:
            raise CascadeDataError(
                f"{name} {position}번 행의 condition/split이 다릅니다."
            )
        sample_id = row.get("id")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id != sample_id.strip()
        ):
            raise CascadeDataError(f"{name} {position}번 행의 id가 올바르지 않습니다.")
        if sample_id in indexed:
            raise CascadeDataError(f"{name}에 중복 id가 있습니다: {sample_id}")
        indexed[sample_id] = row
    return indexed


def build_cascade_rows(
    b0_rows: Sequence[Mapping[str, Any]],
    a_rows: Sequence[Mapping[str, Any]],
    expected_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """완전히 짝지은 B0/A roster에 고정 정책을 적용한다."""
    expected_by_id: dict[str, Mapping[str, Any]] = {}
    for position, row in enumerate(expected_rows, start=1):
        sample_id = row.get("id") if isinstance(row, Mapping) else None
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id != sample_id.strip()
        ):
            raise CascadeDataError(
                f"사전 등록 {position}번 행의 id가 올바르지 않습니다."
            )
        if sample_id in expected_by_id:
            raise CascadeDataError(f"사전 등록에 중복 id가 있습니다: {sample_id}")
        expected_by_id[sample_id] = row

    b0_by_id = _index_rows(b0_rows, name="B0", condition="B0", split="dev")
    a_by_id = _index_rows(a_rows, name="A", condition="A", split="dev")
    expected_ids = set(expected_by_id)
    for name, indexed in (("B0", b0_by_id), ("A", a_by_id)):
        missing = sorted(expected_ids - set(indexed))
        unexpected = sorted(set(indexed) - expected_ids)
        if missing or unexpected:
            raise CascadeDataError(
                f"{name} roster가 사전 등록과 다릅니다. "
                f"missing={missing}, unexpected={unexpected}"
            )

    derived: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for expected in expected_rows:
        sample_id = str(expected["id"])
        b0_row = b0_by_id[sample_id]
        a_row = a_by_id[sample_id]
        for name, row in (("B0", b0_row), ("A", a_row)):
            for key in ("item", "label", "split", "is_defect"):
                if row.get(key) != expected.get(key):
                    raise CascadeDataError(
                        f"{name}/{sample_id}의 {key}가 사전 등록과 다릅니다."
                    )
        verdict, source = cascade_verdict(b0_row, a_row)
        cascade_error = _cascade_error(b0_row, a_row)
        derived.append({
            "id": sample_id,
            "item": expected["item"],
            "label": expected["label"],
            "is_defect": expected["is_defect"],
            "split": "dev",
            "condition": "A+",
            "verdict": verdict,
            "error": cascade_error,
        })
        # 정책이 정답을 입력으로 쓰지 않았음을 검증할 수 있게
        # 공개 행에는 label과 경로를 저장하지 않는다.
        audit_rows.append({
            "id": sample_id,
            "b0_verdict": _usable_verdict(b0_row),
            "a_verdict": _usable_verdict(a_row),
            "cascade_verdict": verdict,
            "selected_source": source,
            "cascade_error": cascade_error,
        })

    # 기존 평가기가 메타데이터·분모 무결성을 다시 검증하게 한다.
    binary_metrics(derived, expected_rows)
    return derived, audit_rows


def _require_manifest_contract(manifest: Mapping[str, Any]) -> None:
    if manifest.get("status") != "complete":
        raise CascadeDataError("완료된 suite만 A+ 분석에 사용할 수 있습니다.")
    if manifest.get("split") != "dev":
        raise CascadeDataError("A+ post-hoc 분석은 dev 전용이며 held-out을 거부합니다.")
    if manifest.get("dry_run") is not False:
        raise CascadeDataError("dry-run은 A+ 실측 분석에 사용할 수 없습니다.")
    planned = manifest.get("planned_conditions")
    records = manifest.get("conditions")
    if not isinstance(planned, list) or not isinstance(records, Mapping):
        raise CascadeDataError("manifest의 조건 정보가 올바르지 않습니다.")
    for condition in ("B0", "A"):
        record = records.get(condition)
        if condition not in planned or not isinstance(record, Mapping):
            raise CascadeDataError(f"manifest에 {condition} 조건이 없습니다.")
        if record.get("status") != "complete":
            raise CascadeDataError(f"{condition} 조건이 complete가 아닙니다.")


def _condition_result_path(
    run_dir: Path,
    manifest: Mapping[str, Any],
    condition: str,
) -> tuple[Path, str]:
    record = manifest["conditions"][condition]["result"]
    if not isinstance(record, Mapping):
        raise CascadeDataError(f"{condition} 결과 기록이 없습니다.")
    relative = record.get("path")
    digest = record.get("sha256")
    if not isinstance(relative, str) or not isinstance(digest, str):
        raise CascadeDataError(f"{condition} 결과 경로·해시가 올바르지 않습니다.")
    try:
        target = (run_dir / relative).resolve(strict=True)
        target.relative_to(run_dir.resolve(strict=True))
    except (FileNotFoundError, ValueError) as exc:
        raise CascadeDataError(f"{condition} 결과 경로가 run 밖으로 나갑니다.") from exc
    if sha256_file(target) != digest:
        raise CascadeDataError(f"{condition} 결과 해시가 manifest와 다릅니다.")
    return target, digest


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CascadeDataError("Git commit을 확인할 수 없습니다.") from exc


def analyze_run(run_dir: str | Path) -> dict[str, Any]:
    """완결·실모델 dev run을 검증하고 A+ 파생 분석을 만든다."""
    directory = Path(run_dir).expanduser().resolve(strict=True)
    try:
        manifest = verify_run_manifest(
            directory,
            reject_dry_run=True,
            reject_fake=True,
            require_complete=True,
            validate_rows=True,
            repo_root=ROOT,
            verify_inputs=True,
        )
    except ProvenanceError as exc:
        raise CascadeDataError(f"원본 run provenance 검증 실패: {exc}") from exc
    _require_manifest_contract(manifest)

    b0_path, b0_sha = _condition_result_path(directory, manifest, "B0")
    a_path, a_sha = _condition_result_path(directory, manifest, "A")
    try:
        b0_rows = load_result_rows(b0_path)
        a_rows = load_result_rows(a_path)
        dataset_record = manifest["artifacts"]["dataset"]
        dataset_path = ROOT / dataset_record["path"]
        expected_rows = load_expected_rows(dataset_path, split="dev")
    except (ResultDataError, EvaluationDataError, KeyError, TypeError) as exc:
        raise CascadeDataError(f"원본 결과/roster 로드 실패: {exc}") from exc

    derived_rows, audit_rows = build_cascade_rows(b0_rows, a_rows, expected_rows)
    b0_metric = binary_metrics(b0_rows, expected_rows)
    a_metric = binary_metrics(a_rows, expected_rows)
    cascade_metric = binary_metrics(derived_rows, expected_rows)
    comparison = {
        "accuracy_delta_vs_b0": (
            float(cascade_metric["accuracy"]) - float(b0_metric["accuracy"])
        ),
        "balanced_accuracy_delta_vs_b0": (
            float(cascade_metric["balanced_accuracy"])
            - float(b0_metric["balanced_accuracy"])
        ),
        "mcnemar_vs_b0": mcnemar(b0_rows, derived_rows, expected_rows),
        "paired_balanced_accuracy_vs_b0": paired_balanced_accuracy_test(
            b0_rows,
            derived_rows,
            expected_rows,
        ),
    }
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise CascadeDataError("manifest run_id가 올바르지 않습니다.")
    return {
        "schema_version": 1,
        "analysis_type": ANALYSIS_TYPE,
        "policy_id": POLICY_ID,
        "post_hoc_dev": True,
        "primary_hypothesis": False,
        "heldout_validated": False,
        "claim_status": CLAIM_STATUS,
        "warning": WARNING,
        "run_id": run_id,
        "split": "dev",
        "generated_at": _utc_now(),
        "policy": {
            "description": "B0 REJECT 우선, 그 외 A PASS 수용, 그 외 ABSTAIN",
            "uses_ground_truth_at_runtime": False,
            "order": ["B0_REJECT", "A_PASS", "ABSTAIN"],
        },
        "source_hashes": {
            "manifest": sha256_file(directory / "manifest.json"),
            "B0_result": b0_sha,
            "A_result": a_sha,
            "dataset": dataset_record["sha256"],
            "analysis_code": sha256_file(Path(__file__)),
        },
        "analysis_git_commit": _git_commit(),
        "metrics": {
            "B0": b0_metric,
            "A": a_metric,
            "A+": cascade_metric,
        },
        "comparison": comparison,
        "metric_semantics": {
            "tp_tn_fp_fn": (
                "전체 roster 기준이며 ABSTAIN·오류·누락을 실패에 포함"
            ),
            "selective_tp_tn_fp_fn": "자동 판정분만 포함하고 ABSTAIN은 제외",
        },
        "limitations": [
            "B0와 A를 모두 실행해야 하며 결합 end-to-end 지연·자원은 미측정",
            "dev 결과를 본 뒤 정의한 사후 정책",
            "held-out 미검증",
        ],
        "rows": audit_rows,
    }


def render_markdown(artifact: Mapping[str, Any]) -> str:
    """탐색적 제약이 눈에 띄는 공개용 Markdown을 만든다."""
    metrics = artifact["metrics"]
    comparison = artifact["comparison"]

    def percent(value: Any) -> str:
        return f"{float(value) * 100:.2f}%"

    lines = [
        "# A+ cascade dev 탐색 결과",
        "",
        f"> {MARKDOWN_WARNING}",
        ">",
        f"> {artifact['warning']}",
        "",
        f"- 원본 run: `{artifact['run_id']}`",
        f"- 정책: `{artifact['policy_id']}`",
        "- 규칙: B0 `REJECT` 우선 → 그 외 A `PASS` → 그 외 `ABSTAIN`",
        "- 정답 라벨은 지표 집계에만 사용했고 정책 입력으로 사용하지 않음",
        "",
        "| 조건 | Accuracy | Balanced Accuracy | 정상 정답 | 결함 정답 | ABSTAIN | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in ("B0", "A", "A+"):
        value = metrics[condition]
        lines.append(
            f"| {condition} | {percent(value['accuracy'])} | "
            f"{percent(value['balanced_accuracy'])} | {value['tn']}/{value['normal_n']} | "
            f"{value['tp']}/{value['defect_n']} | {value['abstain_n']} | "
            f"{percent(value['coverage'])} |"
        )
    overall = comparison["mcnemar_vs_b0"]["overall"]
    lines.extend([
        "",
        f"- B0 대비 Accuracy 차이: "
        f"{float(comparison['accuracy_delta_vs_b0']) * 100:+.2f}%p",
        f"- B0 대비 Balanced Accuracy 차이: "
        f"{float(comparison['balanced_accuracy_delta_vs_b0']) * 100:+.2f}%p",
        f"- B0 대비 McNemar exact p-value: `{float(overall['p_value']):.6f}`",
        "",
        "이 결과는 dev 결과를 본 후 정의한 후속 탐색이다. "
        "A의 primary 성능 결과를 대체하지 않으며 held-out으로 검증되지 않았다.",
        "",
    ])
    return "\n".join(lines)


def write_outputs(
    artifact: Mapping[str, Any],
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """JSON·Markdown을 덮어쓰지 않고 기록한다."""
    directory = Path(out_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    run_id = artifact.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise CascadeDataError("출력 artifact에 run_id가 없습니다.")
    json_path = directory / f"a_plus_dev_{run_id}.json"
    markdown_path = directory / f"a_plus_dev_{run_id}.md"
    conflicts = [path for path in (json_path, markdown_path) if path.exists()]
    if conflicts:
        raise CascadeDataError(
            "기존 파생 결과를 덮어쓸 수 없습니다: "
            + ", ".join(str(path) for path in conflicts)
        )
    try:
        serialized = json.dumps(
            dict(artifact),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
        with json_path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            with markdown_path.open("x", encoding="utf-8") as handle:
                handle.write(render_markdown(artifact))
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            json_path.unlink(missing_ok=True)
            raise
    except (OSError, TypeError, ValueError) as exc:
        raise CascadeDataError(f"A+ 파생 결과 저장 실패: {exc}") from exc
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="완료된 실모델 dev B0/A로 post-hoc A+ cascade를 계산합니다."
    )
    parser.add_argument("--run-dir", required=True, help="완료된 eval/results/<run_id>")
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "docs" / "results"),
        help="파생 JSON·Markdown 저장 디렉터리",
    )
    args = parser.parse_args()
    try:
        artifact = analyze_run(args.run_dir)
        json_path, markdown_path = write_outputs(artifact, args.out_dir)
    except CascadeDataError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(f"A+ post-hoc JSON: {json_path}")
    print(f"A+ post-hoc Markdown: {markdown_path}")
    print(WARNING)


if __name__ == "__main__":
    main()
