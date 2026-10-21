"""평가 실행과 산출물을 결속하고 검증한다.

평가 수치가 어떤 코드·데이터·기준서·모델·설정에서 나왔는지
저장소 상대경로와 해시로 기록한다. 결과 파일에는 절대경로를 남기지
않고, 무결성을 보장할 수 없으면 집계를 중단한다.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path, PureWindowsPath
import platform
import re
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import uuid


MANIFEST_SCHEMA_VERSION = 2
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CONDITION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
_MANIFEST_STATUSES = {"pending", "running", "incomplete", "complete", "failed"}
_CONDITION_STATUSES = {"running", "incomplete", "complete", "failed"}
_TOKEN_ACCOUNTING = {
    "provider_split",
    "estimated_split",
    "mixed_split",
    "total_only",
}
_SETUP_USAGE_STATUSES = {
    "not_applicable",
    "fake_estimate_no_billing",
    "known_total_only",
    "known_split",
    "unknown_tokens_after_success",
    "unknown_after_error",
}
_ROW_USAGE_STATUSES = {
    "fake_estimate_no_billing",
    "fake_error_no_billing",
    "known_total_only",
    "known_split",
    "unknown_tokens_after_success",
    "unknown_after_error",
}
DEFAULT_PACKAGE_NAMES = (
    "fastapi",
    "uvicorn",
    "langgraph",
    "langchain-core",
    "langsmith",
    "fastmcp",
    "mcp",
    "httpx",
    "pydantic",
    "Pillow",
    "numpy",
)
REQUIRED_RESULT_ROW_KEYS = (
    "run_id", "provenance_sha256", "dry_run", "id", "item", "label",
    "is_defect", "split", "condition", "verdict", "defect_type",
    "confidence", "abstain_reason", "approval_required",
    "validation_attempts", "validation_rejections", "retry_count",
    "replan_rounds", "successful_provider_requests",
    "successful_vlm_requests", "unknown_token_usage_successes",
    "unobserved_model_successes", "observed_model_response_count",
    "validation_errors_first", "would_fail", "observations_text",
    "pre_validation_snapshot", "report", "latency_s", "tokens",
    "input_tokens", "output_tokens", "token_accounting", "estimated_usd",
    "billed_usd", "compute_proxy_usd", "cost_basis", "observed_model_ids",
    "langsmith_run_id", "usage_status", "error",
)


class ProvenanceError(ValueError):
    """평가 실행 영수증이 무결성을 보장할 수 없을 때 발생한다."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def validate_run_id(value: object) -> str:
    """경로 탈출을 만들 수 없는 ``run_id``만 허용한다."""
    if not isinstance(value, str) or value != value.strip():
        raise ProvenanceError("run_id는 앞뒤 공백이 없는 문자열이어야 합니다.")
    if value in {".", ".."} or not _RUN_ID_RE.fullmatch(value):
        raise ProvenanceError(
            "run_id는 영문·숫자·밑줄·하이픈·점만 사용할 수 있습니다."
        )
    return value


def new_run_id(*, now: datetime | None = None, random_suffix: str | None = None) -> str:
    """충돌 가능성이 낮은 UTC 시각·난수 접미사로 ``run_id``를 만든다."""
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    stamp = moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    suffix = random_suffix or uuid.uuid4().hex[:12]
    if not re.fullmatch(r"[A-Za-z0-9]{4,32}", suffix):
        raise ProvenanceError("run_id 접미사는 4~32자 영문·숫자여야 합니다.")
    return validate_run_id(f"{stamp}-{suffix}")


def sha256_file(path: str | Path) -> str:
    """파일을 스트림으로 읽어 SHA-256을 계산한다."""
    target = Path(path)
    if not target.is_file():
        raise ProvenanceError(f"해시할 파일이 아닙니다: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(directory: str | Path, *, pattern: str = "**/*") -> str:
    """디렉터리의 상대경로와 내용을 안정적인 트리 SHA-256으로 만든다."""
    root = Path(directory)
    if not root.is_dir():
        raise ProvenanceError(f"해시할 디렉터리가 아닙니다: {root}")
    files = sorted(
        (path for path in root.glob(pattern) if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """JSON 객체를 캐노니컬하여 SHA-256을 계산한다."""
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProvenanceError(f"캐노니컬 JSON으로 변환할 수 없습니다: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ProvenanceError(f"비표준 JSON 수치를 허용하지 않습니다: {value}")


def read_json_object(path: str | Path) -> dict[str, Any]:
    """NaN·Infinity를 거부하며 JSON 개체만 읽는다."""
    target = Path(path)
    try:
        payload = json.loads(
            target.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except OSError as exc:
        raise ProvenanceError(f"JSON 파일을 읽을 수 없습니다: {target}") from exc
    except json.JSONDecodeError as exc:
        raise ProvenanceError(f"올바른 JSON이 아닙니다: {target}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ProvenanceError(f"JSON 최상위 값은 개체여야 합니다: {target}")
    return payload


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """같은 디렉터리에서 임시 파일을 쓴 뒤 JSON을 원자적으로 교체한다."""
    target = Path(path)
    if not target.parent.is_dir():
        raise ProvenanceError(f"JSON 상위 디렉터리가 아닙니다: {target.parent}")
    try:
        serialized = json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise ProvenanceError(f"마니페스트를 JSON으로 기록할 수 없습니다: {exc}") from exc

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, target)
    except OSError as exc:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise ProvenanceError(f"마니페스트를 원자적으로 기록하지 못했습니다: {target}") from exc
    return target


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return canonical_sha256(payload)


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> Path:
    payload = dict(manifest)
    payload["manifest_sha256"] = _manifest_digest(payload)
    if isinstance(manifest, dict):
        manifest["manifest_sha256"] = payload["manifest_sha256"]
    return atomic_write_json(path, payload)


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(repo_root.resolve(strict=True)).as_posix()
    except (FileNotFoundError, ValueError) as exc:
        raise ProvenanceError(f"저장소 안의 파일만 기록할 수 있습니다: {path}") from exc


def _file_record(path: Path, repo_root: Path) -> dict[str, str]:
    return {"path": _repo_relative(path, repo_root), "sha256": sha256_file(path)}


def _records_digest(records: Sequence[Mapping[str, str]]) -> str:
    return canonical_sha256([{"path": row["path"], "sha256": row["sha256"]} for row in records])


def _named_file_records(
    values: Sequence[Mapping[str, Any]],
    *,
    name_key: str,
    repo_root: Path,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen_names: set[str] = set()
    seen_paths: set[str] = set()
    for value in values:
        name = value.get(name_key)
        path = value.get("path")
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise ProvenanceError(f"{name_key}가 빈 값 없는 문자열이어야 합니다.")
        if name in seen_names:
            raise ProvenanceError(f"중복된 {name_key}입니다: {name}")
        candidate = Path(path) if isinstance(path, (str, os.PathLike)) else None
        if candidate is None:
            raise ProvenanceError(f"{name_key}={name}의 파일 경로가 필요합니다.")
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        record = _file_record(candidate, repo_root)
        if record["path"] in seen_paths:
            raise ProvenanceError(f"중복된 입력 파일 경로입니다: {record['path']}")
        records.append({name_key: name, **record})
        seen_names.add(name)
        seen_paths.add(record["path"])
    return records


def _run_git(repo_root: Path, *args: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProvenanceError(f"Git 상태를 확인할 수 없습니다: {' '.join(args)}") from exc
    return completed.stdout


def read_git_state(repo_root: str | Path) -> dict[str, Any]:
    """Git commit·dirty 여부·패치 내용 hash를 절대경로 없이 반환한다."""
    root = Path(repo_root).resolve(strict=True)
    commit = _run_git(root, "rev-parse", "HEAD").decode("ascii", errors="strict").strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", commit):
        raise ProvenanceError("Git commit SHA 형식이 올바르지 않습니다.")
    status = _run_git(root, "status", "--porcelain=v1", "--untracked-files=all")
    tracked_diff = _run_git(root, "diff", "--binary", "HEAD", "--")
    untracked = _run_git(root, "ls-files", "--others", "--exclude-standard", "-z")

    digest = hashlib.sha256()
    digest.update(tracked_diff)
    for raw_name in sorted(value for value in untracked.split(b"\0") if value):
        digest.update(len(raw_name).to_bytes(8, "big"))
        digest.update(raw_name)
        candidate = root / raw_name.decode("utf-8", errors="surrogateescape")
        if candidate.is_file():
            digest.update(bytes.fromhex(sha256_file(candidate)))
    return {
        "commit": commit.lower(),
        "dirty": bool(status),
        "diff_sha256": digest.hexdigest(),
    }


def _default_source_paths(repo_root: Path) -> list[Path]:
    sources: list[Path] = []
    for directory in (repo_root / "agent", repo_root / "eval"):
        if directory.is_dir():
            sources.extend(directory.rglob("*.py"))
    for name in (
        "server.py",
        "mcp_server.py",
        "mcp_server_docs.py",
        "mcp_server_vision.py",
    ):
        candidate = repo_root / name
        if candidate.is_file():
            sources.append(candidate)
    return sorted(set(sources), key=lambda path: path.relative_to(repo_root).as_posix())


def collect_repository_provenance(
    repo_root: str | Path,
    *,
    dataset_path: str | Path | None = None,
    type_map_path: str | Path | None = None,
    specs_dir: str | Path | None = None,
    guides_dir: str | Path | None = None,
    lots_path: str | Path | None = None,
    evaluation_images: Sequence[Mapping[str, Any]] = (),
    reference_images: Sequence[Mapping[str, Any]] = (),
    source_paths: Iterable[str | Path] | None = None,
    package_names: Sequence[str] = DEFAULT_PACKAGE_NAMES,
    git_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """평가 재현에 필요한 저장소 상대경로 provenance를 만든다."""
    root = Path(repo_root).resolve(strict=True)
    dataset = Path(dataset_path) if dataset_path is not None else root / "eval" / "dataset.json"
    type_map = Path(type_map_path) if type_map_path is not None else root / "eval" / "type_map.json"
    specs = Path(specs_dir) if specs_dir is not None else root / "data" / "specs"
    guides = Path(guides_dir) if guides_dir is not None else root / "data" / "guides"
    lots = Path(lots_path) if lots_path is not None else root / "data" / "lots.json"
    if not dataset.is_absolute():
        dataset = root / dataset
    if not type_map.is_absolute():
        type_map = root / type_map
    if not specs.is_absolute():
        specs = root / specs
    if not guides.is_absolute():
        guides = root / guides
    if not lots.is_absolute():
        lots = root / lots

    spec_paths = sorted(specs.glob("*.md")) if specs.is_dir() else []
    guide_paths = sorted(guides.glob("*.md")) if guides.is_dir() else []
    if not spec_paths or not guide_paths:
        raise ProvenanceError("기준서와 조치 가이드 파일이 모두 필요합니다.")

    if source_paths is None:
        resolved_sources = _default_source_paths(root)
    else:
        resolved_sources = []
        for value in source_paths:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = root / candidate
            resolved_sources.append(candidate)
        resolved_sources = sorted(
            set(resolved_sources), key=lambda path: _repo_relative(path, root)
        )
    if not resolved_sources:
        raise ProvenanceError("평가 코드 파일이 모두 필요합니다.")

    spec_records = [_file_record(path, root) for path in spec_paths]
    guide_records = [_file_record(path, root) for path in guide_paths]
    source_records = [_file_record(path, root) for path in resolved_sources]
    evaluation_records = _named_file_records(
        evaluation_images, name_key="id", repo_root=root
    )
    reference_records = _named_file_records(
        reference_images, name_key="item", repo_root=root
    )
    packages: dict[str, str | None] = {}
    for package_name in package_names:
        try:
            packages[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            packages[package_name] = None

    repository = dict(git_state) if git_state is not None else read_git_state(root)
    required_git = {"commit", "dirty", "diff_sha256"}
    if set(repository) != required_git:
        raise ProvenanceError("Git provenance는 commit·dirty·diff_sha256 전부를 포함해야 합니다.")
    if not isinstance(repository["dirty"], bool):
        raise ProvenanceError("Git dirty 값은 bool이어야 합니다.")
    for key in ("commit", "diff_sha256"):
        value = repository[key]
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}|[0-9a-fA-F]{40}", value):
            raise ProvenanceError(f"Git {key} 형식이 올바르지 않습니다.")

    return {
        "repository": repository,
        "artifacts": {
            "dataset": _file_record(dataset, root),
            "type_map": _file_record(type_map, root),
            "lots": _file_record(lots, root),
            "specs": spec_records,
            "specs_sha256": _records_digest(spec_records),
            "guides": guide_records,
            "guides_sha256": _records_digest(guide_records),
            "source": source_records,
            "source_tree_sha256": _records_digest(source_records),
            "evaluation_images": evaluation_records,
            "evaluation_images_sha256": canonical_sha256(evaluation_records),
            "reference_images": reference_records,
            "reference_images_sha256": canonical_sha256(reference_records),
        },
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "packages": packages,
        },
    }


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProvenanceError(f"{name}은 0 이상 정수여야 합니다.")
    return value


def _nonnegative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProvenanceError(f"{name}은 유한한 0 이상 수치여야 합니다.")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ProvenanceError(f"{name}은 유한한 0 이상 수치여야 합니다.")
    return number


def _validate_condition(value: object) -> str:
    if not isinstance(value, str) or not _CONDITION_RE.fullmatch(value):
        raise ProvenanceError("조건은 경로에 안전한 식별자여야 합니다.")
    return value


def _normalize_setup_usage(value: object) -> dict[str, Any]:
    """조건 초기화에서 소비한 provider usage를 엄격하게 검증한다.

    기준서 색인은 샘플 행과 독립적인 조건당 1회 비용이다. 그래서
    샘플이 0행인 중단 경로에서도 manifest에 남을 수 있도록 별도로
    검증한다.
    """
    if not isinstance(value, Mapping):
        raise ProvenanceError("setup_usage는 개체여야 합니다.")
    required = {
        "tokens",
        "input_tokens",
        "output_tokens",
        "token_accounting",
        "estimated_usd",
        "compute_proxy_usd",
        "successful_provider_requests",
        "successful_vlm_requests",
        "unknown_token_usage_successes",
        "unobserved_model_successes",
        "observed_model_response_count",
        "observed_model_ids",
        "latency_s",
        "usage_status",
    }
    if set(value) != required:
        raise ProvenanceError("setup_usage 필드가 올바르지 않습니다.")
    tokens = _nonnegative_int(value.get("tokens"), "setup_usage.tokens")
    accounting = value.get("token_accounting")
    if accounting not in _TOKEN_ACCOUNTING:
        raise ProvenanceError("setup_usage.token_accounting 값이 올바르지 않습니다.")
    if accounting == "total_only":
        if value.get("input_tokens") is not None or value.get("output_tokens") is not None:
            raise ProvenanceError("setup_usage total_only에는 분리 토큰을 둘 수 없습니다.")
        input_tokens = output_tokens = None
    else:
        input_tokens = _nonnegative_int(
            value.get("input_tokens"), "setup_usage.input_tokens"
        )
        output_tokens = _nonnegative_int(
            value.get("output_tokens"), "setup_usage.output_tokens"
        )
        if input_tokens + output_tokens != tokens:
            raise ProvenanceError("setup_usage 입력·출력 토큰 합이 전체와 다릅니다.")
    models = value.get("observed_model_ids")
    if not isinstance(models, list) or any(
        not isinstance(model, str) or not model.strip() or model != model.strip()
        for model in models
    ):
        raise ProvenanceError("setup_usage.observed_model_ids가 올바르지 않습니다.")
    status = value.get("usage_status")
    if status not in _SETUP_USAGE_STATUSES:
        raise ProvenanceError("setup_usage.usage_status가 올바르지 않습니다.")
    successful_requests = _nonnegative_int(
        value.get("successful_provider_requests"),
        "setup_usage.successful_provider_requests",
    )
    unknown_tokens = _nonnegative_int(
        value.get("unknown_token_usage_successes"),
        "setup_usage.unknown_token_usage_successes",
    )
    unobserved_models = _nonnegative_int(
        value.get("unobserved_model_successes"),
        "setup_usage.unobserved_model_successes",
    )
    observed_count = _nonnegative_int(
        value.get("observed_model_response_count"),
        "setup_usage.observed_model_response_count",
    )
    if unknown_tokens > successful_requests:
        raise ProvenanceError(
            "setup_usage 토큰 미지 성공 수가 provider 성공 수보다 큽니다."
        )
    if observed_count + unobserved_models != successful_requests:
        raise ProvenanceError(
            "setup_usage 모델 ID 관측 카운트가 provider 성공 수와 다릅니다."
        )
    if observed_count and not models:
        raise ProvenanceError(
            "setup_usage에 관측 모델 응답이 있지만 model ID가 없습니다."
        )
    if successful_requests and accounting == "total_only" and tokens == 0 and not unknown_tokens:
        raise ProvenanceError(
            "setup_usage의 token usage 누락을 알려야 합니다."
        )
    has_unknown = bool(unknown_tokens or unobserved_models)
    if (status == "unknown_tokens_after_success") is not has_unknown:
        raise ProvenanceError(
            "setup_usage.usage_status와 usage 누락 카운터가 다릅니다."
        )
    if status == "not_applicable" and successful_requests:
        raise ProvenanceError(
            "provider 성공 요청이 있는 setup_usage는 not_applicable일 수 없습니다."
        )
    if status == "unknown_after_error" and successful_requests:
        raise ProvenanceError(
            "setup unknown_after_error는 성공 provider 응답 수를 확정할 수 없을 때만 사용합니다."
        )
    return {
        "tokens": tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "token_accounting": accounting,
        "estimated_usd": _nonnegative_float(
            value.get("estimated_usd"), "setup_usage.estimated_usd"
        ),
        "compute_proxy_usd": _nonnegative_float(
            value.get("compute_proxy_usd"), "setup_usage.compute_proxy_usd"
        ),
        "successful_provider_requests": successful_requests,
        "successful_vlm_requests": _nonnegative_int(
            value.get("successful_vlm_requests"),
            "setup_usage.successful_vlm_requests",
        ),
        "unknown_token_usage_successes": unknown_tokens,
        "unobserved_model_successes": unobserved_models,
        "observed_model_response_count": observed_count,
        "observed_model_ids": list(dict.fromkeys(models)),
        "latency_s": _nonnegative_float(
            value.get("latency_s"), "setup_usage.latency_s"
        ),
        "usage_status": status,
    }


def _immutable_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "run_id",
        "split",
        "dry_run",
        "planned_conditions",
        "seed",
        "expected_n",
        "suite_budget_estimated_usd",
        "backends",
        "models",
        "settings",
        "pricing",
        "repository",
        "artifacts",
        "runtime",
    )
    missing = [key for key in keys if key not in manifest]
    if missing:
        raise ProvenanceError(f"마니페스트 불변 필드 누락: {', '.join(missing)}")
    return {key: manifest[key] for key in keys}


def build_run_manifest(
    *,
    run_id: str,
    split: str,
    planned_conditions: Sequence[str],
    seed: int,
    expected_n: int,
    dry_run: bool,
    suite_budget_estimated_usd: int | float,
    provenance: Mapping[str, Any],
    backends: Mapping[str, str],
    models: Mapping[str, Any],
    settings: Mapping[str, Any] | None = None,
    pricing: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """새 run manifest의 불변 영역과 hash를 만든다."""
    normalized_run_id = validate_run_id(run_id)
    if split not in {"dev", "heldout"}:
        raise ProvenanceError("split은 dev 또는 heldout이어야 합니다.")
    if not isinstance(dry_run, bool):
        raise ProvenanceError("dry_run은 bool이어야 합니다.")
    conditions = [_validate_condition(value) for value in planned_conditions]
    if not conditions or len(set(conditions)) != len(conditions):
        raise ProvenanceError("예정 조건은 비어 있지 않고 중복이 없어야 합니다.")
    normalized_backends = dict(backends)
    if not normalized_backends or any(
        not isinstance(key, str) or not isinstance(value, str) or not value
        for key, value in normalized_backends.items()
    ):
        raise ProvenanceError("백엔드 정보는 빈 문자열이 없는 맵이어야 합니다.")
    required_provenance = {"repository", "artifacts", "runtime"}
    if set(provenance) != required_provenance:
        raise ProvenanceError("provenance는 repository·artifacts·runtime만 포함해야 합니다.")

    timestamp = created_at or _utc_now()
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": normalized_run_id,
        "status": "pending",
        "created_at": timestamp,
        "updated_at": timestamp,
        "completed_at": None,
        "failed_at": None,
        "split": split,
        "dry_run": dry_run,
        "planned_conditions": conditions,
        "seed": _nonnegative_int(seed, "seed"),
        "expected_n": _nonnegative_int(expected_n, "expected_n"),
        "suite_budget_estimated_usd": _nonnegative_float(
            suite_budget_estimated_usd, "suite_budget_estimated_usd"
        ),
        "suite_estimated_spend_usd": 0.0,
        "backends": normalized_backends,
        "models": dict(models),
        "settings": dict(settings or {}),
        "pricing": dict(pricing or {}),
        "repository": dict(provenance["repository"]),
        "artifacts": dict(provenance["artifacts"]),
        "runtime": dict(provenance["runtime"]),
        "conditions": {},
    }
    manifest["immutable_sha256"] = canonical_sha256(_immutable_payload(manifest))
    manifest["manifest_sha256"] = _manifest_digest(manifest)
    _validate_manifest(manifest)
    return manifest


def _is_absolute_path_string(value: str) -> bool:
    return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _validate_file_record(record: object, *, name: str) -> None:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise ProvenanceError(f"{name} 파일 기록이 올바르지 않습니다.")
    path = record.get("path")
    digest = record.get("sha256")
    if not isinstance(path, str) or not path or _is_absolute_path_string(path):
        raise ProvenanceError(f"{name} 경로는 저장소 상대경로여야 합니다.")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ProvenanceError(f"{name} SHA-256 형식이 올바르지 않습니다.")


def _validate_named_file_records(
    records: object,
    *,
    name: str,
    name_key: str,
    digest: object,
) -> None:
    if not isinstance(records, list):
        raise ProvenanceError(f"{name}는 파일 목록이어야 합니다.")
    names: set[str] = set()
    paths: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != {
            name_key, "path", "sha256"
        }:
            raise ProvenanceError(f"{name}[{index}] 기록이 올바르지 않습니다.")
        value = record.get(name_key)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ProvenanceError(f"{name}[{index}].{name_key}가 올바르지 않습니다.")
        _validate_file_record(
            {"path": record.get("path"), "sha256": record.get("sha256")},
            name=f"{name}[{index}]",
        )
        if value in names or record["path"] in paths:
            raise ProvenanceError(f"{name}에 중복 기록이 있습니다.")
        names.add(value)
        paths.add(record["path"])
    if digest != canonical_sha256(records):
        raise ProvenanceError(f"{name} digest가 파일 목록과 다릅니다.")


def _validate_manifest(manifest: Mapping[str, Any], *, run_dir: Path | None = None) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ProvenanceError("지원하지 않는 manifest schema_version입니다.")
    run_id = validate_run_id(manifest.get("run_id"))
    if run_dir is not None and run_dir.name != run_id:
        raise ProvenanceError("run_id와 실행 디렉터리 이름이 다릅니다.")
    if manifest.get("status") not in _MANIFEST_STATUSES:
        raise ProvenanceError("마니페스트 상태가 올바르지 않습니다.")
    if not isinstance(manifest.get("dry_run"), bool):
        raise ProvenanceError("dry_run은 bool이어야 합니다.")
    _nonnegative_int(manifest.get("seed"), "seed")
    _nonnegative_int(manifest.get("expected_n"), "expected_n")
    _nonnegative_float(
        manifest.get("suite_budget_estimated_usd"),
        "suite_budget_estimated_usd",
    )
    suite_spend = _nonnegative_float(
        manifest.get("suite_estimated_spend_usd"),
        "suite_estimated_spend_usd",
    )
    conditions = manifest.get("planned_conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ProvenanceError("planned_conditions는 빈 배열일 수 없습니다.")
    normalized = [_validate_condition(value) for value in conditions]
    if len(normalized) != len(set(normalized)):
        raise ProvenanceError("planned_conditions에 중복이 있습니다.")

    immutable_hash = manifest.get("immutable_sha256")
    if not isinstance(immutable_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", immutable_hash):
        raise ProvenanceError("immutable_sha256 형식이 올바르지 않습니다.")
    actual_immutable = canonical_sha256(_immutable_payload(manifest))
    if immutable_hash != actual_immutable:
        raise ProvenanceError("마니페스트 불변 영역이 변조되었습니다.")
    manifest_hash = manifest.get("manifest_sha256")
    if not isinstance(manifest_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", manifest_hash):
        raise ProvenanceError("manifest_sha256 형식이 올바르지 않습니다.")
    if manifest_hash != _manifest_digest(manifest):
        raise ProvenanceError("마니페스트 자체가 변조되었습니다.")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ProvenanceError("artifacts는 개체여야 합니다.")
    _validate_file_record(artifacts.get("dataset"), name="dataset")
    _validate_file_record(artifacts.get("type_map"), name="type_map")
    _validate_file_record(artifacts.get("lots"), name="lots")
    for group_name in ("specs", "guides", "source"):
        group = artifacts.get(group_name)
        if not isinstance(group, list) or not group:
            raise ProvenanceError(f"{group_name}는 비어 있지 않은 파일 목록이어야 합니다.")
        for index, record in enumerate(group):
            _validate_file_record(record, name=f"{group_name}[{index}]")
        digest_name = {
            "specs": "specs_sha256",
            "guides": "guides_sha256",
            "source": "source_tree_sha256",
        }[group_name]
        recorded_digest = artifacts.get(digest_name)
        if recorded_digest != _records_digest(group):
            raise ProvenanceError(f"{digest_name}가 파일 목록과 다릅니다.")
    _validate_named_file_records(
        artifacts.get("evaluation_images"),
        name="evaluation_images",
        name_key="id",
        digest=artifacts.get("evaluation_images_sha256"),
    )
    _validate_named_file_records(
        artifacts.get("reference_images"),
        name="reference_images",
        name_key="item",
        digest=artifacts.get("reference_images_sha256"),
    )
    if not manifest.get("dry_run"):
        if len(artifacts["evaluation_images"]) != manifest.get("expected_n"):
            raise ProvenanceError("실측 manifest의 평가 이미지 수가 expected_n과 다릅니다.")
        if not artifacts["reference_images"]:
            raise ProvenanceError("실측 manifest에 참조 이미지가 없습니다.")

    condition_records = manifest.get("conditions")
    if not isinstance(condition_records, Mapping):
        raise ProvenanceError("conditions는 개체여야 합니다.")
    started_conditions = set(condition_records)
    expected_prefix = set(normalized[: len(started_conditions)])
    if started_conditions != expected_prefix:
        raise ProvenanceError("실행 조건은 planned_conditions 순서의 prefix여야 합니다.")
    if any(
        condition_records[name].get("status") != "complete"
        for name in normalized[: max(0, len(started_conditions) - 1)]
    ):
        raise ProvenanceError("후속 조건 전에 선행 조건이 complete여야 합니다.")
    for condition, record in condition_records.items():
        if condition not in normalized or not isinstance(record, Mapping):
            raise ProvenanceError("계획되지 않은 조건 기록이 있습니다.")
        if record.get("status") not in _CONDITION_STATUSES:
            raise ProvenanceError(f"{condition} 상태가 올바르지 않습니다.")
        _nonnegative_int(record.get("processed_samples"), f"{condition}.processed_samples")
        _nonnegative_float(
            record.get("estimated_spend_usd"),
            f"{condition}.estimated_spend_usd",
        )
        if "setup_usage" in record:
            _normalize_setup_usage(record["setup_usage"])
        if record.get("status") in {"complete", "incomplete"}:
            if "setup_usage" not in record:
                raise ProvenanceError(f"{condition}.setup_usage가 누락됐습니다.")
            expected_complete = record.get("status") == "complete"
            if record.get("complete_capture") is not expected_complete:
                raise ProvenanceError(
                    f"{condition}.complete_capture가 상태와 다릅니다."
                )
            _nonnegative_float(
                record.get("suite_estimated_spend_usd_at_completion"),
                f"{condition}.suite_estimated_spend_usd_at_completion",
            )
            observed = record.get("observed_model_ids")
            if not isinstance(observed, list) or not observed or any(
                not isinstance(value, str)
                or not value.strip()
                or value != value.strip()
                for value in observed
            ):
                raise ProvenanceError(
                    f"{condition}.observed_model_ids가 올바르지 않습니다."
                )
            if "result" not in record or "summary" not in record:
                raise ProvenanceError(f"{condition} 종료 산출물이 누락됐습니다.")
        for key in ("result", "summary"):
            if key in record:
                _validate_file_record(record[key], name=f"{condition}.{key}")
    condition_spend = math.fsum(
        float(record["estimated_spend_usd"])
        for record in condition_records.values()
    )
    if not math.isclose(suite_spend, condition_spend, rel_tol=0.0, abs_tol=1e-9):
        raise ProvenanceError("suite 누적 예상 비용이 조건별 합계와 다릅니다.")
    if manifest.get("status") == "complete":
        if set(condition_records) != set(normalized) or any(
            record.get("status") != "complete"
            for record in condition_records.values()
        ):
            raise ProvenanceError("완료된 suite만 complete로 표시할 수 있습니다.")
    if manifest.get("status") == "pending" and condition_records:
        raise ProvenanceError("pending suite에는 조건 기록이 있을 수 없습니다.")


def create_run_manifest(run_dir: str | Path, manifest: Mapping[str, Any]) -> Path:
    """고유 run 디렉터리와 manifest를 함께 새로 만든다."""
    target_dir = Path(run_dir)
    _validate_manifest(manifest, run_dir=target_dir)
    try:
        target_dir.mkdir(parents=False, exist_ok=False)
    except OSError as exc:
        raise ProvenanceError(f"새 run 디렉터리를 만들 수 없습니다: {target_dir}") from exc
    try:
        return _write_manifest(target_dir / "manifest.json", manifest)
    except Exception:
        try:
            target_dir.rmdir()
        except OSError:
            pass
        raise


def load_run_manifest(run_dir: str | Path) -> dict[str, Any]:
    """run 디렉터리의 manifest를 읽고 스키마·불변 영역을 검사한다."""
    directory = Path(run_dir)
    payload = read_json_object(directory / "manifest.json")
    _validate_manifest(payload, run_dir=directory)
    return payload


def join_run_manifest(
    run_dir: str | Path,
    expected_manifest: Mapping[str, Any],
    *,
    condition: str,
) -> dict[str, Any]:
    """임의 suite에 후속 조건을 추가하기 전 불변 값을 비교한다."""
    requested = _validate_condition(condition)
    directory = Path(run_dir)
    current = load_run_manifest(directory)
    _validate_manifest(expected_manifest, run_dir=directory)
    if current["immutable_sha256"] != expected_manifest["immutable_sha256"]:
        raise ProvenanceError("기존 suite의 불변 설정과 새 실행 설정이 다릅니다.")
    if requested not in current["planned_conditions"]:
        raise ProvenanceError(f"사전 계획에 없는 조건입니다: {requested}")
    if requested in current["conditions"] or (directory / f".condition-{requested}.reserved").exists():
        raise ProvenanceError(f"이미 시작하거나 예약된 조건입니다: {requested}")
    if any(
        record.get("status") == "running"
        for record in current["conditions"].values()
    ):
        raise ProvenanceError("다른 조건이 실행 중일 때는 새 조건을 시작할 수 없습니다.")
    next_index = len(current["conditions"])
    expected_next = current["planned_conditions"][next_index]
    if requested != expected_next:
        raise ProvenanceError(
            f"조건 실행 순서가 잘못되었습니다: "
            f"다음={expected_next}, 요청={requested}"
        )
    if any(
        record.get("status") != "complete"
        for record in current["conditions"].values()
    ):
        raise ProvenanceError("선행 조건이 complete가 아니어서 후속 조건을 시작할 수 없습니다.")
    if current["status"] in {"complete", "failed"}:
        raise ProvenanceError(f"{current['status']} suite에는 새 조건을 추가할 수 없습니다.")
    return current


def start_condition(
    run_dir: str | Path,
    expected_manifest: Mapping[str, Any],
    *,
    condition: str,
    started_at: str | None = None,
) -> dict[str, Any]:
    """조건을 원자적으로 예약하고 running 상태로 기록한다."""
    directory = Path(run_dir)
    requested = _validate_condition(condition)
    current = join_run_manifest(directory, expected_manifest, condition=requested)
    reservation = directory / f".condition-{requested}.reserved"
    try:
        with reservation.open("x", encoding="utf-8") as output:
            output.write(current["run_id"] + "\n")
    except OSError as exc:
        raise ProvenanceError(f"조건 예약이 이미 존재합니다: {requested}") from exc

    timestamp = started_at or _utc_now()
    current["conditions"][requested] = {
        "status": "running",
        "started_at": timestamp,
        "ended_at": None,
        "processed_samples": 0,
        "estimated_spend_usd": 0.0,
    }
    current["status"] = "running"
    current["updated_at"] = timestamp
    _write_manifest(directory / "manifest.json", current)
    return current


def record_condition_setup_usage(
    run_dir: str | Path,
    *,
    condition: str,
    setup_usage: Mapping[str, Any],
    updated_at: str | None = None,
) -> dict[str, Any]:
    """조건당 1회 색인·초기화 usage를 샘플 실행 전에 결속한다."""
    directory = Path(run_dir)
    requested = _validate_condition(condition)
    normalized = _normalize_setup_usage(setup_usage)
    current = load_run_manifest(directory)
    record = current["conditions"].get(requested)
    if not isinstance(record, dict) or record.get("status") != "running":
        raise ProvenanceError(
            f"running 조건만 초기화 usage를 기록할 수 있습니다: {requested}"
        )
    if "setup_usage" in record:
        raise ProvenanceError(
            f"조건 초기화 usage가 이미 기록됐습니다: {requested}"
        )
    if (
        record.get("processed_samples") != 0
        or float(record.get("estimated_spend_usd", 0.0)) != 0.0
    ):
        raise ProvenanceError(
            "샘플 진행 전에만 초기화 usage를 기록할 수 있습니다."
        )
    timestamp = updated_at or _utc_now()
    record["setup_usage"] = normalized
    record["estimated_spend_usd"] = normalized["estimated_usd"]
    current["suite_estimated_spend_usd"] = math.fsum(
        float(value["estimated_spend_usd"])
        for value in current["conditions"].values()
    )
    current["updated_at"] = timestamp
    _write_manifest(directory / "manifest.json", current)
    return current


def record_condition_progress(
    run_dir: str | Path,
    *,
    condition: str,
    processed_samples: int,
    estimated_spend_usd: int | float,
    updated_at: str | None = None,
) -> dict[str, Any]:
    """조건별 누적 처리 건수와 예상 비용을 원자적으로 갱신한다.

    두 값 모두 조건 내 누적치이며, 이전 기록보다 감소하는 값을 거부한다.
    """
    directory = Path(run_dir)
    requested = _validate_condition(condition)
    next_processed = _nonnegative_int(processed_samples, "processed_samples")
    next_spend = _nonnegative_float(estimated_spend_usd, "estimated_spend_usd")
    current = load_run_manifest(directory)
    record = current["conditions"].get(requested)
    if not isinstance(record, dict) or record.get("status") != "running":
        raise ProvenanceError(f"running 조건만 누적할 수 있습니다: {requested}")
    if next_processed < record["processed_samples"]:
        raise ProvenanceError("processed_samples는 기존값보다 줄어들 수 없습니다.")
    if next_spend + 1e-12 < float(record["estimated_spend_usd"]):
        raise ProvenanceError("estimated_spend_usd는 기존값보다 줄어들 수 없습니다.")
    timestamp = updated_at or _utc_now()
    record["processed_samples"] = next_processed
    record["estimated_spend_usd"] = next_spend
    current["suite_estimated_spend_usd"] = math.fsum(
        float(value["estimated_spend_usd"])
        for value in current["conditions"].values()
    )
    current["updated_at"] = timestamp
    _write_manifest(directory / "manifest.json", current)
    return current


def _safe_run_file(run_dir: Path, path: str | Path) -> tuple[Path, str]:
    value = Path(path)
    if value.is_absolute() or PureWindowsPath(str(path)).is_absolute():
        candidate = value.resolve(strict=True)
    else:
        candidate = (run_dir / value).resolve(strict=True)
    root = run_dir.resolve(strict=True)
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError as exc:
        raise ProvenanceError(f"run 디렉터리 밖의 파일은 허용하지 않습니다: {path}") from exc
    if not candidate.is_file():
        raise ProvenanceError(f"실행 파일이 아닙니다: {path}")
    return candidate, relative


def complete_condition(
    run_dir: str | Path,
    *,
    condition: str,
    result_path: str | Path,
    summary_path: str | Path,
    complete_capture: bool = True,
    observed_model_ids: Sequence[str] = (),
    ended_at: str | None = None,
) -> dict[str, Any]:
    """조건 산출물을 hash로 결속하고 조건·suite 상태를 갱신한다."""
    directory = Path(run_dir)
    requested = _validate_condition(condition)
    current = load_run_manifest(directory)
    record = current["conditions"].get(requested)
    if not isinstance(record, dict) or record.get("status") != "running":
        raise ProvenanceError(f"running 조건만 종료할 수 있습니다: {requested}")
    if not isinstance(complete_capture, bool):
        raise ProvenanceError("complete_capture은 bool이어야 합니다.")
    result, result_relative = _safe_run_file(directory, result_path)
    summary, summary_relative = _safe_run_file(directory, summary_path)
    if not observed_model_ids or any(
        not isinstance(value, str) or not value for value in observed_model_ids
    ):
        raise ProvenanceError("observed_model_ids는 빈 문자열이 없는 목록이어야 합니다.")
    timestamp = ended_at or _utc_now()
    record.update({
        "status": "complete" if complete_capture else "incomplete",
        "ended_at": timestamp,
        "complete_capture": complete_capture,
        "suite_estimated_spend_usd_at_completion": current[
            "suite_estimated_spend_usd"
        ],
        "observed_model_ids": list(dict.fromkeys(observed_model_ids)),
        "result": {"path": result_relative, "sha256": sha256_file(result)},
        "summary": {"path": summary_relative, "sha256": sha256_file(summary)},
    })
    statuses = [
        current["conditions"].get(name, {}).get("status")
        for name in current["planned_conditions"]
    ]
    if all(status == "complete" for status in statuses):
        current["status"] = "complete"
        current["completed_at"] = timestamp
    else:
        current["status"] = "incomplete"
        current["completed_at"] = None
    current["updated_at"] = timestamp
    _write_manifest(directory / "manifest.json", current)
    return current


def fail_condition(
    run_dir: str | Path,
    *,
    condition: str,
    error: str,
    failed_at: str | None = None,
) -> dict[str, Any]:
    """조건을 실패했을 때 재시작 불가 상태로 남긴다."""
    directory = Path(run_dir)
    requested = _validate_condition(condition)
    if not isinstance(error, str) or not error.strip():
        raise ProvenanceError("실패 사유는 비어 있지 않은 문자열이어야 합니다.")
    current = load_run_manifest(directory)
    record = current["conditions"].get(requested)
    if not isinstance(record, dict) or record.get("status") != "running":
        raise ProvenanceError(f"running 조건만 실패로 종료할 수 있습니다: {requested}")
    timestamp = failed_at or _utc_now()
    record.update({"status": "failed", "ended_at": timestamp, "error": error.strip()})
    current["status"] = "failed"
    current["failed_at"] = timestamp
    current["updated_at"] = timestamp
    _write_manifest(directory / "manifest.json", current)
    return current


def _verify_row_usage(
    row: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    location: str,
) -> bool:
    """행 usage 계약을 검증하고 실측 불완전 여부를 반환한다."""
    tokens = _nonnegative_int(row.get("tokens"), f"{location}.tokens")
    accounting = row.get("token_accounting")
    if accounting not in _TOKEN_ACCOUNTING:
        raise ProvenanceError(f"{location}.token_accounting 값이 올바르지 않습니다.")
    if accounting == "total_only":
        if row.get("input_tokens") is not None or row.get("output_tokens") is not None:
            raise ProvenanceError(f"{location} total_only에는 분리 토큰을 둘 수 없습니다.")
    else:
        input_tokens = _nonnegative_int(
            row.get("input_tokens"), f"{location}.input_tokens"
        )
        output_tokens = _nonnegative_int(
            row.get("output_tokens"), f"{location}.output_tokens"
        )
        if input_tokens + output_tokens != tokens:
            raise ProvenanceError(f"{location} 입력·출력 토큰 합이 전체와 다릅니다.")

    _nonnegative_float(row.get("estimated_usd"), f"{location}.estimated_usd")
    _nonnegative_float(
        row.get("compute_proxy_usd"), f"{location}.compute_proxy_usd"
    )
    successful = _nonnegative_int(
        row.get("successful_provider_requests"),
        f"{location}.successful_provider_requests",
    )
    successful_vlm = _nonnegative_int(
        row.get("successful_vlm_requests"),
        f"{location}.successful_vlm_requests",
    )
    if successful_vlm > successful:
        raise ProvenanceError(f"{location} VLM 성공 수가 provider 성공 수보다 큽니다.")
    unknown_tokens = _nonnegative_int(
        row.get("unknown_token_usage_successes"),
        f"{location}.unknown_token_usage_successes",
    )
    unobserved = _nonnegative_int(
        row.get("unobserved_model_successes"),
        f"{location}.unobserved_model_successes",
    )
    observed_count = _nonnegative_int(
        row.get("observed_model_response_count"),
        f"{location}.observed_model_response_count",
    )
    if unknown_tokens > successful:
        raise ProvenanceError(f"{location} token usage 미지 수가 성공 수보다 큽니다.")
    if observed_count + unobserved != successful:
        raise ProvenanceError(f"{location} model ID 관측 카운터가 성공 수와 다릅니다.")
    models = row.get("observed_model_ids")
    if not isinstance(models, list) or any(
        not isinstance(model, str) or not model.strip() or model != model.strip()
        for model in models
    ):
        raise ProvenanceError(f"{location}.observed_model_ids가 올바르지 않습니다.")
    if observed_count and not models:
        raise ProvenanceError(f"{location}의 관측 model ID가 누락됐습니다.")
    if successful and accounting == "total_only" and tokens == 0 and not unknown_tokens:
        raise ProvenanceError(f"{location}의 token usage 누락을 알려야 합니다.")

    error = row.get("error")
    if error is not None and (
        not isinstance(error, str) or not error.strip() or error != error.strip()
    ):
        raise ProvenanceError(f"{location}.error가 올바르지 않습니다.")
    backend_values = set(manifest.get("backends", {}).values())
    fake = backend_values == {"fake"}
    has_unknown = bool(unknown_tokens or unobserved)
    if error is not None:
        expected_status = "fake_error_no_billing" if fake else "unknown_after_error"
    elif fake:
        expected_status = "fake_estimate_no_billing"
    elif has_unknown:
        expected_status = "unknown_tokens_after_success"
    elif accounting == "total_only":
        expected_status = "known_total_only"
    else:
        expected_status = "known_split"
    status = row.get("usage_status")
    if status not in _ROW_USAGE_STATUSES or status != expected_status:
        raise ProvenanceError(
            f"{location}.usage_status와 usage·error 상태가 다릅니다."
        )
    return status in {"unknown_after_error", "unknown_tokens_after_success"}


def _verify_jsonl_rows(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    condition: str,
    require_exact_roster: bool,
    expected_truth: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ProvenanceError(f"결과 JSONL을 읽을 수 없습니다: {path.name}") from exc
    row_count = 0
    estimated_spend: list[float] = []
    observed_model_ids: set[str] = set()
    usage_unknown_count = 0
    expected_ids = {
        record["id"] for record in manifest["artifacts"]["evaluation_images"]
    }
    seen_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ProvenanceError) as exc:
            raise ProvenanceError(f"결과 JSONL이 손상되었습니다: {path.name}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ProvenanceError(f"결과 JSONL 행이 개체가 아닙니다: {path.name}:{line_number}")
        if tuple(row) != REQUIRED_RESULT_ROW_KEYS:
            raise ProvenanceError(
                f"결과 JSONL 행 스키마가 올바르지 않습니다: {path.name}:{line_number}"
            )
        expected = {
            "run_id": manifest["run_id"],
            "condition": condition,
            "split": manifest["split"],
            "dry_run": manifest["dry_run"],
            "provenance_sha256": manifest["immutable_sha256"],
        }
        for key, value in expected.items():
            if row.get(key) != value:
                raise ProvenanceError(f"결과 행의 {key}가 manifest와 다릅니다: {path.name}:{line_number}")
        location = f"{path.name}:{line_number}"
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id.strip() or row_id != row_id.strip():
            raise ProvenanceError(f"{location}.id가 올바르지 않습니다.")
        if row_id in seen_ids:
            raise ProvenanceError(f"결과 JSONL에 중복 id가 있습니다: {row_id}")
        if row_id not in expected_ids:
            raise ProvenanceError(f"결과 id가 manifest 평가 roster에 없습니다: {row_id}")
        seen_ids.add(row_id)
        if expected_truth is not None:
            truth = expected_truth.get(row_id)
            if truth is None:
                raise ProvenanceError(f"결과 id의 dataset 정답을 확인할 수 없습니다: {row_id}")
            if (
                row.get("item") != truth["item"]
                or row.get("label") != truth["label"]
                or row.get("is_defect") is not truth["is_defect"]
            ):
                raise ProvenanceError(
                    f"결과 행의 item·label·is_defect가 dataset과 다릅니다: {row_id}"
                )
        row_count += 1
        estimated_spend.append(
            _nonnegative_float(
                row.get("estimated_usd"),
                f"{location}.estimated_usd",
            )
        )
        models = row.get("observed_model_ids")
        if not isinstance(models, list) or any(
            not isinstance(value, str) or not value.strip() or value != value.strip()
            for value in models
        ):
            raise ProvenanceError(
                f"결과 행의 observed_model_ids가 올바르지 않습니다: "
                f"{path.name}:{line_number}"
            )
        observed_model_ids.update(models)
        if _verify_row_usage(row, manifest=manifest, location=location):
            usage_unknown_count += 1
    if require_exact_roster and seen_ids != expected_ids:
        missing = sorted(expected_ids - seen_ids)
        raise ProvenanceError(
            "complete 결과의 처리 건수·id roster가 manifest와 다릅니다"
            + (f": 누락 {', '.join(missing)}" if missing else "")
        )
    return {
        "processed_samples": row_count,
        "condition_estimated_usd": math.fsum(estimated_spend),
        "observed_model_ids": observed_model_ids,
        "usage_unknown_count": usage_unknown_count,
    }


def _verify_summary(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    condition: str,
) -> dict[str, Any]:
    summary = read_json_object(path)
    expected = {
        "run_id": manifest["run_id"],
        "condition": condition,
        "split": manifest["split"],
        "dry_run": manifest["dry_run"],
        "provenance_sha256": manifest["immutable_sha256"],
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ProvenanceError(f"요약의 {key}가 manifest와 다릅니다: {path.name}")
    _nonnegative_int(summary.get("processed_samples"), "summary.processed_samples")
    _nonnegative_float(
        summary.get("condition_estimated_usd"),
        "summary.condition_estimated_usd",
    )
    _nonnegative_float(
        summary.get("suite_estimated_usd"),
        "summary.suite_estimated_usd",
    )
    if not isinstance(summary.get("complete_capture"), bool):
        raise ProvenanceError("summary.complete_capture은 bool이어야 합니다.")
    _nonnegative_int(
        summary.get("usage_unknown_count"), "summary.usage_unknown_count"
    )
    _normalize_setup_usage(summary.get("setup_usage"))
    return summary


def _verify_condition_accounting(
    *,
    manifest: Mapping[str, Any],
    condition: str,
    record: Mapping[str, Any],
    rows: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    """결과 행·요약·manifest의 계수·비용·모델 출처를 서로 대조한다."""
    processed = _nonnegative_int(
        record.get("processed_samples"),
        f"{condition}.processed_samples",
    )
    summary_processed = _nonnegative_int(
        summary.get("processed_samples"),
        "summary.processed_samples",
    )
    if rows["processed_samples"] != processed or summary_processed != processed:
        raise ProvenanceError(
            f"{condition} 결과 행 수·요약·manifest 처리 건수가 다릅니다."
        )
    summary_unknown = _nonnegative_int(
        summary.get("usage_unknown_count"), "summary.usage_unknown_count"
    )
    if rows["usage_unknown_count"] != summary_unknown:
        raise ProvenanceError(
            f"{condition} 결과 행과 summary의 usage 미지 건수가 다릅니다."
        )
    expected_n = _nonnegative_int(manifest.get("expected_n"), "expected_n")
    if processed > expected_n:
        raise ProvenanceError(f"{condition} 처리 건수가 예상 N을 넘었습니다.")

    record_spend = _nonnegative_float(
        record.get("estimated_spend_usd"),
        f"{condition}.estimated_spend_usd",
    )
    summary_spend = _nonnegative_float(
        summary.get("condition_estimated_usd"),
        "summary.condition_estimated_usd",
    )
    setup = _normalize_setup_usage(record.get("setup_usage"))
    summary_setup = _normalize_setup_usage(summary.get("setup_usage"))
    if summary_setup != setup:
        raise ProvenanceError(
            f"{condition} summary와 manifest의 초기화 usage가 다릅니다."
        )
    rows_plus_setup = rows["condition_estimated_usd"] + setup["estimated_usd"]
    if not math.isclose(
        rows_plus_setup, record_spend, rel_tol=0.0, abs_tol=1e-9
    ) or not math.isclose(summary_spend, record_spend, rel_tol=0.0, abs_tol=1e-9):
        raise ProvenanceError(
            f"{condition} 결과 행·요약·manifest 예상 비용이 다릅니다."
        )

    recorded_models = record.get("observed_model_ids")
    if not isinstance(recorded_models, list) or not recorded_models or any(
        not isinstance(value, str) or not value.strip() or value != value.strip()
        for value in recorded_models
    ):
        raise ProvenanceError(f"{condition}.observed_model_ids가 올바르지 않습니다.")
    expected_models = rows["observed_model_ids"] | set(setup["observed_model_ids"])
    if expected_models != set(recorded_models):
        raise ProvenanceError(
            f"{condition} 결과 행과 manifest의 관측 model ID가 다릅니다."
        )
    backend_values = set(manifest.get("backends", {}).values())
    real_backends = backend_values & {"openai", "ollama"}
    if "fake" not in backend_values and real_backends:
        allowed_prefixes = tuple(f"{backend}:" for backend in sorted(real_backends))
        if any(not value.startswith(allowed_prefixes) for value in recorded_models):
            raise ProvenanceError(
                f"{condition} 관측 model ID가 선택한 backend 응답 형식과 다릅니다."
            )

    complete_capture = summary.get("complete_capture")
    expected_complete = record.get("status") == "complete"
    if complete_capture is not expected_complete or record.get("complete_capture") is not expected_complete:
        raise ProvenanceError(
            f"{condition} summary complete_capture와 manifest 상태가 다릅니다."
        )
    if expected_complete and processed != expected_n:
        raise ProvenanceError(
            f"{condition} complete 조건의 처리 건수가 예상 N과 다릅니다."
        )
    if expected_complete and summary_unknown:
        raise ProvenanceError(
            f"{condition} usage 미지 행이 있는데 complete로 표시됐습니다."
        )
    if expected_complete and (
        setup["unknown_token_usage_successes"]
        or setup["unobserved_model_successes"]
        or setup["usage_status"] in {
            "unknown_tokens_after_success", "unknown_after_error"
        }
    ):
        raise ProvenanceError(
            f"{condition} setup usage가 불완전한데 complete로 표시됐습니다."
        )
    suite_at_completion = _nonnegative_float(
        record.get("suite_estimated_spend_usd_at_completion"),
        f"{condition}.suite_estimated_spend_usd_at_completion",
    )
    summary_suite = _nonnegative_float(
        summary.get("suite_estimated_usd"),
        "summary.suite_estimated_usd",
    )
    if not math.isclose(
        summary_suite, suite_at_completion, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ProvenanceError(
            f"{condition} summary와 manifest의 suite 완료 시점 예상 비용이 다릅니다."
        )
    manifest_suite = _nonnegative_float(
        manifest.get("suite_estimated_spend_usd"),
        "suite_estimated_spend_usd",
    )
    if suite_at_completion > manifest_suite + 1e-9:
        raise ProvenanceError(
            f"{condition} 완료 시점 suite 비용이 manifest 최종 비용보다 큽니다."
        )


def _verify_repository_artifacts(
    manifest: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, dict[str, Any]]:
    root = repo_root.resolve(strict=True)
    artifacts = manifest["artifacts"]
    records: list[Mapping[str, str]] = [
        artifacts["dataset"],
        artifacts["type_map"],
        artifacts["lots"],
    ]
    records.extend(artifacts["specs"])
    records.extend(artifacts["guides"])
    records.extend(artifacts["source"])
    records.extend(artifacts["evaluation_images"])
    records.extend(artifacts["reference_images"])
    for record in records:
        relative = record["path"]
        try:
            target = (root / relative).resolve(strict=True)
        except FileNotFoundError as exc:
            raise ProvenanceError(f"평가 입력 파일이 사라졌습니다: {relative}") from exc
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ProvenanceError(f"저장소 경로가 밖으로 나갔습니다: {relative}") from exc
        if not target.is_file() or sha256_file(target) != record["sha256"]:
            raise ProvenanceError(f"평가 입력 파일이 변경되었습니다: {relative}")

    expected_specs = {record["path"] for record in artifacts["specs"]}
    expected_guides = {record["path"] for record in artifacts["guides"]}
    expected_sources = {record["path"] for record in artifacts["source"]}
    actual_specs = {
        path.relative_to(root).as_posix()
        for path in (root / "data" / "specs").glob("*.md")
        if path.is_file()
    }
    actual_guides = {
        path.relative_to(root).as_posix()
        for path in (root / "data" / "guides").glob("*.md")
        if path.is_file()
    }
    actual_sources = {
        path.relative_to(root).as_posix()
        for path in _default_source_paths(root)
    }
    for name, expected, actual in (
        ("기준서", expected_specs, actual_specs),
        ("조치 가이드", expected_guides, actual_guides),
        ("평가 코드", expected_sources, actual_sources),
    ):
        if expected != actual:
            raise ProvenanceError(f"{name} 파일 목록이 변경되었습니다.")

    dataset = read_json_object(root / artifacts["dataset"]["path"])
    samples = dataset.get("samples")
    if not isinstance(samples, list):
        raise ProvenanceError("평가 dataset의 samples가 배열이 아닙니다.")
    samples_by_id: dict[str, Mapping[str, Any]] = {}
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ProvenanceError(f"평가 dataset.samples[{index}]가 개체가 아닙니다.")
        sample_id = sample.get("id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ProvenanceError(f"평가 dataset.samples[{index}].id가 올바르지 않습니다.")
        if sample_id in samples_by_id:
            raise ProvenanceError(f"평가 dataset에 중복 id가 있습니다: {sample_id}")
        samples_by_id[sample_id] = sample

    selected_items: set[str] = set()
    selected_truth: dict[str, dict[str, Any]] = {}
    for record in artifacts["evaluation_images"]:
        sample = samples_by_id.get(record["id"])
        if sample is None:
            raise ProvenanceError(
                f"평가 이미지 id가 dataset에 없습니다: {record['id']}"
            )
        if sample.get("split") != manifest.get("split"):
            raise ProvenanceError(
                f"평가 이미지 split이 manifest와 다릅니다: {record['id']}"
            )
        raw_path = sample.get("path")
        item = sample.get("item")
        label = sample.get("label")
        is_defect = sample.get("is_defect")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ProvenanceError(f"dataset 이미지 경로가 올바르지 않습니다: {record['id']}")
        if not isinstance(item, str) or not item.strip():
            raise ProvenanceError(f"dataset 품목이 올바르지 않습니다: {record['id']}")
        if not isinstance(label, str) or not label.strip():
            raise ProvenanceError(f"dataset label이 올바르지 않습니다: {record['id']}")
        if not isinstance(is_defect, bool):
            raise ProvenanceError(f"dataset is_defect가 bool이 아닙니다: {record['id']}")
        source_path = Path(raw_path)
        if source_path.is_absolute() or ".." in source_path.parts:
            raise ProvenanceError(f"dataset 이미지 경로가 저장소 안 상대경로가 아닙니다: {record['id']}")
        try:
            expected_path = (root / source_path).resolve(strict=True).relative_to(root).as_posix()
        except (FileNotFoundError, ValueError) as exc:
            raise ProvenanceError(f"dataset 이미지 경로를 확인할 수 없습니다: {record['id']}") from exc
        if record["path"] != expected_path:
            raise ProvenanceError(
                f"평가 이미지 경로가 dataset과 다릅니다: {record['id']}"
            )
        selected_items.add(item)
        selected_truth[record["id"]] = {
            "item": item,
            "label": label,
            "is_defect": is_defect,
        }

    reference_items = {record["item"] for record in artifacts["reference_images"]}
    if artifacts["evaluation_images"] and reference_items != selected_items:
        raise ProvenanceError("참조 이미지 품목 목록이 평가 품목과 다릅니다.")
    for record in artifacts["reference_images"]:
        expected_reference = (
            Path("data") / "mvtec" / record["item"] / "train" / "good" / "000.png"
        ).as_posix()
        if record["path"] != expected_reference:
            raise ProvenanceError(
                f"참조 이미지 경로 규칙이 다릅니다: {record['item']}"
            )
    return selected_truth


def verify_run_manifest(
    run_dir: str | Path,
    *,
    reject_dry_run: bool = True,
    reject_fake: bool = True,
    expected_backends: Mapping[str, str] | None = None,
    require_complete: bool = True,
    validate_rows: bool = True,
    repo_root: str | Path | None = None,
    verify_inputs: bool = False,
) -> dict[str, Any]:
    """manifest와 등록된 결과·요약 hash·행 결속을 fail-closed로 검사한다."""
    directory = Path(run_dir)
    manifest = load_run_manifest(directory)
    if reject_dry_run and manifest["dry_run"]:
        raise ProvenanceError("dry-run 결과는 실측 보고서에 사용할 수 없습니다.")
    backends = manifest.get("backends")
    if not isinstance(backends, Mapping):
        raise ProvenanceError("마니페스트의 backends가 올바르지 않습니다.")
    if reject_fake and any(value == "fake" for value in backends.values()):
        raise ProvenanceError("fake 백엔드 결과는 실측 보고서에 사용할 수 없습니다.")
    if expected_backends is not None and dict(backends) != dict(expected_backends):
        raise ProvenanceError("마니페스트의 백엔드가 예상과 다릅니다.")
    if require_complete and manifest["status"] != "complete":
        raise ProvenanceError(f"완료되지 않은 suite 상태입니다: {manifest['status']}")
    if verify_inputs and repo_root is None:
        raise ProvenanceError("입력 파일 검사에는 repo_root가 필요합니다.")
    expected_truth: dict[str, dict[str, Any]] | None = None
    if verify_inputs:
        expected_truth = _verify_repository_artifacts(
            manifest, repo_root=Path(repo_root)
        )

    registered_jsonl: set[str] = set()
    for condition, record in manifest["conditions"].items():
        if record["status"] not in {"complete", "incomplete"}:
            if require_complete:
                raise ProvenanceError(f"완료되지 않은 조건 상태입니다: {condition}")
            continue
        result_facts: dict[str, Any] | None = None
        summary_payload: dict[str, Any] | None = None
        for key in ("result", "summary"):
            file_record = record.get(key)
            _validate_file_record(file_record, name=f"{condition}.{key}")
            target, relative = _safe_run_file(directory, file_record["path"])
            if sha256_file(target) != file_record["sha256"]:
                raise ProvenanceError(f"{condition} {key} 파일이 변조되었습니다.")
            if key == "result":
                registered_jsonl.add(relative)
                if validate_rows:
                    result_facts = _verify_jsonl_rows(
                        target,
                        manifest=manifest,
                        condition=condition,
                        require_exact_roster=record["status"] == "complete",
                        expected_truth=expected_truth,
                    )
            elif validate_rows:
                summary_payload = _verify_summary(
                    target, manifest=manifest, condition=condition
                )
        if validate_rows:
            if result_facts is None or summary_payload is None:
                raise ProvenanceError(f"{condition} 결과와 요약을 검증할 수 없습니다.")
            _verify_condition_accounting(
                manifest=manifest,
                condition=condition,
                record=record,
                rows=result_facts,
                summary=summary_payload,
            )

    actual_jsonl = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*.jsonl")
        if path.is_file()
    }
    unexpected = sorted(actual_jsonl - registered_jsonl)
    if unexpected:
        raise ProvenanceError(f"manifest에 등록되지 않은 결과 JSONL이 있습니다: {', '.join(unexpected)}")
    return manifest


__all__ = [
    "DEFAULT_PACKAGE_NAMES",
    "MANIFEST_SCHEMA_VERSION",
    "ProvenanceError",
    "REQUIRED_RESULT_ROW_KEYS",
    "atomic_write_json",
    "build_run_manifest",
    "canonical_sha256",
    "collect_repository_provenance",
    "complete_condition",
    "create_run_manifest",
    "fail_condition",
    "join_run_manifest",
    "load_run_manifest",
    "new_run_id",
    "read_git_state",
    "read_json_object",
    "record_condition_progress",
    "record_condition_setup_usage",
    "sha256_file",
    "sha256_tree",
    "start_condition",
    "validate_run_id",
    "verify_run_manifest",
]
