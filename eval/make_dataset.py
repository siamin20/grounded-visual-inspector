"""MVTec AD 샘플 다운로드와 평가 메타데이터 생성.

이 스크립트는 이미지를 열지 않고 파일 경로와 외부 폴더 라벨만 사용한다.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path


SEED = 20261020
DEV_RATIO = 0.30
REPO_ID = "foersben/mvtec-ad"
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "mvtec"
DATASET_PATH = ROOT / "eval" / "dataset.json"
TYPE_MAP_PATH = ROOT / "eval" / "type_map.json"
LOTS_PATH = ROOT / "data" / "lots.json"


def _load_type_map() -> dict[str, dict[str, list[str]]]:
    """품목·유형 목록의 SSOT인 type_map.json을 읽는다."""
    value = json.loads(TYPE_MAP_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise RuntimeError("eval/type_map.json이 비어 있거나 잘못됐습니다.")
    return value


TYPE_MAP = _load_type_map()
ITEM_TYPES = {
    item: tuple(defect_types)
    for item, defect_types in TYPE_MAP.items()
}


def _expected_download_paths() -> list[Path]:
    paths: list[Path] = []
    for item, labels in ITEM_TYPES.items():
        paths.extend(DATA_ROOT / item / "train" / "good" / f"{index:03d}.png" for index in range(5))
        for label in labels:
            paths.extend(
                DATA_ROOT / item / "test" / label / f"{index:03d}.png"
                for index in range(20 if label == "good" else 12)
            )
    return paths


def download_samples() -> None:
    """필요한 품목·유형·번호의 PNG만 다운로드한다."""
    expected = _expected_download_paths()
    if all(path.is_file() for path in expected):
        print("MVTec AD 필요 샘플이 이미 있어 다운로드를 건너뜁니다.")
        return

    from huggingface_hub import snapshot_download

    allow_patterns = [path.relative_to(DATA_ROOT).as_posix() for path in expected]
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    print("MVTec AD 필요 샘플을 다운로드합니다.")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        local_dir=DATA_ROOT,
    )


def _select_paths(item: str, label: str, limit: int) -> tuple[list[Path], set[Path]]:
    paths = sorted((DATA_ROOT / item / "test" / label).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"샘플이 없습니다: data/mvtec/{item}/test/{label}")

    rng = random.Random(f"{SEED}:{item}:{label}")
    selected = sorted(rng.sample(paths, min(limit, len(paths))))
    shuffled = selected[:]
    rng.shuffle(shuffled)

    dev_count = max(1, int(len(selected) * DEV_RATIO + 0.5))
    if len(selected) > 1:
        dev_count = min(dev_count, len(selected) - 1)
    return selected, set(shuffled[:dev_count])


def build_dataset() -> dict[str, object]:
    samples: list[dict[str, object]] = []
    for item, labels in ITEM_TYPES.items():
        for label in labels:
            selected, dev_paths = _select_paths(item, label, 20 if label == "good" else 8)
            for path in selected:
                samples.append(
                    {
                        "id": f"{item}-{label}-{path.stem}",
                        "path": path.relative_to(ROOT).as_posix(),
                        "item": item,
                        "label": label,
                        "is_defect": label != "good",
                        "split": "dev" if path in dev_paths else "heldout",
                    }
                )
    return {"seed": SEED, "items": list(ITEM_TYPES), "samples": samples}


def build_lots(dataset: dict[str, object]) -> dict[str, dict[str, str]]:
    base_time = datetime(2026, 10, 20, 9, 15)
    lots: dict[str, dict[str, str]] = {}
    samples = dataset["samples"]
    assert isinstance(samples, list)
    for index, sample in enumerate(samples):
        assert isinstance(sample, dict)
        sample_id = str(sample["id"])
        lots[sample_id] = {
            "lot": f"L2610-{index // 20 + 1:02d}",
            "line": f"A-{index % 3 + 1}",
            "shift": "D" if (index // 12) % 2 == 0 else "N",
            "ts": (base_time + timedelta(minutes=5 * index)).isoformat(),
        }
    return lots


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def print_counts(dataset: dict[str, object]) -> None:
    samples = dataset["samples"]
    assert isinstance(samples, list)
    print("\n품목별 평가 샘플:")
    for item in ITEM_TYPES:
        item_samples = [sample for sample in samples if isinstance(sample, dict) and sample["item"] == item]
        splits = Counter(str(sample["split"]) for sample in item_samples)
        print(f"  {item}: {len(item_samples)}장 (dev {splits['dev']}, heldout {splits['heldout']})")
    print(f"  합계: {len(samples)}장")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="생성 전에 Hugging Face에서 필요 샘플을 다운로드")
    args = parser.parse_args()

    if args.download:
        download_samples()

    dataset = build_dataset()
    _write_json(DATASET_PATH, dataset)
    _write_json(LOTS_PATH, build_lots(dataset))
    print_counts(dataset)


if __name__ == "__main__":
    main()
