#!/usr/bin/env bash
# heldout 분할에서 B0·B1·A를 실모델로 평가한다.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -x ".venv/bin/python3" ] && export PATH="$PWD/.venv/bin:$PATH"
if [ -f .env ]; then
    set -a; source .env; set +a
fi

LLM_BACKEND="${LLM_BACKEND:-fake}"
VLM_BACKEND="${VLM_BACKEND:-$LLM_BACKEND}"
EMBED_BACKEND="${EMBED_BACKEND:-$LLM_BACKEND}"
EVAL_SUITE_BUDGET_USD="${EVAL_SUITE_BUDGET_USD:-${EVAL_BUDGET_USD:-15}}"
# 이전 환경 변수도 같은 suite 상한으로 맞춰 하위 호환만 유지한다.
EVAL_BUDGET_USD="$EVAL_SUITE_BUDGET_USD"
EVAL_PRICE_TABLE_VERSION="provider-effective-v2"
EVAL_FREEZE_MANIFEST="${EVAL_FREEZE_MANIFEST:-eval/freeze_manifest.json}"
export LLM_BACKEND VLM_BACKEND EMBED_BACKEND
export EVAL_SUITE_BUDGET_USD EVAL_BUDGET_USD EVAL_PRICE_TABLE_VERSION
export EVAL_FREEZE_MANIFEST
CONDITIONS=(B0 B1 A)
SUITE_CONDITIONS="B0,B1,A"

if [ "$LLM_BACKEND" = "fake" ] || [ "$VLM_BACKEND" = "fake" ] || [ "$EMBED_BACKEND" = "fake" ]; then
    echo "ERROR: heldout 평가는 fake LLM/VLM/embedding을 허용하지 않습니다."
    echo "  LLM_BACKEND=openai VLM_BACKEND=openai EMBED_BACKEND=openai ./sh/run_eval_heldout.sh"
    exit 1
fi
if [ ! -f "$EVAL_FREEZE_MANIFEST" ]; then
    echo "ERROR: heldout 동결 manifest가 없습니다: $EVAL_FREEZE_MANIFEST"
    echo "  R7 동결 게이트를 완료한 뒤 실행하세요."
    exit 1
fi

RUN_ID="$(python3 -c 'from eval.provenance import new_run_id; print(new_run_id())')"
RESULTS_DIR="eval/results/$RUN_ID"

python3 - "$EVAL_SUITE_BUDGET_USD" "$EVAL_PRICE_TABLE_VERSION" <<'PY'
import json
import math
import sys
from pathlib import Path

samples = json.loads(Path("eval/dataset.json").read_text(encoding="utf-8"))["samples"]
count = sum(sample["split"] == "heldout" for sample in samples)
try:
    budget = float(sys.argv[1])
except (TypeError, ValueError) as exc:
    raise SystemExit("ERROR: suite 예산은 유한한 0 이상의 수여야 합니다.") from exc
if not math.isfinite(budget) or budget < 0:
    raise SystemExit("ERROR: suite 예산은 유한한 0 이상의 수여야 합니다.")
print(f"heldout 이미지: {count}장")
print(f"조건: B0, B1, A (3개, 최대 {count * 3}건 처리)")
print(f"설계상 nominal provider 요청: 약 {count * 12 + 1}회")
print("  B0 1회/장 + B1 1회/장 + A 10회/장 + A 조건 setup(기준서 색인) 1회")
print("  재시도·재계획·보고서 재작성은 제외한 사전 추정치입니다.")
print(f"가격표 버전: {sys.argv[2]}")
print(f"추정 비용 가드: 전체 suite 공유 상한 ${budget:.2f}")
print("주의: setup과 각 샘플 종료 후 usage로 검사하므로 setup 1회 또는")
print("      한 샘플의 다중 요청 추정 비용만큼 상한을 넘을 수 있습니다.")
print("주의: estimated_usd이며 실제 청구액을 보증하지 않습니다.")
PY
echo "백엔드: llm=$LLM_BACKEND / vlm=$VLM_BACKEND / embed=$EMBED_BACKEND"
echo "동결 manifest: $EVAL_FREEZE_MANIFEST"
echo "run_id: $RUN_ID"
echo "결과 디렉터리: $RESULTS_DIR"
read -r -p "최종 heldout 평가면 HELDOUT 140 FINAL을 입력하세요: " CONFIRM
if [ "$CONFIRM" != "HELDOUT 140 FINAL" ]; then
    echo "중단합니다."
    exit 1
fi

for CONDITION in "${CONDITIONS[@]}"; do
    python3 eval/run_eval.py \
        --condition "$CONDITION" \
        --split heldout \
        --run-id "$RUN_ID" \
        --suite-conditions "$SUITE_CONDITIONS" \
        --final-run \
        --freeze-manifest "$EVAL_FREEZE_MANIFEST"
done
python3 eval/report.py \
    --results "$RESULTS_DIR" \
    --split heldout \
    --conditions "$SUITE_CONDITIONS"
