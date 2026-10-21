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
EVAL_BUDGET_USD="${EVAL_BUDGET_USD:-15}"
EVAL_FREEZE_MANIFEST="${EVAL_FREEZE_MANIFEST:-eval/freeze_manifest.json}"
export LLM_BACKEND VLM_BACKEND EMBED_BACKEND EVAL_BUDGET_USD EVAL_FREEZE_MANIFEST
CONDITIONS=(B0 B1 A)

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

python3 - "$EVAL_BUDGET_USD" <<'PY'
import json
import sys
from pathlib import Path

samples = json.loads(Path("eval/dataset.json").read_text(encoding="utf-8"))["samples"]
count = sum(sample["split"] == "heldout" for sample in samples)
budget = float(sys.argv[1])
print(f"heldout 이미지: {count}장")
print(f"조건: B0, B1, A (3개, 최대 {count * 3}건 처리)")
print(f"비용 가드: 조건별 ${budget:.2f}, 전체 최대 ${budget * 3:.2f}")
PY
echo "백엔드: llm=$LLM_BACKEND / vlm=$VLM_BACKEND / embed=$EMBED_BACKEND"
echo "동결 manifest: $EVAL_FREEZE_MANIFEST"
read -r -p "최종 heldout 평가면 HELDOUT 140 FINAL을 입력하세요: " CONFIRM
if [ "$CONFIRM" != "HELDOUT 140 FINAL" ]; then
    echo "중단합니다."
    exit 1
fi

for CONDITION in "${CONDITIONS[@]}"; do
    python3 eval/run_eval.py \
        --condition "$CONDITION" \
        --split heldout \
        --final-run \
        --freeze-manifest "$EVAL_FREEZE_MANIFEST"
done
python3 eval/report.py --results eval/results --split heldout --conditions B0,B1,A
