#!/usr/bin/env bash
# dev 분할에서 B0와 A를 실모델로 평가한다.
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
export LLM_BACKEND VLM_BACKEND EMBED_BACKEND EVAL_BUDGET_USD
CONDITIONS=(B0 A)

if [ "$LLM_BACKEND" = "fake" ] || [ "$VLM_BACKEND" = "fake" ]; then
    echo "ERROR: 실측 평가는 fake LLM/VLM을 허용하지 않습니다."
    echo "  LLM_BACKEND=openai VLM_BACKEND=openai EMBED_BACKEND=openai ./sh/run_eval_dev.sh"
    exit 1
fi

python3 - "$EVAL_BUDGET_USD" <<'PY'
import json
import sys
from pathlib import Path

samples = json.loads(Path("eval/dataset.json").read_text(encoding="utf-8"))["samples"]
count = sum(sample["split"] == "dev" for sample in samples)
budget = float(sys.argv[1])
print(f"dev 이미지: {count}장")
print(f"조건: B0, A (2개, 최대 {count * 2}건 처리)")
print(f"비용 가드: 조건별 ${budget:.2f}, 전체 최대 ${budget * 2:.2f}")
PY
echo "백엔드: llm=$LLM_BACKEND / vlm=$VLM_BACKEND / embed=$EMBED_BACKEND"
read -r -p "실모델 평가를 시작할까요? [y/N] " CONFIRM
case "$CONFIRM" in
    [yY]|[yY][eE][sS]) ;;
    *) echo "중단합니다."; exit 1 ;;
esac

for CONDITION in "${CONDITIONS[@]}"; do
    python3 eval/run_eval.py --condition "$CONDITION" --split dev
done
python3 eval/report.py --results eval/results --split dev --conditions B0,A
