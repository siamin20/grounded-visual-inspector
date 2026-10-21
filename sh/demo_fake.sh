#!/usr/bin/env bash
# 외부 호출 없이 발표 리허설을 위한 fake 데모를 시작한다.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -x ".venv/bin/python3" ] && export PATH="$PWD/.venv/bin:$PATH"
export LLM_BACKEND=fake
export VLM_BACKEND=fake
export EMBED_BACKEND=fake
export HITL_MODE=interactive
export SUPERVISOR_VERSION=v2

python3 - <<'PY'
import json
from pathlib import Path

from eval.preflight import select_dev_demo_samples

root = Path.cwd()
samples = json.loads((root / "eval/dataset.json").read_text(encoding="utf-8"))["samples"]
good, defect = select_dev_demo_samples(samples)
print("데모용 dev 이미지(브라우저 업로드 창에서 선택):")
print(f"  good  [{good['id']}, split={good['split']}]: {root / good['path']}")
print(f"  defect[{defect['id']}, split={defect['split']}]: {root / defect['path']}")
PY

lsof -ti:8100 | xargs kill -9 2>/dev/null || true
lsof -ti:8600 | xargs kill -9 2>/dev/null || true

cleanup() {
    echo
    echo "fake 데모 서버와 프런트엔드를 종료합니다..."
    lsof -ti:8100 | xargs kill -9 2>/dev/null || true
    lsof -ti:8600 | xargs kill -9 2>/dev/null || true
}
trap cleanup EXIT

python3 server.py &
BACKEND_PID=$!
for _ in $(seq 1 40); do
    curl -s http://127.0.0.1:8100/health >/dev/null 2>&1 && break
    sleep 0.5
done
if ! curl -s --max-time 2 http://127.0.0.1:8100/health >/dev/null 2>&1; then
    echo "ERROR: fake 백엔드 기동에 실패했습니다."
    exit 1
fi

python3 web_demo_server.py &
FRONTEND_PID=$!
for _ in $(seq 1 40); do
    curl -s http://127.0.0.1:8600/health >/dev/null 2>&1 && break
    sleep 0.5
done
if ! curl -s --max-time 2 http://127.0.0.1:8600/health >/dev/null 2>&1; then
    echo "ERROR: 프런트엔드 기동에 실패했습니다."
    exit 1
fi

cat <<'EOF'

################################################################
 fake 데모가 준비됐습니다 → http://localhost:8600
 위에 출력한 두 데모용 이미지 중 하나를 업로드하세요.
################################################################

종료하려면 이 터미널에서 Ctrl+C를 누르세요.
EOF

wait "$FRONTEND_PID"
wait "$BACKEND_PID"
