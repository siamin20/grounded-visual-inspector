#!/usr/bin/env bash
# dev B0/A를 자원 제한 Ollama 서버에서 한 배치씩 실행한다.
#
# 사용법:
#   bash sh/run_eval_dev_batch.sh B0 4 <run-id>
#   bash sh/run_eval_dev_batch.sh B0 4 <run-id> --resume
#   bash sh/run_eval_dev_batch.sh A  4 <run-id>
#
# 같은 run-id를 계속 사용하고, paused 조건을 잇는 호출에만 --resume을 붙인다.
# 배치가 끝나면 이 스크립트가 시작한 Ollama 서버를 내려 통합 메모리를 해제한다.
set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
    echo "사용법: bash sh/run_eval_dev_batch.sh <B0|A> <batch-size> <run-id> [--resume]"
}

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    usage
    exit 2
fi

CONDITION="$1"
BATCH_SIZE="$2"
RUN_ID="$3"
RESUME_ARG="${4:-}"

case "$CONDITION" in
    B0|A) ;;
    *) echo "ERROR: 조건은 B0 또는 A여야 합니다."; exit 2 ;;
esac
case "$BATCH_SIZE" in
    ''|*[!0-9]*) echo "ERROR: batch-size는 1 이상의 정수여야 합니다."; exit 2 ;;
esac
if [ "$BATCH_SIZE" -ne 4 ]; then
    echo "ERROR: 정식 dev B0/A의 batch-size는 4로 고정됩니다."
    exit 2
fi
if [ -n "$RESUME_ARG" ] && [ "$RESUME_ARG" != "--resume" ]; then
    usage
    exit 2
fi
if [ ! -x ".venv/bin/python3" ]; then
    echo "ERROR: .venv가 없습니다. 먼저 bash sh/setup_venv.sh 를 실행하세요."
    exit 1
fi
command -v ollama >/dev/null 2>&1 \
    || { echo "ERROR: ollama가 설치되어 있지 않습니다."; exit 1; }

# 이미 떠 있는 서버를 임의로 종료하지 않는다. 이 실행은 아래 자원 제한으로
# 직접 띄운 서버만 사용해야 배치 간 메모리 해제를 보장할 수 있다.
if curl -s --max-time 1 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "ERROR: 기존 Ollama 서버가 실행 중입니다. 종료한 뒤 다시 실행하세요."
    exit 1
fi

OLLAMA_LOG="${TMPDIR:-/tmp}/grounded-visual-inspector-ollama-batch.log"
OLLAMA_PID=""
cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [ -n "$OLLAMA_PID" ] && kill -0 "$OLLAMA_PID" 2>/dev/null; then
        kill "$OLLAMA_PID" 2>/dev/null || true
        wait "$OLLAMA_PID" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# 동시에 여러 모델이 통합 메모리에 남아 브라우저 렌더러와 경합하지 않게 한다.
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_KEEP_ALIVE=30s
export OLLAMA_CONTEXT_LENGTH=4096
export OLLAMA_NO_CLOUD=1
export OLLAMA_URL=http://127.0.0.1:11434
export OLLAMA_FAST_NUM_PREDICT=512
export OLLAMA_JUDGE_NUM_PREDICT=512
export OLLAMA_QUALITY_NUM_PREDICT=2048
export OLLAMA_LLM_WALL_TIMEOUT_S=300

ollama serve >"$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!
for _ in $(seq 1 60); do
    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
        echo "ERROR: Ollama 서버 프로세스가 종료됐습니다. 로그: $OLLAMA_LOG"
        exit 1
    fi
    if curl -s --max-time 1 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        break
    fi
    sleep 0.25
done
if ! curl -s --max-time 1 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "ERROR: 자원 제한 Ollama 서버를 시작하지 못했습니다. 로그: $OLLAMA_LOG"
    exit 1
fi
if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
    echo "ERROR: 시작한 Ollama 서버가 아닌 다른 서버가 응답했습니다."
    exit 1
fi

echo "run_id=$RUN_ID condition=$CONDITION batch_size=$BATCH_SIZE resume=${RESUME_ARG:-no}"
echo "Ollama 자원 제한: endpoint=127.0.0.1:11434 loaded_models=1 parallel=1 keep_alive=30s context=4096"
echo "Ollama LLM 생성 제한: think=false fast=512 judge=512 quality=2048 wall_timeout=300s"

run_eval_batch() {
    env \
    LLM_BACKEND=ollama \
    VLM_BACKEND=ollama \
    EMBED_BACKEND=ollama \
    OLLAMA_FAST_MODEL=qwen3:0.6b \
    OLLAMA_JUDGE_MODEL=qwen3:8b \
    OLLAMA_QUALITY_MODEL=qwen3:8b \
    VLM_OBSERVER_MODEL=qwen2.5vl:3b \
    VLM_CROSS_MODEL=qwen2.5vl:3b \
    EMBED_MODEL=nomic-embed-text \
    HITL_MODE=auto_approve \
    LANGCHAIN_TRACING=false \
    LANGCHAIN_TRACING_V2=false \
    LANGSMITH_TRACING=false \
    EVAL_SUITE_BUDGET_USD=15 \
    PYTHONUNBUFFERED=1 \
    .venv/bin/python3 eval/run_eval.py \
        --condition "$CONDITION" \
        --split dev \
        --run-id "$RUN_ID" \
        --suite-conditions B0,A \
        --batch-size "$BATCH_SIZE" \
        "$@"
}

if [ "$RESUME_ARG" = "--resume" ]; then
    run_eval_batch --resume
else
    run_eval_batch
fi

