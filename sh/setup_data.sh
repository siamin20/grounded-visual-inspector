#!/usr/bin/env bash
#
# MVTec AD 필요 샘플을 받고 평가 데이터셋·어휘 매핑·로트 메타데이터를 생성한다.
# 이미 받은 파일은 Hugging Face 스냅샷 다운로더가 재사용한다.
#
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x ".venv/bin/python3" ]; then
    echo "ERROR: .venv가 없습니다. 먼저 bash sh/setup_venv.sh를 실행하세요."
    exit 1
fi

./.venv/bin/python3 eval/make_dataset.py --download
