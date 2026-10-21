# grounded-visual-inspector

> 이미지 기반 제품 결함 판정 및 근거 검증 멀티에이전트

Multi-agent visual defect inspector that grounds VLM observations in spec documents (RAG), cross-checks them, and asks a human before risky actions.

## Why

범용 비전언어모델(VLM)은 학습 없이 산업 이미지를 읽지만, 산업 외관검사 벤치마크 [MMAD (ICLR 2025)](https://arxiv.org/abs/2410.09453)에서 GPT-4o도 평균 74.9%에 그치고 틀린 근거를 지어낸다. 이 프로젝트는 VLM을 **관찰기**로 제한하고, 판정·검증·승인을 에이전트 레이어가 맡게 해 그 신뢰도를 숫자로 측정한다.

## 주요 실행 방법

최초 1회 가상환경과 MVTec AD 4품목 데이터를 준비한다. 데이터 다운로드에는 네트워크가 필요하며 이미지는 `data/mvtec/`에만 저장되고 커밋하지 않는다.

```bash
bash sh/setup_venv.sh
bash sh/setup_data.sh
```

외부 모델 호출 없이 전체 흐름과 HITL UI를 확인하려면 결정론적 fake 데모를 실행한다. 표시된 URL을 브라우저에서 열고 스크립트가 출력한 good/결함 이미지를 업로드한다.

```bash
bash sh/demo_fake.sh
```

실모델은 `.env.example`을 `.env`로 복사해 필요한 키를 설정한 뒤 기동한다. Ollama는 최초 1회 `setup_ollama.sh`로 모델을 준비한다. 프런트 프록시는 별도 터미널에서 실행한다.

```bash
cp .env.example .env
bash sh/setup_ollama.sh       # Ollama 최초 1회
bash sh/run_agent_ollma.sh    # 또는: bash sh/run_agent_openai.sh
bash sh/run_frontend.sh       # 별도 터미널, http://localhost:8600
```

테스트는 fake 백엔드로 네트워크 없이 실행한다.

```bash
LLM_BACKEND=fake VLM_BACKEND=fake EMBED_BACKEND=fake .venv/bin/python3 -m pytest -q
```

## Architecture

```
route ─▶ plan ─▶ gate ─▶ research ──▶ analysis ─▶ write ─▶ [HITL] ─▶ validate ─▶ finalize
                          ├ Observer      (VLM, full + 4 crops) ─┐
                          └ Cross-Checker (second-opinion VLM) ──┴─▶ Spec-Judge (RAG)
```

- **Multi-Agent**: Observer와 Cross-Checker를 병렬 실행한 뒤 Spec-Judge가 결과를 기준서와 대조
- **MCP**: `inspect_image` Tool / `spec://{item}`, `guide://{defect}` Resource / `inspection_prompt` Prompt
- **RAG**: 검사 기준서 조항 단위 임베딩 검색 — 판정 기준은 문서가 결정
- **Validate**: 보고서의 모든 관찰·조항 인용을 원문과 대조, 불일치 시 반려
- **HITL**: 폐기·라인 정지 등 위험 조치와 확신 낮은 판정은 사람 승인
- **LangSmith**: 전 구간 trace + Dataset 평가

## 저장소 구조

```text
agent/                 LangGraph 노드·서브그래프, RAG, VLM 백엔드
data/specs/            품목별 검사 기준서
data/guides/           결함 유형별 조치 가이드
eval/                  고정 데이터셋, 평가 실행·지표·보고서
tests/                 fake 백엔드 단위·통합 테스트
docs/adr/              핵심 아키텍처 결정 4건
sh/                    환경·데이터·서버·평가 실행 스크립트
mcp_server_vision.py   VLM Tool·기준서/guide Resource·Prompt
server.py              FastAPI(:8100), SSE, 업로드, HITL 재개
web_demo_server.py     Vue 화면 프록시(:8600)
```

## Evaluation

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) (CVPR 2019, CC BY-NC-SA 4.0) 4품목 held-out 140장. VLM 단독(Before) vs 에이전트(After) vs ablation.

실제 평가는 fake 백엔드를 거부하며 실모델 비용이 발생한다. 각 스크립트는 대상 장수와 예산을 보여 준 뒤 승인받고 실행한다.

```bash
export LLM_BACKEND=openai VLM_BACKEND=openai EMBED_BACKEND=openai
bash sh/run_eval_dev.sh       # dev: B0, A 후 summary_dev.md
bash sh/run_eval_heldout.sh   # held-out: B0, B1, A 후 summary_heldout.md
```

아래 표의 `TBD`는 실측 전 자리표시자다. `--dry-run` 결과를 성능 수치로 채우지 않고, `eval/report.py`의 실모델 집계와 블라인드 사람 대조 결과만 기록한다. 집계 분모는 `eval/dataset.json`의 해당 split 전체 ID로 고정한다. 누락·오류·`ABSTAIN`·무효 판정은 종단 지표에서 오답이며, 자동 판정분만의 선택적 지표는 coverage와 함께 제시한다.

| 지표 | B0: VLM 단독 | B1: 기준서 전문 | A: 전체 에이전트 |
|---|---:|---:|---:|
| 사전 등록 / 결과 N | TBD / TBD | TBD / TBD | TBD / TBD |
| 누락 / 오류 / ABSTAIN N | TBD / TBD / TBD | TBD / TBD / TBD | TBD / TBD / TBD |
| End-to-end Balanced Accuracy | TBD | TBD | TBD |
| False Positive Rate (FPR) / False Negative Rate (FNR) | TBD / TBD | TBD / TBD | TBD / TBD |
| Coverage | TBD | TBD | TBD |
| Selective Accuracy / Selective F1 | TBD / TBD | TBD / TBD | TBD / TBD |
| Defect-type Macro-F1 | TBD | TBD | TBD |
| 인용 충실도 | TBD | TBD | TBD |
| 지연 p50 / p95 (초) | TBD / TBD | TBD / TBD | TBD / TBD |
| 평균 비용 (USD/장) | TBD | TBD | TBD |

H1은 정상/결함 층화 paired permutation의 ΔBalanced Accuracy, H2는
품목/진실 유형 층화 paired permutation의 ΔMacro-F1로 검정하며 각각 paired
bootstrap 95% CI를 함께 기록한다: **TBD**. 전체 이진 정오 McNemar는 보조
분석이다. H3는 B0/A를 섞지 않고 동일 A 실행의 첫 write snapshot과 최종
보고서 20쌍을 블라인드 claim 채점한다(validate 전/후 관측 환각 포함률:
**TBD / TBD**). held-out 결과가 불완전하면 보고서는 최종 가설 결론을 금지한다.

상세 설계와 검증 체계는 [docs/design.md](docs/design.md), 구성도와 시퀀스 다이어그램은 [docs/architecture.md](docs/architecture.md) 참고.

## Status

Phase 1–6과 보정 R0–R4 구현 및 fake 백엔드 회귀 검증을 완료했다. held-out은 동결 게이트 전까지 잠겨 있으며, 표의 성능 수치는 실모델 최종 평가 후 채운다.

## Attribution

MVTec AD: Bergmann et al., "A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection", CVPR 2019.
