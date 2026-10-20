# grounded-visual-inspector

> 근거 검증 루프를 가진 VLM 외관 불량 판정 멀티에이전트

Multi-agent visual defect inspector that grounds VLM observations in spec documents (RAG), cross-checks them, and asks a human before risky actions.

## Why

범용 비전언어모델(VLM)은 학습 없이 산업 이미지를 읽지만, 산업 외관검사 벤치마크 [MMAD (ICLR 2025)](https://arxiv.org/abs/2410.09453)에서 GPT-4o도 평균 74.9%에 그치고 틀린 근거를 지어낸다. 이 프로젝트는 VLM을 **관찰기**로 제한하고, 판정·검증·승인을 에이전트 레이어가 맡게 해 그 신뢰도를 숫자로 측정한다.

## Architecture

```
route ─▶ plan ─▶ gate ─▶ research ──▶ analysis ─▶ write ─▶ [HITL] ─▶ validate ─▶ finalize
                          ├ Observer      (VLM, full + 4 crops, parallel)
                          ├ Spec-Judge    (RAG over inspection specs)
                          └ Cross-Checker (second-opinion VLM pass)
```

- **Multi-Agent**: research를 3개 서브그래프로 분리, supervisor가 병렬 호출
- **MCP**: `inspect_image` Tool / `spec://{item}`, `guide://{defect}` Resource / `inspection_prompt` Prompt
- **RAG**: 검사 기준서 조항 단위 임베딩 검색 — 판정 기준은 문서가 결정
- **Validate**: 보고서의 모든 관찰·조항 인용을 원문과 대조, 불일치 시 반려
- **HITL**: 폐기·라인 정지 등 위험 조치와 확신 낮은 판정은 사람 승인
- **LangSmith**: 전 구간 trace + Dataset 평가

## Evaluation

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) (CVPR 2019, CC BY-NC-SA 4.0) 4품목 held-out 140장. VLM 단독(Before) vs 에이전트(After) vs ablation.

| 지표 | Before | After |
|---|---|---|
| 양불 정확도 | TBD | TBD |
| FNR (결함 미검) | TBD | TBD |
| 유형 정확도 | TBD | TBD |
| 환각률 | TBD | TBD |
| p50 지연 / 비용 | TBD | TBD |

상세 설계와 검증 체계는 [docs/design.md](docs/design.md) 참고.

## Status

Day1 — 설계 완료, 구현 시작.

## Attribution

MVTec AD: Bergmann et al., "A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection", CVPR 2019.
