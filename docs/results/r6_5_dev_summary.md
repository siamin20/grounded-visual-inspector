# R6.5 보정 후 dev B0/A 결과 요약

## 실행 범위

- Run ID: `20261022T002750112528Z-9befe77d286f`
- Split: `dev`
- 고정 roster: 52장(정상 24, 결함 28)
- 조건: B0 단독 VLM, A 전체 Agent
- Batch: 조건별 4장 checkpoint
- 오류·누락·무효 판정: B0/A 모두 0건
- Held-out 접근: 없음

## 모델과 실행 조건

| 역할 | 모델 |
|---|---|
| Text fast | `qwen3:0.6b` |
| Text judge / quality | `qwen3:8b` |
| Observer / Cross-Checker | `qwen2.5vl:3b` |
| Embedding | `nomic-embed-text` |

Ollama는 `max_loaded_models=1`, `parallel=1`, `keep_alive=30s`,
`context=4096`로 실행했다. API 과금은 0원이지만 로컬 계산 자원 proxy를
별도로 기록했다.

## 핵심 결과

| 지표 | B0 | A |
|---|---:|---:|
| 전체 / 결과 | 52 / 52 | 52 / 52 |
| Accuracy | 28.85% | 9.62% |
| Balanced Accuracy | 26.79% | 10.42% |
| `ABSTAIN` | 29 | 42 |
| Coverage | 44.23% | 19.23% |
| 정상 정답 / 24 | 0 | 5 |
| 결함 정답 / 28 | 15 | 0 |
| Selective Accuracy | 65.22% | 50.00% |
| Defect-type Macro-F1 | 13.57% | 0.00% |
| Citation fidelity | 60.00% | 68.40% |
| 지연 p50 / p95 | 4.269초 / 9.110초 | 65.405초 / 84.120초 |
| 평균 compute proxy / 장 | 0.000573 | 0.022970(setup 상각 포함) |
| HITL 필요 발생률 | 100.00% | 88.46% |

`ABSTAIN`은 정확도 분모에서 제외하지 않고 오답으로 처리했다. 배치 실행은
`auto_approve`이므로 HITL 발생률을 사람 판단에 의한 정확도 향상으로 해석하지 않는다.

## 통계와 해석

- H1 ΔBalanced Accuracy(A−B0): `-16.37%p`
- Paired bootstrap 95% CI: `[-28.0, -3.9]%p`
- 층화 paired permutation 양측 p-value: `0.040798`
- H2 ΔMacro-F1(A−B0): `-13.57%p`, `p=0.064453`
- 보조 McNemar(B0 vs A, 전체): `p=0.041389`

dev 탐색 결과이므로 held-out 최종 가설 결론은 내리지 않는다. 다만 현재 A는
B0보다 정확도·balanced accuracy·coverage·결함 재현율이 모두 낮아 성능 개선을
입증하지 못했다. strict Cross 계약은 불일치 근거를 확정 판정으로 넘기지 않았지만,
3B VLM의 관찰 품질을 높이지 못해 `ABSTAIN`과 결함 미검이 증가했다.

## 무결성

- Manifest SHA-256:
  `69572a98da559c41e30cca73fdcd358644150c3a1d151abaedb925066e9af1d0`
- Immutable provenance SHA-256:
  `a690c158ce79d9a1d16cec9eb8db75899d2e3bf78e50563d4f1910eb839b763b`
- 실행 시 Git commit:
  `54eb9d8df7dfb8fe060b9658ffa60861be0c2e0f`
- 실행 시 dirty diff SHA-256:
  `2dfc17b0523c027675d33524938851ea9118265c176b2124d07a11fdbdd66365`

원시 JSONL과 manifest는 로컬 `eval/results/`에 보존한다. 원시 행에는 로컬 이미지
경로가 포함되므로 Git에는 올리지 않고, 제출 저장소에는 이 비식별 요약만 추적한다.
따라서 공개 저장소만으로 원시 행을 독립 재집계할 수는 없다. 이 문서는 실행 snapshot과
hash를 공개하는 범위의 증거이며, 원시 결과 공개를 대신한다고 주장하지 않는다.
