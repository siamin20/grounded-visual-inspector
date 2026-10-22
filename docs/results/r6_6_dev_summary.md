# R6.6 dev B0/A 정식 결과 요약

## 실행 범위

- Run ID: `20261022T112100000000Z-r66dev`
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

Observer와 Cross-Checker는 같은 모델을 써서 반전·프롬프트 차이만 있으며,
모델 독립성은 제한된다. Ollama는 `max_loaded_models=1`, `parallel=1`,
`keep_alive=30s`, `context=4096`로 실행했다. API 과금은 0원이지만
로컬 계산 자원 proxy를 별도로 기록했다.

## 핵심 결과

| 지표 | B0 | A |
|---|---:|---:|
| 전체 / 결과 | 52 / 52 | 52 / 52 |
| Accuracy | 28.85% | 11.54% |
| Balanced Accuracy | 26.79% | 12.50% |
| `ABSTAIN` | 29 | 41 |
| Coverage | 44.23% | 21.15% |
| 정상 정답 / 24 | 0 | 6 |
| 결함 정답 / 28 | 15 | 0 |
| Selective Accuracy | 65.22% | 54.55% |
| Defect-type Macro-F1 | 13.57% | 0.00% |
| Citation fidelity | 60.00% | 68.40% |
| 지연 p50 / p95 | 4.431초 / 9.505초 | 68.314초 / 83.178초 |
| 평균 compute proxy / 장 | 0.000573 | 0.031484(setup 상각 포함) |
| HITL 필요 발생률 | 100.00% | 80.77% |

`ABSTAIN`은 정확도 분모에서 제외하지 않고 오답으로 처리했다. 배치 실행은
`auto_approve`이므로 HITL 발생률을 사람 판단에 의한 정확도 향상으로 해석하지 않는다.

## 통계와 해석

- H1 ΔBalanced Accuracy(A−B0): `-14.29%p`
- Paired bootstrap 95% CI: `[-26.2, -1.2]%p`
- 층화 paired permutation 양측 p-value: `0.084896`
- H2 ΔMacro-F1(A−B0): `-13.57%p`, `p=0.064453`
- 보조 McNemar(B0 vs A, 전체): `p=0.078354`

dev 탐색 결과이므로 held-out 최종 가설 결론은 내리지 않는다. 다만 현재
A는 B0보다 Accuracy·Balanced Accuracy·Coverage·결함 재현율이 모두
낮아 종단 성능 개선에 실패했다. Cross에 정상 참조를 추가하고 구조 계약을
보강한 R6.6은 정상 정답을 R6.5의 5장에서 6장으로 늘렸지만, 결함 28장은
여전히 하나도 맞히지 못했다. 병목은 RAG 후단보다 3B VLM의 초기 관찰 품질이다.

## A+ post-hoc dev 탐색

같은 run의 B0/A 저장 행을 이미지 재호출 없이 결합했다. 규칙은 다음과 같다.

```text
B0 == REJECT  -> REJECT
else A == PASS -> PASS
else           -> ABSTAIN
```

이 규칙은 실행 중 정답 라벨을 사용하지 않는다. 다만 dev 결과를 본 후
정한 사후 탐색 정책이므로 기존 A의 성과가 아니며, held-out 동결 전 후보로만 본다.

| 지표 | A+ |
|---|---:|
| Accuracy | 38.46% |
| Balanced Accuracy | 37.20% |
| 자동 판정분의 `TP / TN / FP / FN` | 15 / 5 / 8 / 1 |
| `ABSTAIN` / Coverage | 23 / 55.77% |
| Selective Accuracy | 68.97% |

B0 대비 Accuracy는 `+9.62%p`, Balanced Accuracy는 `+10.42%p`였다.
짝지은 검정 `p=0.0625`로 유의하지 않고, B0와 A를 둘 다 실행해야 하므로
지연·계산 자원이 늘어난다. 또한 B0가 보류한 실제 결함 1건을 A가 `PASS`로
바꾸는 미검 위험이 있었다. 따라서 이 수치는 “정확도 개선 입증”이 아니라
held-out 재현이 필요한 다음 동결 후보다. 위 혼동행렬은 `ABSTAIN` 23건을
제외한 자동 판정분만 나타내며, 전체 52장 정확도에는 보류를 오답으로 포함했다.

## RAG 검색 비교

이전 R6의 저장된 A 실제 query만 재사용한 결함 28건 top-3 검색 결과는
`nomic-embed-text` 17/28(60.71%), `bge-m3` 23/28(82.14%)였다.
이는 이미지를 다시 판독한 결과가 아니며, **검색 컴포넌트의 개선**이지
종단 A 정확도 개선으로 해석하지 않는다.

## 무결성

- Manifest 내장 self-hash: `311a2a6f6d9d458a9e65267952ba7d19785c356b51c0dc3589357e7ca6fb6a59`
- 최종 manifest 파일 SHA-256:
  `3e314a28198f8b7bcf3a80b80a03ec6d05988f94d33231d71594fe601487c4d4`
- Immutable provenance SHA-256:
  `c4da6b336d09fb645f1826e64701bea013c74150e63394a2ada253fcca6c8446`
- 실행 시 Git commit: `6e0709e89b504cb684e73c60d04ca1efaa36c921`
- 실행 시 repository 상태: clean
- B0 JSONL SHA-256: `59808d0d70d9d5632b9a44c2c3f27a2850fee3165a2624b513e178dff15bc6e0`
- A JSONL SHA-256: `a433a459e0162584c6e980fe37aa49b9a9ed4e3a9d18cefa99024186da97c1c2`

원시 JSONL과 manifest는 로컬 `eval/results/`에 보존한다. 원시 행에는 로컬 이미지
경로가 포함되므로 Git에는 올리지 않고, 제출 저장소에는 이 비식별 요약만 추적한다.
따라서 공개 저장소만으로 원시 행을 독립 재집계할 수 없다는 한계가 남는다.
