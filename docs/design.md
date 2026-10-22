# 설계서 — 이미지 기반 제품 결함 판정 및 근거 검증 멀티에이전트

작성일 2026-10-20, 최종 갱신 2026-10-21 (v2.5). 과제 유형: 2번 신규 프로젝트 (Base AI Agent 구조 이식).
v2.5 변경: v2.4의 자원 제한 배치 checkpoint/재개 계약에
더해, 실측에서 발견한 Qwen3 무제한 생성을
`think=false`·역할별 출력 상한·전체 스트림 제한으로 차단했다.

---

## 0. 한 문단 요약

범용 비전언어모델(VLM)은 학습 없이 산업 이미지를 읽을 수 있지만 산업 외관검사 벤치마크(MMAD, ICLR 2025)에서 최상위 상용 모델도 평균 정확도 74.9%에 그쳐 현장 요구 수준에 못 미친다. 이 프로젝트는 VLM을 "관찰기"로 격하하고, 기준서 RAG로 근거 조항을 찾은 뒤 코드 규칙이 시각 결과와 함께 종합 판정한다. 환각 억제는 별도 관점 교차검증과 원문 대조 validate가, 위험 조치는 사람 승인(HITL)이 맡는 에이전트 운영 레이어를 Base AI Agent 그래프 위에 구현한다. 공개 벤치마크 MVTec AD의 외부 라벨로 VLM 단독(Before) 대비 에이전트(After)의 정확도·환각률·비용 변화를 같은 이미지에 대한 짝지은 비교로 측정하고, 효과가 없으면 없다고 보고한다.

---

## 1. 왜 이 프로젝트인가

### 1.1 현장 문제
제조 외관검사는 전용 CV 모델이 담당해 왔다. 품목·라인별로 라벨을 모으고 재학습해야 하므로 소량 다품종이나 신규 품목에서는 투입이 늦다. 범용 VLM(GPT-4o, Qwen2.5-VL 등)은 학습 없이 "이 나사 목에 긁힘이 있다"고 말할 수 있지만 신뢰도가 부족하다.

- MMAD(Jiang et al., ICLR 2025): 산업 이미지 8,366장·질문 39,672개에서 GPT-4o 74.9%, GPT-4o-mini 66.3%, 최고 오픈소스 70.8%. 저자들이 "산업 요구에 크게 못 미친다"고 명시.
- Eyes Wide Shut(Tong et al., CVPR 2024): 기본 시각 패턴에서 GPT-4V가 틀린 답에 지어낸 설명을 붙인다.
- Vision language models are blind(Rahmanzadehgervi et al., ACCV 2024): 원 겹침·선 교차 같은 초등 과제에서 주요 상용 비전언어모델의 평균 성능이 약 58%.

학습 없는 범용 지각기는 있지만 그 위에서 신뢰할 수 있는 판정을 만드는 운영 레이어는 비어 있다. 이 프로젝트는 그 공백을 겨냥한다.

### 1.2 왜 CV 모델 대신 에이전트인가
| 요구 | CV 모델 단독 | VLM 단독 | 에이전트 레이어 |
|---|---|---|---|
| 신규 품목 즉시 투입 | ✗ 재학습 | ○ | ○ |
| 판정 기준이 품목·고객마다 다름 | 재학습 | 프롬프트에 하드코딩 | 기준서 RAG (문서만 교체) |
| 판정 근거를 사람이 읽을 수 있음 | ✗ | △ 환각 섞임 | 관찰 인용 강제 + validate |
| 오판 가능성 자각(확신도) | 점수만 | ✗ 과신 | 별도 관점 교차검증 → 확신 플래그 |
| 폐기·라인 정지 같은 위험 조치 | 사람 별도 | ✗ | HITL 승인 노드 |
| 판정 과정 추적·감사 | 로그 | ✗ | LangSmith 계층 trace |

### 1.3 왜 이 과제에 맞는가
지시서 9쪽은 과제 2번을 "Base AI Agent의 주제 대신 다른 주제로 바꾸어 신규 AI Agent로 개발"로 정의하고 주제 제약 없음을 두 번 명시한다. 지시서 10쪽 개선 포인트와의 대응은 ①키워드→RAG(기준서·조치 가이드 검색) ②모듈화(노드 분리) ③research 서브그래프 분리 ④write→validate 사이 HITL 승인이다. 네 개를 한 프로젝트에서 모두 다룬다. 비전은 MCP tool 하나에 갇혀 있고 점수는 전부 에이전트 설계로 받는다.

### 1.4 Base 패턴 유지표
평가자가 Base를 만든 사람이므로 무엇을 그대로 쓰고 무엇을 바꾸는지 먼저 밝힌다.

| Base 패턴 (Wave) | 이 프로젝트 | 변경 여부 |
|---|---|---|
| lifespan 싱글턴 `SupervisorDeps` | 유지 + `vector_store`, `vlm` 주입 추가 | 확장 |
| `LLM_BACKEND` fake/ollama/openai 토글 | 유지. VLM도 같은 토글로 fake/ollama/openai | 확장 |
| `SUPERVISOR_VERSION` v1/v2 토글 | v1 자리를 "VLM 단독(B0)"로 재구성 → Before 스위치 | 용도 변경 |
| MCP 멀티 서버 + `find_prompt/find_resource_template` 탐색 | Base ops 서버 파일은 보존하되 기본 연결에서 제외, vision 추가 + docs decoy 유지 | 교체 |
| MCP 의도적 flaky + `with_retry` | 유지 (VLM tool 호출도 with_retry로 감쌈) | 유지 |
| SSE 이벤트 스트리밍 | 유지 + 이벤트 타입 추가 (4.4) | 확장 |
| route → chat / 무거운 파이프라인 분기 | 유지 (이미지 유무로 분기) | 유지 |
| gate 코드 규칙 | 유지 | 유지 |
| replan_check 1회 상한 | 유지 (추가 크롭 task) | 유지 |
| validate → write 반려 루프 | 유지 + 원문 대조 규칙 교체 | 강화 |
| fast/quality 비용 분리 | 유지 (텍스트 fast/quality 역할 분리, 실행 자원량 분리) | 유지 |
| SQLite 세션 저장소 | 유지 + LangGraph checkpointer 추가 (HITL용) | 확장 |
| Vue 채팅 UI + 프록시 | 유지 + 이미지 업로드·승인 버튼 추가 | 확장 |

---

## 2. 가설 (측정 가능한 형태로 사전 등록)

같은 held-out ID를 유지하되 가설의 주 지표와 같은 통계량을 검정한다.
H1은 주 확증 가설, H2·H3은 보조 가설, H4는 탐색적 보정 진단으로
계층화한다. H2·H3의 p-value는 다중성 보정 없이 각각 보고하므로
독립적인 확정 결론으로 일반화하지 않는다. 신뢰구간은 효과 크기 서술용이다.

| ID | 가설 | 지표 | 판정 기준 |
|---|---|---|---|
| H1 | 전체 에이전트 A는 VLM 단독 B0보다 결함 유무 판정이 정확하다 | balanced accuracy, FNR, FPR | 정상/결함 층화 paired permutation 양측 p<0.05 이고 ΔBA(A−B0)>0 |
| H2 | 전체 에이전트 A는 B0보다 결함 유형 분류가 정확하다 | 유형 macro-F1 (MVTec 폴더 라벨 기준) | 진실 유형 층화 paired permutation 양측 p<0.05 이고 Δmacro-F1(A−B0)>0 |
| H3 | validate 반려·재작성은 동일 A 실행의 보고서 환각을 줄인다 | 동일 20 ID의 validate 직전/최종 보고서 환각 포함률 | 관측 After ≤5%, Before보다 낮음, exact McNemar p<0.05 |
| H4 | `confidence=low`는 high보다 자동 판정 오답을 농축한다 | low PASS/REJECT 오답률 vs high PASS/REJECT 오답률 | 각 그룹 20건 이상에서 low 오답률이 high보다 엄격히 높고 ≥2× |
| C1 | 비용: 호출 수 증가로 지연·토큰이 늘어난다 | p50/p95 지연, 토큰, 추정 USD | 실측 보고 (트레이드오프) |

사전 등록은 기준을 먼저 적어두고 결과가 못 미치면 "효과 미입증"으로 그대로 발표한다는 뜻이다. 결과를 본 뒤 기준을 바꾸지 않는다.
R1 계약상 `ABSTAIN`은 모든 주 성능 지표와 McNemar 정오표에서 오답으로 남기며,
선택적 지표에만 제외할 수 있다. coverage 없이 선택적 정확도만 발표하지 않는다.

---

## 3. 데이터

### 3.1 평가 데이터: MVTec AD (외부 라벨)
- 출처: Bergmann et al., "A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection", CVPR 2019. HuggingFace 미러 `foersben/mvtec-ad`(원본 폴더 구조 유지). 라이선스 CC BY-NC-SA 4.0 → 비상업 과제 사용 가능, README에 출처 표기.
- 구조(확인 완료): `{품목}/test/good/`, `{품목}/test/{결함유형}/`, `{품목}/ground_truth/{결함유형}/`(픽셀 마스크). 폴더명이 곧 라벨이므로 내가 라벨을 만들지 않는다.
- 이미지는 레포에 커밋하지 않는다. `sh/setup_data.sh`가 받고 `data/mvtec/`은 `.gitignore`에 둔다.
- 사용 품목과 유형 (언어로 구분 가능한 것만):

| 품목 | 사용 유형 | 제외 유형과 이유 |
|---|---|---|
| screw | good, scratch_head, scratch_neck, thread_side, thread_top | manipulated_front — 앞면 가공 변형은 시각 언어로 기술이 모호 |
| bottle | good, broken_large, broken_small, contamination | 없음 |
| metal_nut | good, bent, color, scratch | flip — 뒤집힘은 결함이 아니라 자세 문제로 기준서 작성이 어색 |
| hazelnut | good, crack, cut, hole, print | 없음 (시간 부족 시 이 품목부터 제외) |

- 유형 어휘 매핑 표 `eval/type_map.json`: 에이전트 출력 유형(한국어) ↔ MVTec 폴더명. 채점 전에 고정.
- 샘플링: 품목당 good 20장 + 결함 유형별 8장 → 4품목 192장 (screw 52, bottle 44, metal_nut 44, hazelnut 52). hazelnut을 빼면 140장. 정상:결함 비율이 1:2라 단순 정확도 대신 전체 ID를 분모로 한 balanced accuracy와 정상·결함 실패율을 주 지표로 쓴다. PASS/REJECT 자동 판정분의 F1은 `selective_f1`로 구분해 coverage와 함께만 보고한다. 난수 시드 고정, 파일 목록을 `eval/dataset.json`에 저장.
- dev / held-out 분리: 품목·유형별 층화 결과 dev 52장과 held-out 140장으로 고정됐다. dev는 프롬프트·임계값 조정에 쓰고 held-out은 최종 보고에만 쓰며 개발 중 열어보지 않는다. 튜닝 누출을 막기 위해서다.

### 3.2 참고 문서 (직접 작성하지만 채점 정답으로는 쓰지 않음)
기준서와 조치 가이드는 공개 데이터에 없으므로 직접 쓴다. 이 문서들은 에이전트가 판정할 때 참고하는 자료일 뿐이고 정확도 채점의 정답은 3.1의 MVTec 라벨이다. 내가 쓴 문서로 내가 채점하는 순환을 피하기 위한 구분이다.

- 검사 기준서 `data/specs/{품목}.md`: 품목별 허용/불허 기준. 조항 ID 부여(예 `SCREW-03: 목 부위에 나사산을 가로지르는 긁힘이 보이면 불합격`). RAG 색인 대상. 메타데이터: 품목, 결함유형.
  - 기준은 이미지에서 확인 가능한 상대적·정성적 표현으로만 쓴다("머리 지름의 1/4 이상", "나사산을 가로지르는", "입구 테두리가 끊긴"). mm 같은 절대 치수는 VLM이 측정할 수 없어 판정이 지어낸 수치에 의존하게 되므로 쓰지 않는다.
  - 작성 규칙(편향 차단): MVTec 논문의 결함 유형 정의와 일반 품질 기준으로만 쓴다. 특정 테스트 이미지를 보고 쓰지 않는다. held-out 이미지는 열지 않는다. 작성 완료 후 git에 고정하고 평가 중 수정하지 않는다.
- 조치 가이드 `data/guides/{결함유형}.md`: 재작업/폐기/라인 점검 등 후속 조치와 위험도(HITL 트리거 여부).
- 정상 참조 이미지: 품목당 `train/good` 1장을 VLM에 함께 제공(MMAD 기본 설정과 동일한 1-shot 비교). Before에도 똑같이 제공한다.
- 언어 정책: 기준서·관찰·보고서 모두 한국어로 통일한다. VLM 프롬프트가 한국어 관찰 출력을 강제하고, 영어로 나오면 후처리에서 재요청 1회. 검색 품질은 5.2의 RAG hit rate로 따로 측정한다.

### 3.3 조회용 메타데이터(합성, 판정에 미사용)
`data/lots.json`: 이미지 → 가상 로트번호/라인/시각. 보고서 서식용이며 정확도 판정에 쓰지 않는다.

---

## 4. 아키텍처

### 4.1 그래프 (Base `graph.py` 노드 유지)
전체 구성도·상태 흐름·시퀀스 다이어그램은 [architecture.md](architecture.md) 에 mermaid 로 있다.

```
route ──chat──▶ chat_answer
  └──inspection──▶ plan ▶ gate ▶ research ▶ replan_check ▶ analysis ▶ write ▶ [HITL approve] ▶ validate ▶ finalize
                                   │                                                              │ 불합격
                                   │  1단계 (병렬): Observer ∥ Cross-Checker                       └──▶ write (최대 2회)
                                   │  2단계       : Spec-Judge (시각 양성 관찰 근거를 입력으로)
                                   ▲ 부족 시 추가 크롭 (1회)
```
research는 2단계다. Spec-Judge가 Observer와 유효한 Cross-Checker가 만든 시각 양성 근거를 쿼리로 쓰기 때문에 두 분기와 동시에 돌 수 없다. Observer 양성이 있으면 그 관찰만 쓰고, Observer 양성이 없지만 구조 계약을 통과한 Cross-Checker가 양성이면 Cross-Checker 근거만 쓴다. 둘 다 양성이 아닐 때만 판독 가능한 Observer 음성 관찰을 정상 근거로 유지한다. 병렬인 것은 Observer와 Cross-Checker, 그리고 Observer 내부의 크롭 5회 호출이다.

- route: 요청에 이미지가 있으면 inspection, 없으면 chat.
- plan: MCP Prompt `inspection_prompt(item, lot)` 조회 → tasks `[observe, cross_check, spec_judge]`.
- gate: 코드 규칙으로 불필요 task 제거. 품목(item)은 필수 입력이다(UI 드롭다운, 배치는 경로에서). VLM에게 품목까지 맞히게 하면 실패 지점이 하나 더 생기므로 범위에서 뺀다. 품목이 없으면 gate가 사용자에게 되묻고 종료.
- research — compiled subgraph 3개
  1. Observer (1단계): MCP Tool `inspect_image(path, region="full")` 1회 + 4분할 크롭 4회 병렬. 출력에는 관찰 사실만(위치·크기·색·형태) 담는다. "불량/합격" 단어 금지(프롬프트 + 후처리 필터). 자유 문장과 함께 구조화 필드 `{anomaly_seen: bool|null, regions: [tl|tr|bl|br|center], features: [...]}`를 반드시 채운다. `null`은 정상 관찰이 아니라 판독 불가다.
  2. Cross-Checker (1단계, Observer와 병렬): 같은 이미지를 좌우 반전해 체크리스트형 프롬프트로 재판독. openai 백엔드에서는 Observer가 `gpt-4o-mini`, Cross-Checker가 `gpt-4o`로 모델도 다르게 한다. ollama 백엔드는 모델이 하나라 반전+프롬프트 차이만 두며, 이 경우 독립성이 약하다는 점을 결과에 적는다. 출력은 `{anomaly_seen, regions, features, checks}`이며 `checks`는 `scratch|break_or_crack|contamination|discoloration|bend_or_deformation|thread_damage` 여섯 키를 정확히 한 번씩 가진 boolean 객체다. `anomaly_seen`은 `any(checks.values())`와 일치해야 하고, 양성에는 위치·특징이 필요하며 음성의 위치·특징은 빈 배열이어야 한다. 키 누락·추가, boolean 이외 값, `checks`·`anomaly_seen`·`regions`·`features` 구조 필드 간 모순은 `C?`로 닫는다. 반전 좌표는 원좌표로 되돌려 기록한다. 자유문장 요약의 의미와 구조 필드가 실제 이미지에 비추어 참인지는 이 형식 검사만으로 증명하지 못한다.
     서술용 `text`만 생략됐고 위 네 구조 필드가 완전히 일치하는 경우에만
     `checks`·`features`에서 한국어 요약을 결정론적으로 만든다. 구조 모순을
     요약으로 복구하지 않으며, 생성한 요약에는
     `summary_synthesized=true`를 남겨 판정에서는 `C?`로 유지한다. 원문이
     없는 응답을 `C-`로 승격시켜 확실한 합격으로 보내지 않는다.
  3. Spec-Judge (2단계): 정답 라벨·파일 경로를 제외하고, 양성 Observer 관찰과 `C+` Cross-Checker 요약·특징을 결정론적으로 합성한 문장을 쿼리로 기준서 RAG top-3 조항 검색 → 조항별 충족/위반 판정 + 결함 유형 후보. 양성이 없을 때는 판독 가능한 Observer 음성 관찰을 정상 근거로 쓴다.
- replan_check: 아래 Observer 집계가 `O?`일 때만 center 크롭 task를 1회 추가한다. 단일 crop의 판독 실패만으로 재계획하지 않는다. center 이후에도 `O?`이면 analysis가 `ABSTAIN`한다.
- analysis(코드 규칙, LLM 아님): After(v2)의 판독 가능 여부·합의·근거를 아래 R1 계약으로 조합한다. B0/B1은 RAG가 없는 비교군이므로 이 표의 `V+` 요건을 적용하지 않고, VLM이 판독하지 못했거나 출력이 무효일 때만 ABSTAIN한다.
  - 판독 가능한 관찰은 `anomaly_seen`이 실제 bool이고 text가 `판독 불가`가 아닌 경우다.
  - `O+`: 판독 가능한 `true`가 1개 이상. `O-`: `true`가 없고 `full=false` 또는 서로 다른 사분면 3개 이상이 `false`. 나머지는 `O?`다. 3개 사분면은 라벨을 보지 않고 정한 75% 공간 커버리지 경계다. center의 `false`는 정상 커버리지를 대신하지 않지만 `true`는 `O+` 증거다.
  - 이상 영역 합집합에는 `anomaly_seen=true`인 관찰의 region만 넣는다. 정상·판독 불가 관찰의 region은 양성 합의에 쓰지 않는다.
  - 사분면 crop의 source region은 호출자가 이미 알고 있으므로 양성 응답의 `regions`는 그 source region 하나와 정확히 같아야 한다. 다른 좌표·복수 좌표를 반환하면 판독 불가로 닫고, 음성 crop의 `regions`는 빈 배열로 정규화한다. 모델이 임의로 만든 좌표가 Cross-Checker와 거짓 중첩을 만들지 못하게 하는 경계다.
  - Cross-Checker는 위 여섯 항목 계약까지 검증해 `C+|C-|C?`로 구분한다. 빈 값·`null`·계약 불일치는 운영 중 `C?`, 명시적 `no_cross` ablation만 `C-skip`이다.
  - `V+`는 실제 기준서에 있는 위반 조항 1개 이상과 type_map의 결함 유형이 함께 검증된 경우다. 빈 판정은 `V0`, 형식 오류·교정 불가능한 모순은 `V?`다.

| Observer | Cross | Spec-Judge | 결과 |
|---|---|---|---|
| 임의 | 임의 | `V?` | `ABSTAIN/low`, `validation_failure` |
| `O+` | `C+` | `V+` | 양성 region이 겹치면 `REJECT/high`, 아니면 `ABSTAIN/low`, `evidence_conflict` |
| `O+` | `C-|C?` | `V+` | `ABSTAIN/low`, `evidence_conflict` |
| `O?` | `C+` | `V+` | `ABSTAIN/low`, `evidence_conflict` |
| `O-` | `C+` | `V+` | `ABSTAIN/low`, `evidence_conflict` |
| 시각 양성(`O+` 또는 `C+`) | 임의 | `V0` | `ABSTAIN/low`, `ungrounded_anomaly` |
| 시각 양성 없음 | 임의 | `V+` | `ABSTAIN/low`, `evidence_conflict` |
| `O-` | `C-` | `V0` | `PASS/high` |
| `O-` | `C?` | `V0` | `PASS/low` |
| `O?` | `C-|C?` | `V0` | `ABSTAIN/low`, `insufficient_visual_evidence` |
| `O+` / `O-` | `C-skip` | 각각 `V+` / `V0` | ablation 표식을 남기고 각각 `REJECT/high` / `PASS/high` |
| 그 밖의 `C-skip` 조합 | `C-skip` | 임의 | `ABSTAIN/low` |

  `ABSTAIN`은 결함 판정도 합격 판정도 아니다. `defect_type=null`,
  `spec_clauses=[]`, `actions=["재검사"]`이며 `abstain_reason`은
  `insufficient_visual_evidence|ungrounded_anomaly|evidence_conflict|validation_failure`
  중 하나다.
- write: 보고서 생성. 스키마 강제: `{verdict, defect_type, confidence, abstain_reason, observations[{text, source_call_id}], spec_clauses[ids], actions[], lot, summary}`. PASS는 `통과`만, REJECT는 `재작업|폐기|라인정지`, ABSTAIN은 `재검사`만 허용한다.
- HITL(interrupt): `verdict=ABSTAIN`, `verdict=REJECT and action in {폐기, 라인정지}`, 또는 `confidence=low`일 때 LangGraph `interrupt()`로 중단한다. ABSTAIN 승인은 재검사 요청을 확인하는 것이며 PASS/REJECT로 바꾸는 승인이 아니다. 4.5 참조.
- validate(코드 규칙):
  1. `observations[].text`가 Observer/Cross-Checker 실제 출력에 존재(정규화 후 부분 일치, 임계값은 dev셋으로 결정)
  2. `spec_clauses` ID가 기준서에 실제 존재하고 품목이 일치
  3. `defect_type`이 해당 품목의 유형 목록 안에 있음
  4. 스키마 통과
  - 하나라도 실패 → write 재작성(최대 2회). 상한 뒤에도 유효하지 않으면 `validation_passed=false`를 유지한 종단 실패로 기록하고, 잘못된 보고서를 강제 통과시키지 않은 채 `ABSTAIN/low`, `validation_failure`, `재검사`로 강제 HITL한다.
- finalize: 보고서 + LangSmith run id + 이벤트 로그.

### 4.2 MCP 서버 (`mcp_server_vision.py`, Base ops 파일 보존·기본 연결 대체)
| capability | 이름 | 내용 |
|---|---|---|
| Tool | `inspect_image(path, region, style, flip)` | VLM 호출. region=full/tl/tr/bl/br. 반환: 관찰 문장 리스트 + 모델명 + 토큰 |
| Resource | `spec://{item}` | 기준서 본문 |
| Resource | `guide://{defect_type}` | 조치 가이드 |
| Prompt | `inspection_prompt(item, lot)` | 검사 절차 프롬프트 |

Base의 `find_prompt / find_resource_template` 멀티 서버 탐색과 docs decoy 서버는 유지한다. VLM tool 호출은 Base의 `with_retry`로 감싼다.

### 4.3 LLM/VLM 백엔드 (의존성 주입)
`LLM_BACKEND=fake | ollama | openai` 유지. VLM도 같은 토글: fake(`/test/<label>/` 폴더 라벨을 우선 읽고 등록 파일은 바이트 SHA-256으로 역조회하는 label-leaking 결정론 백엔드, 테스트·데모용, 실평가는 fake 거부) / ollama `qwen2.5vl:3b` / openai `gpt-4o-mini`(관찰) + `gpt-4o`(교차 검증). 텍스트 노드(plan/write)는 기존 fast/quality 분리 그대로.

주 실모델 실험은 결과를 보기 전 Ollama로 고정한다. 텍스트 fast는
`qwen3:0.6b`, Spec-Judge와 quality는 각각 독립 역할의 `qwen3:8b`,
Observer와 Cross-Checker는 모두
`qwen2.5vl:3b`를 쓴다. OpenAI 결과와 동등한 모델 성능이나 상용 API로의
일반화를 주장하지 않는다.

Ollama 텍스트 호출은 판정과 보고서에 불필요한 내부 추론을 넣지 않도록
`think=false`로 고정하고, fast와 Judge는 각각
`num_predict=512`, quality는 `num_predict=2048`, 전체 스트림은 300초로
제한한다. 출력 상한에 닿아
`done_reason=length`로 끝나거나
완료 기록 없이 닫힌 응답은 부분 JSON을 채택하지 않고 오류로 기록한다.

### 4.4 이미지 입력 경로와 UI
Base `/chat`은 텍스트만 받는다. 다음을 추가한다.
- `POST /inspect` (multipart): 이미지 파일 + 필수 품목 + `thread_id`. `session_id`는 이전 클라이언트 호환용 별칭으로만 받는다. 서버가 `data/uploads/{thread_id}/{uuid}.png`에 저장하고 그래프 state에 `image_path`를 넣어 `/chat/stream`과 같은 SSE 경로로 진행 이벤트를 보낸다.
- 개발·평가용으로 `/chat`에 `image_path` 필드를 허용해 로컬 파일을 직접 가리킬 수 있게 한다(배치 평가가 이 경로를 쓴다).
- SSE 이벤트 추가: `ObservationReady`(영역별), `CrossCheckDone`, `SpecJudged`(조항 ID 포함), `ApprovalRequired`, `Validation`(통과 여부와 사유), `FinalAnswer`.
- `web_chat.html`: 이미지 업로드 버튼, 사용자가 올린 원본 이미지 썸네일, 관찰 영역 텍스트 배지, 승인/반려 버튼 두 개. 현재 UI는 크롭 이미지나 영역 오버레이를 별도로 생성하지 않는다.

### 4.5 HITL 재개 설계
LangGraph `interrupt()`는 checkpointer와 thread_id가 필요하다. Base에는 없으므로 다음을 추가한다.
- checkpointer: `langgraph.checkpoint.sqlite.AsyncSqliteSaver`를 `SupervisorDeps`에 주입. 파일은 기존 `session.db`와 분리(`checkpoints.db`). 외부 `thread_id` 앞에 실행 조건 namespace를 붙여 저장 ID를 `v2:{thread_id}`, `v1:B0:{thread_id}`, `v1:B1:{thread_id}`로 분리한다. 같은 사용자 ID를 써도 After와 두 Before 조건의 체크포인트는 섞이지 않는다.
- 서버는 `interrupt()`가 만든 LangGraph `Interrupt.id`를 `ApprovalRequired.approval_id`로 보내고 report를 함께 제공한다. `POST /chat/resume {thread_id, approval_id, decision: approve|reject, note}`는 `approval_id`를 필수로 받고, 현재 namespace의 유일한 pending interrupt ID와 정확히 일치할 때만 `Command(resume={approval_id: ...})`로 재개한다. stale·재사용·다른 조건의 ID는 409로 거부한다.
- 한 초안의 승인 결정은 그 초안에만 유효하다. 사람 반려 또는 validate 반려로 `write`가 새 초안을 만들면 approval을 초기화하고 새 interrupt ID로 다시 HITL 승인을 요구한다. reject면 원본 위험 보고서를 강제 완료하지 않고 보수적 재작성 경로로 보낸다.
- pending interrupt가 있는 외부 `thread_id`로 `/chat`, `/chat/stream`, `/inspect` 새 요청을 보내면 409로 거부해 대기 체크포인트 덮어쓰기를 막는다. 먼저 `/chat/resume`으로 해당 초안을 처리해야 한다.
- 같은 namespace·`thread_id`의 신규 요청과 resume는 single-flight로 직렬화한다. 이미 실행 중이면 409로 거부하고, SSE는 endpoint가 응답 객체를 만든 시점이 아니라 스트림의 자연 종료·예외·취소까지 예약을 유지한다. 따라서 같은 `approval_id`의 동시 재사용과 두 신규 요청의 checkpoint TOCTOU를 허용하지 않는다.
- `HITL_MODE=interactive | auto_approve | auto_reject`: 배치 평가는 `auto_approve`로 돌려 HITL 발생률만 기록한다. 데모는 interactive. `auto_reject`가 첫 초안을 반려한 뒤 생성한 새 초안은 자동 승인하지 않고 실제 사람의 새 승인에서 멈춘다.
- 서버 재시작 후 재개: SQLite checkpointer에 저장된 같은 namespace·`thread_id`·`approval_id`로 가능하다. L1 테스트에 포함.

### 4.6 RAG
- 임베딩: 주 실험은 Ollama `nomic-embed-text`(운영 기본)와 `bge-m3`(R6 비교용) 2종을 사전 고정해 비교한다. OpenAI `text-embedding-3-small`은 선택적 비교 경로로만 남기고 주 결론에 섞지 않는다. 벡터 스토어는 외부 DB 없이 작동하는 인메모리 `SpecIndex`이며, NumPy cosine 유사도로 상위 조항을 검색한다. 색인 단위는 기준서 조항 1개 = 1 청크이고, 메타데이터는 `{item, defect_type, clause_id}`다. 품목을 먼저 필터한 다음 유사도 검색한다.
- 규모에 대한 솔직한 전제: 현재 기준서 4개·조항 25개 수준에서는 전문을 프롬프트에 넣는 B1이 RAG와 비슷하거나 나을 수 있다. RAG의 가치는 품목이 수십·수백 개로 늘어 전문 삽입이 토큰 한도와 비용을 넘을 때 생긴다. 그래서 B1을 반드시 같이 측정하고, B1이 이기면 그렇게 보고한다.

### 4.7 모듈화
```
agent/
  graph.py            # 조립만
  nodes/route.py plan.py gate.py analysis.py write.py validate.py hitl.py finalize.py
  subgraphs/observer.py cross_checker.py spec_judge.py
  rag/index.py embedder.py
  vision/backends.py  # fake/ollama/openai VLM
tests/                # pytest
eval/                 # run_eval.py, dataset.json, type_map.json, evaluators.py
docs/adr/             # ADR 4건
```

### 4.8 ADR (Architecture Decision Record) — 강의자료의 틀 그대로
배경 → 결정 → 대안 비교 → 결과/트레이드오프 → 재검토 조건 형식으로 4건을 `docs/adr/`에 둔다.
1. ADR-0001 판정 기준을 RAG로 분리 — 대안: 프롬프트 전문 삽입(B1) / 파인튜닝. 재검토 조건: B1이 held-out에서 이기면 소규모 운영은 B1.
2. ADR-0002 analysis 합의를 LLM이 아닌 코드 규칙으로 — 대안: LLM 판정. 이유: 검증 가능·결정론·테스트 가능(Base gate/replan과 같은 원칙).
3. ADR-0003 HITL 위치를 write 뒤 validate 앞으로 — 대안: validate 뒤 / analysis 뒤. 이유: 사람이 보는 것은 보고서 초안이어야 하고, 승인 후 내용이 validate로 한 번 더 걸러져야 함.
4. ADR-0004 VLM을 MCP Tool로 격리 — 대안: 노드 안에서 직접 호출. 이유: 백엔드 교체·재시도·flaky 테스트가 Base와 같은 경로로 가능.

---

## 5. 신뢰성 검증 체계

### 5.1 L1 단위 테스트 (pytest, fake 백엔드, 외부 호출 없음)
- validate 원문 대조: 존재/부분일치/불일치/띄어쓰기 변형
- analysis 합의 규칙: 일치·불일치·부분 조합 전부
- 판독 경계: 전체 5개 `None`, crop 1개만 `None`, full `None`+정상 사분면 2개/3개,
  center 재계획 뒤에도 `O?`, Cross만 판독 불가, 명시적 `no_cross`
- 크롭 분할 좌표·경계, 좌우 반전 좌표 변환
- RAG 검색: 조항 쿼리 → 올바른 품목·조항 top-1 (소규모 고정 셋)
- HITL 트리거 조건, `HITL_MODE` 세 값, 서버 재시작 후 resume
- write 스키마 검증, type_map 매핑 커버리지(모든 사용 유형이 매핑됨)

### 5.2 L2 컴포넌트 평가 (실모델)
- VLM tool 단독 = Before 기준선 B0. dev셋에서 B0 프롬프트도 After와 같은 정성으로 튜닝해 강한 기준선으로 만든다(약한 Before로 차이를 부풀리지 않는다).
- RAG 검색 품질: 관찰 문장 → 정답 결함유형의 조항이 top-3 안에 드는 비율. 임베딩 모델 2종 비교.

### 5.3 L3 종단 평가 — held-out 140장
| 지표 | 정의 |
|---|---|
| 종단 balanced accuracy | 정상은 PASS, 결함은 REJECT일 때만 정답. ABSTAIN·오류·누락은 오답이며 전체 N에서 빼지 않음 |
| 보수적 정상 실패율/FPR | 정상품에서 PASS가 아닌 비율. REJECT와 ABSTAIN을 모두 실패로 포함 |
| 보수적 결함 실패율/FNR | 결함에서 REJECT가 아닌 비율. PASS와 ABSTAIN을 모두 실패로 포함 |
| coverage, abstain rate | `(PASS+REJECT)/전체 N`, `ABSTAIN/전체 N`; 전체·정상·결함별 보고 |
| selective accuracy·F1 | PASS/REJECT 자동 판정분만 계산하되 coverage와 항상 함께 표시 |
| 유형 macro-F1 | defect_type vs 폴더명 (결함 이미지 한정, type_map 적용) |
| 환각률 | 아래 독립 측정 |
| 지연 p50/p95, 토큰, 추정 USD | 실행 row를 집계. manifest는 가격표·provenance, LangSmith trace는 보조 근거 |
| 반려·replan·재시도 횟수, HITL 발생률 | 실행 row·이벤트 로그 |

여기서 `USD`는 실제 청구서가 아니라 모델 응답의 토큰 수와 고정 가격표에 따라 계산한
`estimated_usd`다. 실제 청구액과 같은 값으로 표시하지 않으며, Ollama는 API 과금 0과
계산 자원 사용량 proxy를 분리해 보고한다.

`eval/dataset.json`의 해당 split을 평가 대상의 단일 roster와 정답으로 사용한다.
각 조건 결과를 이 roster에 left join해 누락 행도 오답으로 남기며, 결과의 중복·외부 ID나
`item/label/is_defect/split` 불일치는 조용히 제외하지 않고 집계를 중단한다. 표에는
`expected N`, 실제 결과 N, 누락·오류·ABSTAIN N, 자동 판정 N을 분리해 부분 실행이
완전 실행처럼 보이지 않게 한다.

환각률은 validate의 문자열 매처 규칙으로 재지 않는다. 그 규칙으로
validate의 효과를 측정하면 After가 구조적으로 좋아지는 순환이 생기기 때문이다.
- 주 측정 — 블라인드 사람 대조: A 실행이 첫 `write`에서 만든
  `pre_validation_snapshot` 보고서와 validate 반려·재작성 후의 최종
  보고서를 동일 ID 안에서 짝지어 측정한다. canonical held-out roster에서
  seed `20261020`으로 품목×정상/결함 비율을 유지해 20 ID를 결과 가용성을
  보기 전에 한 번 추출하고,
  오류·누락이 있어도 대체 표본을 뽑지 않으며 H3 결론을 중단한다.
  40개 보고서의 pair·stage·원본 ID는 0600 매핑에만 보관하고 순서를
  섞는다. 채점자는 summary와 observation의 각 claim을
  `SUPPORTED|PARAPHRASE_SUPPORTED|HALLUCINATED|NON_FACTUAL|UNCERTAIN`으로
  표시한다. `UNCERTAIN`은 보수적으로 환각에 포함하고 stage 추측도
  받아 블라인드 실패 가능성을 같이 보고한다.
- held-out 표본 수와 seed는 각각 `20`, `20261020`으로 고정하고 생성 CLI의
  override를 거부한다. unblind 시점에도 mapping의 split·표본 수·seed를
  재검증하여 세 값이 일치할 때만 H3 결론을 허용한다. dev 채점과 임의
  파라미터 채점은 기술 통계이며 H3 지지·미입증 판정에 쓰지 않는다.
- 채점 순서와 rubric을 고정한다. 무작위로 표시된 순서대로 각 행을
  한 번씩 처리하고, 검색·정렬·다른 행과의 대조로 짝을 추정하지 않는다.
  보고서를 읽은 즉시 stage 추측을 먼저 적고, claim 채점 후에 수정하지
  않는다. 관찰 원문의 직접 근거는
  `SUPPORTED`, 의미를 유지한 변형은 `PARAPHRASE_SUPPORTED`, 근거 없는
  사실·모순·과장이 하나라도 섞인 claim은 전체 `HALLUCINATED`로 채점한다.
  순수 판정·조치 권고·형식 문구처럼 관찰 사실이 아닌 claim만
  `NON_FACTUAL`로 제외하고, 판별할 수 없으면 `UNCERTAIN`을 쓴다.
- 20건의 1건은 5%p이다. H3의 `≤5%`는 선택된 관측 표본의 point
  rate 기준일 뿐 모집단 상한 주장이 아니다. Wilson 95% 구간과 1인
  채점 한계를 항상 함께 제시한다.
- 보조 진단 — 독립 매처: validate보다 엄격한 별도 검사기(정규화 없이 8자 이상 연속 n-gram 일치). 이 값은 바꿔 쓴 문장도 실패로 잡으므로 환각률이 아니라 "원문 인용 충실도"로 이름 붙여 따로 보고한다. validate 통과분이 여기서 얼마나 떨어지는지가 validate 임계값의 느슨함을 보여준다.
- H3의 판정은 사람 대조 값으로 한다.

### 5.4 L4 신뢰도 보정
- 플래그 유효성: 오류 없는 PASS/REJECT 자동 판정 안에서
  `confidence=low`와 `high`의 오답률을 비교한다. 각 그룹이 20건
  미만이면 H4는 보류한다. ABSTAIN·실행 오류·누락·무효는 종단
  오답에는 남기되 두 플래그 그룹에 합치지 않고 별도 보고한다.
- 일관성: dev셋 20장을 3회 반복 → verdict 변동률. temperature 0 고정.
- 커버리지-정확도: low를 사람에게 넘겼을 때 자동 처리분의 정확도.

### 5.5 L5 교란·결함 주입 (dev셋, 시간 남을 때)
- 이미지 교란: 90° 회전, 밝기 ±30% → 판정 변동률.
- 기준서 조항 삭제: 해당 결함 조항을 빼고 돌리면 "기준 없음, 판정 보류"를 내는가. 지어낸 조항 ID를 인용하면 validate가 잡는가.
- 관찰 조작 주입: Observer 기록 자체에 가짜 문장이 들어가면 write가 그 기록을
  그대로 인용한 것은 validate만으로 잡을 수 없다. validate는 보고서–관찰 기록–기준서
  사이의 인용·일관성을 검사하며 픽셀 수준의 관찰 진실성을 독립 검증하지 않는다.
  Cross-Checker 일치와 MVTec 외부 라벨 실측은 최종 판정 성능을 별도 평가하지만,
  개별 관찰 문장이나 지목한 픽셀 위치의 진실성까지 직접 증명하지는 않는다.
- 품목 오인: screw 이미지에 bottle 기준서를 강제하면 gate/validate가 걸러내는가.

### 5.6 LangSmith와 통계
- Dataset 업로드는 명시적 `--push-dataset`에서만 수행한다. 지표는
  `eval/evaluators.py`의 코드로 채점하며 LLM-as-judge를 쓰지 않는다.
  trace 프로젝트는 `vision-insp-{run_id}-{condition}`으로 실행·조건을
  분리한다.
- H1: 사전 등록 ID 전체를 유지하고 누락·오류·ABSTAIN을 오답으로
  남긴 정상/결함 층화 paired permutation으로 Δbalanced accuracy를
  검정한다. paired stratified bootstrap 10,000회 95% CI를 함께 제시한다.
- H2: 결함 ID의 B0/A 유형 예측을 짝 안에서 교환해 Δmacro-F1을
  검정한다. 품목/진실 유형 층화 bootstrap 10,000회 95% CI를 쓴다.
  macro-F1와 다른 이진 정오 McNemar를 H2 검정으로 쓰지 않는다.
- 보조 분석: 전체 이진 정오에 대한 exact McNemar는 전체·정상·결함
  층으로 따로 보고하되 H1/H2의 주 검정으로 대체하지 않는다.
  순열·bootstrap 시드는 `20261020`, Monte-Carlo 순열은 20,000회로
  고정하고 가능한 교환 짝이 16개 이하면 정확 전수한다.
- H4: low/high 자동 판정 중 한 그룹이 20건 미만이면 가설 판정을
  보류하고 건수·오답률·Wilson 95% CI만 보고한다. 결과를 본 후
  dev를 합쳐 미리 정한 표본 조건을 바꾸지 않는다.
- 품목별 분해 표도 함께 제시한다.

---

## 6. 실험 설계와 예산

| 조건 | 구성 | 분리하려는 효과 | 데이터 | nominal provider 요청/장 |
|---|---|---|---|---|
| B0 | VLM 1회 (참조 이미지, 품목명, 해당 품목 결함 유형 목록 제공, dev 튜닝 프롬프트) | 단일 VLM 기준선 | held-out 140 | 1 |
| B1 | B0와 같은 단일 VLM + 기준서 전문 삽입 | 소규모 기준서의 전문 삽입 대안 | held-out 140 | 1 |
| A | 전체 에이전트 | B0 대비 전체 시스템 효과 | held-out 140 | 10 |
| A(pre) | A 실행의 첫 write 불변 snapshot | 같은 실행에서 validate 전 보고서 | held-out A 중 사전 고정 20 | 추가 호출 없음 |
| A−X | Cross-Checker 제거 + C-skip 정책 | 탐색적 묶음 ablation | dev 52 | 9 |
| A−C | 사분면 crop과 center 재계획 제거(full만) | 탐색적 묶음 ablation | dev 52 | 6 |
| A−V | validate 비활성 별도 실행 | 실행 진단 전용, H3 인과 근거로 사용하지 않음 | dev 52 | 10 |

- 위 요청 수는 재시도·재계획·보고서 재작성이 없는 정상 경로의
  사전 추정치다. 이미지 입력의 route는 코드 분기라 provider를
  호출하지 않는다. A의 10회는 plan 1 + Observer 5 +
  Cross-Checker 1 + query embedding 1 + judge 1 + write 1이다. A 계열
  조건은 샘플 외에도 조건 시작 때 기준서
  색인 setup provider 요청 1회를 실행한다. 실측 결과에서는 성공한
  provider/VLM 응답 수와 retry를 분리해 보고한다.
- R6 dev B0+A의 nominal 요청은 `52×(1+10)+setup 1=573`, R8
  held-out B0+B1+A는 `140×(1+1+10)+setup 1=1,681`회다. dev
  ablation은 각 조건을 별도 실행하므로 이 두 suite 수에 포함하지 않는다.
- 위 573회는 한 프로세스에서 A 색인을 한 번만 만드는 알고리즘
  nominal이다. 자원 제한 배치는 A 재개마다 SpecIndex를 다시
  만들어 embedding setup 1회가 추가된다. 추가 호출·토큰·지연·
  계산 proxy를 누적하고, 이 실행 overhead를 샘플별 추론 비용과
  구분해 보고한다.
- 주 실측은 Ollama 로컬 축이다. API 과금은 0이고 계산 자원·시간은
  `compute_proxy_usd`로 분리해 보고한다. 실행 모델은 fast `qwen3:0.6b`,
  Judge/quality `qwen3:8b`, Observer/Cross-Checker `qwen2.5vl:3b`, 운영 embedding
  `nomic-embed-text`로 고정한다.
- dev 정식 B0/A 실측은 모두 `--batch-size 4`로 고정 roster의
  prefix만 실행하고
  정상 배치 경계에서만 `paused`로 남긴다. `--resume`은 checkpoint
  hash, ID 순서, 누적 usage, 배치 크기, 코드·데이터·
  기준서·모델·설정이
  동일할 때만 다음 ID를 append한다. Ctrl+C·비정상 종료·usage
  미확정은 `failed`로 종료하여 재개하지 않는다. 전체 roster가
  모일 때만 정식 summary와 `complete`를 생성한다.
- OpenAI 경로는 코드에 선택지로 유지하지만 이 실험의 주 결과에
  섞지 않는다. 후속으로 실행할 경우에만 고정 가격표의 `estimated_usd`와
  suite soft 상한을 적용하고, 실제 청구액 상한이 아님을 밝힌다.
- 반복: 5.4의 일관성 측정만 3회(20장). 주 결과는 1회 실행이며 temperature 0.
- 고정 조건: 같은 이미지, 같은 참조 이미지, 같은 날 실행. `SUPERVISOR_VERSION=v1`을 B0로 재구성해 Base의 버전 토글을 Before 스위치로 쓴다.
- B0와 A는 모델 강도·호출 수·crop·교차검증·RAG·규칙 판정·validate가 함께
  달라지므로 “전체 시스템 효과”만 주장한다. B1과 A도 전문 삽입 단일 VLM 대
  전체 에이전트라는 대안 비교이지 RAG 검색만의 인과 효과가 아니다. RAG 고유
  효과를 분리하려면 향후 같은 A 파이프라인의 `A-fullspec` 대조군이 필요하다.
- A−X/A−C/A−V는 dev의 탐색적 별도 재실행이므로 효과 크기·비용 진단만 하며
  확정적 인과 주장을 하지 않는다. H3만 동일 A 실행의 A(pre)/최종 보고서를 쓴다.

---

## 7. "신뢰할 수 없다"고 판정하는 조건 (사전 등록)
- H1 미달(ΔBA paired permutation p≥0.05 또는 ΔBA≤0) → "전체 에이전트가
  이 기준선보다 정확도를 올린다는 증거 없음"으로 발표. H3·H4가 성립하면
  "정확도 아닌 신뢰성 개선"으로 재해석하되 재해석임을 명시.
- H2 미달(Δmacro-F1 paired permutation p≥0.05 또는 Δ≤0) → 유형 분류 개선
  효과 미입증으로 발표한다. 이진 McNemar로 대체 설명하지 않는다.
- 선택된 20쌍의 관측 환각 포함률 After >5%, Before 이상, 또는 paired
  exact McNemar p≥0.05 → H3 효과 미입증으로 보고한다. 모집단 5% 이하로
  일반화하지 않는다.
- FNR이 Before보다 증가 → 안전 관점 후퇴로 명시. FPR 개선으로 상쇄 주장 금지.
- 일관성 변동률 > 10% → 판정 재현성 부족 명시.
- B1 ≥ A → "이 작은 기준서 규모에서 더 복잡한 전체 에이전트의 추가 효용이
  관측되지 않음"으로 보고하고 ADR-0001 재검토 조건을 발동한다. RAG 검색
  단독 효과나 불필요성을 입증했다고 표현하지 않는다.

---

## 8. 한계와 트레이드오프
1. 지연·비용: nominal provider 요청이 B0 1회에서 A 10회로
   늘고 A 조건 setup 1회가 별도다. 실측 성공 응답·retry·지연·추정
   비용을 함께 제시한다. 완화는 크롭 병렬과 fast/quality 분리다.
2. validate의 false reject: 정당한 표현 변형도 반려 가능. 임계값은 dev셋으로 조정하되 held-out에 반영된 비용을 적는다.
3. 기준서 작성자 편향: 기준서를 내가 썼다. 라벨 미참조·held-out 분리·git 고정으로 완화하지만 완전히 제거되진 않음. 실무에서는 고객 기준서가 대체.
4. 교차 검증의 상관 오류: 같은 계열 모델은 같은 곳에서 틀릴 수 있다. 모델 분리와 이미지 반전으로 줄이지만 독립 검증이라 부르기엔 한계가 있다. ollama 축에서는 특히 약하다.
5. RAG 규모·비교군: 이 규모에서는 B1이 이길 수 있다. B1/A는 다른 요소도
   함께 달라 retrieval 단독 효과를 식별하지 못한다. 그 경우 RAG는 확장성
   투자로만 설명하고, 고유 효과는 향후 같은 파이프라인 대조군으로 확인한다.
6. MVTec의 통제 환경: 조명·배경이 균일한 벤치마크. 이 프로젝트가 증명하는 것은 "파이프라인이 설계대로 신뢰성을 올리는가"이고 현장 적용 성능이 아니다.
7. 소형 로컬 VLM 품질: 3B 모델 B0가 매우 낮으면 A의 상승폭이 부풀려
   보일 수 있다. 그러므로 Ollama 축의 동일 모델 내 B0/A 비교로만 해석하고,
   GPT 계열이나 현장 모델로 일반화하지 않는다.
8. 환각률 사람 채점이 1인·20쌍: 블라인드와 stage 추측 기록을 써도 채점자
   편향이 남고, 관측 5%는 모집단 5% 상한을 뜻하지 않는다.
9. HITL 중단과 상태 복원: checkpointer로 해결하지만 배치 평가에서는 auto_approve라 사람 판단의 효과는 측정되지 않는다.
10. 루프 상한: write↔validate 2회, replan 1회. 상한 도달 뒤에도 판독·검증이
    유효하지 않으면 `ABSTAIN/low`와 재검사로 HITL하며, 무효 보고서를 강제 통과시키지 않는다.
11. 라이선스: CC BY-NC-SA → 비상업 데모 한정. 이미지는 레포에 올리지 않는다.
12. 에이전트 레이어의 상한: 지각기가 못 본 결함은 어떤 레이어도 복구 못 한다. FNR 하한은 VLM이 정한다.

---

## 9. 등록 당시 초기 일정 (Day1 15:00 → Day3 13:00)

> 아래는 주제 등록 당시의 초기 일정이다. 현재 실행 순서와 완료 게이트는
> `_tasks/00_MASTER_PROMPT.md`의 보정 Phase R0~R10을 따른다.

| 시점 | 작업 | 산출물 |
|---|---|---|
| Day1 15–17 | Base 기동 확인. API 계약(4.4)·checkpointer 방식(4.5) 결정. MVTec 4품목 다운로드 스크립트, dev/held-out 분리, `qwen2.5vl:3b` pull | `eval/dataset.json`, `eval/type_map.json`, `sh/setup_data.sh` |
| Day1 17–19 | 기준서 4개·조치 가이드 작성(라벨 미참조) 후 git 고정. MCP vision 서버 | `data/specs`, `data/guides`, `mcp_server_vision.py` |
| Day1 19–21 | Observer 서브그래프 + fake VLM, route/plan 수정, `/chat` image_path 경로 | 이미지 입력 종단 1회 동작 |
| Day2 09–12 | Cross-Checker, Spec-Judge(RAG), analysis 합의 규칙 | research 2단계 완성 |
| Day2 13–15 | write 스키마 + validate + HITL(checkpointer, resume, HITL_MODE) | P0 완료 |
| Day2 15–17 | B0/B1 구성, `eval/run_eval.py`, 독립 환각 매처, 선택적 LangSmith Dataset 업로드, dev셋 튜닝 | 평가 파이프라인 |
| Day2 17–21 | held-out B0/B1/A 실측 → ablation(dev) → 일관성 20장×3 → 지표별 paired 검정 | 결과 표 |
| Day3 09–10 | 동일 A 실행 validate 전·후 블라인드 환각 채점 20쌍(40건), 품목별 분해 | 최종 수치 |
| Day3 10–12 | 모듈 분리, pytest, ADR 4건, README(보고서 대체), `/inspect` 업로드 UI(시간 되면) | 제출물 |
| Day3 12–13 | PPT(문제→개선→수치→한계), fake 백엔드 데모 리허설, 데모 GIF 녹화 | 발표 |
| Day3 13:00 / 14:00 | 제출 / 발표 | |

우선순위: P0 = 4.1·4.5 전체 + L1 + Ollama B0 vs A + ADR. P1 = B1, 독립 환각 측정, L4, ablation. P2 = L5 교란, `/inspect` 업로드 UI, 커버리지 곡선. 시간 부족 시 P2부터 버리고 "미수행"으로 명시한다. 업로드 UI가 밀리면 데모는 `image_path` 경로로 한다.

---

## 10. 데모와 발표 리스크
- 라이브 데모는 fake VLM 백엔드로 결정론 경로를 먼저 보여주고(외부 모델 장애와 무관하게 전 노드 동작), 이어서 Ollama 축으로 dev 이미지 1장을 돌린다.
- 데모 GIF를 Day3 오전에 녹화해 PPT에 넣는다. 네트워크가 죽어도 흐름은 보여줄 수 있다.
- HITL 승인 장면은 데모의 핵심이므로 `confidence=low`가 확실히 나오는 dev 이미지 1장을 미리 골라 둔다.

---

## 11. 범위 밖
- CV 모델 학습·파인튜닝, 픽셀 마스크 기반 위치 정확도(IoU) 평가(마스크는 보유만)
- 실시간 카메라 연동, 다중 이미지 배치 UI
- LLM-as-judge 평가(평가자 환각 배제를 위해 코드 채점과 블라인드 사람 채점만)
- 다수 채점자 간 일치도 측정

---

## 12. 발표 구조 (지시서 12쪽 순서)
1. 문제: VLM은 산업 검사에서 25%를 틀리고 틀린 이유를 지어낸다(MMAD·CVPR 근거)
2. 개선: 관찰/교차검증/판정 2단계 research, 기준서 RAG, validate, HITL (그래프 1장 + Base 패턴 유지표)
3. Before/After 수치: balanced accuracy·FNR·유형 F1·환각률(독립 측정)·확신
   플래그 유효성·지연·비용. 지표별 paired p값·95% CI와 보조 McNemar. 품목별 분해
4. 한계·추가 개선: 8장 항목 중 실측으로 드러난 것 우선. B1이 이겼다면 그것부터.
