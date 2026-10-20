# 설계서 — 근거 검증 루프를 가진 외관 불량 판정 에이전트 (VLM + Multi-Agent + RAG)

작성일 2026-10-20 (Day1). 과제 유형: 2번 신규 프로젝트 (Base AI Agent 구조 이식).

---

## 0. 한 문단 요약

범용 비전언어모델(VLM)은 학습 없이 산업 이미지를 읽을 수 있지만 산업 외관검사 벤치마크(MMAD, ICLR 2025)에서 최상위 상용 모델도 평균 정확도 74.9%에 그쳐 현장 요구 수준에 못 미친다. 이 프로젝트는 VLM을 "관찰기"로 격하하고 판정은 검사 기준서 RAG가, 환각 억제는 교차 검증과 원문 대조 validate가, 위험 조치는 사람 승인(HITL)이 맡는 에이전트 운영 레이어를 Base AI Agent 그래프 위에 구현한다. 공개 벤치마크 MVTec AD의 외부 라벨로 VLM 단독(Before) 대비 에이전트(After)의 정확도·환각률·비용 변화를 측정하고 효과가 없으면 없다고 보고한다.

---

## 1. 왜 이 프로젝트인가 (문제 정의와 필요성)

### 1.1 현장 문제
- 제조 외관검사는 전용 CV 모델(학습 필요, 품목·라인별 재학습, 라벨 수집 비용)이 담당해 왔다. 소량 다품종·신규 품목에서는 학습 데이터가 없어 투입이 늦다.
- 범용 VLM(GPT-4o, Qwen2.5-VL 등)은 학습 없이 "이 나사 목에 긁힘이 있다"고 말할 수 있다. 그러나:
  - MMAD(Jiang et al., ICLR 2025): 산업 이미지 8,366장·질문 39,672개에서 GPT-4o 74.9%, GPT-4o-mini 66.3%, 최고 오픈소스 70.8%. 저자들이 "산업 요구에 크게 못 미친다"고 명시.
  - Eyes Wide Shut(Tong et al., CVPR 2024): 기본 시각 패턴에서 GPT-4V가 틀린 답에 지어낸 설명을 붙임(환각).
  - Vision language models are blind(Rahmanzadehgervi et al., ACCV 2024): 원 겹침·선 교차 같은 초등 과제에서 GPT-4o·Gemini·Claude 평균 약 58%.
- 학습 없는 범용 지각기는 있지만 그 위에서 신뢰할 수 있는 판정을 만드는 운영 레이어는 비어 있다. 이 프로젝트는 그 공백을 겨냥한다.

### 1.2 왜 CV 모델 대신 에이전트인가
| 요구 | CV 모델 단독 | VLM 단독 | 에이전트 레이어 |
|---|---|---|---|
| 신규 품목 즉시 투입 | ✗ 재학습 | ○ | ○ |
| 판정 기준이 품목·고객마다 다름 | 재학습 | 프롬프트에 하드코딩 | **기준서 RAG** (문서만 교체) |
| 판정 근거를 사람이 읽을 수 있음 | ✗ | △ 환각 섞임 | **관찰 인용 강제 + validate** |
| 오판 가능성 자각(확신도) | 점수만 | ✗ 과신 | **교차 검증 → 확신 플래그** |
| 폐기·라인 정지 같은 위험 조치 | 사람 별도 | ✗ | **HITL 승인 노드** |
| 판정 과정 추적·감사 | 로그 | ✗ | **LangSmith 계층 trace** |

### 1.3 왜 이 과제(Mini-Project)에 맞는가
- 지시서 9쪽: 과제 2번은 "Base AI Agent의 주제 대신 다른 주제로 바꾸어 신규 AI Agent로 개발". 주제 제약 없음을 두 번 명시.
- 지시서 10쪽 개선 포인트와의 대응: ①키워드→RAG(기준서·조치 가이드 검색) ②모듈화(노드 분리) ③research 서브그래프 분리(관찰/판정/교차검증 3개) ④write→validate 사이 HITL 승인. 네 개를 한 프로젝트에서 모두 다룬다.
- 평가 기준: 동작(30) 기술요소(30) 트레이드오프(20) 발표(20) + Before/After(10). 비전은 MCP tool 하나에 갇혀 있고 점수는 전부 에이전트 설계로 받는다.

### 1.4 포트폴리오 관점
제조·자동차 부품 공장에서 비전 검사는 가장 보편적인 AI 적용 영역이다. "비전 판정을 에이전트로 보고서화·조치 연결"은 현재 실무 추세이고 특정 회사용으로 보이지 않는다. 전하려는 메시지는 "틀리는 지각기 위에서 신뢰할 수 있는 판정을 설계하고, 그 신뢰를 숫자로 증명했다."이다.

---

## 2. 가설 (측정 가능한 형태로 사전 등록)

| ID | 가설 | 지표 | 판정 기준 |
|---|---|---|---|
| H1 | 에이전트 레이어는 VLM 단독보다 결함 유무 판정 정확도가 높다 | 양불 정확도, FPR(정상품 오탐), FNR(결함 미탐) | After − Before ≥ 5%p, 95% CI 비겹침 |
| H2 | 기준서 RAG는 결함 유형 분류 정확도를 높인다 | 유형 정확도(MVTec 폴더 라벨 기준) | 동일 |
| H3 | validate는 보고서 환각(관찰에 없는 주장)을 줄인다 | 환각률 = 근거 없는 문장 수 / 전체 주장 문장 수 | After ≤ 5%, Before 대비 감소 |
| H4 | 교차 검증의 "확신 낮음" 플래그는 실제 오답을 선별한다 | 플래그된 건의 오답률 vs 전체 오답률 | 플래그 내 오답률 ≥ 2× 전체 |
| C1 | 비용: 호출 수 증가로 지연·토큰이 늘어난다 | p50/p95 지연, 토큰, USD | 실측 보고 (가설 아님, 트레이드오프) |

사전 등록은 기준을 먼저 적어두고 결과가 못 미치면 "효과 미입증"으로 그대로 발표한다는 뜻이다. 결과 보고 기준을 바꾸지 않는다.

---

## 3. 데이터

### 3.1 평가 데이터: MVTec AD (외부 라벨)
- 출처: Bergmann et al., "A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection", CVPR 2019. HuggingFace 미러 `foersben/mvtec-ad`(원본 폴더 구조 유지). 라이선스 CC BY-NC-SA 4.0 → 비상업 과제 사용 가능, README에 출처 표기.
- 구조(확인 완료): `{품목}/test/good/`, `{품목}/test/{결함유형}/`, `{품목}/ground_truth/{결함유형}/`(픽셀 마스크). 폴더명이 곧 라벨이므로 내가 라벨을 만들지 않는다.
- 사용 품목(4개, 결함이 언어로 구분 가능한 것 우선):
  - screw: good / scratch_head / scratch_neck / thread_side / thread_top / manipulated_front
  - bottle: good / broken_large / broken_small / contamination
  - metal_nut: good / bent / color / flip / scratch
  - hazelnut(예비): good / crack / cut / hole / print
- 샘플링: 품목당 good 20장 + 결함 유형별 8장 → 약 200장. 난수 시드 고정, 파일 목록을 `eval/dataset.json`에 저장.
- dev / held-out 분리: 30%(약 60장)는 프롬프트·임계값·기준서 조정에 쓰고 70%(약 140장)는 최종 보고에만 사용하며 개발 중 열어보지 않는다. 튜닝 누출을 막기 위해서다.

### 3.2 참고 문서(내가 작성, 정답이 아님)
- 검사 기준서 `data/specs/{품목}.md`: 품목별 허용/불허 기준. 조항 ID 부여(예 `SCREW-03: 목 부위 긁힘 길이 1mm 초과 시 불합격`). RAG 색인 대상. 메타데이터: 품목, 결함유형.
  - 작성 규칙(편향 차단): MVTec 논문의 결함 유형 정의와 일반 품질 기준으로만 쓴다. 특정 테스트 이미지를 보고 쓰지 않는다. held-out 이미지는 열지 않는다. 작성 완료 후 git에 고정하고 평가 중 수정하지 않는다.
- 조치 가이드 `data/guides/{결함유형}.md`: 재작업/폐기/라인 점검 등 후속 조치와 위험도(HITL 트리거 여부).
- 정상 참조 이미지: 품목당 `train/good` 1장을 VLM에 함께 제공(MMAD 기본 설정과 동일한 1-shot 비교).

### 3.3 조회용 메타데이터(합성, 판정에 미사용)
- `data/lots.json`: 이미지 → 가상 로트번호/라인/시각. 보고서 서식용이며 정확도 판정에 쓰지 않는다.

---

## 4. 아키텍처

### 4.1 그래프 (Base `graph.py` 노드 유지)
```
route ──chat──▶ chat_answer
  └──inspection──▶ plan ▶ gate ▶ research(서브그래프 3개 병렬) ▶ replan_check ▶ analysis ▶ write ▶ [HITL approve] ▶ validate ▶ finalize
                                        ▲ 부족                                                               │ 불합격
                                        └──────────────────────────────────────────────────────────────────────┘ (최대 2회)
```
- **route**: 입력에 이미지 경로/업로드가 있으면 inspection, 없으면 chat.
- **plan**: MCP Prompt `inspection_prompt(item, lot)` 조회 → tasks 생성 `[observe_full, observe_crops, spec_judge, cross_check]`.
- **gate**: 코드 규칙으로 불필요 task 제거(예: 품목 미식별 시 spec_judge 보류).
- **research — 서브그래프 3개 (compiled subgraph, 병렬)**
  1. **Observer**: MCP Tool `inspect_image(path, region="full")` 1회 + 4분할 크롭 4회 병렬. 출력에는 관찰 사실만(위치·크기·색·형태) 담는다. "불량/합격" 단어 금지(프롬프트 + 후처리 필터).
  2. **Spec-Judge**: 관찰 문장을 쿼리로 기준서 RAG 검색(top-3 조항) → 조항별 충족/위반 판정 + 결함 유형 후보.
  3. **Cross-Checker**: 같은 이미지를 다른 프롬프트 스타일(체크리스트형)과 가능하면 다른 모델로 재판독 → Observer와 결함 유무·위치 일치 여부.
- **replan_check**: Observer가 "판독 불가/해상도 부족" 반환 시 추가 크롭 task 1회.
- **analysis(코드 규칙, LLM 아님)**: 합의 규칙
  - Observer·Cross-Checker 일치 + Spec-Judge 위반 조항 있음 → `REJECT, confidence=high`
  - 둘 일치 + 위반 없음 → `PASS, confidence=high`
  - 불일치 → `confidence=low` → HITL 대상
- **write**: 보고서 생성. 스키마 강제: `{verdict, defect_type, confidence, observations[{text, source_call_id}], spec_clauses[ids], actions[], lot}`.
- **HITL(interrupt)**: `verdict=REJECT and action in {폐기, 라인정지}` 또는 `confidence=low`일 때 LangGraph `interrupt()`로 중단, 사람 승인/수정 후 재개. 체크포인트는 세션 저장소에 둔다.
- **validate(코드 규칙)**:
  1. `observations[].text`가 Observer/Cross-Checker 실제 출력에 존재(정규화 후 부분 일치, 임계값 dev셋으로 결정)
  2. `spec_clauses` ID가 기준서에 실제 존재하고 품목이 일치
  3. `defect_type`이 해당 품목의 유형 목록 안에 있음
  4. 스키마 통과
  - 하나라도 실패 → write 재작성(최대 2회), 이후 `confidence=low`로 강제 HITL.
- **finalize**: 보고서 + LangSmith run id + 이벤트 로그.

### 4.2 MCP 서버 (`mcp_server_vision.py`, Base ops 서버 자리)
| capability | 이름 | 내용 |
|---|---|---|
| Tool | `inspect_image(path, region, style)` | VLM 호출. region=full/tl/tr/bl/br. 반환: 관찰 문장 리스트 + 모델명 + 토큰 |
| Resource | `spec://{item}` | 기준서 본문 |
| Resource | `guide://{defect_type}` | 조치 가이드 |
| Prompt | `inspection_prompt(item, lot)` | 검사 절차 프롬프트 |
- Base의 `find_prompt / find_resource_template` 멀티 서버 탐색 유지. docs decoy 서버도 유지(탐색 데모).

### 4.3 LLM/VLM 백엔드 (의존성 주입)
- `LLM_BACKEND=fake | ollama | openai` 유지. VLM도 같은 토글: fake(결정론 관찰 반환, 테스트용) / ollama `qwen2.5vl:3b` / openai `gpt-4o-mini`(관찰) + `gpt-4o`(교차검증, 품질).
- 텍스트 노드(plan/write)는 기존 fast/quality 분리 그대로.

### 4.4 RAG
- 임베딩: OpenAI `text-embedding-3-small`(실측) 또는 Ollama 로컬 임베딩(무료 경로). 벡터 스토어: 인메모리/Chroma. 색인 단위: 기준서 조항 1개 = 1 청크, 메타 `{item, defect_type, clause_id}`.
- `SupervisorDeps`에 `vector_store` 주입. 서버 수명 동안 1회 색인.

### 4.5 모듈화
```
agent/
  graph.py            # 조립만
  nodes/route.py plan.py gate.py analysis.py write.py validate.py hitl.py finalize.py
  subgraphs/observer.py spec_judge.py cross_checker.py
  rag/index.py retriever.py
  vision/backends.py  # fake/ollama/openai VLM
tests/                # pytest
eval/                 # run_eval.py, dataset.json, evaluators.py
```

---

## 5. 신뢰성 검증 체계 (5개 층)

### L1 단위 테스트 (pytest, fake 백엔드, 외부 호출 없음)
- validate 원문 대조: 존재/부분일치/불일치/띄어쓰기 변형 케이스
- analysis 합의 규칙: 일치·불일치·부분 조합 전부
- 크롭 분할: 4분할 좌표·경계
- RAG 검색: 조항 쿼리 → 올바른 품목·조항 top-1 (소규모 고정 셋)
- HITL 트리거 조건: 위험 조치/확신 low 조합
- write 스키마 검증

### L2 컴포넌트 평가 (실모델)
- VLM tool 단독 정확도 = Before 기준선 B0. dev셋에서 프롬프트 고정 후 held-out 측정.
- RAG 검색 품질: 관찰 문장 → 정답 결함유형의 조항이 top-3 안에 드는 비율(라벨 기준). 임베딩 모델 2종 비교.

### L3 종단(End-to-End) 평가 — held-out 140장
| 지표 | 정의 |
|---|---|
| 양불 정확도 | verdict(PASS/REJECT) vs 폴더(good/결함) |
| FPR | 정상품을 REJECT한 비율 (과검 → 생산성 손실) |
| FNR | 결함을 PASS한 비율 (미검 → 품질 사고, 더 위험) |
| 유형 정확도 | defect_type vs 폴더명 (결함 이미지에 한정) |
| 환각률 | validate 전 보고서에서 근거 없는 주장 문장 비율 (validate 효과 측정용으로 전·후 모두 기록) |
| 지연 p50/p95, 토큰, USD | LangSmith run 메타데이터 |
| retry/반려 횟수 | validate 반려, replan 횟수 |
| HITL 발생률 | 사람 개입이 필요한 비율 (자동화율의 역) |

### L4 신뢰도 보정 (confidence가 의미 있는가)
- 플래그 유효성: `confidence=low` 건의 오답률 vs 전체 오답률 (H4).
- 일관성: 동일 이미지 3회 반복 → verdict 변동률. temperature 0 고정.
- 커버리지-정확도 곡선: low를 사람에게 넘겼을 때 자동 처리분의 정확도 (실무 운영 지표).

### L5 교란·결함 주입 테스트
- 이미지 교란: 90° 회전, 밝기 ±30% → 판정 변동률.
- 기준서 조항 삭제: 해당 결함 조항을 빼고 돌리면 "기준 없음, 판정 보류"를 내는가(abstention). 지어낸 조항 ID를 인용하면 validate가 잡는가.
- 관찰 조작 주입: Observer 출력에 가짜 문장을 섞었을 때 write가 그걸 인용하면 validate가 반려하는가(fault injection).
- 품목 오인: screw 이미지에 bottle 기준서를 강제하면 gate/validate가 걸러내는가.

### LangSmith 연결
- Dataset: held-out 이미지 경로 + 라벨. Evaluator 함수: 위 지표 전부 코드로 채점(LLM-as-judge 없음 → 평가자 환각 배제).
- 프로젝트 분리: `vision-insp-before`, `vision-insp-after`, `vision-insp-ablation-*`.

### 통계 처리
140장 기준 정확도의 95% 신뢰구간(Wilson)을 구하고 Before/After 차이가 CI 안에 들면 "유의하지 않음"으로 보고한다. 평균이 특정 품목에 가려지지 않게 품목별 분해 표도 함께 제시한다.

---

## 6. 실험 설계 (Before / After / Ablation)

| 조건 | 구성 | 분리하려는 효과 |
|---|---|---|
| B0 | VLM 1회: "불량인가, 어떤 유형인가" | 기준선 |
| B1 | VLM 1회 + 기준서 전문을 프롬프트에 통째로 삽입 | RAG 검색 자체의 가치 (B1 vs A) |
| A | 전체 에이전트 | 제안 방식 |
| A−X | A에서 Cross-Checker 제거 | 교차 검증 기여 |
| A−C | A에서 크롭 병렬 제거(full만) | 멀티 관찰 기여 |
| A−V | A에서 validate 비활성 | 환각 억제 기여 |

- 고정 조건: 같은 held-out 140장, 같은 VLM 모델, temperature 0, 같은 날 실행, 3회 반복 평균.
- 모델 2축: `qwen2.5vl:3b`(로컬·무료) / `gpt-4o-mini`(API). 비용 상한 USD 10. 초과하면 held-out을 100장으로 줄이도록 사전에 정해 둔다.
- `SUPERVISOR_VERSION=v1`을 B0로 재구성해 Base의 버전 토글을 그대로 Before 스위치로 사용.

---

## 7. "신뢰할 수 없다"고 판정하는 조건 (사전 등록)
- H1 미달(5%p 미만 또는 CI 겹침) → "에이전트 레이어가 정확도를 올린다는 증거 없음"으로 발표. 그래도 H3(환각 억제)·H4(확신 플래그)가 성립하면 "정확도 아닌 신뢰성 개선"으로 재해석하되, 재해석임을 명시.
- 환각률 After > 5% → validate 설계 실패로 보고.
- FNR이 Before보다 증가 → 안전 관점 후퇴로 명시(FPR 개선으로 상쇄 주장 금지).
- 일관성 변동률 > 10% → 판정 재현성 부족 명시.

---

## 8. 한계와 트레이드오프 (미리 적어두는 것)
1. **지연·비용**: 호출 수 1 → 7~9회(관찰 5 + 교차 1 + 텍스트 노드). 지연 수 배, 비용 수 배. 실측치로 제시. 완화: 크롭 병렬, fast/quality 분리.
2. **validate의 false reject**: 정당한 표현 변형도 반려 가능 → 임계값 트레이드오프. dev셋으로 조정하되 held-out 결과에 반영된 비용을 적는다.
3. **기준서 작성자 편향**: 기준서를 내가 썼다. 라벨 미참조·held-out 분리·git 고정으로 완화하지만 완전히 제거되진 않음. 실무에서는 고객 기준서가 대체.
4. **MVTec의 통제 환경**: 조명·배경이 균일한 벤치마크. 현장 일반화는 별도 검증 필요. 이 프로젝트가 증명하는 건 "파이프라인이 설계대로 신뢰성을 올리는가"이고 "현장 적용 성능"이 아님.
5. **소형 로컬 VLM 품질**: 3b 모델 Before가 매우 낮을 수 있음. 그러면 After 상승폭이 과장돼 보일 수 있으므로 gpt-4o-mini 축 결과를 주 결과로 삼는다.
6. **HITL 중단과 상태 복원**: interrupt 중 서버 재시작 시 체크포인트 복원 필요. SQLite 세션 저장소 재사용.
7. **무한 루프**: write↔validate 2회, replan 1회 상한. 상한 도달 시 low로 HITL.
8. **라이선스**: CC BY-NC-SA → 비상업 데모 한정. 포트폴리오 공개 시 이미지 재배포 대신 링크·출처 표기.
9. **에이전트 레이어의 상한**: 지각기가 못 본 결함은 어떤 레이어도 복구 못 함. FNR 하한은 VLM이 결정.

---

## 9. 일정 (Day1 15:00 → Day3 13:00)

| 시점 | 작업 | 산출물 |
|---|---|---|
| Day1 15–17 | Base 기동 확인, MVTec 4품목 샘플 다운로드 스크립트, dev/held-out 분리, `qwen2.5vl:3b` pull | `eval/dataset.json`, `sh/setup_data.sh` |
| Day1 17–19 | 기준서 4개·조치 가이드 작성(라벨 미참조), MCP vision 서버(Tool/Resource/Prompt) | `data/specs`, `data/guides`, `mcp_server_vision.py` |
| Day1 19–21 | Observer 서브그래프 + fake VLM, route/plan 수정 | 이미지 입력 종단 1회 동작 |
| Day2 09–12 | Spec-Judge(RAG) + Cross-Checker 서브그래프, analysis 합의 규칙 | 3 서브그래프 병렬 |
| Day2 13–15 | write 스키마 + validate 원문 대조 + HITL interrupt | P0 완료 |
| Day2 15–18 | B0/B1 구성, `eval/run_eval.py`, LangSmith Dataset·evaluator, dev셋 튜닝 | 평가 파이프라인 |
| Day2 18–21 | held-out 실측(B0/B1/A/ablation), 통계 | 결과 표 |
| Day3 09–11 | 모듈 분리, pytest, README(보고서 대체) | 제출물 |
| Day3 11–13 | PPT(문제→개선→수치→한계), 리허설 | 발표 |
| Day3 13:00 / 14:00 | 제출 / 발표 | |

우선순위: P0 = 4.1 전체 + L1 + B0 vs A. P1 = RAG 비교(B1), L4, ablation. P2 = L5 교란, qwen 축, 커버리지 곡선. 시간 부족 시 P2부터 버리고 "미수행"으로 명시.

---

## 10. 범위 밖
- CV 모델 학습·파인튜닝, 픽셀 마스크 기반 위치 정확도(IoU) 평가(마스크는 보유만)
- 실시간 카메라 연동, 다중 이미지 배치 처리 UI
- LLM-as-judge 평가(평가자 환각 배제를 위해 코드 채점만)

---

## 11. 발표 구조 (지시서 12쪽 순서)
1. 문제: VLM은 산업 검사에서 25%를 틀리고 틀린 이유를 지어낸다(MMAD·CVPR 근거)
2. 개선: 관찰/판정/검증 분리 — 3 서브에이전트, 기준서 RAG, validate, HITL (그래프 1장)
3. Before/After 수치: 정확도·FNR·환각률·확신 플래그 유효성·지연·비용 (표 1장 + 품목별 분해)
4. 한계·추가 개선: 8장 항목 중 실측으로 드러난 것 우선
