# 설계서 — 이미지 기반 제품 결함 판정 및 근거 검증 멀티에이전트

작성일 2026-10-20 (Day1, v2.1). 과제 유형: 2번 신규 프로젝트 (Base AI Agent 구조 이식).
v1 대비 변경: research를 2단계 구조로 정정, 환각률 독립 측정, McNemar 검정, 실험 예산 축소, 이미지 입력 경로·HITL 재개 설계 추가, ADR·Base 패턴 유지표 추가.

---

## 0. 한 문단 요약

범용 비전언어모델(VLM)은 학습 없이 산업 이미지를 읽을 수 있지만 산업 외관검사 벤치마크(MMAD, ICLR 2025)에서 최상위 상용 모델도 평균 정확도 74.9%에 그쳐 현장 요구 수준에 못 미친다. 이 프로젝트는 VLM을 "관찰기"로 격하하고, 판정은 검사 기준서 RAG가, 환각 억제는 독립 교차 검증과 원문 대조 validate가, 위험 조치는 사람 승인(HITL)이 맡는 에이전트 운영 레이어를 Base AI Agent 그래프 위에 구현한다. 공개 벤치마크 MVTec AD의 외부 라벨로 VLM 단독(Before) 대비 에이전트(After)의 정확도·환각률·비용 변화를 같은 이미지에 대한 짝지은 비교로 측정하고, 효과가 없으면 없다고 보고한다.

---

## 1. 왜 이 프로젝트인가

### 1.1 현장 문제
제조 외관검사는 전용 CV 모델이 담당해 왔다. 품목·라인별로 라벨을 모으고 재학습해야 하므로 소량 다품종이나 신규 품목에서는 투입이 늦다. 범용 VLM(GPT-4o, Qwen2.5-VL 등)은 학습 없이 "이 나사 목에 긁힘이 있다"고 말할 수 있지만 신뢰도가 부족하다.

- MMAD(Jiang et al., ICLR 2025): 산업 이미지 8,366장·질문 39,672개에서 GPT-4o 74.9%, GPT-4o-mini 66.3%, 최고 오픈소스 70.8%. 저자들이 "산업 요구에 크게 못 미친다"고 명시.
- Eyes Wide Shut(Tong et al., CVPR 2024): 기본 시각 패턴에서 GPT-4V가 틀린 답에 지어낸 설명을 붙인다.
- Vision language models are blind(Rahmanzadehgervi et al., ACCV 2024): 원 겹침·선 교차 같은 초등 과제에서 GPT-4o·Gemini·Claude 평균 약 58%.

학습 없는 범용 지각기는 있지만 그 위에서 신뢰할 수 있는 판정을 만드는 운영 레이어는 비어 있다. 이 프로젝트는 그 공백을 겨냥한다.

### 1.2 왜 CV 모델 대신 에이전트인가
| 요구 | CV 모델 단독 | VLM 단독 | 에이전트 레이어 |
|---|---|---|---|
| 신규 품목 즉시 투입 | ✗ 재학습 | ○ | ○ |
| 판정 기준이 품목·고객마다 다름 | 재학습 | 프롬프트에 하드코딩 | 기준서 RAG (문서만 교체) |
| 판정 근거를 사람이 읽을 수 있음 | ✗ | △ 환각 섞임 | 관찰 인용 강제 + validate |
| 오판 가능성 자각(확신도) | 점수만 | ✗ 과신 | 독립 교차 검증 → 확신 플래그 |
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
| MCP 멀티 서버 + `find_prompt/find_resource_template` 탐색 | ops 서버 자리를 vision 서버로 교체, docs decoy 유지 | 교체 |
| MCP 의도적 flaky + `with_retry` | 유지 (VLM tool 호출도 with_retry로 감쌈) | 유지 |
| SSE 이벤트 스트리밍 | 유지 + 이벤트 타입 추가 (4.4) | 확장 |
| route → chat / 무거운 파이프라인 분기 | 유지 (이미지 유무로 분기) | 유지 |
| gate 코드 규칙 | 유지 | 유지 |
| replan_check 1회 상한 | 유지 (추가 크롭 task) | 유지 |
| validate → write 반려 루프 | 유지 + 원문 대조 규칙 교체 | 강화 |
| fast/quality 비용 분리 | 유지 (관찰 mini / 교차 검증 상위 모델) | 유지 |
| SQLite 세션 저장소 | 유지 + LangGraph checkpointer 추가 (HITL용) | 확장 |
| Vue 채팅 UI + 프록시 | 유지 + 이미지 업로드·승인 버튼 추가 | 확장 |

---

## 2. 가설 (측정 가능한 형태로 사전 등록)

같은 held-out 이미지를 Before와 After에 모두 넣는 짝지은 설계이므로 검정은 McNemar(양측, α=0.05)로 한다. 신뢰구간은 효과 크기 서술용으로 함께 적는다.

| ID | 가설 | 지표 | 판정 기준 |
|---|---|---|---|
| H1 | 에이전트 레이어는 VLM 단독보다 결함 유무 판정이 정확하다 | balanced accuracy, FNR(결함 미검), FPR(정상 오탐) | McNemar p<0.05 이고 balanced accuracy 상승 |
| H2 | 기준서 기반 판정은 결함 유형 분류가 정확하다 | 유형 macro-F1 (MVTec 폴더 라벨 기준) | B0 대비 상승, McNemar p<0.05 |
| H3 | validate는 보고서 환각을 줄인다 | 환각률 (5.3의 독립 측정) | After ≤ 5%, Before 대비 감소 |
| H4 | 확신 낮음 플래그는 실제 오답을 선별한다 | 플래그 건 오답률 vs 전체 오답률 | 플래그 내 오답률 ≥ 2× 전체 |
| C1 | 비용: 호출 수 증가로 지연·토큰이 늘어난다 | p50/p95 지연, 토큰, USD | 실측 보고 (트레이드오프) |

사전 등록은 기준을 먼저 적어두고 결과가 못 미치면 "효과 미입증"으로 그대로 발표한다는 뜻이다. 결과를 본 뒤 기준을 바꾸지 않는다.

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
- 샘플링: 품목당 good 20장 + 결함 유형별 8장 → 4품목 192장 (screw 52, bottle 44, metal_nut 44, hazelnut 52). hazelnut을 빼면 140장. 정상:결함 비율이 1:2라 정확도 대신 balanced accuracy와 F1을 주 지표로 쓴다. 난수 시드 고정, 파일 목록을 `eval/dataset.json`에 저장.
- dev / held-out 분리: 품목·유형별 층화로 약 30%(60장)는 프롬프트·임계값·기준서 조정에 쓰고 나머지 약 130장은 최종 보고에만 쓰며 개발 중 열어보지 않는다. 튜닝 누출을 막기 위해서다. 아래에서 held-out을 120장으로 적은 곳은 이 수치의 근사다.

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
```
route ──chat──▶ chat_answer
  └──inspection──▶ plan ▶ gate ▶ research ▶ replan_check ▶ analysis ▶ write ▶ [HITL approve] ▶ validate ▶ finalize
                                   │                                                              │ 불합격
                                   │  1단계 (병렬): Observer ∥ Cross-Checker                       └──▶ write (최대 2회)
                                   │  2단계       : Spec-Judge (Observer 관찰문을 입력으로)
                                   ▲ 부족 시 추가 크롭 (1회)
```
research는 2단계다. Spec-Judge가 Observer의 관찰 문장을 쿼리로 쓰기 때문에 Observer와 동시에 돌 수 없다. 병렬인 것은 Observer와 Cross-Checker, 그리고 Observer 내부의 크롭 5회 호출이다.

- route: 요청에 이미지가 있으면 inspection, 없으면 chat.
- plan: MCP Prompt `inspection_prompt(item, lot)` 조회 → tasks `[observe, cross_check, spec_judge]`.
- gate: 코드 규칙으로 불필요 task 제거. 품목(item)은 필수 입력이다(UI 드롭다운, 배치는 경로에서). VLM에게 품목까지 맞히게 하면 실패 지점이 하나 더 생기므로 범위에서 뺀다. 품목이 없으면 gate가 사용자에게 되묻고 종료.
- research — compiled subgraph 3개
  1. Observer (1단계): MCP Tool `inspect_image(path, region="full")` 1회 + 4분할 크롭 4회 병렬. 출력에는 관찰 사실만(위치·크기·색·형태) 담는다. "불량/합격" 단어 금지(프롬프트 + 후처리 필터). 자유 문장과 함께 구조화 필드 `{anomaly_seen: bool, regions: [tl|tr|bl|br|center], features: [...]}`를 반드시 채운다.
  2. Cross-Checker (1단계, Observer와 병렬): 같은 이미지를 좌우 반전해 체크리스트형 프롬프트로 재판독. openai 백엔드에서는 Observer가 `gpt-4o-mini`, Cross-Checker가 `gpt-4o`로 모델도 다르게 한다. ollama 백엔드는 모델이 하나라 반전+프롬프트 차이만 두며, 이 경우 독립성이 약하다는 점을 결과에 적는다. 출력은 같은 구조화 필드 `{anomaly_seen, regions}`(반전 좌표는 원좌표로 되돌려 기록).
  3. Spec-Judge (2단계): Observer 관찰 문장을 쿼리로 기준서 RAG top-3 조항 검색 → 조항별 충족/위반 판정 + 결함 유형 후보.
- replan_check: Observer가 "판독 불가/해상도 부족"을 반환하면 추가 크롭 task 1회.
- analysis(코드 규칙, LLM 아님): 합의 규칙. "일치"는 `anomaly_seen`이 같고, 둘 다 true일 때 `regions` 교집합이 비어 있지 않은 경우로 정의한다(자유 문장은 비교하지 않는다).
  - Observer·Cross-Checker 일치 + Spec-Judge 위반 조항 있음 → `REJECT, confidence=high`
  - 둘 일치 + 위반 없음 → `PASS, confidence=high`
  - 불일치 → `confidence=low` → HITL 대상
- write: 보고서 생성. 스키마 강제: `{verdict, defect_type, confidence, observations[{text, source_call_id}], spec_clauses[ids], actions[], lot}`.
- HITL(interrupt): `verdict=REJECT and action in {폐기, 라인정지}` 또는 `confidence=low`일 때 LangGraph `interrupt()`로 중단. 4.5 참조.
- validate(코드 규칙):
  1. `observations[].text`가 Observer/Cross-Checker 실제 출력에 존재(정규화 후 부분 일치, 임계값은 dev셋으로 결정)
  2. `spec_clauses` ID가 기준서에 실제 존재하고 품목이 일치
  3. `defect_type`이 해당 품목의 유형 목록 안에 있음
  4. 스키마 통과
  - 하나라도 실패 → write 재작성(최대 2회), 이후 `confidence=low`로 강제 HITL.
- finalize: 보고서 + LangSmith run id + 이벤트 로그.

### 4.2 MCP 서버 (`mcp_server_vision.py`, Base ops 서버 자리)
| capability | 이름 | 내용 |
|---|---|---|
| Tool | `inspect_image(path, region, style, flip)` | VLM 호출. region=full/tl/tr/bl/br. 반환: 관찰 문장 리스트 + 모델명 + 토큰 |
| Resource | `spec://{item}` | 기준서 본문 |
| Resource | `guide://{defect_type}` | 조치 가이드 |
| Prompt | `inspection_prompt(item, lot)` | 검사 절차 프롬프트 |

Base의 `find_prompt / find_resource_template` 멀티 서버 탐색과 docs decoy 서버는 유지한다. VLM tool 호출은 Base의 `with_retry`로 감싼다.

### 4.3 LLM/VLM 백엔드 (의존성 주입)
`LLM_BACKEND=fake | ollama | openai` 유지. VLM도 같은 토글: fake(이미지 경로 해시로 결정론 관찰 반환, 테스트·데모용) / ollama `qwen2.5vl:3b` / openai `gpt-4o-mini`(관찰) + `gpt-4o`(교차 검증). 텍스트 노드(plan/write)는 기존 fast/quality 분리 그대로.

### 4.4 이미지 입력 경로와 UI
Base `/chat`은 텍스트만 받는다. 다음을 추가한다.
- `POST /inspect` (multipart): 이미지 파일 + 품목(optional) + session_id. 서버가 `data/uploads/{session_id}/{uuid}.png`에 저장하고 그래프 state에 `image_path`를 넣어 `/chat/stream`과 같은 SSE 경로로 진행 이벤트를 보낸다.
- 개발·평가용으로 `/chat`에 `image_path` 필드를 허용해 로컬 파일을 직접 가리킬 수 있게 한다(배치 평가가 이 경로를 쓴다).
- SSE 이벤트 추가: `ObservationReady`(크롭별), `CrossCheckDone`, `SpecJudged`(조항 ID 포함), `ApprovalRequired`, `ValidateRejected`(사유), `FinalReport`.
- `web_chat.html`: 이미지 업로드 버튼, 관찰 결과 썸네일(크롭 영역 표시), 승인/반려 버튼 두 개.

### 4.5 HITL 재개 설계
LangGraph `interrupt()`는 checkpointer와 thread_id가 필요하다. Base에는 없으므로 다음을 추가한다.
- checkpointer: `langgraph.checkpoint.sqlite.SqliteSaver`를 `SupervisorDeps`에 주입. 파일은 기존 `session.db`와 분리(`checkpoints.db`). thread_id = session_id.
- `POST /chat/resume {session_id, decision: approve|reject, note}`: `Command(resume=...)`로 그래프 재개. reject면 write 노드로 돌아가 actions를 보수적으로 재작성.
- `HITL_MODE=interactive | auto_approve | auto_reject`: 배치 평가는 `auto_approve`로 돌려 HITL 발생률만 기록한다. 데모는 interactive.
- 서버 재시작 후 재개: checkpointer가 SQLite라 가능. L1 테스트에 포함.

### 4.6 RAG
- 임베딩: OpenAI `text-embedding-3-small`(실측) 또는 Ollama 로컬 임베딩(무료 경로). 벡터 스토어 인메모리(Chroma). 색인 단위: 기준서 조항 1개 = 1 청크, 메타 `{item, defect_type, clause_id}`. 품목 메타 필터 후 유사도 검색.
- 규모에 대한 솔직한 전제: 기준서 4개·조항 30~40개 수준에서는 전문을 프롬프트에 넣는 B1이 RAG와 비슷하거나 나을 수 있다. RAG의 가치는 품목이 수십·수백 개로 늘어 전문 삽입이 토큰 한도와 비용을 넘을 때 생긴다. 그래서 B1을 반드시 같이 측정하고, B1이 이기면 그렇게 보고한다.

### 4.7 모듈화
```
agent/
  graph.py            # 조립만
  nodes/route.py plan.py gate.py analysis.py write.py validate.py hitl.py finalize.py
  subgraphs/observer.py cross_checker.py spec_judge.py
  rag/index.py retriever.py
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
- 크롭 분할 좌표·경계, 좌우 반전 좌표 변환
- RAG 검색: 조항 쿼리 → 올바른 품목·조항 top-1 (소규모 고정 셋)
- HITL 트리거 조건, `HITL_MODE` 세 값, 서버 재시작 후 resume
- write 스키마 검증, type_map 매핑 커버리지(모든 사용 유형이 매핑됨)

### 5.2 L2 컴포넌트 평가 (실모델)
- VLM tool 단독 = Before 기준선 B0. dev셋에서 B0 프롬프트도 After와 같은 정성으로 튜닝해 강한 기준선으로 만든다(약한 Before로 차이를 부풀리지 않는다).
- RAG 검색 품질: 관찰 문장 → 정답 결함유형의 조항이 top-3 안에 드는 비율. 임베딩 모델 2종 비교.

### 5.3 L3 종단 평가 — held-out 120장
| 지표 | 정의 |
|---|---|
| balanced accuracy, F1 | verdict(PASS/REJECT) vs 폴더(good/결함) |
| FPR | 정상품을 REJECT한 비율 (과검 → 생산성 손실) |
| FNR | 결함을 PASS한 비율 (미검 → 품질 사고, 더 위험) |
| 유형 macro-F1 | defect_type vs 폴더명 (결함 이미지 한정, type_map 적용) |
| 환각률 | 아래 독립 측정 |
| 지연 p50/p95, 토큰, USD | LangSmith run 메타데이터 |
| 반려·replan 횟수, HITL 발생률 | 이벤트 로그 |

환각률은 validate와 독립적으로 잰다. validate가 정의한 기준으로 validate의 효과를 재면 After가 구조적으로 0에 수렴하는 순환이 생기기 때문이다.
- 주 측정 — 블라인드 사람 대조: held-out 보고서 중 Before 15건·After 15건을 무작위 추출해 조건 표시를 가리고 관찰 원문과 대조, "관찰에 없는 사실을 단정한 문장"을 센다. 표현을 바꿔 쓴 것은 환각으로 세지 않는다. 1인 채점이라는 한계는 결과에 적는다.
- 보조 진단 — 독립 매처: validate보다 엄격한 별도 검사기(정규화 없이 8자 이상 연속 n-gram 일치). 이 값은 바꿔 쓴 문장도 실패로 잡으므로 환각률이 아니라 "원문 인용 충실도"로 이름 붙여 따로 보고한다. validate 통과분이 여기서 얼마나 떨어지는지가 validate 임계값의 느슨함을 보여준다.
- H3의 판정은 사람 대조 값으로 한다.

### 5.4 L4 신뢰도 보정
- 플래그 유효성: `confidence=low` 건의 오답률 vs 전체 오답률 (H4).
- 일관성: dev셋 20장을 3회 반복 → verdict 변동률. temperature 0 고정.
- 커버리지-정확도: low를 사람에게 넘겼을 때 자동 처리분의 정확도.

### 5.5 L5 교란·결함 주입 (dev셋, 시간 남을 때)
- 이미지 교란: 90° 회전, 밝기 ±30% → 판정 변동률.
- 기준서 조항 삭제: 해당 결함 조항을 빼고 돌리면 "기준 없음, 판정 보류"를 내는가. 지어낸 조항 ID를 인용하면 validate가 잡는가.
- 관찰 조작 주입: Observer 출력에 가짜 문장을 섞었을 때 write가 그걸 인용하면 validate가 반려하는가.
- 품목 오인: screw 이미지에 bottle 기준서를 강제하면 gate/validate가 걸러내는가.

### 5.6 LangSmith와 통계
- Dataset: held-out 이미지 경로 + 라벨. Evaluator 함수는 전부 코드 채점(LLM-as-judge 없음). 프로젝트 분리: `vision-insp-before`, `vision-insp-after`, `vision-insp-ablation-*`.
- 검정: 같은 이미지에 대한 Before/After 정오 짝 → McNemar. 전체와 정상/결함 층별로 각각 수행한다(balanced accuracy는 두 층의 평균이므로).
- H4의 검정력 한계: `confidence=low`가 held-out의 5% 수준이면 6건 안팎이라 비율 비교가 불안정하다. 그 경우 dev셋 결과를 합쳐 서술 통계로만 제시하고 가설 판정은 보류한다. 120장에서 정확도 75% 근처의 단순 95% 신뢰구간은 약 ±8%p라 독립 표본 비교로는 검정력이 없고, 짝지은 검정이 맞는 방법이다. 신뢰구간은 효과 크기 서술용으로만 쓴다.
- 품목별 분해 표도 함께 제시한다.

---

## 6. 실험 설계와 예산

| 조건 | 구성 | 분리하려는 효과 | 데이터 | 호출/장 |
|---|---|---|---|---|
| B0 | VLM 1회 (참조 이미지, 품목명, 해당 품목 결함 유형 목록 제공, dev 튜닝 프롬프트) | 기준선 | held-out 120 | 1 |
| B1 | VLM 1회 + 기준서 전문 삽입 | RAG 검색 자체의 가치 | held-out 120 | 1 |
| A | 전체 에이전트 | 제안 방식 | held-out 120 | 약 8 |
| A−X | Cross-Checker 제거 | 교차 검증 기여 | dev 60 | 약 7 |
| A−C | 크롭 병렬 제거(full만) | 멀티 관찰 기여 | dev 60 | 약 4 |
| A−V | validate 비활성 | 환각 억제 기여 | dev 60 | 약 8 |

- 주 모델 축: `gpt-4o-mini`(관찰) + `gpt-4o`(교차). 호출 수 합계 약 120×10 + 60×19 ≈ 2,350회. 이미지 1장 호출당 약 0.003~0.01 USD로 잡아 상한 15 USD. 초과 시 ablation을 dev 30장으로 줄이도록 사전에 정해 둔다.
- 로컬 축 `qwen2.5vl:3b`: dev 60장에서 B0와 A만. 비용은 없지만 시간이 들어 P2.
- 반복: 5.4의 일관성 측정만 3회(20장). 주 결과는 1회 실행이며 temperature 0.
- 고정 조건: 같은 이미지, 같은 참조 이미지, 같은 날 실행. `SUPERVISOR_VERSION=v1`을 B0로 재구성해 Base의 버전 토글을 Before 스위치로 쓴다.

---

## 7. "신뢰할 수 없다"고 판정하는 조건 (사전 등록)
- H1 미달(McNemar p≥0.05 또는 balanced accuracy 하락) → "에이전트 레이어가 정확도를 올린다는 증거 없음"으로 발표. H3·H4가 성립하면 "정확도 아닌 신뢰성 개선"으로 재해석하되 재해석임을 명시.
- 환각률(보수적인 쪽) After > 5% → validate 설계 실패로 보고.
- FNR이 Before보다 증가 → 안전 관점 후퇴로 명시. FPR 개선으로 상쇄 주장 금지.
- 일관성 변동률 > 10% → 판정 재현성 부족 명시.
- B1 ≥ A → "이 규모에서 RAG는 불필요"로 보고하고 ADR-0001 재검토 조건 발동.

---

## 8. 한계와 트레이드오프
1. 지연·비용: 호출 수 1 → 약 8회. 지연·비용 수 배. 실측치로 제시. 완화는 크롭 병렬과 fast/quality 분리.
2. validate의 false reject: 정당한 표현 변형도 반려 가능. 임계값은 dev셋으로 조정하되 held-out에 반영된 비용을 적는다.
3. 기준서 작성자 편향: 기준서를 내가 썼다. 라벨 미참조·held-out 분리·git 고정으로 완화하지만 완전히 제거되진 않음. 실무에서는 고객 기준서가 대체.
4. 교차 검증의 상관 오류: 같은 계열 모델은 같은 곳에서 틀릴 수 있다. 모델 분리와 이미지 반전으로 줄이지만 독립 검증이라 부르기엔 한계가 있다. ollama 축에서는 특히 약하다.
5. RAG 규모: 4.6에 적은 대로 이 규모에서는 B1이 이길 수 있다. 그 경우 RAG는 확장성 투자로만 정당화된다.
6. MVTec의 통제 환경: 조명·배경이 균일한 벤치마크. 이 프로젝트가 증명하는 것은 "파이프라인이 설계대로 신뢰성을 올리는가"이고 현장 적용 성능이 아니다.
7. 소형 로컬 VLM 품질: 3b 모델 Before가 매우 낮을 수 있음. After 상승폭이 과장돼 보일 수 있으므로 gpt-4o-mini 축 결과를 주 결과로 삼는다.
8. 환각률 사람 채점이 1인: 블라인드로 하지만 채점자 편향이 남는다.
9. HITL 중단과 상태 복원: checkpointer로 해결하지만 배치 평가에서는 auto_approve라 사람 판단의 효과는 측정되지 않는다.
10. 루프 상한: write↔validate 2회, replan 1회. 상한 도달 시 low로 HITL.
11. 라이선스: CC BY-NC-SA → 비상업 데모 한정. 이미지는 레포에 올리지 않는다.
12. 에이전트 레이어의 상한: 지각기가 못 본 결함은 어떤 레이어도 복구 못 한다. FNR 하한은 VLM이 정한다.

---

## 9. 일정 (Day1 15:00 → Day3 13:00)

| 시점 | 작업 | 산출물 |
|---|---|---|
| Day1 15–17 | Base 기동 확인. API 계약(4.4)·checkpointer 방식(4.5) 결정. MVTec 4품목 다운로드 스크립트, dev/held-out 분리, `qwen2.5vl:3b` pull | `eval/dataset.json`, `eval/type_map.json`, `sh/setup_data.sh` |
| Day1 17–19 | 기준서 4개·조치 가이드 작성(라벨 미참조) 후 git 고정. MCP vision 서버 | `data/specs`, `data/guides`, `mcp_server_vision.py` |
| Day1 19–21 | Observer 서브그래프 + fake VLM, route/plan 수정, `/chat` image_path 경로 | 이미지 입력 종단 1회 동작 |
| Day2 09–12 | Cross-Checker, Spec-Judge(RAG), analysis 합의 규칙 | research 2단계 완성 |
| Day2 13–15 | write 스키마 + validate + HITL(checkpointer, resume, HITL_MODE) | P0 완료 |
| Day2 15–17 | B0/B1 구성, `eval/run_eval.py`, 독립 환각 매처, LangSmith Dataset, dev셋 튜닝 | 평가 파이프라인 |
| Day2 17–21 | held-out B0/B1/A 실측 → ablation(dev) → 일관성 20장×3 → McNemar | 결과 표 |
| Day3 09–10 | 블라인드 환각 채점 30건, 품목별 분해 | 최종 수치 |
| Day3 10–12 | 모듈 분리, pytest, ADR 4건, README(보고서 대체), `/inspect` 업로드 UI(시간 되면) | 제출물 |
| Day3 12–13 | PPT(문제→개선→수치→한계), fake 백엔드 데모 리허설, 데모 GIF 녹화 | 발표 |
| Day3 13:00 / 14:00 | 제출 / 발표 | |

우선순위: P0 = 4.1·4.5 전체 + L1 + B0 vs A + ADR. P1 = B1, 독립 환각 측정, L4, ablation. P2 = L5 교란, qwen 축, `/inspect` 업로드 UI, 커버리지 곡선. 시간 부족 시 P2부터 버리고 "미수행"으로 명시한다. 업로드 UI가 밀리면 데모는 `image_path` 경로로 한다.

---

## 10. 데모와 발표 리스크
- 라이브 데모는 fake VLM 백엔드로 결정론 경로를 먼저 보여주고(API 장애와 무관하게 전 노드 동작), 이어서 openai 축으로 실제 이미지 1장을 돌린다.
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
3. Before/After 수치: balanced accuracy·FNR·유형 F1·환각률(독립 측정)·확신 플래그 유효성·지연·비용. McNemar p값. 품목별 분해
4. 한계·추가 개선: 8장 항목 중 실측으로 드러난 것 우선. B1이 이겼다면 그것부터.
