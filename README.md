# Grounded Visual Inspector

> 이미지 기반 제품 결함 판정 및 근거 검증 멀티에이전트

| 항목 | 내용 |
|---|---|
| 작성자 | **광주 4반 · G123 · 신민서** |
| 프로젝트 주제 | VLM 관찰을 검사 기준서 RAG·교차검증·규칙 판정·HITL로 검증하는 제품 외관검사 서비스 |
| 대상 품목 | MVTec AD의 `screw`, `bottle`, `metal_nut`, `hazelnut` |
| 저장소 | [siamin20/grounded-visual-inspector](https://github.com/siamin20/grounded-visual-inspector) |
| 현재 상태 | R6.5 dev B0/A 스냅샷 공개(성능 개선 미입증), R6.6 재측정 전 성능 NO-GO·held-out 미실행 |

## 30초 요약

범용 VLM에 합격·불합격을 한 번에 맡기지 않고 역할을 분리했다.

```text
이미지 업로드
  ├─ Observer: 전체·분할 이미지에서 보이는 사실 관찰 ───────┐
  └─ Cross-Checker: 반전 이미지와 6항목 체크리스트 재확인 ──┤ 병렬
                                                            ▼
                      Spec-Judge: 검사 기준서 RAG로 위반 근거 확인
                                                            ▼
                      코드 결정표 판정 → 보고서 작성 → HITL → 인용 검증
```

핵심 목표는 VLM 출력이 그럴듯하다는 이유만으로 판정을 확정하지 않고,
“무엇을 관찰했고, 어떤 기준서 조항을 위반했으며, 왜 사람이 확인해야 하는가”를
추적 가능한 상태로 남기는 것이다.

## 1. 진행 사항 소개

### 과제 1 — Base Agent에서 추가·수정한 Task

기존 FastAPI + LangGraph + MCP + Vue 기반 장애 진단 Agent의 구조를 보존하면서
제품 외관검사 도메인으로 이식했다.

| 구분 | Base Agent | 추가·수정 내용 |
|---|---|---|
| 도메인 | 운영 장애 진단 | 이미지 기반 제품 결함 판정 |
| MCP | ops Tool·runbook | vision Tool, 검사 기준서·조치 가이드 Resource, 검사 Prompt 추가 |
| Graph | 단일 research 흐름 | Observer·Cross-Checker 병렬 실행 후 Spec-Judge가 기준서 대조 |
| 상태 | 텍스트 질의 중심 | 이미지·품목·관찰·교차검증·조항·판정·승인 상태 추가 |
| 판정 | LLM 분석 | `gate / analysis / validate`를 결정론적 코드 규칙으로 구현 |
| HITL | 없음 | 위험 조치·저확신·보류 결과 interrupt, 승인·반려 후 resume |
| UI | 텍스트 채팅 | 품목·이미지 업로드, SSE 진행 이벤트, 승인·반려, 최종 보고서 |
| 평가 | 일반 질의 평가 | MVTec AD 고정 roster, B0/A 비교, 오류 포함 지표, 실행 provenance |

Base의 `AgentState`, 노드 시그니처, `SupervisorDeps` 주입, `@traceable`, SSE 포맷과
이벤트 누적 방식은 유지했다. 기존 ops 서버 파일도 삭제하지 않았다.

### 과제 2 — 선정 주제와 주요 Task

주제는 **“이미지 기반 제품 결함 판정 및 근거 검증 멀티에이전트”**다.

주요 Task는 다음과 같다.

1. 정상 참조와 검사 이미지를 전체·사분면·중앙 영역으로 관찰한다.
2. 같은 이미지를 반전하고 6개 항목 체크리스트로 별도 재확인한다. 단, 주 실험에서는 Observer와 Cross-Checker가 모두 `qwen2.5vl:3b`를 사용해 모델 독립성은 제한된다.
3. 관찰문만 사용해 품목별 검사 기준서의 관련 조항을 RAG로 검색한다.
4. LLM이 아니라 코드 결정표로 `PASS / REJECT / ABSTAIN`을 판정한다.
5. 위험 조치·저확신·판정 보류는 사람 승인 또는 반려를 받는다.
6. 보고서가 실제 관찰문과 기준서 조항을 인용했는지 다시 검증한다.
7. 동일 이미지의 단독 VLM(B0)과 전체 Agent(A)를 고정 dev 셋에서 비교한다.

## 2. 사용 기술과 구현 결과

| 기술 요소 | 실제 적용 | 근거 파일 |
|---|---|---|
| Multi-Agent | Observer·Cross-Checker 병렬, Spec-Judge 순차 대조 | [`agent/subgraphs/`](agent/subgraphs) |
| LangGraph | route→plan→gate→research→analysis→write→HITL→validate→finalize | [`agent/graph.py`](agent/graph.py) |
| MCP | `inspect_image` Tool, `spec://`, `guide://`, `inspection_prompt` | [`mcp_server_vision.py`](mcp_server_vision.py) |
| RAG | 기준서 조항 top-3 검색, 품목 필터, 결과 스키마 검증 | [`agent/rag/`](agent/rag) |
| 의존성 주입 | LLM·MCP·Vector Store·Session Store를 `SupervisorDeps`로 주입 | [`agent/deps.py`](agent/deps.py) |
| HITL | LangGraph interrupt/resume, 승인 ID 검증, stale ID 409 차단 | [`agent/nodes/hitl.py`](agent/nodes/hitl.py), [`server.py`](server.py) |
| 근거 검증 | 보고서 관찰문·조항·결함 유형·스키마를 코드로 검사 | [`agent/nodes/validate.py`](agent/nodes/validate.py) |
| LangSmith | 노드·LLM·Tool 계층 trace, 선택적 Dataset 업로드 | `@traceable`, [`eval/run_eval.py`](eval/run_eval.py) |
| 모듈화 | nodes / subgraphs / vision / rag / eval 계층 분리 | [`agent/`](agent), [`eval/`](eval) |
| 단위·통합 테스트 | fake 외부 호출 차단, 결정표·경계값·재개·provenance 검증 | [`tests/`](tests) |

### 안전하게 실패하는 판정 계약

- Cross-Checker는 정확히 6개의 boolean `checks`를 채워야 한다.
- `checks`, 이상 여부, 위치, 특징이 모순되거나 누락되면 `C?`로 처리한다.
- 두 시각 분기와 기준서 위반이 일치하고 양성 위치가 겹칠 때만 `REJECT/high`다.
- 근거가 없거나 서로 충돌하면 정상으로 넘기지 않고 `ABSTAIN/재검사`로 보낸다.
- `ABSTAIN`은 불량 판정도 합격 판정도 아니다.

자유문장과 실제 픽셀의 진실성까지 구조 검사만으로 증명할 수는 없다. 이 한계는
MVTec 외부 라벨에 대한 종단 성능과 Cross-Checker 일치율로 별도 측정한다.

## 3. 실행 화면

아래 필수 네 장과 선택 비교 한 장이 공고의 “기능별 화면 캡처 + 최종 실행 화면”에 해당한다.
아래 UI 4장은 외부 호출이 없는 fake 백엔드로 기능 흐름을 재현한 캡처이며,
성능 수치는 별도 Ollama 실측 결과만 사용했다.

1. `docs/images/01-upload.jpg` — 품목 선택과 이미지 업로드
2. `docs/images/02-agent-events.jpg` — Observer·Cross-Checker·Spec-Judge 진행 이벤트
3. `docs/images/03-hitl.jpg` — ApprovalRequired와 승인·반려 버튼
4. `docs/images/04-final-report.jpg` — 판정·근거 조항·조치가 포함된 최종 보고서
5. `docs/images/05-before-after.jpg` — 동일 입력 B0/A 결과 비교(선택 보너스)

![이미지 업로드 화면](docs/images/01-upload.jpg)
![멀티에이전트 진행 이벤트](docs/images/02-agent-events.jpg)
![HITL 승인·반려 화면](docs/images/03-hitl.jpg)
![최종 검사 보고서](docs/images/04-final-report.jpg)
![동일 입력 B0/A 비교](docs/images/05-before-after.jpg)

## 4. 실행 방법

### 빠른 확인 — fake 백엔드

외부 모델 호출과 API 비용 없이 전체 Graph와 HITL UI를 확인한다.

```bash
bash sh/setup_venv.sh
bash sh/setup_data.sh
bash sh/demo_fake.sh
```

스크립트가 출력한 주소를 브라우저에서 열어 품목과 이미지를 선택한다.

### 실제 모델 — 로컬 Ollama

OpenAI API 대신 로컬 모델을 기본 경로로 사용한다. API 과금은 없지만 CPU/GPU·메모리와
실행 시간이 필요하다.

```bash
cp .env.example .env
bash sh/setup_ollama.sh       # 최초 1회
bash sh/run_agent_ollma.sh    # 백엔드: http://localhost:8100
bash sh/run_frontend.sh       # 별도 터미널: http://localhost:8600
```

OpenAI는 선택적 백엔드다. `bash sh/run_agent_openai.sh`를 사용하면 ChatGPT 구독과
별개로 API 사용량이 과금된다.

### 테스트

```bash
LLM_BACKEND=fake VLM_BACKEND=fake EMBED_BACKEND=fake \
  .venv/bin/python3 -m pytest -q
```

최신 전체 회귀 결과: **665 passed in 31.62s**.

## 5. Before / After 비교

### 비교 조건

- 데이터: 사전 고정한 MVTec AD dev 52장
- B0: VLM 단독 판정
- A: Observer + Cross-Checker + 기준서 RAG + 코드 판정 + 보고서 검증
- 오류·누락·`ABSTAIN`을 정확도 분모에서 제외하지 않음
- batch는 `auto_approve`이므로 사람 승인이 정확도를 높였다고 해석하지 않음

### R6.5 dev 스냅샷 실측(현재 R6.6 성능이 아님)

정식 run `20261022T002750112528Z-9befe77d286f`에서 R6.5 코드로 동일한 dev
52장을 B0와 A에 각각 입력했다. 양쪽 모두 누락·오류·무효 판정 없이 완료했다.

| 지표 | B0: 단독 VLM | A: 전체 Agent |
|---|---:|---:|
| 결과 / 전체 | 52 / 52 | 52 / 52 |
| 오류 / 누락 / 무효 | 0 / 0 / 0 | 0 / 0 / 0 |
| End-to-end Accuracy | 28.85% | 9.62% |
| Balanced Accuracy | 26.79% | 10.42% |
| `ABSTAIN` / Coverage | 29 / 44.23% | 42 / 19.23% |
| 정상 정답 / 정상 24장 | 0 / 24 | 5 / 24 |
| 결함 정답 / 결함 28장 | 15 / 28 | 0 / 28 |
| Selective Accuracy | 65.22% | 50.00% |
| 유형 Macro-F1 | 13.57% | 0.00% |
| 인용 충실도 | 60.00% | 68.40% |
| 지연 p50 / p95 | 4.269초 / 9.110초 | 65.405초 / 84.120초 |
| 평균 계산 자원 proxy / 장 | 0.000573 | 0.022970(setup 상각 포함) |

A는 정상 5장을 PASS로 회복했지만 결함 28장을 하나도 맞히지 못했고, 42장을
`ABSTAIN`으로 보냈다. Balanced Accuracy는 B0보다 16.37%p 낮았다
(`Δ=-16.37%p`, 95% CI `[-28.0, -3.9]`%p, `p=0.040798`). 이는 dev 탐색 결과라
held-out 가설 결론은 아니지만, **현재 3B 로컬 모델에서 전체 Agent의 성능 개선은
입증되지 않았으며 오히려 악화됐다.** 엄격한 Cross 계약은 근거 부족을 확정 판정으로
넘기지 않았지만, 관찰 품질 자체를 개선하지 못해 coverage와 결함 재현율이 무너졌다.

R6.6에서는 Observer와 Cross-Checker에 같은 정상 참조를 제공하고 응답 계약을
강화했지만, 아직 같은 고정 dev 전체를 다시 측정하지 않았다. 따라서 R6.5 스냅샷을
R6.6 성능으로 해석하지 않으며, 재측정 결과가 나오기 전 현재 성능 판정은 **NO-GO**다.

상세 지표·모델·무결성 정보는 [R6.5 dev 공개 요약](docs/results/r6_5_dev_summary.md)에
고정했다. 원시 JSONL은 이미지 경로를 포함하므로 Git에는 올리지 않는다.

이전 R6 실행에서 저장된 A의 실제 RAG query만 다시 사용한 결함 28건 top-3 결과는
`nomic-embed-text` 17/28(60.71%), `bge-m3` 23/28(82.14%)였다.
같은 이전 R6 코드에서 고정 dev 20장을 3회 반복했을 때 verdict 변동은
0/20이었지만, strict Cross 계약 보정 후 반복 측정값은 아니다. 낮은 정확도의
결과도 일관될 수 있으므로 성능 지표와 분리해 해석한다.

held-out 140장은 개발 중 보지 않았으며, 코드·기준서·설정을 동결한 뒤 B0/B1/A를
한 번씩 실행하는 최종 평가용으로 남겨 두었다.

## 6. 검증 방법

블랙박스와 화이트박스 기법을 기능 위험에 맞게 조합했다.

| 기법 | 적용 예 |
|---|---|
| 결정표 테스트 | Observer / Cross / Spec-Judge 조합별 PASS·REJECT·ABSTAIN |
| 경계값 분석 | crop 0/1개 양성, 2/3개 음성, 재작성 횟수, 업로드 크기 |
| 동등 분할 | 정상·결함·판독불가, valid·invalid schema, 승인·반려 |
| 결함 주입 | MCP 첫 호출 timeout, 잘못된 JSON·조항·모델 usage |
| 변형 테스트 | 좌우 반전 좌표 복원, 경로·라벨을 바꿔도 규칙 불변 |
| 상태 전이 테스트 | HITL 중단→승인/반려→새 승인 ID→stale ID 차단 |
| 통합·회귀 테스트 | MCP subprocess, LangGraph E2E, SSE, eval resume, manifest hash |

실험·리뷰 과정과 해결 내역은 [`docs/troubleshooting.md`](docs/troubleshooting.md)에
`증상 → 왜 문제인가 → 조치 → 결과` 형식으로 누적했다.

## 7. 어려웠던 점과 해결 경험

### 그럴듯한 자유문장을 근거로 오인

Cross-Checker가 실제 6개 항목을 확인하지 않고 “제품 외관이 보인다”라고만 답해도
양성으로 처리되는 문제가 있었다. 자유문장 신뢰 대신 exact 6-key boolean 계약을 만들고,
누락·추가·타입 오류·구조 모순은 판독 불가로 닫았다.

### 보고서 검증과 이미지 진실성의 차이

보고서가 VLM 관찰문을 정확히 인용해도 최초 관찰 자체가 틀릴 수 있었다. 따라서
인용 충실도와 실제 판정 정확도를 별도 지표로 분리했고, 근거가 충돌하면 REJECT가 아니라
ABSTAIN으로 보내도록 결정표를 보수적으로 바꿨다.

### 로컬 모델 자원과 긴 평가 시간

텍스트 LLM과 VLM을 동시에 적재하면 브라우저와 메모리가 경합했다. Ollama의 적재 모델과
병렬 수를 1로 제한하고, 고정 roster를 4장 checkpoint 단위로 재개하도록 만들었다.
실패·누락 행도 분모에 남겨 좋은 결과만 선택할 수 없게 했다.

## 8. 느낀 점과 향후 개선

### 느낀 점

- Agent를 여러 개 연결한다고 정확도가 자동으로 올라가지는 않는다.
- RAG는 틀린 관찰도 기준서와 그럴듯하게 연결할 수 있어, 입력 근거 품질이 먼저다.
- `ABSTAIN`과 HITL은 실패가 아니라 근거 부족을 숨기지 않는 운영 안전장치다.
- 평가에서는 정확도 하나보다 오류 수, 보류율, 지연, 비용, 재현성을 함께 봐야 한다.
- 구현 자체뿐 아니라 “어떤 코드·데이터·모델로 나온 결과인가”를 남기는 일이 중요했다.

### 향후 개선

1. 더 강한 VLM 또는 결함 위치 탐지 모델을 도입해 관찰 정확도를 개선한다.
2. 품목 재식별기를 별도로 두어 다른 사물명 환각을 구조적으로 차단한다.
3. 반려 사유가 보고서 조치 수정에 실제 반영되는 HITL 재작성 루프를 강화한다.
4. Observer·Cross-Checker 모델과 프롬프트를 dev에서 개선한 뒤 다시 측정하고,
   성능 게이트를 통과할 때만 설정을 동결해 held-out을 1회 평가한다.
5. 실제 검사자가 판정·근거·조치 유용성을 평가하는 사용자 실험을 추가한다.

## 9. 저장소 구조와 상세 문서

```text
agent/                 LangGraph 노드·서브그래프, RAG, VLM 백엔드
data/specs/            품목별 검사 기준서
data/guides/           결함 유형별 조치 가이드
eval/                  고정 데이터셋, 실행기, 지표, provenance
tests/                 단위·통합·회귀 테스트
docs/adr/              핵심 아키텍처 결정 기록
sh/                    환경·데이터·서버·평가 실행 스크립트
mcp_server_vision.py   VLM Tool·기준서/guide Resource·Prompt
server.py              FastAPI(:8100), SSE, 업로드, HITL 재개
web_demo_server.py     Vue 화면 프록시(:8600)
```

- [설계 SSOT](docs/design.md)
- [아키텍처·시퀀스 다이어그램](docs/architecture.md)
- [구현·검증 계획](docs/plan.md)
- [트러블슈팅 기록](docs/troubleshooting.md)
- [ADR: RAG 판정](docs/adr/0001-spec-judgement-via-rag.md)
- [ADR: 코드 합의 규칙](docs/adr/0002-consensus-as-code-rule.md)
- [ADR: HITL 위치](docs/adr/0003-hitl-between-write-and-validate.md)
- [ADR: VLM MCP 격리](docs/adr/0004-vlm-behind-mcp-tool.md)

## 제출 전 마지막 확인

- [x] 반·번호·성명 입력
- [x] 실제 프로젝트 화면 캡처 4장 추가
- [x] 선택 보너스용 동일 입력 B0/A 비교 화면 추가
- [x] GitHub 저장소 Public 및 링크 접속 확인
- [ ] 최종 변경사항 커밋·푸시
- [x] fake 전체 회귀 통과
- [x] dev B0/A·RAG·일관성 결과와 한계 공개
- [x] held-out 미열람 유지

## Attribution

MVTec AD: Bergmann et al., “A Comprehensive Real-World Dataset for Unsupervised
Anomaly Detection,” CVPR 2019, CC BY-NC-SA 4.0.
