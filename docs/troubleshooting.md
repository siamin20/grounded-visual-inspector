# grounded-visual-inspector 트러블슈팅 기록

- 작성일: 2026-10-20
- 범위: Phase 1 데이터·기준서, Phase 2 vision MCP, Phase 3 그래프·RAG 이식,
  Phase 4 서버·HITL·웹 UI, Phase 5 평가 파이프라인
- 원칙: 기준서를 작성·검토할 때 테스트 이미지를 열지 않고, MVTec 유형 정의와 일반 품질 기준만 사용했다.

## 1. Phase 1 최초 구성

### 환경

`requirements.txt`에 다음 의존성을 추가했다.

- `pillow`
- `numpy`
- `huggingface_hub`
- `pytest`
- `pytest-asyncio`
- `langgraph-checkpoint-sqlite`
- `python-multipart`

`bash sh/setup_venv.sh`로 `.venv`를 생성하고 전체 패키지 import를 확인했다. 첫 시도에서는 제한된 네트워크 환경의 DNS 조회가 실패했고, 네트워크 접속이 가능한 환경에서 동일한 스크립트를 재실행해 성공했다. 이는 의존성 정의 문제가 아니라 네트워크 접속 문제였다.

### 데이터 생성기

`eval/make_dataset.py`와 `sh/setup_data.sh`를 추가했다.

- Hugging Face `foersben/mvtec-ad`에서 지정된 4개 품목과 유형만 다운로드한다.
- `ground_truth`와 제외 유형(`manipulated_front`, `flip`)은 다운로드하지 않는다.
- 난수 시드는 `20261020`으로 고정한다.
- 결함 유형당 다운로드 풀은 12장, 평가 샘플은 8장을 사용한다.
- 품목·유형별로 dev/heldout을 결정론적으로 나눈다.
- `eval/dataset.json`, `eval/type_map.json`, `data/lots.json`을 생성한다.

최초 실행에서는 train 정상 참조 이미지 20장과 test 이미지 216장, 총 236개 파일을 받았다. 이 때 test 정상 샘플이 품목당 12장으로 제한되어 평가 데이터셋은 총 160장이었다.

## 2. 사양 불일치

### 조치 가이드 개수

`data/_TASK.md`에는 조치 가이드가 “총 16개”로 적혀 있지만, `docs/design.md` §3.1에서 선택한 실제 결함 유형은 14개이다. 제외한 `manipulated_front`와 `flip`을 임의로 다시 추가하지 않고, 현재 평가에 사용하는 14개 결함 유형에 대해서만 가이드를 작성했다. 해당 판단은 `_tasks/QUESTIONS.md`에도 남겨 두었다.

## 3. 검사 기준서 검토와 고정

4개 기준서에 총 25개 조항을 작성했다.

| 품목 | 조항 수 | 판정 요소 |
|---|---:|---|
| screw | 6 | 전체 윤곽, 나사산 연속성, 머리·목 긁힘, 측면·상단 나사산 변형 |
| bottle | 6 | 입구·외곽 연속성, 큰 파손, 작은 파손, 이물·얼룩·부착물 |
| metal_nut | 6 | 외곽·중앙 구멍 대칭, 간격, 변색, 선형 긁힘 |
| hazelnut | 7 | 껍질 무결성, 균열, 절단면, 구멍, 인쇄 흔적 |

검토 중 다음 문구를 다듬었다.

- `SCREW-05`: `닳히거나` → `닳거나`
- `SCREW-06`: `달리 지어지거나` → `달리 찌그러지거나`
- `BOTTLE-01`: 촬영 방향에 의존하는 `밑면`을 `용기 외곽`으로 명확히 했다.
- `BOTTLE-04`: `내용물 보존`이라는 기능 추론을 제거하고, 안쪽까지 이어진 `열린 틈`으로 바꿔 시각 판정만 사용했다.
- `BOTTLE-05`, `BOTTLE-06`: 국소 조각 결손과 오염 형태를 자연스럽고 구체적인 표현으로 다듬었다.
- `METAL_NUT-03`, `METAL_NUT-04`: 대칭과 간격의 비교 대상을 명시했다.
- `METAL_NUT-06`: 모호한 `표면의 연속적인 흐름`을 `주변의 정상 질감과 구별되는 연속된 선형 긁힘`으로 다듬었다.
- hazelnut 기준서는 필수 수정 사항이 없어 유지했다.

절대 치수를 사용하지 않고 위치·형태·연속성·색·질감처럼 이미지에서 확인 가능한 상대적·정성적 표현만 남겼다. 기준서는 에이전트의 판정 근거이지만 정확도 채점의 정답은 아니며, 채점 정답은 MVTec 폴더 라벨이다.

### 기준서 고정본 SHA-256

```text
a4ee1709bbe81fd760b5ada592b614fbfb065b2bd6cebf315b6c0f1e2832fc0b  data/specs/screw.md
abae8c57172eadb02109ba911d7de6d2e983ca9fb3fa7e0e432e011cf17d35ab  data/specs/bottle.md
1b98d98e198e545ea108390dc1d41026278dc48aa8d468591480f219c0d32a95  data/specs/metal_nut.md
0cd55244251342937625db2545ce5476c8dfde4466aea2a530de01296fc21a22  data/specs/hazelnut.md
```

평가를 시작한 후에는 위 네 문서를 수정하지 않는다. 변경이 불가피하면 기존 평가와 섞지 않고 별도 버전과 변경 사유를 남겨야 한다.

## 4. `good` 샘플 수 부족

### 현상

최초 `bash sh/setup_data.sh` 실행 후 품목당 정상 샘플이 12장만 선택되었다. `docs/design.md` §3.1의 목표는 품목당 20장이므로 총 데이터셋도 예정한 192장이 아닌 160장으로 생성되었다.

### 원인

`eval/make_dataset.py` 샘플링 로직은 이미 `good=20`, `defect=8`이었다. 하지만 `_expected_download_paths()`가 `good`을 포함한 모든 test 유형에 동일하게 `000.png`부터 `011.png`까지만 요청했다. 샘플러가 20장을 요청해도 로컬에 12장만 있어 12장을 모두 사용한 것이 원인이었다.

### 보정

test 다운로드 범위를 다음과 같이 분리했다.

```python
for index in range(20 if label == "good" else 12)
```

- `test/good`: `000.png`~`019.png` 20장
- `test/{defect}`: `000.png`~`011.png` 12장
- 데이터셋 선택: `good` 20장, 각 결함 유형 8장
- 시드: `20261020` 유지

`eval/make_dataset.py`의 다운로드 패턴 한 곳만 수정했고, 기준서·가이드·분할 로직은 변경하지 않았다.

### 재실행

```bash
bash sh/setup_data.sh
```

추가로 품목당 정상 샘플 8장씩을 받은 후 `eval/dataset.json`과 `data/lots.json`이 재생성되었다.

| 품목 | 전체 | good | dev | heldout |
|---|---:|---:|---:|---:|
| screw | 52 | 20 | 14 | 38 |
| bottle | 44 | 20 | 12 | 32 |
| metal_nut | 44 | 20 | 12 | 32 |
| hazelnut | 52 | 20 | 14 | 38 |
| **합계** | **192** | **80** | **52** | **140** |

### dev 52장과 예상치 약 58장의 차이

192장의 30%는 57.6장이지만 현재 코드는 전체에서 30%를 한 번에 뽑지 않고 각 `품목×유형` 층에서 다음을 각각 계산한다.

```python
int(len(selected) * 0.30 + 0.5)
```

- good 층: 20장 × 30% → 6장
- 결함 층: 8장 × 30% → 2.4장 → 2장
- good 4층 × 6장 + 결함 14층 × 2장 = dev 52장

전체 dev를 58장에 맞추려면 층화 비율의 나머지를 전역으로 배분하는 별도 규칙이 필요하다. 이번 요청은 `good` 다운로드 수 부족 한 건만 보정하는 범위이므로 분할 규칙은 바꾸지 않고 실제 결과를 기록했다.

## 5. 최종 검증

```text
dataset: 192 unique samples, seed=20261020
lots: 192/192 IDs matched
screw: total=52, good=20, dev=14, heldout=38
bottle: total=44, good=20, dev=12, heldout=32
metal_nut: total=44, good=20, dev=12, heldout=32
hazelnut: total=52, good=20, dev=14, heldout=38
```

추가로 다음을 확인했다.

- 모든 샘플 ID 192개가 고유하다.
- `dataset.json`의 모든 경로가 실제 파일과 대응한다.
- `lots.json`의 키 집합이 샘플 ID 집합과 일치한다.
- 품목당 `test/good` 20장이 존재한다.
- `git diff --check`가 통과했다.
- 데이터 보정 중 기준서 SHA-256이 변하지 않았다.

Phase 1 당시 자동 테스트 파일이 아직 없어 `pytest -q`는 `no tests ran`(종료 코드 5)를 반환했다. 코드 테스트 실패가 아니라 수집할 테스트가 없었던 상태였다.

## 6. Phase 2 리뷰에서 발견한 것

### 결함 유형 목록 하드코딩

- 증상: `agent/vision/backends.py` 에 품목별 결함 유형 목록(`_DEFECT_TYPES`)이 딕셔너리로
  직접 박혀 있었다. 같은 정보가 `eval/type_map.json` 에도 있어 출처가 둘이 됐다.
- 왜 문제인가: 품목을 추가하거나 유형을 제외할 때 두 곳을 같이 고쳐야 하고, 하나만 고치면
  baseline 프롬프트와 채점 기준이 어긋난다. Base Agent 에서도 시스템명 하드코딩이 같은 종류의
  회귀를 냈던 전례가 있다(`docs/base/regression_test_queries.md`).
- 조치: Phase 3 에서 `type_map.json` 을 단일 출처로 삼아 vision 백엔드도 거기서 읽도록 변경.
  해소되면 아래에 결과를 적는다.
- 결과: `agent/typemap.py` 의 `load_type_map()` 하나로 통일했다. `_DEFECT_TYPES` 딕셔너리는
  삭제됐고(`grep -rn "_DEFECT_TYPES" agent/` 결과 0건), 노드·서브그래프·vision 백엔드·baseline
  프롬프트가 전부 `eval/type_map.json` 을 읽는다. 유형을 바꾸려면 파일 하나만 고치면 된다.
- 결과: `agent/typemap.py`로 공통 로더를 분리했고, `agent/nodes/common.py`와
  `agent/vision/backends.py`가 이 로더를 공유한다. `eval/make_dataset.py`의 품목·유형
  하드코딩도 제거해 `eval/type_map.json`을 직접 읽는다. Fake 전체 18개 품목·유형
  경로가 같은 매핑으로 판정되는 것을 확인했다.

## 7. Phase 3 그래프 이식

### 병렬 서브그래프의 상태 충돌 가능성

- 현상: Observer와 Cross-Checker가 LangGraph의 같은 단계에서 `events`, `tasks`,
  `retry_count` 같은 필드를 동시에 갱신하면 병렬 update 충돌이 발생할 수 있었다.
- 왜 문제인가: 병렬 분기의 공통 필드 overwrite는 결과 손실 또는 LangGraph
  `InvalidUpdateError`로 이어질 수 있다.
- 조치: 각 분기가 `_observer_result`, `_cross_result`라는 독립 임시 필드만 쓰고,
  Spec-Judge가 두 결과와 이벤트·task 상태·재시도 횟수를 한 번에 병합하게 했다.
- 결과: Observer 5개 영역과 Cross-Checker가 병렬 분기로 실행되고, 그 뒤에만
  Spec-Judge가 실행되는 순서를 단독 테스트와 HTTP 통합 테스트에서 확인했다.

### 코드는 병렬인데 MCP 호출은 직렬인 문제

- 현상: research는 `asyncio.gather` 병렬 구조였지만 `MCPClient.call_tool()`이
  전역 lock을 잡고 있어 실제 vision MCP 호출은 한 번에 하나씩만 실행됐다.
- 왜 문제인가: 코드 형태만 병렬이고 실제 지연은 6개 호출의 합으로 늘어나
  설계의 병렬 관찰 효과가 사라진다.
- 조치: resource·prompt·tool 탐색의 lock은 보존하되, request ID로 응답을 분리하는
  `ClientSession.call_tool()`은 전역 lock 밖에서 실행하게 바꿘다.
- 결과: 두 tool 요청이 실제로 겹쳐 진입하는 회귀 테스트와 실제 stdio MCP를
  사용한 POST·SSE 통합 테스트가 통과했다.

### `metal_nut/color`가 `scratch`로 판정된 Fake 회귀

- 현상: `type_map.json`의 good 포함 18개 경로를 전수 점검했을 때 17개는 맞았지만
  `metal_nut/color` 하나가 `scratch`로 판정됐다.
- 왜 문제인가: Fake judge의 조사 정규화 후 일반어 `구별되`가 stopword에서
  누락됐다. `scratch`의 `구별되`와 `color`의 `변색`이 각각 1점으로 동점이 되자
  RAG에서 먼저 나온 scratch 조항이 선택됐다.
- 조치: 정규화된 일반 어간 `구별되`를 stopword에 추가하고 해당 동점 사례를
  회귀 테스트로 고정했다.
- 결과: Fake 전체 매트릭스가 `screw 5/5`, `bottle 4/4`, `metal_nut 4/4`,
  `hazelnut 5/5`, 합계 `18/18`로 통과했다.

### 작성 LLM 출력을 HITL 전에 사용하는 문제

- 현상: Fake 작성기의 1차 보고서에 실제 관찰에 없는 문장이 섞여 검증에서
  반려됐다. 추가 리뷰에서 `actions` 원소가 문자열이 아니면 validate에 도달하기
  전 HITL의 `set()` 변환이 예외를 낼 수 있는 것도 발견했다.
- 왜 문제인가: write는 LLM 출력이므로 JSON 스키마뿐 아니라 코드 규칙이 내린
  `verdict/defect_type/confidence`와 근거·조항이 일치하는지 다시 검증해야 한다.
- 조치: 관찰 문장 grounding, 조항 존재·품목·Spec-Judge 판정, 결함 유형,
  스키마와 코드 판정 일치를 검증했다. HITL은 문자열 action만 안전하게 집합으로
  만들고, 3회 실패 강제 통과 시 오류는 `validation_errors`에 남기되 재작성을
  의미하는 `validation_feedback`은 비우게 했다.
- 결과: `scratch_neck` HTTP 검사는 1차 보고서를 반려하고 2차에 근거가 확인된
  보고서로 완료했다. 잘못된 action 형태와 판정 변조는 그래프 중단 없이
  validator가 반려하는 테스트로 고정했다.

### Before 기준선에 RAG 의존성이 섞인 문제

- 현상: 서버 lifespan이 `SUPERVISOR_VERSION=v1`에서도 임베더와 SpecIndex를 생성해
  B0/B1이 실제로는 임베딩 백엔드에 의존했다.
- 왜 문제인가: Before는 VLM 1회만 쓰는 비교 기준이므로 불필요한 RAG 초기화가
  비용·실패 조건을 변경하고, OpenAI 임베딩 키가 없으면 B0/B1까지 실패할 수 있다.
- 조치: `SUPERVISOR_VERSION=v2`일 때만 임베더와 SpecIndex를 lifespan에서 한 번
  생성하고, v1의 `vector_store`는 `None`으로 두었다.
- 결과: `EMBED_BACKEND=openai`이고 API key가 없는 환경에서도 v1/B1 POST 검사가
  성공하며, VLM Tool은 딱 1회만 호출된다.

### Phase 3 완료 검증

- `POST /chat` fake 검사: `REJECT / scratch_neck / SCREW-04`, 검증 2회 후 완료.
- SSE: `RouteDecided → PlanCreated → GateApplied → ObservationReady×5 →
  CrossCheckDone → SpecJudged → ReplanCheck → AnalysisDecided → 작성·승인·검증
  2회 → Finalize → FinalAnswer → [DONE]` 순서 확인.
- RAG: 4개 기준서 25개 조항 색인, 품목 필터 top-3, clause/full-text 조회 통과.
- fake 전체 테스트: `44 passed`; `import agent.graph`, `compileall`, `git diff --check` 통과.
- 기준서·가이드와 실제 테스트 이미지는 열거나 수정하지 않았다.

## 8. Phase 4 서버·HITL·웹 UI

### 같은 관찰문이 보고서에 반복되는 문제

- 증상: Observer의 5개 크롭이 같은 문장을 반환하면 `report.observations`에 같은 문장이
  최대 5번 반복됐다.
- 왜 문제인가: 같은 근거가 여러 개인 것처럼 보이고 최종 보고서가 불필요하게 길어지며,
  단순히 한 항목을 제거하면 어떤 VLM 호출에서 나온 근거인지 추적 정보가 사라진다.
- 조치: `write_node`가 보고서 JSON을 파싱한 직후 `text`가 완전히 같은 항목을 하나로
  합치고, 모든 `source_call_id`를 등장 순서대로 중복 없이 리스트에 보존한다. validator와
  finalizer는 기존 문자열과 새 리스트 형식을 모두 처리한다.
- 결과: 같은 문장은 한 번만 렌더되고 `obs-full`, `obs-tl` 같은 원본 호출 ID는 모두 남는다.
  공백까지 다른 문장은 임의로 합치지 않는 회귀 테스트도 통과했다.

### `ApprovalRequired`의 거짓·중복 이벤트

- 증상: 승인이 필요하지 않은 보고서도 `required=false`인 `ApprovalRequired`를 냈고,
  validate 반려 뒤 write가 재실행될 때마다 같은 승인 이벤트가 반복됐다.
- 왜 문제인가: UI가 불필요한 승인 카드를 만들 수 있고, 승인한 요청이 재작성 루프에서 다시
  멈추거나 반려 후 같은 위험 조치를 계속 재승인해야 하는 루프가 생길 수 있다.
- 조치: required가 false이면 승인 이벤트를 생략한다. interactive에서는 최초
  `interrupt()` payload만 `ApprovalRequired`로 보내고 resume 후에는 `ApprovalResolved`만
  누적한다. 승인·반려 상태를 요청 안에서 보존해 이후 write 재진입은 다시 중단하지 않는다.
- 결과: 위험 보고서는 요청당 `ApprovalRequired`가 정확히 1회 발생하고 `[PAUSED]`에서
  멈춘 뒤 `/chat/resume`으로 이어진다. 일반 보고서는 승인 이벤트 없이 완료된다.

### 서버 재시작 직후 resume이 간헐적으로 상태를 못 찾는 문제

- 증상: `__interrupt__` 청크를 받은 즉시 `astream` 반복기를 종료하면, 바로 서버 lifespan을
  닫고 다시 연 테스트에서 저장된 중단 상태가 없는 경우가 있었다.
- 왜 문제인가: 설치된 `AsyncSqliteSaver`는 비동기 쓰기를 사용하므로 스트림 조기 종료와
  SQLite checkpoint 저장 취소가 경합하면 “서버 재시작 후 재개” 계약을 보장할 수 없다.
- 조치: 설치 버전(`langgraph 1.2.16`, `langgraph-checkpoint-sqlite 3.1.2`)의 실제 API를
  확인해 `AsyncSqliteSaver.from_conn_string()`을 lifespan 전체에서 열었다. 모든 v2 호출은
  `durability="sync"`와 같은 `configurable.thread_id`를 쓰고, interrupt 뒤 반복기를 자연
  종료한 다음 `[PAUSED]`를 보낸다.
- 결과: 첫 lifespan에서 중단하고 SQLite 연결을 닫은 뒤, 두 번째 lifespan에서 같은
  thread ID로 `/chat/resume`해 `FinalAnswer → [DONE]`까지 완료했다. 중단 전 이벤트도
  재전송되지 않았다.

### 같은 thread ID의 다음 요청이 이전 승인을 재사용하는 문제

- 증상: checkpointer를 붙인 뒤 같은 thread ID로 새 검사를 시작하면 이전 요청의
  `approval`, `report`, `validation_attempts`가 새 입력과 병합될 수 있었다.
- 왜 문제인가: 앞선 요청의 `decision=approve`가 남으면 다음 위험 보고서가 HITL을 건너뛸
  수 있고, 재작성 횟수와 관찰 결과도 요청 사이에서 오염된다.
- 조치: 새 그래프 입력을 만들 때 승인·보고서뿐 아니라 task, 관찰, 판정, 검증, 비용,
  분기 임시 결과 등 모든 요청 단위 필드를 명시적으로 초기화하고 대화 history만 유지한다.
- 결과: 같은 thread ID로 두 번째 위험 검사를 보내도 다시 정확히 한 번 중단되며, 첫 요청을
  resume하기 전에는 어느 요청도 세션 이력에 저장되지 않는다.

### 업로드·상태 관측 계약 누락

- 증상: 서버에 `/inspect`, `/chat/resume`이 없고 `/health`와 `/chat` metadata만으로는
  실행 중인 VLM·임베딩·HITL 설정과 판정 결과를 확인할 수 없었다. 진행 이벤트에도 실제
  관찰문과 교차 판독 요약이 없어 UI가 구체적인 문장을 표시할 수 없었다.
- 왜 문제인가: 브라우저가 POST 업로드 스트림이나 승인 재개를 수행할 수 없고, 평가·운영
  로그에서 실행 조건과 `verdict`, `defect_type`, VLM 호출 수를 재현하기 어렵다.
- 조치: 10MB 제한과 Pillow 검증을 거친 이미지만
  `data/uploads/{thread_id}/{uuid}.png`로 저장하고, 프록시는 multipart/JSON SSE를 그대로
  중계한다. `/health`에 `vlm_backend`, `embed_backend`, `hitl_mode`, `ablation`을,
  metadata에 판정·승인·호출 수 필드를 추가했다. Vue UI에는 품목 선택, 썸네일, 진행 카드,
  승인·반려 버튼을 연결했고 관찰·교차 이벤트에 문장 필드를 추가했다.
- 결과: 경로 조작 파일명, 비이미지, 10MB 초과 업로드를 거부하고 UUID PNG만 남긴다.
  `HITL_MODE=auto_approve`에서는 중단 없이 `[DONE]`까지 완료하며, interactive에서는
  승인 카드에서 재개할 수 있다.

### Phase 4 완료 검증

- fake 전체 테스트: `53 passed in 12.08s`.
- `/inspect`: UUID PNG 저장, 경로 조작·비이미지·10MB 초과 차단 확인.
- HITL: interactive 중단·승인·반려, auto-approve 무중단, SQLite 재시작 복원 확인.
- `import agent.graph`, Python compile, Vue 인라인 JavaScript 문법, `git diff --check` 통과.
- 기준서·가이드는 수정하지 않았고 테스트 이미지를 시각적으로 열지 않았다.

## 9. Phase 5 평가 파이프라인

### fake 결과가 실험 결과와 섞일 위험

- 증상: 서버의 기본 LLM·VLM이 fake이므로, 평가 실행기가 같은 기본값을 쓰면
  폴더 라벨을 읽는 결정론 결과가 실측처럼 저장될 수 있었다.
- 왜 문제인가: fake VLM은 파일 경로의 정답 라벨을 사용하므로 정확도 평가에 쓰면
  결과 조작이다. 로컬 검증 결과가 실제 실험 JSONL과 섞이면 집계도 오염된다.
- 조치: `eval/run_eval.py`는 `LLM_BACKEND` 또는 `VLM_BACKEND`가 fake이면 즉시
  종료하고, `--dry-run`을 명시한 경우만 허용했다. dry-run 파일에는
  `_DRYRUN`을 붙이고 LangSmith 전송을 비활성화했다. Dataset 업로드도
  `--push-dataset`이 있을 때만 실행한다. 오타난 백엔드가 암묵적으로 fake로
  떨어지지 않도록 지원 목록 밖의 LLM·VLM 값도 즉시 거부한다.
- 결과: 허용된 `A/dev/limit 5` dry-run은
  `eval/results/A_dev_DRYRUN.jsonl`에 5줄을 저장했고, 실제 API·LangSmith 호출은
  발생하지 않았다.

### 토큰 사용량이 평가 행에 남지 않는 문제

- 증상: 기존 상태는 `cost_usd`만 누적하고 `tokens`를 보존하지 않았다.
- 왜 문제인가: Phase 5 비용 표에는 조건별 평균 토큰이 필요하다. VLM·LLM·임베딩 중
  하나라도 빠지면 에이전트 층의 비용을 과소 계산한다.
- 조치: `AgentState` 및 초기 상태에 `tokens`를 추가했다. route·plan·judge·write의
  LLM 사용량, Observer·Cross-Checker의 VLM 사용량, RAG query 임베딩 사용량을
  Spec-Judge에서 병합하고 서버 metadata와 평가 JSONL에 전달했다.
- 결과: dry-run 5건 모두 1,882 tokens로 기록됐고, 행당 추정 비용은
  0.006217 USD, 누적은 0.031085 USD였다.

### 조건 비교를 독립 표본처럼 계산할 위험

- 증상: B0·B1·A는 같은 이미지를 판정하므로 조건별 정확도만 나란히 놓으면
  짝지은 정보를 버리게 된다.
- 왜 문제인가: 실험 설계는 같은 `id`의 Before/After 정·오답 변화를 검정해야
  하며, 정상·결함 비율이 다른 자료에서 단순 독립 비교는 검정력을 잃는다.
- 조치: `eval/evaluators.py`에 코드 기반 이진·유형·확신도·인용·비용 지표와,
  scipy 없이 정확 이항분포를 쓰는 McNemar 양측 검정을 구현했다. McNemar는
  전체·정상·결함 층을 따로 낸다. `eval/report.py`는 조건·품목·비용 표를
  마크다운으로 생성하고, `eval/human_check.py`는 B0/A 보고서를 조건 비노출로
  섞어 사람 환각 채점을 보조한다.
- 결과: 손계산 예제 `b=10, c=2`의 정확 p-value `0.03857421875`를 일치시켰고,
  평가 지표·리포트·블라인드 도구 테스트 16개가 통과했다.

### Phase 5 완료 검증

- `A/dev/limit 5` dry-run: 5건 전부 `PASS`, 오류 0건, JSONL 스키마 20개 필드 일치.
- 샘플 ID: `screw-good-004`, `005`, `007`, `010`, `012`.
- 평가 지표·출력·실행기 대상 테스트: `20 passed`.
- fake 전체 회귀 테스트: `73 passed in 11.38s`.
- 실제 API 실험, held-out 실험, 다른 조건 실험은 실행하지 않았다.
- 기준서·가이드는 수정하지 않았고 테스트 이미지를 시각적으로 열지 않았다.
