# grounded-visual-inspector 트러블슈팅 기록

- 작성일: 2026-10-20
- 최종 갱신: 2026-10-21
- 범위: Phase 1 데이터·기준서, Phase 2 vision MCP, Phase 3 그래프·RAG 이식,
  Phase 4 서버·HITL·웹 UI, Phase 5 평가 파이프라인, Phase 6 테스트·문서·실행 마감,
  R0 heldout 오염 방화벽, R1 판독 불가·재검사 계약, R2 안전 계약 구현,
  R3 평가 완전성·보수 집계, R4 통계·블라인드·ablation 실험 재설계
- 원칙: 기준서를 작성·검토할 때 테스트 이미지를 열지 않고, MVTec 유형 정의와 일반 품질 기준만 사용했다.

## 1. Phase 1 최초 구성

### 환경

`requirements.txt`에 추가한 의존성은 다음과 같다.

- `pillow`
- `numpy`
- `huggingface_hub`
- `pytest`
- `pytest-asyncio`
- `langgraph-checkpoint-sqlite`
- `python-multipart`

설치 과정에서는 한 번 막혔다.

- **증상**: 첫 시도에서 `bash sh/setup_venv.sh`를 실행하니 DNS 조회가 실패했다.
- **왜 문제?**: 패키지 설치 전에 멈춰 전체 import를 확인할 수 없었다.
- **조치**: 네트워크 접속이 가능한 환경에서 같은 스크립트를 다시 실행했다.
- **결과**:
  - `.venv`가 생성됐다.
  - 전체 패키지 import가 통과했다.
  - 의존성 정의가 아니라 네트워크 접속 문제였다.

### 데이터 생성기

- **조치**: `eval/make_dataset.py`와 `sh/setup_data.sh`를 추가했다.
  - Hugging Face `foersben/mvtec-ad`에서 지정된 4개 품목과 유형만 다운로드한다.
  - `ground_truth`와 제외 유형(`manipulated_front`, `flip`)은 다운로드하지 않는다.
  - 난수 시드는 `20261020`으로 고정한다.
  - 결함 유형당 다운로드 풀은 12장, 평가 샘플은 8장을 사용한다.
  - 품목·유형별로 dev/heldout을 결정론적으로 나눈다.
- **결과**:
  - `eval/dataset.json`, `eval/type_map.json`, `data/lots.json`이 생성됐다.
  - 최초 실행에서 train 정상 참조 이미지 20장과 test 이미지 216장, 총 236개 파일을 받았다.
  - 이때 test 정상 샘플이 품목당 12장으로 제한돼 평가 데이터셋은 총 160장이었다.

## 2. 사양 불일치

### 조치 가이드 개수

- **증상**:
  - `data/_TASK.md`: 조치 가이드 “총 16개”
  - `docs/design.md` §3.1: 실제 선택한 결함 유형 14개
- **왜 문제?**: 문서끼리 숫자가 달라 어떤 범위를 따라야 하는지 모호했다.
- **조치**: 제외한 `manipulated_front`와 `flip`을 임의로 다시 추가하지 않았다. 현재 평가에 사용하는 14개 결함 유형만 가이드로 작성했다.
- **결과**: 판단 근거를 `_tasks/QUESTIONS.md`에도 남겼다.

## 3. 검사 기준서 검토와 고정

4개 기준서에 총 25개 조항을 작성했다.

| 품목 | 조항 수 | 판정 요소 |
|---|---:|---|
| screw | 6 | 전체 윤곽, 나사산 연속성, 머리·목 긁힘, 측면·상단 나사산 변형 |
| bottle | 6 | 입구·외곽 연속성, 큰 파손, 작은 파손, 이물·얼룩·부착물 |
| metal_nut | 6 | 외곽·중앙 구멍 대칭, 간격, 변색, 선형 긁힘 |
| hazelnut | 7 | 껍질 무결성, 균열, 절단면, 구멍, 인쇄 흔적 |

검토에서 다듬은 문구는 다음과 같다.

- `SCREW-05`: `닳히거나` → `닳거나`
- `SCREW-06`: `달리 지어지거나` → `달리 찌그러지거나`
- `BOTTLE-01`: 촬영 방향에 의존하는 `밑면`을 `용기 외곽`으로 명확히 했다.
- `BOTTLE-04`: `내용물 보존`이라는 기능 추론을 제거하고, 안쪽까지 이어진 `열린 틈`으로 바꿔 시각 판정만 사용했다.
- `BOTTLE-05`, `BOTTLE-06`: 국소 조각 결손과 오염 형태를 자연스럽고 구체적인 표현으로 다듬었다.
- `METAL_NUT-03`, `METAL_NUT-04`: 대칭과 간격의 비교 대상을 명시했다.
- `METAL_NUT-06`: 모호한 `표면의 연속적인 흐름`을 `주변의 정상 질감과 구별되는 연속된 선형 긁힘`으로 다듬었다.
- hazelnut 기준서는 필수 수정 사항이 없어 유지했다.

- **작성 원칙**: 절대 치수는 쓰지 않았다. 이미지에서 확인 가능한 위치·형태·연속성·색·질감 같은 상대적·정성적 표현만 남겼다.
- **채점 원칙**: 기준서는 에이전트의 판정 근거다. 정확도 채점의 정답은 기준서가 아니라 MVTec 폴더 라벨이다.

### 기준서 고정본 SHA-256

```text
a4ee1709bbe81fd760b5ada592b614fbfb065b2bd6cebf315b6c0f1e2832fc0b  data/specs/screw.md
abae8c57172eadb02109ba911d7de6d2e983ca9fb3fa7e0e432e011cf17d35ab  data/specs/bottle.md
1b98d98e198e545ea108390dc1d41026278dc48aa8d468591480f219c0d32a95  data/specs/metal_nut.md
0cd55244251342937625db2545ce5476c8dfde4466aea2a530de01296fc21a22  data/specs/hazelnut.md
```

- **고정 규칙**: 평가를 시작한 뒤에는 위 네 문서를 수정하지 않는다.
- **예외 처리**: 변경이 불가피하면 기존 평가와 섞지 않는다. 별도 버전과 변경 사유를 남긴다.

## 4. `good` 샘플 수 부족

### 증상 → 원인 → 조치

- **증상**:
  - 최초 `bash sh/setup_data.sh` 실행 후 품목당 정상 샘플이 12장만 선택됐다.
  - `docs/design.md` §3.1의 목표는 품목당 20장이다.
  - 총 데이터셋도 예정한 192장이 아니라 160장으로 생성됐다.
- **왜 문제?**:
  - `eval/make_dataset.py`의 샘플링 로직은 이미 `good=20`, `defect=8`이었다.
  - 하지만 `_expected_download_paths()`가 `good`을 포함한 모든 test 유형에 `000.png`부터 `011.png`까지만 요청했다.
  - 샘플러가 20장을 요청해도 로컬에는 12장뿐이라 12장을 모두 사용했다.
- **조치**: test 다운로드 범위를 다음처럼 나눴다.

```python
for index in range(20 if label == "good" else 12)
```

| 항목 | 적용값 |
|---|---|
| `test/good` | `000.png`~`019.png` 20장 |
| `test/{defect}` | `000.png`~`011.png` 12장 |
| 데이터셋 선택 | `good` 20장, 각 결함 유형 8장 |
| 시드 | `20261020` 유지 |
| 수정 범위 | `eval/make_dataset.py`의 다운로드 패턴 한 곳 |
| 유지 항목 | 기준서·가이드·분할 로직 |

- **결과**: 아래 명령을 다시 실행했다.

```bash
bash sh/setup_data.sh
```

품목당 정상 샘플 8장씩을 추가로 받았다. 이후 `eval/dataset.json`과 `data/lots.json`이 재생성됐다.

| 품목 | 전체 | good | dev | heldout |
|---|---:|---:|---:|---:|
| screw | 52 | 20 | 14 | 38 |
| bottle | 44 | 20 | 12 | 32 |
| metal_nut | 44 | 20 | 12 | 32 |
| hazelnut | 52 | 20 | 14 | 38 |
| **합계** | **192** | **80** | **52** | **140** |

### dev 52장과 예상치 약 58장의 차이

- **증상**: 192장의 30%는 57.6장인데 실제 dev는 52장이다.
- **왜 문제?**:
  - 현재 코드는 전체에서 30%를 한 번에 뽑지 않는다.
  - 각 `품목×유형` 층에서 다음 값을 따로 계산한다.

```python
int(len(selected) * 0.30 + 0.5)
```

| 계산 단위 | 결과 |
|---|---:|
| good 층 | 20장 × 30% → 6장 |
| 결함 층 | 8장 × 30% → 2.4장 → 2장 |
| 최종 dev | good 4층 × 6장 + 결함 14층 × 2장 = 52장 |

- **조치**: 이번 요청은 `good` 다운로드 수 부족 한 건만 보정했다. 분할 규칙은 바꾸지 않았다.
- **결과**:
  - 실제 dev 52장을 기록했다.
  - dev를 58장에 맞추려면 층화 비율의 나머지를 전역으로 배분하는 별도 규칙이 필요하다.

## 5. 최종 검증

```text
dataset: 192 unique samples, seed=20261020
lots: 192/192 IDs matched
screw: total=52, good=20, dev=14, heldout=38
bottle: total=44, good=20, dev=12, heldout=32
metal_nut: total=44, good=20, dev=12, heldout=32
hazelnut: total=52, good=20, dev=14, heldout=38
```

- **결과**: 위 분포와 함께 다음 항목을 확인했다.
  - 모든 샘플 ID 192개가 고유하다.
  - `dataset.json`의 모든 경로가 실제 파일과 대응한다.
  - `lots.json`의 키 집합이 샘플 ID 집합과 일치한다.
  - 품목당 `test/good` 20장이 존재한다.
  - `git diff --check`가 통과했다.
  - 데이터 보정 중 기준서 SHA-256이 변하지 않았다.

- **주의**: Phase 1 당시에는 자동 테스트 파일이 아직 없었다.
  - `pytest -q`: `no tests ran`
  - 종료 코드: 5
  - 의미: 코드 테스트 실패가 아니라 수집할 테스트가 없었던 상태

## 6. Phase 2 리뷰에서 발견한 것

### 결함 유형 목록 하드코딩

- **증상**:
  - `agent/vision/backends.py`에 품목별 결함 유형 목록(`_DEFECT_TYPES`)이 딕셔너리로 직접 박혀 있었다.
  - 같은 정보가 `eval/type_map.json`에도 있어 출처가 둘이었다.
- **왜 문제?**:
  - 품목을 추가하거나 유형을 제외할 때 두 곳을 같이 고쳐야 했다.
  - 한 곳만 고치면 baseline 프롬프트와 채점 기준이 어긋난다.
  - Base Agent에서도 시스템명 하드코딩이 같은 종류의 회귀를 냈던 전례가 있다(`docs/base/regression_test_queries.md`).
- **조치**: Phase 3에서 `type_map.json`을 단일 출처로 삼았다. vision 백엔드도 여기서 읽도록 바꿨다. 해소 결과는 아래에 적었다.
- **결과**:
  - `agent/typemap.py`의 `load_type_map()` 하나로 통일했다.
  - `_DEFECT_TYPES` 딕셔너리를 삭제했다. `grep -rn "_DEFECT_TYPES" agent/` 결과는 0건이다.
  - 노드·서브그래프·vision 백엔드·baseline 프롬프트가 모두 `eval/type_map.json`을 읽는다.
  - 유형을 바꾸려면 파일 하나만 고치면 된다.
  - `agent/typemap.py`로 공통 로더를 분리했다. `agent/nodes/common.py`와 `agent/vision/backends.py`가 이 로더를 공유한다.
  - `eval/make_dataset.py`의 품목·유형 하드코딩도 제거했다. 이제 `eval/type_map.json`을 직접 읽는다.
  - Fake 전체 18개 품목·유형 경로가 같은 매핑으로 판정되는 것을 확인했다.

## 7. Phase 3 그래프 이식

### 병렬 서브그래프의 상태 충돌 가능성

- **증상**: Observer와 Cross-Checker가 같은 단계에서 공통 필드를 동시에 갱신했다.
  대상은 `events`, `tasks`, `retry_count` 등이었다.
- **왜 문제?**: 병렬 분기의 overwrite 때문에 결과가 사라지거나 LangGraph
  `InvalidUpdateError`가 날 수 있었다.
- **조치**: 두 분기는 각각 `_observer_result`, `_cross_result`만 쓰게 했다.
  공통 상태 병합은 Spec-Judge 한 곳에서 처리했다.
- **결과**: Observer 5개 영역과 Cross-Checker가 병렬로 실행되고, 완료 뒤에만
  Spec-Judge가 실행됐다. 단독 테스트와 HTTP 통합 테스트로 순서를 확인했다.

### 코드는 병렬인데 MCP 호출은 직렬인 문제

- **증상**: research는 `asyncio.gather` 구조였지만 실제 vision MCP 호출은
  한 번에 하나씩 실행됐다. `MCPClient.call_tool()`의 전역 lock 때문이었다.
- **왜 문제?**: 겉으로만 병렬이었다. 실제 지연은 6개 호출의 합이 되어
  병렬 관찰 효과가 사라졌다.
- **조치**: resource·prompt·tool 탐색의 lock은 유지했다.
  request ID로 응답을 분리했다.
  `ClientSession.call_tool()`은 전역 lock 밖에서 실행하게 바꾼다.
- **결과**: 두 tool 요청이 실제로 겹쳐 진입하는 회귀 테스트와 실제 stdio MCP를
  사용한 POST·SSE 통합 테스트가 통과했다.

### `metal_nut/color`가 `scratch`로 판정된 Fake 회귀

- **증상**: `type_map.json`의 good 포함 18개 경로를 전수 점검했을 때 17개는 맞았지만
  `metal_nut/color` 하나가 `scratch`로 판정됐다.
- **왜 문제?**: Fake judge의 조사 정규화 후 일반어 `구별되`가 stopword에서
  누락됐다. `scratch`의 `구별되`와 `color`의 `변색`이 각각 1점으로 동점이 되자
  RAG에서 먼저 나온 scratch 조항이 선택됐다.
- **조치**: 정규화된 일반 어간 `구별되`를 stopword에 추가하고 해당 동점 사례를
  회귀 테스트로 고정했다.
- **결과**: Fake 전체 매트릭스가 `screw 5/5`, `bottle 4/4`, `metal_nut 4/4`,
  `hazelnut 5/5`, 합계 `18/18`로 통과했다.

### 작성 LLM 출력을 HITL 전에 사용하는 문제

- **증상**:
  - Fake 작성기의 1차 보고서에 실제 관찰에 없는 문장이 섞였다.
  - `actions` 원소가 문자열이 아니면 validate 전에 HITL의 `set()` 변환이
    예외를 낼 수 있었다.
- **왜 문제?**: write는 LLM 출력이다. JSON 스키마만 맞는다고 신뢰할 수 없다.
  코드가 정한 `verdict/defect_type/confidence`와 근거·조항도 일치해야 한다.
- **조치**:
  - 관찰 문장 grounding을 검사했다.
  - 조항 존재 여부·품목·Spec-Judge 판정·결함 유형을 검사했다.
  - 스키마와 코드 판정의 일치를 검사했다.
  - HITL은 문자열 action만 안전하게 집합으로 만들게 했다.
  - 재작성 상한 뒤에도 무효이면 원문을 강제 통과시키지 않았다.
    `ABSTAIN/low`, `validation_failure`, `재검사` 보고서로 교체하고 다시 HITL을 거치게 했다.
- **결과**: `scratch_neck` HTTP 검사는 1차 보고서를 반려하고 2차에 근거가 확인된
  보고서로 완료했다. 잘못된 action 형태와 판정 변조는 그래프 중단 없이
  validator가 반려하는 테스트로 고정했다.

### Before 기준선에 RAG 의존성이 섞인 문제

- **증상**: `SUPERVISOR_VERSION=v1`에서도 임베더와 SpecIndex를 만들었다.
  B0/B1이 실제로는 임베딩 백엔드에 의존했다.
- **왜 문제?**: Before는 VLM 1회만 쓰는 비교 기준이다.
  불필요한 RAG 초기화가 비용·실패 조건을 바꿨다.
  OpenAI 임베딩 키가 없으면 B0/B1까지 실패할 수 있었다.
- **조치**:
  - `SUPERVISOR_VERSION=v2`일 때만 임베더와 SpecIndex를 만들었다.
  - v1의 `vector_store`는 `None`으로 뒀다.
  - v1의 ABSTAIN·위험 조치도 사람 승인이 필요했다.
    그래서 R2부터 RAG와 무관한 SQLite checkpointer만 공통으로 만들었다.
- **결과**: `EMBED_BACKEND=openai`이고 API key가 없는 환경에서도 v1/B1 검사가
  VLM Tool 1회로 동작한다. interactive 승인 대기는 서버 재시작 후 같은
  `thread_id`로 재개되며, 임베딩·RAG는 여전히 초기화하지 않는다.

### Phase 3 완료 검증

- **종단 검사**: `POST /chat`에서 `REJECT / scratch_neck / SCREW-04`.
  검증 2회 후 완료됐다.
- **SSE 순서**: `RouteDecided → PlanCreated → GateApplied → ObservationReady×5 →
  CrossCheckDone → SpecJudged → ReplanCheck → AnalysisDecided → 작성·승인·검증
  2회 → Finalize → FinalAnswer → [DONE]`.
- **RAG**: 4개 기준서 25개 조항 색인, 품목 필터 top-3,
  clause/full-text 조회가 통과했다.
- **회귀 검사**: fake 전체 테스트 `44 passed`.
  `import agent.graph`, `compileall`, `git diff --check`도 통과했다.
- **보호 범위**: 기준서·가이드와 실제 테스트 이미지는 열거나 수정하지 않았다.

## 8. Phase 4 서버·HITL·웹 UI

### 같은 관찰문이 보고서에 반복되는 문제

- **증상**: Observer의 5개 크롭이 같은 문장을 내면 같은 문장이 최대 5번 반복됐다.
- **왜 문제?**: 문장만 합치면 `source_call_id: str` 계약을 깨게 된다.
  어느 호출의 원문인지도 모호해진다.
- **조치**:
  - 관찰 하나에는 비어 있지 않은 `source_call_id` 문자열 하나만 허용했다.
  - 복수 ID는 ID별 항목으로 나눴다.
  - `(text, source_call_id)`가 모두 같은 경우만 중복으로 제거했다.
- **결과**: 각 관찰은 호출 원문 하나와 일대일로 연결됐다.
  `obs-full`, `obs-tl`처럼 호출이 다르면 같은 문장도 따로 보존했다.

### `ApprovalRequired`의 거짓 이벤트와 재작성 승인 재사용

- **증상**:
  - 승인이 필요 없는 보고서도 `required=false`인 `ApprovalRequired`를 냈다.
  - validate 뒤 새로 쓴 보고서가 이전 초안의 승인을 이어받을 수 있었다.
- **왜 문제?**: UI에는 불필요한 승인 카드가 생겼다.
  더 심각하게는 사람이 보지 않은 변경본이 과거 승인으로 통과하는 fail-open이었다.
  같은 초안의 이벤트를 다시 만들면 승인 카드도 중복됐다.
- **조치**:
  - required가 false이면 승인 이벤트를 생략했다.
  - 같은 초안은 `ApprovalRequired`를 한 번만 냈다.
  - 새 초안이 생기면 `approval`을 초기화했다.
  - 새 `interrupt()`와 새 `approval_id`로 다시 승인받게 했다.
- **결과**: 일반 보고서는 승인 이벤트 없이 완료됐다.
  승인 대상은 초안별로 한 번 `[PAUSED]`에서 멈췄다.
  재작성본은 과거 승인을 재사용하지 않고 `/chat/resume` 승인을 다시 받았다.

### 서버 재시작 직후 resume이 간헐적으로 상태를 못 찾는 문제

- **증상**: `__interrupt__`를 받은 직후 `astream`을 닫으면 중단 상태가
  저장되지 않는 경우가 있었다. 서버를 바로 재시작한 테스트에서 발견했다.
- **왜 문제?**: `AsyncSqliteSaver`는 비동기 쓰기를 사용한다.
  스트림 종료와 checkpoint 저장 취소가 경합하면 재시작 뒤 재개를 보장할 수 없다.
- **조치**:
  - 설치 버전은 `langgraph 1.2.16`, `langgraph-checkpoint-sqlite 3.1.2`였다.
  - 실제 API를 확인하고 `AsyncSqliteSaver.from_conn_string()`을 lifespan 전체에서 열었다.
  - 그래프 호출에 `durability="sync"`와 namespace가 적용된
    `configurable.thread_id`를 사용했다.
  - interrupt 뒤 반복기를 자연 종료하고 `[PAUSED]`를 보냈다.
- **결과**: 첫 lifespan에서 중단하고 SQLite 연결을 닫은 뒤, 두 번째 lifespan에서 같은
  namespace·thread ID와 현재 `approval_id`로 `/chat/resume`해
  `FinalAnswer → [DONE]`까지 완료했다. 중단 전 이벤트도 재전송되지 않았다.

### 같은 thread ID의 다음 요청이 이전 승인을 재사용하는 문제

- **증상**: 같은 thread ID로 새 검사를 시작하면 이전 요청의 `approval`, `report`,
  `validation_attempts`가 새 입력과 섞일 수 있었다.
- **왜 문제?**: 이전 `decision=approve`가 남으면 다음 위험 보고서가 HITL을 건너뛴다.
  재작성 횟수와 관찰 결과도 요청 사이에서 오염된다.
- **조치**: 새 그래프 입력에서 모든 요청 단위 필드를 명시적으로 초기화했다.
  승인·보고서·task·관찰·판정·검증·비용·분기 임시 결과가 대상이다.
  대화 history만 유지했다.
- **결과**: 첫 요청을 완료한 뒤 같은 thread ID로 두 번째 위험 검사를 보내도 과거 승인을
  재사용하지 않고 새 초안의 HITL에서 중단된다. 중단된 요청은 resume으로 완료되기 전까지
  세션 이력에 저장되지 않는다.

### 업로드·상태 관측 계약 누락

- **증상**:
  - 서버에 `/inspect`, `/chat/resume`이 없었다.
  - `/health`와 `/chat` metadata만으로 실행 설정과 판정 결과를 확인할 수 없었다.
  - 진행 이벤트에 실제 관찰문과 교차 판독 요약이 없었다.
- **왜 문제?**: 브라우저에서 업로드 스트림과 승인 재개를 쓸 수 없었다.
  로그만으로 실행 조건, `verdict`, `defect_type`, VLM 호출 수도 재현하기 어려웠다.
- **조치**:
  - 10MB 제한과 Pillow 검증을 통과한 이미지만
    `data/uploads/{thread_id}/{uuid}.png`에 저장했다.
  - 프록시는 multipart/JSON SSE를 그대로 중계했다.
  - `/health`에 `vlm_backend`, `embed_backend`, `hitl_mode`, `ablation`을 추가했다.
  - metadata에 판정·승인·호출 수를 추가했다.
  - Vue UI에 품목 선택, 썸네일, 진행 카드, 승인·반려 버튼을 연결했다.
  - 관찰·교차 이벤트에는 문장 필드를 추가했다.
- **결과**: 경로 조작 파일명, 비이미지, 10MB 초과 업로드를 거부하고 UUID PNG만 남긴다.
  `HITL_MODE=auto_approve`에서는 중단 없이 `[DONE]`까지 완료하며, interactive에서는
  승인 카드에서 재개할 수 있다.

### Phase 4 완료 검증

- **회귀 검사**: fake 전체 테스트 `53 passed in 12.08s`.
- **업로드**: `/inspect`의 UUID PNG 저장과 경로 조작·비이미지·10MB 초과 차단 확인.
- **HITL**: interactive 중단·승인·반려, auto-approve 무중단,
  SQLite 재시작 복원 확인.
- **정적 검사**: `import agent.graph`, Python compile,
  Vue 인라인 JavaScript 문법, `git diff --check` 통과.
- **보호 범위**: 기준서·가이드는 수정하지 않았다.
  테스트 이미지도 시각적으로 열지 않았다.

## 9. Phase 5 평가 파이프라인

### fake 결과가 실험 결과와 섞일 위험

- **증상**: 평가 실행기가 서버의 fake 기본값을 그대로 쓸 수 있었다.
  그러면 폴더 라벨을 읽은 결정론 결과가 실측처럼 저장된다.
- **왜 문제?**: fake VLM은 파일 경로의 정답 라벨을 사용한다.
  정확도 평가에 쓰면 결과 조작이고, 실제 실험 JSONL과 섞이면 집계도 오염된다.
- **조치**:
  - `LLM_BACKEND` 또는 `VLM_BACKEND`가 fake이면 즉시 종료했다.
  - `--dry-run`을 명시한 경우만 예외로 허용했다.
  - dry-run 파일에는 `_DRYRUN`을 붙이고 LangSmith 전송을 껐다.
  - Dataset 업로드는 `--push-dataset`이 있을 때만 실행했다.
  - 오타난 백엔드가 fake로 떨어지지 않게 지원 목록 밖의 LLM·VLM 값도 즉시 거부했다.
- **결과**: 허용된 `A/dev/limit 5` dry-run은
  `eval/results/A_dev_DRYRUN.jsonl`에 5줄을 저장했고, 실제 API·LangSmith 호출은
  발생하지 않았다.

### 토큰 사용량이 평가 행에 남지 않는 문제

- **증상**: 기존 상태는 `cost_usd`만 누적하고 `tokens`를 보존하지 않았다.
- **왜 문제?**: Phase 5 비용 표에는 조건별 평균 토큰이 필요하다.
  VLM·LLM·임베딩 중 하나라도 빠지면 비용이 실제보다 작게 계산된다.
- **조치**:
  - `AgentState`와 초기 상태에 `tokens`를 추가했다.
  - route·plan·judge·write의 LLM 사용량을 모았다.
  - Observer·Cross-Checker의 VLM 사용량을 모았다.
  - RAG query 임베딩 사용량까지 Spec-Judge에서 병합했다.
  - 최종 값을 서버 metadata와 평가 JSONL에 전달했다.
- **결과**: dry-run 5건 모두 1,882 tokens로 기록됐고, 행당 추정 비용은
  0.006217 USD, 누적은 0.031085 USD였다.

### 조건 비교를 독립 표본처럼 계산할 위험

- **증상**: B0·B1·A는 같은 이미지를 판정한다.
  그런데 조건별 정확도만 나란히 놓아 짝지은 정보를 버릴 수 있었다.
- **왜 문제?**: 같은 `id`의 Before/After 정·오답 변화를 검정해야 한다.
  정상·결함 비율이 다른 자료를 독립 표본처럼 비교하면 검정력을 잃는다.
- **조치**:
  - `eval/evaluators.py`에 이진·유형·확신도·인용·비용 지표를 구현했다.
  - scipy 없이 정확 이항분포를 쓰는 McNemar 양측 검정을 구현했다.
  - McNemar 결과는 전체·정상·결함 층으로 나눴다.
  - `eval/report.py`는 조건·품목·비용 표를 마크다운으로 만들게 했다.
  - `eval/human_check.py`는 B0/A 보고서를 조건 비노출로 섞게 했다.
- **결과**: 손계산 예제 `b=10, c=2`의 정확 p-value `0.03857421875`를 일치시켰고,
  평가 지표·리포트·블라인드 도구 테스트 16개가 통과했다.

### Phase 5 완료 검증

- **dry-run**: `A/dev/limit 5`의 5건이 전부 `PASS`.
  오류 0건, JSONL 스키마 20개 필드가 일치했다.
- **샘플 ID**: `screw-good-004`, `005`, `007`, `010`, `012`.
- **대상 테스트**: 평가 지표·출력·실행기 `20 passed`.
- **전체 회귀**: fake 전체 `73 passed in 11.38s`.
- **미실행 범위**: 실제 API, held-out, 다른 조건 실험은 실행하지 않았다.
- **보호 범위**: 기준서·가이드는 수정하지 않았다.
  테스트 이미지도 시각적으로 열지 않았다.

## 10. Phase 6 테스트·문서·실행 마감

### 업로드한 이미지가 fake 데모에서 판독 불가가 되는 문제

- **증상**: `/inspect`로 올린 이미지는 fake 데모에서 항상 `판독 불가`가 됐다.
  업로드 경로 `data/uploads/<thread_id>/<uuid>.png`에는 `/test/<label>/` 정보가 없기 때문이다.
- **왜 문제?**: 발표 리허설용 `demo_fake.sh`에서 실제 업로드 UI를 쓸 수 없었다.
  good과 결함 이미지 모두 판정하지 못해 종단 데모가 성립하지 않았다.
- **조치**:
  - `FakeVLM` 초기화 때 `eval/dataset.json`의 등록 파일을 바이트 SHA-256으로 색인했다.
  - 경로 라벨이 없을 때만 해시로 라벨을 찾게 했다.
  - Pillow가 PNG로 판별한 업로드는 원본 바이트를 보존했다.
  - JPEG 등 비-PNG만 PNG로 다시 인코딩했다.
  - 미등록 파일과 서로 다른 라벨에 같은 해시가 있는 파일은 `판독 불가`로 뒀다.
- **결과**: 실제 평가셋 192개의 해시 맵을 생성했다.
  good·결함 업로드 복사본은 원본과 같은 결과를 냈다.
  미등록 파일은 `판독 불가`를 유지했다. PNG 해시 보존과 JPEG 변환도 테스트했다.

### fake 테스트가 LangSmith 전송을 시도하는 문제

- **증상**: fake 테스트 73개는 통과했지만 종료 때 LangSmith 전송을 시도했다.
  로컬 `.env`의 추적 설정을 읽었기 때문이다. 네트워크 차단 환경에서는 오류도 출력됐다.
- **왜 문제?**: 세 백엔드가 fake이면 외부 호출 없이 재현돼야 한다.
  프로젝트 계약과 어긋났고, 성공한 테스트 로그에도 불필요한 오류가 섞였다.
- **조치**: 세 백엔드가 모두 fake이면 `server.py` import 단계에서
  `LANGCHAIN_TRACING_V2=false`, `LANGSMITH_TRACING=false`를 강제했다.
  실모델 조합의 추적 설정은 건드리지 않았다.
- **결과**: 같은 fake 명령에서 두 추적 플래그가 모두 `false`임을 확인했다.
  이후 전체 테스트는 외부 전송 오류 없이 종료됐다.

### 실행 스크립트가 이전 프로젝트 설정을 쓰는 문제

- **증상**:
  - 실모델 스크립트가 이전 Base 프로젝트의 `LANGCHAIN_PROJECT`를 덮어썼다.
  - 실행 전에 VLM·임베딩·HITL 설정과 이미지당 호출 수를 보여 주지 않았다.
  - dev·held-out 실측과 fake UI 리허설용 단일 명령도 없었다.
- **왜 문제?**: 추적이 다른 프로젝트와 섞일 수 있었다.
  모델·승인 모드·비용 범위를 확인하지 않고 평가를 시작할 수도 있었다.
  그러면 실행 조건을 재현하기 어렵다.
- **조치**:
  - OpenAI/Ollama 프로젝트명을 `grounded-visual-inspector-*`로 분리했다.
  - 세 백엔드 설정을 export하고 출력했다.
  - dev(B0, A)와 held-out(B0, B1, A)은 장수·조건별 비용 가드를 먼저 보여 줬다.
  - 실측 전에는 y/N 승인을 받게 했다.
  - fake 데모는 서버와 프런트를 함께 기동하게 했다.
- **결과**: 셸 구문과 실행 권한을 확인했고 dev 52장·최대 104건, held-out 140장·최대 420건이
  승인 전에 표시된다. 실제 유료 평가는 실행하지 않았다.

### Phase 6 완료 검증

- **전체 회귀**: fake 테스트 `120 passed in 16.94s`.
  외부 추적 전송 시도는 없었다.
- **MCP**: vision Tool·Resource·Prompt, 첫 Timeout 재시도, 기준서 25조항 계약을 실제 stdio
  subprocess 경로로 확인했다.
- **에이전트 계약**: analysis 합의 전 조합, validate 4규칙·임계값·3회 상한,
  HITL 3모드, good/결함 종단,
  B0/B1, 비동기 ASGI 업로드·중단·재개 계약을 보강했다.
- **문서**: ADR 4건의 필수 섹션과 실제 모듈 참조,
  README의 실행법·구조·실측 표 자리를 확인했다.
- **데모**: `demo_fake.sh`로 8100 백엔드와 8600 프런트엔드 health 응답을 확인하고 종료했다.
- **정적 검사**: `import agent.graph`, Python compile, 셸 구문,
  실행 권한, `git diff --check`가 통과했다.
- **보호 범위**: 기준서·가이드를 수정하지 않았다.
  테스트 이미지도 시각적으로 열지 않았다.

## 11. 보정 Phase R0~R10을 재설계한 분석

### 왜 R0~R10을 다시 짰나

- **증상**: Phase 1~6 완료 기준과 fake 회귀 테스트 120개는 통과했다.
  하지만 아래 항목은 아직 증명되지 않았다.
  - 실모델 B0/B1/A 전체 실행과 heldout 결과
  - H1~H4 비교가 실제로 격리됐는지
  - 오류·누락·부분 실행까지 포함한 보수적 성능
  - 기준서·데이터·코드·모델을 평가 전에 고정했다는 증거
  - 최종 보고서·발표자료·원격 저장소 반영
- **왜 문제?**: 구현 완료와 평가 결과의 신뢰성은 다른 문제다.
  결과를 먼저 본 뒤 기준·임계값·통계를 고치면 결과에 맞춘 것으로 보일 수 있다.
  fail-open 상태에서 정확도만 내면 판독 불가·근거 누락·실행 실패가 분모에서 빠진다.
- **조치**: 기능 구현 순서가 아니라 다음 의존성 순서로 다시 짰다.
  - 평가 오염 차단 → 안전 계약 → 집계 완전성 → 통계 사전 고정
  - dev 리허설 → 동결 → heldout
- **결과**: 보정 작업을 R0~R10으로 나눴다.
  각 Phase는 독립 검증 뒤 멈추고, R7 동결 전에는 heldout을 열지 않게 했다.

### 어떤 분석을 했나

정적·정량 근거를 아래 순서로 대조했다.
테스트 이미지는 열지 않았다. 파일 메타데이터·폴더 라벨·문서·코드만 사용했다.

1. **요구사항 추적 분석**
   - **확인**: 공식 평가 항목을 구현·테스트·문서·발표 산출물과 연결했다.
   - **기준**: `docs/design.md`, `docs/plan.md`를 SSOT로 삼았다.
     외부 사본, README, 결과 폴더, ADR, 발표 산출물도 확인했다.
   - **발견**: “Phase 구현 완료”와 “가설 검증·제출 완료”는 따로 봐야 했다.

2. **설계–구현–테스트 정합성 분석**
   - **확인**: 샘플 수, Chroma, API·SSE 이름, UI crop, LangSmith,
     비용 산식을 실제 코드와 줄 단위로 대조했다.
   - **발견**:
     - 실제 분포 `192 = dev 52 + heldout 140`과 문서의 `60/120` 근사값이 달랐다.
     - 실제 구현은 `SpecIndex + NumPy cosine`인데 문서에는 Chroma로 적힌 drift가 있었다.

3. **안전 데이터 흐름과 fail-open 분석**
   - **확인**: `analysis → write → HITL → validate → finalize` 흐름을 추적했다.
     `None`, 빈 근거, 재작성, 승인 재사용 처리도 확인했다.
   - **최우선 위험**:
     - 판독 불가가 `PASS/high`에 합쳐지는 경로
     - 음성 관찰 영역이 양성 합의에 쓰이는 경로
     - 빈 근거·모순 action이 validator를 통과하는 경로
     - 재작성 보고서가 기존 승인을 재사용하는 경로

4. **실험 타당성과 인과 격리 분석**
   - **확인**: 가설 문장, 비교군, 코드에서 실제로 달라지는 요소를 대조했다.
   - **발견**:
     - B0/A는 모델 강도·호출 수·crop·RAG·validate가 모두 다르다.
       따라서 전체 시스템 비교로만 해석했다.
     - B1/A도 RAG 단독 효과를 말할 수 없어 인과 주장을 제한했다.
     - validate 효과는 같은 validate 직전 상태를 A/A-V로 나눠야 했다.

5. **통계·표본·블라인드 타당성 분석**
   - **발견**:
     - H1의 이진 정오 McNemar와 H2의 유형 macro-F1 주장이 맞지 않았다.
     - 약속한 95% 신뢰구간이 없었다.
     - 기존 H3는 B0/A에서 서로 다른 15개를 뽑고 조건 단서도 노출했다.
       paired blind가 아니었다.
     - 15개 중 1건은 6.67%다. 5% 이하 주장을 검증하기에는 해상도가 부족했다.

6. **집계 편향과 재현성 분석**
   - **집계 위험**:
     - 예외 행이 성능 분모에서 빠졌다.
     - McNemar가 성공 ID 교집합만 사용했다.
     - 예산 중단·`--limit`·`--items` 결과도 보고서에 들어갈 수 있었다.
   - **재현성 위험**: 결과 덮어쓰기, fake embedding, 모델·Git SHA·기준서·데이터셋
     해시 누락을 동결 증거의 빈틈으로 분류했다.

### 테스트 설계 기법

R0 및 후속 보정 Phase의 테스트는 한 가지 기법이 아니라 다음을 조합한다.

| 기법 | 이 프로젝트의 적용 |
|---|---|
| 요구사항 추적 | 공식 평가 항목 → 구현 파일 → 테스트 → 보고서·발표 증거 연결 |
| 동등 분할 | dev/heldout, fake/real, 정상/결함, 완전/부분 실행 분리 |
| 결정표 테스트 | split·backend·dry-run·final-run·manifest 조합별 허용/거부 |
| 경계값 분석 | heldout 139/140건, 사람 채점 0/1건, low 표본 최소 건수 |
| 부정·fail-closed 테스트 | manifest 누락·오류, 잘못된 해시, fake backend, 빈 근거를 명시적 거부 |
| 결함 주입 | 오류 행, 누락·중복 ID, timeout, 비용 중단을 넣어 집계가 보수적으로 실패하는지 확인 |
| 불변조건·회귀 | Base 시그니처·SSE·events 누적, dev 데모 선택, fake 전체 테스트 유지 |

따라서 R0의 `139/140`은 경계값 분석이지만, R0 전체는 동등 분할·결정표·
부정 테스트·불변조건 검증이 함께 쓰인 구조다.

### 결과: Phase 순서로 변환

| 분석에서 찾은 위험 | 보정 Phase | 먼저 해야 하는 이유 |
|---|---|---|
| 데모·직접 CLI의 heldout 누출 | R0 방화벽 | 다른 보정 중에 평가 데이터를 보지 않게 먼저 차단 |
| 판독 불가·재검사 표현 불일치 | R1 계약 | 공통 스키마 결정 전에 코드를 고치면 재작업 발생 |
| PASS/high, 합의 영역, 빈 근거, 승인 재사용 | R2 안전성 | 성능 측정 전에 판정 의미와 안전 계약 보장 |
| 오류 제외·ID 교집합·partial 결과 | R3 평가 완전성 | 좋은 샘플만 남는 성능 부풀리기 차단 |
| H1~H4 비교군·통계·블라인드 불일치 | R4 실험 재설계 | 결과를 보기 전에 주 지표와 성공·보류 규칙 고정 |
| 결과 덮어쓰기·해시·모델·비용 추적 누락 | R5 재현성·문서 | dev 리허설과 동결에 쓸 불변 실행 단위 확립 |
| 실모델 케이스와 비용 미검증 | R6 dev 리허설 | heldout을 쓰지 않고 실행·임계값·예산 문제 해결 |
| 평가 후 수정 가능성 | R7 동결 | 코드·기준·통계·설정·Git SHA를 최종 실행 전에 고정 |
| 최종 140건 실측 | R8 heldout | 동결 통과 후에만 단회 평가 |
| 사람 채점·가설 판정·결과 일치 | R9 결과 보고 | 수수료·오류·CI·p-value·한계를 포함한 증거 생성 |
| README TBD·발표자료·원격 반영 누락 | R10 제출 | clean clone과 GitHub 링크까지 평가자 관점으로 최종 확인 |

이 순서는 구현 편의로 정하지 않았다.
증거 오염 가능성과 다음 Phase의 선행 조건을 기준으로 정했다.
각 Phase는 독립 테스트 뒤 멈춘다. heldout은 R7 동결 게이트 통과 전에 열지 않는다.

### 각 보정 Phase의 채점자 관점 종료 게이트

- **증상**: 기능 완료와 테스트 통과만으로는 채점 근거가 충분하지 않았다.
- **왜 문제?**: fail-open, 오류·누락 제외 편향, heldout 오염, 문서 drift,
  실측하지 않은 주장이 남을 수 있었다.
- **조치**:
  - 공식 요구와 design SSOT를 변경 파일·테스트·출력 증거에 다시 연결한다.
  - 코드·문서·테스트·UI 정합성을 확인한다.
  - 발표에서 방어할 수 있는 근거인지 따로 검토한다.
  - Phase 범위 안의 문제는 바로 고치고 재검증한다.
  - 후속 Phase가 필요한 위험은 담당 Phase와 검증 공백을 적는다.
- **결과**: 종료 판정을 아래 세 가지로 고정했다.
  - `GO`: 완료 기준과 채점 근거를 모두 충족했고 알려진 차단 위험이 없음.
  - `GO WITH RISKS`: Phase 완료 기준은 충족했지만 후속 Phase에서 해소할 위험이 명시됨.
  - `NO-GO`: 요구 위반·검증 실패·증거 오염 가능성이 남아 다음 Phase로 진행하면 안 됨.

## 12. R0 heldout 오염 방화벽

### 데모가 heldout 경로를 출력하는 문제

- **증상**
  - `sh/demo_fake.sh`가 split 구분 없이 첫 good·결함 샘플을 골랐다.
  - 그 결과 `screw-good-000`, `screw-scratch_head-000`이 선택됐다.
  - 두 샘플은 모두 heldout이었다.

- **왜 문제?**
  - 리허설 도중 이미지를 열면 heldout을 미리 본 것이 된다.
  - 그러면 최종 평가의 독립성을 설명하기 어렵다.

- **조치**
  - 데모 샘플 선택을 `eval.preflight.select_dev_demo_samples()`로 분리했다.
  - `split == "dev"`인 good·결함만 선택한다.
  - dev 샘플이 없으면 heldout으로 대체하지 않고 종료한다.

- **결과**
  - 현재 데모 ID는 `screw-good-004`, `screw-scratch_head-004`다.
  - 둘 다 dev임을 메타데이터로 확인했다.
  - 실제 이미지는 열지 않았다.
  - 기존에 출력된 heldout 두 건도 업로드·열람하지 않았다.
  - heldout 오염은 없었고 split과 seed는 그대로 유지했다.

### heldout 부분·fake 실행을 직접 호출로 우회할 수 있는 문제

- **증상**
  - `eval/run_eval.py`를 직접 실행하면 heldout에서도 다음 옵션을 쓸 수 있었다.
    - `--dry-run`
    - `--items`
    - `--limit`
  - fake embedding과 동결 전 실행을 막는 공통 가드도 없었다.

- **왜 문제?**
  - shell의 `y/N` 확인은 Python 직접 호출로 우회할 수 있다.
  - 부분 표본이나 fake 결과가 최종 실측처럼 남을 수 있다.

- **조치**
  - 모든 진입점이 공유하는 `eval/preflight.py`를 추가했다.
  - heldout에서는 다음을 거부한다.
    - fake LLM·VLM·embedding
    - `--dry-run`
    - `--items`, `--limit`
  - heldout 실행에는 `--final-run --freeze-manifest`를 모두 요구한다.
  - R0 최소 manifest는 다음을 검증한다.
    - `status=frozen`
    - `split=heldout`
    - 표본 140건
    - `dataset.json` SHA-256
  - 가드는 샘플 로드·LangSmith 업로드·결과 폴더 생성보다 먼저 실행된다.

- **결과**
  - `heldout --dry-run`, `heldout --limit 1`은 부수 효과 전에 종료됐다.
  - 결과 폴더도 만들어지지 않았다.
  - 대상 테스트 `26 passed`, fake 전체 `142 passed`.
  - 최종 manifest가 없으므로 R7 전까지 heldout은 계속 잠긴다.

## 13. R1 판독 불가·재검사 계약 고정

### 판독 불가가 PASS/high로 합쳐지는 문제

- **증상**
  - 모든 `anomaly_seen`가 `None`이어도 Observer가 정상으로 집계될 수 있었다.
  - 빈 Cross-Checker 결과도 `no_cross` ablation과 구분하지 않았다.
  - 그 결과 판독 불가 상태가 `PASS/high`로 끝날 수 있었다.
  - `data/_TASK.md`에는 `재검사`가 있었지만 report 스키마는 이 조치를 허용하지 않았다.

- **왜 문제?**
  - 읽지 못한 제품을 합격시키는 fail-open이다.
  - 판독 불가를 `REJECT`로 처리해도 안전하지 않다. 결함 정답으로 채점돼 정확도가 부풀 수 있다.
  - 기존 집계에 합성 3건을 넣었을 때 ABSTAIN이 빠져 `n=2, accuracy=1.0`이 나왔다.

- **결정**

  | 대안 | 판단 |
  |---|---|
  | `PASS` 유지 | 판독 불가를 합격시키므로 제외 |
  | `REJECT+assessable` | 운영 보류와 결함 판정이 섞여 제외 |
  | `ABSTAIN` 추가 | 재검사와 평가 실패를 동시에 표현할 수 있어 선택 |

- **조치**
  - 공통 판정을 `PASS|REJECT|ABSTAIN`으로 고정했다.
  - ABSTAIN은 다음 값을 강제한다.
    - `defect_type=null`
    - `confidence=low`
    - 빈 조항
    - `actions=["재검사"]`
    - 구조화된 사유
    - HITL
  - Observer 집계는 다음처럼 고정했다.
    - 양성 1건 이상: `O+`
    - full 정상 또는 서로 다른 정상 사분면 3개: `O-`
    - 나머지: `O?`
  - center 재계획 1회는 `O?`에서만 허용한다.
  - 정상·판독 불가 region은 양성 합의에서 뺀다.

- **결과**
  - Master, design, ADR-0002, 각 폴더 Task 문서가 같은 계약을 가리킨다.
  - ABSTAIN은 주 평가의 전체 N에 오답으로 남는다.
  - coverage와 selective 지표는 분리해 보고한다.
  - R1은 계약만 고정했다. 코드는 R2, 집계는 R3에서 구현하도록 나눴다.
  - 기준서·가이드·`type_map`·평가 데이터는 수정하지 않았다.
  - 테스트 이미지는 열지 않았다.
  - `git diff --check`, `import agent.graph`, fake 전체 `142 passed in 15.57s`.


## 14. R2 판정·보고서·HITL 안전 계약 구현

### R2 한눈에 보기

| 핵심 위험 | 막은 방법 |
|---|---|
| 판독 불가가 PASS로 나감 | `ABSTAIN/low`로 fail-closed |
| 모델이 lot·요약·조치를 바꿈 | state·원문·가이드로 재구성 |
| 사람이 보지 않은 재작성본이 통과 | 새 초안마다 새 `approval_id` |
| pending thread가 덮어써짐 | 새 요청 `409` + namespace 분리 |
| 동시 요청이 둘 다 통과 | thread별 single-flight |
| 잘못된 thread ID로 재개 불가 | 네 진입점의 ID 검증 통일 |
| 불완전한 Spec-Judge 응답이 `V0`로 변환 | 의미를 보충하지 않고 `V?`로 닫기 |
| 스키마 밖 JSON이 남음 | 허용 필드만 보존 |

> 결론: R2의 핵심은 “애매하면 통과”가 아니라 “애매하면 멈추고 다시 확인”이다.

### 판독 불가와 합의 경계가 fail-open이 되는 문제

- **증상**
  - 빈 관찰과 `anomaly_seen=null`이 정상으로 합쳐질 수 있었다.
  - “판별하기 어려움”, “unable to assess” 같은 표현도 `O-`·`C-`가 될 수 있었다.
  - UI가 합의 여부를 따로 계산해 백엔드와 다른 결과를 보여줄 수 있었다.

- **왜 문제?**
  - 판독하지 못한 제품이 `PASS/high`로 나갈 수 있다.
  - 서버와 UI가 서로 다른 판정을 보여줄 수 있다.
  - 한국어만 검사하면 영어 불확실 표현을 놓친다.

- **조치**
  - R1 결정표를 코드 규칙으로 옮겼다.
  - `O+|O-|O?`, `C+|C-|C?|C-skip`, `V+|V0|V?` 전 조합을 테스트했다.
  - 한국어·영어 판독 불가 표현은 `agent/readability.py`에서 한 번만 판단한다.
  - vision, v1 baseline, analysis가 같은 규칙을 쓴다.
  - UI는 `AnalysisDecided.agree`를 그대로 표시한다.

- **결과**
  - 판독 불가는 `ABSTAIN/low` 또는 `O?/C?`로 닫힌다.
  - 정상·판독 불가 region은 양성 합의에서 뺀다.
  - 결정표·표현 변형·재계획 경계 테스트가 통과했다.

### 과거 Phase 4 작업 지시와 현재 공통 계약이 충돌하는 문제

- **증상**
  - `_tasks/phase4.md`에 현재 계약과 다른 과거 지시가 남아 있었다.
    - `source_call_id`를 리스트로 합치기
    - 재작성 후 `ApprovalRequired`를 다시 내지 않기

- **왜 문제?**
  - 현재 계약은 관찰별 단일 `source_call_id`를 쓴다.
  - 새 초안은 새로 승인받아야 한다.
  - 과거 지시를 적용하면 두 규칙이 모두 깨진다.

- **조치**
  - `_tasks/phase4.md`는 과거 작업 이력이므로 수정하지 않았다.
  - 현재 `docs/design.md` SSOT와 R1 공통 계약이 과거 지시를 대체한다고 명시했다.
  - Phase 4 회고 설명만 현재 계약에 맞게 고쳤다.

- **결과**
  - 관찰은 `(text, source_call_id)` 단위로 남는다.
  - 새 초안은 항상 새로 승인받는다.
  - 과거 지시는 감사 이력으로 보존했다.

### 작성 모델의 보고서 필드를 그대로 신뢰하는 문제

- **증상**
  - 작성 모델이 `source_call_id`, lot, summary, action을 임의로 바꿀 수 있었다.
  - `draft`도 보정된 report가 아니라 raw 응답이었다.

- **왜 문제?**
  - 관찰문이 맞아도 lot·요약·조치에 환각이 남을 수 있다.
  - 모델이 위험한 조치를 임의로 약화할 수 있다.

- **조치**
  - 관찰당 `source_call_id` 하나만 허용했다.
  - ID는 실제 원문과 함께 대조한다.
  - lot은 state 값으로 덮어쓴다.
  - summary는 검증된 관찰·판정·조항·조치로만 다시 만든다.
  - `draft=json.dumps(report)`를 강제했다.
  - action은 결함별 가이드의 허용 목록 안에서만 고른다.
  - 무효 action은 `라인정지 > 폐기 > 재작업` 순으로 보수 대체한다.
  - 가이드가 없으면 `ABSTAIN/validation_failure`로 전환한다.

- **결과**
  - 위조된 source ID·lot·summary·action은 validator가 거부한다.
  - 14개 결함 가이드의 action을 계약 테스트로 고정했다.
  - UI에서 스트리밍 작성문은 `검증 전 초안(확정 아님)`으로 보인다.

### baseline ABSTAIN과 반려된 위험 조치가 안전하게 재승인되지 않는 문제

- **증상**
  - v1 interactive가 승인 전에도 `[DONE]`과 최종 보고서를 내보낼 수 있었다.
  - 모든 REJECT action은 `재작업`으로 고정됐다.
  - 실행 모드 오타가 v1 fallback이나 자동 승인으로 연결될 수 있었다.

- **왜 문제?**
  - `ABSTAIN은 항상 HITL`이라는 계약을 깨뜨린다.
  - `broken_large` 같은 고위험 결함이 사람 확인을 피할 수 있다.
  - 반려한 초안이 과거 승인으로 다시 완료될 수 있다.
  - 설정 오타로 약한 기준선이 실행되는 것도 fail-open이다.

- **조치**
  - v1을 `baseline → hitl → finalize` LangGraph로 바꿘다.
  - SQLite checkpointer와 기존 `hitl_node`를 재사용했다.
  - 승인 전에는 `final`을 만들지 않는다.
  - REJECT action은 고정 가이드에서 고른다.
  - 폐기·라인정지는 HITL을 거친다.
  - REJECT를 반려하면 다음 새 초안을 만든다.
    - `ABSTAIN/low`
    - `validation_failure`
    - `actions=["재검사"]`
  - 새 초안은 approval을 초기화하고 다시 승인받는다.
  - auto-reject도 같은 재작성 계약을 쓴다.
  - 설정 enum 오타는 MCP 시작 전에 거부한다.

- **결과**
  - ABSTAIN과 `broken_large`는 interactive에서 `[PAUSED]`로 멈춘다.
  - 서버 재시작 후에도 같은 `thread_id`로 재개할 수 있다.
  - REJECT 반려 후에는 새 `approval_id`로 다시 멈춘다.
  - auto 모드도 초안별 ApprovalRequired/Resolved 기록을 남긴다.
  - v1은 checkpointer만 쓰고 RAG·embedding은 초기화하지 않는다.

### pending thread 덮어쓰기·stale 승인·조건 간 checkpoint 오염 위험

- **증상**
  - 승인 대기 thread에 새 요청을 보내면 pending state를 덮어쓸 수 있었다.
  - 이전 `approval_id`를 새 interrupt에 재사용할 수 있었다.
  - v2·v1:B0·v1:B1이 같은 checkpoint key를 쓸 수 있었다.

- **왜 문제?**
  - 사용자가 본 보고서와 실제 재개되는 state가 달라질 수 있다.
  - Before/After 간 관찰·판정·승인이 섞여 평가까지 오염된다.

- **조치**
  - pending interrupt가 있으면 새 요청을 파일 저장 전에 `409`로 거부한다.
  - `/chat/resume`은 현재 `approval_id`를 필수로 받는다.
  - pending ID와 exact-match할 때만 재개한다.
  - 새 초안은 새 ID를 받는다.
  - checkpoint key에 `v2`, `v1:B0`, `v1:B1` namespace를 붙였다.

- **결과**
  - paused thread를 새 요청으로 덮어쓸 수 없다.
  - 누락 ID는 `422`, stale·다른 조건 ID는 `409`다.
  - 같은 namespace의 `thread_id`+현재 `approval_id`만 서버 재시작 후 재개된다.

### 동시 요청이 pending·approval 검사를 함께 통과하는 문제

- **증상**
  - 같은 thread에 신규 SSE 두 개를 동시에 보내면 둘 다 검사를 통과할 수 있었다.
  - 같은 `approval_id`를 동시에 resume해도 같았다.
  - 검사와 stream 시작 사이의 TOCTOU였다.

- **왜 문제?**
  - 한 요청이 다른 checkpoint를 덮어쓸 수 있다.
  - 한 번만 쓰어야 할 승인이 두 번 처리될 수 있다.

- **조치**
  - `namespace + thread_id`를 키로 process 내 single-flight 예약을 두었다.
  - `/chat`, `/chat/stream`, `/inspect`, `/chat/resume`은 검사 전에 예약한다.
  - SSE 예약은 iterator의 종료·예외·취소 시점에 풀린다.

- **결과**
  - 동시 요청은 하나만 `200`, 나머지는 `409`다.
  - stream이 끝나면 예약이 풀려 다음 정상 요청을 막지 않는다.

### 테스트가 heldout 실파일을 참조하는 문제

- **증상**: 서버·재시작·그래프 테스트가 실제
  `data/mvtec/*/test/*`의 heldout ID를 가리켰다.
- **왜 문제?**:
  - 사람이 이미지를 열지 않아도 R7 전 heldout을 쓰지 않는다는 설명과 어긋났다.
  - 실수로 실제 decoder가 호출될 여지도 남았다.
- **조치**:
  - 테스트 이미지는 `tmp_path`에서 만든 합성 PNG fixture로 바꿨다.
  - 평가 runner 예시 ID는 metadata로 확인한 dev ID `004`로 바꿨다.
  - 이미지의 시각 내용은 열지 않았다.
- **결과**:
  - R2 테스트는 실제 heldout 파일을 읽지 않는다.
  - 기준서·조치 가이드·`eval/type_map.json`·평가 데이터·seed는 수정하지 않았다.

### `/chat`만 잘못된 thread ID를 허용하는 문제

- **증상**: `/chat/stream`, `/inspect`, `/chat/resume`은 공통 정규식으로
  `thread_id`를 검사했다. `POST /chat`만 요청 값을 checkpoint key에 그대로 썼다.
- **왜 문제?**:
  - `bad:id`로 interrupt를 만들면 resume은 같은 값을 422로 거부한다.
  - 그러면 영구적으로 재개할 수 없는 checkpoint가 남는다.
  - 빈 문자열의 default 해석도 엔드포인트마다 달라질 수 있었다.
- **조치**:
  - `/chat`도 그래프 호출 전에 `_validated_thread_id()`를 통과시켰다.
  - 명시적인 빈 `thread_id`·`session_id`는 누락과 구분해 422로 거부했다.
- **결과**:
  - 빈 값, 콜론, `.`·`..`, 경로 탈출 문자열을 그래프 실행 전에 거부했다.
  - endpoint 회귀 테스트를 추가했다.
  - 네 진입점이 같은 외부 ID 계약을 사용한다.

### ID 없는 업로드가 공용 default checkpoint를 쓰는 문제

- **증상**: `/inspect`의 `thread_id`와 호환용 `session_id`가 모두 optional이었다.
  둘 다 빠지면 일반 채팅과 같은 `default`로 변환됐다.
- **왜 문제?**:
  - 무관한 업로드가 `v2:default` checkpoint를 공유할 수 있었다.
  - 첫 요청의 승인 대기가 다음 업로드를 막거나 상태를 섞을 수 있었다.
  - 응답은 선택된 암묵 ID도 알려주지 않았다.
- **조치**:
  - 일반 `/chat`과 `/chat/stream`의 기본 ID는 유지했다.
  - `/inspect`만 두 ID 중 하나를 필수로 요구했다.
  - 기존 클라이언트의 `session_id` 단독 사용은 계속 허용했다.
- **결과**:
  - 두 ID가 모두 없으면 파일 저장·그래프 실행 전에 422가 된다.
  - legacy `session_id`는 같은 검증을 거쳐 명시적 thread로 사용된다.

### Spec-Judge의 불완전한 응답이 정상 판정으로 보정되는 문제

- **증상**:
  - OpenAI/Ollama 응답 정규화가 누락·중첩·알 수 없는 조항을 빈 위반 목록으로 바꿀 수 있었다.
  - 리스트 안의 비객체 값에서는 예외가 날 수 있었다.
  - 모델이 `V0`를 말하지 않아도 코드가 위반 없음으로 해석할 여지가 있었다.
- **왜 문제?**: 불완전한 judge 응답을 `V0`로 바꾸면 fail-open이다.
  결함 관찰이 있어도 `PASS` 또는 근거 없는 확정 판정으로 이어질 수 있다.
- **조치**:
  - 백엔드 정규화는 원문 구조를 보존하고 의미를 보충하지 않게 했다.
  - 공통 validator가 필드 형태와 비어 있지 않은 rationale을 검사했다.
  - 해당 품목의 실제 조항 ID인지 검사했다.
  - 원문 defect type과 조항 defect type의 일치를 검사했다.
  - 누락·중첩·미지원 조항·모순은 `V?`로 보냈다.
  - 실제 조항과 미지원 조항이 섞여도 일부만 골라 쓰지 않았다.
  - 명시적인 `violations=[]`, `defect_type=null`만 `V0`로 인정했다.
- **결과**:
  - 두 실백엔드 정규화와 공통 validator에 malformed·unknown·contradiction·non-dict
    회귀 테스트를 추가했다.
  - `V?`는 결정표에서 `ABSTAIN/low`, `validation_failure`, `재검사`로 fail-closed된다.

### baseline의 누락된 결함 유형이 명시적 정상처럼 취급되는 문제

- **증상**: baseline JSON의 아래 값이 모두 Python `None`과 같은 정상 응답으로
  합쳐질 수 있었다.
  - `defect_type` 키 누락
  - `"none"`, `"null"`, `"good"`, `"정상"` 문자열
- **왜 문제?**: 스키마 불완전과 명시적 정상 판정을 구분하지 못했다.
  판독 실패가 `PASS`가 될 수 있었다.
- **조치**:
  - JSON `null`만 명시적 정상으로 허용했다.
  - 키 누락·빈 문자열·별칭 문자열·잘못된 타입은 모두
    `ABSTAIN/validation_failure`로 처리했다.
  - vision backend와 v1 supervisor의 정규화를 같은 계약으로 맞췄다.
- **결과**: 명시적 `null`의 PASS와 누락·별칭의 ABSTAIN을
  각각 회귀 테스트로 고정했다.

### 보고서의 비표준 JSON 값과 추가 필드가 최종 응답에 남는 문제

- **증상**:
  - Python `json.loads()`가 `NaN`·`Infinity`를 허용했다.
  - 모델이 만든 top-level·observation 추가 키가 최종 보고서에 남을 수 있었다.
- **왜 문제?**:
  - 브라우저 `JSON.parse`와 저장 결과가 달라질 수 있었다.
  - 스키마 밖 데이터가 검증을 우회해 감사 가능한 보고서 계약을 깨뜨렸다.
- **조치**:
  - 작성 응답 파싱에서 비표준 상수를 거부했다.
  - top-level과 observation은 허용 필드만 남게 정규화했다.
  - validator도 추가 키와 모든 중첩 비유한 수를 독립적으로 거부했다.
- **결과**:
  - `NaN`·`Infinity` 파싱을 회귀 테스트로 고정했다.
  - 추가 top-level/observation 필드와 중첩 비유한 수도 회귀 테스트로 고정했다.
  - 최종 보고서에는 공통 스키마 필드만 남는다.

### 재작성된 초안에 이전 승인 결정이 남는 문제

- **증상**: 승인·반려 또는 validate 반려 뒤 새 초안이 만들어져도
  이전 승인 결정으로 완료될 수 있었다.
  새 초안의 action만 보면 HITL 조건이 사라질 수 있기 때문이다.
- **왜 문제?**: 사용자가 본 보고서와 최종 보고서가 달라진다.
  승인 범위를 벗어나는 동작이다.
- **조치**:
  - 새 초안을 만들 때 이전 초안의 승인 결정·승인 필요성을 추적했다.
  - 해당하면 `reapproval_required`를 세웠다.
  - hitl은 새 초안의 위험도와 무관하게 다시 interrupt했다.
  - 승인 결정과 플래그는 초안 단위로 초기화했다.
- **결과**:
  - approve 뒤 validate 반려 경로가 새 `ApprovalRequired`에서 멈췄다.
  - reject 뒤 안전 재작성 경로도 새 `ApprovalRequired`에서 멈췄다.
  - 과거 결정을 자동 재사용하지 않는 테스트가 통과했다.

### 채점자 관점 잔여 위험과 후속 Phase

- **증상**:
  - `no_crops` ablation도 `O?`이면 center 재계획을 실행했다.
  - grounding 부분 일치에는 정규화 문자열의 최소 길이가 없었다.
  - 비용·토큰 usage가 유한한 0 이상 수인지 최종 집계 전에 검사하지 않았다.
  - evaluator는 오류·ABSTAIN을 일부 지표에서 제외했다.
  - McNemar는 성공 ID 교집합만 사용했다.
- **왜 문제?**:
  - center 재계획은 R4의 “full만” 인과 격리를 흐릴 수 있다.
  - 한 글자 인용처럼 설명력이 낮은 보고서를 허용할 수 있다.
  - 음수나 `NaN` usage는 R5 비용 표를 왜곡할 수 있다.
  - 오류·누락 제외는 최종 성능을 실제보다 높게 보이게 한다.
- **조치**:
  - `no_crops`는 heldout 결과를 보기 전에 R4 실험 재설계에서 결정하기로 했다.
  - grounding 최소 길이는 R6 dev 임계값 리허설에서 결정하기로 했다.
  - usage 유효성은 R5 비용·집계 계약에서 막기로 했다.
  - evaluator의 분모·짝지음은 바로 다음 R3에서 보수적으로 고치기로 했다.
- **결과**:
  - 현재 R2 안전 계약을 막는 fail-open으로 보지는 않았다.
  - R3/R4/R5/R6 종료 게이트에서 테스트와 문서를 함께 고정하기로 했다.
  - single-flight는 단일 worker 계약에는 맞지만 process-local이다.
    multi-worker 배포에는 DB 기반 상호 배제가 따로 필요하다.
  - pending 승인 ID 조회 UX는 후속 P2로 남겼다.

### R2 완료 검증

- **증상**: R2는 판정표, 두 supervisor, 보고서, HITL, 재개 API, UI까지 바뀌었다.
  대상 테스트만 통과해도 연동 회귀가 남을 수 있었다.
- **왜 문제?**: 개별 안전 규칙만 맞아서는 부족하다.
  그래프·SQLite checkpoint·SSE 직렬화가 함께 동작해야 발표와 평가에서 재현할 수 있다.
- **조치**:
  - heldout과 실제 테스트 이미지는 열지 않았다.
  - fake LLM/VLM/embedding으로 전체 pytest를 실행했다.
  - compileall, 핵심 import, `git diff --check`를 실행했다.
  - 기준서·가이드·`type_map.json`·dataset·lots·역사 지시 파일의 diff도 따로 확인했다.
- **결과**:
  - 전체 회귀: `338 passed in 29.67s`.
  - compileall/import/diff 검사도 통과했다.
  - 기준서·가이드·평가 데이터·seed는 바뀌지 않았다.
  - R7 전 heldout 잠금도 유지됐다.
  - 독립 채점자 감사에서 동시 신규 요청과 동일 승인 ID 재사용을 실제 ASGI로 재현했다.
  - 결과는 각각 `200/409`, 종료 후 예약 해제, ID 없는 `/inspect`의 422였다.
  - R2 범위의 잔존 P0/P1은 없다.
  - 최종 판정은 `GO WITH RISKS`다.
  - R3 집계 보정 전에는 전체 성능을 주장하지 않는 조건으로 다음 Phase 진행이 가능하다.

## 15. R3 평가 완전성·보수 집계

### R3 한눈에 보기

| 기존 문제 | R3 규칙 |
|---|---|
| 성공한 행만 정확도에 포함 | 사전 등록 전체 ID를 분모로 고정 |
| 오류·누락·ABSTAIN이 사라짐 | 모두 주 지표의 오답으로 유지 |
| 자동 판정만의 성능이 전체 성능처럼 보임 | `selective_*`+ coverage로 분리 |
| 부분 실행이 완전 실행처럼 보임 | expected/result/missing N과 `결론 금지` 표시 |
| ID·메타데이터 오염을 조용히 제외 | 즉시 hard-fail |

> 결론: R3에서는 “못 한 샘플”도 평가 결과의 일부다.

### 오류·ABSTAIN·누락이 주 지표 분모에서 빠지는 문제

- **증상**:
  - 기존 `binary_metrics`는 `error` 행을 제외했다.
  - verdict가 `PASS|REJECT`가 아닌 행도 제외했다.
  - 결과 파일에 없는 샘플은 존재 자체를 알 수 없었다.
- **왜 문제?**: 실패·판독 불가·미처리 샘플이 많아질수록
  성공한 행만의 정확도가 높아 보이는 생존자 편향이 생긴다.
- **조치**:
  - `eval/dataset.json`의 split 전체 ID와 정답을 canonical roster로 고정했다.
  - 정상은 clean `PASS`, 결함은 clean `REJECT`만 주 지표 정답으로 인정했다.
  - `ABSTAIN`·error·missing·invalid는 전체 N에 남기고 오답으로 채점했다.
  - clean `PASS|REJECT`만의 지표는 `selective_*`로 분리했다.
  - selective 지표는 coverage와 함께 표시했다.
- **결과**: 정상 PASS/REJECT/ABSTAIN/error와 결함 REJECT/PASS/missing/invalid
  8건의 합성 결정표에서 아래 값이 나왔다.
  - `expected=8`, `result=7`, `missing=1`, `error=1`
  - `ABSTAIN=1`, `invalid=1`, `selective=4`
  - accuracy와 balanced accuracy는 각각 `0.25`
  - coverage는 `0.5`

### McNemar와 유형 macro-F1에서 실패 ID·유형이 사라지는 문제

- **증상**:
  - McNemar는 두 조건의 성공 ID 교집합만 짝지었다.
  - 유형 지표는 오류 행을 제외했다.
  - verdict와 무관하게 `defect_type`을 인정할 수 있었다.
- **왜 문제?**:
  - 한 조건에서만 실패한 샘플이 검정에서 사라졌다.
  - 통째로 누락된 결함 유형도 macro 평균에서 사라졌다.
  - 결함을 `PASS`하고 유형명만 맞힌 행을 성공으로 보면 양불 판정과 모순된다.
- **조치**:
  - McNemar를 canonical ID 전체에서 짝지었다.
  - missing·error·ABSTAIN·invalid는 `false` 정오로 남겼다.
  - 유형 support와 클래스 목록은 roster의 결함 샘플로 고정했다.
  - clean `REJECT`의 type_map 정규화 예측만 인정했다.
- **결과**:
  - 두 조건이 모두 0행이어도 McNemar N은 유지됐다.
  - dev는 `52/24/28`(전체/정상/결함), held-out은 `140/56/84`다.
  - 한쪽 missing과 양쪽 missing 경계에서도 `N=3, b=1, c=1, both_wrong=1`이었다.
  - 결함 `PASS` 행은 유형명이 맞아도 type TP가 0이었다.

### 부분 실행·결과 오염이 완전한 실측처럼 보이는 문제

- **증상**:
  - 비용 가드가 중단한 tail은 run summary 분모에서 빠졌다.
  - 보고서는 사전 등록 N과 실제 결과 N을 구분하지 않았다.
  - 파일을 직접 지정하면 `_DRYRUN` 이름 차단도 우회됐다.
- **왜 문제?**:
  - partial 실행의 성공 행만으로 최종 결론을 낼 수 있었다.
  - fake 결과를 실측 표에 섞을 수 있었다.
- **조치**:
  - runner summary에 `requested_samples`, `processed_samples`, `missing_samples`,
    `complete_capture`를 남겼다.
  - 미처리 tail은 오답으로 집계했다.
  - 조건·층·품목·비용 표에 expected/result/missing/error/invalid/ABSTAIN/selective N과
    캡처 상태를 반복 표시했다.
  - partial이면 보고서 상단에 `결론 금지`를 표시했다.
  - 0행 결과는 전체 missing으로 표시했다.
  - 직접 지정한 `_DRYRUN.jsonl`과 비표준 `NaN/Infinity`도 거부했다.
- **결과**:
  - 2건 중 1건에서 비용 중단한 테스트는 `requested=2`, `processed=1`,
    `missing=1`, `complete_capture=false`였다.
  - 분모는 2로 고정됐다.
  - 0행·요청 조건 전체 누락도 정확도 0과 결론 금지로 보고됐다.

### Canonical roster·type_map 오염을 조용히 받아들이는 문제

- **증상**: 결과 ID·메타데이터·type alias의 무결성을 보장하는 공통 검증이 없었다.
- **왜 문제?**: 아래 오염을 임의로 버리거나 첫 키에 맞추면 채점 정답이 바뀐다.
  - 중복·미등록 ID
  - `item/label/is_defect/split` 드리프트
  - 한 품목의 두 canonical type이 같은 alias를 쓰는 경우
- **조치**:
  - canonical ID는 공백 없는 문자열로 제한했다.
  - `is_defect`는 exact bool로 제한했다.
  - 라벨과 결함 여부의 일관성을 검사했다.
  - 빈·비문자·공백·중복·미등록 ID는 집계를 중단하게 했다.
  - canonical 메타데이터 불일치와 대상 ID의 wrong split도 중단하게 했다.
  - 한 품목 안의 type alias 충돌도 중단하게 했다.
- **결과**: 각 부정 사례가 `EvaluationDataError` 또는 `ResultDataError`로
  hard-fail하는 회귀 테스트와 독립 합성 probe가 통과했다.

### R3 완료 후 채점자 관점 검토

- **증상**: 평가 코드의 단위 테스트만으로는 아래를 한 번에 입증하지 못했다.
  - 전체 서버·그래프 회귀
  - 실제 roster 크기
  - 보고서 표의 증거 충분성
- **왜 문제?**:
  - 성능 부풀리기를 막아도 사용자 흐름을 깨뜨릴 수 있다.
  - README·설계·실제 표의 용어가 다르면 발표 근거가 약해진다.
- **조치**:
  - 표적 테스트 46개를 실행했다.
  - fake 백엔드 전체 회귀, `compileall`, `git diff --check`를 실행했다.
  - 실제 dataset 메타데이터 분포와 합성 공격 probe를 독립 채점자 관점에서 확인했다.
  - README·design·architecture에 고정 N·coverage·selective·partial 결론 금지 계약을
    같은 표현으로 반영했다.
- **결과**:
  - 표적: `46 passed in 0.19s`.
  - 전체: `362 passed in 29.26s`.
  - eval compileall과 diff-check가 통과했다.
  - 실제 roster는 dev `52`(정상 24/결함 28), held-out `140`(정상 56/결함 84)이다.
  - 실제 roster와 R3 분모가 일치했다.
  - 기준서·가이드·`type_map.json`·dataset·lots의 diff는 없었다.
  - 테스트 이미지를 열거나 held-out 평가를 실행하지 않았다.
  - R3 범위의 P0/P1은 없으며 판정은 `GO`다.
  - 파일명을 바꾼 dry-run의 provenance·manifest/backend/hash 결속은 R5에서 검증한다.
  - 사람 블라인드·H2 통계 계약은 R4에서 검증한다.

## 16. R4 통계·블라인드·ablation 실험 재설계

### R4 한눈에 보기

| 검증 대상 | 고정한 방식 |
|---|---|
| H1 | 짝지은 `Δbalanced accuracy` 순열 검정 |
| H2 | 짝지은 `Δmacro-F1` 순열 검정 |
| H3 | 같은 A 실행의 validate 전·후 20쌍 블라인드 채점 |
| H4 | low/high 상호 배타 그룹, 각 20건 이상일 때만 판정 |
| ablation | heldout 금지, dev에서만 탐색적 해석 |

> 결론: 결과를 보고 통계·표본·비교군을 바꾸지 못하게 먼저 잠겼다.

### 가설 지표와 검정 통계량이 다른 문제

- **증상**:
  - H1 주 지표는 balanced accuracy다.
  - H2 주 지표는 결함 유형 macro-F1이다.
  - 그런데 두 가설을 전체 이진 정오의 McNemar로 판정할 여지가 있었다.
- **왜 문제?**:
  - 이진 감지가 같고 유형 분류만 좋아지면 H2 효과를 찾지 못한다.
  - 가설이 정한 효과가 아닌 정오 차이로 결과를 과장할 수도 있다.
- **조치**:
  - H1은 정상/결함 층을 유지한 짝지은 Δbalanced accuracy 순열 검정을 썼다.
  - H2는 품목/진실 유형 층을 유지한 짝지은 Δmacro-F1 순열 검정을 썼다.
  - 둘 다 같은 ID 안에서 교환하고 seed `20261020`을 사용했다.
  - 교환 짝 16개 이하는 정확 전수, 그 이상은 20,000회 Monte-Carlo로 고정했다.
  - 층화 paired bootstrap 10,000회와 95% CI를 고정했다.
  - McNemar는 이진 정오의 보조 분석으로만 남겼다.
- **결과**:
  - 이진 정오는 같아 McNemar가 `p=1.0`인 반례를 만들었다.
  - 12개 결함 짝의 macro-F1은 `0.0→1.0`이었다.
  - H2 정확 순열 검정은 `p=2/4096`을 냈다.
  - 순서 변경, 누락·오류·`ABSTAIN`, 총 오류·0 효과 경계도 회귀 테스트로 고정했다.

### H3가 독립 재실행·선택 가능 표본을 비교하는 문제

- **증상**:
  - 기존 A−V는 독립 재실행이었다.
  - B0/A 보고서를 서로 다른 ID로 뽑을 수도 있었다.
  - 성공한 보고서만 보고 표본을 바꿀 수 있었다.
  - dev·임의 seed 결과도 H3 지지로 표시할 수 있었다.
- **왜 문제?**:
  - 독립 실행은 VLM 출력·재작성·비용까지 함께 달라진다.
  - 실패 표본 대체와 사후 seed 선택은 좋은 보고서만 남기는 선택 편향이다.
  - 결과를 본 뒤 방법을 바꿨다는 의심을 막을 수 없다.
- **조치**:
  - A 한 실행의 첫 `write` 보고서를 deep copy했다.
  - canonical SHA-256과 함께 `pre_validation_snapshot`에 보존했다.
  - 재작성으로 snapshot을 덮어쓰지 않게 했다.
  - 결과 가용성을 보기 전에 canonical roster에서 20 ID를 선택했다.
  - 품목×정상/결함 비율로 뽑고 validate 전/후 40건을 무작위로 섞었다.
  - 오류·누락·무효·hash 불일치는 재표집 없이 생성을 중단했다.
- **결과**:
  - held-out 생성은 `sample=20`, `seed=20261020`만 허용한다.
  - unblind는 mapping의 `split=heldout`, 정확한 seed, 20쌍, 2N 정합,
    payload hash를 다시 검사한다.
  - 모두 통과해야 `conclusion_allowed=true`가 된다.
  - dev·임의 seed·임의 N도 Wilson CI와 exact McNemar는 계산한다.
  - 하지만 `observed_support=false`, `결론 금지`로 남는 회귀 테스트가 통과했다.

### 블라인드 형식은 있지만 채점 rubric이 빈 문제

- **증상**: claim 판정 라벨만 있었다. 아래 규칙은 고정되지 않았다.
  - 복합 claim 처리
  - 판정·조치 문구 처리
  - 애매한 근거 처리
  - stage 추측 순서
- **왜 문제?**:
  - 같은 보고서도 시점에 따라 `NON_FACTUAL`과 `HALLUCINATED`가 달라질 수 있다.
  - claim 채점 뒤 stage를 맞히면 남아 있는 스타일 단서를 축소해 보여줄 수 있다.
- **조치**:
  - 행은 무작위 순서로 한 번씩만 처리했다.
  - 검색·정렬·쌍 대조를 금지했다.
  - 내용을 읽은 직후 stage 추측을 claim 채점보다 먼저 기록했다.
  - 기록한 stage 추측은 수정하지 않게 했다.
  - 직접 근거, 의미 보존 변형, 비근거 사실이 하나라도 섞인 claim,
    순수 판정·권고, 판별 불가를 각각 정해진 라벨에 대응시켰다.
- **결과**:
  - `UNCERTAIN`은 보수적으로 환각에 포함한다.
  - 빈·미지·미완료 판정은 unblind에서 hard-fail한다.
  - 페이로드 변조와 쌍·stage 중복도 hard-fail한다.
  - 같은 ID의 관찰 원문 때문에 쌍·stage를 추정할 수 있다는 한계가 남는다.
  - 1인 채점 한계와 stage-guess 정확도를 반드시 공개한다.

### ablation 비교가 의도한 구성 효과를 고립하지 못한 문제

- **증상**:
  - `no_crops`에서도 Observer가 `O?`를 내면 center crop을 다시 실행했다.
  - B0/A·B1/A·A/A−V 차이를 RAG·validate 단독 효과처럼 설명할 여지가 있었다.
- **왜 문제?**:
  - `A−C`에 center crop이 섞이면 full-only 비교가 아니다.
  - B0/A는 모델·호출 수·crop·교차 검증·RAG·validate가 함께 바뀐다.
  - B1/A도 전문 삽입 단일 VLM과 전체 에이전트 비교다.
  - 따라서 retrieval 단독 인과를 말할 수 없다.
- **조치**:
  - `no_crops`+`O?`에서 center 재계획을 막았다.
  - 이벤트에 `replan_skipped=true`, `skip_reason=no_crops_full_only`를 남겼다.
  - 알 수 없는 `ABLATION`은 deps·서버 시작에서 fail-fast하게 했다.
  - held-out은 `B0|B1|A`만 허용했다.
  - `A−X|A−C|A−V`는 샘플 로드·디렉터리 생성 전에 거부했다.
  - 세 ablation은 dev 전용 탐색 조건으로 고정했다.
- **결과**:
  - 일반 조건의 center 1회 재계획은 유지했다.
  - A−C만 full-only가 되는 경계 테스트가 통과했다.
  - B0/A는 전체 시스템 효과로만 표현했다.
  - B1/A는 전문 삽입 대안과 전체 시스템의 비교로 표현했다.
  - 세 ablation은 dev의 탐색적 비용·효과 진단으로만 표현했다.

### H4와 dev 보고서가 작은 그룹으로 확정 결론을 낼 수 있는 문제

- **증상**:
  - `confidence=low` 오답률을 전체 결과와 비교해 low 행이 분모에 중복됐다.
  - 몇 건만으로도 H4를 지지할 수 있었다.
  - dev의 H1·H2·H4 표에도 `지지` 또는 `효과 미입증`이 나올 수 있었다.
- **왜 문제?**:
  - 중첩 분모와 작은 표본은 신뢰도 플래그의 변별력을 과장한다.
  - 탐색·튜닝용 dev의 확증적 표현은 held-out 결과로 오인될 수 있다.
  - 사후 선택도 유도할 수 있다.
- **조치**:
  - 오류 없는 PASS/REJECT 안에서 low와 high를 상호 배타 그룹으로 나눴다.
  - 그룹별 오답률과 Wilson 95% CI를 표시했다.
  - 두 그룹 중 하나라도 20건 미만이면 보류했다.
  - `ABSTAIN`·오류·누락·무효 판정·무효 confidence는 별도 상태로 보고했다.
  - dev는 수치와 무관하게 H1·H2·H4를 모두 `보류(dev 탐색)`으로 강제했다.
- **결과**:
  - low/high 0건 경계가 통과했다.
  - 한쪽 부족, 둘 다 20건 이상, 오답률 2배 경계도 통과했다.
  - dev 보고서 회귀 테스트가 통과했다.
  - 최종 판정은 held-out 완전 캡처에서만 가능하다.

### R4 완료 후 채점자 관점 검토

- **증상**: 첫 구현 통과 뒤 종료 리뷰에서 P1이 추가로 발견됐다.
  - dev 가설 판정으로 사후 해석을 우회할 수 있었다.
  - held-out ablation으로 비교군을 바꿀 수 있었다.
  - seed·표본 가드가 블라인드 생성 시점에만 있었다.
- **왜 문제?**: 기능 테스트가 통과해도 아래 가능성이 남으면 증거 신뢰가 깨진다.
  - 결과 선택
  - 비교군 변경
  - 기술 통계를 확정 결론처럼 바꾸는 행위
- **조치**:
  - 주 지표·분모·표본·seed·결론 가드를 독립된 지점에서 다시 검사했다.
  - 각 우회는 부작용 전에 hard-fail하도록 부정 테스트를 추가했다.
  - 설계·Task·보고서와 실행 코드의 인과 주장 범위를 다시 맞췄다.
- **결과**:
  - R4 표적 평가 회귀: `83 passed`.
  - fake LLM/VLM 전체 회귀: `404 passed in 31.26s`.
  - `compileall`, 핵심 import, 수정 shell 문법, `git diff --check`가 통과했다.
  - 기준서·가이드·`type_map.json`·dataset·lots의 diff는 없었다.
  - held-out 평가와 이미지 열람은 하지 않았다.
  - 두 보고서에 같은 기준서 인용(reference)이 반복되면, 채점자가 두 보고서가
    validate 전·후 한 쌍이라는 사실을 눈치챌 수 있다. 완전한 블라인드 평가는
    아니므로 이 점과 1인 채점의 한계를 결과 보고서에 명시한다.
  - 실제 실행 provenance 전체 동결은 R5 종료 게이트에서 확인한다.
  - R4 범위 P0/P1은 없다.
  - 독립 종료 판정은 `GO WITH RISKS`다.
  - R5 완료 전에 held-out 결과를 확정하지 않는 조건으로 다음 Phase 진행이 가능하다.
