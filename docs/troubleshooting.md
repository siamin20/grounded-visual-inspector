# grounded-visual-inspector 트러블슈팅 기록

- 작성일: 2026-10-20
- 범위: Phase 1 데이터·기준서, Phase 2 vision MCP, Phase 3 그래프·RAG 이식
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
