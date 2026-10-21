# Mini-Project 진행 계획 (Google Spread 등록용) — 외관 불량 판정 에이전트

> 이 문서는 주제 등록 당시의 진행 계획을 현재 용어에 맞춰 보존한 참고 자료다.
> 구현·평가 계약의 단일 기준은 `docs/design.md`이며, 두 문서가 다르면 설계를 따른다.

- 과제 유형: **2번 신규 프로젝트** (Base AI Agent의 LangGraph·MCP·SSE·세션 구조 재사용, 도메인 교체)
- 주제: **근거 검증 루프를 가진 외관 불량 판정 에이전트** — 제품 이미지를 VLM이 "관찰"하고, 검사 기준서 RAG가 "판정"하며, 교차 검증·원문 대조 validate가 환각을 막고, 폐기·라인 정지 같은 조치는 사람 승인(HITL)을 거친다.
- 문제: 범용 VLM은 산업 외관검사 벤치마크(MMAD, ICLR 2025)에서 GPT-4o도 74.9%에 그치고 틀린 근거를 지어낸다. 학습 없는 지각기 위에 신뢰할 수 있는 판정 레이어가 필요하다.
- 데이터: MVTec AD(CVPR 2019, 공개 벤치마크, 폴더명이 결함 유형 라벨) 4품목 192장(dev 52, held-out 140). 정답을 내가 만들지 않음.
- 상세 설계: `docs/design.md`

## 세부 기능 리스트

| 순번 | 영역 | 무엇을 만드나 | 지시서 개선 포인트 | 우선순위 |
|---|---|---|---|---|
| 1 | MCP | vision 서버: Tool `inspect_image` / Resource `spec://{item}`, `guide://{defect}` / Prompt `inspection_prompt` | — | P0 |
| 2 | 그래프 | research를 Observer(VLM 전체+4크롭 병렬) / Spec-Judge(RAG) / Cross-Checker 3개 서브그래프로 분리 | 3번 | P0 |
| 3 | 그래프 | analysis 합의 규칙(코드) → write 스키마 → validate 원문·조항 대조 → 반려 재작성(≤2회) | — | P0 |
| 4 | 그래프 | write→validate 사이 HITL interrupt (위험 조치·확신 낮음) | 4번 | P0 |
| 5 | RAG | 기준서 조항 임베딩 검색, 결함 유형·조치 매핑 | 1번 | P1 |
| 6 | 평가 | held-out B0/B1/A와 dev 전용 ablation: 종단 balanced accuracy·FNR·유형 macro-F1·환각률·확신 플래그 유효성·지연·추정 비용, LangSmith Dataset | 6번 유사 | P1 |
| 7 | 모듈화·테스트 | nodes/subgraphs 분리, pytest(validate·합의 규칙·RAG) | 2번 | P2 |

## Before/After 측정
held-out 140장은 B0 VLM 단독, B1 기준서 전문 삽입, A 전체 에이전트를 같은 ID로 비교한다. ablation은 dev 전용 탐색으로 분리한다. 지표는 end-to-end balanced accuracy, 보수적 FPR/FNR, defect-type macro-F1, 블라인드 사람 채점 환각 포함률, 확신 플래그별 오답률, p50/p95 지연, 토큰·추정 USD, 반려·재계획·재시도 횟수다. 95% 신뢰구간을 제시하고 효과가 없으면 없다고 보고한다.

## 예상 한계
호출 수 증가로 지연·비용 수 배 / validate false reject / 기준서 작성자 편향 / MVTec 통제 환경의 현장 일반화 한계 / 소형 로컬 VLM 품질 / HITL 상태 복원 / 루프 상한 / 비상업 라이선스

## 등록 당시 일정
Day1 15–21 데이터·MCP·Observer | Day2 오전 서브그래프·RAG | Day2 오후 validate·HITL·평가 파이프라인 | Day2 저녁 실측 | Day3 오전 모듈화·README·PPT | Day3 13:00 제출, 14:00 발표

현재 실행 순서와 완료 게이트는 `_tasks/00_MASTER_PROMPT.md`의 보정 Phase R0~R10을 따른다.
