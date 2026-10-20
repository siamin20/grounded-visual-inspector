# Mini-Project 진행 계획 (Google Spread 등록용) — 외관 불량 판정 에이전트

- 과제 유형: **2번 신규 프로젝트** (Base AI Agent의 LangGraph·MCP·SSE·세션 구조 재사용, 도메인 교체)
- 주제: **근거 검증 루프를 가진 외관 불량 판정 에이전트** — 제품 이미지를 VLM이 "관찰"하고, 검사 기준서 RAG가 "판정"하며, 교차 검증·원문 대조 validate가 환각을 막고, 폐기·라인 정지 같은 조치는 사람 승인(HITL)을 거친다.
- 문제: 범용 VLM은 산업 외관검사 벤치마크(MMAD, ICLR 2025)에서 GPT-4o도 74.9%에 그치고 틀린 근거를 지어낸다. 학습 없는 지각기 위에 신뢰할 수 있는 판정 레이어가 필요하다.
- 데이터: MVTec AD(CVPR 2019, 공개 벤치마크, 폴더명이 결함 유형 라벨) 4품목 약 200장. 정답을 내가 만들지 않음.
- 상세 설계: `설계서_비전검사에이전트.md`

## 세부 기능 리스트

| 순번 | 영역 | 무엇을 만드나 | 지시서 개선 포인트 | 우선순위 |
|---|---|---|---|---|
| 1 | MCP | vision 서버: Tool `inspect_image` / Resource `spec://{item}`, `guide://{defect}` / Prompt `inspection_prompt` | — | P0 |
| 2 | 그래프 | research를 Observer(VLM 전체+4크롭 병렬) / Spec-Judge(RAG) / Cross-Checker 3개 서브그래프로 분리 | 3번 | P0 |
| 3 | 그래프 | analysis 합의 규칙(코드) → write 스키마 → validate 원문·조항 대조 → 반려 재작성(≤2회) | — | P0 |
| 4 | 그래프 | write→validate 사이 HITL interrupt (위험 조치·확신 낮음) | 4번 | P0 |
| 5 | RAG | 기준서 조항 임베딩 검색, 결함 유형·조치 매핑 | 1번 | P1 |
| 6 | 평가 | Before(VLM 단독)/After(에이전트)/Ablation 실측: 정확도·FNR·환각률·확신 플래그 유효성·지연·비용, LangSmith Dataset | 6번 유사 | P1 |
| 7 | 모듈화·테스트 | nodes/subgraphs 분리, pytest(validate·합의 규칙·RAG) | 2번 | P2 |

## Before/After 측정
held-out 140장 고정. B0 VLM 단독 vs A 에이전트 (+ ablation). 지표: 양불 정확도, FPR/FNR, 유형 정확도, 환각률, 확신 플래그 내 오답률, p50/p95 지연, 토큰·USD, 반려 횟수. 95% 신뢰구간 제시, 효과 없으면 없다고 보고.

## 예상 한계
호출 수 증가로 지연·비용 수 배 / validate false reject / 기준서 작성자 편향 / MVTec 통제 환경의 현장 일반화 한계 / 소형 로컬 VLM 품질 / HITL 상태 복원 / 루프 상한 / 비상업 라이선스

## 일정
Day1 15–21 데이터·MCP·Observer | Day2 오전 서브그래프·RAG | Day2 오후 validate·HITL·평가 파이프라인 | Day2 저녁 실측 | Day3 오전 모듈화·README·PPT | Day3 13:00 제출, 14:00 발표
