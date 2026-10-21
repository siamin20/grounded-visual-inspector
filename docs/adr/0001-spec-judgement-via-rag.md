# ADR-0001: 판정 기준을 RAG로 분리
- 상태: Accepted
- 관련 모듈: `agent/rag/index.py::SpecIndex`, `agent/subgraphs/spec_judge.py::spec_judge_node`, `agent/supervisor_v1.py::run_v1`

## 배경 (Context)
검사 결과는 VLM의 일반 지식이 아니라 품목별 기준서 조항에 근거해야 하며, 현재 4품목 규모에서는 전문 삽입도 경쟁력이 있다.
## 결정 (Decision)
`SpecIndex.from_dir()`로 조항 단위 색인을 만들고, `SpecIndex.search()`가 품목 필터 후 상위 3개를 `spec_judge_node()`에 제공한다. B1은 `run_v1()`의 기준서 전문 삽입 비교군으로 유지한다.
## 대안 비교 (Alternatives Considered)
| 대안 | 설명 | 장점 | 단점 |
|---|---|---|---|
| 프롬프트 전문 삽입(B1) | 품목 기준서 전체를 VLM 입력에 포함 | 구현이 간단하고 누락이 적음 | 품목 확장 시 토큰·비용 증가 |
| 파인튜닝 | 기준을 모델 가중치에 반영 | 추론 시 검색이 불필요 | 기준 변경 때 재학습·검증 필요 |
## 결과 / 트레이드오프 (Consequences)
인용 조항을 추적·검증하고 기준서를 독립 갱신할 수 있지만, 검색 오류와 임베딩 비용이 추가된다.
## 재검토 조건
held-out 동일 샘플에서 B1의 성능이 A 이상이면 소규모 운영은 B1을 우선하고 이 결정을 재검토한다.
