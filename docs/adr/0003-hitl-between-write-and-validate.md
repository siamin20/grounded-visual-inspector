# ADR-0003: HITL을 write와 validate 사이에 배치
- 상태: Accepted
- 관련 모듈: `agent/graph.py::build_graph`, `agent/nodes/hitl.py::hitl_node`, `agent/nodes/validate.py::validate_node`

## 배경 (Context)
사람은 폐기·라인정지 또는 low 판정의 근거와 조치를 읽고 승인해야 하며, 승인이 보고서 근거 검증을 우회해서는 안 된다.
## 결정 (Decision)
`build_graph()`의 순서를 `write → hitl → validate`로 고정한다. `hitl_node()`는 초안을 보여 중단·재개하고, 승인된 내용도 `validate_node()`의 코드 규칙을 통과해야 한다. validate가 반려하고 재작성 여지가 있으면 `write`가 새 초안을 만들고 과거 승인을 초기화한 뒤 HITL 승인을 다시 받는다. 재작성 상한의 `validation_failure` ABSTAIN도 별도 초안으로 다시 승인받는다.
## 대안 비교 (Alternatives Considered)
| 대안 | 설명 | 장점 | 단점 |
|---|---|---|---|
| validate 뒤 | 검증 완료본만 사람에게 제시 | 초안 변경 가능성이 적음 | 사람 승인 후 독립 안전망이 없음 |
| analysis 뒤 | 판정만 보여 승인 후 작성 | 조기 중단 가능 | 근거·조치를 보지 못한 승인 |
## 결과 / 트레이드오프 (Consequences)
사람은 실제 초안을 판단하고 승인 후에도 근거·조항·스키마 검증을 받는다. 검증 반려나 사람 반려로 내용이 바뀌면 `write → hitl`을 다시 지나고, 상한에서 만든 ABSTAIN도 새 초안으로 승인하므로 보지 않은 변경본이 과거 승인으로 통과하지 않는다. 대신 재작성 횟수만큼 승인 대기가 늘 수 있다.
## 재검토 조건
규제상 검증 완료본만 사람에게 노출해야 하거나 반복 승인 비용을 줄여야 하면 배치를 재검토한다. 단, 재승인을 생략하려면 승인 뒤 보고서가 불변임을 보장하는 동등한 통제가 먼저 있어야 한다.
