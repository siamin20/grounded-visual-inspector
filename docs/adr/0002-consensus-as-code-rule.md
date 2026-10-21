# ADR-0002: 합의 판정을 코드 규칙으로 구현
- 상태: Accepted
- 관련 모듈: `agent/nodes/analysis.py::analysis_node`, `agent/nodes/gate.py::gate_node`, `agent/nodes/replan_check.py::replan_check_node`

## 배경 (Context)
Observer·Cross-Checker의 이상 여부와 영역, Spec-Judge의 위반 조항을 합쳐 동일 입력에 재현 가능한 판정을 내려야 한다. 판독 불가를 bool `False`로 합치면 근거 없는 PASS가 되고, REJECT로 숨기면 결함 샘플에서 정답으로 채점될 수 있다.
## 결정 (Decision)
`analysis_node()`가 판독 가능 범위, 양성 영역 일치, 검증된 위반 조항을 코드로 조합한다. 근거 있는 결함은 REJECT, 충분한 정상 근거는 PASS, 증거 부족·미근거 이상·모순은 `ABSTAIN/low`와 재검사로 보낸다. ABSTAIN은 주 평가에서 오답이며 coverage를 별도 보고한다.
## 대안 비교 (Alternatives Considered)
| 대안 | 설명 | 장점 | 단점 |
|---|---|---|---|
| LLM 합의 판정 | 세 에이전트 결과를 프롬프트로 종합 | 복잡한 맥락을 표현 가능 | 비결정적이고 규칙 단위 검증이 어려움 |
| 코드 규칙 | 조건문으로 합의와 보수적 분기를 고정 | 재현·단위 테스트·감사 가능 | 예외 맥락을 수동 반영해야 함 |
| REJECT + assessable 필드 | 판독 불가를 출하 보류로 표시 | 기존 이진 enum 유지 | 결함 판독 불가가 TP처럼 보이고 판정·운영 의미가 섞임 |
## 결과 / 트레이드오프 (Consequences)
Base의 `gate_node()`·`replan_check_node()`와 같은 원칙으로 LLM이 안전 분기를 바꾸지 못한다. 대신 API·UI·평가기가 세 번째 값을 명시적으로 처리하고 선택적 정확도와 coverage를 함께 보여야 한다.
## 재검토 조건
고정 규칙이 dev/held-out에서 특정 불일치 유형을 반복적으로 오판하고 검증 가능한 대체 방식이 확보되면 재검토한다.
