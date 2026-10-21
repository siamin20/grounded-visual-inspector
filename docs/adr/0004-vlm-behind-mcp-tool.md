# ADR-0004: VLM을 MCP Tool 뒤에 격리
- 상태: Accepted
- 관련 모듈: `mcp_server_vision.py::inspect_image`, `agent/mcp_client.py::MCPClient`, `agent/retry.py::with_retry`

## 배경 (Context)
Observer와 Cross-Checker가 동일한 관찰 계약을 쓰면서 fake·Ollama·OpenAI VLM 교체, 재시도, 일시 장애 테스트를 공유해야 한다.
## 결정 (Decision)
노드는 `MCPClient.find_tool()`로 발견한 `inspect_image`만 `call_tool()`로 호출한다. `mcp_server_vision.py`가 `make_vlm()`로 백엔드를 선택하고, 노드의 호출은 `with_retry()`를 거친다.
## 대안 비교 (Alternatives Considered)
| 대안 | 설명 | 장점 | 단점 |
|---|---|---|---|
| 노드 내 직접 호출 | 각 노드가 VLM SDK/HTTP를 직접 사용 | 프로세스 경계 비용이 없음 | 백엔드·재시도·모의 장애 경로가 분산 |
| MCP Tool 격리 | 구조화 계약과 stdio 서버로 접근 | 백엔드 교체와 통합 테스트 경로 공유 | 서브프로세스·직렬화 오버헤드 |
## 결과 / 트레이드오프 (Consequences)
`MCPClient`의 첫 Tool 호출 Timeout 주입과 실제 호출이 같은 재시도 경로를 쓰지만, MCP 프로세스 생명주기와 오류 처리가 추가된다.
## 재검토 조건
MCP 경계의 지연·운영 비용이 측정상 허용 범위를 넘고 백엔드 교체 요구도 없어지면 직접 호출을 재검토한다.
