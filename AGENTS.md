# AGENTS.md — grounded-visual-inspector

SK AX AI Capstone의 Base Agent(FastAPI + LangGraph + MCP + Vue 채팅)를
"이미지 기반 제품 결함 판정 및 근거 검증 멀티에이전트"로 이식하는 프로젝트다.

## 먼저 읽을 것 (이 순서로, 필요한 부분만)
1. `docs/design.md` — 설계 SSOT. 지시가 서로 충돌하면 이 문서가 이긴다.
2. `_tasks/00_MASTER_PROMPT.md` — Phase 순서, 완료 기준, 멈춤 지점.
3. 지금 작업하는 폴더의 `_TASK.md` — 그 폴더의 상세 사양.

## 지켜야 할 것
- Base 구조 보존. `AgentState`, 노드 시그니처 `async def x_node(state, deps) -> dict`,
  `SupervisorDeps` 주입(`functools.partial`), `@traceable`, SSE `data: {json}\n\n` 포맷,
  `events` 누적 방식은 그대로 둔다. Base 파일을 통째로 다시 쓰지 말고 필요한 부분만 고친다.
- gate / analysis / validate 는 LLM 호출 없이 코드 규칙으로 판단한다.
- `LLM_BACKEND=fake VLM_BACKEND=fake` 에서 외부 호출 없이 `pytest` 전부 통과해야 한다.
- `eval/run_eval.py` 는 fake 백엔드를 거부한다(결과 조작 방지).
- `data/mvtec/`, `data/uploads/`, `.env`, `*.db` 는 커밋 대상이 아니다. 이미 .gitignore 에 있다.
- 기본적으로 git 커밋·push는 하지 않는다. 사용자가 명시적으로 요청한 경우에만
  요청 범위의 파일을 선별해 커밋·push하고, 작업 트리의 다른 변경은 섞지 않는다.
  커밋 메시지·코드 주석·문서 어디에도 AI 도구 이름이나 Co-Authored-By를 넣지 않는다.
- 주석·문서는 한국어, 식별자는 영어.
- 확실하지 않으면 추측으로 구현하지 말고 `_tasks/QUESTIONS.md` 에 질문을 적은 뒤
  그 결정에 의존하지 않는 다음 작업을 계속한다.
- 한 Phase가 끝나면 멈추고 보고한다: 바뀐 파일 목록, 실행한 검증 명령과 그 출력 요약,
  미해결 질문.
- 각 Phase의 구현·검증 뒤에는 반드시 채점자 관점 종료 리뷰를 별도로 수행한다.
  공식 요구·design SSOT·Phase 완료 기준 추적, fail-open·오류 제외·heldout 오염,
  코드-문서-테스트-UI 정합성, 결과 과장 가능성, 발표에서 방어할 수 있는 근거를 확인한다.
  Phase 범위 안의 지적은 고치고 재검증한 뒤에만 완료 처리한다. 후속 Phase 의존 사항은
  담당 Phase와 검증 공백을 명시하며, 최종 보고에 `GO | GO WITH RISKS | NO-GO` 판정을 넣는다.
- `docs/troubleshooting.md` 는 누적 기록이다. Phase 마다 막혔던 것, 리뷰에서 지적된 것,
  그 해소 방법과 결과를 `증상 → 왜 문제인가 → 조치 → 결과` 형식으로 덧붙인다. 리뷰어가
  "(Phase N 완료 후 기록)" 으로 비워 둔 항목은 해당 Phase 에서 채운다.

## 실행 환경
- `bash sh/setup_venv.sh` 로 `.venv` 생성. 이후 `.venv/bin/python3` 사용.
- 서버: `python3 server.py` (8100), 프런트 프록시: `python3 web_demo_server.py` (8600).
- 테스트: `LLM_BACKEND=fake VLM_BACKEND=fake .venv/bin/python3 -m pytest -q`.
