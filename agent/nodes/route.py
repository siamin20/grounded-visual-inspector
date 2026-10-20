"""입력에 이미지가 있는지로 검사와 일반 채팅을 나눈다."""

from langsmith import traceable

from agent.deps import SupervisorDeps
from agent.state import AgentState

from .common import KNOWN_ITEMS, _cost, _tokens, event


@traceable(name="route_node")
async def route_node(state: AgentState, deps: SupervisorDeps) -> dict:
    if state.get("image_path"):
        route = "inspection" if state.get("item") in KNOWN_ITEMS else "needs_item"
        return {
            "route": route,
            "item": state.get("item"),
            "events": event(state, "RouteDecided", route=route),
        }

    # Base 라우터 호출과 비용 집계는 보존하되, incident 도메인은 더 이상 없다.
    result, usage = deps.fast_llm.route(state.get("query", ""))
    return {
        "route": "chat",
        "needs_analysis": False,
        "cost_usd": _cost(state, usage),
        "tokens": _tokens(state, usage),
        "events": event(
            state, "RouteDecided", route="chat", classifier_route=result.get("route")
        ),
    }
