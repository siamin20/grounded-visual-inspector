"""일반 채팅과 품목 선택 안내 노드."""

from langgraph.config import get_stream_writer
from langsmith import traceable

from agent.deps import SupervisorDeps
from agent.state import AgentState

from .common import _cost, _tokens, event, load_type_map


@traceable(name="chat_answer_node")
async def chat_answer_node(state: AgentState, deps: SupervisorDeps) -> dict:
    if state.get("route") == "needs_item":
        items = "/".join(load_type_map())
        reply = f"품목을 선택해 주세요({items})"
        return {
            "final": reply,
            "events": event(state, "Finalize", pipeline="needs-item"),
        }

    writer = get_stream_writer()
    reply, usage = await deps.fast_llm.chat(
        state.get("query", ""),
        state.get("history", []),
        on_token=lambda piece: writer({"type": "AnswerToken", "delta": piece}),
    )
    return {
        "final": reply,
        "last_prompt": usage.get("prompt"),
        "cost_usd": _cost(state, usage),
        "tokens": _tokens(state, usage),
        "events": event(state, "Finalize", pipeline="chat-direct"),
    }
