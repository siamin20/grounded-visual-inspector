"""web_demo_server.py — 제품 결함 판정 에이전트의 브라우저 데모.

web_chat.html이 바닐라 JS 대신 Vue 3(CDN 글로벌 빌드, 별도 빌드 단계 없음)로 되어
있다는 점만 all_total과 다르다. server.py(8100, 기존 통합 테스트 서버)는 전혀
건드리지 않는다. 이 스크립트는
- web_chat.html 한 장을 같은 출처(origin)로 띄우고
- /chat/stream, /inspect, /chat/resume 요청을 8100으로 중계해서

브라우저의 CORS 제약 없이 실제 에이전트와 대화할 수 있게 해주는 아주 얇은 레이어다.
server.py는 평소처럼(fake/ollama/openai 아무 백엔드로나) 8100에 미리 띄워 놓아야 한다.

실행:
    python3 web_demo_server.py        # http://localhost:8600 에서 채팅 화면 오픈
"""
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

BACKEND = "http://127.0.0.1:8100"
HERE = Path(__file__).parent

app = FastAPI()


@app.get("/")
async def index():
    return FileResponse(HERE / "web_chat.html")


@app.get("/health")
async def proxy_health():
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{BACKEND}/health")
    return JSONResponse(r.json(), status_code=r.status_code)


@app.post("/chat")
async def proxy_chat(req: Request):
    body = await req.body()
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{BACKEND}/chat", content=body,
                              headers={"content-type": "application/json"})
    return JSONResponse(r.json(), status_code=r.status_code)


async def _stream_response(
    method: str,
    path: str,
    *,
    content: bytes | None = None,
    content_type: str | None = None,
    params: dict[str, str] | None = None,
) -> StreamingResponse:
    """상류 응답을 버퍼링하지 않고 상태 코드와 함께 전달한다."""
    client = httpx.AsyncClient(timeout=None)
    headers = {"accept": "text/event-stream"}
    if content_type:
        # multipart boundary를 포함한 원본 Content-Type을 그대로 보존한다.
        headers["content-type"] = content_type
    request = client.build_request(
        method, f"{BACKEND}{path}", content=content, headers=headers, params=params
    )
    try:
        upstream = await client.send(request, stream=True)
    except Exception:
        await client.aclose()
        raise

    async def relay():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        relay(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "text/event-stream"),
        headers={"cache-control": "no-cache"},
    )


@app.get("/chat/stream")
async def proxy_stream(
    message: str,
    thread_id: str = "web-demo",
    image_path: str | None = None,
    item: str | None = None,
):
    params = {"message": message, "thread_id": thread_id}
    if image_path is not None:
        params["image_path"] = image_path
    if item is not None:
        params["item"] = item
    return await _stream_response("GET", "/chat/stream", params=params)


@app.post("/inspect")
async def proxy_inspect(req: Request):
    return await _stream_response(
        "POST",
        "/inspect",
        content=await req.body(),
        content_type=req.headers.get("content-type"),
    )


@app.post("/chat/resume")
async def proxy_resume(req: Request):
    return await _stream_response(
        "POST",
        "/chat/resume",
        content=await req.body(),
        content_type=req.headers.get("content-type"),
    )


if __name__ == "__main__":
    print("브라우저에서 http://localhost:8600 을 열어보세요 (server.py가 8100에 떠 있어야 합니다)")
    uvicorn.run(app, host="0.0.0.0", port=8600)
