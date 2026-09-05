"""FastAPI 版 Web 界面(轻量) + SSE 流式接口。

不带浏览器也能当 API 用。启动：
    python -m web.fastapi_app          # 打开 http://127.0.0.1:8001
界面 HTML 是项目根目录下的独立 index.html(可单独打开/传 GitHub Pages)。
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from agent.graph import get_agent, detect_mode

app = FastAPI(title="知友 AI 辅导+科普 Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 独立前端文件：位于项目根(与仓库一起推送，GitHub Pages 也能直接展示该壳)
INDEX_HTML_PATH = Path(__file__).resolve().parent.parent / "index.html"


class ChatBody(BaseModel):
    message: str
    session_id: str = "web-user"


@app.get("/", response_class=HTMLResponse)
def index():
    """返回独立 index.html(实时读取，改完刷新即生效)。"""
    try:
        return HTMLResponse(INDEX_HTML_PATH.read_text(encoding="utf-8"))
    except OSError:
        return HTMLResponse(
            "<h3>未找到 index.html</h3>"
            "<p>请在项目根目录下运行 <code>python -m web.fastapi_app</code>。</p>",
            status_code=500,
        )


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(body: ChatBody):
    """非流式 JSON 接口。"""
    try:
        agent = get_agent()
        answer = agent.invoke(body.session_id, body.message)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"内部错误: {exc}")
    try:
        agent.remember(body.session_id, body.message, answer)
    except Exception:
        pass
    return {"answer": answer, "mode": detect_mode(body.message), "session_id": body.session_id}


@app.get("/api/chat/stream")
async def chat_stream(
    message: str = Query(...),
    session_id: str = Query(default="web-user"),
):
    """SSE 流式接口，事件为 JSON：{type: start|token|tool|tool_result|done, ...}。"""

    async def gen():
        agent = get_agent()
        answer = ""
        try:
            async for ev in agent.stream_events(session_id, message):
                if ev["type"] == "done":
                    answer = ev.get("answer", "")
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except RuntimeError as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': f'内部错误: {exc}'}, ensure_ascii=False)}\n\n"
        finally:
            if answer:
                try:
                    agent.remember(session_id, message, answer)
                except Exception:
                    pass

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import os

    import uvicorn

    # 端口优先读环境变量 PORT(Hugging Face Spaces 会自动注入 7860；本地默认 8001)
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run("web.fastapi_app:app", host="0.0.0.0", port=port, reload=False)
