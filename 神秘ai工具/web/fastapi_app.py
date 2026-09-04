"""FastAPI 版 Web 界面(轻量) + SSE 流式接口。

不带浏览器也能当 API 用。启动：
    python -m web.fastapi_app          # 打开 http://127.0.0.1:8001
"""
from __future__ import annotations

import json

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

HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>知友 · AI 辅导/科普</title>
<style>
  :root{--bg:#0f172a;--panel:#1e293b;--accent:#38bdf8;--text:#e2e8f0;--muted:#94a3b8}
  *{box-sizing:border-box;margin:0}
  body{background:var(--bg);color:var(--text);font-family:system-ui,'Segoe UI',sans-serif;height:100vh;display:flex;flex-direction:column}
  header{padding:16px 22px;background:var(--panel);border-bottom:1px solid #334155;display:flex;gap:12px;align-items:center}
  header h1{font-size:18px} header .tag{color:var(--muted);font-size:13px}
  #chat{flex:1;overflow-y:auto;padding:22px;display:flex;flex-direction:column;gap:14px}
  .msg{max-width:78%;padding:12px 16px;border-radius:14px;line-height:1.65;white-space:pre-wrap;font-size:15px}
  .user{align-self:flex-end;background:#0369a1}
  .bot{align-self:flex-start;background:var(--panel);border:1px solid #334155}
  .meta{color:var(--muted);font-size:12px;margin-bottom:4px}
  .tool{color:#c084fc;font-style:italic;font-size:13px}
  #bar{display:flex;gap:10px;padding:14px 22px;background:var(--panel)}
  #input{flex:1;background:#0f172a;border:1px solid #334155;border-radius:10px;color:var(--text);padding:11px 14px;font-size:15px;outline:none}
  #btn{background:var(--accent);color:#082f49;border:0;border-radius:10px;padding:0 22px;font-weight:700;cursor:pointer}
  #btn:disabled{opacity:.5} .katex-err{color:#f87171}
</style>
</head>
<body>
<header><h1>知友</h1><span class="tag">辅导讲题(苏格拉底引导) · 科普(What-Why-Wow) · 记忆你的画像</span></header>
<div id="chat"></div>
<div id="bar">
  <input id="input" placeholder="输入题目、作业或任何“为什么”…" autofocus/>
  <button id="btn" onclick="send()">发送</button>
</div>
<script>
const chat=document.getElementById('chat');
function add(cls,html){const d=document.createElement('div');d.className='msg '+cls;d.innerHTML=html;chat.appendChild(d);chat.scrollTop=chat.scrollHeight;return d;}
let last=null;
async function send(){
  const inp=document.getElementById('input'),btn=document.getElementById('btn');
  const text=inp.value.trim(); if(!text)return;
  inp.value=''; btn.disabled=true;
  add('user',text.replace(/</g,'&lt;'));
  const box=add('bot','');
  const sid='web-user-'+Math.floor(Math.random()*1e9);
  try{
    const r=await fetch('/api/chat/stream?session_id='+sid+'&message='+encodeURIComponent(text));
    const reader=r.body.getReader(); const dec=new TextDecoder(); let buf='';
    while(true){
      const {done,value}=await reader.read(); if(done)break;
      buf+=dec.decode(value,{stream:true});
      let idx;
      while((idx=buf.indexOf('\\n\\n'))>=0){
        const raw=buf.slice(0,idx); buf=buf.slice(idx+2);
        const line=raw.split('\\n').find(l=>l.startsWith('data:'));
        if(!line)continue;
        try{const ev=JSON.parse(line.slice(5));
          if(ev.type==='token') box.innerHTML+=ev.text;
          else if(ev.type==='tool') box.innerHTML+='<div class="tool">🔧 调用 '+ev.tool+' …</div>';
          else if(ev.type==='tool_result') box.innerHTML+='<div class="tool">✅ '+ev.tool+' 完成</div>';
          else if(ev.type==='start'){/* 模式已由服务端拼进第一段文本 */}
          chat.scrollTop=chat.scrollHeight;
        }catch(e){}
      }
    }
  }catch(e){box.innerHTML='连接失败: '+e;}
  btn.disabled=false;
}
document.getElementById('input').addEventListener('keydown',e=>{if(e.key==='Enter')send();});
</script>
</body>
</html>"""


class ChatBody(BaseModel):
    message: str
    session_id: str = "web-user"


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


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
    from fastapi.responses import StreamingResponse

    async def gen():
        agent = get_agent()
        answer = ""
        try:
            async for ev in agent.stream_events(session_id, message):
                if ev["type"] == "done":
                    answer = ev.get("answer", "")
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except RuntimeError as exc:
            yield f"data: {json.dumps({'type':'error','detail':str(exc)}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type':'error','detail':f'内部错误: {exc}'}, ensure_ascii=False)}\n\n"
        finally:
            if answer:
                try:
                    agent.remember(session_id, message, answer)
                except Exception:
                    pass

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web.fastapi_app:app", host="0.0.0.0", port=8001, reload=False)
