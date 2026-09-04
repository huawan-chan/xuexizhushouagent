"""Chainlit Web 界面。

启动：
    chainlit run web/chainlit_app.py -w      # -w 热重载(可选)
"""
from __future__ import annotations

import chainlit as cl
from chainlit.types import ThreadDict

from agent.graph import get_agent
from memory.session import get_store


def _user_id() -> str:
    """用 Chainlit 会话 id 作为用户画像主键(web 端匿名用户)。"""
    uid = cl.user_session.get("uid")
    if not uid:
        uid = cl.user_session.get("id") or "web-user"
        cl.user_session.set("uid", uid)
    return uid


@cl.on_chat_start
async def on_chat_start() -> None:
    user_id = _user_id()
    profile = get_store().get_profile(user_id)
    meta = profile.get("meta", {})
    times = meta.get("turns", 0)
    if times:
        msg = (
            f"欢迎回来！这是我们第 {times + 1} 次对话。\n\n"
            f"我会记住你的水平与薄弱点。之前了解到的信息:\n"
            + get_store().to_prompt_text(profile)
            + "\n\n直接告诉我今天想学什么，或问任何『为什么』吧。"
        )
    else:
        msg = (
            "你好，我是「知友」——会引导你自己想明白问题的 AI 老师。\n\n"
            "• 想让我**辅导作业/讲题** → 把题目发给我（我不会直接给答案，会一步步问你）\n"
            "• 想听**科普** → 直接问“为什么天空是蓝色的？”这类问题\n\n"
            "开始吧！"
        )
    await cl.Message(content=msg).send()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict) -> None:
    user_id = _user_id()
    await cl.Message(
        content=f"继续上次的对话。你的画像：{get_store().to_prompt_text(get_store().get_profile(user_id))}"
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    user_id = _user_id()
    text = message.content.strip()
    if not text:
        return
    agent = get_agent()
    answer = ""
    main_msg = cl.Message(content="")
    tool_note: cl.Message | None = None

    try:
        async for ev in agent.stream_events(user_id, text):
            typ = ev["type"]
            if typ == "start":
                mode_cn = "📚 学习辅导(苏格拉底引导)" if ev["mode"] == "tutor" else "🔭 科普问答(What-Why-Wow)"
                await main_msg.stream_token(f"【{mode_cn}】\n\n")
                await main_msg.send()
            elif typ == "token":
                await main_msg.stream_token(ev["text"])
            elif typ == "tool":
                await main_msg.stream_token(f"\n\n_🔧 正在调用 {ev['tool']}…_\n\n")
            elif typ == "tool_result":
                content = ev["content"].strip().replace("\n", " ")
                if len(content) > 180:
                    content = content[:180] + "…"
                await main_msg.stream_token(f"_✅ {ev['tool']}: {content}_\n\n")
            elif typ == "done":
                answer = ev.get("answer", "")
    except Exception as exc:
        await cl.Message(content=f"出错了：{exc}\n\n请检查 `.env` 里 GROQ_API_KEY 是否配置正确。").send()
        return

    await main_msg.update()
    # 记忆：记录本轮并把画像写回 sqlite
    try:
        agent.remember(user_id, text, answer)
    except Exception:
        pass
