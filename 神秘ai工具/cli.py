"""命令行入口。

用法示例：
  # 交互式辅导/科普对话(默认，流式输出)
  python cli.py chat

  # 一次性提问(结果整体返回)
  python cli.py ask "为什么天空是蓝色的？"

  # 把 data/pdfs 下的 PDF 建成 Chroma 索引(知识库检索前提)
  python cli.py build-index

  # 查看某用户的长期画像 / 重置某用户的短期会话
  python cli.py profile --user alice
  python cli.py reset --user alice
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from agent.graph import get_agent
from memory.session import get_store


def _ainput(prompt: str = "") -> str:
    return input(prompt)


async def chat_loop(user_id: str) -> None:
    agent = get_agent()
    store = get_store()
    print("=" * 60)
    print("知友 · AI 学生辅导 + 科普助手(输入 /exit 退出，/profile 看画像)")
    print("=" * 60)
    first = True
    while True:
        try:
            text = _ainput("\n" + ("> " if first else "> "))
        except (EOFError, KeyboardInterrupt):
            print()
            break
        first = False
        text = text.strip()
        if not text:
            continue
        if text in ("/exit", "exit", "quit", "q", "退出"):
            print("再见！学得开心～")
            break
        if text == "/profile":
            p = store.get_profile(user_id)
            print("---- 我的长期画像 ----")
            print(store.to_prompt_text(p))
            continue
        if text == "/reset":
            agent.reset_thread(user_id)
            continue
        if text == "/help":
            print("命令: /exit 退出 · /profile 查看画像 · /reset 清空本次会话(不删画像)")
            continue

        await ask_once(agent, user_id, text, stream=True)


async def ask_once(agent, user_id: str, text: str, stream: bool) -> str:
    answer = ""
    try:
        if stream:
            async for ev in agent.stream_events(user_id, text):
                typ = ev["type"]
                if typ == "start":
                    mode_cn = "📚 辅导模式" if ev["mode"] == "tutor" else "🔭 科普模式"
                    print(f"\n[{mode_cn}]", flush=True)
                elif typ == "token":
                    print(ev["text"], end="", flush=True)
                elif typ == "tool":
                    print(f"\n  🧰 调用工具: {ev['tool']}", flush=True)
                elif typ == "tool_result":
                    content = str(ev["content"]).strip().replace("\n", " ")
                    print(f"     ↳ {content[:120]}{'…' if len(content) > 120 else ''}", flush=True)
                elif typ == "done":
                    answer = ev.get("answer", "")
            print()
        else:
            answer = agent.invoke(user_id, text)
            print("\n" + answer + "\n")
    except RuntimeError as exc:
        print(f"\n[错误] {exc}")
    except Exception as exc:
        print(f"\n[错误] 内部错误: {exc}")
    if answer:
        try:
            agent.remember(user_id, text, answer)
        except Exception:
            pass
    return answer


def build_index(args) -> None:
    from rag.loader import build_index

    count = build_index(args.pdf_dir, force=args.force)
    print(f"建库完成，共 {count} 个块。" if count else "没有新内容入库。")


def main() -> None:
    parser = argparse.ArgumentParser(description="知友 AI：辅导 + 科普 Agent(CLI)")
    parser.add_argument("--user", default="cli-user", help="用户 id(用于长期画像/记忆)")
    sub = parser.add_subparsers(dest="cmd")

    p_chat = sub.add_parser("chat", help="交互式对话")
    p_chat.add_argument("--user", help="用户 id")

    p_ask = sub.add_parser("ask", help="一次性提问")
    p_ask.add_argument("question", nargs="+")
    p_ask.add_argument("--user")

    p_idx = sub.add_parser("build-index", help="扫描 PDF 建 Chroma 索引")
    p_idx.add_argument("--pdf-dir", default=None)
    p_idx.add_argument("--force", action="store_true", help="重建整个索引")

    p_stats = sub.add_parser("rag-stats", help="查看知识库统计")
    p_prof = sub.add_parser("profile", help="查看用户长期画像")
    p_prof.add_argument("--user")
    p_reset = sub.add_parser("reset", help="重置某用户短期会话")
    p_reset.add_argument("--user")

    args = parser.parse_args()
    cmd = args.cmd or "chat"
    user_id = getattr(args, "user", None) or args.user or "cli-user"

    if cmd == "chat":
        asyncio.run(chat_loop(user_id))
    elif cmd == "ask":
        text = " ".join(args.question)
        asyncio.run(ask_once(get_agent(), user_id, text, stream=True))
    elif cmd == "build-index":
        build_index(args)
    elif cmd == "rag-stats":
        from rag import vector_store as vs

        print(vs.index_stats())
    elif cmd == "profile":
        store = get_store()
        print(store.to_prompt_text(store.get_profile(user_id)))
    elif cmd == "reset":
        get_agent().reset_thread(user_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
