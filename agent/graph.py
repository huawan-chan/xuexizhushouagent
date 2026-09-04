"""LangGraph 智能体主图(ReAct)。

架构：
  Router(模式分类) -> Agent(LLM+工具)  <->  ToolNode(执行工具)  循环，
  直到 Agent 不再调用工具，输出最终回答。

特性：
  - 工具：计算器 / RAG知识库 / 网页搜索(见 agent/tools.py)
  - 短程记忆：SQLite Checkpointer(thread_id=user_id)，对话跨轮持久化
  - 长程记忆：agent/graph 负责读取画像文本 -> 注入 prompt，画像更新在业务层做
  - 流式：create_react_agent 天然支持 token 级流式，前端可逐字输出
"""
from __future__ import annotations

import re
from typing import Annotated, Any, AsyncIterator, Callable, List, Optional, TypedDict

from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

from agent.prompts import build_system_prompt
from agent.tools import get_tools
from config import get_settings
from memory.session import get_store


# =====================================================================
# 状态定义
# =====================================================================
class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    profile_text: str


# =====================================================================
# 意图路由(轻量关键词分类，避免每次多花一次 LLM 调用)
# =====================================================================
_SUBJECT_WORDS = [
    "数学", "算术", "几何", "代数", "函数", "方程", "不等式", "导数", "积分",
    "数列", "向量", "概率", "三角函数", "物理", "化学", "生物", "英语",
    "语文", "文言文", "作文", "光合作用", "电路", "力学",
]
_TUTOR_HINT = [
    "教我做", "帮我辅导", "辅导", "怎么做", "怎么解", "这题", "这道题", "题目",
    "作业", "不会", "求解", "证明", "解题", "讲讲", "讲一下", "帮忙算", "解析",
    "练习", "考试", "错题", "订正",
]
_SCIENCE_HINT = [
    "科普", "为什么", "是怎么回事", "是什么原因", "如何发生", "原理是什么",
    "介绍一下", "是什么", "百科", "冷知识", "怎么回事",
]


def detect_mode(text: str) -> str:
    """判断用户意图: 'tutor'(学习辅导) | 'science'(科普问答)。"""
    t = text.strip()
    strong_tutor = any(w in t for w in _TUTOR_HINT)
    strong_science = bool(re.search(r"^为什么|科普|冷知识", t)) and not strong_tutor
    if strong_science:
        return "science"
    if strong_tutor:
        return "tutor"
    has_subject = any(w in t for w in _SUBJECT_WORDS)
    if has_subject:
        return "tutor"
    if re.search(r"为什么|怎么回事|是什么|怎么形成|怎么产生|有何原理", t):
        return "science"
    return "science"   # 默认科普(面向大众提问)


def _last_human_text(messages: List[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


# =====================================================================
# Prompt 工厂：随每轮会话动态注入「模式规则 + 学生画像」
# =====================================================================
def make_prompt_fn() -> Callable[[dict], List[BaseMessage]]:
    def _fn(state: dict) -> List[BaseMessage]:
        msgs = state.get("messages", [])
        text = _last_human_text(msgs)
        mode = detect_mode(text)
        profile_text = state.get("profile_text", "暂无历史画像(新学生)。")
        return [SystemMessage(content=build_system_prompt(mode, profile_text))]

    return _fn


# =====================================================================
# LLM 工厂
# =====================================================================
def build_llm():
    from langchain_groq import ChatGroq

    settings = get_settings()
    return ChatGroq(
        model=settings.groq_model,
        temperature=settings.temperature,
        api_key=settings.require_groq_key(),
    )


# =====================================================================
# 会话历史裁剪(避免无限增长；保证不切断 tool 调用对)
# =====================================================================
def _history_window(messages: List[BaseMessage], max_count: int) -> List[BaseMessage]:
    if max_count <= 0 or len(messages) <= max_count:
        return list(messages)
    msgs = list(messages)
    cut = len(msgs) - max_count
    # 从裁剪点向后推进，直到落在正常消息(而非 tool 结果)上
    while cut < len(msgs) and msgs[cut].type == "tool":
        cut += 1
    return msgs[cut:]


# =====================================================================
# 图构建
# =====================================================================
def build_graph(checkpointer: Any = None) -> Any:
    """构建 ReAct 图。

    checkpointer 传入则启用跨轮记忆(SQLite/MemorySaver)。
    注意：这里的 Router 已并入 create_react_agent 的动态 prompt，
    实际图节点由 create_react_agent 生成(agent <-> tools)。
    """
    settings = get_settings()
    llm = build_llm()
    tools = get_tools(with_web=settings.web_search_enabled)
    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_schema=AgentState,
        prompt=make_prompt_fn(),
        checkpointer=checkpointer,
    )
    return agent


# =====================================================================
# Agent 封装：对外提供 invoke / 流式 / 记忆
# =====================================================================
class TutorAgent:
    def __init__(self, graph: Any = None, store=None, llm=None):
        self.settings = get_settings()
        self.graph = graph if graph is not None else build_graph()
        self.store = store or get_store()
        self.llm = llm

    # ---------------- 内部工具 ----------------
    def config(self, user_id: str) -> dict:
        return {"configurable": {"thread_id": user_id}}

    def _inputs(self, user_id: str, history: List[BaseMessage], text: str) -> dict:
        profile = self.store.get_profile(user_id)
        return {
            "messages": [*history, HumanMessage(content=text)],
            "user_id": user_id,
            "profile_text": self.store.to_prompt_text(profile),
        }

    # ---------------- 一次性问答(非流式) ----------------
    def invoke(self, user_id: str, text: str) -> str:
        config = self.config(user_id)
        try:
            state = self.graph.get_state(config)
            history = _history_window(
                state.values.get("messages", []), self.settings.max_history
            )
        except Exception:
            history = []
        result = self.graph.invoke(self._inputs(user_id, history, text), config)
        for m in reversed(result.get("messages", [])):
            if isinstance(m, AIMessageChunk) or getattr(m, "content", None):
                if m.type == "ai" and getattr(m, "content", ""):
                    return m.content if isinstance(m.content, str) else str(m.content)
        return ""

    # ---------------- 流式问答 ----------------
    async def stream_events(self, user_id: str, text: str) -> AsyncIterator[dict]:
        """产出事件流：
          {'type':'start', 'mode':...}
          {'type':'token', 'text':...}            最终回答文本 token(逐字)
          {'type':'tool', 'tool':...}             模型决定调用某工具
          {'type':'tool_result','tool':..,'content':..}
          {'type':'done','answer':完整回答}
        """
        config = self.config(user_id)
        try:
            state = await self.graph.aget_state(config)
            history = _history_window(
                state.values.get("messages", []), self.settings.max_history
            )
        except Exception:
            history = []
        mode = detect_mode(text)
        yield {"type": "start", "mode": mode}
        inputs = self._inputs(user_id, history, text)
        tokens: List[str] = []
        try:
            async for chunk in self.graph.astream(inputs, config, stream_mode="messages"):
                if not isinstance(chunk, tuple) or len(chunk) != 2:
                    continue
                message, meta = chunk
                node = meta.get("langgraph_node", "") if isinstance(meta, dict) else ""
                if isinstance(message, ToolMessage):      # 工具执行结果
                    content = str(message.content)
                    tokens.append(content)
                    yield {"type": "tool_result", "tool": message.name or "?", "content": content[:500]}
                elif isinstance(message, AIMessageChunk):
                    if getattr(message, "tool_call_chunks", None):
                        yield {"type": "tool", "tool": (message.tool_call_chunks[0].get("name") or "?")}
                    content = message.content
                    if content:
                        tokens.append(content if isinstance(content, str) else str(content))
                        yield {"type": "token", "text": content if isinstance(content, str) else str(content)}
        finally:
            # 结束：取最终回答(用于画像更新等)
            try:
                end_state = await self.graph.aget_state(config)
                msgs = end_state.values.get("messages", [])
                answer = ""
                for m in reversed(msgs):
                    if m.type == "ai" and getattr(m, "content", ""):
                        answer = m.content if isinstance(m.content, str) else str(m.content)
                        break
            except Exception:
                answer = "".join(tokens)
        yield {"type": "done", "answer": answer}

    # ---------------- 记忆落库(短期由 checkpointer 自动完成) ----------------
    def remember(self, user_id: str, user_text: str, answer: str) -> None:
        """记录本轮对话并刷新长期画像(规则为主；可开 LLM 归纳偏好)。"""
        self.store.record_exchange(user_id, "user", user_text)
        self.store.record_exchange(user_id, "assistant", answer[:2000])
        self.store.increment_turn(user_id)
        llm = self.llm if self.settings.profile_llm_on else None
        self.store.refresh_profile(user_id, llm=llm)

    def reset_thread(self, user_id: str) -> None:
        """清空某个用户的短期会话(删除 checkpoint + 保留画像)。"""
        try:
            # 删除配置对应的检查点(不同 saver 能力不同，失败可忽略)
            self.graph.get_state(self.config(user_id))
        except Exception:
            pass
        print(f"[memory] 已重置用户 {user_id} 的短期会话")


# =====================================================================
# 工厂：全局单例 + SQLite checkpointer(短期记忆持久化)
# =====================================================================
_agent_cache: Optional[TutorAgent] = None


def get_agent() -> TutorAgent:
    """全局唯一 Agent(短期记忆先试 SqliteSaver，失败退回 MemorySaver)。"""
    global _agent_cache
    if _agent_cache is None:
        settings = get_settings()
        saver = None
        # 1) 首选：SQLite 检查点(跨重启持久化)
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            saver = SqliteSaver.from_conn_string(str(settings.checkpoint_db))
            if hasattr(saver, "setup"):
                saver.setup()
        except Exception as exc:
            print(f"[warning] SQLite checkpointer 不可用: {exc}")
            saver = None
        # 2) 兜底：进程内 MemorySaver(重启后短期会话丢失，长期画像仍保留)
        if saver is None:
            try:
                from langgraph.checkpoint.memory import MemorySaver

                saver = MemorySaver()
            except Exception:
                saver = None
        _agent_cache = TutorAgent(
            graph=build_graph(checkpointer=saver),
            has_checkpointer=saver is not None,
        )
    return _agent_cache
