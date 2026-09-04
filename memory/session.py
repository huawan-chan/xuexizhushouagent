"""长期用户画像 / 会话记忆(sqlite)。

职责：
  1. 用 sqlite 持久化每个用户(user_id = thread_id)的学习画像：
     年级、科目、能力水平、薄弱点、已掌握点、互动次数、最近话题。
  2. 记录最近若干轮对话(原始文本)作为画像提取的原料。
  3. 提供“画像 -> Prompt 文本”的序列化，方便注入系统提示词。

短期上下文由 LangGraph 的 checkpointer(SQLite)负责，见 agent/graph.py。
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import get_settings

DEFAULT_PROFILE: dict = {
    "grade": "",          # 年级/学段
    "subjects": [],       # 涉及的学科
    "level": "",          # 基础 / 中等 / 进阶
    "weak_areas": [],     # 薄弱点
    "mastered": [],       # 已掌握
    "preference": "",     # 学习偏好(由 LLM 归纳，可选)
    "meta": {"turns": 0, "last_topics": []},
}

# 关键词 -> 年级
_GRADE_RULES = [
    (r"小学|三年级|四年级|五年级|六年级", "小学"),
    (r"初一|七年级", "初一"),
    (r"初二|八年级", "初二"),
    (r"初三|九年级|中考", "初三"),
    (r"高一|十年级", "高一"),
    (r"高二|十一年级", "高二"),
    (r"高三|十二年级|高考", "高三"),
    (r"大学|本科|高等数学|微积分", "大学"),
    (r"考研|研究生", "考研"),
]

# 关键词 -> 学科
_SUBJECT_RULES = [
    ("数学", ["数学", "算术", "几何", "代数", "函数", "方程", "微积分", "积分", "导数", "三角函数", "数列", "概率"]),
    ("物理", ["物理", "力学", "牛顿", "加速度", "电学", "电路", "光学", "磁场", "功和能", "动量", "热学"]),
    ("化学", ["化学", "元素", "摩尔", "酸碱", "氧化还原", "方程式配平", "有机化学", "周期表"]),
    ("生物", ["生物", "细胞", "光合作用", "DNA", "基因", "遗传", "生态系统", "酶"]),
    ("英语", ["英语", "语法", "单词", "阅读理解", "完形填空", "作文"]),
    ("语文", ["语文", "文言文", "古诗", "阅读理解", "写作"]),
]

_LEVEL_UP_RULES = [
    (r"懂了|会了|明白了|理解了|掌握了|看懂了", "进阶"),
    (r"大概懂|有点懂|半懂|还可以", "中等"),
    (r"完全不会|不懂|听不懂|一脸懵|不会做|太难", "基础"),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ProfileStore:
    """线程安全的 sqlite 画像存取。"""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS user_profiles(
                    user_id    TEXT PRIMARY KEY,
                    profile    TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS user_exchanges(
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id    TEXT NOT NULL,
                    role       TEXT NOT NULL,          -- user | assistant | tool
                    content    TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_ex_user
                    ON user_exchanges(user_id, id);
                """
            )
            self._conn.commit()

    # ---------- 基本存取 ----------
    def get_profile(self, user_id: str) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT profile FROM user_profiles WHERE user_id=?", (user_id,)
            ).fetchone()
        if row is None:
            return json.loads(json.dumps(DEFAULT_PROFILE))  # 深拷贝
        try:
            return {**json.loads(row["profile"])}          # 浅合并默认字段
        except json.JSONDecodeError:
            return json.loads(json.dumps(DEFAULT_PROFILE))

    def save_profile(self, user_id: str, profile: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO user_profiles(user_id, profile, updated_at) VALUES(?,?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET profile=excluded.profile, "
                "updated_at=excluded.updated_at",
                (user_id, json.dumps(profile, ensure_ascii=False), _now()),
            )
            self._conn.commit()

    def record_exchange(self, user_id: str, role: str, content: str) -> None:
        """记录一轮原始文本，并只保留最近 30 条，防止 sqlite 无限膨胀。"""
        with self._lock:
            self._conn.execute(
                "INSERT INTO user_exchanges(user_id, role, content, created_at) "
                "VALUES(?,?,?,?)",
                (user_id, role, content, _now()),
            )
            self._conn.execute(
                "DELETE FROM user_exchanges WHERE user_id=? AND id NOT IN ("
                " SELECT id FROM user_exchanges WHERE user_id=? ORDER BY id DESC LIMIT 30)",
                (user_id, user_id),
            )
            self._conn.commit()

    def recent_exchanges(self, user_id: str, limit: int = 12) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM user_exchanges "
                "WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def increment_turn(self, user_id: str) -> None:
        p = self.get_profile(user_id)
        meta = p.setdefault("meta", {})
        meta["turns"] = int(meta.get("turns", 0)) + 1
        self.save_profile(user_id, p)

    # ---------- 画像提取(规则版，免 LLM) ----------
    def extract_rule_based(self, exchanges: list[dict]) -> dict:
        """从最近的对话里按关键词提取画像增量，与旧画像合并见 refresh_profile。"""
        text_all = "\n".join(e["content"] for e in exchanges if e["role"] == "user")
        if not text_all:
            return {}
        delta: dict = {"subjects": []}
        for grade, pat in _GRADE_RULES:
            if re.search(pat, text_all):
                delta["grade"] = grade
                break
        for subj, words in _SUBJECT_RULES:
            if any(w in text_all for w in words):
                delta["subjects"].append(subj)
        level = ""
        for pat, lv in _LEVEL_UP_RULES:   # 越靠后优先级越低
            if re.search(pat, text_all):
                level = lv
        if level:
            delta["level"] = level
        # 弱项：带否定词的主观句
        weak = [s for s in re.split(r"[。！？!?\n]", text_all)
                if re.search(r"不会|不懂|没听懂|不清楚|没学会|卡在|总是错|分不清", s) and len(s) <= 60]
        delta["weak_areas"] = [s.strip("，,、 ") for s in weak][:5]
        # 话题
        topics = [s.strip("，,、 ?？") for s in re.split(r"[。？！\n]", text_all)
                  if re.search(r"(辅导|讲讲|怎么|为什么|如何|计算|求|教|帮我)", s)][:3]
        delta["meta"] = {"last_topics": topics}
        return delta

    def refresh_profile(self, user_id: str, exchanges: Optional[list[dict]] = None,
                        llm: Optional[object] = None) -> dict:
        """读取旧画像 -> 用最近对话提取增量 -> 合并写回。返回最新画像。

        llm 非空且 settings.profile_llm_on 为真时，会额外让 LLM 归纳学习偏好；
        LLM 失败不影响规则结果。
        """
        old = self.get_profile(user_id)
        exchanges = exchanges if exchanges is not None else self.recent_exchanges(user_id)
        delta = self.extract_rule_based(exchanges)
        self._merge_delta(old, delta)
        if llm is not None and get_settings().profile_llm_on:
            try:
                old["preference"] = self._summarize_with_llm(llm, exchanges)
            except Exception:
                pass  # LLM 归纳失败不阻断
        self.save_profile(user_id, old)
        return old

    @staticmethod
    def _merge_delta(profile: dict, delta: dict) -> None:
        if delta.get("grade"):
            profile["grade"] = delta["grade"]
        if delta.get("subjects"):
            seen = set(profile.get("subjects", []))
            for s in delta["subjects"]:
                if s not in seen:
                    profile.setdefault("subjects", []).append(s)
                    seen.add(s)
        if delta.get("level"):
            # 只往上提档，不轻易降级，避免一句“太难”就清零进度
            order = {"基础": 0, "中等": 1, "进阶": 2}
            if order.get(delta["level"], 1) >= order.get(profile.get("level", ""), -1):
                profile["level"] = delta["level"]
        if delta.get("weak_areas"):
            wset = set(profile.get("weak_areas", []))
            for w in delta["weak_areas"]:
                if w not in wset and len(profile.get("weak_areas", [])) < 6:
                    profile.setdefault("weak_areas", []).append(w)
                    wset.add(w)
        if delta.get("meta"):
            profile.setdefault("meta", {}).update(delta["meta"])
        profile["meta"]["turns"] = int(profile.get("meta", {}).get("turns", 0))

    @staticmethod
    def _summarize_with_llm(llm, exchanges: list[dict]) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        convo = "\n".join(
            f"{'学生' if e['role']=='user' else '老师'}: {e['content'][:200]}"
            for e in exchanges[-6:]
        )
        resp = llm.invoke(
            [
                SystemMessage(content="你是学情分析师。请用不超过 60 个字，概括该学生最近对话体现出的学习偏好/风格"
                                      "(例如：喜欢类比、需要图形辅助、节奏偏慢、爱追问原理)。只输出结论，不要寒暄。"),
                HumanMessage(content=convo),
            ]
        )
        return (resp.content or "").strip()

    def to_prompt_text(self, profile: dict) -> str:
        """把画像渲染成系统提示词里的一段文字。"""
        parts = []
        if profile.get("grade"):
            parts.append(f"- 学段/年级: {profile['grade']}")
        if profile.get("subjects"):
            parts.append(f"- 涉及学科: {', '.join(profile['subjects'])}")
        if profile.get("level"):
            parts.append(f"- 估计水平: {profile['level']}")
        if profile.get("weak_areas"):
            parts.append(f"- 薄弱点: {'; '.join(profile['weak_areas'])}")
        if profile.get("mastered"):
            parts.append(f"- 已掌握: {'; '.join(profile['mastered'])}")
        if profile.get("preference"):
            parts.append(f"- 学习偏好: {profile['preference']}")
        if profile.get("meta", {}).get("last_topics"):
            parts.append(f"- 最近话题: {', '.join(profile['meta']['last_topics'])}")
        if not parts:
            return "暂无历史画像(新学生)。"
        return "该学生画像如下，辅导时要因材施教：\n" + "\n".join(parts)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# 全局单例(进程内复用)
_store: Optional[ProfileStore] = None


def get_store(db_path: Optional[Path] = None) -> ProfileStore:
    global _store
    if _store is None:
        _store = ProfileStore(db_path or get_settings().memory_db)
    return _store
