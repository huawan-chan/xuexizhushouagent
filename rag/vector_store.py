"""Chroma 向量库：Embedding 生成、持久化、检索。

设计要点：
  - Embedding 用本地 sentence-transformers（免费、离线可推理），首次运行会下载模型。
  - Chroma 持久化到 data/chroma_db，重启不丢。
  - 模块内 lazy import，保证“不装 heavy 依赖也能先跑通 Chat”。
"""
from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from config import get_settings, DATA_DIR

COLLECTION_NAME = "pdf_kb"

_lock = threading.Lock()
_vectorstore = None          # 进程内缓存，避免重复连接
_indexed_docs = None         # 已经建过的源文件清单缓存


# ---------------- Embedding ----------------
@lru_cache(maxsize=1)
def _embeddings():
    """懒加载 HuggingFace Embedding。"""
    from langchain_huggingface import HuggingFaceEmbeddings

    settings = get_settings()
    cache_folder = DATA_DIR / "hf_cache"
    cache_folder.mkdir(parents=True, exist_ok=True)
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
        cache_folder=str(cache_folder),
    )


# ---------------- Chroma ----------------
def _get_vectorstore(force: bool = False):
    """获取(或创建) Chroma 持久化向量库。进程内缓存；force 时删旧库重建。"""
    global _vectorstore
    settings = get_settings()
    with _lock:
        if force and _vectorstore is not None:
            try:
                _vectorstore.delete_collection()
            except Exception:
                pass
            _vectorstore = None
        if _vectorstore is None:
            from langchain_chroma import Chroma

            _vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                persist_directory=str(settings.chroma_dir),
                embedding_function=_embeddings(),
            )
    return _vectorstore


def get_retriever(top_k: Optional[int] = None):
    """RAG 工具使用的检索器(兼容 langchain RetrieverLike)。"""
    settings = get_settings()
    return _get_vectorstore().as_retriever(
        search_kwargs={"k": top_k or settings.rag_top_k}
    )


def index_chunks(chunks: List, force: bool = False) -> int:
    """把 Document 块写入向量库。返回写入数量。"""
    global _indexed_docs
    if not chunks:
        return 0
    vs = _get_vectorstore(force=force)
    # 已入库且非 force 时去重(按 (source, chunk) 查已存 id 前缀)
    if not force:
        ids = [f"{c.metadata.get('source')}:{c.metadata.get('chunk')}" for c in chunks]
        existed = {i for i in vs.get(ids=ids, include=[])["ids"]} if ids else set()
        to_add = [c for c, i in zip(chunks, ids) if i not in existed]
        if not to_add:
            print("[vector_store] 所有块已存在，跳过重复入库")
            return len(chunks)
        chunks = to_add
    texts = [c.page_content for c in chunks]
    metas = [c.metadata for c in chunks]
    ids = [f"{c.metadata.get('source')}:{c.metadata.get('chunk')}" for c in chunks]
    # 批量写；文本为空会抛异常，先过滤
    valid = [(t, m, i) for t, m, i in zip(texts, metas, ids) if t and t.strip()]
    if not valid:
        return 0
    vs.add_texts(
        texts=[v[0] for v in valid],
        metadatas=[v[1] for v in valid],
        ids=[v[2] for v in valid],
    )
    _indexed_docs = None
    return len(valid)


def similarity_search(query: str, top_k: Optional[int] = None) -> List:
    """对外检索接口：返回 Document 列表。"""
    settings = get_settings()
    k = top_k or settings.rag_top_k
    try:
        return _get_vectorstore().similarity_search_with_score(query, k=k)
    except Exception as exc:      # 集合为空时兼容
        print(f"[vector_store] 检索失败: {exc}")
        return []


def index_stats() -> dict:
    """当前向量库里有多少条、来自哪些文件。"""
    try:
        vs = _get_vectorstore()
        coll = vs._collection
        data = coll.get(include=["metadatas"])
        sources: dict = {}
        for m in data.get("metadatas", []) or []:
            src = (m or {}).get("source", "?")
            sources[src] = sources.get(src, 0) + 1
        return {"count": len(data.get("ids", []) or []), "sources": sources}
    except Exception as exc:
        return {"count": 0, "sources": {}, "error": str(exc)}


def has_index() -> bool:
    """data/chroma_db 里是否已有数据。"""
    vs = _get_vectorstore()
    try:
        n = vs._collection.count()
        return n > 0
    except Exception:
        return False


def reset_index() -> None:
    """清空向量库(重建空集合)。"""
    _get_vectorstore(force=True)
    print("[vector_store] 已清空索引")
