"""PDF 加载与切分。

流程：PDF 目录 -> 逐页提取文本(pypdf) -> 清洗 -> 按块切分(保留来源页码)。
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from config import get_settings


def extract_text_from_pdf(pdf_path: Path) -> str:
    """用 pypdf 提取整份 PDF 文本，返回带分页标记的纯文本。"""
    from pypdf import PdfReader

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")
    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append(f"\n[第{i}页] {text}")
    full = "\n".join(pages).replace("\x00", "")
    return full


def load_pdfs(pdf_dir: Path) -> List[dict]:
    """扫描目录下所有 .pdf，返回 [{file, text, page_hint}]。扫描版(无文字层)会被跳过。"""
    settings = get_settings()
    pdf_dir = Path(pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    docs: List[dict] = []
    files = sorted(pdf_dir.glob("*.pdf")) + sorted(pdf_dir.glob("*.PDF"))
    seen = set()
    for f in files:
        if f in seen:
            continue
        seen.add(f)
        try:
            text = extract_text_from_pdf(f)
        except Exception as exc:                     # 单个坏文件不影响整体建库
            print(f"[loader] 跳过 {f.name}: {exc}")
            continue
        if not text.strip():
            print(f"[loader] 跳过无文字内容(可能是扫描版图片PDF): {f.name}")
            continue
        docs.append({"file": f.name, "path": str(f), "text": text})
        print(f"[loader] 已加载 {f.name} ({len(text)} 字符)")
    return docs


def chunk_documents(docs: List[dict], chunk_size: int | None = None,
                    chunk_overlap: int | None = None) -> List:
    """把整份文本切成长文档块，并带上来源与页码元数据。

    使用 langchain RecursiveCharacterTextSplitter，中文按常见分隔符递归切分。
    """
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ". ", " "],
        length_function=len,
    )
    chunks: List[Document] = []
    for doc in docs:
        # 每个块独立分页切分，块与块之间不跨文件
        raw_chunks = splitter.split_text(doc["text"])
        for idx, raw in enumerate(raw_chunks):
            page = "?"
            # 取块内第一个分页标记作为大致页码(格式 [第N页])
            for line in raw.splitlines():
                if line.startswith("[第") and "页]" in line:
                    page = line.strip("[]页")
                    break
            metadata = {"source": doc["file"], "page": page, "chunk": idx}
            chunks.append(Document(page_content=raw.strip(), metadata=metadata))
    return chunks


def build_index(pdf_dir: Path | None = None, force: bool = False) -> int:
    """一步到位：扫描 PDF -> 切分 -> 写入 Chroma。返回入库块数。"""
    from rag.vector_store import index_chunks

    settings = get_settings()
    pdf_dir = Path(pdf_dir) if pdf_dir else settings.pdf_dir
    docs = load_pdfs(pdf_dir)
    if not docs:
        print(f"[loader] {pdf_dir} 下没有可用的 PDF，请先放入教材/讲义/资料。")
        return 0
    chunks = chunk_documents(docs)
    count = index_chunks(chunks, force=force)
    print(f"[loader] 建库完成: {len(docs)} 份 PDF -> {count} 个向量块")
    return count


if __name__ == "__main__":
    # 用法: python -m rag.loader
    import sys

    arg = sys.argv[1] if len(sys.argv) > 1 else None
    build_index(Path(arg) if arg else None)
