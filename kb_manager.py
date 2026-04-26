import os
import shutil
import logging
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from vector_store import add_documents, load_vector_store, create_vector_store
from document_loader import load_documents, SUPPORTED_EXT

logger = logging.getLogger(__name__)


def _load_single_file(file_path: str) -> List[Document]:
    """加载单个文件为 Document 列表。"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXT:
        raise ValueError(f"不支持的文件类型: {ext}，仅支持 {list(SUPPORTED_EXT.keys())}")

    from langchain_community.document_loaders import PyPDFLoader, TextLoader

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".md" or ext == ".txt":
        loader = TextLoader(file_path, autodetect_encoding=True)
    else:
        raise ValueError(f"未处理的文件类型: {ext}")

    docs = loader.load()
    for doc in docs:
        doc.metadata.setdefault("source", file_path)
        doc.metadata["file_type"] = SUPPORTED_EXT[ext]
    return docs


def add_file_to_kb(vectordb: Chroma, file_path: str) -> str:
    """
    将单个文件添加到知识库。

    Args:
        vectordb: Chroma 向量库实例。
        file_path: 要添加的文件路径。

    Returns:
        操作结果描述字符串。
    """
    if not os.path.exists(file_path):
        return f"文件不存在: {file_path}"

    try:
        docs = _load_single_file(file_path)
        if not docs:
            return f"文件内容为空或加载失败: {file_path}"
        add_documents(vectordb, docs)
        return f"已成功添加文件到知识库: {os.path.basename(file_path)} (共 {len(docs)} 个片段)"
    except Exception as e:
        logger.exception("添加文件到知识库失败")
        return f"添加失败: {e}"


def delete_file_from_kb(vectordb: Chroma, file_path: str) -> str:
    """
    从知识库中删除指定文件的所有文档块。

    Args:
        vectordb: Chroma 向量库实例。
        file_path: 要删除的文件路径（与添加时的 source 元数据匹配）。

    Returns:
        操作结果描述字符串。
    """
    try:
        result = vectordb.get(where={"source": file_path})
        ids = result.get("ids", [])
        if not ids:
            return f"知识库中未找到该文件的记录: {file_path}"
        vectordb.delete(ids=ids)
        return f"已删除文件 '{os.path.basename(file_path)}' 的 {len(ids)} 个文档块。"
    except Exception as e:
        logger.exception("从知识库删除文件失败")
        return f"删除失败: {e}"


def rebuild_kb(persist_dir: str, docs_directory: str) -> Chroma:
    """
    重建知识库：清空旧向量库，重新加载指定目录下的所有文档。

    Args:
        persist_dir: 向量库持久化目录。
        docs_directory: 文档源目录。

    Returns:
        新创建的 Chroma 向量库实例。
    """
    import gc
    import time

    # 强制垃圾回收，释放可能持有的文件句柄
    gc.collect()

    if not os.path.exists(docs_directory):
        raise FileNotFoundError(f"文档目录不存在: {docs_directory}")

    docs = load_documents(docs_directory)
    if not docs:
        logger.warning(f"在 {docs_directory} 中未找到任何支持的文档。")

    # Windows 上 HNSW 索引文件（data_level0.bin）的 mmap 锁非常顽固，
    # 物理删除目录经常失败。因此采用"复用目录 + 逻辑清空 collection"策略。
    if os.path.exists(persist_dir):
        try:
            existing_db = load_vector_store(persist_dir)
            # 优先尝试直接删除整个 collection（释放 HNSW 索引文件最彻底）
            try:
                existing_db._client.delete_collection("langchain")
                logger.info("已删除现有向量库 collection。")
            except Exception:
                # 回退：逐个删除所有文档
                result = existing_db.get()
                existing_ids = result.get("ids", []) if result else []
                if existing_ids:
                    existing_db.delete(ids=existing_ids)
                    logger.info(f"已清空现有向量库中的 {len(existing_ids)} 个文档。")
            # 关闭底层 client，给 Windows 释放文件锁的机会
            if hasattr(existing_db, "_client") and hasattr(existing_db._client, "close"):
                existing_db._client.close()
            del existing_db
            gc.collect()
            time.sleep(1.0)
        except Exception as e:
            logger.warning(f"无法复用现有向量库，尝试物理删除目录: {e}")
            max_retries = 10
            for i in range(max_retries):
                try:
                    shutil.rmtree(persist_dir)
                    logger.info(f"已删除旧向量库目录: {persist_dir}")
                    break
                except (PermissionError, OSError) as exc:
                    if i < max_retries - 1:
                        time.sleep(1.0)
                        gc.collect()
                    else:
                        raise PermissionError(
                            f"无法删除旧向量库目录 '{persist_dir}'，文件仍被占用。"
                            f"请尝试重启 Streamlit 应用后再重建。原始错误: {exc}"
                        ) from exc

    vectordb = create_vector_store(docs, persist_dir=persist_dir)
    logger.info(f"知识库重建完成，持久化到: {persist_dir}")
    return vectordb


def search_kb(vectordb: Chroma, query: str, k: int = 4) -> str:
    """
    在知识库中搜索与查询最相似的文档。

    Args:
        vectordb: Chroma 向量库实例。
        query: 搜索查询文本。
        k: 返回结果数量。

    Returns:
        格式化后的搜索结果字符串。
    """
    try:
        docs = vectordb.similarity_search(query, k=k)
        if not docs:
            return "未找到相关文档。"

        lines = [f"搜索 '{query}' 的结果（共 {len(docs)} 条）:\n"]
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知来源")
            content = doc.page_content[:300].replace("\n", " ")
            lines.append(f"[{i}] 来源: {source}\n    {content}...\n")
        return "\n".join(lines)
    except Exception as e:
        logger.exception("知识库搜索失败")
        return f"搜索失败: {e}"
