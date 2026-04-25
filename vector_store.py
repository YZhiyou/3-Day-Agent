import os
import logging
from pathlib import Path
from dotenv import load_dotenv

from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma

load_dotenv()
logger = logging.getLogger(__name__)


def _get_embeddings() -> DashScopeEmbeddings:
    """统一初始化 DashScope 嵌入模型。"""
    return DashScopeEmbeddings(model="text-embedding-v4")


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    使用 RecursiveCharacterTextSplitter 对文档进行中文友好分割。

    Args:
        documents: 原始文档列表。
        chunk_size: 每块最大字符数。
        chunk_overlap: 块间重叠字符数。

    Returns:
        分割后的文档块列表。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]  # 中文友好
    )
    split_docs = splitter.split_documents(documents)
    logger.info(f"文档已分割为 {len(split_docs)} 个块 (chunk_size={chunk_size}, overlap={chunk_overlap}).")
    return split_docs


def create_vector_store(
    documents: List[Document],
    persist_dir: str = "./data/chroma",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Chroma:
    """
    从原始文档创建 Chroma 向量存储并持久化。

    Args:
        documents: 原始文档列表。
        persist_dir: 持久化目录路径。
        chunk_size: 分块大小。
        chunk_overlap: 分块重叠大小。

    Returns:
        构建完成的 Chroma 向量存储实例。
    """
    # 1. 分割文档
    split_docs = split_documents(documents, chunk_size, chunk_overlap)

    # 2. 初始化嵌入模型
    embeddings = _get_embeddings()

    # 3. 确保持久化目录存在
    Path(persist_dir).parent.mkdir(parents=True, exist_ok=True)

    # 4. 构建 Chroma 向量库并持久化
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    logger.info(f"Vector store created and persisted to '{persist_dir}'.")
    return vectordb


def load_vector_store(
    persist_dir: str = "./data/chroma",
) -> Chroma:
    """
    从持久化目录加载已有的 Chroma 向量存储。

    Args:
        persist_dir: 持久化目录路径。

    Returns:
        加载完成的 Chroma 向量存储实例。

    Raises:
        FileNotFoundError: 如果持久化目录不存在或为空。
    """
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        raise FileNotFoundError(f"在 {persist_dir} 未找到已持久化的向量库。")

    embeddings = _get_embeddings()
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    logger.info(f"已从 '{persist_dir}' 加载向量库。")
    return vectordb


def add_documents(
    vectordb: Chroma,
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[str]:
    """
    向现有向量存储中增量添加文档。

    Args:
        vectordb: 已有的 Chroma 实例。
        documents: 要添加的新文档列表。
        chunk_size: 分块大小。
        chunk_overlap: 分块重叠大小。

    Returns:
        新添加文档对应的 ID 列表。
    """
    split_docs = split_documents(documents, chunk_size, chunk_overlap)
    ids = vectordb.add_documents(split_docs)
    logger.info(f"已向向量库添加 {len(ids)} 个文档块。")
    return ids


def delete_documents(
    vectordb: Chroma,
    ids: Optional[List[str]] = None,
    filter: Optional[Dict[str, Any]] = None
) -> Optional[bool]:
    """
    根据文档 ID 或元数据过滤条件删除向量存储中的文档。

    Args:
        vectordb: Chroma 向量存储实例。
        ids: 要删除的文档 ID 列表。
        filter: 可选的元数据过滤条件。

    Returns:
        删除是否成功，或 None（取决于底层实现）。
    """
    result = vectordb.delete(ids=ids, filter=filter)
    logger.info(f"已删除符合条件的文档 (ids={ids is not None}, filter={filter is not None}).")
    return result


def get_collection_stats(vectordb: Chroma) -> Dict[str, Any]:
    """
    获取向量存储集合的统计信息。

    Args:
        vectordb: Chroma 向量存储实例。

    Returns:
        包含集合名称、文档数量等信息的字典。
    """
    collection = vectordb._collection
    stats = {
        "name": collection.name,
        "count": collection.count(),
        "metadata": collection.metadata,
    }
    logger.info(f"向量库统计信息: {stats}")
    return stats


