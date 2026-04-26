import os
import re
import logging
from pathlib import Path
from dotenv import load_dotenv

from typing import List, Tuple, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma

load_dotenv()
logger = logging.getLogger(__name__)


def _get_embeddings() -> DashScopeEmbeddings:
    """统一初始化 DashScope 嵌入模型。"""
    return DashScopeEmbeddings(model="text-embedding-v4")


# ---------- Markdown 论文专用保护性分块 ----------

# 需要作为原子单元保护的正则模式（避免在内部切断）
_ATOMIC_BLOCK_PATTERNS: List[Tuple[str, str]] = [
    (r"```[\s\S]*?```", "code_block"),           # 围栏代码块
    (r"\$\$[\s\S]*?\$\$", "math_block"),          # 块级 LaTeX 公式
    (r"(?<!\$)\$(?!\$)[^\$\n]*?\$(?!\$)", "math_inline"),  # 行内 LaTeX 公式
    (r"(?:\|[^\n]*\|[\r\n]?)+", "table"),         # Markdown 表格（简化匹配）
]


def _extract_atomic_units(text: str) -> List[Tuple[str, str]]:
    """
    将文本拆分为原子单元，识别并标记需要保护的特殊块。

    返回的每个元组为 (unit_type, content)，unit_type 取值：
    "code_block" | "math_block" | "math_inline" | "table" | "text"
    """
    # 1. 找出所有保护块的位置
    blocks: List[Tuple[int, int, str, str]] = []
    for pattern, block_type in _ATOMIC_BLOCK_PATTERNS:
        for match in re.finditer(pattern, text):
            blocks.append((match.start(), match.end(), match.group(0), block_type))

    # 2. 按起始位置排序
    blocks.sort(key=lambda x: x[0])

    # 3. 合并重叠的块（保留最长的）
    merged: List[Tuple[int, int, str, str]] = []
    for block in blocks:
        if merged and block[0] < merged[-1][1]:
            # 重叠：保留长度更大的
            if block[1] - block[0] > merged[-1][1] - merged[-1][0]:
                merged[-1] = block
        else:
            merged.append(block)

    # 4. 构建原子单元列表
    units: List[Tuple[str, str]] = []
    last_end = 0
    for start, end, content, block_type in merged:
        if start > last_end:
            units.append(("text", text[last_end:start]))
        units.append((block_type, content))
        last_end = end
    if last_end < len(text):
        units.append(("text", text[last_end:]))

    return units


def _group_units_into_chunks(
    units: List[Tuple[str, str]],
    chunk_size: int,
    chunk_overlap: int
) -> List[str]:
    """
    将原子单元组合成 chunk，尽量不在保护块中间切断。
    """
    chunks: List[str] = []
    current_units: List[Tuple[str, str]] = []
    current_size = 0

    i = 0
    while i < len(units):
        unit_type, content = units[i]
        content_len = len(content)

        # 如果单个原子单元就超过 chunk_size，只能强行独占一个 chunk
        if content_len > chunk_size:
            if current_units:
                chunks.append("".join(u[1] for u in current_units))
                current_units = []
                current_size = 0
            chunks.append(content)
            i += 1
            continue

        # 如果加入当前单元会超出限制，则先结束当前 chunk
        if current_size + content_len > chunk_size and current_units:
            chunks.append("".join(u[1] for u in current_units))

            # 构建 overlap：从尾部尽可能多地取单元
            overlap_units: List[Tuple[str, str]] = []
            overlap_size = 0
            for u in reversed(current_units):
                u_len = len(u[1])
                if overlap_size + u_len > chunk_overlap:
                    break
                overlap_units.insert(0, u)
                overlap_size += u_len

            current_units = overlap_units
            current_size = overlap_size

        current_units.append((unit_type, content))
        current_size += content_len
        i += 1

    if current_units:
        chunks.append("".join(u[1] for u in current_units))

    return chunks


def split_markdown_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    针对 Markdown 论文的专用分块策略。

    分块流程：
    1. 一级分割：按 Markdown 标题层级（# / ## / ### / ####）分割，保留章节结构；
       每个 chunk 的 metadata 中会携带 Header 1 / Header 2 等标题路径。
    2. 二级分割：对仍然超过 chunk_size 的章节，使用原子单元保护性分割：
       - 代码块、块级/行内数学公式、表格 被视为不可分割的原子单元；
       - 分割优先发生在普通文本的段落/句子边界，尽量避免破坏上述结构。

    Args:
        documents: 原始 Document 列表（要求 metadata 中已标记 file_type="markdown"）。
        chunk_size: 每块最大字符数。
        chunk_overlap: 块间重叠字符数。

    Returns:
        分割后的文档块列表。
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ],
        strip_headers=False,  # 保留标题行在内容中，方便定位
    )

    all_chunks: List[Document] = []

    for doc in documents:
        # 一级：按标题层级分割
        header_docs = header_splitter.split_text(doc.page_content)

        for h_doc in header_docs:
            # 合并原始文档的元数据（source、file_type 等）
            h_doc.metadata.update(doc.metadata)

            # 如果大小合适，直接保留
            if len(h_doc.page_content) <= chunk_size:
                all_chunks.append(h_doc)
                continue

            # 二级：原子单元保护性分割
            units = _extract_atomic_units(h_doc.page_content)
            sub_contents = _group_units_into_chunks(units, chunk_size, chunk_overlap)

            for content in sub_contents:
                # 过滤掉纯空白 chunk
                if not content or not content.strip():
                    continue
                all_chunks.append(
                    Document(page_content=content, metadata=h_doc.metadata.copy())
                )

    logger.info(
        f"Markdown 文档已分割为 {len(all_chunks)} 个块 (chunk_size={chunk_size}, "
        f"overlap={chunk_overlap}, 标题感知+原子保护)."
    )
    return all_chunks


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    使用 RecursiveCharacterTextSplitter 对文档进行中文友好分割。
    适用于 PDF、TXT 等非 Markdown 文档。

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


def _smart_split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    根据文档类型自动选择最佳分块策略。
    - Markdown 论文 → 标题感知 + 原子单元保护
    - 其他（PDF / TXT）→ 通用中文友好分割
    """
    md_docs = [d for d in documents if d.metadata.get("file_type") == "markdown"]
    other_docs = [d for d in documents if d.metadata.get("file_type") != "markdown"]

    result: List[Document] = []
    if md_docs:
        result.extend(split_markdown_documents(md_docs, chunk_size, chunk_overlap))
    if other_docs:
        result.extend(split_documents(other_docs, chunk_size, chunk_overlap))
    return result


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
    # 1. 分割文档（自动按文件类型路由最佳策略）
    split_docs = _smart_split_documents(documents, chunk_size, chunk_overlap)

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
    split_docs = _smart_split_documents(documents, chunk_size, chunk_overlap)
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


