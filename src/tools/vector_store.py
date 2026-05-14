import json
import os
import re
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Callable, Iterator, Sequence
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.stores import BaseStore
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever

load_dotenv()
logger = logging.getLogger(__name__)


def _get_embeddings() -> DashScopeEmbeddings:
    """统一初始化 DashScope 嵌入模型。"""
    return DashScopeEmbeddings(model="text-embedding-v4")


def _clean_text(text: str) -> str:
    """
    清理文本中的非法 Unicode surrogate 字符（0xD800 - 0xDFFF 范围）。

    PDF 提取或某些编码转换可能产生 surrogate 字符，这些字符在 UTF-8 编码中不合法，
    会导致 DashScope API 请求触发 UnicodeEncodeError。
    """
    if not text:
        return text
    return "".join(c for c in text if not (0xD800 <= ord(c) <= 0xDFFF))


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
    chunk_overlap: int = 200,
    batch_size: int = 10,
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
) -> Chroma:
    """
    从原始文档创建 Chroma 向量存储并持久化。

    Args:
        documents: 原始文档列表。
        persist_dir: 持久化目录路径。
        chunk_size: 分块大小。
        chunk_overlap: 分块重叠大小。
        batch_size: 每批嵌入的文档块数量，避免一次性请求过多导致 API 超时。
        progress_callback: 进度回调函数，签名 (stage, current, total, message)。

    Returns:
        构建完成的 Chroma 向量存储实例。
    """
    # 1. 分割文档（自动按文件类型路由最佳策略）
    if progress_callback:
        progress_callback("split", 0, 1, "正在分割文档...")

    split_docs = _smart_split_documents(documents, chunk_size, chunk_overlap)

    if progress_callback:
        progress_callback("split", 1, 1, f"文档已分割为 {len(split_docs)} 个块")

    # 2. 初始化嵌入模型
    embeddings = _get_embeddings()

    # 3. 确保持久化目录存在
    Path(persist_dir).parent.mkdir(parents=True, exist_ok=True)

    # 4. 创建空 Chroma 实例（复用已有目录或新建）
    if progress_callback:
        progress_callback("create", 0, 1, "正在初始化向量库...")

    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    if progress_callback:
        progress_callback("create", 1, 1, "向量库初始化完成")

    # 5. 分批添加文档，避免单次嵌入请求过大导致 API 超时
    total = len(split_docs)
    for i in range(0, total, batch_size):
        batch = split_docs[i : i + batch_size]
        # 清理非法 Unicode surrogate 字符（常见于 PDF 提取的数学符号）
        for doc in batch:
            doc.page_content = _clean_text(doc.page_content)

        # 对每批增加重试，应对偶发的网络中断（IncompleteRead / Connection broken）
        for attempt in range(3):
            try:
                vectordb.add_documents(batch)
                break
            except Exception as exc:
                if attempt < 2:
                    wait = 2 ** (attempt + 1)  # 2s, 4s
                    logger.warning(
                        f"嵌入批次失败（{i + 1}-{min(i + batch_size, total)}），"
                        f"{wait}s 后第 {attempt + 2} 次重试: {exc}"
                    )
                    import time
                    time.sleep(wait)
                else:
                    logger.error(
                        f"嵌入批次最终失败（{i + 1}-{min(i + batch_size, total)}）: {exc}"
                    )
                    raise
        if progress_callback:
            progress_callback(
                "embed",
                min(i + batch_size, total),
                total,
                f"正在嵌入文档块... ({min(i + batch_size, total)}/{total})",
            )

    logger.info(
        f"Vector store created and persisted to '{persist_dir}' with {total} chunks."
    )
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


# ---------- Parent-Child 模式支持 ----------

class JsonDocstore(BaseStore[str, Document]):
    """
    基于 JSON 文件的 Document 存储，用于 Parent-Child 模式的父文档持久化。

    实现 langchain_core.stores.BaseStore 接口，与 ParentDocumentRetriever 兼容。
    每次 mset/mdelete 后自动同步到磁盘，确保进程重启后父文档不丢失。
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._data: Dict[str, Document] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                for k, v in raw.items():
                    self._data[k] = Document(
                        page_content=v.get("page_content", ""),
                        metadata=v.get("metadata", {}),
                    )
            except Exception as exc:
                logger.warning(f"加载 docstore 失败: {exc}，将以空 store 启动")

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        raw = {
            k: {"page_content": doc.page_content, "metadata": doc.metadata}
            for k, doc in self._data.items()
        }
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)

    def mget(self, keys: Sequence[str]) -> List[Optional[Document]]:
        return [self._data.get(k) for k in keys]

    def mset(self, key_value_pairs: Sequence[tuple[str, Document]]) -> None:
        for k, v in key_value_pairs:
            self._data[k] = v
        self._save()

    def mdelete(self, keys: Sequence[str]) -> None:
        for k in keys:
            self._data.pop(k, None)
        self._save()

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        for k in self._data:
            if prefix is None or k.startswith(prefix):
                yield k


def is_parent_child_mode(persist_dir: str = "./data/chroma") -> bool:
    """检查指定目录是否使用了 Parent-Child 模式（通过 docstore.json 存在性判断）。"""
    return os.path.exists(os.path.join(persist_dir, "docstore.json"))


def create_parent_child_vector_store(
    documents: List[Document],
    persist_dir: str = "./data/chroma",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    child_chunk_size: int = 300,
    child_chunk_overlap: int = 50,
    batch_size: int = 10,
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
) -> tuple[Chroma, JsonDocstore]:
    """
    使用 Parent-Child 模式创建 Chroma 向量存储。

    工作流程：
    1. 先用 _smart_split_documents 创建父块（Markdown 标题感知 / 通用中文友好）
    2. ParentDocumentRetriever 将每个父块细分为子块（chunk_size=300）
    3. 子块存入 Chroma 向量库（用于向量检索），父块存入 JsonDocstore（用于返回完整上下文）

    Args:
        documents: 原始文档列表。
        persist_dir: 持久化目录路径。
        chunk_size: 父块分块大小。
        chunk_overlap: 父块分块重叠大小。
        child_chunk_size: 子块分块大小。
        child_chunk_overlap: 子块分块重叠大小。
        batch_size: 每批嵌入的父块数量。
        progress_callback: 进度回调函数。

    Returns:
        (Chroma 向量库实例, JsonDocstore 父文档存储实例)
    """
    # 1. 父块分割（保持原有的文件类型感知路由）
    if progress_callback:
        progress_callback("split", 0, 1, "正在分割父块文档...")

    parent_docs = _smart_split_documents(documents, chunk_size, chunk_overlap)
    for doc in parent_docs:
        doc.page_content = _clean_text(doc.page_content)

    if progress_callback:
        progress_callback("split", 1, 1, f"父块已分割为 {len(parent_docs)} 个")

    # 2. 初始化嵌入模型与 Chroma
    embeddings = _get_embeddings()
    Path(persist_dir).parent.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback("create", 0, 1, "正在初始化 Parent-Child 向量库...")

    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    docstore_path = os.path.join(persist_dir, "docstore.json")
    docstore = JsonDocstore(docstore_path)

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )

    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectordb,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=None,  # 父块已由 _smart_split_documents 完成
        search_kwargs={"k": 20},
    )

    if progress_callback:
        progress_callback("create", 1, 1, "Parent-Child 向量库初始化完成")

    # 3. 分批添加父块（ParentDocumentRetriever 内部自动完成子块分割与双写）
    total = len(parent_docs)
    for i in range(0, total, batch_size):
        batch = parent_docs[i : i + batch_size]

        for attempt in range(3):
            try:
                parent_retriever.add_documents(batch)
                break
            except Exception as exc:
                if attempt < 2:
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        f"Parent-Child 嵌入批次失败（{i + 1}-{min(i + batch_size, total)}），"
                        f"{wait}s 后第 {attempt + 2} 次重试: {exc}"
                    )
                    import time
                    time.sleep(wait)
                else:
                    logger.error(
                        f"Parent-Child 嵌入批次最终失败（{i + 1}-{min(i + batch_size, total)}）: {exc}"
                    )
                    raise

        if progress_callback:
            progress_callback(
                "embed",
                min(i + batch_size, total),
                total,
                f"正在嵌入父块... ({min(i + batch_size, total)}/{total})",
            )

    logger.info(
        f"Parent-Child vector store created: {total} parent chunks, "
        f"persisted to '{persist_dir}' with docstore '{docstore_path}'."
    )
    return vectordb, docstore


def load_parent_child_vector_store(
    persist_dir: str = "./data/chroma",
) -> tuple[Chroma, JsonDocstore]:
    """
    加载已有的 Parent-Child 模式向量存储。

    Args:
        persist_dir: 持久化目录路径。

    Returns:
        (Chroma 实例, JsonDocstore 实例)

    Raises:
        FileNotFoundError: 如果向量库或 docstore 不存在。
    """
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        raise FileNotFoundError(f"在 {persist_dir} 未找到已持久化的向量库。")

    embeddings = _get_embeddings()
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    docstore_path = os.path.join(persist_dir, "docstore.json")
    if not os.path.exists(docstore_path):
        raise FileNotFoundError(
            f"在 {docstore_path} 未找到 Parent-Child 的 docstore。"
            f"如需使用旧版向量库，请调用 load_vector_store()。"
        )

    docstore = JsonDocstore(docstore_path)
    logger.info(f"已从 '{persist_dir}' 加载 Parent-Child 向量库及 docstore。")
    return vectordb, docstore


def add_documents_parent_child(
    vectordb: Chroma,
    docstore: JsonDocstore,
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    child_chunk_size: int = 300,
    child_chunk_overlap: int = 50,
) -> None:
    """
    向 Parent-Child 向量存储中增量添加文档。

    Args:
        vectordb: Chroma 实例。
        docstore: JsonDocstore 实例。
        documents: 要添加的新文档列表。
        chunk_size: 父块分块大小。
        chunk_overlap: 父块分块重叠大小。
        child_chunk_size: 子块分块大小。
        child_chunk_overlap: 子块分块重叠大小。
    """
    parent_docs = _smart_split_documents(documents, chunk_size, chunk_overlap)
    for doc in parent_docs:
        doc.page_content = _clean_text(doc.page_content)

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )

    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectordb,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=None,
        search_kwargs={"k": 20},
    )

    parent_retriever.add_documents(parent_docs)
    logger.info(f"已向 Parent-Child 向量库添加 {len(parent_docs)} 个父块。")


def delete_documents_parent_child(
    vectordb: Chroma,
    docstore: JsonDocstore,
    ids: Optional[List[str]] = None,
    filter: Optional[Dict[str, Any]] = None,
) -> Optional[bool]:
    """
    在 Parent-Child 模式下删除文档（同时清理 vectorstore 中的子块和 docstore 中的父块）。

    Args:
        vectordb: Chroma 向量存储实例。
        docstore: JsonDocstore 实例。
        ids: 要删除的文档 ID 列表。
        filter: 可选的元数据过滤条件（会按 source 查找到所有子块，再联动删除父块）。

    Returns:
        删除是否成功。
    """
    try:
        if filter is not None:
            result = vectordb.get(where=filter)
            ids = result.get("ids", []) if result else []
            metadatas = result.get("metadatas", []) if result else []
            doc_ids = set()
            for meta in metadatas:
                if isinstance(meta, dict) and "doc_id" in meta:
                    doc_ids.add(meta["doc_id"])
            if doc_ids:
                docstore.mdelete(list(doc_ids))
                logger.info(f"已从 docstore 删除 {len(doc_ids)} 个父块。")
        elif ids is not None:
            # 按子块 ID 查 doc_id 再删父块（效率较低，建议优先用 filter）
            result = vectordb.get(ids=ids)
            metadatas = result.get("metadatas", []) if result else []
            doc_ids = set()
            for meta in metadatas:
                if isinstance(meta, dict) and "doc_id" in meta:
                    doc_ids.add(meta["doc_id"])
            if doc_ids:
                docstore.mdelete(list(doc_ids))

        result = vectordb.delete(ids=ids, filter=filter)
        logger.info(f"已删除 vectorstore 中的子块 (ids={ids is not None}, filter={filter is not None})。")
        return result
    except Exception as exc:
        logger.error(f"Parent-Child 删除失败: {exc}")
        raise

