import logging
import re
from typing import Optional, Dict, Any, List

from pydantic import Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_chroma import Chroma

from vector_store import load_vector_store, get_collection_stats
from reranker import get_compression_retriever


def _chinese_tokenizer(text: str) -> List[str]:
    """
    简易中文分词器：连续中文字符逐字切分，英文/数字保留为整体。

    用于 BM25Retriever 的 preprocess_func，解决默认空格分词对中文无效的问题。
    """
    return [match.group(0) for match in re.finditer(r"[a-zA-Z0-9]+|[^a-zA-Z0-9\s]", text)]

logger = logging.getLogger(__name__)

_DEFAULT_SEARCH_KWARGS = {"k": 4}


def build_retriever(
    persist_dir: str = "./data/chroma",
    search_type: str = "similarity",
    search_kwargs: Optional[Dict[str, Any]] = None
) -> BaseRetriever:
    """
    工厂函数：构建并返回配置好的 LangChain Retriever。

    这是接入 LCEL、Agent 等上层组件的推荐入口，
    返回标准的 BaseRetriever，可直接用于链式调用。

    Args:
        persist_dir: 向量库持久化目录。
        search_type: 搜索类型，"similarity" 或 "mmr"。
        search_kwargs: 搜索参数，如 {"k": 6, "fetch_k": 50}。

    Returns:
        配置好的 BaseRetriever 实例。
    """
    if search_kwargs is None:
        search_kwargs = _DEFAULT_SEARCH_KWARGS.copy()

    vectordb = load_vector_store(persist_dir)
    retriever = vectordb.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    logger.info(
        f"Retriever built: search_type={search_type}, search_kwargs={search_kwargs}"
    )
    return retriever


def build_rerank_retriever(
    persist_dir: str = "./data/chroma",
    search_type: str = "similarity",
    top_k: int = 20,
    top_n: int = 5,
    model: str = "qwen3-vl-rerank",
    search_kwargs: Optional[Dict[str, Any]] = None,
) -> BaseRetriever:
    """
    构建带重排序的检索器。

    工作流程：
        1. 向量检索器先从 Chroma 中召回 top_k 个文档
        2. DashScope Reranker 对这 top_k 个文档重新打分
        3. 返回得分最高的 top_n 个文档

    Args:
        persist_dir: 向量库持久化目录。
        search_type: 搜索类型，"similarity" 或 "mmr"。
        top_k: 初步检索数量（给 rerank 更大的候选池）。
        top_n: 最终返回的文档数量。
        model: 重排序模型名称。
        search_kwargs: 额外的搜索参数。

    Returns:
        带重排序能力的 ContextualCompressionRetriever 实例。
    """
    if search_kwargs is None:
        search_kwargs = {"k": top_k}
    else:
        search_kwargs = {**search_kwargs, "k": top_k}

    vectordb = load_vector_store(persist_dir)
    base_retriever = vectordb.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    compression_retriever = get_compression_retriever(
        base_retriever, model=model, top_n=top_n
    )
    logger.info(
        f"Rerank retriever built: search_type={search_type}, top_k={top_k}, "
        f"top_n={top_n}, model={model}"
    )
    return compression_retriever


class HybridRetriever(BaseRetriever):
    """
    混合检索器：语义粗筛 + BM25 关键词精排 + RRF 融合。

    工作流程：
    1. 语义检索器先召回 semantic_k 个候选文档
    2. 在这些候选上构建 BM25 索引，进行关键词匹配，召回 bm25_k 个
    3. 使用倒数排名融合（RRF）合并两条通路的结果

    若 rank-bm25 未安装，则自动降级为纯语义检索。
    """
    semantic_retriever: BaseRetriever
    semantic_k: int = 20
    bm25_k: int = 5
    weights: List[float] = Field(default_factory=lambda: [0.5, 0.5])

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """同步混合检索入口。"""
        semantic_docs = self.semantic_retriever.invoke(query)
        if not semantic_docs:
            return []

        bm25_docs = self._bm25_search(query, semantic_docs)
        return self._rrf_fuse(semantic_docs, bm25_docs)

    def _bm25_search(self, query: str, candidates: List[Document]) -> List[Document]:
        """在语义候选上构建 BM25 索引并进行关键词检索。"""
        try:
            from langchain_community.retrievers import BM25Retriever
        except ImportError:
            logger.warning("BM25Retriever 不可用，跳过关键词检索")
            return []

        if len(candidates) <= 1:
            return candidates

        try:
            bm25 = BM25Retriever.from_documents(
                candidates, preprocess_func=_chinese_tokenizer
            )
            bm25.k = min(self.bm25_k, len(candidates))
            return bm25.invoke(query)
        except Exception as exc:
            logger.warning(f"BM25 检索失败: {exc}，降级为纯语义检索")
            return []

    def _rrf_fuse(
        self, semantic_docs: List[Document], bm25_docs: List[Document]
    ) -> List[Document]:
        """倒数排名融合（Reciprocal Rank Fusion）。"""
        k = 60  # RRF 常数
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        lists = [
            (self.weights[0], semantic_docs),
            (
                self.weights[1] if len(self.weights) > 1 else 0.5,
                bm25_docs,
            ),
        ]

        for weight, docs in lists:
            for rank, doc in enumerate(docs):
                key = doc.page_content
                scores[key] = scores.get(key, 0.0) + weight * (1.0 / (k + rank + 1))
                doc_map[key] = doc

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[key] for key, _ in sorted_items]


def build_hybrid_rerank_retriever(
    persist_dir: str = "./data/chroma",
    search_type: str = "similarity",
    semantic_k: int = 20,
    bm25_k: int = 5,
    top_n: int = 5,
    model: str = "qwen3-vl-rerank",
    weights: Optional[List[float]] = None,
) -> BaseRetriever:
    """
    构建混合检索 + Rerank 精排检索器。

    工作流程：
    1. 向量检索器从 Chroma 中召回 semantic_k 个候选文档（语义粗筛）
    2. BM25 在这批候选上进行关键词精排，召回 bm25_k 个
    3. RRF 算法融合语义和关键词两条通路的结果
    4. DashScope Reranker 对融合结果最终精排，返回 top_n

    Args:
        persist_dir: 向量库持久化目录。
        search_type: 搜索类型，"similarity" 或 "mmr"。
        semantic_k: 语义通路召回数量。
        bm25_k: 关键词通路召回数量。
        top_n: Rerank 后最终返回数量。
        model: 重排序模型名称。
        weights: RRF 融合权重 [语义权重, 关键词权重]，默认 [0.5, 0.5]。

    Returns:
        带混合检索和重排序能力的检索器实例。
    """
    if weights is None:
        weights = [0.5, 0.5]

    vectordb = load_vector_store(persist_dir)
    semantic_retriever = vectordb.as_retriever(
        search_type=search_type,
        search_kwargs={"k": semantic_k}
    )

    hybrid = HybridRetriever(
        semantic_retriever=semantic_retriever,
        semantic_k=semantic_k,
        bm25_k=bm25_k,
        weights=weights,
    )

    compression_retriever = get_compression_retriever(
        hybrid, model=model, top_n=top_n
    )
    logger.info(
        f"Hybrid rerank retriever built: search_type={search_type}, "
        f"semantic_k={semantic_k}, bm25_k={bm25_k}, top_n={top_n}, "
        f"weights={weights}, model={model}"
    )
    return compression_retriever


class RetrievalEngine:
    """
    检索引擎：封装检索策略，提供比 BaseRetriever 更丰富的控制接口。

    适用于需要直接操控搜索参数（k、filter、MMR、阈值过滤等）的场景，
    也可作为后续扩展（重排序、混合检索）的挂载点。
    """

    def __init__(self, persist_dir: str = "./data/chroma"):
        self.vectordb: Chroma = load_vector_store(persist_dir)

    # ------------------------------------------------------------------
    # 基础检索接口（直接操作向量库）
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        相似度搜索。

        Args:
            query: 查询文本。
            k: 返回最相似的结果数量。
            filter: 可选的元数据过滤条件，例如 {"file_type": "pdf"}。

        Returns:
            相似度最高的文档列表。
        """
        docs = self.vectordb.similarity_search(query, k=k, filter=filter)
        logger.debug(
            f"相似度搜索返回 {len(docs)} 条结果 (query='{query[:30]}...', k={k})."
        )
        return docs

    def mmr_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        MMR（最大边际相关性）搜索，平衡相关性与多样性。

        Args:
            query: 查询文本。
            k: 返回结果数量。
            fetch_k: 初始检索候选数量。
            lambda_mult: 多样性权重（0~1，越大越注重相关性）。
            filter: 可选的元数据过滤条件。

        Returns:
            MMR 筛选后的文档列表。
        """
        docs = self.vectordb.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter
        )
        logger.debug(
            f"MMR 搜索返回 {len(docs)} 条结果 (query='{query[:30]}...', k={k})."
        )
        return docs

    def similarity_search_with_threshold(
        self,
        query: str,
        k: int = 10,
        score_threshold: float = 0.7,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        带相似度阈值的搜索，过滤掉低质量结果。

        注意：Chroma 的 similarity_search_with_score 返回的 score
        是距离（越小越相似），这里统一转换为相似度（越大越相似）。
        如需接入重排序模型，可在此方法后追加 rerank 步骤。

        Args:
            query: 查询文本。
            k: 初始检索候选数量。
            score_threshold: 相似度阈值（0~1），低于此值的结果被过滤。
            filter: 可选的元数据过滤条件。

        Returns:
            通过阈值过滤后的文档列表。
        """
        docs_with_scores = self.vectordb.similarity_search_with_score(
            query, k=k, filter=filter
        )
        # Chroma score 是距离（L2），需归一化转换。
        # 这里采用简单的 1 / (1 + distance) 作为近似相似度。
        filtered = [
            doc for doc, distance in docs_with_scores
            if (1.0 / (1.0 + distance)) >= score_threshold
        ]
        logger.info(
            f"阈值过滤搜索: 原始 {len(docs_with_scores)} 条，"
            f"过滤后 {len(filtered)} 条 (threshold={score_threshold})."
        )
        return filtered

    # ------------------------------------------------------------------
    # Retriever 封装（便于接入 LangChain 链）
    # ------------------------------------------------------------------

    def get_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None
    ) -> BaseRetriever:
        """
        获取配置好的 BaseRetriever。

        Args:
            search_type: 搜索类型，"similarity" 或 "mmr"。
            search_kwargs: 搜索参数。

        Returns:
            BaseRetriever 实例。
        """
        if search_kwargs is None:
            search_kwargs = _DEFAULT_SEARCH_KWARGS.copy()
        return self.vectordb.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )

    def invoke(self, query: str, **kwargs) -> List[Document]:
        """同步检索快捷入口。"""
        return self.get_retriever().invoke(query, **kwargs)

    async def ainvoke(self, query: str, **kwargs) -> List[Document]:
        """异步检索快捷入口。"""
        return await self.get_retriever().ainvoke(query, **kwargs)

    # ------------------------------------------------------------------
    # 元数据与诊断
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """获取向量库统计信息。"""
        return get_collection_stats(self.vectordb)