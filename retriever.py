import logging
from typing import Optional, Dict, Any, List

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_chroma import Chroma

from vector_store import load_vector_store, get_collection_stats

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