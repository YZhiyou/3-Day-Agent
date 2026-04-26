"""
重排序（Rerank）模块
使用阿里云 DashScope Rerank API 对初步检索结果进行语义重排序
"""

from typing import List, Sequence

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.document_compressors import DashScopeRerank


class ContextualCompressionRetriever(BaseRetriever):
    """
    上下文压缩检索器：先通过基础检索器召回文档，
    再通过 compressor（如重排序模型）对结果精排/压缩。

    注：当前 langchain 1.2.15 未内置此类，故在模块内自行实现。
    """

    base_compressor: DashScopeRerank
    base_retriever: BaseRetriever

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """同步获取相关文档并压缩。"""
        docs = self.base_retriever.invoke(query)
        if not docs:
            return []
        compressed = self.base_compressor.compress_documents(
            documents=docs, query=query
        )
        return list(compressed)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """异步获取相关文档并压缩。"""
        docs = await self.base_retriever.ainvoke(query)
        if not docs:
            return []
        compressed = await self.base_compressor.acompress_documents(
            documents=docs, query=query
        )
        return list(compressed)


def get_reranker(model: str = "qwen3-vl-rerank", top_n: int = 5) -> DashScopeRerank:
    """
    获取 DashScope Reranker 实例。

    注意：当前 langchain_community 的 DashScopeRerank 在初始化时会强制将
    model 覆盖为 gte-rerank，因此创建后需手动覆盖为期望的模型名称。

    Args:
        model: 重排序模型名称，支持 qwen3-vl-rerank 等
        top_n: 重排序后返回的文档数量

    Returns:
        DashScopeRerank 实例
    """
    reranker = DashScopeRerank(model=model, top_n=top_n)
    reranker.model = model
    return reranker


def get_compression_retriever(
    base_retriever: BaseRetriever,
    model: str = "qwen3-vl-rerank",
    top_n: int = 5,
) -> ContextualCompressionRetriever:
    """
    创建一个带重排序能力的压缩检索器。

    Args:
        base_retriever: 基础向量检索器（Chroma.as_retriever() 的返回值）
        model: 重排序模型名称
        top_n: 最终返回的文档数量

    Returns:
        ContextualCompressionRetriever 实例，可直接用于 agent 工具
    """
    compressor = get_reranker(model=model, top_n=top_n)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )
