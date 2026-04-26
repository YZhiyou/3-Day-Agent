# 需要安装 duckduckgo-search: pip install duckduckgo-search
from typing import List
from langchain_core.tools import tool, BaseTool
from langchain_core.retrievers import BaseRetriever

from memory_manager import save_user_info as _save_user_info_impl, get_user_info as _get_user_info_impl


def _create_web_search_tool(max_results: int = 5) -> BaseTool:
    """
    创建联网搜索工具（使用 Tavily）。

    Args:
        max_results: 返回的最大搜索结果数。

    Returns:
        配置好的联网搜索工具。
    """
    from langchain_tavily import TavilySearch

    @tool
    def web_search(query: str) -> str:
        """
        当用户询问实时信息、最新新闻、当前事件、需要联网查询的内容时使用此工具。参数 query 为搜索关键词。
        """
        search = TavilySearch(max_results=max_results)
        return search.invoke(query)

    return web_search


def create_tools(
    user_id: str,
    retriever: BaseRetriever,
    enable_web_search: bool = True,
    web_search_max_results: int = 5
) -> List[BaseTool]:
    """
    工厂函数：创建并返回绑定 user_id 和 retriever 的工具列表。

    Args:
        user_id: 当前用户标识，用于长期记忆的读写。
        retriever: 已配置的 LangChain Retriever，用于文档检索。
        enable_web_search: 是否启用联网搜索工具。
        web_search_max_results: 联网搜索返回的最大结果数。

    Returns:
        Agent 可用的 BaseTool 列表。
    """

    @tool
    def retrieve_documents(query: str) -> str:
        """
        在个人知识库中搜索与用户问题相关的文档。
        检索结果经过混合检索（语义理解 + 精准关键词匹配）与语义重排序，
        越靠前的结果越相关，尤其适用于缩写和特定术语的精确查找。
        当用户询问某个特定知识点、概念或需要参考已有资料时，使用此工具。
        参数 query 应该是用自然语言描述的具体搜索内容。
        """
        docs = retriever.invoke(query)
        if not docs:
            return "没有找到相关文档。"
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知来源")
            content = doc.page_content[:300]
            results.append(f"[{i}] 来源: {source}\n{content}\n")
        return "\n".join(results)

    @tool
    def save_user_info(key: str, value: str) -> str:
        """
        保存当前用户的偏好或个人信息。例如：姓名、喜欢的颜色、专业领域等。
        参数 key 是信息的名称，value 是信息的内容。
        """
        _save_user_info_impl(user_id, key, value)
        return f"已保存信息: {key} = {value}"

    @tool
    def get_user_info(key: str) -> str:
        """
        查询当前用户之前保存的个人信息。如果用户提问涉及他们的个人情况，先尝试这里查询。
        参数 key 是要查询的信息名称。
        """
        result = _get_user_info_impl(user_id, key)
        if result is None:
            return f"未找到关于 '{key}' 的信息。"
        return f"{key}: {result}"

    tools: List[BaseTool] = [retrieve_documents, save_user_info, get_user_info]

    if enable_web_search:
        tools.append(_create_web_search_tool(web_search_max_results))

    return tools
