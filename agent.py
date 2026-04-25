from typing import List, Callable

from langchain.agents import create_agent as _create_langchain_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.tools import BaseTool

import os
import sqlite3


def create_agent(
    tools: List[BaseTool],
    get_session_history: Callable[[str], object] = None,
    llm_model: str = "deepseek-chat",
    temperature: float = 0.7,
    max_iterations: int = 5,
):
    """
    创建配置了工具、记忆的 Agent 执行器。

    Args:
        tools: Agent 可用的工具列表。
        get_session_history: 获取会话历史的工厂函数（来自 memory_manager，
            在 LangChain 1.2.15 中保留此参数以保持接口兼容，实际短期记忆
            由内部 MemorySaver checkpointer 自动管理）。
        llm_model: DeepSeek 模型名。
        temperature: LLM 温度参数。
        max_iterations: Agent 最大推理步数（映射为 LangGraph 的 recursion_limit）。

    Returns:
        可直接调用 invoke() 的 CompiledStateGraph 实例。
        调用方式：
            agent.invoke(
                {"messages": [{"role": "user", "content": "..."}]},
                {"configurable": {"thread_id": "session_id"}}
            )
    """

    system_prompt = """你是一个个人知识库助手，能帮助用户管理信息和回答基于知识库的问题。

你的能力：
- 搜索用户的个人知识库（当用户询问某个知识点、文档内容时使用）
- 保存用户的个人信息（姓名、偏好等）
- 查询用户之前保存的个人信息

行为准则：
1. 当用户询问关于他们自己的情况时，先用 get_user_info 查询是否已保存相关信息。
2. 当用户询问某个概念、定义或需要参考资料时，用 retrieve_documents 搜索知识库。
3. 不要编造信息。知识库中没有的内容，诚实地告诉用户。
4. 回答简洁明了，用中文回复。"""

    # 检查工具列表中是否存在联网搜索工具，若存在则追加联网规则
    tool_names = [t.name for t in tools]
    if "web_search" in tool_names:
        system_prompt += """
【联网搜索规则】
- 当用户询问实时信息、最新新闻、当前事件、股票价格、天气等动态内容时，使用 web_search 工具搜索互联网。
- 知识库中没有的实时信息也应尝试联网搜索。
- 搜索结果通常包含来源 URL，请在回答中引用来源网址。
- 如果用户明确说"不要联网搜索"或"只查我的知识库"，则跳过联网搜索。
- 对搜索结果进行归纳总结，不要直接堆砌原始内容。
"""

    # 确保 data 目录存在
    os.makedirs("./data", exist_ok=True)
    conn = sqlite3.connect("./data/checkpoints.db", check_same_thread=False)
    memory = SqliteSaver(conn)
    agent = _create_langchain_agent(
        llm_model,
        tools,
        system_prompt=system_prompt,
        checkpointer=memory,
    )

    # 设置默认 recursion_limit
    agent = agent.with_config({"recursion_limit": max_iterations * 10})

    return agent
