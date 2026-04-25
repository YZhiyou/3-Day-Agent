import os

# 清理可能干扰 API 连接的代理环境变量
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    os.environ.pop(k, None)

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from retriever import build_retriever
from memory_manager import get_session_history
from tools import create_tools
from agent import create_agent


def main():
    # 准备
    retriever = build_retriever()
    user_id = "test_user"
    tools = create_tools(user_id, retriever)
    agent_executor = create_agent(tools, get_session_history)

    config = {"configurable": {"thread_id": "test_session"}}

    # 测试 1：让用户自我介绍并保存信息
    print("=" * 50)
    print("用户: 你好，我叫 Alice，请记住我的名字。")
    print("=" * 50)
    response = agent_executor.invoke(
        {"messages": [{"role": "user", "content": "你好，我叫 Alice，请记住我的名字。"}]},
        config
    )
    print(f"Agent: {response['messages'][-1].content}")

    # 测试 2：验证是否记住了名字
    print("\n" + "=" * 50)
    print("用户: 我叫什么名字？")
    print("=" * 50)
    response = agent_executor.invoke(
        {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
        config
    )
    print(f"Agent: {response['messages'][-1].content}")

    # 测试 3：搜索知识库
    print("\n" + "=" * 50)
    print("用户: 什么是 PSR？")
    print("=" * 50)
    response = agent_executor.invoke(
        {"messages": [{"role": "user", "content": "什么是 PSR？"}]},
        config
    )
    print(f"Agent: {response['messages'][-1].content}")


if __name__ == "__main__":
    main()
