## 模块二：用 `RunnableWithMessageHistory` 改造 Agent

### 2.1 为什么要用 `RunnableWithMessageHistory`？

在模块一中，我们已经实现了三种记忆的存储方式。但存储只是第一步——关键在于如何让 Agent **在执行时自动读取历史、并在执行后自动写入新消息**。

`RunnableWithMessageHistory` 就是 LangChain 提供的“记忆包装器”。它做了三件事：
1. **执行前**：根据 `session_id` 从存储中加载历史消息，拼接到当前输入中。
2. **执行中**：让 Agent 带着完整上下文运行。
3. **执行后**：把本轮的新消息（用户输入 + AI 输出）保存回存储。

**多会话隔离原理**：每个 `session_id` 对应一个独立的消息列表。不同用户（或同一用户的不同对话）使用不同的 `session_id`，彼此完全隔离，互不干扰。

---

### 2.2 改造前的 Agent（回顾上周成果）

我们先看一个**无记忆**的 Agent，这是你上周实现的基础版本：

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv
import os

load_dotenv()

# --- 模型 & 工具准备 ---
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0,
)

search = TavilySearchResults(
    api_key=os.getenv("TAVILY_API_KEY"),
    max_results=2,
)
tools = [search]

# --- 直接创建 Agent（无记忆）---
agent = create_agent(model=llm, tools=tools)
```

**失忆测试**：
```python
# 第一轮：告诉 Agent 名字
response1 = agent.invoke({"messages": [{"role": "user", "content": "你好，我叫小明，我是一名程序员"}]})
print(response1["messages"][-1].content)

# 第二轮：问它“我叫什么”——它会忘掉！
response2 = agent.invoke({"messages": [{"role": "user", "content": "你还记得我叫什么吗？"}]})
print(response2["messages"][-1].content)  
# 输出类似："抱歉，我没有关于你名字的信息。"
```

原因：每轮 `invoke` 都是独立的，Agent 看不到上一轮的对话。

---

### 2.3 核心改造：三步装上有记忆

改造分三步：

#### 步骤①：实现 `get_session_history` 回调函数

这是记忆的“后勤部门”——LangChain 会在需要时调用它来获取或创建会话历史。我们先用一个字典作为简单的会话存储。

```python
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

# 全局字典：key 是 session_id，value 是 ChatMessageHistory 实例
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """根据 session_id 返回对应的历史记录对象。
    如果不存在，就新建一个 InMemoryChatMessageHistory 并存入 store。
    """
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

```

**关键点**：
- `session_id` 是会话的唯一标识，由我们（调用方）传入。
- 每个 `session_id` 对应一个独立的消息列表，这就是多会话隔离的核心。
- 如果你想用**滑动窗口**或**摘要记忆**，只需把 `InMemoryChatMessageHistory` 替换为你在模块一中自定义的 `WindowedChatMessageHistory` 或 `SummaryChatMessageHistory` 即可——接口完全兼容。

---

#### 步骤②：用 `invoke_with_memory` 函数手动管理历史消息

- `create_agent` 创建的 LangGraph Agent 与 `RunnableWithMessageHistory` 不兼容。在 LangChain 1.x 中，需要**手动管理历史消息**，而不是用 `RunnableWithMessageHistory` 包装。

```python
def invoke_with_memory(agent, session_id: str, user_content: str):
    """手动管理历史记忆调用 Agent。
    1. 获取历史消息
    2. 添加当前用户消息
    3. 调用 Agent
    4. 保存用户消息和 AI 响应到历史
    5. 返回响应
    """
    history = get_session_history(session_id)

    # 构建包含历史消息的输入
    messages = list(history.messages) + [HumanMessage(content=user_content)]

    # 调用 Agent
    response = agent.invoke({"messages": messages})

    # 将当前交互保存到历史
    history.add_message(HumanMessage(content=user_content))
    history.add_message(AIMessage(content=response["messages"][-1].content))

    return response
```

**数据流解释**：
```
用户调用 → RunnableWithMessageHistory 
           ├─ 调用 get_session_history(session_id) 获取历史
           ├─ 将历史消息合并到 "messages" 字段
           ├─ 调用原始 Agent 执行
           ├─ 从输出中提取新的消息
           └─ 调用 history.add_messages() 保存新消息
         → 返回完整输出
```

---

#### 步骤③：调用时传入 `config` 指定 `session_id`

与无记忆版本最大的区别：调用时必须传入 `config`，其中包含 `session_id`。

```python
# --- 直接创建 Agent ---
agent = create_agent(model=llm, tools=tools)

# --- 有记忆调用 ---
SESSION_ID = "user_xiaoming"

# 第一轮
response1 = invoke_with_memory(agent, SESSION_ID, "你好，我叫小明，我是一名程序员")
print(response1["messages"][-1].content)
print("---")

# 第二轮——这次能记住！
response2 = invoke_with_memory(agent, SESSION_ID, "你还记得我叫什么吗？")
print(response2["messages"][-1].content)
```

---

### 2.4 多会话隔离验证

```python
# 用户A的会话
SESSION_ID = "user_xiaoming"

response1 = invoke_with_memory(agent, SESSION_ID, "你好，我叫小明，我是一名程序员")
print(response1["messages"][-1].content)
print("---")


response2 = invoke_with_memory(agent, SESSION_ID, "你还记得我叫什么吗？")
print(response2["messages"][-1].content)

# 用户B的会话
SESSION_ID_2 = "user_xiaohong"

response1 = invoke_with_memory(agent, SESSION_ID_2, "你好，我叫小红，我是一名学生")
print(response1["messages"][-1].content)
print("---")

response2 = invoke_with_memory(agent, SESSION_ID, "你还记得我叫什么吗？")
print(response2["messages"][-1].content)

```

**原理**：`store` 字典中，`"user_a"` 和 `"user_b"` 各自维护独立的消息列表，互不干扰。

---

### 2.5 完整改造后代码（可直接运行）

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

# 全局字典：key 是 session_id，value 是 InMemoryChatMessageHistory 实例
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """根据 session_id 返回对应的历史记录对象。
    如果不存在，就新建一个 InMemoryChatMessageHistory 并存入 store。
    """
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def invoke_with_memory(agent, session_id: str, user_content: str):
    """手动管理历史记忆调用 Agent。
    1. 获取历史消息
    2. 添加当前用户消息
    3. 调用 Agent
    4. 保存用户消息和 AI 响应到历史
    5. 返回响应
    """
    history = get_session_history(session_id)

    # 构建包含历史消息的输入
    messages = list(history.messages) + [HumanMessage(content=user_content)]

    # 调用 Agent
    response = agent.invoke({"messages": messages})

    # 将当前交互保存到历史
    history.add_message(HumanMessage(content=user_content))
    history.add_message(AIMessage(content=response["messages"][-1].content))

    return response

# --- 模型 & 工具准备 ---
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0,
)

search = TavilySearch(
    tavily_api_key=os.getenv("TAVILY_API_KEY"),
    max_results=2,
)
tools = [search]

# --- 直接创建 Agent ---
agent = create_agent(model=llm, tools=tools)

# --- 有记忆调用 ---
SESSION_ID = "user_xiaoming"

# 第一轮
response1 = invoke_with_memory(agent, SESSION_ID, "你好，我叫小明，我是一名程序员")
print(response1["messages"][-1].content)
print("---")

# 第二轮——这次能记住！
response2 = invoke_with_memory(agent, SESSION_ID, "你还记得我叫什么吗？")
print(response2["messages"][-1].content)

SESSION_ID_2 = "user_xiaohong"

response1 = invoke_with_memory(agent, SESSION_ID_2, "你好，我叫小红，我是一名学生")
print(response1["messages"][-1].content)
print("---")

response2 = invoke_with_memory(agent, SESSION_ID, "你还记得我叫什么吗？")
print(response2["messages"][-1].content)

```

---

这个模块的内容清楚吗？有任何疑问吗？

特别是关于 `input_messages_key` 和 `history_messages_key` 的配置，以及 `config` 中 `session_id` 的原理——如果没问题，我们就进入模块三：**进阶记忆策略——摘要记忆实战**，把你在模块一中自定义的 `SummaryChatMessageHistory` 和这里的 `RunnableWithMessageHistory` 结合起来。