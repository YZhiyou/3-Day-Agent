## 综合拓展：RAG + Memory 融合设计思路

### 4.1 为什么需要融合？

单纯的 Agent 记忆能记住“用户说过什么”，但无法主动影响系统的**行为策略**。如果我们将记忆与 RAG（检索增强生成）结合，就能实现更智能的个性化：

- **场景举例**：
  - 一个技术客服 Agent，记住用户是“后端开发，主要用 Go 语言”，后续检索文档时可以**优先返回 Go 相关的结果**。
  - 一个电商导购 Agent，记住用户偏好“价格区间 200-500 元，喜欢极简风格”，检索商品时自动加上过滤条件。

本质上是：**用记忆存储用户画像/偏好，用这些画像动态调整 RAG 的检索参数**。

---

### 4.2 架构设计（文字描述）

```
用户输入
  │
  ▼
┌─────────────────────────┐
│  偏好管理器              │
│  (从 session 历史提取    │
│   用户画像/偏好标签)     │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  RAG 检索增强            │
│  (根据偏好改写查询，     │
│   或添加元数据过滤)      │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Agent (带工具)          │
│  (接收检索结果+历史消息) │
└───────────┬─────────────┘
            │
            ▼
         最终回答
```

**关键数据流**：
1. 用户输入进入后，先经过“偏好管理器”，它从当前会话历史中提取用户画像（可以是简单的关键词提取，或调用 LLM 生成结构化偏好）。
2. 偏好标签去影响 RAG 检索：可以改写查询语句（如原始查询“推荐电脑” → “推荐电脑 轻便 适合编程”），或者作为向量检索的元数据过滤条件。
3. 检索到的文档连同历史消息一起传给 Agent，Agent 综合生成回答。
4. 回答中的新偏好可以再次更新记忆。

---

### 4.3 简化版可运行代码骨架

下面演示一个**最小可行版本**：用一个简单的规则从历史中提取偏好，并影响检索查询的改写。

```python
import os
from dotenv import load_dotenv
from typing import List

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.tools import tool

load_dotenv()

# ========== 1. 基础组件 ==========
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0,
)

embeddings = DashScopeEmbeddings(model="text-embedding-v4")  # 使用 DashScope 的文本嵌入模型

# 模拟一个简单的商品文档库
docs = [
    "轻薄笔记本，仅重1.2kg，适合编程与出差, 价格: 4500",
    "高性能游戏本，RTX 4060显卡，适合游戏与渲染, 价格: 8000",
    "极简设计超极本，全金属机身，适合商务办公, 价格: 6000",
    "学生本，大屏护眼，适合网课学习, 价格: 3000",
]
vectorstore = Chroma.from_texts(docs, embedding=embeddings)

# ========== 2. 偏好提取函数 ==========
def extract_preferences(history: InMemoryChatMessageHistory) -> dict:
    """从历史消息中提取用户偏好（简化版：关键词匹配）"""
    all_text = " ".join([m.content for m in history.messages])
    prefs = {}
    if "编程" in all_text or "开发" in all_text:
        prefs["usage"] = "编程"
    if "轻" in all_text or "轻薄" in all_text:
        prefs["feature"] = "轻薄"
    if "游戏" in all_text:
        prefs["usage"] = "游戏"
    return prefs

# ========== 3. 智能检索工具 ==========
@tool
def smart_search(query: str, session_id: str) -> str:
    """根据用户偏好增强的搜索工具"""
    history = get_session_history(session_id)
    prefs = extract_preferences(history)
    
    # 根据偏好改写查询
    enhanced_query = query
    if prefs.get("usage") == "编程":
        enhanced_query = f"{query} 适合编程"
    if prefs.get("feature") == "轻薄":
        enhanced_query = f"{enhanced_query} 轻薄"
    
    print(f"原始查询: {query} → 增强查询: {enhanced_query}")
    
    # 执行检索
    results = vectorstore.similarity_search(enhanced_query, k=2)
    return "\n".join([doc.page_content for doc in results])

# ========== 4. 会话管理（复用之前模块） ==========
store = {}
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def invoke_with_memory(agent, session_id: str, user_content: str):
    """手动管理历史记忆调用 Agent（支持工具使用 session_id）"""
    history = get_session_history(session_id)
    messages = list(history.messages) + [HumanMessage(content=user_content)]
    # 在调用时传入 session_id，让工具能获取偏好
    response = agent.invoke({"messages": messages, "session_id": session_id})
    history.add_message(HumanMessage(content=user_content))
    history.add_message(AIMessage(content=response["messages"][-1].content))
    return response

# ========== 5. 创建 Agent ==========
agent = create_agent(
    model=llm, 
    tools=[smart_search],
    # 注意：create_agent 的 state_schema 需要包含 session_id 字段，这里简化处理，
    # 实际中可能需要自定义 State 或通过 RunnableConfig 传递
)

# ========== 6. 模拟对话流程 ==========
SESSION_ID = "user_tech"

# 第一轮：表达偏好
invoke_with_memory(agent, SESSION_ID, "我需要一台适合编程的笔记本，希望轻便一些")
print("---")

# 第二轮：使用增强搜索
response = invoke_with_memory(agent, SESSION_ID, "推荐一款笔记本")
print("Agent 回答：", response["messages"][-1].content)

# 此时 smart_search 会看到历史中的“编程”和“轻便”偏好，
# 自动改写查询为“推荐一款笔记本 适合编程 轻薄”，返回相关款。
```

**运行效果**：
- 第一次询问时，用户表达了“编程”和“轻便”的偏好，这些信息被保存在历史中。
- 第二次“推荐一款笔记本”时，`smart_search` 工具提取历史偏好，将查询增强为“推荐一款笔记本 适合编程 轻薄”，使向量检索精准返回轻薄编程本（4500元款），而非游戏本或学生本。

---

### 4.4 拓展方向

这个骨架可以进一步升级为：
1. **LLM 驱动的偏好提取**：用户模型自动分析整个历史，输出结构化 JSON（如 `{"role": "developer", "preferred_price_range": "4000-6000"}`），存入持久化存储，用于多会话持续影响。
2. **元数据过滤**：在 Milvus / Pinecone 等向量库中给文档打标签，偏好直接转为过滤条件（`filter={"usage": "programming", "weight": {"$lt": 1.5}}`），比改写查询更精确。
3. **主动偏好澄清**：Agent 在发现偏好缺失时主动提问（“您主要用于游戏还是办公？”），新偏好立刻写入记忆，下一轮检索立即生效。

这样的架构将“记忆”从被动的对话记录升级为**主动的策略驱动者**，是 RAG+Agent 的实用进阶方向。

---

### ✅ 模块四总结

现在你已掌握：
- 记忆在对话中的角色
- 三种经典记忆策略的实现与选择
- 用 `invoke_with_memory` 为 Agent 添加手动记忆管理
- 摘要记忆的实战集成
- RAG + Memory 融合的设计思路

---

## 🎓 第六周学习笔记摘要

#### 记忆管理架构图
```
用户输入 → [提取 session_id] 
          → get_session_history(session_id) 返回该会话的 ChatMessageHistory 实例
          → 历史消息 + 新用户消息 → Agent (带工具)
          → 响应中的新消息写回 history.add_message()
          
多会话隔离：全局 store 字典以 session_id 为键，
不同 id 指向独立的历史对象，互不干扰。
```

#### 关键代码资产
```python
# 完整的带记忆 Agent（手动管理历史）
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

store = {}
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def invoke_with_memory(agent, session_id: str, user_content: str):
    history = get_session_history(session_id)
    messages = list(history.messages) + [HumanMessage(content=user_content)]
    response = agent.invoke({"messages": messages})
    history.add_message(HumanMessage(content=user_content))
    history.add_message(AIMessage(content=response["messages"][-1].content))
    return response

# 创建 Agent
agent = create_agent(model=llm, tools=tools)

# 调用示例
response1 = invoke_with_memory(agent, "user1", "我叫小明")
response2 = invoke_with_memory(agent, "user1", "我叫什么？")  # 正确回答
```

#### 选型建议备忘

| 场景特点                         | 推荐记忆策略                        | 理由                             |
| -------------------------------- | ----------------------------------- | -------------------------------- |
| 极短对话（<5轮）或调试           | `InMemoryChatMessageHistory` (全量) | 简单，无信息丢失                 |
| 近期上下文敏感，需控制成本       | `WindowedChatMessageHistory` (窗口) | Token 上限固定，适合客服FAQ      |
| 长对话，需要全局理解，细节可损失 | `SummaryChatMessageHistory` (摘要)  | 摘要捕获关键信息，Token 增长缓慢 |
| 个性化 RAG 系统，需长期偏好      | 自定义画像记忆 + 元数据过滤         | 记忆驱动检索策略，而非仅记录对话 |
