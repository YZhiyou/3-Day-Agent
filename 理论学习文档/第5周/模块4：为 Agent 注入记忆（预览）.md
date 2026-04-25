好的，我们现在进入最后一个模块——**模块四：为 Agent 注入记忆（预览）**。这个模块作为第六周的桥梁，让你提前感知 Agent 从“一次性对话”进化到“多轮交互”的关键跃迁。

---

### 模块四：为 Agent 注入记忆（预览）

#### 1. 为什么 Agent 需要记忆？

到目前为止，我们构造的 Agent 每次执行都是**无状态**的。如果你连续问两个相关问题，它不会记得你第一轮说过什么。

比如：

- 第一轮：*“北京天气如何？”*
- 第二轮：*“那需要带伞吗？”*  
→ Agent 此时**并不知道**第一轮已经查过北京天气是晴天，它可能会再查一次天气，或者干脆无法理解“那”指什么。

**记忆**就是让 Agent 能够将过去的交互上下文保留下来，从而进行连贯对话。

#### 2. 最简单粗暴的记忆实现：拼接历史

在 LangChain 1.x 中，为 `create_agent` 添加记忆最简单的方式，就是**用 `chat_history` 参数手动传递历史消息**。

思路：
1. 维护一个消息列表（`history`），存储用户和 AI 的历史消息。
2. 在新对话时，将历史消息连同新问题一起发给 Agent。
3. Agent 结束后，将新问答追加到历史中。

#### 3. 完整带记忆的 Agent 示例代码

请在你的 `ai.py` 基础上添加以下代码（或新建一个文件）：

```python

import math
import os
from dotenv import load_dotenv

# LangChain 核心组件导入
import math
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek  # 如果您使用 DeepSeek
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.globals import set_verbose
from langchain_core.messages import HumanMessage, AIMessage

# 1. 加载 .env 环境变量
load_dotenv()

# 2. 配置 DeepSeek 模型实例
llm = ChatOpenAI(
    model="deepseek-chat",               # DeepSeek 模型名称
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"), # 从环境变量读取密钥
    base_url="https://api.deepseek.com/v1",       # DeepSeek API 地址
    temperature=0.7,                    
)

@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气情况。
    
    Args:
        city: 需要查询天气的城市名称（中文或英文）。
    
    Returns:
        一个字符串，描述该城市的天气状况和气温。
    """
    # 这是一个模拟函数，实际应用中可替换为真实 API 调用
    weather_data = {
        "北京": "晴天，气温18℃",
        "上海": "多云转阴，气温24℃",
        "广州": "雷阵雨，气温28℃",
        "深圳": "晴间多云，气温27℃"
    }
    # 简单容错处理
    return weather_data.get(city, f"{city}天气：局部多云，气温22℃。")

@tool
def calculator(expression: str) -> str:
    """执行数学计算。支持基本的四则运算和部分数学函数（如 math.sqrt、math.sin）。
    
    Args:
        expression: 一个有效的 Python 数学表达式字符串，例如 "3 * (5 + 2)" 或 "math.sqrt(16)"。
    
    Returns:
        计算结果的字符串形式；如果表达式无效则返回错误信息。
    """
    # 使用 Python 内置的 eval，但注意在生产环境中需严格限制命名空间以确保安全
    try:
        # 提供安全的 math 模块函数，避免执行危险代码
        allowed_names = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "math": math,
            **{k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"
    
tools = [get_weather, calculator]

# ----- 创建 Agent（不需要 AgentExecutor） -----
agent = create_agent(llm, tools)

# 开启全局 verbose，可看到工具调用详情（可选）
set_verbose(True)

# ----- 记忆管理 -----
history = []

def chat_with_memory(user_input: str, verbose: bool = True):
    global history
    # 构建完整消息列表：历史 + 新问题
    messages = history + [HumanMessage(content=user_input)]

    # 直接 invoke，输入格式是 {"messages": [...]}
    result = agent.invoke({"messages": messages})

    # --- 观察工具调用详情 ---
    if verbose:
        print("\n========== Agent 执行详情 ==========")
        for i, msg in enumerate(result["messages"]):
            msg_type = type(msg).__name__
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                # AI 决定调用工具
                print(f"\n[{i}] {msg_type} - 工具调用决定:")
                for tc in msg.tool_calls:
                    print(f"    工具名: {tc.get('name')}")
                    print(f"    参数:   {tc.get('args')}")
                    print(f"    ID:     {tc.get('id')}")
            elif hasattr(msg, "tool_call_id") and msg.tool_call_id:
                # 工具执行结果
                print(f"\n[{i}] {msg_type} - 工具返回结果:")
                print(f"    工具ID: {msg.tool_call_id}")
                print(f"    名称:   {msg.name}")
                print(f"    内容:   {msg.content}")
            elif msg_type == "AIMessage":
                # AI 最终回复
                print(f"\n[{i}] {msg_type} - 最终回复:\n    {msg.content}")
            elif msg_type == "HumanMessage":
                print(f"\n[{i}] {msg_type} - 用户输入:\n    {msg.content}")
            else:
                print(f"\n[{i}] {msg_type}:\n    {msg}")
        print("====================================\n")

    # 提取最后一条 AI 消息的内容
    ai_response = result["messages"][-1].content

    # 更新历史
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=ai_response))

    return ai_response

# ----- 多轮交互测试 -----
print("🤖 带记忆的 Agent 已启动（输入 'quit' 退出）\n")
while True:
    user_input = input("你：")
    if user_input.lower() in ["quit", "exit"]:
        break
    response = chat_with_memory(user_input)
    print(f"Agent：{response}\n")
```

#### 4. 测试记忆效果

假设我们进行如下交互：

```text
你：北京今天天气怎么样？
Agent：北京今天晴天，气温18℃。

你：那适合户外运动吗？
Agent：18℃的晴天非常适合户外运动，建议做好防晒。
```

Agent 在第二轮时**不需要再次调用天气工具**，因为它知道“那”指的就是北京天气，这个信息已经存在于对话历史中。这就是记忆的力量。

#### 5. 注意事项与下一步展望

- **API 调用成本**：历史越长，每次发送给模型的 token 越多。实际生产环境中需要做**记忆压缩**或**摘要**。
- **更优雅的记忆管理**：LangChain 提供了 `BaseChatMessageHistory` 和 `RunnableWithMessageHistory` 等封装，可以轻松将历史存储到数据库。**这些正是第六周将要深入学习的重点。**

#### 6. 可选实验（强烈建议）

1. **开启 Verbose 对比**：在带记忆的场景下开启 `verbose=True`，观察 Agent 是否在第二轮直接跳过了工具调用。
2. **测试遗忘**：对话六七轮后，再问一个依赖第一轮信息的问题，看看 Agent 是否还能提取出正确的上下文。

---

### 📌 模块四总结

- 记忆的本质是**在每次调用时把过去的对话上下文传进去**。
- 最简单的实现是手动维护 `HumanMessage` 和 `AIMessage` 列表。
- 这种方法为第六周学习的 `ConversationBufferMemory`、`ConversationSummaryMemory` 等高级记忆策略奠定了感性基础。
