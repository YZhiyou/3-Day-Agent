

### 模块三：组装 Agent 并可视化推理过程（LangGraph 重制版）

#### 1. 理解 LangGraph 的 ReAct Agent

LangGraph 是一个用于构建状态多步工作流的框架。它的 `create_react_agent` 函数会生成一个标准 ReAct Agent，底层维护一个消息列表作为状态。当我们用 `stream` 方法运行时，每一步状态变化都会直观暴露，让我们看到模型的内心独白。

#### 2. 基础导入与模型初始化

```python
import os
import math
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek  # 如果您使用 DeepSeek
from langchain.agents import create_agent
from langchain_core.tools import tool

# 选择您的模型（任选其一）
# 使用 OpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 或使用 DeepSeek（如果您在用 langchain-deepseek）
# llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
```

#### 3. 定义工具（复用模块二代码）

```python
@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气情况。
    
    Args:
        city: 需要查询天气的城市名称（中文或英文）。
    
    Returns:
        一个字符串，描述该城市的天气状况和气温。
    """
    weather_data = {
        "北京": "晴天，气温18℃",
        "上海": "多云转阴，气温24℃",
        "广州": "雷阵雨，气温28℃",
        "深圳": "晴间多云，气温27℃"
    }
    return weather_data.get(city, f"{city}天气：局部多云，气温22℃。")

@tool
def calculator(expression: str) -> str:
    """执行数学计算。支持基本的四则运算和部分数学函数（如 math.sqrt、math.sin）。
    
    Args:
        expression: 一个有效的 Python 数学表达式字符串，例如 "3 * (5 + 2)"。
    
    Returns:
        计算结果的字符串形式；如果表达式无效则返回错误信息。
    """
    try:
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"

tools = [get_weather, calculator]
```

#### 4. 创建 ReAct Agent

这一步极其简单，一行代码搞定。

```python
agent = create_agent(llm, tools)
```

注意：LangGraph 的 ReAct Agent 内置了精确的 `max_iterations` 控制吗？实际上，它通过 `recursion_limit` 在 `config` 中控制最大步数。我们稍后在流式调用时会看到。

#### 5. 流式执行并分析推理过程（核心环节）

LangGraph Agent 的 `stream` 方法会返回一个生成器，每一步产生一个状态块。我们将这个块打印出来，您就能亲眼看到 Agent 的决策循环。

```python
query = "今天北京气温多少度？如果低于25度，请帮我计算150美元等于多少人民币，假设汇率是7.25。"

print(f"用户提问: {query}\n")
print("=" * 60)

# 流式执行，返回每个节点的数据
for chunk in agent.stream(
    {"messages": [("user", query)]},
    config={"recursion_limit": 10}  # 控制最大循环次数（包括工具调用和思考轮次）
):
    # 每个 chunk 是一个包含节点名称和该节点状态更新的字典
    for node_name, node_output in chunk.items():
        print(f"\n--- 节点: {node_name} ---")
        if "messages" in node_output:
            # 获取最新的一条消息
            last_msg = node_output["messages"][-1]
            # 打印角色和内容
            role = getattr(last_msg, "type", "unknown")
            if role == "ai":
                print(f"AI 思考或行动: {last_msg.content}")
                # 如果消息中含有工具调用请求，打印出来
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        print(f"   -> 调用工具: {tc['name']}")
                        print(f"   -> 传入参数: {tc['args']}")
            elif role == "tool":
                print(f"工具观测结果: {last_msg.content}")
        print("-" * 40)
```

**预期输出分析**（您将看到的类似日志）：

```
用户提问: 今天北京气温多少度？如果低于25度，请帮我计算150美元等于多少人民币，假设汇率是7.25。

============================================================

--- 节点: agent ---
AI 思考或行动: 
   -> 调用工具: get_weather
   -> 传入参数: {'city': '北京'}
----------------------------------------

--- 节点: tools ---
工具观测结果: 北京：晴天，气温18℃
----------------------------------------

--- 节点: agent ---
AI 思考或行动: 
   -> 调用工具: calculator
   -> 传入参数: {'expression': '150 * 7.25'}
----------------------------------------

--- 节点: tools ---
工具观测结果: 1087.5
----------------------------------------

--- 节点: agent ---
AI 思考或行动: 今天北京气温为18℃，低于25度。按照7.25的汇率，150美元相当于1087.5元人民币。
----------------------------------------
```

> 🔍 **深度剖析**：
> - **`agent` 节点**对应模型的推理步骤。当它输出 `tool_calls` 时，就是 ReAct 中的 **Action** 阶段。
> - **`tools` 节点**对应外部工具的执行与结果返回，即 ReAct 的 **Observation** 阶段。
> - 在 `agent` 节点无工具调用且有自然语言内容时，即为 **Final Answer**。

这种流式输出完美实现了 **verbose 的可视化思考**，而且更加结构化！

#### 6. 获取最终回答（非流式调用）

如果您只需要最终答案而不需要中间过程，也可以使用 `invoke` 方法：

```python
result = agent.invoke({"messages": [("user", query)]})
final_answer = result["messages"][-1].content
print("最终回答:\n", final_answer)
```

#### 7. 参数控制：最大迭代数

如前所述，LangGraph ReAct Agent 通过 `recursion_limit` 控制总体步数。`create_react_agent` 实际内部构建了一个循环图，每次模型调用和每次工具执行都算一步。一般设置 10~15 即可。

#### 8. 课内小实验：观察 Agent 的挣扎

为了加深理解，建议您尝试以下变体：

- **实验 1**：将 `get_weather` 的 docstring 改成「获取当前时间」，再问相同问题，观察 Agent 是否会错误地选用天气工具。
- **实验 2**：在 `calculator` 工具中故意传入错误的表达式（例如在流式过程中手动打断是不可能的，但可以构造一个必定报错的表达式），看看 Agent 如何处理错误信息。

---

### 📌 模块三小结

- 使用 LangGraph 的 `create_react_agent` 是目前 LangChain 1.x 中最接近原生 ReAct 教学体验的方法。
- 通过 `agent.stream()` 遍历输出，我们可以清晰地看到 **Thought（隐式在 AI 的工具调用请求中）**、**Action（工具调用）**、**Observation（工具响应）** 的完整流转。
- `recursion_limit` 替代了旧版 `max_iterations` 的角色。

---

**这个模块的内容清楚吗？尤其是关于流式输出如何对应 ReAct 循环，还有任何疑问吗？**

如果没问题，我们即可进入最后一个模块——**模块四：为 Agent 注入记忆（预览）**。