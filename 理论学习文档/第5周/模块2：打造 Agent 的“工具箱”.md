### 模块二：打造 Agent 的“工具箱”

Agent 的“能力”来源于工具（Tools）。没有工具，Agent 只是一个只会聊天的模型；有了工具，它才能与外部世界交互。本模块我们将用 **`@tool` 装饰器**打造两件趁手的工具，并深刻理解 **docstring 是如何指导模型决策的**。

#### 1. 使用 `@tool` 装饰器定义工具

`@tool` 是 LangChain 提供的最便捷的工具定义方式。它将一个普通的 Python 函数包装成一个能被 Agent 理解并调用的 `Tool` 对象。

**核心要点**：
- 函数名称 -> **工具名称**（Agent 用它来标识工具）
- 函数 docstring -> **工具描述**（Agent 用它来**判断何时使用该工具以及如何传参**）
- 函数参数及类型注解 -> **工具输入参数 Schema**（Agent 根据它生成正确的 Action Input）

> ⚠️ **Docstring 至关重要！** 它是 Agent 在使用工具时唯一能看到的“说明书”。写得好不好，直接决定了 Agent 能否在正确的场景唤起正确的工具。

#### 2. 定义工具一：模拟天气查询工具

这个工具将返回指定城市的模拟天气信息。

```python
from langchain_core.tools import tool

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
```

#### 3. 定义工具二：计算器工具

这个工具执行数学表达式并返回结果。

```python
import math

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
            **{k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"
```

#### 4. 测试工具（非常重要！）

在将工具挂载到 Agent 之前，**务必先在 Python 脚本中独立测试它们**，以确保它们的行为符合预期。这是良好的开发习惯。

```python
# --- 测试代码 ---
if __name__ == "__main__":
    # 由于 @tool 包装后返回的是 Tool 对象，调用时需使用 .invoke() 或直接作为函数调用
    print("测试 get_weather 工具：")
    print(get_weather.invoke({"city": "北京"}))
    print(get_weather.invoke({"city": "成都"}))  # 测试不存在的城市
    
    print("\n测试 calculator 工具：")
    print(calculator.invoke({"expression": "3 * 5 + 2"}))
    print(calculator.invoke({"expression": "math.sqrt(25) + math.sin(math.pi/2)"}))
    print(calculator.invoke({"expression": "10 / 0"}))  # 测试错误情况
```

**运行上述测试代码，你应该看到类似输出：**

```
测试 get_weather 工具：
北京：晴天，气温18℃
成都：局部多云，气温22℃。

测试 calculator 工具：
17
6.0
计算错误：division by zero
```

#### 5. 理解 `@tool` 的内部结构

当你用 `@tool` 装饰一个函数后，它实际上会生成一个 `StructuredTool` 对象。我们可以窥探一下它的几个关键属性：

```python
# 观察工具元数据
print(get_weather.name)          # 输出: get_weather
print(get_weather.description)   # 输出: 查询指定城市的实时天气情况。...
print(get_weather.args_schema)   # 输出: 自动生成的 Pydantic 模型，包含 city 字段
```

Agent 在执行时，就是依据这些元数据来理解工具的用途和所需参数的。

