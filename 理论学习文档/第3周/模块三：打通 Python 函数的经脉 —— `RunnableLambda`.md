## 模块三：打通 Python 函数的经脉 —— `RunnableLambda`

### 3.1 为什么需要 `RunnableLambda`？

到目前为止，我们串联的都是 LangChain 内置的组件（`ChatPromptTemplate`、`ChatOpenAI`、`StrOutputParser`）。但在真实项目中，你几乎一定会遇到需要插入自定义逻辑的场景：

- 对用户输入做**预处理**：去除首尾空格、过滤敏感词、计算字符串长度。
- 对模型输出做**后处理**：转换为大写、提取特定字段、保存到数据库。
- **调用外部 API**：在生成回复前，先查询天气接口或知识库。

`RunnableLambda` 的作用就是：**将任意 Python 函数包装成一个符合 `Runnable` 协议的对象**，从而可以用 `|` 管道符串联。

### 3.2 基础用法：包装一个简单函数

```python
from langchain_core.runnables import RunnableLambda

# 1. 定义一个普通的 Python 函数
def to_uppercase(text: str) -> str:
    return text.upper()

# 2. 包装成 Runnable
uppercase_runnable = RunnableLambda(to_uppercase)

# 3. 单独调用
print(uppercase_runnable.invoke("hello langchain"))
# 输出: HELLO LANGCHAIN
```

**关键点**：函数的**输入和输出类型需要明确**，因为 LCEL 会在连接时做类型检查。通常我们期望函数接收一个 `dict` 或 `str`，并返回一个可以被下游组件消费的类型（如 `str` 或 `dict`）。

### 3.3 实战：将函数插入链中

我们来构建一个带后处理的链：生成笑话 → 转为大写。

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8)
prompt = ChatPromptTemplate.from_template("讲一个关于 {topic} 的笑话，一句话。")
parser = StrOutputParser()

# 自定义后处理函数
def add_emphasis(text: str) -> str:
    """在笑话前后加上强调符号"""
    return f"🔥 {text.upper()} 🔥"

# 将函数包装为 Runnable
emphasis_runnable = RunnableLambda(add_emphasis)

# 串联链
chain = prompt | model | parser | emphasis_runnable

# 测试
result = chain.invoke({"topic": "程序员"})
print(result)
# 输出示例: 🔥 为什么程序员总分不清万圣节和圣诞节？因为 OCT 31 == DEC 25！ 🔥
```

### 3.4 处理更复杂的输入输出（字典操作）

很多时候，函数需要从字典中提取特定字段，处理后返回新的字典。这时我们需要注意函数签名。

```python
# 场景：计算用户输入文本的字数，然后将字数附加到输入中传给下游

def count_words(input_dict: dict) -> dict:
    """从输入字典中读取 text，计算字数，并添加到字典中"""
    text = input_dict.get("text", "")
    word_count = len(text.split())
    # 返回增强后的字典，保留原有字段
    return {
        **input_dict,          # 保留原始字段
        "word_count": word_count
    }

# 包装
word_counter = RunnableLambda(count_words)

# 一个简单的链：计算字数 -> 模型根据字数给出反馈
feedback_prompt = ChatPromptTemplate.from_template(
    "用户输入了 {word_count} 个字。请根据这个信息说一句鼓励的话。"
)

chain = word_counter | feedback_prompt | model | parser

result = chain.invoke({"text": "LangChain 真的很有趣，我想深入学习它。"})
print(result)
```

### 3.5 异步函数支持

如果你的自定义逻辑涉及网络请求或文件 I/O，可以定义为 `async def` 函数。`RunnableLambda` 会自动识别并提供异步调用能力。

```python
import asyncio

async def fetch_weather(city: str) -> str:
    """模拟异步获取天气（实际中可能是 aiohttp 请求）"""
    await asyncio.sleep(1)  # 模拟网络延迟
    return f"{city} 的天气是晴天，温度 22°C"

weather_runnable = RunnableLambda(fetch_weather)

# 在异步环境中调用
# result = await weather_runnable.ainvoke("北京")
```

### 3.6 使用装饰器 `@chain`（进阶可选）

LangChain 还提供了一个 `@chain` 装饰器，让你可以用更 Pythonic 的方式定义复杂的自定义链，但核心依然是 `RunnableLambda` 的思想。这里不展开，我们聚焦基础用法即可满足大多数场景。

---

### 模块小结

- `RunnableLambda` 让你能够把**任何 Python 函数**变成 LCEL 中的标准零件。
- 函数的输入输出应当与上下游组件的期望匹配（通常是 `str` 或 `dict`）。
- 它是连接 AI 逻辑与业务逻辑的桥梁。

现在你已经拥有了**自定义数据处理节点**的能力，不再受限于 LangChain 内置组件。

---

这个模块的内容清楚吗？关于 `RunnableLambda` 的使用，或者函数签名处理，有任何疑问吗？如果没有，我们就进入今天的最后一个模块——`RunnableBranch` 与 `.bind()`，学习如何让链拥有**条件判断**和**参数配置**的能力。