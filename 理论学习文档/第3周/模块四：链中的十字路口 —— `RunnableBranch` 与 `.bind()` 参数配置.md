## 模块四：链中的十字路口 —— `RunnableBranch` 与 `.bind()` 参数配置

### 4.1 场景引入：为什么需要分支？

之前的链都是**线性**的：无论输入什么，数据都沿着固定路线流动。但在真实场景中，我们往往需要根据输入的性质来**切换处理逻辑**。例如：

- 用户输入是“问候语” → 走**礼貌回复链**（简单回应即可）
- 用户输入是“技术问题” → 走**专家解答链**（需要详细推理）
- 用户输入是“投诉” → 走**安抚与转接链**（特殊话术）

`RunnableBranch` 就是专门用于实现这种**条件路由**的组件。

### 4.2 `RunnableBranch` 的基本语法

`RunnableBranch` 接收一个列表，列表中的每一项是一个 `(条件函数, 目标链)` 的元组。当条件函数返回 `True` 时，就会执行对应的目标链。如果所有条件都不满足，则执行一个默认链。

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: "hello" in x["input"].lower(), greeting_chain),  # 如果包含 hello，走 greeting_chain
    (lambda x: "error" in x["input"].lower(), error_chain),     # 如果包含 error，走 error_chain
    default_chain                                                # 否则走默认链
)
```

**条件函数**的签名通常是 `(input_dict) -> bool`，它接收当前链的输入，返回一个布尔值。

### 4.3 实战：构建一个智能客服路由系统

我们将构建一个链，根据用户消息的内容，自动选择不同的“人格”模型来回复。

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda

# 初始化基础模型（温度稍高，让回复更有风格）
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
parser = StrOutputParser()

# --- 定义三条不同风格的子链 ---

# 链1：友好闲聊风格
greeting_prompt = ChatPromptTemplate.from_template(
    "你是一个热情友好的助手。请用温暖愉快的语气回复：{input}"
)
greeting_chain = greeting_prompt | model | parser

# 链2：专业技术支持风格
tech_prompt = ChatPromptTemplate.from_template(
    "你是一个资深技术专家。请用专业、准确的语言回答以下技术问题，必要时提供代码示例：{input}"
)
tech_chain = tech_prompt | model | parser

# 链3：默认通用助手
general_prompt = ChatPromptTemplate.from_template(
    "你是一个乐于助人的助手。请回答用户的问题：{input}"
)
general_chain = general_prompt | model | parser

# --- 定义分支条件（基于输入文本的简单关键词判断）---

def is_greeting(input_dict: dict) -> bool:
    """判断是否为问候语"""
    text = input_dict["input"].lower()
    greeting_words = ["hello", "hi", "你好", "您好", "早上好", "晚上好"]
    return any(word in text for word in greeting_words)

def is_tech_question(input_dict: dict) -> bool:
    """判断是否为技术问题"""
    text = input_dict["input"].lower()
    tech_keywords = ["python", "代码", "bug", "错误", "编程", "api", "数据库", "langchain"]
    return any(keyword in text for keyword in tech_keywords)

# 构建分支链
branch_chain = RunnableBranch(
    (is_greeting, greeting_chain),
    (is_tech_question, tech_chain),
    general_chain  # 默认链
)

# 测试不同输入
inputs = [
    {"input": "Hello! 今天天气真好！"},
    {"input": "Python 里面如何读取 CSV 文件？请给我一段代码。"},
    {"input": "给我讲个笑话吧。"}
]

print("=== 智能路由测试 ===\n")
for i, inp in enumerate(inputs, 1):
    print(f"用户输入 {i}: {inp['input']}")
    response = branch_chain.invoke(inp)
    print(f"助手回复 {i}: {response}\n")
```

**运行效果说明：**
- 第一条输入触发了问候链，回复会带有热情友好的语气。
- 第二条输入触发了技术链，回复会包含代码示例和专业解释。
- 第三条输入不满足任何条件，走默认通用链。

您观察得非常仔细，感谢指正！我上一节的示例确实**没有展示 `.bind()` 的真正常用场景**——用不同模型实例来区分温度虽然可行，但并没有体现 `.bind()` 在参数传递上的灵活性。让我重新用一个更准确的例子来演示 `.bind()` 的核心价值。

---

### 4.4 .bind()` 的真正用法：动态覆盖模型参数

`.bind()` 的作用是：**在链的构建阶段，为某个 Runnable 预先绑定一组参数。** 当链被调用时，这些参数会自动附加到该组件的实际执行参数中，**优先级高于运行时传入的同名参数**。

最典型的使用场景是 **Function Calling（工具调用）**，但为了聚焦概念，我们依然用 `temperature` 参数来演示，关键在于：**我们只用一个模型实例，但通过 `.bind()` 让它在不同分支表现出不同的温度**。

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch

# 初始化一个基础模型（温度不固定，将在分支中通过 .bind() 覆盖）
base_model = ChatOpenAI(model="gpt-3.5-turbo")

# 提示词（略）
greeting_prompt = ChatPromptTemplate.from_template("热情回复：{input}")
tech_prompt = ChatPromptTemplate.from_template("专业回复：{input}")
general_prompt = ChatPromptTemplate.from_template("通用回复：{input}")

parser = StrOutputParser()

# === .bind() 的关键用法：为同一模型实例绑定不同参数 ===

# 注意：这里不是直接使用 base_model，而是使用 base_model.bind(temperature=xxx)
greeting_chain = greeting_prompt | base_model.bind(temperature=0.9) | parser
tech_chain = tech_prompt | base_model.bind(temperature=0.1) | parser
general_chain = general_prompt | base_model.bind(temperature=0.5) | parser

# 分支条件函数（同前）
def is_greeting(d): return any(w in d["input"].lower() for w in ["hello", "hi", "你好"])
def is_tech(d): return any(w in d["input"].lower() for w in ["python", "代码", "bug"])

branch_chain = RunnableBranch(
    (is_greeting, greeting_chain),
    (is_tech, tech_chain),
    general_chain
)

# 测试
print(branch_chain.invoke({"input": "Hello!"}))
print(branch_chain.invoke({"input": "Python 列表推导式怎么用？"}))
```

**这里发生了什么？**
- 我们只创建了一个 `ChatOpenAI` 实例 `base_model`。
- 在定义链时，通过 `.bind(temperature=0.9)` **创建了一个新的 Runnable 副本**，该副本在调用时会强制使用 `temperature=0.9`。
- 三个分支使用了**同一个底层模型连接配置**，但每次调用的参数不同。

### 为什么这比新建三个模型实例更好？

1. **代码语义更清晰**：`.bind()` 明确表达了“我在为这个节点固定特定参数”。
2. **更灵活的参数覆盖**：除了温度，你还可以绑定 `stop`、`max_tokens`、`tools` 等任意模型支持的关键字参数。
3. **与 LangChain 运行时的深度集成**：某些高级场景（如配置中心化、动态配置注入）依赖于 `.bind()` 这种参数声明方式。

### `.bind()` 与 `.with_config()` 的区别（补充知识）

你可能还会看到 `.with_config()`，它的作用类似但更底层。简单区分：
- `.bind(**kwargs)`：专门用于向**模型 API 调用**传递额外参数（如 `temperature`、`tools`）。
- `.with_config(config)`：用于传递 LangChain 内部运行时配置（如 `tags`、`metadata`、`callbacks`）。

---

### 4.5 模块小结

- **`RunnableBranch`**：根据输入条件动态选择执行路径。
- **`.bind()`**：在不改变组件核心实例的前提下，为特定调用节点**静态绑定**参数，实现精细化行为控制。

