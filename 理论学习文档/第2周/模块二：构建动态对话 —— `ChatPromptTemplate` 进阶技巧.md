### 模块二：构建动态对话 —— `ChatPromptTemplate` 进阶技巧

在第一周，你使用了 `ChatPromptTemplate.from_template()` 来创建简单的单轮提示词模板。然而，真实应用中的对话往往是多轮次的，并且需要包含系统指令、历史记录和示例。今天我们将学习如何用 `from_messages` 构建结构更丰富的对话模板。

#### 1. 使用 `from_messages` 构建多角色提示

`ChatPromptTemplate.from_messages()` 方法允许我们明确指定每条消息的角色（`system`、`human`、`ai`），从而构建一个完整的对话上下文。

**基础语法示例：**

```python
from langchain_core.prompts import ChatPromptTemplate

# 使用元组列表定义消息角色和内容
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一位专业的{profession}，用友好而专业的口吻回答问题。"),
    ("human", "你好，我想咨询一下关于{topic}的问题。")
])

# 格式化消息，传入变量
messages = prompt_template.format_messages(
    profession="园艺师",
    topic="多肉植物养护"
)

print("=== 生成的消息列表 ===")
for msg in messages:
    print(f"角色: {msg.type.upper()}, 内容: {msg.content}")
```

**输出结果：**
```
=== 生成的消息列表 ===
角色: SYSTEM, 内容: 你是一位专业的园艺师，用友好而专业的口吻回答问题。
角色: HUMAN, 内容: 你好，我想咨询一下关于多肉植物养护的问题。
```

这种结构的好处是清晰地定义了对话的参与者和上下文，模型能更好地理解自己的定位。

#### 2. 管理对话历史 —— `MessagesPlaceholder`

对于聊天机器人，我们需要将历史对话记录动态地插入到提示词中。`MessagesPlaceholder` 就是为此而生的占位符，它可以在运行时被一个消息列表替换。

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 创建一个包含历史记录占位符的模板
prompt_with_history = ChatPromptTemplate.from_messages([
    ("system", "你是一个友善的助手。"),
    MessagesPlaceholder(variable_name="chat_history"),  # 历史记录插槽
    ("human", "{user_input}")
])

# 模拟一些历史消息
history = [
    HumanMessage(content="你好，我叫小明。"),
    AIMessage(content="你好小明！很高兴认识你。有什么可以帮你的吗？"),
    HumanMessage(content="今天天气怎么样？"),
    AIMessage(content="抱歉，我无法获取实时天气信息。你可以查看天气预报应用。")
]

# 格式化模板
messages = prompt_with_history.format_messages(
    chat_history=history,
    user_input="你能记住我的名字吗？"
)

print("=== 包含历史记录的提示消息 ===")
for msg in messages:
    print(f"角色: {msg.type.upper()}, 内容: {msg.content}")
```

**输出部分内容：**
```
角色: SYSTEM, 内容: 你是一个友善的助手。
角色: HUMAN, 内容: 你好，我叫小明。
角色: AI, 内容: 你好小明！很高兴认识你。有什么可以帮你的吗？
角色: HUMAN, 内容: 今天天气怎么样？
角色: AI, 内容: 抱歉，我无法获取实时天气信息。你可以查看天气预报应用。
角色: HUMAN, 内容: 你能记住我的名字吗？
```

这样，模型就能“看到”完整的对话上下文，从而做出连贯的回应。

#### 3. 少样本提示（Few-Shot Prompting）

有时我们想引导模型按照特定的格式或风格输出，提供几个“输入-输出”示例是最直接有效的方法。我们可以在提示词中直接嵌入示例对话。

```python
from langchain_core.prompts import ChatPromptTemplate, AIMessage, HumanMessage, SystemMessage

# 构建包含示例的提示词模板
few_shot_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="将用户输入的英文句子翻译成中文，并按照'原文：xxx\n译文：xxx'的格式输出。"),
    # 示例1
    HumanMessage(content="Hello, how are you?"),
    AIMessage(content="原文：Hello, how are you?\n译文：你好，你怎么样？"),
    # 示例2
    HumanMessage(content="The weather is nice today."),
    AIMessage(content="原文：The weather is nice today.\n译文：今天天气很好。"),
    # 真实用户输入
    HumanMessage(content="{input}")
])

# 格式化
messages = few_shot_prompt.format_messages(input="I love programming.")
print("=== 少样本提示消息列表 ===")
for msg in messages:
    print(f"{msg.type.upper()}: {msg.content[:50]}...")  # 只打印前50个字符
```

运行这段代码，模型会接收到两个示例对话，然后处理真实的用户输入。这样它就会明白，应该严格遵循“原文：xxx\n译文：xxx”的格式来输出。

#### 4. 模块小结

- `from_messages` 提供了比 `from_template` 更精细的对话结构控制。
- `MessagesPlaceholder` 是处理对话历史的利器，让聊天应用能记住上下文。
- 在提示词中直接插入示例是实现少样本提示的简单有效方式。

