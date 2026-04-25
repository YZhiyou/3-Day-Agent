

## 模块二：数据流的魔术师 —— `RunnablePassthrough` 与 `RunnableParallel`

### 2.1 场景引入：链的数据流动困境

回顾模块一的链：`prompt | model | parser`，数据流向非常清晰，一路向前。但实际开发中我们常遇到两个痛点：

1. **数据丢失**：经过层层处理后，原始的输入（比如 `topic`）在最后一步找不到了，而我们希望在输出中保留它。
2. **单任务瓶颈**：模型调用通常是最耗时的环节，但我们可能需要在**同一时刻**对同一份输入做多件事（例如：既要生成摘要，又要判断情感极性）。

LCEL 提供了两个专门解决这两个问题的原语：
- `RunnablePassthrough`：数据流的“透明管道”，负责把数据原封不动传递下去。
- `RunnableParallel`：数据流的“分叉路口”，负责并行执行多个子链，并自动合并结果。

### 2.2 `RunnablePassthrough`：数据的“直通车”

它的功能极其简单：输入什么，就输出什么。你可能会问：“那要它干嘛？” 答案是：**占位**。

当你用 LCEL 构建一个复杂的输入字典时，`RunnablePassthrough` 可以用来表示“把用户原始输入整体放到这里”。

**基础用法示例：**

```python
from langchain_core.runnables import RunnablePassthrough

# 它本身就是一个 Runnable
passthrough = RunnablePassthrough()
print(passthrough.invoke({"name": "LangChain", "version": 3}))
# 输出: {'name': 'LangChain', 'version': 3} （完全不变）
```

**在链中的实战意义：**

假设我们希望链的最后输出不仅包含模型生成的笑话，还要附带用户最初输入的 `topic`。可以这样构建：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

model = ChatOpenAI(model="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_template("关于 {topic} 的一句话笑话")
parser = StrOutputParser()

# 构建一个并行字典，同时保留原始输入和模型结果
chain = (
    # 第一部分：告诉链，最终输出字典的结构是什么样的
    {
        "original_topic": RunnablePassthrough(),  # 直接把整个输入拿过来
        "joke": prompt | model | parser           # 执行原来的笑话生成链
    }
    # 注意：这里没有 | 连接后续组件，因为字典本身就是一个 Runnable
)

# 调用
result = chain.invoke({"topic": "人工智能"})
print(result)
# 输出示例: {'original_topic': {'topic': '人工智能'}, 'joke': '人工智能为什么不会迷路？因为它有算法导航。'}
```

### 2.3 `RunnableParallel`：任务的“并发调度器”

实际上，上面的字典 `{}` 语法在 LCEL 中会自动被转换为 `RunnableParallel` 对象。也就是说，下面两种写法是**完全等价**的：

```python
# 显式使用 RunnableParallel
from langchain_core.runnables import RunnableParallel
parallel_chain = RunnableParallel(key1=chain1, key2=chain2)
```

**关键特性**：字典里的每个值（`chain1`、`chain2`）会被**并发执行**。如果 `chain1` 调用模型需要 2 秒，`chain2` 调用模型也需要 2 秒，那么整个 `parallel_chain` 的总耗时约等于 2 秒（而不是 4 秒）。

### 2.4 综合实战：同时生成摘要和关键词

让我们把学到的结合起来，构建一个实用性较强的链：输入一段文本，**同时**输出摘要和关键词列表，并保留原文。

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 初始化
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 两个专用提示词模板
summary_prompt = ChatPromptTemplate.from_template(
    "用一句话概括以下文本的主要内容：\n{text}"
)
keyword_prompt = ChatPromptTemplate.from_template(
    "从以下文本中提取 3 个最重要的关键词，用逗号分隔：\n{text}"
)

# 两个子链
summary_chain = summary_prompt | model | StrOutputParser()
keyword_chain = keyword_prompt | model | StrOutputParser()

# 并行主链：输入字典包含 "text" 字段
parallel_chain = RunnableParallel(
    original_text=RunnablePassthrough(),
    summary=summary_chain,
    keywords=keyword_chain
)

# 测试
input_data = {
    "text": "LangChain 是一个用于开发由语言模型驱动的应用程序的框架。它提供了模块化组件和预置链，使得构建复杂 AI 应用变得简单。LCEL 是 LangChain 的表达语言，允许开发者用管道符串联组件。"
}

result = parallel_chain.invoke(input_data)

print("=== 原文 ===")
print(result["original_text"]["text"])
print("\n=== 摘要 ===")
print(result["summary"])
print("\n=== 关键词 ===")
print(result["keywords"])
```

**输出预览：**
```
=== 原文 ===
LangChain 是一个用于开发由语言模型驱动的应用程序的框架...

=== 摘要 ===
LangChain 是一个简化由语言模型驱动的复杂 AI 应用开发的框架，并提供 LCEL 表达语言以便串联组件。

=== 关键词 ===
LangChain, 语言模型, LCEL
```

### 2.5 模块小结

- **`RunnablePassthrough`** 解决了“数据保真”问题，确保原始信息不会在链中被丢弃。
- **字典语法（即 `RunnableParallel`）** 解决了“多任务并发”问题，显著提升了多步处理的效率。

你现在已经能够构建带有“数据旁路”和“并行分支”的复杂流水线了。

---

这个模块的内容清楚吗？关于 `RunnablePassthrough` 和并行字典的用法，或者那个综合示例的逻辑，有任何疑问吗？如果没有，我们就进入下一个模块，学习如何将你自定义的 Python 函数变成链中的一环——`RunnableLambda`。