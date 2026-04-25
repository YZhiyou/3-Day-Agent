## 模块一：一切皆是 Runnable —— 统一接口的威力

### 1.1 什么是 `Runnable` 协议？

在 LangChain 的世界里，几乎你能接触到的所有组件——`ChatModel`、`PromptTemplate`、`OutputParser`、甚至是上一周你用过的 `PydanticOutputParser`——它们底层都实现了同一个接口：**`Runnable`**。

这个接口定义了一套标准化的“运行方法”。正因为大家都遵循同一套规则，我们才能用 `|` 管道符像搭积木一样把它们串起来。这就是 LCEL 能够工作的根本原因。

你可以把 `Runnable` 想象成一个**统一规格的水管接头**：
- 提示词模板是一个接头（输入 → 格式化输出）
- 大模型是一个接头（输入 → 推理输出）
- 解析器是一个接头（输入 → 结构化对象）

把它们拧在一起，数据就能像水流一样顺畅通过。

### 1.2 三个核心方法：`invoke`、`batch`、`stream`

任何 `Runnable` 对象都至少提供以下三个方法。理解它们的区别，是高效使用 LCEL 的关键。

| 方法     | 输入类型         | 执行方式                 | 适用场景                       |
| :------- | :--------------- | :----------------------- | :----------------------------- |
| `invoke` | 单个 `dict`      | 单次同步执行             | 实时对话、单条数据处理         |
| `batch`  | `list` of `dict` | 内部自动并发执行（异步） | 批量离线处理、评估测试集       |
| `stream` | 单个 `dict`      | 逐 Token 流式返回        | 聊天界面打字机效果、长文本生成 |

### 1.3 实战演练：亲身体验三种调用方式

我们来创建一个简单的链，并用这三种方法分别调用它，直观感受差异。

**请在你的 Jupyter Notebook 或 Python 脚本中运行以下代码：**

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 准备组件（都是 Runnable）
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
prompt = ChatPromptTemplate.from_template("给我讲一个关于 {topic} 的冷笑话，一句话。")
parser = StrOutputParser()

# 2. 用 LCEL 管道符串联成链（也是一个 Runnable）
chain = prompt | model | parser

# 3. 准备测试数据
single_input = {"topic": "程序员"}
batch_inputs = [
    {"topic": "程序员"},
    {"topic": "香蕉"},
    {"topic": "WiFi"}
]

# --- 方式一：invoke ---
print("=== invoke 结果 ===")
result_invoke = chain.invoke(single_input)
print(result_invoke)
print("\n")

# --- 方式二：batch (注意：它内部会并发调用 API) ---
print("=== batch 结果 ===")
results_batch = chain.batch(batch_inputs)
for i, res in enumerate(results_batch):
    print(f"第 {i+1} 个: {res}")
print("\n")

# --- 方式三：stream ---
print("=== stream 效果 ===")
for chunk in chain.stream(single_input):
    # chunk 是逐渐产生的一个个字符/词块
    print(chunk, end="", flush=True)
print("\n")
```

**运行效果说明：**
- `invoke` 会等待 API 返回完整字符串后再打印。
- `batch` 会一次性发出三个请求（几乎同时），然后按顺序返回结果，处理多个话题时速度极快。
- `stream` 会像 ChatGPT 网页端一样，一个字一个字蹦出来。

### 1.4 LCEL 的核心心智模型

通过上面的例子，你应该能感受到：**在 LangChain 中，一切都是数据管道。** 你的工作就是设计管道的走向（串联/并联/分支），而 `Runnable` 协议保证每个接头都能严丝合缝。

---

这个模块的内容清楚吗？关于 `Runnable` 的统一接口，或者 `invoke/batch/stream` 的用法，有任何疑问吗？如果没有，我们就进入下一个模块，学习如何让管道**分叉**——即并行处理数据。