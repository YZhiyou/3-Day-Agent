### 模块一：精准控制模型 —— `ChatModel` 的关键参数与行为

在第一周，你已经用 `ChatOpenAI` 创建过模型实例并成功调用。现在，我们来深入了解几个控制模型输出行为的关键参数。这些参数相当于模型的“调节旋钮”，理解它们能让你在不同场景下获得更符合预期的结果。

#### 1. 核心参数详解

我们以 OpenAI 的 GPT 系列模型为例，主要关注以下三个参数：

| 参数              | 含义                                                         | 取值范围  | 典型场景                                                     |
| :---------------- | :----------------------------------------------------------- | :-------- | :----------------------------------------------------------- |
| **`temperature`** | **控制随机性**。值越高，输出越随机、有创造性；值越低，输出越确定、保守。 | 0.0 ~ 2.0 | **低 (0.0~0.3)**：事实问答、代码生成、数据提取。<br>**高 (0.7~1.0)**：创意写作、头脑风暴。 |
| **`top_p`**       | **核采样 (Nucleus Sampling)**。模型会从概率累加达到 `top_p` 的候选词中随机选择。通常建议**与 `temperature` 二选一调整**，不要同时大幅修改。 | 0.0 ~ 1.0 | 作用类似 `temperature`，提供了另一种控制随机性的方式。设为 `0.1` 表示只考虑前 10% 概率质量的词。 |
| **`max_tokens`**  | **最大输出长度**。限制模型生成回复的最大 token 数（一个 token 约等于 0.75 个英文单词或 0.5 个汉字）。 | 正整数    | 控制回答长度、节省成本。如果设置过小，回答可能会被截断。     |

#### 2. 直观对比：`temperature` 参数实验

光看概念可能有些抽象。我们直接运行一段代码，对比 `temperature=0.0` 和 `temperature=1.0` 时的输出差异。

**请确保你的环境已安装 `langchain-openai` 并配置好 `OPENAI_API_KEY`。**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 准备一个需要一点创意的 Prompt
prompt = ChatPromptTemplate.from_template(
    "为一个能自动给植物浇水的智能花盆，想一个吸引人的产品口号。"
)

# 创建两个模型实例，仅 temperature 不同
model_creative = ChatOpenAI(model="gpt-3.5-turbo", temperature=1.0, max_tokens=50)
model_deterministic = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, max_tokens=50)

# 构建链
chain_creative = prompt | model_creative
chain_deterministic = prompt | model_deterministic

print("=== 高温度 (temperature=1.0) 的输出 ===")
# 多次运行可能会得到不同的结果
for i in range(3):
    response = chain_creative.invoke({})
    print(f"尝试 {i+1}: {response.content}")

print("\n=== 低温度 (temperature=0.0) 的输出 ===")
# 多次运行结果几乎相同
for i in range(3):
    response = chain_deterministic.invoke({})
    print(f"尝试 {i+1}: {response.content}")
```

**预期观察结果：**

- **高温度模型**：三次尝试可能分别给出 "生命之泉，智在掌握"、"忘记浇水？它说：我在。"、"润物细无声，智能伴一生。" 等不同风格的答案。
- **低温度模型**：三次尝试的输出内容**几乎完全一致**，通常是最符合统计规律、最“安全”的答案，比如 "智能花盆，让你的植物茁壮成长。"

#### 3. 参数使用建议与避坑点

- **`temperature=0` 不代表绝对确定性**：由于底层 GPU 浮点数运算的微小差异，`temperature=0` 的模型在极少数情况下输出仍可能有细微不同。但在逻辑和事实上，它是高度确定性的。
- **`top_p` 与 `temperature` 的选择**：对于大多数应用，调整 `temperature` 更直观易懂。`top_p` 是更精细的控制方式，如果想深入研究，可以参考 OpenAI 官方文档。初学阶段，掌握 `temperature` 就够了。
- **`max_tokens` 设置**：如果模型回答到一半戛然而止，大概率是 `max_tokens` 设置过小。可以在代码中根据预估的回答长度动态设置，或者先设置一个较大的值（如 1024），再根据实际情况调优。

#### 4. 模块小结

通过调整 `ChatModel` 的 `temperature` 参数，我们能够有效地在模型的**创造力**和**确定性**之间进行切换。这是构建可靠、可控 LangChain 应用的第一步。

