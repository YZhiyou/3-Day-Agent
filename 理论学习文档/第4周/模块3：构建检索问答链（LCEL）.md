## 🧩 模块3：构建检索问答链（LCEL）

这个模块的目标：  
**将 retriever（检索器）+ prompt（提示模板）+ model（DeepSeek） 串联成一个完整的 RAG 流水线，实现“问题 → 上下文 → 答案”的自动化流程。**

---

## 📌 一、核心组件回顾

在开始之前，确保你已经有了以下三个对象（来自模块2）：

1. **`retriever`**：从 Chroma 向量库中检索相关文档块的接口。
2. **`embeddings`**：`DashScopeEmbeddings(model="text-embedding-v4")` —— 用于将查询问题向量化（内部自动调用）。
3. **DeepSeek Chat Model**：用于生成最终答案的 LLM。

> 注意：DeepSeek API 是 OpenAI 兼容的，所以我们可以直接使用 LangChain 的 `ChatOpenAI` 类，只需要修改 `base_url` 和 `api_key`。

---

## 🔧 二、初始化 DeepSeek 模型

首先安装依赖（如果还没装）：
```bash
pip install langchain-openai
```

然后配置 DeepSeek：

```python
from langchain_openai import ChatOpenAI

# DeepSeek API 配置
deepseek_api_key = "你的-deepseek-api-key"   # 建议从环境变量读取
deepseek_base_url = "https://api.deepseek.com/v1"

llm = ChatOpenAI(
    model="deepseek-chat",           # 或者 deepseek-coder
    api_key=deepseek_api_key,
    base_url=deepseek_base_url,
    temperature=0.1,                 # 降低随机性，提高答案准确性
    max_tokens=1024
)
```

> 💡 **为什么用 ChatOpenAI 而不是专用类？**  
> DeepSeek 提供了与 OpenAI 完全兼容的 API 接口，因此 LangChain 的 `ChatOpenAI` 可以直接调用，只需修改 `base_url` 和 `model` 名称即可。

---

## 🧪 三、构建 RAG 链（LCEL 方式）

### 3.1 准备提示模板

我们需要一个能够接收 **问题（question）** 和 **检索到的上下文（context）** 的提示模板。

```python
from langchain_core.prompts import ChatPromptTemplate

template = """你是一个基于文档内容的问答助手。请根据以下上下文片段回答用户的问题。
如果上下文不足以回答问题，请如实说“根据现有文档无法回答”，不要编造信息。

<上下文>
{context}
</上下文>

用户问题：{question}

请给出准确、简洁的回答："""

prompt = ChatPromptTemplate.from_template(template)
```

### 3.2 实现 `context` 的自动填充（重点：`RunnablePassthrough`）

RAG 链的关键在于：如何将用户输入的 `question` 经过检索得到 `context`，然后传给 prompt。

使用 `RunnablePassthrough` 可以优雅地实现这个流程：

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# 定义一个函数：输入 question，返回检索到的文档拼接成的字符串
def retrieve_docs(question):
    docs = retriever.invoke(question)   # retriever 来自 vectordb.as_retriever()
    # 将每个文档的内容拼接，用 \n\n 分隔
    return "\n\n".join([doc.page_content for doc in docs])

# 构建 LCEL 链
rag_chain = (
    {
        "context": RunnableLambda(retrieve_docs),   # 将问题转为上下文
        "question": RunnablePassthrough()           # 原样传递问题
    }
    | prompt
    | llm
)
```

**解释：**
- `RunnablePassthrough()`：直接将用户输入（问题）原样传递给字典的 `question` 键。
- `RunnableLambda(retrieve_docs)`：将用户输入问题传给 `retrieve_docs` 函数，返回拼接后的文本，赋给 `context` 键。
- 然后 `{ "context": ..., "question": ... }` 这个字典被传入 `prompt`，格式化后发给 `llm`。

### 3.3 更简洁的写法（推荐）

LangChain 提供了一个 `RunnablePassthrough.assign()` 方法，可以更简洁：

```python
rag_chain = (
    RunnablePassthrough.assign(context=lambda x: retrieve_docs(x["question"]))
    | prompt
    | llm
)
```

调用时传入一个字典 `{"question": "你的问题"}`。

为了更直观，我选择使用第一种显式写法，方便调试。

---

## 🚀 四、完整可运行代码（模块3整合）

将模块2中生成的 `vectordb` 和 `retriever` 与上面的链结合，得到一个完整的 RAG 脚本：

```python
# module3_rag_chain.py
import os
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# ----------------------------
# 1. 配置 API Keys（建议用环境变量）
# ----------------------------
os.environ["DASHSCOPE_API_KEY"] = "你的-阿里云-key"
deepseek_api_key = "你的-deepseek-api-key"
deepseek_base_url = "https://api.deepseek.com/v1"

# ----------------------------
# 2. 加载已有的向量数据库
# ----------------------------
embeddings = DashScopeEmbeddings(model="text-embedding-v4")
persist_directory = "./chroma_db"
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})  # 检索3个最相关块

# ----------------------------
# 3. 初始化 DeepSeek 模型
# ----------------------------
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=deepseek_api_key,
    base_url=deepseek_base_url,
    temperature=0.1,
)

# ----------------------------
# 4. 定义提示模板
# ----------------------------
template = """你是一个基于文档内容的问答助手。请根据以下上下文片段回答用户的问题。
如果上下文不足以回答问题，请如实说“根据现有文档无法回答”，不要编造信息。

<上下文>
{context}
</上下文>

用户问题：{question}

请给出准确、简洁的回答："""

prompt = ChatPromptTemplate.from_template(template)

# ----------------------------
# 5. 检索函数
# ----------------------------
def retrieve_docs(question):
    docs = retriever.invoke(question)
    if not docs:
        return "没有找到相关文档片段。"
    return "\n\n".join([doc.page_content for doc in docs])

# ----------------------------
# 6. 构建 RAG 链
# ----------------------------
rag_chain = (
    {
        "context": RunnableLambda(retrieve_docs),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

# ----------------------------
# 7. 测试问答
# ----------------------------
if __name__ == "__main__":
    question = "这份PDF文档主要讲了什么内容？"
    response = rag_chain.invoke(question)
    print("回答:", response.content)
```

---

## 🧠 五、核心要点总结

| 组件                  | 作用                     | 代码位置                                   |
| --------------------- | ------------------------ | ------------------------------------------ |
| `retriever`           | 根据问题召回相关文档块   | `vectordb.as_retriever()`                  |
| `RunnablePassthrough` | 原样传递用户输入（问题） | `"question": RunnablePassthrough()`        |
| `RunnableLambda`      | 包装自定义检索函数       | `"context": RunnableLambda(retrieve_docs)` |
| `prompt`              | 合并上下文和问题         | `ChatPromptTemplate`                       |
| `llm`                 | 生成最终答案             | `ChatOpenAI` (DeepSeek)                    |

---

## ✅ 模块3 小结

你现在已经能够：
- 加载持久化的 Chroma 向量库并创建 retriever。
- 使用 OpenAI 兼容接口调用 DeepSeek 模型。
- 通过 LCEL 中的 `RunnablePassthrough` + `RunnableLambda` 构建完整的 RAG 链。
- 输入一个问题，获得基于文档内容的回答。

**这个模块的内容清楚吗？有任何疑问吗？如果没有，我们就进入最后一个模块（调试与优化）。**