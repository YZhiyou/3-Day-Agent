## 🧭 一、RAG 核心数据流（先记下这张图）

```
Load → Split → Embed → Store → Retrieve → Generate
```

- **Load**：读取原始文档（PDF、Word、网页等）
- **Split**：将长文档切成有语义的小块（chunk）
- **Embed**：将每个块转换成向量（需要 Embedding 模型）
- **Store**：将向量存入向量数据库（如 Chroma）
- **Retrieve**：根据用户问题，从数据库中召回最相关的几个块
- **Generate**：将问题 + 召回的块一起发给 LLM，生成最终答案

**本周我们会一步步实现这个流程。**  
今天先完成最前面两步中的 **Load** 部分。

---

## 📄 二、文档加载：`PyPDFLoader`

LangChain 提供了多种文档加载器。对于 PDF，我们使用 `PyPDFLoader`。

### 1. 安装依赖
在运行代码前，请确保安装：

```bash
pip install langchain langchain-community pypdf
```

> 说明：  
> - `langchain-community` 包含了社区版加载器  
> - `pypdf` 是 `PyPDFLoader` 背后的 PDF 解析库

### 2. 基础用法

假设你有一个本地 PDF 文件，路径为 `"./data/我的文档.pdf"`。

```python
from langchain_community.document_loaders import PyPDFLoader

# 加载 PDF
loader = PyPDFLoader("./data/我的文档.pdf")
documents = loader.load()

# 查看加载结果
print(f"共加载了 {len(documents)} 页")
print(documents[0].page_content[:200])   # 第一页的前200个字符
print(documents[0].metadata)             # 包含页码、源文件路径等信息
```

**关键点：**
- `loader.load()` 返回一个 `List[Document]`，每一页是一个 `Document` 对象。
- `Document` 包含两个属性：
  - `page_content`：该页的文本内容
  - `metadata`：字典，例如 `{"source": "...", "page": 0}`

### 3. 处理多页 PDF 的实践

`PyPDFLoader` 默认每页一个 `Document`，但这对于后续的分割来说并不理想（我们希望在模块2中把相关段落合并成块）。  
**目前你只需知道如何拿到所有页的内容即可**，模块2我们会用 `RecursiveCharacterTextSplitter` 重新切分。

---

## 🧪 三、完整可运行示例（模块1结束时可测）

请创建一个测试 PDF 文件（或者手头随便一个 PDF），然后运行以下脚本：

```python
# module1_load_pdf.py
from langchain_community.document_loaders import PyPDFLoader

# 替换成你的 PDF 路径
pdf_path = "./sample.pdf"

loader = PyPDFLoader(pdf_path)
pages = loader.load()

print(f"PDF 共 {len(pages)} 页")
print("\n--- 第一页内容预览 ---")
print(pages[0].page_content[:500])
print("\n--- 第一页元数据 ---")
print(pages[0].metadata)
```

**期望输出**：能看到 PDF 第一页的文本内容（可能包含少量乱码或格式问题，正常）。

---

## 🤔 关于 Embedding 模型的提前沟通

你提到 DeepSeek 不提供 Embedding 模型，这是对的。在 **模块2** 中，我会建议你使用以下两种方案之一：

1. **本地 Embedding 模型**（免费，无需 API）：  
   `sentence-transformers/all-MiniLM-L6-v2`（轻量，适合学习）
2. **云端 Embedding 模型**（需要 API key）：  
   OpenAI `text-embedding-3-small` 或 阿里云 `dashscope` 等

我会在模块2给出两种代码示例，让你根据自己情况选择。  
**现在只需先确认：你是否有 OpenAI API Key？或者更想用本地免费模型？**（这个问题留给你思考，模块2开始前我会明确给出两种写法的切换方式）

---

## ✅ 模块1 小结

你现在已经能够：
- 描述 RAG 的 6 个基本步骤
- 使用 `PyPDFLoader` 加载 PDF 文件
- 访问每一页的文本和元数据

**这个模块的内容清楚吗？有任何疑问吗？如果没有，我们就进入下一个模块（文档分割 & 向量存储）。**