## 1. 项目实现：1 天全栈开发流程

本项目从架构设计到功能完整运行，全部在 **Day 3** 内完成。高效的核心并非手速，而是 **AI 工具链的精准分工**：把"业务逻辑设计"留给人类，把"代码实现与 Debug"交给 AI IDE。

### 1.1 三步工作流

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   主控窗口       │────▶│    AI IDE       │────▶│   开发者（我）   │
│ (DeepSeek 对话) │     │  (代码助手)      │     │  (审校与集成)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**第一步：主控窗口负责架构设计**

开发者首先在对话窗口中完成系统级思考：模块怎么划分？数据如何流动？每个模块的输入输出是什么？随后，将思考结果转化为**结构化的编码 prompt**——不是模糊的"帮我写个知识库"，而是精确的"创建一个 vector_store.py，职责是文档分块与 Chroma 持久化，需要以下函数：create_vector_store、load_vector_store、add_documents……"

**第二步：AI IDE 全权负责编码与 Debug**

AI IDE 接收结构化 prompt 后，独立完成具体编码、依赖安装、运行测试和错误修复。所有编译错误、接口变动、依赖冲突都在 IDE 内部闭环解决，**绝不把报错信息贴回对话窗口污染上下文**。这正是"AI 驱动开发"的关键：让工具处理细节，让人脑保持对架构的清晰掌控。

**第三步：开发者审校、集成与微调**

开发者审查 AI IDE 产出的代码，确认其是否符合预设的数据流；将各模块集成为完整系统；最后调整前端交互（如 Streamlit 页面跳转、按钮布局）。整个过程中，开发者始终聚焦于**数据流向和业务逻辑**，而非某一行代码的语法细节。

### 1.2 方法论验证：框架会变，逻辑不变

Day 3 的开发过程验证了一个核心洞察：**LangChain 的 import 路径可能每月一变，但 RAG 的数据流（加载→分块→向量化→存储→检索→增强→生成）和 Agent 的 ReAct 循环（观察→思考→行动）是稳定的业务逻辑。**

当 AI IDE 发现 `langchain` 某接口已废弃时，它会自动查找替代方案并完成迁移；而开发者只需要确保"文档依然能被正确分块和检索"、"Agent 依然能根据用户问题选择合适工具"。这种分工，让一天内交付全栈应用成为可能。

---

## 2. 快速开始

### 2.1 环境要求

- Python >= 3.13
- 依赖管理工具：[uv](https://github.com/astral-sh/uv)

### 2.2 核心依赖

主要依赖已在 `pyproject.toml` 中声明，包括：

| 依赖 | 用途 |
|------|------|
| `langchain>=1.2.15` | LangChain 核心框架 |
| `langchain-classic` | 兼容包（ParentDocumentRetriever / EnsembleRetriever） |
| `langchain-chroma>=1.1.0` | Chroma 向量存储集成 |
| `langchain-deepseek>=1.0.1` | DeepSeek LLM 接入 |
| `langgraph>=1.1.9` | ReAct Agent 图编排 |
| `langgraph-checkpoint-sqlite>=2.0.0` | 会话状态 SQLite 持久化 |
| `chromadb>=1.5.8` | 本地向量数据库 |
| `dashscope>=1.25.17` | 灵积平台 Embedding / Rerank 服务 |
| `rank-bm25>=0.2.2` | BM25 关键词检索（混合检索关键词通道） |
| `pypdf>=6.10.2` | PDF 文档解析 |
| `streamlit` | WebUI 框架 |
| `duckduckgo-search>=8.1.1` | 联网搜索 |

### 2.3 安装与启动

```bash
# 1. 克隆项目
git clone <repository-url>
cd 3-Day-Agent

# 2. 安装依赖
uv sync

# 3. 配置环境变量
# 复制 .env 模板并填入你的 API Key
cp .env.example .env
```

`.env` 文件模板：

```ini
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DASHSCOPE_API_KEY=your-dashscope-api-key
```

```bash
# 4. 将需要检索的文档放入 ./documents 目录
# 支持格式：PDF、TXT、Markdown

# 5. 启动应用

# 方式一：WebUI（推荐）
python start.py
# 或直接使用 streamlit
streamlit run streamlit_app.py

# 方式二：CLI
python cli.py
```

首次启动时，系统会自动尝试加载已持久化的向量库。若向量库不存在，可通过 CLI 的 `/kb rebuild` 命令或 WebUI 的"重建知识库"功能，从 `./documents` 目录构建初始索引。

---

## 3. 使用说明

### 3.1 CLI 使用方式

启动 CLI 后，先使用 `/login <用户名>` 登录，随后即可进行多轮对话。

**常用命令：**

| 命令 | 说明 |
|------|------|
| `/login <用户名>` | 登录或切换用户，自动创建新会话 |
| `/new` | 开启一个新会话 |
| `/switch <会话ID>` | 切换到已有会话 |
| `/history` | 查看当前会话的历史消息 |
| `/userinfo` | 查看当前用户的长期记忆 |
| `/kb add <文件路径>` | 添加文件到知识库 |
| `/kb delete <文件路径>` | 从知识库删除文件 |
| `/kb rebuild` | 重建知识库（会确认） |
| `/kb search <关键词>` | 在知识库中搜索 |
| `/help` | 显示帮助 |
| `/quit` | 退出 |

CLI 支持**流式输出**：Agent 思考过程和最终回复会实时逐字显示。

### 3.2 WebUI 使用方式（Streamlit）

WebUI 采用 Streamlit 多页面架构，入口为 `streamlit_app.py`。

**使用流程：**

1. **登录页**：输入用户名，系统自动加载向量库、构建检索器、创建 Agent，并生成唯一会话 ID。
2. **聊天页**：主对话界面，支持流式回复。当任务较复杂时，侧边栏会自动显示 **📋 执行计划** 卡片，实时展示计划步骤、执行进度和状态（已完成标绿、执行中蓝色闪烁、失败标红）。侧边栏可快速切换：
   - 新对话：创建全新会话
   - 知识库：进入知识库管理页
   - 长期记忆：查看和编辑用户记忆
   - 历史对话：浏览并恢复过往会话
3. **知识库页**：文件列表查看、文档搜索、上传新文档、重建知识库。
4. **长期记忆页**：以键值对形式查看和编辑 Agent 记住的个人信息。
5. **历史对话页**：会话列表，支持点击进入或删除。

### 3.3 交互示例

**示例 1：询问知识库内容**

```
用户：请解释一下 PSR 的详细证明过程
助手：（调用 retrieve_documents）
    [1] 来源: ./documents/PSR详细理论证明.md
    假设策略参数为 θ，平滑后的策略为 ...
    
根据知识库中的《PSR详细理论证明》，PSR 的核心思路是...
```

**示例 2：保存和查询个人信息**

```
用户：我叫 Alice，是一名强化学习研究员
助手：已保存信息: 姓名 = Alice

用户：我的研究方向是什么？
助手：（调用 get_user_info）
    你的研究方向是强化学习。
```

**示例 3：联网搜索实时信息**

```
用户：今天上海天气怎么样？
助手：（调用 web_search）
    根据搜索结果（来源: weather.example.com），
    上海今天多云，气温 22-28°C...
```

**示例 4：复杂任务 Plan-Act 分解（WebUI 侧边栏显示执行计划）**

```
用户：先搜索知识库中关于 LangChain 的内容，然后保存我的专业领域为 AI

[侧边栏显示 📋 执行计划]
⏳ 步骤 1: retrieve_documents （执行中...）
⚪ 步骤 2: save_user_info

助手：（执行步骤 1）
    [1] 来源: ./documents/PSR详细理论证明.md
    ...

[侧边栏更新]
✅ 步骤 1: retrieve_documents
⏳ 步骤 2: save_user_info （执行中...）

助手：（执行步骤 2）
    已保存信息: 专业领域 = AI

[侧边栏更新]
✅ 步骤 1: retrieve_documents
✅ 步骤 2: save_user_info

助手：根据您的请求，我已成功执行了以下操作：
    1. 搜索知识库...
    2. 保存您的专业领域为 AI...
```

---

## 4. 模块设计与业务逻辑

本章按数据流自下而上讲解各模块。阅读时建议抓住两个问题：**这个模块解决了什么业务问题？数据如何流入和流出？**

---

### 4.1 `vector_store.py` —— 文档分块、向量化与 Chroma 持久化

**业务职责**：将原始文档转换为可语义检索的向量表示，并持久化到本地 Chroma 数据库。

**核心业务逻辑**：

```
原始文档 (List[Document])
    │
    ▼
┌──────────────────────────────────────────┐
│  _smart_split_documents()                │
│  ├── Markdown (.md)                      │
│  │   └── MarkdownHeaderTextSplitter      │
│  │       （按 # / ## / ### 标题层级分割） │
│  │       └── 仍过长？                     │
│  │           └── 原子单元保护性二次分割   │
│  │               （代码块/公式/表格不切断）│
│  │                                        │
│  └── PDF / TXT                           │
│      └── RecursiveCharacterTextSplitter  │
│          （中文友好分隔符）                │
└──────────────────────────────────────────┘
    │
    ▼
文档块 (List[Document])
    │
    ▼
┌──────────────────────────────────────────┐
│  分批嵌入（batch_size=10）                │
│  ├── 每批 _clean_text() 过滤 surrogate   │
│  └── 失败则指数退避重试（2s → 4s）        │
└──────────────────────────────────────────┘
    │
    ▼
Chroma.add_documents() ──▶ 持久化到 ./data/chroma
```

**关键设计决策**：

- **文件类型感知路由**：`_smart_split_documents()` 根据 `metadata["file_type"]` 自动选择分块策略。PDF / TXT 沿用原有的 `RecursiveCharacterTextSplitter`；Markdown 论文走专用 pipeline。
- **Markdown 双层分块（标题感知 + 原子保护）**：
  - **一级**：`MarkdownHeaderTextSplitter` 按 `# / ## / ### / ####` 分割，每个 chunk 的 metadata 自动携带 `Header 1/2/3/4` 路径，RAG 检索时模型能感知章节归属。
  - **二级**：对仍超过 `chunk_size` 的章节，启用**原子单元保护性分割**。文本被拆解为"普通文本"和"保护块"两类原子单元：代码块（`` ```...``` ``）、块级公式（`$$...$$`）、行内公式（`$...$`）、Markdown 表格被视为不可分割单元，分块时优先在普通文本处切断，尽量避免破坏上述结构。
- **中文友好的通用分块**：非 Markdown 文档沿用 `RecursiveCharacterTextSplitter`，分隔符序列按 `\n\n` → `\n` → `。` → `！` → `？` → `；` → ` ` → `""` 优先级递归切分，中文段落优先在句末断点分割。
- **chunk_size=1000, chunk_overlap=200**：在信息密度和检索粒度之间取平衡。1000 字符足以容纳一个完整的论证段落；200 字符重叠确保跨块边界的关键信息不会丢失。
- **DashScope `text-embedding-v4`**：选用灵积平台的 Embedding 模型，对中文语义理解效果稳定，且 API 调用成本可控。
- **分批嵌入与指数退避重试**：`create_vector_store` 不再一次性提交全部文档，而是按 `batch_size=10` 分批调用 `vectordb.add_documents()`。每批失败时自动重试最多 3 次，等待时间按指数退避（2s → 4s），有效应对 `IncompleteRead` / `SSLEOFError` / `Connection broken` 等网络抖动。
- **Unicode surrogate 字符过滤**：PDF 提取的数学符号或编码转换可能产生 `0xD800-0xDFFF` 范围的非法 surrogate 字符，导致 DashScope API 触发 `UnicodeEncodeError`。`_clean_text()` 在每批嵌入前自动过滤这些字符，根治此类崩溃。
- **进度回调支持**：`create_vector_store` 支持传入 `progress_callback(stage, current, total, message)`，外部 UI（如 Streamlit）可分阶段（split / create / embed）实时展示大批量论文入库进度。
- **增量添加与删除**：`add_documents` 和 `delete_documents` 支持在已有向量库上增删，无需每次都全量重建。
- **Parent-Child 分块策略（默认）**：`create_parent_child_vector_store` 采用 Parent-Child 双层分块：
  - **父块**（chunk_size=1000, chunk_overlap=200）：由 `_smart_split_documents()` 按文件类型路由生成，保留完整段落/章节语义，存入 `JsonDocstore` 持久化。
  - **子块**（chunk_size=300, chunk_overlap=50）：由 `ParentDocumentRetriever` 自动将每个父块细分为更小的子块，子块经 `DashScopeEmbedding` 嵌入后存入 Chroma 向量库。
  - **检索时返回父块**：向量检索命中的是子块，但 `ParentDocumentRetriever` 会根据子块 metadata 中的 `doc_id` 从 `JsonDocstore` 取出完整父块返回，既保证嵌入精度（小块语义更聚焦），又保证上下文完整性（大块信息不截断）。
- **JsonDocstore 持久化**：官方 `InMemoryStore` 在进程重启后父文档会丢失。自定义 `JsonDocstore(BaseStore[str, Document])` 将父文档序列化为 JSON 文件（`./data/chroma/docstore.json`），每次 `mset/mdelete` 后自动写盘，实现 Parent-Child 模式下父文档的跨会话持久化。
- **向后兼容**：保留 `create_vector_store` / `load_vector_store` / `add_documents` / `delete_documents` 等旧版接口。旧向量库（无 `docstore.json`）仍可通过旧接口正常使用。

**对外接口**：

| 函数 | 作用 |
|------|------|
| `create_vector_store(docs, persist_dir, chunk_size, chunk_overlap, batch_size, progress_callback)` | 从原始文档创建并持久化新的向量库（旧版普通分块） |
| `create_parent_child_vector_store(docs, persist_dir, chunk_size, chunk_overlap, child_chunk_size, child_chunk_overlap, batch_size, progress_callback)` | 使用 Parent-Child 模式创建向量库；父块存 JsonDocstore，子块存 Chroma |
| `load_vector_store(persist_dir)` | 加载已持久化的普通向量库 |
| `load_parent_child_vector_store(persist_dir)` | 加载 Parent-Child 向量库，返回 (Chroma, JsonDocstore) |
| `add_documents(vectordb, docs)` | 向现有普通向量库增量添加文档 |
| `add_documents_parent_child(vectordb, docstore, docs)` | 向 Parent-Child 向量库增量添加文档 |
| `delete_documents(vectordb, ids/filter)` | 按 ID 或元数据过滤条件删除普通向量库文档 |
| `delete_documents_parent_child(vectordb, docstore, ids/filter)` | 删除 Parent-Child 向量库文档（联动清理子块和父块） |
| `is_parent_child_mode(persist_dir)` | 检查指定目录是否为 Parent-Child 模式 |
| `get_collection_stats(vectordb)` | 获取集合名称、文档数量等统计信息 |
| `split_markdown_documents(docs, chunk_size, chunk_overlap)` | Markdown 专用分块：标题感知 + 原子单元保护 |
| `split_documents(docs, chunk_size, chunk_overlap)` | 通用中文友好分块（适用于 PDF / TXT） |

---

### 4.2 `retriever.py` / `reranker.py` —— 检索策略封装与混合检索

**业务职责**：从向量库中召回相关文档，并封装多阶段检索 pipeline（语义检索 → 关键词精排 → RRF 融合 → Rerank 精排），以标准 `BaseRetriever` 形式供上层链式调用。

**核心业务逻辑（混合检索 pipeline）**：

```
用户查询字符串
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1: 语义粗筛（Chroma 向量检索）                    │
│  vectordb.as_retriever(search_kwargs={"k": 20})          │
│  ──▶ 召回语义相关的 Top-20 候选文档                      │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2: 关键词精排（BM25）                             │
│  BM25Retriever.from_documents(candidates,                │
│                  preprocess_func=_chinese_tokenizer)     │
│  ──▶ 在候选集上做字面匹配，召回 Top-5                    │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3: RRF 倒数排名融合                               │
│  score = w_semantic * 1/(60+rank_semantic+1)             │
│        + w_bm25     * 1/(60+rank_bm25+1)                 │
│  ──▶ 合并双通道结果，重新排序                            │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 4: Rerank 语义精排（DashScope qwen3-vl-rerank）   │
│  ContextualCompressionRetriever                          │
│  ──▶ 对融合结果二次打分，返回最相关 Top-5                │
└─────────────────────────────────────────────────────────┘
```

> 若 `rank-bm25` 未安装或 BM25 初始化失败，Stage 2/3 自动跳过，降级为纯语义检索 + Rerank。

**关键设计决策**：

- **四档检索器按需切换**：
  - `build_retriever()`：基础语义检索（向后兼容）。
  - `build_rerank_retriever()`：语义检索 + DashScope Rerank 精排。
  - `build_hybrid_rerank_retriever()`：完整混合检索 pipeline（语义 + BM25 + RRF + Rerank），旧版默认入口。
  - `build_parent_child_hybrid_rerank_retriever()`：**当前默认入口**。在混合检索基础上叠加 Parent-Child 分块：子块负责向量嵌入与检索，返回完整父块上下文，再经 BM25 + RRF + Rerank 精排。
- **粗筛架构避免全库 BM25 索引**：BM25 只在语义召回的 Top-20 候选上动态构建索引，既保证关键词匹配的精确性，又避免大知识库上全量 BM25 的内存与性能开销。
- **自定义中文分词器 `_chinese_tokenizer`**：`rank-bm25` 默认按空格分词，对中文完全失效。`_chinese_tokenizer` 将连续中文字符逐字切分、英文/数字保留为整体，使 BM25 能正确识别 "SAM-PPO" 等中英混合术语。
- **RRF 权重可配置**：默认 `[0.5, 0.5]`，语义与关键词通道平衡。若某类查询更依赖字面匹配，可调高关键词权重。
- **阈值过滤的 score 转换**：Chroma 返回的是 L2 距离（越小越相似），代码中通过 `1 / (1 + distance)` 转换为近似的相似度分数（0~1），便于设置直观阈值。

**对外接口**：

| 函数/类 | 作用 |
|---------|------|
| `build_retriever(persist_dir, search_type, search_kwargs)` | 基础语义检索工厂函数（向后兼容） |
| `build_rerank_retriever(persist_dir, top_k, top_n, model)` | 语义检索 + DashScope Rerank 精排 |
| `build_hybrid_rerank_retriever(persist_dir, semantic_k, bm25_k, top_n, model, weights)` | 混合检索完整 pipeline（旧版默认入口） |
| `build_parent_child_hybrid_rerank_retriever(persist_dir, semantic_k, bm25_k, top_n, model, weights)` | Parent-Child 混合检索 + Rerank（当前默认入口） |
| `HybridRetriever` | 语义+BM25+RRF 混合检索器（BaseRetriever 子类） |
| `RetrievalEngine.similarity_search(query, k, filter)` | 相似度搜索 |
| `RetrievalEngine.mmr_search(query, k, fetch_k, lambda_mult)` | MMR 多样性搜索 |
| `RetrievalEngine.similarity_search_with_threshold(query, score_threshold)` | 带阈值过滤的搜索 |
| `RetrievalEngine.get_retriever()` | 从引擎实例获取标准 `BaseRetriever` |

---

### 4.3 `kb_manager.py` —— 知识库的增删重建管理

**业务职责**：以**文件**为粒度管理知识库内容，支持单文件增删、全库重建和知识库内搜索。

**核心业务逻辑**：

```
文件路径
    │
    ├──▶ _load_single_file ──▶ PyPDFLoader / TextLoader ──▶ List[Document]
    │                                                                │
    │                                                                ▼
    │                                                    add_documents(vectordb, docs)
    │                                                                │
    └──▶ delete_file_from_kb ──▶ 按 source 元数据过滤 ──▶ vectordb.delete(ids)

全量重建：
docs_directory ──▶ load_documents() ──▶ create_parent_child_vector_store() ──▶ 新的 Parent-Child 向量库 + JsonDocstore
```

**关键设计决策**：

- **文件级元数据追踪**：每个文档块都携带 `source`（原始文件路径）和 `file_type`（pdf/txt/markdown）元数据。这让"删除某个文件的所有内容"成为可能——只需按 `source` 过滤后批量删除对应 IDs。
- **重建时的 Windows 兼容处理**：`rebuild_kb` 在删除旧向量库目录前执行 `gc.collect()`，并带 10 次重试机制，避免 Windows 下文件句柄未释放导致的 `PermissionError`。
- **加载与向量化的解耦**：`kb_manager` 只负责"文件→Document"的加载和"对 vectordb 的操作"，实际的向量化与持久化仍然委托给 `vector_store.py`，保持职责清晰。
- **Parent-Child 模式自动检测**：`add_file_to_kb` 和 `delete_file_from_kb` 通过 `is_parent_child_mode()` 自动检测当前向量库是否为 Parent-Child 模式，自动选择 `add_documents_parent_child` / `delete_documents_parent_child` 或旧版接口，无需调用方手动判断。

**对外接口**：

| 函数 | 作用 |
|------|------|
| `add_file_to_kb(vectordb, file_path, persist_dir)` | 将单个文件加载并加入向量库；自动检测 Parent-Child 模式 |
| `delete_file_from_kb(vectordb, file_path, persist_dir)` | 按 source 删除某文件的所有块；自动检测 Parent-Child 模式 |
| `rebuild_kb(persist_dir, docs_directory)` | 清空并重建整个知识库（采用 Parent-Child 模式） |
| `search_kb(vectordb, query, k)` | 在知识库中搜索并返回格式化结果 |

---

### 4.4 `tools.py` —— Agent 可调用的工具定义

**业务职责**：为 Agent 封装三个核心能力——**知识库检索**、**长期记忆读写**、**联网实时搜索**——并以 LangChain `BaseTool` 列表形式暴露。

**核心业务逻辑**：

```
create_tools(user_id, retriever, enable_web_search)
    │
    ├──▶ retrieve_documents(query) ──▶ retriever.invoke(query) ──▶ 格式化文档片段+来源
    │
    ├──▶ save_user_info(key, value) ──▶ memory_manager.save_user_info(user_id, key, value)
    │
    ├──▶ get_user_info(key) ──▶ memory_manager.get_user_info(user_id, key)
    │
    └──▶ web_search(query) ──▶ DuckDuckGoSearchResults ──▶ 联网搜索结果（可选）
```

**关键设计决策**：

- **工具即函数的简洁设计**：使用 `@tool` 装饰器将普通函数直接转为 Agent 可调用的 `BaseTool`，无需继承复杂类。工具 docstring 即为 Agent 的"使用说明书"——Agent 根据 docstring 描述自主决定何时调用哪个工具。
- **user_id 在工具创建时绑定**：`create_tools` 是工厂函数，在创建时就把 `user_id` 注入到 `save_user_info` 和 `get_user_info` 的闭包中。这保证多用户场景下，Agent 始终操作当前登录用户的记忆，而不会串号。
- **联网搜索为可选项**：通过 `enable_web_search` 参数控制，方便在无网络环境或不需要实时信息的场景下关闭。
- **检索结果截断与格式化**：`retrieve_documents` 将每段内容截断至 300 字符并标注来源，防止超长文档块撑爆 LLM 上下文窗口，同时让来源可追溯。
- **检索工具描述反映真实 pipeline**：`retrieve_documents` 的 docstring 明确告知 Agent"检索结果经过混合检索（语义理解 + 精准关键词匹配）与语义重排序"，让 Agent 在回答时能更自信地引用排名靠前的结果，尤其适用于缩写和特定术语的精确查找。

**对外接口**：

| 函数 | 作用 |
|------|------|
| `create_tools(user_id, retriever, enable_web_search)` | 工厂函数，返回 Agent 可用的工具列表 |

---

### 4.5 `memory_manager.py` —— 短期会话记忆 + 长期用户记忆

**业务职责**：管理两层记忆：
- **短期记忆**：当前会话的多轮对话历史，支持跨重启的连续对话。
- **长期记忆**：每个用户的持久化键值对信息（姓名、偏好等）。

**核心业务逻辑**：

```
短期记忆（会话级）：
session_id ──▶ FileChatMessageHistory(f"./data/chat_history/{session_id}.json")
    │                    │
    │                    └── 由 LangGraph SqliteSaver 实际主导持久化
    │                        （FileChatMessageHistory 保留以兼容历史接口）
    │
长期记忆（用户级）：
user_id ──▶ ./data/user_info/{user_id}.json
    │
    ├──▶ save_user_info(user_id, key, value) ──▶ 读取 → 更新 → 写回 JSON
    │
    └──▶ get_user_info(user_id, key) ──▶ 读取 JSON → 返回 value
```

**关键设计决策**：

- **短期记忆由 LangGraph SqliteSaver 接管**：代码中保留了 `get_session_history` 返回 `FileChatMessageHistory` 的接口以兼容旧模式，但实际 Agent 运行时的会话状态持久化由 `agent.py` 中的 `SqliteSaver`（写入 `./data/checkpoints.db`）负责。SQLite 方案比 JSON 文件更适合高频读写和事务性操作。
- **滚动式摘要+窗口记忆模式**：为控制 LLM 上下文长度，在 `agent.py` 中引入 `manage_memory` 节点实现记忆压缩：
  - **触发条件**：对话总轮数达到 `MEMORY_WINDOW_LIMIT=20` 时触发首次摘要，之后每新增 `MEMORY_TRIGGER_BATCH=15` 轮未摘要历史再次触发。
  - **摘要范围**：每次将"保留窗口之外"的所有历史（`summary_covered_rounds` 之后到最近 5 轮之前）生成摘要，与旧摘要合并更新。
  - **Prompt 注入**：`_build_dialogue_history()` 根据 `summary_covered_rounds` 精确截断，向 LLM 输出 `[此前对话摘要]\n{summary}\n\n[最近 5 轮对话]\n...` 格式。完整原始消息仍保留在 `checkpoints.db` 中，仅 LLM 可见层做逻辑截断。
  - **状态追踪**：`AgentState` 新增 `conversation_summary`（摘要文本）和 `summary_covered_rounds`（已覆盖轮数），`SqliteSaver` 自动持久化，跨会话恢复后摘要状态不丢失。
- **长期记忆使用简单 JSON 文件**：用户个人信息通常是低频读写、结构简单的键值对，JSON 足够轻量且便于人工查看和备份。
- **用户隔离**：每个用户拥有独立的 JSON 文件，天然隔离，无需复杂权限控制。

**对外接口**：

| 函数 | 作用 |
|------|------|
| `get_session_history(session_id)` | 获取指定会话的历史管理器 |
| `save_user_info(user_id, key, value)` | 保存用户的长期记忆 |
| `get_user_info(user_id, key)` | 查询用户的长期记忆 |

---

### 4.6 `agent.py` —— Plan-Act 智能体调度层

**业务职责**：将所有组件（LLM、工具、系统提示词、记忆持久化）装配成一个**Plan-Act 模式**的智能体。原 ReAct Agent 作为子图嵌入，复杂任务自动分解执行，简单任务直接走原 ReAct 流程。

**核心业务逻辑**：

```
系统提示词（角色定义 + 行为准则 + 联网规则）
    │
    ▼
_build_react_agent(tools, system_prompt, checkpointer) ──▶ 原 ReAct 子图
    │
    ▼
StateGraph(AgentState) ──▶ Plan-Act 外层状态图
    │
    ├──▶ manage_memory ──▶ 检查对话长度，超阈值则生成/更新滚动摘要
    │       └──▶ classify_complexity ──▶ LLM 判断复杂度
    │               ├──▶ 简单 ──▶ react_agent（原 ReAct 子图）──▶ END
    │               └──▶ 复杂 ──▶ generate_plan ──▶ execute_step
    │                                           │
    │                                           ▼
    │                                      check_progress
    │                                           │
    │                            need_replan ──▶ generate_plan（重规划，最多 3 次）
    │                            未完成    ──▶ execute_step
    │                            全部完成  ──▶ summarize_results ──▶ END
    │
    ▼
CompiledStateGraph（配置 recursion_limit + SqliteSaver）
```

**关键设计决策**：

- **Plan-Act 双层架构**：外层 StateGraph 负责"判断复杂度 → 生成计划 → 执行步骤 → 检查进度 → 汇总结果"的固定流程；内层 `react_agent` 子图保留原 ReAct 循环能力，处理简单查询和单步工具调用。两层共享同一个 `SqliteSaver`，状态自动持久化到 `./data/checkpoints.db`。
- **复杂度自动判断**：`classify_complexity` 节点通过 LLM + `ComplexityJudgment`（Pydantic 结构化输出）判断用户请求是否需要多步计划。判断标准：≥2 个工具调用、多步依赖、涉及多个子任务 → 复杂；简单闲聊、单次检索、仅读写用户信息 → 简单。
- **计划生成与执行**：`generate_plan` 输出 `Plan` 对象（含 `steps` 列表和 `overall_goal`），每步指定 `action`（工具名）、`input`（参数，多参数工具用 JSON）、`expected`（期望输出）。`execute_step` 通过 `tool_map` 动态选择工具并调用，自动尝试 JSON 解析以支持 `save_user_info` 等多参数工具。
- **动态进度检查与重规划**：`check_progress` 使用 LLM + `DeviationCheck` 判断上一步结果是否严重偏离预期。若失败或偏离 → `need_replan=True`，路由回 `generate_plan` 重新制定计划。`replan_count` 限制最多 3 次，防止无限循环。
- **Pydantic 状态模型**：
  - `PlanStep`：action / input / expected
  - `Plan`：steps / overall_goal
  - `StepResult`：step_index / success / output / deviation
  - `ComplexityJudgment`：is_complex / reason
  - `DeviationCheck`：deviation / reason
- **对话历史全量注入**：`_classify_complexity`、`_generate_plan`、`_summarize_results` 三个节点调用 LLM 时，均通过 `_build_dialogue_history()` 注入完整对话历史（含滚动摘要+窗口原始消息），确保 Agent 在多轮对话中能正确理解代词指代和省略上下文。
- **子图流式传播**：`chat.py` 调用 `agent.stream(..., subgraphs=True)`，使外层 Plan-Act 图能捕获内层 `react_agent` 子图的 `AIMessageChunk` 流式事件。简单模式下 ReAct Agent 的回答也能逐字流式显示，而非一次性返回完整消息。
- **状态定义 `AgentState`**：在原有 `messages`（`Annotated[list, add_messages]`）基础上扩展 `plan`、`current_step_index`、`step_results`、`need_replan`、`is_complex`、`final_summary`、`replan_count`、`conversation_summary`、`summary_covered_rounds`。

**对外接口**：

| 函数 | 作用 |
|------|------|
| `create_agent(tools, get_session_history, llm_model, temperature, max_iterations)` | 创建并返回配置好的 Plan-Act CompiledStateGraph |

**调用方式**：

```python
agent.invoke(
    {"messages": [{"role": "user", "content": "..."}]},
    {"configurable": {"thread_id": "session_id"}}
)
```

---

### 4.7 `streamlit_app.py` —— WebUI 入口与模块装配

**业务职责**：Streamlit 多页面应用的入口页面，负责用户登录、全局状态初始化、以及各后端模块的**装配**。

**核心业务逻辑**：

```
用户输入用户名 ──▶ 登录按钮
    │
    ├──▶ load_vector_store() ──▶ 加载 Chroma 向量库
    │
    ├──▶ build_parent_child_hybrid_rerank_retriever() ──▶ 构建 Parent-Child 混合检索器
    │       （子块语义粗筛 → 返回父块 → BM25 关键词精排 → RRF 融合 → DashScope Rerank）
    │
    ├──▶ create_tools(user_id, retriever) ──▶ 创建工具集
    │
    ├──▶ create_agent(tools) ──▶ 创建 Agent
    │
    ├──▶ 生成 session_id ──▶ 记录会话
    │
    └──▶ st.switch_page("pages/chat.py")
```

**关键设计决策**：

- **登录即装配**：所有核心对象（vectordb、retriever、tools、agent）在登录时一次性创建并注入 `st.session_state`。其中检索器默认使用 `build_parent_child_hybrid_rerank_retriever()`（子块语义粗筛 → 返回父块 → BM25 关键词精排 + RRF 融合 + DashScope Rerank 精排）。后续页面直接从 session_state 读取，避免重复初始化。
- **错误拦截与友好提示**：向量库加载失败、检索器构建失败、工具创建失败、Agent 创建失败——每一步都有 `try/except` 捕获，并通过 `st.error()` 向用户展示中文错误信息，而不是抛出堆栈。
- **隐藏默认侧边栏导航**：通过 CSS 隐藏 Streamlit 原生的多页面导航，改用自定义侧边栏按钮控制页面跳转，保持 UI 风格统一。
- **多页面状态共享**：`st.session_state` 充当跨页面的全局状态容器，user_id、session_id、agent 实例等在各页面间无缝传递。

**对外接口**：无直接对外函数，作为 Streamlit 应用入口运行。

---

### 4.8 `pages/chat.py` —— 聊天页面与执行计划可视化

**业务职责**：Streamlit 聊天主页面，负责对话渲染、用户输入处理、Agent 流式输出，以及**复杂任务时的执行计划实时可视化**。

**核心业务逻辑**：

```
用户输入消息
    │
    ├──▶ st.chat_message("user") 显示用户消息
    │
    ├──▶ agent.stream(..., stream_mode=["messages", "updates"])
    │       │
    │       ├──▶ messages 模式 ──▶ AIMessageChunk 逐字累加 ──▶ 实时更新助手回复
    │       │
    │       └──▶ updates 模式 ──▶ 节点状态变更
    │               ├──▶ classify_complexity (is_complex=True)
    │               │       └── 侧边栏渲染 📋 执行计划卡片
    │               ├──▶ generate_plan ──▶ 显示计划步骤列表
    │               ├──▶ execute_step ──▶ 更新当前步骤高亮
    │               ├──▶ check_progress ──▶ 标记步骤成功/失败
    │               └──▶ summarize_results ──▶ 汇总生成最终回答
    │
    └──▶ st.rerun() 从历史加载完整对话
```

**关键设计决策**：

- **双模式流式输出 `stream_mode=["messages", "updates"]`**：`messages` 模式保留 AI 回复的逐字流式效果（`AIMessageChunk` 累加）；`updates` 模式捕获每个 LangGraph 节点的状态变更，用于驱动执行计划卡片的实时更新。两种模式并行，互不干扰。
- **执行计划卡片 `_render_plan_card`**：在侧边栏动态渲染计划步骤，使用 Streamlit 原生组件区分状态：
  - 已完成且成功 → `st.success()`（绿色 ✅）
  - 已完成但失败/偏离 → `st.error()`（红色 ❌）
  - 当前执行中 → `st.markdown(unsafe_allow_html=True)` + CSS `@keyframes` 闪烁动画（蓝色 ⏳）
  - 未开始 → 普通 markdown（灰色 ⚪）
- **Loading 状态提示**：根据 Agent 所处阶段（制定计划、执行步骤、重规划、生成回答），在主聊天区上方显示动态 `st.info()` / `st.warning()` 提示，让用户感知当前处理进度。
- **计划状态持久化**：通过 `st.session_state["execution_plan"]` 保存最终计划结果，页面刷新（rerun）后执行计划卡片仍然保留在侧边栏，不会消失。发送新消息时自动清空上一次计划。

**对外接口**：无直接对外函数，作为 Streamlit 页面运行。

---

## 5. 架构图

下图展示了系统的五层架构与各模块之间的数据依赖关系：

```mermaid
graph TB
    subgraph UI["用户界面层"]
        CLI["cli.py<br/>命令行交互"]
        WEB["streamlit_app.py + pages/<br/>Streamlit WebUI"]
        CHAT["pages/chat.py<br/>聊天页 + 执行计划卡片"]
    end

    subgraph AGENT["智能体调度层"]
        AGENT_CORE["agent.py<br/>Plan-Act 外层 StateGraph"]
        CLASSIFY["classify_complexity<br/>复杂度判断"]
        REACT["react_agent<br/>原 ReAct 子图"]
        GEN_PLAN["generate_plan<br/>生成执行计划"]
        EXEC_STEP["execute_step<br/>执行步骤"]
        CHECK_PROG["check_progress<br/>检查进度 + replan"]
        SUMMARIZE["summarize_results<br/>汇总结果"]
        SYS_PROMPT["系统提示词<br/>决策优先级 + 行为准则"]
        CHECKPOINTER["SqliteSaver<br/>checkpoints.db"]
    end

    subgraph TOOLS["工具层"]
        RETRIEVE_DOC["retrieve_documents<br/>Parent-Child 混合检索 + Rerank"]
        USER_INFO["save/get_user_info<br/>长期记忆读写"]
        WEB_SEARCH["web_search<br/>联网搜索"]
    end

    subgraph SERVICE["服务层"]
        RETRIEVER["retriever.py<br/>检索策略封装"]
        RERANKER["reranker.py<br/>DashScope Rerank 精排"]
        HYBRID["HybridRetriever<br/>语义+BM25+RRF 混合检索"]
        PARENT_CHILD["ParentDocumentRetriever<br/>子块检索 → 返回父块"]
        KB_MGR["kb_manager.py<br/>知识库增删重建"]
    end

    subgraph DATA["数据持久层"]
        VECTOR["vector_store.py<br/>Chroma 向量库 + DashScope Embedding"]
        DOCSTORE["JsonDocstore<br/>docstore.json<br/>父文档持久化"]
        DOC_LOADER["document_loader.py<br/>PDF/TXT/MD 加载"]
        MEM_SHORT["SQLite checkpoints<br/>短期会话记忆"]
        MEM_LONG["JSON user_info<br/>长期用户记忆"]
    end

    CLI --> AGENT_CORE
    WEB --> CHAT
    CHAT --> AGENT_CORE
    AGENT_CORE --> CLASSIFY
    CLASSIFY --> REACT
    CLASSIFY --> GEN_PLAN
    GEN_PLAN --> EXEC_STEP
    EXEC_STEP --> CHECK_PROG
    CHECK_PROG --> GEN_PLAN
    CHECK_PROG --> EXEC_STEP
    CHECK_PROG --> SUMMARIZE
    REACT --> SYS_PROMPT
    REACT --> CHECKPOINTER
    AGENT_CORE --> SYS_PROMPT
    AGENT_CORE --> CHECKPOINTER
    AGENT_CORE --> RETRIEVE_DOC
    AGENT_CORE --> USER_INFO
    AGENT_CORE --> WEB_SEARCH

    RETRIEVE_DOC --> PARENT_CHILD
    PARENT_CHILD --> HYBRID
    HYBRID --> RETRIEVER
    HYBRID --> RERANKER
    USER_INFO --> MEM_LONG
    WEB_SEARCH --> DUCK["DuckDuckGo"]

    RETRIEVER --> VECTOR
    RERANKER --> DASHSCOPE_RE["DashScope<br/>qwen3-vl-rerank"]
    PARENT_CHILD --> DOCSTORE
    KB_MGR --> VECTOR
    KB_MGR --> DOCSTORE
    KB_MGR --> DOC_LOADER

    VECTOR --> CHROMA[("Chroma<br/>./data/chroma")]
    DOC_LOADER --> DOCS["./documents"]
    MEM_SHORT --> CHECKPOINTER
    MEM_LONG --> USER_FILES["./data/user_info/*.json"]
```

**数据流说明**：

1. **用户提问**（UI 层）→ `chat.py` 接收消息，通过 `stream_mode=["messages", "updates"]` 同时获取流式回复和节点状态更新。
2. **复杂度判断**（智能体调度层）→ `classify_complexity` 判断任务简单或复杂。简单任务直接走 `react_agent` 子图；复杂任务进入 Plan-Act 流程。
3. **计划生成与执行** → `generate_plan` 输出多步计划，`execute_step` 逐条调用工具，`check_progress` 检查执行结果。若偏离预期则自动重规划（最多 3 次），全部完成后由 `summarize_results` 汇总生成最终回答。
4. **执行计划可视化** → `chat.py` 根据 `updates` 模式实时捕获节点状态，在侧边栏渲染执行计划卡片：已完成标绿、失败标红、当前步骤蓝色闪烁。
5. **工具调用**（工具层）→ Agent 按需调用检索、记忆或搜索工具，获取外部信息。
6. **检索与存储**（服务层 + 数据持久层）→ Parent-Child 混合检索 pipeline：Chroma 子块语义粗筛召回 Top-20 → ParentDocumentRetriever 返回完整父块 → BM25 在父块候选集上做关键词精排 → RRF 融合双通道结果 → DashScope Rerank 最终精排返回 Top-5。父文档由 JsonDocstore 持久化到 `./data/chroma/docstore.json`。记忆读写操作对应 SQLite/JSON 文件。
7. **状态持久化** → 整个对话状态（包括 Plan、StepResult 等新增字段）由 SqliteSaver 写入 SQLite，实现跨会话恢复。

---


