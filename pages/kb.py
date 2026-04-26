import streamlit as st
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kb_manager import add_file_to_kb, delete_file_from_kb, rebuild_kb, search_kb
from vector_store import get_collection_stats
from retriever import build_retriever
from tools import create_tools
from agent import create_agent
from utils_web import _preprocess_latex

st.set_page_config(page_title="知识库", layout="wide")

# 隐藏侧边栏默认的英文多页面导航
st.markdown(
    """<style>[data-testid="stSidebarNav"] {display: none;}</style>""",
    unsafe_allow_html=True
)

# 权限检查
if not st.session_state.get("user_id"):
    st.switch_page("streamlit_app.py")

# 侧边栏
if st.sidebar.button("⬅️ 返回聊天", width='stretch'):
    st.switch_page("pages/chat.py")

st.title("📚 知识库管理")

vectordb = st.session_state.get("vectordb")
if not vectordb:
    st.error("知识库未初始化，请返回登录页重新登录。")
    st.stop()

tab_files, tab_search, tab_add, tab_rebuild = st.tabs(
    ["文件列表", "搜索文档", "新增文档", "重建知识库"]
)

# ---------------- Tab 1: 文件列表 ----------------
with tab_files:
    try:
        result = vectordb.get(include=["metadatas"])
        metadatas = result.get("metadatas", []) if result else []

        # 按来源去重，统计类型
        sources = {}
        for meta in metadatas:
            source = meta.get("source", "未知")
            file_type = meta.get("file_type", "未知")
            sources[source] = file_type

        if sources:
            data = [{"来源": s, "类型": t} for s, t in sorted(sources.items())]
            st.dataframe(data, width='stretch')

            st.divider()
            st.subheader("删除文件")
            file_to_delete = st.selectbox("选择要删除的文件", list(sources.keys()))
            if st.button("删除", key="btn_delete_file"):
                msg = delete_file_from_kb(vectordb, file_to_delete)
                if "已删除" in msg or "成功" in msg:
                    st.success(msg)
                else:
                    st.error(msg)
                st.rerun()
        else:
            st.info("知识库暂无文档")

        # 显示总块数
        stats = get_collection_stats(vectordb)
        st.metric("向量库总块数", stats.get("count", 0))
    except Exception as e:
        st.error(f"加载文件列表失败: {e}")

# ---------------- Tab 2: 搜索文档 ----------------
with tab_search:
    query = st.text_input("搜索关键词", placeholder="输入关键词搜索知识库...")
    if st.button("搜索", key="btn_search") and query:
        result = search_kb(vectordb, query)
        st.markdown(_preprocess_latex(result))

# ---------------- Tab 3: 新增文档 ----------------
with tab_add:
    uploaded = st.file_uploader(
        "上传文档 (PDF, TXT, MD)",
        type=["pdf", "txt", "md"]
    )
    if uploaded and st.button("添加到知识库", key="btn_add"):
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name
        try:
            msg = add_file_to_kb(vectordb, tmp_path)
            if "成功" in msg:
                st.success(msg)
            else:
                st.error(msg)
        finally:
            os.unlink(tmp_path)

# ---------------- Tab 4: 重建知识库 ----------------
with tab_rebuild:
    st.warning("重建知识库将清空现有向量数据，并重新索引指定目录下的所有文档。")
    docs_dir = st.text_input("文档目录路径", value="./documents")
    confirm = st.checkbox("我确认要重建知识库")
    if st.button("重建", key="btn_rebuild"):
        if confirm:
            progress_bar = st.progress(0.0, text="准备重建...")
            try:
                with st.status("正在重建知识库...", expanded=True) as status:
                    # 1. 彻底释放旧向量库的文件句柄
                    old_vectordb = st.session_state.pop("vectordb", None)
                    st.session_state.pop("retriever", None)
                    st.session_state.pop("agent", None)

                    if old_vectordb:
                        try:
                            if hasattr(old_vectordb, "_client") and hasattr(old_vectordb._client, "close"):
                                old_vectordb._client.close()
                            if hasattr(old_vectordb, "_collection"):
                                old_vectordb._collection = None
                        except Exception:
                            pass
                        del old_vectordb

                    try:
                        del vectordb
                    except NameError:
                        pass

                    import gc
                    gc.collect()
                    gc.collect()
                    import time
                    time.sleep(2.0)

                    # 进度跟踪器：将各阶段进度映射到总体进度条
                    class _ProgressTracker:
                        def __init__(self, bar_placeholder):
                            self.bar = bar_placeholder
                            self.weights = {
                                "scan":  (0.00, 0.05),
                                "load":  (0.05, 0.30),
                                "clear": (0.30, 0.40),
                                "split": (0.40, 0.50),
                                "create":(0.50, 0.55),
                                "embed": (0.55, 1.00),
                            }

                        def __call__(self, stage, current, total, message):
                            start, end = self.weights.get(stage, (0.0, 1.0))
                            ratio = current / total if total > 0 else 1.0
                            overall = start + (end - start) * ratio
                            self.bar.progress(overall, text=message)

                    tracker = _ProgressTracker(progress_bar)
                    new_vectordb = rebuild_kb(
                        "./data/chroma", docs_dir, progress_callback=tracker
                    )
                    new_retriever = build_retriever()
                    new_tools = create_tools(st.session_state.user_id, new_retriever)
                    new_agent = create_agent(new_tools)
                    st.session_state.vectordb = new_vectordb
                    st.session_state.retriever = new_retriever
                    st.session_state.agent = new_agent

                    progress_bar.empty()
                    status.update(label="知识库重建成功！", state="complete", expanded=False)
            except Exception as e:
                progress_bar.empty()
                st.error(f"重建失败: {e}")
        else:
            st.error("请先勾选确认框以确认重建操作。")
