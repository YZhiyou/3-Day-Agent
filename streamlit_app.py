import streamlit as st
import uuid
import os
import sys

# 确保能导入根目录下的后端模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vector_store import load_vector_store
from retriever import build_retriever, build_rerank_retriever
from tools import create_tools
from agent import create_agent
from utils_web import _record_session


def init_session_state():
    """首次运行时初始化全局 session_state 键为 None。"""
    for key in ["user_id", "session_id", "agent", "vectordb", "retriever"]:
        if key not in st.session_state:
            st.session_state[key] = None


st.set_page_config(page_title="个人知识库助手", layout="wide")

# 隐藏侧边栏默认的英文多页面导航
st.markdown(
    """<style>[data-testid="stSidebarNav"] {display: none;}</style>""",
    unsafe_allow_html=True
)

init_session_state()

# 如果已登录，直接跳转到聊天页
if st.session_state.get("user_id"):
    st.switch_page("pages/chat.py")

# 居中登录卡片（三列布局，中间列放置登录卡片）
left_col, center_col, right_col = st.columns([1, 2, 1])
with center_col:
    st.markdown("<div style='margin-top: 80px;'></div>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>🔐 个人知识库助手</h1>", unsafe_allow_html=True)

    user_id_input = st.text_input(
        "用户名",
        placeholder="请输入用户名",
        label_visibility="collapsed"
    )

    if st.button("登录", use_container_width=True):
        user_id = user_id_input.strip()
        if not user_id:
            st.error("用户名不能为空")
        else:
            # 加载知识库
            try:
                vectordb = load_vector_store()
            except Exception as e:
                st.error(f"加载知识库失败: {e}")
                st.stop()

            # 构建检索器（带重排序）
            try:
                retriever = build_rerank_retriever(top_k=20, top_n=5)
            except Exception as e:
                st.error(f"构建检索器失败: {e}")
                st.stop()

            # 创建工具
            try:
                tools = create_tools(user_id, retriever, enable_web_search=True)
            except Exception as e:
                st.error(f"创建工具失败: {e}")
                st.stop()

            # 创建 Agent
            try:
                agent = create_agent(tools)
            except Exception as e:
                st.error(f"创建 Agent 失败: {e}")
                st.stop()

            # 生成会话 ID
            session_id = str(uuid.uuid4())[:8]

            # 存入全局状态
            st.session_state.user_id = user_id
            st.session_state.session_id = session_id
            st.session_state.vectordb = vectordb
            st.session_state.retriever = retriever
            st.session_state.agent = agent

            # 记录会话
            _record_session(user_id, session_id)

            # 跳转到聊天页
            st.switch_page("pages/chat.py")
