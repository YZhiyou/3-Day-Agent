import streamlit as st
import uuid
import os
import sys

# 将项目根目录加入 sys.path，以便导入后端模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import AIMessageChunk
from agent import create_agent
from tools import create_tools
from utils_web import _record_session, _load_chat_history

st.set_page_config(page_title="聊天", layout="wide")

# 隐藏侧边栏默认的英文多页面导航
st.markdown(
    """<style>[data-testid="stSidebarNav"] {display: none;}</style>""",
    unsafe_allow_html=True
)

# 权限检查：未登录则返回首页
if not st.session_state.get("user_id"):
    st.switch_page("streamlit_app.py")

user_id = st.session_state.user_id

# ---------------- 侧边栏 ----------------
st.sidebar.markdown(f"**当前用户：** `{user_id}`")
st.sidebar.divider()

if st.sidebar.button("🆕 新对话", use_container_width=True):
    st.session_state.session_id = str(uuid.uuid4())[:8]
    _record_session(user_id, st.session_state.session_id)
    tools = create_tools(user_id, st.session_state.retriever, enable_web_search=True)
    st.session_state.agent = create_agent(tools)
    st.rerun()

if st.sidebar.button("📚 知识库", use_container_width=True):
    st.switch_page("pages/kb.py")

if st.sidebar.button("🧠 长期记忆", use_container_width=True):
    st.switch_page("pages/memory.py")

if st.sidebar.button("📜 历史对话", use_container_width=True):
    st.switch_page("pages/history.py")

# 仅管理员可见
if user_id == "admin":
    if st.sidebar.button("🔧 管理员", use_container_width=True):
        st.switch_page("pages/admin.py")

st.sidebar.divider()

if st.sidebar.button("🚪 退出登录", use_container_width=True):
    for key in ["user_id", "session_id", "agent", "vectordb", "retriever"]:
        st.session_state[key] = None
    st.switch_page("streamlit_app.py")

# ---------------- 主区域 ----------------
st.title("💬 聊天")
st.caption(f"当前会话：`{st.session_state.session_id}`")

# 加载并显示历史对话
history = _load_chat_history(st.session_state.agent, st.session_state.session_id)
for msg in history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 底部聊天输入
user_input = st.chat_input("请输入消息...")
if user_input:
    # 先显示用户刚发送的消息（流式输出期间可见）
    with st.chat_message("user"):
        st.markdown(user_input)

    # 流式输出助手回复
    def stream_response(input_text: str):
        for chunk, metadata in st.session_state.agent.stream(
            {"messages": [{"role": "user", "content": input_text}]},
            {"configurable": {"thread_id": st.session_state.session_id}},
            stream_mode="messages"
        ):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                yield chunk.content

    with st.chat_message("assistant"):
        response = st.write_stream(stream_response(user_input))

    # 流式完成后刷新页面，从历史加载完整对话
    st.rerun()
