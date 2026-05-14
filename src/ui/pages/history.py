import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ui.utils_web import _load_sessions, _delete_session_record
from src.core.agent import create_agent
from src.tools.tools import create_tools

st.set_page_config(page_title="历史对话", layout="wide")

# 隐藏侧边栏默认的英文多页面导航
st.markdown(
    """<style>[data-testid="stSidebarNav"] {display: none;}</style>""",
    unsafe_allow_html=True
)

# 权限检查
if not st.session_state.get("user_id"):
    st.switch_page("streamlit_app.py")

# 侧边栏
user_id = st.session_state.user_id
st.sidebar.markdown(f"**当前用户：** `{user_id}`")
st.sidebar.divider()

if st.sidebar.button("💬 聊天", width='stretch'):
    st.switch_page("pages/chat.py")
if st.sidebar.button("📚 知识库", width='stretch'):
    st.switch_page("pages/kb.py")
if st.sidebar.button("🧠 长期记忆", width='stretch'):
    st.switch_page("pages/memory.py")
if st.sidebar.button("📜 历史对话", width='stretch'):
    st.switch_page("pages/history.py")
if st.sidebar.button("🤖 多Agent协作", width='stretch'):
    st.switch_page("pages/multi_agent.py")

# 仅管理员可见
if user_id == "admin":
    if st.sidebar.button("🔧 管理员", width='stretch'):
        st.switch_page("pages/admin.py")

st.sidebar.divider()

if st.sidebar.button("🚪 退出登录", width='stretch'):
    for key in ["user_id", "session_id", "agent", "vectordb", "retriever"]:
        st.session_state[key] = None
    st.switch_page("streamlit_app.py")

st.title("📜 历史对话")

sessions = _load_sessions(user_id)

if not sessions:
    st.info("暂无历史对话")
else:
    # 表头
    header_cols = st.columns([3, 3, 2, 2])
    header_cols[0].markdown("**会话标题**")
    header_cols[1].markdown("**创建时间**")
    header_cols[2].markdown("**进入**")
    header_cols[3].markdown("**删除**")

    st.divider()

    for s in sessions:
        sid = s.get("session_id", "")
        title = s.get("title") or sid
        created = s.get("created", "")
        cols = st.columns([3, 3, 2, 2])
        cols[0].markdown(f"**{title}**")
        if s.get("title"):
            cols[0].caption(f"`{sid}`")
        cols[1].text(created)

        if cols[2].button("进入", key=f"enter_{sid}"):
            st.session_state.session_id = sid
            # 重新创建 agent，确保 checkpoint 连接正确
            tools = create_tools(user_id, st.session_state.retriever)
            st.session_state.agent = create_agent(tools)
            st.switch_page("pages/chat.py")

        if cols[3].button("删除", key=f"delete_{sid}"):
            _delete_session_record(user_id, sid)
            st.success(f"已删除会话 {sid}")
            st.rerun()
