import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_manager import _load_user_data, _user_info_file
from utils_web import _load_sessions, _delete_session_record

st.set_page_config(page_title="管理员", layout="wide")

# 隐藏侧边栏默认的英文多页面导航
st.markdown(
    """<style>[data-testid="stSidebarNav"] {display: none;}</style>""",
    unsafe_allow_html=True
)

# 权限检查：仅 admin 可访问
if st.session_state.get("user_id") != "admin":
    st.warning("权限不足，仅管理员可访问此页面。")
    st.switch_page("pages/chat.py")

# 侧边栏
if st.sidebar.button("⬅️ 返回聊天", use_container_width=True):
    st.switch_page("pages/chat.py")

st.title("🔧 管理员控制台")

USER_INFO_DIR = "./data/user_info"

# 列出所有用户
users = []
if os.path.exists(USER_INFO_DIR):
    users = sorted([
        f.replace(".json", "")
        for f in os.listdir(USER_INFO_DIR)
        if f.endswith(".json")
    ])

if not users:
    st.info("暂无用户数据")
    st.stop()

selected_user = st.selectbox("选择用户", users)

st.subheader("长期记忆")
memory = _load_user_data(selected_user)
if memory:
    st.json(memory)
else:
    st.info("该用户暂无长期记忆")

st.subheader("会话列表")
sessions = _load_sessions(selected_user)
if sessions:
    st.dataframe(sessions, use_container_width=True)
else:
    st.info("该用户暂无会话")

st.divider()
st.subheader("危险操作")
confirm = st.checkbox(f"我确认要删除用户 `{selected_user}` 的所有数据（包括长期记忆和会话记录）")
if st.button("删除用户数据", key="btn_delete_user"):
    if confirm:
        # 删除长期记忆文件
        info_file = _user_info_file(selected_user)
        if os.path.exists(info_file):
            os.remove(info_file)

        # 删除会话列表及相关 checkpoints
        sessions_file = os.path.join("./data/user_sessions", f"{selected_user}_sessions.json")
        if os.path.exists(sessions_file):
            sess_list = _load_sessions(selected_user)
            for s in sess_list:
                _delete_session_record(selected_user, s.get("session_id", ""))
            os.remove(sessions_file)

        st.success(f"已删除用户 `{selected_user}` 的所有数据")
        st.rerun()
    else:
        st.error("请先勾选确认框")
