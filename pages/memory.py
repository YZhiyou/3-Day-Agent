import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_manager import _load_user_data, _save_user_data

st.set_page_config(page_title="长期记忆", layout="wide")

# 隐藏侧边栏默认的英文多页面导航
st.markdown(
    """<style>[data-testid="stSidebarNav"] {display: none;}</style>""",
    unsafe_allow_html=True
)

# 权限检查
if not st.session_state.get("user_id"):
    st.switch_page("streamlit_app.py")

# 侧边栏
if st.sidebar.button("⬅️ 返回聊天", use_container_width=True):
    st.switch_page("pages/chat.py")

st.title("🧠 长期记忆")

user_id = st.session_state.user_id
data = _load_user_data(user_id)

# 将 dict 转为 list of dicts，便于 data_editor 编辑
records = [{"键": k, "值": v} for k, v in data.items()]

st.subheader("编辑记忆")
edited = st.data_editor(
    records,
    use_container_width=True,
    num_rows="dynamic",
    key="memory_editor"
)

if st.button("保存修改", key="btn_save_memory"):
    new_data = {r["键"]: r["值"] for r in edited if r.get("键")}
    _save_user_data(user_id, new_data)
    st.success("保存成功！")
    st.rerun()

st.divider()
st.subheader("新增键值对")
col1, col2 = st.columns(2)
with col1:
    new_key = st.text_input("键", key="new_key_input")
with col2:
    new_value = st.text_input("值", key="new_value_input")

if st.button("添加", key="btn_add_memory"):
    key_str = new_key.strip()
    if key_str:
        data[key_str] = new_value
        _save_user_data(user_id, data)
        st.success(f"已添加: {key_str} = {new_value}")
        st.rerun()
    else:
        st.error("键不能为空")
