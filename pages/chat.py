import streamlit as st
import uuid
import os
import sys

# 将项目根目录加入 sys.path，以便导入后端模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import AIMessage, AIMessageChunk
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

# ---------------- 执行计划卡片渲染 ----------------

def _render_plan_card(plan, step_results, current_step_index, placeholder):
    """在侧边栏渲染执行计划卡片：已完成标绿，当前步骤闪烁，失败标红。"""
    if not plan or not plan.steps:
        placeholder.empty()
        return

    placeholder.empty()
    with placeholder.container():
        st.markdown("### 📋 执行计划")
        for i, step in enumerate(plan.steps):
            step_num = i + 1
            if i < len(step_results):
                result = step_results[i]
                if result.success and not result.deviation:
                    st.success(f"✅ 步骤 {step_num}: `{step.action}`")
                else:
                    st.error(f"❌ 步骤 {step_num}: `{step.action}`")
            elif i == current_step_index:
                st.markdown(
                    f"""
                    <style>
                    @keyframes plan-blink-{step_num} {{
                        0%, 100% {{ opacity: 1; background-color: #e3f2fd; }}
                        50% {{ opacity: 0.5; background-color: #bbdefb; }}
                    }}
                    </style>
                    <div style="animation: plan-blink-{step_num} 1.2s infinite; padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; border-left: 4px solid #2196f3;">
                        <strong>⏳ 步骤 {step_num}: <code>{step.action}</code></strong><br/>
                        <small style="color: #555;">{step.input[:80]}{'...' if len(step.input) > 80 else ''}</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"⚪ 步骤 {step_num}: `{step.action}`")
            if i < len(plan.steps) - 1:
                st.markdown("<div style='margin: 2px 0;'></div>", unsafe_allow_html=True)
        st.divider()


# 侧边栏执行计划卡片占位
plan_card = st.sidebar.empty()

# 恢复上一次的执行计划卡片（rerun 后保留）
if st.session_state.get("execution_plan"):
    ep = st.session_state["execution_plan"]
    _render_plan_card(ep.get("plan"), ep.get("step_results", []), ep.get("current_step_index", 0), plan_card)

# 底部聊天输入
user_input = st.chat_input("请输入消息...")
if user_input:
    # 清空上一次的计划状态
    st.session_state["execution_plan"] = None
    plan_card.empty()

    # 先显示用户刚发送的消息
    with st.chat_message("user"):
        st.markdown(user_input)

    # loading 状态提示
    loading_placeholder = st.empty()

    # 助手回复区域
    response_buffer = ""
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # 当前计划状态（用于 updates 模式）
        current_plan = None
        current_step_results = []
        current_step_index = 0

        for chunk in st.session_state.agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            {"configurable": {"thread_id": st.session_state.session_id}},
            stream_mode=["messages", "updates"]
        ):
            mode, payload = chunk

            if mode == "messages":
                msg, metadata = payload
                if isinstance(msg, (AIMessage, AIMessageChunk)) and msg.content:
                    response_buffer += msg.content
                    message_placeholder.markdown(response_buffer)

            elif mode == "updates":
                for node_name, state_update in payload.items():
                    if node_name == "classify_complexity":
                        if state_update.get("is_complex"):
                            loading_placeholder.info("🤔 任务较复杂，正在制定执行计划...")
                        else:
                            loading_placeholder.info("💭 正在思考...")

                    elif node_name == "generate_plan":
                        current_plan = state_update.get("plan")
                        current_step_results = []
                        current_step_index = 0
                        loading_placeholder.info("🔧 开始执行计划...")
                        _render_plan_card(current_plan, current_step_results, current_step_index, plan_card)
                        st.session_state["execution_plan"] = {
                            "plan": current_plan,
                            "step_results": current_step_results,
                            "current_step_index": current_step_index,
                        }

                    elif node_name == "execute_step":
                        current_step_results = state_update.get("step_results", [])
                        current_step_index = state_update.get("current_step_index", 0)
                        loading_placeholder.info(f"⏳ 执行步骤 {current_step_index}...")
                        _render_plan_card(current_plan, current_step_results, current_step_index, plan_card)
                        st.session_state["execution_plan"] = {
                            "plan": current_plan,
                            "step_results": current_step_results,
                            "current_step_index": current_step_index,
                        }

                    elif node_name == "check_progress":
                        current_step_results = state_update.get("step_results", current_step_results)
                        if state_update.get("need_replan"):
                            loading_placeholder.warning("⚠️ 步骤偏离预期，正在重新规划...")
                        _render_plan_card(current_plan, current_step_results, current_step_index, plan_card)
                        st.session_state["execution_plan"] = {
                            "plan": current_plan,
                            "step_results": current_step_results,
                            "current_step_index": current_step_index,
                        }

                    elif node_name in ("react_agent", "summarize_results"):
                        loading_placeholder.info("📝 正在生成回答...")

        # 流式结束，清空 loading，保留最终计划状态
        loading_placeholder.empty()
        if current_plan:
            st.session_state["execution_plan"] = {
                "plan": current_plan,
                "step_results": current_step_results,
                "current_step_index": current_step_index,
            }

    # 流式完成后刷新页面，从历史加载完整对话
    st.rerun()
