# -*- coding: utf-8 -*-
"""多 Agent 协作仪表盘页面 — 论文评审委员会

提供多智能体协作的可视化交互界面：
- 左侧 60%：全局状态条 + 对话历史 + 用户输入 + AI 实时输出
- 右侧 40%：Agent 实时卡片 x3 + 任务队列 + 全局进度条 + 消息流

执行流程：
    用户输入 → 构造 MultiAgentState → graph.stream(stream_mode="updates")
    → 逐步更新仪表盘 → 提取最终 AIMessage 显示在对话区
"""

import os
import sys

# 将项目根目录加入 sys.path，以便导入后端模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk

from src.multi_agent.multi_agent_state import (
    create_initial_multi_agent_state,
    AgentCapability,
)
from src.multi_agent.multi_agent_graph import build_multi_agent_graph
from src.ui.utils_web import _preprocess_latex


# ============================================================
# 工具函数
# ============================================================

def _deep_merge(base: dict, update: dict) -> dict:
    """深合并：对嵌套 dict 递归合并，避免浅合并丢失嵌套数据"""
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _get_agent_from_namespace(namespace: tuple) -> str:
    """从 subgraphs=True 的 namespace 元组中解析 Agent 来源

    namespace 格式示例：
    - () 表示顶层图
    - ("researcher:abc123",) 表示 researcher 子图
    - ("researcher:abc123", "inner:xyz") 表示嵌套子图

    Returns:
        Agent 名称字符串，如 "researcher"、"analyst"、"writer"
    """
    if not namespace:
        return ""
    # 取最外层子图标识，如 "researcher:abc123" → "researcher"
    first_seg = namespace[0]
    return first_seg.split(":")[0] if ":" in first_seg else first_seg


def render_agent_plan(agent_name: str, plan_data: dict, current_step_index: int, step_results: list):
    """在 Agent 卡片内展示 Plan-Act 执行计划

    Args:
        agent_name: Agent 名称，如 "researcher"
        plan_data: 计划数据，可能为 dict 或带 .steps 属性的对象
        current_step_index: 当前执行步骤索引（0-based）
        step_results: 步骤执行结果列表
    """
    if not plan_data:
        return

    # 提取 steps 列表
    steps = []
    if isinstance(plan_data, dict):
        steps = plan_data.get("steps", [])
    elif hasattr(plan_data, "steps"):
        steps = plan_data.steps

    if not steps:
        return

    st.markdown("<div style='margin-top:6px'>📋 <b>执行计划</b></div>", unsafe_allow_html=True)
    for i, step in enumerate(steps):
        # 提取 action 和 expected
        if isinstance(step, dict):
            action = step.get("action", "")
            expected = step.get("expected", "")
        elif hasattr(step, "action"):
            action = step.action
            expected = getattr(step, "expected", "")
        else:
            action = str(step)
            expected = ""

        if i < current_step_index:
            st.markdown(
                f"<div style='font-size:12px;padding:2px 8px;'>"
                f"<span style='color:green'>✅ 步骤{i+1}: {action}</span></div>",
                unsafe_allow_html=True,
            )
        elif i == current_step_index:
            hint = f" - {expected[:20]}" if expected else ""
            st.markdown(
                f"<div style='font-size:12px;padding:2px 8px;'>"
                f"<span style='color:#1565c0'>🔵 步骤{i+1}: {action}{hint}</span></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='font-size:12px;padding:2px 8px;'>"
                f"<span style='color:gray'>⚪ 步骤{i+1}: {action}</span></div>",
                unsafe_allow_html=True,
            )


# ============================================================
# 页面配置
# ============================================================

st.set_page_config(layout="wide", page_title="多 Agent 协作评审系统")

# 隐藏侧边栏默认的英文多页面导航
st.markdown(
    """<style>[data-testid="stSidebarNav"] {display: none;}</style>""",
    unsafe_allow_html=True,
)

# ============================================================
# 登录检查与懒初始化
# ============================================================

# 未登录则返回首页
if not st.session_state.get("user_id"):
    st.switch_page("streamlit_app.py")

user_id = st.session_state.user_id

# 懒初始化：首次进入页面时构建多 Agent 协作图
if "multi_agent_graph" not in st.session_state or st.session_state.multi_agent_graph is None:
    from src.tools.tools import create_tools
    tools = create_tools(user_id, st.session_state.retriever, enable_web_search=True)
    st.session_state.multi_agent_graph = build_multi_agent_graph(
        plan_act_graph=st.session_state.agent,
        tools=tools,
    )

# 初始化多 Agent 对话历史
if "multi_agent_messages" not in st.session_state:
    st.session_state.multi_agent_messages = []

graph = st.session_state.multi_agent_graph

# ============================================================
# 侧边栏导航
# ============================================================

st.sidebar.markdown(f"**当前用户：** `{user_id}`")
st.sidebar.divider()

if st.sidebar.button("💬 新对话", width="stretch"):
    st.switch_page("pages/chat.py")
if st.sidebar.button("📚 知识库", width="stretch"):
    st.switch_page("pages/kb.py")
if st.sidebar.button("🧠 长期记忆", width="stretch"):
    st.switch_page("pages/memory.py")
if st.sidebar.button("📜 历史对话", width="stretch"):
    st.switch_page("pages/history.py")
if st.sidebar.button("🤖 多Agent协作", width="stretch"):
    st.switch_page("pages/multi_agent.py")

# 仅管理员可见
if user_id == "admin":
    if st.sidebar.button("🔧 管理员", width="stretch"):
        st.switch_page("pages/admin.py")

st.sidebar.divider()

# 清空会话按钮
if st.sidebar.button("🗑️ 清空会话", width="stretch"):
    st.session_state.multi_agent_messages = []
    st.rerun()

if st.sidebar.button("🚪 退出登录", width="stretch"):
    for key in ["user_id", "session_id", "agent", "vectordb", "retriever",
                "multi_agent_graph", "multi_agent_messages"]:
        st.session_state[key] = None
    st.switch_page("streamlit_app.py")


# ============================================================
# 仪表盘渲染组件
# ============================================================

def render_global_status(state):
    """渲染全局状态条：显示当前活跃 Agent 描述"""
    progress = state.get("agent_progress", {})
    busy_agents = []
    for name, info in progress.items():
        if isinstance(info, dict) and info.get("status") == "BUSY":
            step = info.get("current_step", "执行中")
            busy_agents.append(f"{name}: {step}")
    if busy_agents:
        status_text = " | ".join(busy_agents)
    else:
        status_text = "所有 Agent 空闲，等待任务..."
    st.markdown(f"""<div style='background:linear-gradient(90deg,#e8f5e9,#f3e5f5);
        padding:12px 20px;border-radius:8px;margin-bottom:16px;
        font-size:14px;color:#333'>
        <b>系统状态：</b>{status_text}</div>""", unsafe_allow_html=True)


def render_agent_cards(state, agent_plans=None):
    """渲染 Agent 实时卡片：遍历 agent_registry，结合 agent_progress

    Args:
        state: 当前全局状态字典
        agent_plans: 各 Agent 的计划数据，格式 {agent_name: {plan, current_step_index, step_results}}
    """
    if agent_plans is None:
        agent_plans = {}
    registry = state.get("agent_registry", [])
    progress = state.get("agent_progress", {})
    private = state.get("agent_private", {})

    for agent in registry:
        name = agent.agent_name if hasattr(agent, 'agent_name') else agent.get("agent_name", "")
        info = progress.get(name, {})
        if not isinstance(info, dict):
            info = {}
        status = info.get("status", "IDLE")
        current_step = info.get("current_step", "空闲")
        detail = info.get("detail", "")
        prog = info.get("progress", 0)
        last_log = info.get("last_log", "")
        inbox_count = len(private.get(name, {}).get("inbox", []))

        # 状态灯颜色
        light = "🟢" if status == "BUSY" else "⚪"

        with st.container():
            bg_color = "#f0fff0" if status == "BUSY" else "#fafafa"
            detail_html = f"<br><span style='color:#888;font-size:12px'>{detail}</span>" if detail else ""
            log_html = f"<br><span style='color:#999;font-size:11px'>日志: {last_log}</span>" if last_log else ""
            st.markdown(f"""<div style='border:1px solid #ddd;border-radius:8px;
                padding:12px;margin-bottom:8px;
                background:{bg_color}'>
                <b>{light} {name}</b>
                <span style='float:right;font-size:12px;color:#666'>
                    收件箱: {inbox_count}</span>
                <br><span style='color:#555;font-size:13px'>{current_step}</span>
                {detail_html}
                {log_html}
                </div>""", unsafe_allow_html=True)
            if status == "BUSY" and prog > 0:
                st.progress(prog / 100)

            # 在 Agent 卡片内展示 Plan-Act 执行计划
            plan_info = agent_plans.get(name)
            if plan_info and plan_info.get("plan"):
                render_agent_plan(
                    name,
                    plan_info["plan"],
                    plan_info.get("current_step_index", 0),
                    plan_info.get("step_results", []),
                )


def render_task_queue(state):
    """渲染任务队列 + 全局进度条"""
    tasks = state.get("task_queue", [])
    if not tasks:
        st.caption("暂无任务")
        return

    completed = 0
    for t in tasks:
        s = t.status if hasattr(t, "status") else t.get("status", "?")
        if s == "COMPLETED":
            completed += 1
    total = len(tasks)
    st.progress(completed / total if total > 0 else 0)
    st.caption(f"完成 {completed}/{total}")

    for task in tasks:
        status = task.status if hasattr(task, "status") else task.get("status", "?")
        desc = task.description if hasattr(task, "description") else task.get("description", "?")
        if status == "COMPLETED":
            icon = "✅"
            color = "#e8f5e9"
        elif status == "IN_PROGRESS":
            icon = "🔄"
            color = "#fff8e1"
        else:
            icon = "⏳"
            color = "#f5f5f5"
        st.markdown(f"""<div style='background:{color};padding:6px 10px;
            border-radius:4px;margin:4px 0;font-size:13px'>
            {icon} {desc}</div>""", unsafe_allow_html=True)


def render_message_log(state):
    """渲染消息流：从 processing + archived 取最近 5 条"""
    bus = state.get("message_bus")
    if not bus:
        st.caption("暂无消息")
        return

    # 从 processing + archived 取最近 5 条
    messages = []
    if hasattr(bus, 'processing'):
        messages.extend(bus.processing[-3:])
    if hasattr(bus, 'archived'):
        messages.extend(bus.archived[-5:])
    messages = messages[-5:]

    if not messages:
        st.caption("暂无消息记录")
        return

    for msg in reversed(messages):
        msg_type = msg.type.value if hasattr(msg.type, "value") else str(getattr(msg, "type", "?"))
        sender = getattr(msg, "sender", "?")
        payload = getattr(msg, "payload", {})
        if isinstance(payload, dict):
            desc = str(payload.get("description", payload.get("task_id", "")))[:40]
        else:
            desc = str(payload)[:40]
        st.markdown(f"""<div style='font-size:12px;padding:4px 8px;
            border-left:3px solid #2196F3;margin:4px 0;color:#444'>
            <b>[{msg_type}]</b> {sender} → {desc}</div>""", unsafe_allow_html=True)


def render_dashboard(state, agent_plans=None):
    """渲染完整右侧仪表盘"""
    st.markdown("### Agent 状态")
    render_agent_cards(state, agent_plans)
    st.markdown("---")
    st.markdown("### 任务队列")
    render_task_queue(state)
    st.markdown("---")
    st.markdown("### 消息流")
    render_message_log(state)


# ============================================================
# 主区域：左右两列布局
# ============================================================

st.title("论文评审委员会")

left_col, right_col = st.columns([3, 2])

# ---- 左侧（60%）：全局状态条 + 对话历史 + 输入框 ----
with left_col:
    # 全局状态条占位（执行期间实时更新）
    global_status_placeholder = st.empty()
    with global_status_placeholder.container():
        render_global_status({})

    # 渲染历史对话
    for msg in st.session_state.multi_agent_messages:
        with st.chat_message(msg["role"]):
            st.markdown(_preprocess_latex(msg["content"]))

    # 用户输入框
    user_input = st.chat_input("输入你的请求（如：请评审这篇论文）")

# ---- 右侧（40%）：静态占位（非执行期间显示） ----
with right_col:
    st.subheader("实时状态")
    st.caption("输入请求后，此处将实时显示各 Agent 的协作状态")

# ============================================================
# 核心执行流程：用户输入后触发
# ============================================================

# Agent 名称到图标/标签的映射
AGENT_LABELS = {
    "researcher": "🔍 Researcher",
    "analyst": "📊 Analyst",
    "writer": "✍️ Writer",
    "supervisor": "🎯 Supervisor",
}

if user_input:
    # 记录用户消息到对话历史
    st.session_state.multi_agent_messages.append({"role": "user", "content": user_input})

    # 显示用户刚发送的消息
    with left_col:
        with st.chat_message("user"):
            st.markdown(user_input)

    # 构造初始 MultiAgentState
    initial_state = create_initial_multi_agent_state(
        messages=[HumanMessage(content=user_input)],
        agent_registry=[
            AgentCapability(
                agent_name="researcher",
                skills=["retrieve", "web_search"],
                status="IDLE",
                max_concurrent=3,
            ),
            AgentCapability(
                agent_name="analyst",
                skills=["reason", "summarize"],
                status="IDLE",
                max_concurrent=2,
            ),
            AgentCapability(
                agent_name="writer",
                skills=["format_output"],
                status="IDLE",
                max_concurrent=1,
            ),
        ],
    )

    # LangGraph 执行配置
    run_id = uuid.uuid4().hex[:8]
    config = {
        "configurable": {
            "thread_id": f"multi_{st.session_state.get('session_id', 'default')}_{run_id}"
        },
        "recursion_limit": 100,
    }

    # 右侧仪表盘占位符（执行期间不断重绘）
    dashboard_placeholder = right_col.empty()
    # 左侧进度指示器
    status_text = left_col.empty()

    # 追踪最新状态（逐步合并 node 更新）
    latest_state = dict(initial_state)
    full_response = ""
    # 流式文本缓冲：按 Agent 分别缓存
    streaming_buffers = {}  # {agent_name: accumulated_text}
    # 各 Agent 的 Plan-Act 计划数据
    agent_plans = {}  # {agent_name: {plan, current_step_index, step_results}}

    # 助手回复区域
    with left_col:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            try:
                for chunk in graph.stream(
                    initial_state, config=config,
                    stream_mode=["messages", "updates"],
                    subgraphs=True,
                ):
                    # subgraphs=True 时格式为 (path, mode, payload)
                    # 参考 chat.py 第 189-194 行的解析逻辑
                    if len(chunk) == 3:
                        namespace, mode, payload = chunk
                    else:
                        mode, payload = chunk
                        namespace = ()

                    # ---- 处理 messages 类型事件：流式文本增量输出 ----
                    if mode == "messages":
                        msg, metadata = payload
                        if isinstance(msg, (AIMessage, AIMessageChunk)) and msg.content:
                            # 判断来自哪个 Agent
                            agent_source = _get_agent_from_namespace(namespace)
                            # 若无法从 namespace 解析，尝试从 metadata 获取
                            if not agent_source and isinstance(metadata, dict):
                                agent_source = metadata.get("langgraph_node", "")

                            # 缓存每个 Agent 的流式输出
                            if agent_source not in streaming_buffers:
                                streaming_buffers[agent_source] = ""
                            streaming_buffers[agent_source] += msg.content

                            # 拼接所有 Agent 的流式输出用于显示
                            display_parts = []
                            for a_name, a_text in streaming_buffers.items():
                                if a_text.strip():
                                    label = AGENT_LABELS.get(a_name, f"🤖 {a_name}") if a_name else "🤖"
                                    display_parts.append(f"**{label}:**\n\n{a_text}")
                            full_response = "\n\n---\n\n".join(display_parts)
                            message_placeholder.markdown(
                                _preprocess_latex(full_response) + " ▌"
                            )

                    # ---- 处理 updates 类型事件：节点状态更新 ----
                    elif mode == "updates":
                        for node_name, node_update in payload.items():
                            if isinstance(node_update, dict):
                                _deep_merge(latest_state, node_update)

                                # ---- 捕获子图的 Plan-Act 计划数据 ----
                                # 当适配节点完成时，子图结果已合入 node_update
                                # 尝试从 node_update 中提取 plan 相关字段
                                if "plan" in node_update and node_update["plan"]:
                                    # 尝试从 namespace 判断来源 Agent
                                    plan_agent = _get_agent_from_namespace(namespace)
                                    if not plan_agent:
                                        # 回退：从 agent_progress 判断哪个 Agent 在 BUSY
                                        prog = latest_state.get("agent_progress", {})
                                        for a_name, a_info in prog.items():
                                            if isinstance(a_info, dict) and a_info.get("status") == "BUSY":
                                                plan_agent = a_name
                                                break
                                    if plan_agent:
                                        existing = agent_plans.get(plan_agent, {})
                                        agent_plans[plan_agent] = {
                                            "plan": node_update["plan"],
                                            "current_step_index": node_update.get("current_step_index", existing.get("current_step_index", 0)),
                                            "step_results": node_update.get("step_results", existing.get("step_results", [])),
                                        }

                                if "current_step_index" in node_update:
                                    step_agent = _get_agent_from_namespace(namespace)
                                    if not step_agent:
                                        prog = latest_state.get("agent_progress", {})
                                        for a_name, a_info in prog.items():
                                            if isinstance(a_info, dict) and a_info.get("status") == "BUSY":
                                                step_agent = a_name
                                                break
                                    if step_agent and step_agent in agent_plans:
                                        agent_plans[step_agent]["current_step_index"] = node_update["current_step_index"]
                                    if step_agent and "step_results" in node_update and step_agent in agent_plans:
                                        agent_plans[step_agent]["step_results"] = node_update["step_results"]

                                # ---- 从 agent_progress 推断执行阶段（简化版 Plan 卡片） ----
                                if "agent_progress" in node_update:
                                    prog_update = node_update["agent_progress"]
                                    for a_name, a_info in prog_update.items():
                                        if isinstance(a_info, dict) and a_info.get("status") == "BUSY":
                                            # Agent 正在执行：从 current_step 和 detail 推断阶段
                                            current_step_desc = a_info.get("current_step", "")
                                            detail_desc = a_info.get("detail", "")
                                            # 如果该 Agent 还没有 plan 数据，则根据进度推断
                                            if a_name not in agent_plans or not agent_plans[a_name].get("plan"):
                                                # 构造简化的执行阶段描述
                                                inferred_steps = []
                                                if current_step_desc:
                                                    inferred_steps.append({"action": current_step_desc, "expected": detail_desc})
                                                if inferred_steps:
                                                    agent_plans[a_name] = {
                                                        "plan": {"steps": inferred_steps},
                                                        "current_step_index": 0,
                                                        "step_results": [],
                                                    }

                        # 更新左侧全局状态条
                        with global_status_placeholder.container():
                            render_global_status(latest_state)

                        # 重绘右侧仪表盘（含计划卡片）
                        with dashboard_placeholder.container():
                            render_dashboard(latest_state, agent_plans)

                        # 显示当前执行节点
                        node_names = list(payload.keys())
                        status_text.info(f"正在执行: {', '.join(node_names)}")

                        # 尝试从最新消息中提取增量 AI 回复（兜底逻辑）
                        messages = latest_state.get("messages", [])
                        if messages:
                            last_msg = messages[-1]
                            if isinstance(last_msg, AIMessage) and last_msg.content:
                                # 仅在流式文本未捕获到内容时使用此兜底
                                if not full_response:
                                    full_response = last_msg.content
                                    message_placeholder.markdown(
                                        _preprocess_latex(full_response)
                                    )

            except Exception as e:
                st.error(f"执行出错：{e}")
                status_text.error("执行异常，请重试")

            else:
                # 流式结束
                status_text.success("执行完成")

                # 移除流式光标
                if full_response:
                    message_placeholder.markdown(_preprocess_latex(full_response))

                # 如果流式中未提取到内容，从最终状态中回溯查找
                if not full_response:
                    messages = latest_state.get("messages", [])
                    for msg in reversed(messages):
                        if isinstance(msg, AIMessage) and msg.content:
                            full_response = msg.content
                            message_placeholder.markdown(
                                _preprocess_latex(full_response)
                            )
                            break

    # 清空仪表盘占位符（执行结束后右侧恢复静态状态）
    dashboard_placeholder.empty()

    # 重置全局状态条为空闲
    with global_status_placeholder.container():
        render_global_status({})

    # 记录 AI 回复到对话历史
    if full_response:
        st.session_state.multi_agent_messages.append(
            {"role": "assistant", "content": full_response}
        )

    # 刷新页面以完整渲染对话历史
    st.rerun()
