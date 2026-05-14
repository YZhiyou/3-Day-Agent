"""多 Agent 协作图构建模块

使用 LangGraph 的 StateGraph 构建完整的多智能体协作图，
将 supervisor、message_bus、agents 三个模块的节点组合成可执行的协作流程。

协作流程：
    START → supervisor → publish → router → dispatch → route_to_agents（条件边）

    route_to_agents 根据各 Agent 的 inbox 状态决定下一节点：
    - researcher inbox 非空 → researcher
    - analyst inbox 非空 → analyst
    - writer inbox 非空 → writer
    - 所有 inbox 为空 → 检查 final_summary_ready
      - True → END
      - False → supervisor（继续循环）

    Agent 出口边：
    - researcher → publish（RESULT 消息经总线路由回 supervisor）
    - analyst → publish
    - writer → publish（RESULT 消息经总线路由回 supervisor）
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.multi_agent.multi_agent_state import MultiAgentState
from src.multi_agent.supervisor import supervisor_node
from src.multi_agent.message_bus import publish_node, router_node, dispatch_node
from src.multi_agent.agents import writer_node

logger = logging.getLogger(__name__)


# ============================================================
# 条件路由函数
# ============================================================

def route_to_agents(state: dict) -> str:
    """条件路由：根据各 Agent 的 inbox 状态决定下一节点

    检查顺序：researcher → analyst → writer
    - 优先级按检查顺序，先到先得
    - 所有 inbox 为空时，检查是否所有任务完成
      - final_summary_ready == True → 返回 "__end__"（图结束）
      - 否则 → 返回 "supervisor"（继续循环处理更多任务）

    Args:
        state: LangGraph 全局状态字典

    Returns:
        下一节点名称字符串
    """
    agent_private = state.get("agent_private", {})

    # 按优先级检查各 Agent 的 inbox
    for agent_name in ("researcher", "analyst", "writer"):
        inbox = agent_private.get(agent_name, {}).get("inbox", [])
        if inbox:
            logger.info(
                "路由决策: %s inbox 非空（%d 条消息）→ %s",
                agent_name, len(inbox), agent_name,
            )
            return agent_name

    # 所有 inbox 为空，检查是否全部完成
    if state.get("final_summary_ready") is True:
        logger.info("路由决策: 所有任务已完成 → END")
        return "__end__"

    # 尚未完成，回到 supervisor 继续处理
    logger.info("路由决策: inbox 均为空但未完成 → supervisor")
    return "supervisor"


# ============================================================
# 构建协作图
# ============================================================

def build_multi_agent_graph(
    plan_act_graph=None,
    tools=None,
) -> Any:
    """构建多 Agent 协作图

    节点组成：
    - supervisor: 任务分解与调度
    - publish:    消息发布到总线
    - router:     消息路由决策
    - dispatch:   消息投递到 Agent inbox
    - researcher: 文献检索 Agent（模拟 / Plan-Act 适配）
    - analyst:    文献分析 Agent（模拟 / Plan-Act 适配）
    - writer:     报告撰写 Agent

    边与路由：
    - START → supervisor
    - supervisor → publish → router → dispatch
    - dispatch → route_to_agents（条件边）
    - researcher → publish
    - analyst → publish
    - writer → publish

    Checkpointer：MemorySaver（内存版，不依赖 SQLite）

    Args:
        plan_act_graph: create_agent() 返回的 CompiledStateGraph 实例。
            若提供，researcher/analyst 将使用 Plan-Act 适配节点（真实推理）；
            若为 None，则使用 agents.py 中的模拟节点（向后兼容）。
        tools: Agent 可用工具列表，传入 Plan-Act 适配节点使用。

    Returns:
        编译后的 LangGraph CompiledGraph 实例
    """
    # 创建状态图
    graph = StateGraph(MultiAgentState)

    # ---- 添加节点 ----
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("publish", publish_node)
    graph.add_node("router", router_node)
    graph.add_node("dispatch", dispatch_node)

    # ---- researcher / analyst 节点：根据是否传入 plan_act_graph 决定使用适配节点或模拟节点 ----
    if plan_act_graph is not None:
        # 使用 Plan-Act 适配节点：将真实的 Plan-Act 子图包装为多 Agent 节点
        from src.multi_agent.agent_adapters import create_adapted_researcher_node, create_adapted_analyst_node
        adapted_researcher = create_adapted_researcher_node(plan_act_graph, tools)
        adapted_analyst = create_adapted_analyst_node(plan_act_graph, tools)
        graph.add_node("researcher", adapted_researcher)
        graph.add_node("analyst", adapted_analyst)
        logger.info("已启用 Plan-Act 适配节点（researcher / analyst）")
    else:
        # 向后兼容：使用 agents.py 中的模拟节点
        from src.multi_agent.agents import researcher_node, analyst_node
        graph.add_node("researcher", researcher_node)
        graph.add_node("analyst", analyst_node)
        logger.info("使用模拟节点（researcher / analyst）")

    graph.add_node("writer", writer_node)

    # ---- 入口边 ----
    graph.add_edge(START, "supervisor")

    # ---- 线性流水线：supervisor → publish → router → dispatch ----
    graph.add_edge("supervisor", "publish")
    graph.add_edge("publish", "router")
    graph.add_edge("router", "dispatch")

    # ---- 条件边：dispatch → route_to_agents ----
    # 映射覆盖 route_to_agents 所有可能的返回值
    graph.add_conditional_edges(
        "dispatch",
        route_to_agents,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "__end__": END,
            "supervisor": "supervisor",
        },
    )

    # ---- Agent 出口边 ----
    # researcher / analyst 处理完后发 RESULT 消息，需经总线路由回 supervisor
    graph.add_edge("researcher", "publish")
    graph.add_edge("analyst", "publish")
    # Writer 生成最终报告后经消息总线回传给 supervisor
    graph.add_edge("writer", "publish")

    # ---- 编译图，使用内存版 checkpointer ----
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ============================================================
# __main__ 演示代码
# ============================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    from langchain_core.messages import HumanMessage
    from src.multi_agent.multi_agent_state import (
        create_initial_multi_agent_state, AgentCapability, MessageBus, SharedScratchpad
    )

    # 初始化状态
    initial_state = create_initial_multi_agent_state(
        messages=[HumanMessage(content="请评审这篇论文：Transformer 架构比 CNN 好")],
        agent_registry=[
            AgentCapability(agent_name="researcher", skills=["retrieve", "web_search"], status="IDLE", max_concurrent=3),
            AgentCapability(agent_name="analyst", skills=["reason", "summarize"], status="IDLE", max_concurrent=2),
            AgentCapability(agent_name="writer", skills=["format_output"], status="IDLE", max_concurrent=1),
        ],
    )

    # 构建并运行图
    # 默认使用模拟节点运行，确保独立演示可用
    # 如需使用真实的 Plan-Act 子图，可传入 plan_act_graph：
    #   from src.core.agent import create_agent
    #   from src.tools.tools import create_tools
    #   tools = create_tools(user_id="user", retriever=my_retriever)
    #   plan_act_graph = create_agent(tools)
    #   graph = build_multi_agent_graph(plan_act_graph=plan_act_graph, tools=tools)
    graph = build_multi_agent_graph()
    config = {"configurable": {"thread_id": "review-demo-001"}, "recursion_limit": 100}

    print("=" * 60)
    print("论文评审委员会 - 多 Agent 协作演示")
    print("=" * 60)

    final_state = graph.invoke(initial_state, config=config)

    # 打印结果
    print("\n" + "=" * 60)
    print("任务队列最终状态:")
    print("=" * 60)
    for task in final_state.get("task_queue", []):
        status = task.status if hasattr(task, 'status') else task.get('status', '?')
        desc = task.description if hasattr(task, 'description') else task.get('description', '?')
        print(f"  [{status}] {desc}")

    print("\n" + "=" * 60)
    print("共享记事本内容:")
    print("=" * 60)
    scratchpad = final_state.get("shared_scratchpad")
    if scratchpad and hasattr(scratchpad, 'data'):
        for key, item in scratchpad.data.items():
            print(f"  {key}: (version={item.version}) {str(item.value)[:100]}...")

    print("\n" + "=" * 60)
    print("最终输出消息:")
    print("=" * 60)
    messages = final_state.get("messages", [])
    for msg in messages:
        msg_type = getattr(msg, 'type', type(msg).__name__)
        content = getattr(msg, 'content', str(msg))
        if msg_type == "ai":
            print(f"\n[AI 回复]\n{content}")
