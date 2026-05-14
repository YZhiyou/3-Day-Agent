"""消息总线节点模块

本模块实现基于 LangGraph 状态图的消息总线三阶段流水线：
1. publish_node  — 将 new_message_dict 转换为 MessageBusEntry 并发布到 pending 队列
2. router_node   — 遍历 pending 消息，根据消息类型生成路由决策
3. dispatch_node — 根据路由决策将消息副本投递到目标智能体的 inbox

三节点组成线性图：publish → router → dispatch → END
"""

import copy
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.multi_agent.multi_agent_state import (
    MessageBus,
    MessageBusEntry,
    MessageType,
    MessageStatus,
    MultiAgentState,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# 1. publish_node — 发布消息
# ============================================================

def publish_node(state: dict) -> dict:
    """将 state['new_message_dict'] 和/或 state['new_message_batch'] 转换为 MessageBusEntry 并追加到 pending 队列

    支持两种发布方式：
    - new_message_dict: 单条消息发布
    - new_message_batch: 批量消息发布
    两者可同时存在，也可仅存在其中一个。
    当两者均为空时，说明本轮无需发布，直接返回。
    """
    msg_dict = state.get("new_message_dict")
    msg_batch = state.get("new_message_batch")

    # 两者都为空时，无需更新
    if msg_dict is None and not msg_batch:
        return {}

    # 构建增量返回字典，只包含实际修改的字段
    updates: Dict[str, Any] = {}

    # 处理单条消息
    if msg_dict is not None:
        entry = MessageBusEntry(
            id=uuid.uuid4().hex,
            type=MessageType(msg_dict["type"]),
            sender=msg_dict.get("sender", "supervisor"),
            receiver=msg_dict.get("receiver"),  # 可能为 None
            payload=msg_dict.get("payload", {}),
            status=MessageStatus.PENDING,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )

        state["message_bus"].pending.append(entry)
        updates["new_message_dict"] = None

        logger.info("消息已发布: id=%s, type=%s, sender=%s", entry.id, entry.type.value, entry.sender)

    # 处理批量消息
    if msg_batch:
        for batch_item in msg_batch:
            entry = MessageBusEntry(
                id=uuid.uuid4().hex,
                type=MessageType(batch_item["type"]),
                sender=batch_item.get("sender", "supervisor"),
                receiver=batch_item.get("receiver"),
                payload=batch_item.get("payload", {}),
                status=MessageStatus.PENDING,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
            )
            state["message_bus"].pending.append(entry)
            logger.info("批量消息已发布: id=%s, type=%s, sender=%s", entry.id, entry.type.value, entry.sender)
        updates["new_message_batch"] = None

    # 只要有消息发布，message_bus 就被修改了
    updates["message_bus"] = state["message_bus"]

    return updates


# ============================================================
# 2. 辅助函数 _get_agent_load
# ============================================================

def _get_agent_load(state: dict, agent_name: str) -> int:
    """计算指定智能体当前负载（inbox 长度）"""
    return len(state.get("agent_private", {}).get(agent_name, {}).get("inbox", []))


# ============================================================
# 3. router_node — 路由决策
# ============================================================

def router_node(state: dict) -> dict:
    """遍历 pending 中状态为 PENDING 的消息，生成路由决策

    路由规则：
    - TASK:       按 required_capability 在 agent_registry 中匹配，优先负载最小的在线 agent
    - QUERY:      投递给指定 receiver（若在线），否则记录警告
    - BROADCAST:  投递给所有在线且非 sender 的 agent
    - RESULT:     投递给 supervisor

    决策存储在 state["agent_private"]["__bus_routing__"]
    """
    bus: MessageBus = state["message_bus"]
    agent_registry = state.get("agent_registry", [])
    agent_private = state.get("agent_private", {})

    # 构建在线 agent 集合（status != "OFFLINE"）
    online_agents = [
        cap.agent_name for cap in agent_registry
        if cap.status != "OFFLINE"
    ]

    # 路由决策: { message_id: [agent_name, ...] }
    routing_decisions: Dict[str, List[str]] = {}

    for entry in bus.pending:
        if entry.status != MessageStatus.PENDING:
            continue

        if entry.type == MessageType.TASK:
            # 从 payload 获取所需能力
            required_cap = entry.payload.get("required_capability", "")
            # 在注册表中查找拥有该能力且在线的 agent
            candidates = [
                cap.agent_name for cap in agent_registry
                if cap.status != "OFFLINE" and required_cap in cap.skills
            ]
            if candidates:
                # 多个匹配时选负载最小的
                candidates.sort(key=lambda name: _get_agent_load(state, name))
                routing_decisions[entry.id] = [candidates[0]]
            else:
                # 完全找不到则广播给所有在线 agent
                routing_decisions[entry.id] = list(online_agents)
                logger.warning("TASK 消息 %s 未找到能力 '%s' 的 agent，广播至所有在线 agent", entry.id, required_cap)

        elif entry.type == MessageType.QUERY:
            receiver = entry.receiver
            if receiver and receiver in online_agents:
                routing_decisions[entry.id] = [receiver]
            else:
                logger.warning("QUERY 消息 %s 的 receiver '%s' 不在线或为空，无法投递", entry.id, receiver)

        elif entry.type == MessageType.BROADCAST:
            # 投递给所有在线且非 sender 的 agent
            targets = [name for name in online_agents if name != entry.sender]
            routing_decisions[entry.id] = targets

        elif entry.type == MessageType.RESULT:
            # 结果消息投递给 supervisor
            routing_decisions[entry.id] = ["supervisor"]

    # 存储路由决策
    agent_private = state.get("agent_private", {})
    if "__bus_routing__" not in agent_private:
        agent_private["__bus_routing__"] = {}
    agent_private["__bus_routing__"].update(routing_decisions)

    logger.info("路由决策: %s", routing_decisions)
    return {"agent_private": agent_private}


# ============================================================
# 4. dispatch_node — 投递消息
# ============================================================

def dispatch_node(state: dict) -> dict:
    """根据路由决策将消息副本投递到目标智能体的 inbox

    流程：
    1. 读取 __bus_routing__ 中的路由决策
    2. 对每个 message_id，从 pending 中找到对应消息
    3. 深拷贝消息，状态改为 DELIVERED，更新 updated_at
    4. 将消息副本追加到各目标 agent 的 inbox
    5. 原消息从 pending 移除，状态改为 DELIVERED，插入 processing 列表
    6. 清除 __bus_routing__
    """
    bus: MessageBus = state["message_bus"]
    agent_private = state.get("agent_private", {})
    routing: Dict[str, List[str]] = agent_private.get("__bus_routing__", {})

    # 记录哪些 agent 收到了新消息，用于更新 agent_progress
    agents_with_new_messages: set = set()

    for message_id, target_agents in routing.items():
        # 在 pending 中找到对应消息
        entry_index = None
        entry = None
        for i, pending_entry in enumerate(bus.pending):
            if pending_entry.id == message_id:
                entry_index = i
                entry = pending_entry
                break

        if entry is None:
            logger.warning("路由决策中的消息 %s 在 pending 中未找到，跳过", message_id)
            continue

        # 对每个目标 agent 投递消息副本
        for agent_name in target_agents:
            msg_copy = copy.deepcopy(entry)
            msg_copy.status = MessageStatus.DELIVERED
            msg_copy.updated_at = datetime.now().isoformat()

            # 确保 agent_private 中该 agent 有 inbox 列表
            if agent_name not in agent_private:
                agent_private[agent_name] = {}
            if "inbox" not in agent_private[agent_name]:
                agent_private[agent_name]["inbox"] = []

            # 追加消息副本的 dict 形式到 inbox
            agent_private[agent_name]["inbox"].append(msg_copy.model_dump())
            agents_with_new_messages.add(agent_name)

        # 原消息从 pending 移除
        bus.pending.pop(entry_index)

        # 原消息状态改为 DELIVERED，插入 processing 列表
        entry.status = MessageStatus.DELIVERED
        entry.updated_at = datetime.now().isoformat()
        bus.processing.append(entry)

        logger.info("消息 %s 已投递至: %s", message_id, target_agents)

    # 清除路由决策
    agent_private.pop("__bus_routing__", None)

    # 投递消息后，更新 agent_progress 为 BUSY
    progress = dict(state.get("agent_progress") or {})
    for agent_name_key in agents_with_new_messages:
        agent_info = dict(progress.get(agent_name_key) or {})
        agent_info["status"] = "BUSY"
        agent_info["current_step"] = "收到新任务，等待执行"
        agent_info["progress"] = 10
        progress[agent_name_key] = agent_info

    return {
        "message_bus": bus,
        "agent_private": agent_private,
        "agent_progress": progress,
    }


# ============================================================
# 5. 构建演示图
# ============================================================

def build_message_bus_demo_graph():
    """构建消息总线演示图

    线性流水线：publish → router → dispatch → END
    使用 MemorySaver 作为 checkpointer（内存版，不依赖 sqlite）
    """
    graph = StateGraph(MultiAgentState)

    # 添加节点
    graph.add_node("publish", publish_node)
    graph.add_node("router", router_node)
    graph.add_node("dispatch", dispatch_node)

    # 设置入口
    graph.set_entry_point("publish")

    # 添加边
    graph.add_edge("publish", "router")
    graph.add_edge("router", "dispatch")
    graph.add_edge("dispatch", END)

    # 编译图，使用内存版 checkpointer
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ============================================================
# 6. __main__ 示例
# ============================================================

if __name__ == "__main__":
    from src.multi_agent.multi_agent_state import (
        create_initial_multi_agent_state, AgentCapability, MessageBus, SharedScratchpad
    )

    # 初始化状态
    initial_state = create_initial_multi_agent_state(
        agent_registry=[
            AgentCapability(agent_name="researcher", skills=["retrieve"], status="IDLE", max_concurrent=3),
            AgentCapability(agent_name="analyst", skills=["reason"], status="IDLE", max_concurrent=2),
        ],
        new_message_dict={
            "type": "TASK",
            "sender": "supervisor",
            "payload": {"task_id": "1", "required_capability": "retrieve", "query": "transformer architecture"}
        }
    )

    # 构建并运行图
    graph = build_message_bus_demo_graph()
    config = {"configurable": {"thread_id": "demo-001"}}
    final_state = graph.invoke(initial_state, config=config)

    # 验证结果
    print("=== Researcher Inbox ===")
    inbox = final_state.get("agent_private", {}).get("researcher", {}).get("inbox", [])
    for msg in inbox:
        print(f"  消息ID: {msg['id']}, 类型: {msg['type']}, 状态: {msg['status']}")

    print("\n=== Message Bus Processing ===")
    bus = final_state.get("message_bus")
    if hasattr(bus, 'processing'):
        for msg in bus.processing:
            print(f"  消息ID: {msg.id}, 已投递, 状态: {msg.status}")

    print("\n=== 消息总线演示完成 ===")
