"""多智能体共享状态模型

本模块定义了多智能体协作所需的核心数据结构：
1. MessageBus — 消息总线，用于智能体间异步通信
2. AgentCapability / AgentTask — 代理能力注册与任务分配
3. SharedScratchpad — 乐观锁共享记事本，避免并发写入冲突
4. MultiAgentState — 继承自 AgentState 的 TypedDict，兼容 LangGraph 状态图
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Annotated, Dict, List, Optional

from pydantic import BaseModel, Field

from src.core.agent import AgentState, Plan, PlanStep, StepResult
from langgraph.graph.message import add_messages


# ============================================================
# 1. 消息总线模型
# ============================================================

class MessageType(str, Enum):
    """消息类型枚举"""
    TASK = "TASK"            # 任务派发
    RESULT = "RESULT"        # 结果回传
    QUERY = "QUERY"          # 查询请求
    BROADCAST = "BROADCAST"  # 广播通知


class MessageStatus(str, Enum):
    """消息状态枚举，追踪消息生命周期"""
    CREATED = "CREATED"        # 已创建
    PENDING = "PENDING"        # 等待投递
    DELIVERED = "DELIVERED"    # 已投递
    PROCESSING = "PROCESSING"  # 处理中
    COMPLETED = "COMPLETED"    # 已完成
    ARCHIVED = "ARCHIVED"      # 已归档


class MessageBusEntry(BaseModel):
    """消息总线中的单条消息

    receiver 为 None 时表示广播消息，所有智能体均可消费。
    时间戳采用 isoformat 字符串存储，确保 SqliteSaver 序列化兼容。
    """
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    type: MessageType
    sender: str
    receiver: Optional[str] = None  # None 表示广播
    payload: Dict[str, Any] = Field(default_factory=dict)
    status: MessageStatus = MessageStatus.CREATED
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class MessageBus(BaseModel):
    """消息总线：管理智能体间异步通信

    消息流: publish → pending → consume → processing → complete → archived
    """
    pending: List[MessageBusEntry] = Field(default_factory=list)
    processing: List[MessageBusEntry] = Field(default_factory=list)
    archived: List[MessageBusEntry] = Field(default_factory=list)

    def publish(self, entry: MessageBusEntry) -> None:
        """发布消息到待处理队列"""
        entry.status = MessageStatus.PENDING
        entry.updated_at = datetime.now().isoformat()
        self.pending.append(entry)

    def consume(self, receiver: str) -> Optional[MessageBusEntry]:
        """从 pending 队列消费一条消息

        查找第一条 receiver 匹配或广播消息（receiver=None），
        将其移入 processing 队列并返回；无匹配时返回 None。
        """
        for i, entry in enumerate(self.pending):
            if entry.receiver is None or entry.receiver == receiver:
                entry.status = MessageStatus.PROCESSING
                entry.updated_at = datetime.now().isoformat()
                self.pending.pop(i)
                self.processing.append(entry)
                return entry
        return None

    def complete(self, message_id: str) -> None:
        """将 processing 中的消息标记为完成并归档"""
        for i, entry in enumerate(self.processing):
            if entry.id == message_id:
                entry.status = MessageStatus.COMPLETED
                entry.updated_at = datetime.now().isoformat()
                self.processing.pop(i)
                self.archived.append(entry)
                return


# ============================================================
# 2. 任务和代理能力模型
# ============================================================

class AgentCapability(BaseModel):
    """智能体能力描述，用于任务分配时的能力匹配"""
    agent_name: str
    skills: List[str] = Field(default_factory=list)
    status: str = "IDLE"  # IDLE / BUSY / OFFLINE
    max_concurrent: int = 1


class AgentTask(BaseModel):
    """智能体任务，支持依赖关系声明

    dependencies 为依赖的 task_id 列表，只有依赖全部完成后才可执行。
    """
    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    description: str
    required_capability: str  # 如 "retrieve"
    assigned_agent: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    status: str = "PENDING"  # PENDING / IN_PROGRESS / COMPLETED / FAILED
    result: Optional[Any] = None


# ============================================================
# 3. 共享记事本（乐观锁）
# ============================================================

class ScratchpadItem(BaseModel):
    """记事本中的单个条目，携带版本号用于乐观锁控制"""
    value: Any
    version: int = 0


class SharedScratchpad(BaseModel):
    """共享记事本：多智能体并发读写，采用乐观锁避免冲突

    写入时须提供 expected_version，仅在版本匹配时才允许更新，
    不匹配则返回 False，由调用方决定重试或放弃。
    """
    data: Dict[str, ScratchpadItem] = Field(default_factory=dict)

    def get(self, key: str) -> Optional[Any]:
        """安全获取值，key 不存在返回 None"""
        item = self.data.get(key)
        return item.value if item is not None else None

    def set(self, key: str, value: Any, expected_version: int = 0) -> bool:
        """乐观锁写入

        - key 不存在：直接写入，返回 True
        - key 存在且 version == expected_version：更新 value 并 version+1，返回 True
        - key 存在但 version != expected_version：写入失败，返回 False
        """
        if key not in self.data:
            self.data[key] = ScratchpadItem(value=value, version=0)
            return True
        item = self.data[key]
        if item.version != expected_version:
            return False
        item.value = value
        item.version += 1
        return True

    def get_version(self, key: str) -> int:
        """获取指定 key 的版本号，不存在返回 0"""
        item = self.data.get(key)
        return item.version if item is not None else 0


# ============================================================
# 4. 自定义 Reducer
# ============================================================

def merge_shared_scratchpad(left: SharedScratchpad, right: SharedScratchpad) -> SharedScratchpad:
    """自定义 reducer：深合并两个 scratchpad，右侧(新数据)覆盖左侧"""
    if right is None:
        return left
    if left is None:
        return right
    # 将右侧新增/更新的 key 合入左侧
    merged = SharedScratchpad()
    merged.data = {**left.data, **right.data}
    return merged


# ============================================================
# 5. MultiAgentState（TypedDict，兼容 LangGraph）
# ============================================================

class MultiAgentState(AgentState):
    """多智能体共享状态，继承自 AgentState

    新增字段：
    - message_bus: 消息总线，管理智能体间通信
    - task_queue: 任务队列，等待分配与执行的任务
    - agent_registry: 代理注册表，记录各智能体的能力与状态
    - shared_scratchpad: 共享记事本，乐观锁并发控制（自定义 reducer 深合并）
    - agent_private: 各智能体的私有状态字典
    - new_message_dict: 待发布到消息总线的原始消息字典
    - new_message_batch: 批量消息列表（供 publish_node 批量发布）
    - final_summary_ready: 所有任务完成标记
    - agent_progress: 各智能体的执行进度字典（默认覆盖策略）
    """
    message_bus: MessageBus
    task_queue: List[AgentTask]
    agent_registry: List[AgentCapability]
    shared_scratchpad: Annotated[SharedScratchpad, merge_shared_scratchpad]
    agent_private: Dict[str, Dict[str, Any]]
    new_message_dict: Optional[Dict[str, Any]]
    new_message_batch: Optional[List[Dict[str, Any]]]
    final_summary_ready: bool
    agent_progress: Dict[str, Any]


# ============================================================
# 6. 工具函数
# ============================================================

def get_agent_private(state: MultiAgentState, agent_name: str) -> Dict[str, Any]:
    """安全获取智能体私有状态

    若该智能体尚未注册私有状态，返回空字典。
    """
    return state.get("agent_private", {}).get(agent_name, {})


def set_agent_private(state: MultiAgentState, agent_name: str, key: str, value: Any) -> None:
    """设置智能体私有状态中的某个 key"""
    if "agent_private" not in state:
        state["agent_private"] = {}
    if agent_name not in state["agent_private"]:
        state["agent_private"][agent_name] = {}
    state["agent_private"][agent_name][key] = value


def create_initial_multi_agent_state(**kwargs) -> dict:
    """创建初始多智能体状态字典

    为所有 MultiAgentState 新增字段提供默认值，
    同时保留原 AgentState 的默认值。
    可通过 kwargs 覆盖任意字段。
    """
    state: dict = {
        # AgentState 原有字段
        "messages": [],
        "plan": None,
        "current_step_index": 0,
        "step_results": [],
        "need_replan": False,
        "is_complex": None,
        "final_summary": None,
        "replan_count": 0,
        "conversation_summary": None,
        "summary_covered_rounds": 0,
        "pending_clarification": None,
        "error_log": [],
        "last_slots": {},
        "confidence": None,
        # MultiAgentState 新增字段
        "message_bus": MessageBus(),
        "task_queue": [],
        "agent_registry": [],
        "shared_scratchpad": SharedScratchpad(),
        "agent_private": {},
        "new_message_dict": None,
        "new_message_batch": None,
        "final_summary_ready": False,
        "agent_progress": {},
    }
    state.update(kwargs)
    return state


# ============================================================
# 7. 示例用法
# ============================================================

if __name__ == "__main__":
    # ---- 创建初始状态 ----
    state = create_initial_multi_agent_state()

    # ---- 注册三个智能体能力 ----
    state["agent_registry"] = [
        AgentCapability(agent_name="Researcher", skills=["retrieve", "search"]),
        AgentCapability(agent_name="Analyst", skills=["analyze", "summarize"]),
        AgentCapability(agent_name="Writer", skills=["write", "format"]),
    ]
    print("已注册智能体:")
    for cap in state["agent_registry"]:
        print(f"  {cap.agent_name}: skills={cap.skills}, status={cap.status}")

    # ---- 发布一条 TASK 消息 ----
    bus: MessageBus = state["message_bus"]
    task_msg = MessageBusEntry(
        type=MessageType.TASK,
        sender="Coordinator",
        receiver="Researcher",
        payload={"query": "检索强化学习相关论文"},
    )
    bus.publish(task_msg)
    print(f"\n消息已发布: id={task_msg.id}, status={task_msg.status}")

    # Researcher 消费消息
    consumed = bus.consume("Researcher")
    print(f"Researcher 消费消息: {consumed.id if consumed else None}, status={consumed.status if consumed else None}")

    # 完成消息
    bus.complete(consumed.id)
    print(f"消息已归档，archived 数量: {len(bus.archived)}")

    # ---- 在 scratchpad 中写入数据 ----
    pad: SharedScratchpad = state["shared_scratchpad"]
    ok1 = pad.set("retrieved_papers", ["PPO论文", "SAC论文", "TD3论文"])
    print(f"\n写入 retrieved_papers: 成功={ok1}, version={pad.get_version('retrieved_papers')}")

    # 读取
    papers = pad.get("retrieved_papers")
    print(f"读取 retrieved_papers: {papers}")

    # ---- 演示乐观锁成功更新 ----
    current_ver = pad.get_version("retrieved_papers")
    ok2 = pad.set("retrieved_papers", ["PPO论文", "SAC论文", "TD3论文", "DS-PPO论文"], expected_version=current_ver)
    print(f"\n乐观锁更新（version={current_ver}）: 成功={ok2}, 新version={pad.get_version('retrieved_papers')}")

    # ---- 演示乐观锁失败场景 ----
    stale_ver = 0  # 使用过期的版本号
    ok3 = pad.set("retrieved_papers", "旧数据", expected_version=stale_ver)
    print(f"乐观锁更新（stale version={stale_ver}）: 成功={ok3} ← 预期失败，数据未被覆盖")
    print(f"retrieved_papers 当前值: {pad.get('retrieved_papers')}")

    # ---- 演示序列化 ----
    print(f"\n序列化测试: MessageBus.model_dump() 类型={type(bus.model_dump())}")
    print(f"序列化测试: SharedScratchpad.model_dump() 类型={type(pad.model_dump())}")
