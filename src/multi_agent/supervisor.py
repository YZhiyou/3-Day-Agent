"""Supervisor 调度逻辑

负责：
1. 任务分解 — decompose_task
2. 冲突检测 — detect_conflicts
3. 调度节点 — supervisor_node（LangGraph 状态图节点）
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from src.multi_agent.multi_agent_state import AgentTask, MultiAgentState

logger = logging.getLogger(__name__)


# ============================================================
# 1. 任务分解
# ============================================================

def decompose_task(
    user_request: str,
    llm_call: Optional[Callable] = None,
) -> List[AgentTask]:
    """将用户请求分解为 AgentTask 列表

    当前为硬编码实现：若请求包含"评审"或"论文"，返回 4 个评审子任务；
    否则返回一个通用任务。llm_call 参数预留，当前不使用。

    Args:
        user_request: 用户的原始请求文本
        llm_call: 可选的 LLM 调用函数（预留接口）

    Returns:
        分解后的 AgentTask 列表
    """
    # 论文评审场景：4 个子任务，task_4 依赖前三个
    if "评审" in user_request or "论文" in user_request:
        logger.info("检测到论文评审场景，分解为 4 个子任务")
        task_1 = AgentTask(
            task_id="task_1",
            description="提取论文核心论点",
            required_capability="reason",
            dependencies=["task_2"],
            status="PENDING",
        )
        task_2 = AgentTask(
            task_id="task_2",
            description="查证关键事实",
            required_capability="retrieve",
            dependencies=[],
            status="PENDING",
        )
        task_3 = AgentTask(
            task_id="task_3",
            description="批判逻辑漏洞",
            required_capability="reason",
            dependencies=["task_2"],
            status="PENDING",
        )
        task_4 = AgentTask(
            task_id="task_4",
            description="撰写评审报告",
            required_capability="format_output",
            dependencies=["task_1", "task_2", "task_3"],
            status="PENDING",
        )
        return [task_1, task_2, task_3, task_4]

    # 通用场景：单个任务
    logger.info("未匹配特定场景，生成通用任务")
    return [
        AgentTask(
            task_id="task_1",
            description=f"处理请求: {user_request}",
            required_capability="general",
            dependencies=[],
            status="PENDING",
        )
    ]


# ============================================================
# 2. 冲突检测
# ============================================================

def detect_conflicts(task_results: Dict[str, Any]) -> List[Dict]:
    """检查多个 agent 的结果是否存在矛盾结论

    当前采用简单的关键词匹配：
    - 若同一 key 下多个 agent 的结论中一个包含"优于"而另一个包含"不足"，视为冲突

    Args:
        task_results: 形如 {"task_id": {"agent": str, "conclusion": Any, ...}, ...}

    Returns:
        冲突列表，每项为 {"key": str, "agents": [str], "values": [Any]}
    """
    conflicts: List[Dict] = []

    # 按 key 聚合结果：以 conclusion 中出现的 key 为检测维度
    # 简单实现 — 检查是否有 agent 结论中同时出现互斥关键词
    positive_agents: List[str] = []
    positive_values: List[Any] = []
    negative_agents: List[str] = []
    negative_values: List[Any] = []

    for task_id, result in task_results.items():
        if not isinstance(result, dict):
            continue
        agent_name = result.get("agent", task_id)
        conclusion = result.get("conclusion")
        # 将结论转为字符串做关键词匹配
        conclusion_str = str(conclusion) if conclusion is not None else ""

        if "优于" in conclusion_str:
            positive_agents.append(agent_name)
            positive_values.append(conclusion)
        if "不足" in conclusion_str:
            negative_agents.append(agent_name)
            negative_values.append(conclusion)

    # 若同时存在正向和负向结论，视为冲突
    if positive_agents and negative_agents:
        conflict = {
            "key": "conclusion_sentiment",
            "agents": positive_agents + negative_agents,
            "values": positive_values + negative_values,
        }
        conflicts.append(conflict)
        logger.warning("检测到结论冲突: %s", conflict)

    return conflicts


# ============================================================
# 3. Supervisor 调度节点
# ============================================================

def supervisor_node(state: dict) -> dict:
    """Supervisor 调度节点，作为 LangGraph 状态图节点使用

    两个分支：
    - 分支 A（首次运行，task_queue 为空）：分解任务 → 派发无依赖任务
    - 分支 B（后续运行，task_queue 非空）：处理 RESULT → 解锁新任务 → 判断是否全部完成

    Args:
        state: LangGraph 传入的状态字典

    Returns:
        部分更新字典（仅包含需要更新的字段）
    """
    task_queue: List[AgentTask] = state.get("task_queue", [])

    # ----------------------------------------------------------
    # 分支 A：首次运行 — task_queue 为空
    # ----------------------------------------------------------
    if not task_queue:
        logger.info("分支 A：首次运行，开始任务分解")

        # 从 messages 中找到最后一条 HumanMessage
        user_request = ""
        messages = state.get("messages", [])
        for msg in reversed(messages):
            # HumanMessage 的 type 属性为 "human"
            if getattr(msg, "type", None) == "human":
                user_request = getattr(msg, "content", "")
                break

        if not user_request:
            logger.warning("未找到用户消息，无法分解任务")
            return {
                "task_queue": [],
                "new_message_batch": None,
                "final_summary_ready": True,
            }

        # 调用 decompose_task 生成任务列表
        tasks = decompose_task(user_request)
        logger.info("任务分解完成，共 %d 个任务", len(tasks))

        # 为无依赖的任务生成 TASK 消息
        batch: List[Dict[str, Any]] = []
        for task in tasks:
            if not task.dependencies:
                batch.append({
                    "type": "TASK",
                    "sender": "supervisor",
                    "payload": {
                        "task_id": task.task_id,
                        "required_capability": task.required_capability,
                        "description": task.description,
                        "user_query": user_request,
                    },
                })
                logger.info("派发任务: task_id=%s, capability=%s", task.task_id, task.required_capability)

        return {"task_queue": tasks, "new_message_batch": batch}

    # ----------------------------------------------------------
    # 分支 B：后续运行 — 处理 RESULT 消息，解锁依赖任务
    # ----------------------------------------------------------
    logger.info("分支 B：后续运行，处理回传结果")

    # 读取 supervisor 的 inbox
    inbox = state.get("agent_private", {}).get("supervisor", {}).get("inbox", [])
    if not inbox:
        logger.info("supervisor inbox 为空，无新结果需处理")

    # 处理每条 RESULT 消息
    processed_ids: List[str] = []
    for msg in inbox:
        msg_type = msg.get("type", "")
        if msg_type != "RESULT":
            continue

        payload = msg.get("payload", {})
        task_id = payload.get("task_id")

        # 在 task_queue 中找到对应任务，更新状态和结果
        for task in task_queue:
            if task.task_id == task_id:
                task.status = "COMPLETED"
                task.result = payload.get("result")
                processed_ids.append(msg.get("id", task_id))
                logger.info("任务完成: task_id=%s", task_id)
                break

    # 从 inbox 中移除已处理的消息
    if processed_ids:
        updated_inbox = [
            m for m in inbox
            if m.get("id", "") not in processed_ids
        ]
        # 构造 agent_private 的增量更新（merge 模式，保留其他 Agent 的私有状态）
        old_agent_private = state.get("agent_private", {})
        agent_private_update = {
            **old_agent_private,
            "supervisor": {
                **old_agent_private.get("supervisor", {}),
                "inbox": updated_inbox,
            },
        }
    else:
        agent_private_update = {}

    # 从 messages 中重新提取用户原始查询（分支 B 中 state 仍持有 messages）
    user_request_br = ""
    messages_br = state.get("messages", [])
    for msg in reversed(messages_br):
        if getattr(msg, "type", None) == "human":
            user_request_br = getattr(msg, "content", "")
            break

    # 扫描 task_queue，找到所有 PENDING 且依赖已完成的任务
    completed_ids = {t.task_id for t in task_queue if t.status == "COMPLETED"}
    batch: List[Dict[str, Any]] = []
    for task in task_queue:
        if task.status == "PENDING" and task.dependencies:
            # 检查依赖是否全部完成
            if all(dep_id in completed_ids for dep_id in task.dependencies):
                batch.append({
                    "type": "TASK",
                    "sender": "supervisor",
                    "payload": {
                        "task_id": task.task_id,
                        "required_capability": task.required_capability,
                        "description": task.description,
                        "user_query": user_request_br,
                    },
                })
                logger.info("依赖满足，派发任务: task_id=%s", task.task_id)

    # 检查是否所有任务都已完成
    all_completed = all(t.status == "COMPLETED" for t in task_queue)

    # 构造返回字典
    result: Dict[str, Any] = {"task_queue": task_queue, "new_message_batch": batch}

    if agent_private_update:
        result["agent_private"] = agent_private_update

    if all_completed:
        result["final_summary_ready"] = True
        logger.info("所有任务已完成，设置 final_summary_ready=True")

    return result
