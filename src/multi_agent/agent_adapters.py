"""Plan-Act 子图适配器

将 agent.py 中 create_agent 返回的 Plan-Act CompiledStateGraph
包装为多 Agent 协作图中可用的节点函数。

核心设计：
- 每个 Plan-Act 子图拥有独立的 SqliteSaver checkpointer，
  通过独立 thread_id 实现会话隔离
- 子图调用用 try/except 包裹，失败时返回错误 RESULT，
  不让整个多 Agent 图崩溃
- 写入 shared_scratchpad 时使用乐观锁，与现有 agents.py 一致
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage, AIMessage

from src.multi_agent.multi_agent_state import MultiAgentState

logger = logging.getLogger(__name__)


def _update_progress(state: dict, agent_name: str, **kwargs) -> dict:
    """返回更新后的完整 agent_progress 字典

    用于在适配节点中分阶段更新 agent_progress，
    避免手动拼接字典的重复代码。

    Args:
        state: 当前 LangGraph 状态字典
        agent_name: 智能体名称
        **kwargs: 要更新的进度字段（如 status, current_step, detail 等）

    Returns:
        更新后的完整 agent_progress 字典
    """
    progress = dict(state.get("agent_progress") or {})
    agent_info = dict(progress.get(agent_name) or {})
    agent_info.update(kwargs)
    progress[agent_name] = agent_info
    return progress


def _resolve_scratchpad_key(agent_name: str, task_description: str) -> str:
    """根据 agent 名称和任务描述确定 scratchpad 写入 key

    使适配节点直接写入 writer_node 预期的 key，避免 key 不匹配导致数据丢失。

    规则：
    - researcher 任务 → "retrieved_docs"
    - 包含"论点"或"提取"的任务 → "claims"
    - 包含"批判"或"逻辑"的任务 → "critique"
    - 其他 → f"{agent_name}_result"（兜底，仍可被 writer_node fallback 逻辑读取）

    Args:
        agent_name: 智能体名称，如 "researcher"、"analyst"
        task_description: 任务描述文本

    Returns:
        scratchpad key 字符串
    """
    # researcher 统一写入 retrieved_docs
    if agent_name == "researcher":
        return "retrieved_docs"

    # analyst 根据任务描述细分
    if agent_name == "analyst":
        if "论点" in task_description or "提取" in task_description:
            return "claims"
        if "批判" in task_description or "逻辑" in task_description:
            return "critique"

    # 兜底：使用 agent_name_result
    return f"{agent_name}_result"


# ============================================================
# 核心：Plan-Act 子图 → 多 Agent 节点适配器
# ============================================================

def wrap_plan_act_as_agent(
    agent_name: str,
    plan_act_graph: Any,
    tools: Optional[List] = None,
) -> Callable[[dict], dict]:
    """将 Plan-Act 子图包装为多 Agent 协作图中的节点函数

    返回的节点函数签名满足 LangGraph 要求：(state: dict) -> dict

    内部流程：
    1. 从 agent_private[agent_name]["inbox"] 取第一条消息
    2. 提取 payload.description → 构造 HumanMessage
    3. 构建子状态字典（AgentState 格式）
    4. 使用独立 thread_id 调用子图 invoke
    5. 提取最后一条 AIMessage 的 content 作为结果
    6. 将结果写入 shared_scratchpad（乐观锁）
    7. 构造 RESULT 消息，追加到 new_message_batch
    8. 从 inbox 移除已处理消息
    9. 返回部分状态更新字典

    Args:
        agent_name: 智能体名称，如 "researcher"、"analyst"
        plan_act_graph: create_agent() 返回的 CompiledStateGraph 实例
        tools: 工具列表（保留扩展接口，当前未直接使用）

    Returns:
        可作为 LangGraph 节点的函数 (state: dict) -> dict
    """

    def agent_node(state: dict) -> dict:
        # ---- 1. 从 inbox 取第一条消息 ----
        inbox: List[dict] = (
            state.get("agent_private", {})
            .get(agent_name, {})
            .get("inbox", [])
        )

        if not inbox:
            logger.debug(f"[{agent_name}] inbox 为空，跳过执行")
            return {}

        message: dict = inbox[0]
        payload = message.get("payload", {})
        task_id = payload.get("task_id", "unknown")
        task_description = payload.get("description") or payload.get("query") or ""
        user_query = payload.get("user_query", "")

        logger.info(
            f"[{agent_name}] 开始处理任务: task_id={task_id}, "
            f"description={task_description[:50]}..."
        )

        # ---- 2. 构建子状态字典（AgentState 格式） ----
        # 组合用户原始查询与任务描述，让子图获得完整上下文
        if user_query:
            full_prompt = (
                f"任务：{task_description}\n\n"
                f"用户原始请求：{user_query}\n\n"
                f"请根据用户的原始请求，完成上述任务。直接给出分析结果，不要反问用户。"
            )
        else:
            full_prompt = task_description

        sub_state = {
            "messages": [HumanMessage(content=full_prompt)],
            "plan": None,
            "current_step_index": 0,
            "step_results": [],
            "need_replan": False,
            "is_complex": None,
            "final_summary": None,
            "replan_count": 0,
            "conversation_summary": state.get("conversation_summary"),
            "summary_covered_rounds": 0,
            "pending_clarification": None,
            "error_log": [],
            "last_slots": {},
            "confidence": None,
        }

        # ---- 3. 使用独立 thread_id 调用子图 ----
        run_id = uuid.uuid4().hex[:8]
        sub_config = {"configurable": {"thread_id": f"{agent_name}_{task_id}_{run_id}"}}

        try:
            result_state = plan_act_graph.invoke(sub_state, config=sub_config)
        except Exception as exc:
            # 子图调用失败：返回错误 RESULT，不让整个多 Agent 图崩溃
            logger.error(
                f"[{agent_name}] Plan-Act 子图调用失败: {exc}",
                exc_info=True,
            )
            error_content = f"[{agent_name}] 子图执行异常: {exc}"

            # 构造错误 RESULT 消息
            result_msg = {
                "type": "RESULT",
                "sender": agent_name,
                "receiver": "supervisor",
                "payload": {
                    "task_id": task_id,
                    "result": {"content": error_content, "source": agent_name, "error": True},
                },
            }

            # 从 inbox 移除已处理消息
            updated_inbox = list(inbox)
            updated_inbox.pop(0)

            # 合并 new_message_batch
            existing_batch = state.get("new_message_batch") or []
            new_batch = existing_batch + [result_msg]

            logger.info(f"[{agent_name}] 错误 RESULT 已放入 new_message_batch")

            error_progress = _update_progress(
                state, agent_name,
                status="IDLE",
                current_step="执行出错",
                detail=str(exc)[:50],
                progress=0,
                last_log="子图执行异常",
            )

            return {
                "agent_private": {
                    **state.get("agent_private", {}),
                    agent_name: {
                        **state.get("agent_private", {}).get(agent_name, {}),
                        "inbox": updated_inbox,
                    },
                },
                "new_message_batch": new_batch,
                "agent_progress": error_progress,
            }

        # ---- 4. 从结果状态中提取最后一条 AIMessage ----
        ai_content = ""
        result_messages = result_state.get("messages", [])
        for msg in reversed(result_messages):
            if isinstance(msg, AIMessage):
                ai_content = msg.content
                break
            # 兼容字典格式的消息
            if isinstance(msg, dict) and msg.get("type") == "ai":
                ai_content = msg.get("content", "")
                break

        if not ai_content:
            ai_content = str(result_state.get("final_summary") or "（子图未产生有效输出）")

        logger.info(
            f"[{agent_name}] Plan-Act 子图执行完成, "
            f"结果长度={len(ai_content)} 字符"
        )

        # ---- 5. 将结果写入 shared_scratchpad（乐观锁） ----
        scratchpad = state.get("shared_scratchpad")
        # 根据任务描述确定 scratchpad key，与 writer_node 预期的 key 对齐
        scratchpad_key = _resolve_scratchpad_key(agent_name, task_description)
        if scratchpad is not None:
            result_data = {"content": ai_content, "source": agent_name}
            current_version = scratchpad.get_version(scratchpad_key)
            success = scratchpad.set(scratchpad_key, result_data, expected_version=current_version)
            if success:
                logger.info(f"[{agent_name}] 结果已写入 shared_scratchpad['{scratchpad_key}']")
            else:
                # 版本冲突，重试一次
                logger.warning(
                    f"[{agent_name}] shared_scratchpad 写入 '{scratchpad_key}' "
                    f"失败（版本冲突），重试"
                )
                current_version = scratchpad.get_version(scratchpad_key)
                scratchpad.set(scratchpad_key, result_data, expected_version=current_version)

        # ---- 6. 构造 RESULT 消息 ----
        result_msg = {
            "type": "RESULT",
            "sender": agent_name,
            "receiver": "supervisor",
            "payload": {
                "task_id": task_id,
                "result": {"content": ai_content, "source": agent_name},
            },
        }

        # ---- 7. 从 inbox 移除已处理消息 ----
        updated_inbox = list(inbox)
        updated_inbox.pop(0)

        # ---- 8. 合并 new_message_batch ----
        existing_batch = state.get("new_message_batch") or []
        new_batch = existing_batch + [result_msg]

        logger.info(f"[{agent_name}] 任务完成，RESULT 已放入 new_message_batch")

        # ---- 9. 返回部分状态更新字典 ----
        final_progress = _update_progress(
            state, agent_name,
            status="IDLE",
            current_step="任务完成",
            detail=f"已完成: {task_description[:30]}",
            progress=100,
            last_log=f"结果写入 {scratchpad_key}",
        )

        return {
            "agent_private": {
                **state.get("agent_private", {}),
                agent_name: {
                    **state.get("agent_private", {}).get(agent_name, {}),
                    "inbox": updated_inbox,
                },
            },
            "new_message_batch": new_batch,
            # shared_scratchpad 已通过引用原地修改，但仍需返回以触发状态合并
            "shared_scratchpad": scratchpad,
            "agent_progress": final_progress,
        }

    return agent_node


# ============================================================
# 便捷工厂函数
# ============================================================

def create_adapted_researcher_node(
    plan_act_graph: Any,
    tools: Optional[List] = None,
) -> Callable[[dict], dict]:
    """创建适配后的 Researcher 节点

    将 Plan-Act 子图包装为 "researcher" Agent 节点，
    可直接传入 graph.add_node("researcher", adapted_researcher)

    Args:
        plan_act_graph: create_agent() 返回的 CompiledStateGraph 实例
        tools: 工具列表（保留扩展接口）

    Returns:
        适配后的节点函数
    """
    return wrap_plan_act_as_agent("researcher", plan_act_graph, tools)


def create_adapted_analyst_node(
    plan_act_graph: Any,
    tools: Optional[List] = None,
) -> Callable[[dict], dict]:
    """创建适配后的 Analyst 节点

    将 Plan-Act 子图包装为 "analyst" Agent 节点，
    可直接传入 graph.add_node("analyst", adapted_analyst)

    Args:
        plan_act_graph: create_agent() 返回的 CompiledStateGraph 实例
        tools: 工具列表（保留扩展接口）

    Returns:
        适配后的节点函数
    """
    return wrap_plan_act_as_agent("analyst", plan_act_graph, tools)
