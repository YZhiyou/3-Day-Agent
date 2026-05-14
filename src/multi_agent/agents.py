"""多智能体执行节点

实现三个专业 Agent 的 LangGraph 执行节点：
1. researcher_node — 文献检索 Agent
2. analyst_node    — 文献分析 Agent
3. writer_node     — 报告撰写 Agent

通用模板 generic_agent_step 封装了「取 inbox → 处理 → 回传 RESULT」的标准循环。
writer_node 因需直接写入 messages 而走独立逻辑。
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage

from src.multi_agent.multi_agent_state import MultiAgentState

logger = logging.getLogger(__name__)


def _update_progress(state: dict, agent_name: str, **kwargs) -> dict:
    """返回更新后的完整 agent_progress 字典"""
    progress = dict(state.get("agent_progress") or {})
    agent_info = dict(progress.get(agent_name) or {})
    agent_info.update(kwargs)
    progress[agent_name] = agent_info
    return progress


# ============================================================
# 通用 Agent 循环模板
# ============================================================

def generic_agent_step(
    state: dict,
    agent_name: str,
    capability_handler: Callable[[dict, dict], dict],
) -> dict:
    """通用 Agent 执行步骤

    流程：
    1. 从 agent_private[agent_name]["inbox"] 取出第一条消息
    2. 若 inbox 为空，直接返回空字典（无需更新状态）
    3. 调用 capability_handler 执行具体任务逻辑
    4. 构造 RESULT 消息放入 new_message_batch
    5. 从 inbox 中移除已处理消息
    6. 返回部分状态更新字典

    Args:
        state: LangGraph 全局状态字典
        agent_name: 智能体名称，如 "researcher"
        capability_handler: 具体能力处理函数，签名为 (state, message) -> result_payload_dict

    Returns:
        部分状态更新字典，或空字典（inbox 为空时）
    """
    # 获取 inbox
    inbox: List[dict] = (
        state.get("agent_private", {})
        .get(agent_name, {})
        .get("inbox", [])
    )

    if not inbox:
        logger.debug(f"[{agent_name}] inbox 为空，跳过执行")
        return {}

    # 取出第一条消息
    message: dict = inbox[0]
    logger.info(
        f"[{agent_name}] 处理消息: type={message.get('type')}, "
        f"sender={message.get('sender')}, task_id={message.get('payload', {}).get('task_id', '')}"
    )

    # 调用具体能力处理函数
    result_payload: dict = capability_handler(state, message)

    # 构造 RESULT 消息
    result_msg: dict = {
        "type": "RESULT",
        "sender": agent_name,
        "receiver": "supervisor",
        "payload": {
            "task_id": message.get("payload", {}).get("task_id", ""),
            "result": result_payload,
        },
    }

    # 从 inbox 中移除已处理消息（深拷贝后 pop，避免原地修改）
    updated_inbox = list(inbox)
    updated_inbox.pop(0)

    # 构造 agent_private 更新
    # 需要保留其他 agent 的私有状态，只更新当前 agent 的 inbox
    agent_private_update = {
        agent_name: {
            **state.get("agent_private", {}).get(agent_name, {}),
            "inbox": updated_inbox,
        }
    }

    # 合并 new_message_batch：追加而非覆盖
    existing_batch = state.get("new_message_batch") or []
    new_batch = existing_batch + [result_msg]

    logger.info(f"[{agent_name}] 任务完成，RESULT 已放入 new_message_batch")

    # 处理完成后
    final_progress = _update_progress(
        state, agent_name,
        status="IDLE",
        current_step="任务完成",
        detail=f"已处理: {message.get('payload', {}).get('description', '')[:30]}",
        progress=100,
        last_log=f"结果已发送",
    )

    return {
        "agent_private": {
            **state.get("agent_private", {}),
            **agent_private_update,
        },
        "new_message_batch": new_batch,
        "shared_scratchpad": state.get("shared_scratchpad"),
        "agent_progress": final_progress,
    }


# ============================================================
# Researcher Agent — 文献检索
# ============================================================

def _handle_retrieve(state: dict, message: dict) -> dict:
    """Researcher 能力处理函数：模拟文献检索

    从消息 payload 中提取查询描述，返回模拟检索结果，
    并将结果同步写入 shared_scratchpad 供其他 Agent 读取。

    Args:
        state: 全局状态
        message: inbox 中的消息字典

    Returns:
        检索结果字典
    """
    payload = message.get("payload", {})
    # 兼容 description 和 query 两种字段名
    query = payload.get("description") or payload.get("query") or ""

    logger.info(f"[researcher] 执行检索任务: {query}")

    # 模拟检索结果（通用占位）
    result: dict = {
        "docs": [
            {
                "title": "[模拟模式] 检索文档 1",
                "summary": "此为模拟检索结果。实际部署时由 Plan-Act 子图执行真实检索。",
                "relevance": 0.95,
            },
            {
                "title": "[模拟模式] 检索文档 2",
                "summary": "此为模拟检索结果。实际部署时由 Plan-Act 子图执行真实检索。",
                "relevance": 0.88,
            },
        ],
        "source": "mock_mode",
    }

    # 将检索结果写入 shared_scratchpad（乐观锁）
    scratchpad = state.get("shared_scratchpad")
    if scratchpad is not None:
        current_version = scratchpad.get_version("retrieved_docs")
        success = scratchpad.set("retrieved_docs", result, expected_version=current_version)
        if success:
            logger.info("[researcher] 检索结果已写入 shared_scratchpad['retrieved_docs']")
        else:
            logger.warning("[researcher] shared_scratchpad 写入失败（版本冲突），将重试")
            # 重试一次：获取最新版本号
            current_version = scratchpad.get_version("retrieved_docs")
            scratchpad.set("retrieved_docs", result, expected_version=current_version)

    return result


def researcher_node(state: dict) -> dict:
    """Researcher 执行节点：文献检索

    Args:
        state: LangGraph 全局状态

    Returns:
        部分状态更新字典
    """
    return generic_agent_step(state, "researcher", _handle_retrieve)


# ============================================================
# Analyst Agent — 文献分析
# ============================================================

def _handle_analyze(state: dict, message: dict) -> dict:
    """Analyst 能力处理函数：根据任务类型进行文献分析

    从 shared_scratchpad 读取 researcher 写入的检索结果，
    根据消息 payload 中的 description 判断分析类型，
    将分析结论写回 shared_scratchpad。

    Args:
        state: 全局状态
        message: inbox 中的消息字典

    Returns:
        分析结果字典
    """
    payload = message.get("payload", {})
    description: str = payload.get("description", "")

    logger.info(f"[analyst] 执行分析任务: {description}")

    # 从 shared_scratchpad 读取检索结果
    scratchpad = state.get("shared_scratchpad")
    retrieved_docs = scratchpad.get("retrieved_docs") if scratchpad else None

    # 根据描述中的关键词判断分析类型
    if "论点" in description or "提取" in description:
        # 论点提取
        result: dict = {
            "analysis_type": "claim_extraction",
            "claims": [
                {
                    "claim": "[模拟模式] 核心论点 1",
                    "evidence": "模拟数据",
                    "confidence": 0.90,
                },
                {
                    "claim": "[模拟模式] 核心论点 2",
                    "evidence": "模拟数据",
                    "confidence": 0.85,
                },
            ],
        }
        scratchpad_key = "claims"
    elif "逻辑" in description or "批判" in description:
        # 逻辑批判
        result = {
            "analysis_type": "logical_critique",
            "critique": [
                {
                    "target": "[模拟模式] 论点",
                    "issue": "模拟逻辑批判",
                    "severity": "medium",
                },
            ],
        }
        scratchpad_key = "critique"
    else:
        # 通用分析
        result = {
            "analysis_type": "general",
            "summary": "[模拟模式] 此为通用分析占位结果。实际部署时由 Plan-Act 子图执行真实分析。",
            "key_findings": [
                "[模拟模式] 核心发现 1",
                "[模拟模式] 核心发现 2",
            ],
        }
        scratchpad_key = "analysis"

    # 将分析结果写入 shared_scratchpad
    if scratchpad is not None:
        current_version = scratchpad.get_version(scratchpad_key)
        success = scratchpad.set(scratchpad_key, result, expected_version=current_version)
        if success:
            logger.info(f"[analyst] 分析结果已写入 shared_scratchpad['{scratchpad_key}']")
        else:
            logger.warning(f"[analyst] shared_scratchpad 写入 '{scratchpad_key}' 失败（版本冲突），重试")
            current_version = scratchpad.get_version(scratchpad_key)
            scratchpad.set(scratchpad_key, result, expected_version=current_version)

    return result


def analyst_node(state: dict) -> dict:
    """Analyst 执行节点：文献分析

    Args:
        state: LangGraph 全局状态

    Returns:
        部分状态更新字典
    """
    return generic_agent_step(state, "analyst", _handle_analyze)


# ============================================================
# Writer Agent — 报告撰写
# ============================================================

def writer_node(state: dict) -> dict:
    """Writer 执行节点：生成最终评审报告

    不使用 generic_agent_step，因为 writer 需要直接将
    最终报告以 AIMessage 形式写入 messages。

    流程：
    1. 检查 inbox，取出 TASK 消息
    2. 从 shared_scratchpad 汇总所有结构化结论
    3. 生成 Markdown 格式的评审报告
    4. 以 AIMessage 写入 messages
    5. 构造 RESULT 消息回传 supervisor
    6. 返回部分状态更新字典

    Args:
        state: LangGraph 全局状态

    Returns:
        部分状态更新字典，包含 messages（add_messages 合并）
    """
    agent_name = "writer"

    # 获取 inbox
    inbox: List[dict] = (
        state.get("agent_private", {})
        .get(agent_name, {})
        .get("inbox", [])
    )

    if not inbox:
        logger.debug(f"[{agent_name}] inbox 为空，跳过执行")
        return {}

    # 取出第一条 TASK 消息
    message: dict = inbox[0]
    logger.info(
        f"[{agent_name}] 处理消息: type={message.get('type')}, "
        f"sender={message.get('sender')}"
    )

    # 读取 scratchpad 数据（兼容模拟节点和适配节点两种 key 格式）
    scratchpad = state.get("shared_scratchpad")

    # 模拟节点写入的 key
    retrieved_docs = scratchpad.get("retrieved_docs") if scratchpad else None
    claims = scratchpad.get("claims") if scratchpad else None
    critique = scratchpad.get("critique") if scratchpad else None
    analysis = scratchpad.get("analysis") if scratchpad else None

    # 适配节点写入的 key（fallback）
    if retrieved_docs is None and scratchpad:
        researcher_result = scratchpad.get("researcher_result")
        if researcher_result:
            # 适配节点的结果是 {"content": "...", "source": "researcher"}
            retrieved_docs = researcher_result

    if (claims is None and critique is None and analysis is None) and scratchpad:
        analyst_result = scratchpad.get("analyst_result")
        if analyst_result:
            # 适配节点的 analyst 结果统一放在 analysis 中
            analysis = analyst_result

    # 生成 Markdown 格式的评审报告
    final_report = _build_report(retrieved_docs, claims, critique, analysis)

    # 以 AIMessage 写入 messages（LangGraph 用 add_messages 合并，传入列表即可）
    report_message = AIMessage(content=final_report)

    # 构造 RESULT 消息
    result_msg: dict = {
        "type": "RESULT",
        "sender": agent_name,
        "receiver": "supervisor",
        "payload": {
            "task_id": message.get("payload", {}).get("task_id", ""),
            "result": {"report_length": len(final_report), "status": "completed"},
        },
    }

    # 从 inbox 中移除已处理消息
    updated_inbox = list(inbox)
    updated_inbox.pop(0)

    # 构造 agent_private 更新
    agent_private_update = {
        agent_name: {
            **state.get("agent_private", {}).get(agent_name, {}),
            "inbox": updated_inbox,
        }
    }

    # 合并 new_message_batch
    existing_batch = state.get("new_message_batch") or []
    new_batch = existing_batch + [result_msg]

    logger.info(f"[{agent_name}] 评审报告已生成，长度={len(final_report)} 字符")

    final_progress = _update_progress(
        state, "writer",
        status="IDLE",
        current_step="报告生成完毕",
        detail="评审报告已输出",
        progress=100,
        last_log=f"报告长度: {len(final_report)} 字符",
    )

    return {
        "messages": [report_message],
        "agent_private": {
            **state.get("agent_private", {}),
            **agent_private_update,
        },
        "new_message_batch": new_batch,
        "agent_progress": final_progress,
    }


def _is_adapter_format(data: Any) -> bool:
    """判断数据是否为适配节点格式（{"content": "...", "source": "..."})"""
    return isinstance(data, dict) and "content" in data and "source" in data


def _build_report(
    retrieved_docs: Optional[dict],
    claims: Optional[dict],
    critique: Optional[dict],
    analysis: Optional[dict],
) -> str:
    """根据 scratchpad 中的结构化数据构建 Markdown 评审报告

    Args:
        retrieved_docs: 检索到的文档列表（模拟节点格式或适配节点格式）
        claims: 提取的论点（模拟节点格式或适配节点格式）
        critique: 逻辑批判（模拟节点格式或适配节点格式）
        analysis: 通用分析结果（模拟节点格式或适配节点格式）

    Returns:
        Markdown 格式的评审报告字符串
    """
    sections: List[str] = []

    # 报告标题
    sections.append("# 文献评审报告\n")

    # 检索文献概览
    sections.append("## 一、检索文献概览\n")
    if retrieved_docs:
        if _is_adapter_format(retrieved_docs):
            # 适配节点格式：直接显示 content 文本
            sections.append(f"{retrieved_docs['content']}\n")
        elif retrieved_docs.get("docs"):
            # 模拟节点格式：结构化文档列表
            for i, doc in enumerate(retrieved_docs["docs"], 1):
                sections.append(f"### {i}. {doc.get('title', '未知标题')}\n")
                sections.append(f"{doc.get('summary', '无摘要')}\n")
            sections.append(f"> 数据来源：{retrieved_docs.get('source', '未知')}\n")
        else:
            sections.append("暂无检索结果。\n")
    else:
        sections.append("暂无检索结果。\n")

    # 论点提取
    sections.append("## 二、核心论点提取\n")
    if claims:
        if _is_adapter_format(claims):
            # 适配节点格式：直接显示 content 文本
            sections.append(f"{claims['content']}\n")
        elif claims.get("claims"):
            # 模拟节点格式：结构化论点列表
            for i, claim in enumerate(claims["claims"], 1):
                sections.append(
                    f"- **论点 {i}**：{claim.get('claim', '未知')}  \n"
                    f"  - 依据：{claim.get('evidence', '未知')}  \n"
                    f"  - 置信度：{claim.get('confidence', 'N/A')}"
                )
        else:
            sections.append("暂无论点提取结果。\n")
    else:
        sections.append("暂无论点提取结果。\n")

    # 逻辑批判
    sections.append("## 三、逻辑批判与问题识别\n")
    if critique:
        if _is_adapter_format(critique):
            # 适配节点格式：直接显示 content 文本
            sections.append(f"{critique['content']}\n")
        elif critique.get("critique"):
            # 模拟节点格式：结构化批判列表
            for i, item in enumerate(critique["critique"], 1):
                severity_label = {"high": "高", "medium": "中", "low": "低"}.get(
                    item.get("severity", ""), item.get("severity", "")
                )
                sections.append(
                    f"- **问题 {i}**（严重度：{severity_label}）：  \n"
                    f"  - 目标：{item.get('target', '未知')}  \n"
                    f"  - 问题：{item.get('issue', '未知')}"
                )
        else:
            sections.append("暂无逻辑批判结果。\n")
    else:
        sections.append("暂无逻辑批判结果。\n")

    # 综合评述
    sections.append("## 四、综合评述\n")
    if analysis:
        if _is_adapter_format(analysis):
            # 适配节点格式：直接显示 content 文本
            sections.append(f"{analysis['content']}\n")
        elif analysis.get("summary"):
            # 模拟节点格式：结构化分析
            sections.append(f"{analysis.get('summary', '暂无综合分析。')}\n")
            key_findings = analysis.get("key_findings", [])
            if key_findings:
                sections.append("**核心发现：**\n")
                for finding in key_findings:
                    sections.append(f"- {finding}")
        else:
            # 基于已有 claims 和 critique 生成综合评述
            _build_summary_from_claims_critique(sections, claims, critique)
    else:
        # 基于已有 claims 和 critique 生成综合评述
        _build_summary_from_claims_critique(sections, claims, critique)

    # 结论与建议
    sections.append("## 五、结论与建议\n")
    # 基于实际数据动态生成结论
    has_content = any([retrieved_docs, claims, critique, analysis])
    if has_content:
        sections.append("基于以上分析，主要发现与建议如下：\n\n")
        if claims:
            if _is_adapter_format(claims):
                # 从 claims content 中提取要点作为建议
                sections.append("**核心论点方面**：已完成论点提取与分析。\n\n")
            elif claims.get("claims"):
                sections.append("**核心论点方面**：\n")
                for c in claims["claims"][:3]:
                    sections.append(f"- {c.get('claim', '')}\n")
                sections.append("\n")
        if critique:
            if _is_adapter_format(critique):
                sections.append("**逻辑分析方面**：已完成逻辑批判与问题识别。\n\n")
            elif critique.get("critique"):
                sections.append("**需要关注的问题**：\n")
                for c in critique["critique"][:3]:
                    sections.append(f"- {c.get('issue', '')}\n")
                sections.append("\n")
        sections.append("建议基于以上分析进一步深入研究相关主题。\n")
    else:
        sections.append("暂无足够数据生成结论，请确保各分析任务已正确完成。\n")

    return "\n".join(sections)


def _build_summary_from_claims_critique(
    sections: List[str],
    claims: Optional[dict],
    critique: Optional[dict],
) -> None:
    """基于已有的 claims 和 critique 生成综合评述，直接追加到 sections 列表"""
    summary_points: List[str] = []
    if claims and not _is_adapter_format(claims) and claims.get("claims"):
        for c in claims["claims"]:
            summary_points.append(f"论点：{c.get('claim', '未知')}（置信度 {c.get('confidence', 'N/A')}）")
    if critique and not _is_adapter_format(critique) and critique.get("critique"):
        for c in critique["critique"]:
            summary_points.append(f"问题：{c.get('issue', '未知')}（严重度 {c.get('severity', 'N/A')}）")
    if summary_points:
        sections.append("基于已有论点提取与逻辑批判，综合要点如下：\n")
        for pt in summary_points:
            sections.append(f"- {pt}")
    else:
        sections.append("暂无综合分析结果。\n")
