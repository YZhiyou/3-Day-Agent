from typing import List, Callable, Optional, TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import json
import os
import sqlite3


# ---------------- Pydantic 模型 ----------------

class PlanStep(BaseModel):
    action: str = Field(description="要执行的工具名称")
    input: str = Field(description="传给工具的输入参数（字符串，多参数工具请用 JSON 格式）")
    expected: str = Field(description="期望的输出结果")


class Plan(BaseModel):
    steps: list[PlanStep] = Field(description="执行步骤列表")
    overall_goal: str = Field(description="整体目标描述")


class StepResult(BaseModel):
    step_index: int
    success: bool
    output: str
    deviation: bool


class ComplexityJudgment(BaseModel):
    is_complex: bool = Field(description="是否为复杂任务")
    reason: str = Field(description="判断理由")


class DeviationCheck(BaseModel):
    deviation: bool = Field(description="是否严重偏离预期")
    reason: str = Field(description="判断理由")


# ---------------- 状态定义 ----------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: Optional[Plan]
    current_step_index: int
    step_results: List[StepResult]
    need_replan: bool
    is_complex: Optional[bool]
    final_summary: Optional[str]
    replan_count: int


# ---------------- 工具函数 ----------------

def _get_last_user_content(messages: list) -> str:
    """从消息列表中提取最后一条用户消息的内容。"""
    for msg in reversed(messages):
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                return msg.get("content", "")
        elif isinstance(msg, HumanMessage):
            return msg.content
        elif hasattr(msg, "type") and msg.type == "human":
            return msg.content
    return ""


def _build_tool_descriptions(tools: List[BaseTool]) -> str:
    """生成工具描述文本。"""
    lines = []
    for t in tools:
        lines.append(f"- {t.name}: {t.description or '无描述'}")
    return "\n".join(lines)


# ---------------- 内部 ReAct Agent 创建器 ----------------

def _build_react_agent(
    tools: List[BaseTool],
    system_prompt: str,
    checkpointer: SqliteSaver,
    llm_model: str,
    max_iterations: int,
):
    """创建原 ReAct Agent（作为子图嵌入）。"""
    from langchain.agents import create_agent as _create_langchain_agent

    agent = _create_langchain_agent(
        llm_model,
        tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
    )
    agent = agent.with_config({"recursion_limit": max_iterations * 10})
    return agent


# ---------------- Plan-Act 节点函数 ----------------

def _classify_complexity(state: AgentState, llm, tools: List[BaseTool]) -> dict:
    """判断用户请求的复杂度。"""
    user_content = _get_last_user_content(state["messages"])
    if not user_content:
        return {"is_complex": False}

    tool_desc = _build_tool_descriptions(tools)

    prompt = f"""你是一个任务复杂度判断助手。请分析用户的请求，判断它是否需要复杂的计划-执行流程。

可用工具：
{tool_desc}

判断标准：
- 复杂任务：需要调用 ≥2 个不同工具，或需要多步依赖（某一步的输出是下一步的输入），或涉及多个子任务
- 简单任务：简单闲聊、单次检索、仅读/写用户信息、打招呼、简单问答

用户请求：{user_content}

请输出判断结果。"""

    structured_llm = llm.with_structured_output(ComplexityJudgment)
    result = structured_llm.invoke([SystemMessage(content=prompt)])
    return {"is_complex": result.is_complex}


def _generate_plan(state: AgentState, llm, tools: List[BaseTool]) -> dict:
    """为复杂任务生成执行计划。"""
    user_content = _get_last_user_content(state["messages"])
    tool_desc = _build_tool_descriptions(tools)

    prompt = f"""你是一个任务规划助手。请根据用户请求制定一个详细的执行计划。

可用工具：
{tool_desc}

要求：
1. 将任务拆解为具体的步骤
2. 每步必须指定 action（工具名）、input（工具参数）、expected（期望输出）
3. 对于多参数工具（如 save_user_info），input 请使用 JSON 格式，例如 {{"key": "name", "value": "Alice"}}
4. 步骤之间可以有依赖关系
5. 如果任务无法完成，也要给出计划并说明限制

用户请求：{user_content}

请输出执行计划。"""

    structured_llm = llm.with_structured_output(Plan)
    plan = structured_llm.invoke([SystemMessage(content=prompt)])

    replan_count = state.get("replan_count", 0)
    return {
        "plan": plan,
        "current_step_index": 0,
        "step_results": [],
        "need_replan": False,
        "replan_count": replan_count + 1,
    }


def _execute_step(state: AgentState, tools: List[BaseTool]) -> dict:
    """执行当前计划步骤。"""
    plan = state.get("plan")
    current_idx = state.get("current_step_index", 0)

    if not plan or current_idx >= len(plan.steps):
        return {}

    step = plan.steps[current_idx]
    tool_map = {t.name: t for t in tools}

    if step.action not in tool_map:
        result = StepResult(
            step_index=current_idx,
            success=False,
            output=f"未找到工具: {step.action}",
            deviation=True,
        )
    else:
        tool = tool_map[step.action]
        try:
            # 尝试将 input 解析为 JSON，支持多参数工具
            parsed_input = json.loads(step.input)
            if isinstance(parsed_input, dict):
                output = tool.invoke(parsed_input)
            else:
                output = tool.invoke(step.input)
            result = StepResult(
                step_index=current_idx,
                success=True,
                output=str(output),
                deviation=False,
            )
        except json.JSONDecodeError:
            try:
                output = tool.invoke(step.input)
                result = StepResult(
                    step_index=current_idx,
                    success=True,
                    output=str(output),
                    deviation=False,
                )
            except Exception as e:
                result = StepResult(
                    step_index=current_idx,
                    success=False,
                    output=str(e),
                    deviation=True,
                )
        except Exception as e:
            result = StepResult(
                step_index=current_idx,
                success=False,
                output=str(e),
                deviation=True,
            )

    step_results = list(state.get("step_results", []))
    step_results.append(result)

    return {
        "step_results": step_results,
        "current_step_index": current_idx + 1,
    }


def _check_progress(state: AgentState, llm) -> dict:
    """检查执行进度，判断是否需要重规划或继续执行。"""
    step_results = state.get("step_results", [])
    plan = state.get("plan")

    if not step_results:
        return {"need_replan": False}

    last_result = step_results[-1]
    if not last_result.success:
        return {"need_replan": True}

    # 使用 LLM 判断是否偏离预期
    if plan and last_result.step_index < len(plan.steps):
        step = plan.steps[last_result.step_index]
        prompt = f"""判断以下步骤执行结果是否严重偏离预期。

步骤目标：{step.expected}
实际结果：{last_result.output}

是否严重偏离？如果结果完全不符合预期或无法继续后续步骤，回答 True；否则回答 False。"""

        structured_llm = llm.with_structured_output(DeviationCheck)
        try:
            check = structured_llm.invoke([SystemMessage(content=prompt)])
        except Exception:
            check = DeviationCheck(deviation=False, reason="检查失败，默认不偏离")

        step_results = list(step_results)
        step_results[-1] = StepResult(
            step_index=last_result.step_index,
            success=last_result.success,
            output=last_result.output,
            deviation=check.deviation,
        )

        return {
            "need_replan": check.deviation,
            "step_results": step_results,
        }

    return {"need_replan": False}


def _summarize_results(state: AgentState, llm) -> dict:
    """汇总执行结果，生成最终回答。"""
    messages = state["messages"]
    step_results = state.get("step_results", [])
    plan = state.get("plan")

    user_content = _get_last_user_content(messages)

    outputs = "\n\n".join([
        f"步骤 {r.step_index + 1}（{plan.steps[r.step_index].action if plan else 'unknown'}）：\n{r.output}"
        for r in step_results
    ])

    prompt = f"""根据以下执行结果，生成对用户请求的完整回答。

用户原始请求：{user_content}

执行计划目标：{plan.overall_goal if plan else ''}

执行结果：
{outputs}

请综合以上信息，给出简洁明了的最终回答。用中文回复。"""

    response = llm.invoke([SystemMessage(content=prompt)])

    return {
        "messages": [AIMessage(content=response.content)],
        "final_summary": response.content,
    }


# ---------------- 条件路由 ----------------

def _route_by_complexity(state: AgentState) -> str:
    if state.get("is_complex"):
        return "generate_plan"
    return "react_agent"


def _route_after_check(state: AgentState) -> str:
    if state.get("need_replan"):
        replan_count = state.get("replan_count", 0)
        if replan_count >= 3:
            # 超过最大重规划次数，强制汇总已有结果
            return "summarize_results"
        return "generate_plan"
    plan = state.get("plan")
    current = state.get("current_step_index", 0)
    if plan and current >= len(plan.steps):
        return "summarize_results"
    return "execute_step"


# ---------------- 对外接口 ----------------

def create_agent(
    tools: List[BaseTool],
    get_session_history: Callable[[str], object] = None,
    llm_model: str = "deepseek-chat",
    temperature: float = 0.7,
    max_iterations: int = 5,
):
    """
    创建配置了工具、记忆的 Agent 执行器（Plan-Act 模式）。

    Args:
        tools: Agent 可用的工具列表。
        get_session_history: 获取会话历史的工厂函数（来自 memory_manager，
            在 LangChain 1.2.15 中保留此参数以保持接口兼容，实际短期记忆
            由内部 SqliteSaver checkpointer 自动管理）。
        llm_model: DeepSeek 模型名。
        temperature: LLM 温度参数。
        max_iterations: Agent 最大推理步数（映射为 LangGraph 的 recursion_limit）。

    Returns:
        可直接调用 invoke() / stream() 的 CompiledStateGraph 实例。
        调用方式：
            agent.invoke(
                {"messages": [{"role": "user", "content": "..."}]},
                {"configurable": {"thread_id": "session_id"}}
            )
    """

    system_prompt = """你是一个个人知识库助手，能帮助用户管理信息和回答基于知识库的问题。

你的能力：
- 搜索用户的个人知识库（当用户询问某个知识点、文档内容时使用）
- 保存用户的个人信息（姓名、偏好等）
- 查询用户之前保存的个人信息

行为准则：
1. 当用户询问关于他们自己的情况时，先用 get_user_info 查询是否已保存相关信息。
2. 当用户询问某个概念、定义或需要参考资料时，用 retrieve_documents 搜索知识库。
3. 不要编造信息。知识库中没有的内容，诚实地告诉用户。
4. 回答简洁明了，用中文回复。"""

    # 检查工具列表中是否存在联网搜索工具，若存在则追加联网规则
    tool_names = [t.name for t in tools]
    if "web_search" in tool_names:
        system_prompt += """
【联网搜索规则】
- 当用户询问实时信息、最新新闻、当前事件、股票价格、天气等动态内容时，使用 web_search 工具搜索互联网。
- 知识库中没有的实时信息也应尝试联网搜索。
- 搜索结果通常包含来源 URL，请在回答中引用来源网址。
- 如果用户明确说"不要联网搜索"或"只查我的知识库"，则跳过联网搜索。
- 对搜索结果进行归纳总结，不要直接堆砌原始内容。
"""

    # 确保 data 目录存在
    os.makedirs("./data", exist_ok=True)
    conn = sqlite3.connect("./data/checkpoints.db", check_same_thread=False)
    memory = SqliteSaver(conn)

    # 创建内部 ReAct Agent（原逻辑，作为子图嵌入）
    react_agent = _build_react_agent(
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=memory,
        llm_model=llm_model,
        max_iterations=max_iterations,
    )

    # 初始化 LLM
    llm = init_chat_model(llm_model, temperature=temperature)

    # 构建 Plan-Act 外层图
    builder = StateGraph(AgentState)

    builder.add_node("classify_complexity", lambda state: _classify_complexity(state, llm, tools))
    builder.add_node("react_agent", react_agent)
    builder.add_node("generate_plan", lambda state: _generate_plan(state, llm, tools))
    builder.add_node("execute_step", lambda state: _execute_step(state, tools))
    builder.add_node("check_progress", lambda state: _check_progress(state, llm))
    builder.add_node("summarize_results", lambda state: _summarize_results(state, llm))

    builder.set_entry_point("classify_complexity")
    builder.add_conditional_edges("classify_complexity", _route_by_complexity)
    builder.add_edge("react_agent", END)
    builder.add_edge("generate_plan", "execute_step")
    builder.add_edge("execute_step", "check_progress")
    builder.add_conditional_edges("check_progress", _route_after_check)
    builder.add_edge("summarize_results", END)

    graph = builder.compile(checkpointer=memory)

    return graph
