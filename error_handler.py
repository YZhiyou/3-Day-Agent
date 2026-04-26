import re
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser

# ---------------- 术语映射表（可扩展） ----------------

TERM_MAP: Dict[str, str] = {
    "sam": "Sharpness Aware Minimization",
    "gpt": "Generative Pre-trained Transformer",
    "llm": "Large Language Model",
    "rag": "Retrieval-Augmented Generation",
    "ppo": "Proximal Policy Optimization",
    "dqn": "Deep Q-Network",
    "gan": "Generative Adversarial Network",
    "cnn": "Convolutional Neural Network",
    "rnn": "Recurrent Neural Network",
    "transformer": "Transformer",
    "bert": "Bidirectional Encoder Representations from Transformers",
    "vit": "Vision Transformer",
    "clip": "Contrastive Language-Image Pre-training",
    "diffusion": "Diffusion Model",
    "rlhf": "Reinforcement Learning from Human Feedback",
    "sft": "Supervised Fine-Tuning",
    "dpo": "Direct Preference Optimization",
    "moe": "Mixture of Experts",
    "lora": "Low-Rank Adaptation",
    "qlora": "Quantized Low-Rank Adaptation",
}

# ---------------- 负面信号词（用户纠正意图） ----------------

NEGATIVE_SIGNALS: List[str] = [
    "不对",
    "不是",
    "错了",
    "改成",
    "应该是",
    "换一个",
    "不对了",
    "不正确",
    "你理解错了",
    "我说的是",
    "我的意思是",
    "你搞错了",
    "并非如此",
    "别搞混了",
]

# ---------------- Pydantic 模型 ----------------

class ErrorHandlerResult(BaseModel):
    needs_clarification: bool = Field(description="是否需要追问或澄清")
    clarification_type: str = Field(
        description="澄清类型: confidence_low / disambiguation / slot_filling / correction / fallback / none"
    )
    clarification_message: str = Field(description="待发送给用户的澄清消息，无需澄清时为空字符串")
    confidence: float = Field(description="对用户意图理解的置信度，范围 0.0-1.0")
    detected_slots: Dict[str, str] = Field(description="从用户输入中提取的槽位信息")
    slot_missing: List[str] = Field(description="缺失的关键槽位列表")
    detected_correction: bool = Field(description="是否检测到用户纠正意图")
    normalized_input: str = Field(description="术语归一化后的用户输入")


# ---------------- 内部工具函数 ----------------

def _normalize_terms(text: str) -> tuple[str, Dict[str, str]]:
    """术语归一化：将口语/别称/简写替换为标准术语。"""
    normalized = text
    replacements: Dict[str, str] = {}
    for term, standard in TERM_MAP.items():
        # 大小写不敏感，匹配词边界
        pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
        if pattern.search(normalized):
            replacements[term.upper()] = standard
            normalized = pattern.sub(standard, normalized)
    return normalized, replacements


def _check_negative_signals(text: str) -> bool:
    """检测用户输入中是否包含负面信号词（纠正意图）。"""
    return any(signal in text for signal in NEGATIVE_SIGNALS)


def _heuristic_confidence(user_input: str) -> float:
    """启发式估算置信度：输入越短、越模糊，置信度越低。"""
    stripped = user_input.strip()
    if len(stripped) <= 3:
        return 0.15
    # 模糊词汇
    vague_words = ["那个", "什么", "随便", "大概", "好像"]
    if any(w in stripped for w in vague_words):
        return 0.35
    # 仅包含标点或数字
    if re.match(r'^[\d\W]+$', stripped):
        return 0.1
    # 过短句
    if len(stripped) <= 8:
        return 0.5
    return 0.8


def _get_last_user_content(messages: list) -> str:
    """从消息列表中提取最后一条用户消息的内容。"""
    for msg in reversed(messages):
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                return msg.get("content", "")
        elif hasattr(msg, "type") and msg.type == "human":
            return msg.content
    return ""


def _get_last_assistant_content(messages: list) -> str:
    """从消息列表中提取最后一条助手消息的内容。"""
    for msg in reversed(messages):
        if isinstance(msg, dict):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        elif hasattr(msg, "type") and msg.type == "ai":
            return msg.content
    return ""


def _parse_confidence_tag(text: str) -> Optional[float]:
    """解析显式置信度标记，如 [confidence:0.3] 或 [置信度:0.5]。"""
    match = re.search(r'\[\s*(?:confidence|置信度)\s*[:：]\s*(\d(?:\.\d+)?)\s*\]', text, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        return min(max(val, 0.0), 1.0)
    return None


# ---------------- 主节点创建器 ----------------

def create_error_handler_node(llm):
    """
    创建容错处理节点，返回一个 LangGraph state_node 函数。

    职责：
    1. 术语归一化
    2. 检测用户纠正意图
    3. 估算/解析置信度
    4. 判断是否需要追问澄清（意图消歧、槽位补全）
    5. 最终兜底
    """

    def error_handler(state: dict) -> dict:
        messages = state.get("messages", [])
        last_user_msg = _get_last_user_content(messages)

        if not last_user_msg:
            return {}

        # 1. 术语归一化
        normalized_input, term_replacements = _normalize_terms(last_user_msg)

        # 2. 检测纠正意图
        detected_correction = _check_negative_signals(last_user_msg)

        # 3. 置信度：先尝试解析显式标记，再用启发式
        explicit_conf = _parse_confidence_tag(last_user_msg)
        heuristic_conf = _heuristic_confidence(last_user_msg)
        confidence = explicit_conf if explicit_conf is not None else heuristic_conf

        # 4. 快速规则层：极低置信度直接兜底（避免 LLM 调用）
        if confidence < 0.25:
            error_log = list(state.get("error_log", []))
            error_log.append({
                "type": "confidence_low",
                "input": last_user_msg,
                "confidence": confidence,
                "reason": "启发式置信度过低",
            })
            return {
                "pending_clarification": (
                    "抱歉，我暂时没能理解你的意思。"
                    "你能换一种方式描述一下吗？比如告诉我你想查哪篇论文，或者想了解什么概念。"
                ),
                "confidence": confidence,
                "error_log": error_log,
                "last_slots": state.get("last_slots", {}),
            }

        # 5. 快速规则层：检测到纠正意图，直接生成确认追问
        if detected_correction:
            last_assistant = _get_last_assistant_content(messages)
            error_log = list(state.get("error_log", []))
            error_log.append({
                "type": "correction",
                "input": last_user_msg,
                "confidence": confidence,
            })
            # 尝试提取用户纠正后的关键信息
            corrected = last_user_msg
            for signal in NEGATIVE_SIGNALS:
                corrected = corrected.replace(signal, "")
            corrected = corrected.strip("，。 \t")
            clarification = (
                f"抱歉我理解错了，你是说{corrected}对吗？"
                if corrected
                else "抱歉我理解错了，你能再明确一下你的需求吗？"
            )
            return {
                "pending_clarification": clarification,
                "confidence": confidence,
                "error_log": error_log,
                "last_slots": {},  # 纠正时清空槽位，等待重新确认
            }

        # 6. LLM 层：对边界情况做精细判断（意图消歧、槽位补全）
        parser = PydanticOutputParser(pydantic_object=ErrorHandlerResult)

        prompt = f"""你是一个对话容错助手。请分析用户的最新输入，判断是否需要追问澄清。

用户最新输入：{last_user_msg}
术语归一化后：{normalized_input}
已归一化术语映射：{term_replacements}

关键信息槽位定义：
- paper_title: 论文标题（当用户提及论文时使用）
- author: 作者姓名
- year: 发表年份
- concept: 概念/术语名称
- topic: 主题/话题
- query: 具体的搜索/查询内容

判断规则（严格按以下顺序）：
1. 如果用户输入非常模糊（如"那个什么"、"帮我找一下"、"关于那个"），needs_clarification=true, type=confidence_low
2. 如果用户同时提到多个可能不相关的意图且没有明确侧重（如"查一下论文和天气"），needs_clarification=true, type=disambiguation
3. 如果用户意图看起来是查论文/查概念，但缺少必要的具体信息（如"找那篇论文"但没给标题），needs_clarification=true, type=slot_filling
4. 否则 needs_clarification=false, type=none

澄清消息规范：
- confidence_low: 礼貌地请用户具体重述，例如"抱歉，我没太明白你的意思，你能更具体地描述一下吗？"
- disambiguation: 列出选项让用户选择，例如"你是想 A 还是 B？"
- slot_filling: 自然地追问缺失信息，例如"请问你指的是哪篇论文？能不能说一下大概的发表年份？"
- none: 空字符串

注意：
- 不要对正常的闲聊、清晰的查询进行过度追问
- 如果用户已经提供了足够明确的信息（即使简短），type=none
- 如果用户输入"SAM"这种缩写，已经被归一化为标准术语，不算模糊

{parser.get_format_instructions()}"""

        try:
            response = llm.invoke([SystemMessage(content=prompt)])
            result = parser.parse(response.content)
        except Exception:
            # 解析失败时回退到启发式结果
            needs_clarification = heuristic_conf < 0.4
            result = ErrorHandlerResult(
                needs_clarification=needs_clarification,
                clarification_type="confidence_low" if needs_clarification else "none",
                clarification_message=(
                    "抱歉，我暂时没能理解你的意思。你能换一种方式描述一下吗？"
                    if needs_clarification else ""
                ),
                confidence=heuristic_conf,
                detected_slots={},
                slot_missing=[],
                detected_correction=False,
                normalized_input=normalized_input,
            )

        # 7. 组装返回状态
        error_log = list(state.get("error_log", []))
        if result.needs_clarification:
            error_log.append({
                "type": result.clarification_type,
                "input": last_user_msg,
                "confidence": result.confidence,
                "message": result.clarification_message,
            })

        # 合并新旧槽位
        last_slots = dict(state.get("last_slots", {}))
        if result.detected_slots:
            last_slots.update(result.detected_slots)

        return {
            "pending_clarification": result.clarification_message if result.needs_clarification else None,
            "confidence": result.confidence if result.confidence else confidence,
            "error_log": error_log,
            "last_slots": last_slots,
        }

    return error_handler
