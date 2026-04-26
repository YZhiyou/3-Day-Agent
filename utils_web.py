import json
import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage

USER_SESSIONS_DIR = "./data/user_sessions"
CHECKPOINTS_DB = "./data/checkpoints.db"


def _ensure_sessions_dir():
    """确保会话记录目录存在。"""
    os.makedirs(USER_SESSIONS_DIR, exist_ok=True)


def _sessions_file(user_id: str) -> str:
    """获取用户会话记录文件路径。"""
    _ensure_sessions_dir()
    return os.path.join(USER_SESSIONS_DIR, f"{user_id}_sessions.json")


def _record_session(user_id: str, session_id: str, title: str = None) -> None:
    """
    记录会话到用户会话列表中。
    维护 data/user_sessions/{user_id}_sessions.json，记录 session_id、标题和创建时间。
    """
    file_path = _sessions_file(user_id)
    sessions = []
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            sessions = json.load(f)
    # 避免重复记录
    if not any(s.get("session_id") == session_id for s in sessions):
        sessions.append({
            "session_id": session_id,
            "title": title,
            "created": datetime.now().isoformat()
        })
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)


def _update_session_title(user_id: str, session_id: str, title: str) -> None:
    """
    更新已有会话的标题。
    """
    file_path = _sessions_file(user_id)
    if not os.path.exists(file_path):
        return
    with open(file_path, "r", encoding="utf-8") as f:
        sessions = json.load(f)
    for s in sessions:
        if s.get("session_id") == session_id:
            s["title"] = title
            break
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)


def _load_sessions(user_id: str) -> List[Dict[str, Any]]:
    """
    读取用户的所有会话记录，按创建时间倒序排列（最新的在最上面）。
    从 data/user_sessions/{user_id}_sessions.json 读取。
    """
    file_path = _sessions_file(user_id)
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        sessions = json.load(f)
    sessions.sort(key=lambda s: s.get("created", ""), reverse=True)
    return sessions


def _delete_session_record(user_id: str, session_id: str) -> None:
    """
    从 JSON 会话列表和 SQLite checkpoints 中删除会话记录。
    """
    # 1. 从 JSON 中删除
    file_path = _sessions_file(user_id)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            sessions = json.load(f)
        sessions = [s for s in sessions if s.get("session_id") != session_id]
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)

    # 2. 从 SQLite checkpoints 中删除对应 thread 记录
    if os.path.exists(CHECKPOINTS_DB):
        try:
            conn = sqlite3.connect(CHECKPOINTS_DB)
            cursor = conn.cursor()
            # SqliteSaver 使用的表（langgraph 默认）
            cursor.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?",
                (session_id,)
            )
            cursor.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id = ?",
                (session_id,)
            )
            conn.commit()
            conn.close()
        except Exception:
            # 忽略数据库清理错误（如表不存在）
            pass


def _load_chat_history(agent, session_id: str) -> List[Dict[str, str]]:
    """
    从 agent.get_state 获取消息并转为可显示的聊天历史格式。
    返回: list of dicts: {"role": "user"/"assistant", "content": ...}
    """
    try:
        config = {"configurable": {"thread_id": session_id}}
        state = agent.get_state(config)
        if state is None:
            return []
        values = getattr(state, "values", None)
        if not values:
            return []
        messages = values.get("messages", [])
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": str(msg.content)})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": str(msg.content)})
        return history
    except Exception:
        return []
