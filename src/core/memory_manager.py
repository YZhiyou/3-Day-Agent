import json
import os
from pathlib import Path
from typing import Optional, Dict

USER_INFO_DIR = "./data/user_info"
CHAT_HISTORY_DIR = "./data/chat_history"

# ---------------- 短期记忆 ----------------
from langchain_community.chat_message_histories import FileChatMessageHistory

def get_session_history(session_id: str) -> FileChatMessageHistory:
    """返回指定会话的历史管理器（会被RunnableWithMessageHistory调用）"""
    Path(CHAT_HISTORY_DIR).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(CHAT_HISTORY_DIR, f"{session_id}.json")
    return FileChatMessageHistory(file_path)

# ---------------- 长期用户记忆 ----------------
def _user_info_file(user_id: str) -> str:
    Path(USER_INFO_DIR).mkdir(parents=True, exist_ok=True)
    return os.path.join(USER_INFO_DIR, f"{user_id}.json")

def _load_user_data(user_id: str) -> Dict[str, str]:
    file = _user_info_file(user_id)
    if not os.path.exists(file):
        return {}
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_user_data(user_id: str, data: Dict[str, str]) -> None:
    file = _user_info_file(user_id)
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_user_info(user_id: str, key: str, value: str) -> None:
    data = _load_user_data(user_id)
    data[key] = value
    _save_user_data(user_id, data)

def get_user_info(user_id: str, key: str) -> Optional[str]:
    data = _load_user_data(user_id)
    return data.get(key)