import os
import sys
import io
import uuid
import logging
from typing import Optional

# Windows 下强制使用 UTF-8 处理标准输入输出，避免编码问题
if sys.platform == "win32":
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 清理可能干扰 API 连接的代理环境变量
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(k, None)

from langchain_core.messages import AIMessageChunk, HumanMessage, AIMessage

from retriever import build_retriever, build_rerank_retriever
from memory_manager import get_session_history
from tools import create_tools
from agent import create_agent
from kb_manager import add_file_to_kb, delete_file_from_kb, rebuild_kb, search_kb
from vector_store import load_vector_store, get_collection_stats

logger = logging.getLogger(__name__)

PERSIST_DIR = "./data/chroma"
DOCS_DIR = "./documents"


def print_help():
    print("""
可用命令:
  /login <用户名>         登录或切换用户
  /new                    开始新会话
  /switch <会话ID>        切换到已有会话
  /history                显示当前会话历史
  /userinfo               查看当前用户的长期记忆
  /kb add <file_path>     添加文件到知识库
  /kb delete <file_path>  从知识库删除文件
  /kb rebuild             重建知识库（需确认）
  /kb search <query>      搜索知识库
  /help                   显示本帮助
  /quit   或 /exit        退出程序
""")


def show_history_from_state(agent_executor, session_id: str):
    """从 LangGraph state 中获取并显示会话历史。"""
    if agent_executor is None:
        print("Agent 未初始化。")
        return
    try:
        config = {"configurable": {"thread_id": session_id}}
        state = agent_executor.get_state(config)
        messages = state.values.get("messages", [])
        if not messages:
            print("（无历史）")
            return
        for i, msg in enumerate(messages):
            if isinstance(msg, HumanMessage):
                role = "用户"
            elif isinstance(msg, AIMessage):
                role = "助手"
            else:
                continue
            content = msg.content[:100] if msg.content else ""
            print(f"{i+1}. {role}: {content}...")
    except Exception as e:
        logger.exception("获取历史失败")
        print(f"获取历史失败: {e}")


def show_user_info(user_id: str):
    import json
    file_path = f"./data/user_info/{user_id}.json"
    if not os.path.exists(file_path):
        print(f"用户 {user_id} 暂无长期记忆。")
        return
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        print("长期记忆为空。")
        return
    for k, v in data.items():
        print(f"  {k}: {v}")


def main():
    print("=== 个人知识库助手 ===")
    print("输入 /help 查看命令。")

    # 启动前检查向量库是否存在，若不存在可提示先构建
    retriever = build_rerank_retriever(top_k=20, top_n=5)

    # 加载向量库实例（用于知识库管理）
    try:
        vectordb = load_vector_store(PERSIST_DIR)
    except FileNotFoundError:
        print("警告: 向量库不存在，知识库管理命令将不可用。请先添加文档。")
        vectordb = None

    current_user_id: Optional[str] = None
    current_session_id: Optional[str] = None
    agent_executor = None

    # 初始提示登录
    print("请先登录: 输入 /login <用户名>")

    while True:
        try:
            raw = input(f"\n[{current_user_id or '未登录'}][会话:{current_session_id or '无'}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not raw:
            continue

        # 处理命令
        if raw.startswith("/"):
            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else None

            if cmd == "/quit" or cmd == "/exit":
                print("再见！")
                break

            elif cmd == "/help":
                print_help()

            elif cmd == "/login":
                if not arg:
                    print("用法: /login <用户名>")
                    continue
                current_user_id = arg.strip()
                # 为新用户创建新的默认会话
                current_session_id = str(uuid.uuid4())[:8]  # 简短ID
                # 重新创建 tools 和 agent (因为 user_id 变了)
                tools = create_tools(current_user_id, retriever)
                agent_executor = create_agent(tools, get_session_history)
                print(f"已登录为 {current_user_id}，新会话: {current_session_id}")

            elif cmd == "/new":
                if not current_user_id:
                    print("请先登录！")
                    continue
                current_session_id = str(uuid.uuid4())[:8]
                print(f"新会话: {current_session_id}")

            elif cmd == "/switch":
                if not arg:
                    print("用法: /switch <会话ID>")
                    continue
                if not current_user_id:
                    print("请先登录！")
                    continue
                # 可简单检查 history 文件是否存在
                hist_file = f"./data/chat_history/{arg}.json"
                if not os.path.exists(hist_file):
                    print(f"会话 {arg} 不存在，将创建新会话。")
                current_session_id = arg
                print(f"已切换到会话 {current_session_id}")

            elif cmd == "/history":
                if not current_session_id:
                    print("无活动会话。")
                    continue
                show_history_from_state(agent_executor, current_session_id)

            elif cmd == "/kb":
                if not arg:
                    print("用法: /kb add <file> | /kb delete <file> | /kb rebuild | /kb search <query>")
                    continue
                kb_parts = arg.split(maxsplit=1)
                subcmd = kb_parts[0].lower()
                subarg = kb_parts[1] if len(kb_parts) > 1 else None

                if subcmd == "add":
                    if not subarg:
                        print("用法: /kb add <file_path>")
                        continue
                    if vectordb is None:
                        print("向量库未加载，无法添加。")
                        continue
                    print(add_file_to_kb(vectordb, subarg))

                elif subcmd == "delete":
                    if not subarg:
                        print("用法: /kb delete <file_path>")
                        continue
                    if vectordb is None:
                        print("向量库未加载，无法删除。")
                        continue
                    print(delete_file_from_kb(vectordb, subarg))

                elif subcmd == "rebuild":
                    if vectordb is None:
                        print("向量库未加载，无法重建。")
                        continue
                    confirm = input("确认重建知识库？这将删除现有数据并重新加载 documents 目录。(y/n): ").strip().lower()
                    if confirm == "y":
                        try:
                            vectordb = rebuild_kb(PERSIST_DIR, DOCS_DIR)
                            # 重建后 retriever 可能持有旧的 vectordb 引用，需要重建
                            retriever = build_rerank_retriever(top_k=20, top_n=5)
                            print("知识库重建完成。")
                        except Exception as e:
                            print(f"重建失败: {e}")
                    else:
                        print("已取消重建。")

                elif subcmd == "search":
                    if not subarg:
                        print("用法: /kb search <query>")
                        continue
                    if vectordb is None:
                        print("向量库未加载，无法搜索。")
                        continue
                    print(search_kb(vectordb, subarg))

                else:
                    print(f"未知 /kb 子命令: {subcmd}")

            elif cmd == "/userinfo":
                if not current_user_id:
                    print("未登录。")
                    continue
                show_user_info(current_user_id)

            else:
                print(f"未知命令: {cmd}。输入 /help 查看帮助。")

        else:
            # 普通对话
            if not current_user_id:
                print("请先登录: /login <用户名>")
                continue
            if not agent_executor:
                print("系统未初始化，请重新登录。")
                continue

            # 调用 Agent（流式输出）
            try:
                # 防御：清理输入中可能的非法 surrogate 字符
                if any(0xD800 <= ord(ch) <= 0xDFFF for ch in raw):
                    raw = raw.encode("utf-8", "ignore").decode("utf-8")
                print("\n助手: ", end="", flush=True)
                assistant_msg = ""

                for chunk, metadata in agent_executor.stream(
                    {"messages": [{"role": "user", "content": raw}]},
                    {"configurable": {"thread_id": current_session_id}},
                    stream_mode="messages",
                ):
                    # 只输出 AIMessageChunk 中的文本片段，排除 ToolMessage 等内部消息
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        print(chunk.content, end="", flush=True)
                        assistant_msg += chunk.content

                print()  # 最后换行
            except Exception as e:
                logger.exception("Agent 调用失败")
                print(f"\n发生错误: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)  # 减少日志干扰
    main()
