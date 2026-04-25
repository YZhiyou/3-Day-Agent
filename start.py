#!/usr/bin/env python3
"""一键启动 Streamlit 前端。"""

import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
VENV_STREAMLIT = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "streamlit.exe")

def main():
    # 使用虚拟环境中的 python -m streamlit 启动（避免 streamlit.exe 路径硬编码问题）
    if os.path.exists(VENV_PYTHON):
        cmd = [VENV_PYTHON, "-m", "streamlit", "run", os.path.join(PROJECT_ROOT, "streamlit_app.py")]
    else:
        # 回退到系统环境
        cmd = [sys.executable, "-m", "streamlit", "run", os.path.join(PROJECT_ROOT, "streamlit_app.py")]

    print(f"启动命令: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_ROOT)

if __name__ == "__main__":
    main()
