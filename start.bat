@echo off
chcp 65001 >nul
cd /d "%~dp0"
.venv\Scripts\python.exe -m streamlit run src\ui\streamlit_app.py
pause
