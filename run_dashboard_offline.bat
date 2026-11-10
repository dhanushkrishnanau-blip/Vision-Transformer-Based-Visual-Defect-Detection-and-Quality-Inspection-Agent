@echo off
cd /d "%~dp0"
set DASHBOARD_OFFLINE=1
call ".venv\Scripts\activate.bat"
python -m streamlit run app\dashboard.py
