@echo off
cd /d "%~dp0"
call ".venv\Scripts\activate.bat"
python -m uvicorn backend.main:app --port 8000
