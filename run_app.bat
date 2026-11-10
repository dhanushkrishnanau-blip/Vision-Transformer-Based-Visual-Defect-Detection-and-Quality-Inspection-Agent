@echo off
cd /d "%~dp0"
if not exist "frontend\dist\index.html" (
  cd /d "%~dp0frontend"
  call npm install
  if errorlevel 1 exit /b 1
  call npm run build
  if errorlevel 1 exit /b 1
  cd /d "%~dp0"
)
call ".venv\Scripts\activate.bat"
python -m uvicorn backend.main:app --port 8000
