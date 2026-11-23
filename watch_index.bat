@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PY=%~dp0venv_minrag\Scripts\python.exe"
"%PY%" -c "import sys" >nul 2>&1
if errorlevel 1 set "PY=py"
"%PY%" -u watch_index.py %*
if "%~1"=="" pause
