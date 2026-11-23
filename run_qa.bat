@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"

set "PY=%~dp0venv_minrag\Scripts\python.exe"
"%PY%" -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo [run_qa] venv python unusable, falling back to "py -3.12"
    set "PY=py"
    set "PYARGS=-3.12"
) else (
    set "PYARGS="
)

echo [run_qa] interpreter: %PY% %PYARGS%
echo [run_qa] cwd        : %CD%
echo.

"%PY%" %PYARGS% -u reproduce\Step_1_QA.py %*
set "RC=%ERRORLEVEL%"

echo.
echo [run_qa] finished, exit code = %RC%
pause
endlocal
