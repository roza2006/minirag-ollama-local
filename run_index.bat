@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"

REM Use the project venv if it works, otherwise the py launcher.
set "PY=%~dp0venv_minrag\Scripts\python.exe"
"%PY%" -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo [run_index] venv python unusable, falling back to "py -3.12"
    set "PY=py"
    set "PYARGS=-3.12"
) else (
    set "PYARGS="
)

echo [run_index] interpreter: %PY% %PYARGS%
echo [run_index] cwd        : %CD%
echo.

"%PY%" %PYARGS% -u reproduce\Step_0_index.py %*
set "RC=%ERRORLEVEL%"

echo.
echo [run_index] finished, exit code = %RC%
if not "%RC%"=="0" (
    echo [run_index] something went wrong - the last lines above say what.
)
pause
endlocal
