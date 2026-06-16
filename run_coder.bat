@echo off
title LLMT Orchestrator — Coder Execution
cd /d "%~dp0"

if not exist venv\Scripts\activate.bat (
    echo [ERROR] venv が見つかりません。先に setup.bat を実行してください。
    pause
    exit /b 1
)

echo Starting LLMT Coder (Phase 4)...
echo prompt_dump.json を読み込んでコード生成を実行します。
echo.

call venv\Scripts\activate
python coder.py

pause
