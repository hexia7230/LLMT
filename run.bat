@echo off
title LLM-Translator
cd /d "%~dp0"

if not exist venv\Scripts\activate.bat (
    echo [ERROR] venv が見つかりません。先に setup.bat を実行してください。
    pause
    exit /b 1
)

echo Starting LLM-Translator...
echo ブラウザが自動で開きます。このウィンドウを閉じると停止します。
echo.

call venv\Scripts\activate
python translator.py

pause
