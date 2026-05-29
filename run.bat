@echo off
title LLM-Translator
cd /d "%~dp0"

if not exist venv\Scripts\activate.bat (
    echo [ERROR] Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

echo Starting server via virtual environment (Low Priority Mode)...
echo Browser will open automatically.
echo Close this window to stop.
echo.

call venv\Scripts\activate

start /low /b "" python translator.py

pause