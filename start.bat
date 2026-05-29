@echo off
title LLM-Translator

echo ============================================================
echo  LLM-Translator
echo ============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo Install Python 3.10 or later: https://www.python.org/downloads/
    pause
    exit /b 1
)

python --version
echo.

python -m pip install --upgrade pip --quiet --disable-pip-version-check

echo Starting server...
echo Browser will open automatically.
echo Close this window to stop.
echo.

cd /d "%~dp0"
python translator.py

pause
