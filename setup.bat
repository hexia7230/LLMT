@echo off
title LLM-Translator Setup
cd /d "%~dp0"

echo ============================================================
echo  LLM-Translator Environment Setup
echo ============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10 or later.
    pause
    exit /b 1
)

echo [1/3] Creating virtual environment (venv)...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

echo [2/3] Installing PyTorch with CUDA 12.4 support...
call venv\Scripts\activate
python -m pip install --upgrade pip --quiet
pip install torch --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo [ERROR] Failed to install PyTorch.
    pause
    exit /b 1
)

echo [3/3] Installing dependency packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Setup completed successfully.
echo  Please execute run.bat to start the server.
echo ============================================================
pause