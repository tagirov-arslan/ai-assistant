@echo off
setlocal

cd /d "%~dp0"

set "PYTHON=.venv-win\Scripts\pythonw.exe"
set "PYTHON_CONSOLE=.venv-win\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo .venv-win was not found.
    echo Create it with:
    echo   C:\Users\admin\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv-win
    echo   .venv-win\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

"%PYTHON_CONSOLE%" --version >nul 2>nul
if errorlevel 1 (
    echo .venv-win exists, but its Python launcher is not working.
    echo Recreate it with the project Windows Python:
    echo   C:\Users\admin\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv-win
    echo   .venv-win\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

start "" "%PYTHON%" app.py
