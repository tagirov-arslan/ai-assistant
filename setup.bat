@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"

echo === Установка ИИ-ассистента ===
echo.

set "VENV=.venv-win"
set "VENV_PY=%VENV%\Scripts\python.exe"

:: 1. Найти Python 3.12 (предпочтительно через launcher py -3.12)
set "BASE_PY="
py -3.12 --version >nul 2>nul
if not errorlevel 1 (
    set "BASE_PY=py -3.12"
) else (
    python --version >nul 2>nul
    if not errorlevel 1 (
        for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PYVER=%%v"
        echo Найден python !PYVER! (launcher py -3.12 недоступен).
        set "BASE_PY=python"
    )
)

if "%BASE_PY%"=="" (
    echo [ОШИБКА] Python не найден. Установите Python 3.12 с https://www.python.org/downloads/
    echo Важно: нужна именно версия 3.12, не 3.13.
    pause
    exit /b 1
)

:: 2. Создать виртуальное окружение
if not exist "%VENV_PY%" (
    echo Создаю виртуальное окружение %VENV% ...
    %BASE_PY% -m venv "%VENV%"
    if errorlevel 1 (
        echo [ОШИБКА] Не удалось создать venv.
        pause
        exit /b 1
    )
) else (
    echo Виртуальное окружение уже существует.
)

:: 3. Установить зависимости
echo Обновляю pip и устанавливаю зависимости ...
"%VENV_PY%" -m pip install --upgrade pip
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ВНИМАНИЕ] Ошибка установки зависимостей.
    echo Если не ставится PyAudio — установите готовое колесо под Python 3.12.
)

:: 4. Скачать модели Whisper и Supertonic (пропустит уже скачанные)
echo.
echo Проверяю/скачиваю модели Whisper и Supertonic (~1.2 ГБ при первом запуске) ...
"%VENV_PY%" download_models.py
if errorlevel 1 (
    echo [ВНИМАНИЕ] Не все модели скачались. Повторите позже: .venv-win\Scripts\python.exe download_models.py
)

:: 5. Создать настройки из шаблона, если их ещё нет
if not exist "assistant_settings.json" (
    if exist "assistant_settings.example.json" (
        copy /y "assistant_settings.example.json" "assistant_settings.json" >nul
        echo Создан assistant_settings.json из шаблона (микрофон выберется автоматически).
    )
)

:: 6. Проверка окружения
echo.
"%VENV_PY%" check_setup.py
echo.
echo === Готово. Запуск: run_app.bat ===
pause
