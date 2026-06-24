"""Проверка готовности окружения ИИ-ассистента на новой машине.

Запуск:  .venv-win\\Scripts\\python.exe check_setup.py
Печатает понятный отчёт OK/ВНИМАНИЕ/ОШИБКА и возвращает код выхода != 0,
если есть блокирующие проблемы (Python/зависимости/модели).

Скрипт зависит только от стандартной библиотеки, поэтому его можно запускать
даже до установки зависимостей (тогда он подскажет, что именно поставить).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

OK = "[ OK ]"
WARN = "[ВНИМ]"
ERR = "[ОШИБ]"

errors = 0
warnings = 0


def report(level: str, message: str) -> None:
    global errors, warnings
    if level is ERR:
        errors += 1
    elif level is WARN:
        warnings += 1
    print(f"{level} {message}")


def check_python() -> None:
    major, minor = sys.version_info[:2]
    version = f"{major}.{minor}.{sys.version_info[2]}"
    if major == 3 and minor == 12:
        report(OK, f"Python {version}")
    elif major == 3 and minor >= 13:
        report(ERR, f"Python {version}: нужен 3.12. В 3.13 удалён модуль audioop — звук работать не будет.")
    elif major == 3 and minor < 12:
        report(WARN, f"Python {version}: рекомендуется 3.12 (проект тестировался на 3.12.9).")
    else:
        report(ERR, f"Python {version}: нужен 3.12.")


def check_dependencies() -> None:
    deps = {
        "pyaudio": "PyAudio (запись/воспроизведение звука)",
        "faster_whisper": "faster-whisper (распознавание речи)",
        "supertonic": "supertonic (озвучка TTS)",
    }
    for module, description in deps.items():
        if importlib.util.find_spec(module) is None:
            report(ERR, f"Не установлен пакет '{module}' — {description}. Выполните: pip install -r requirements.txt")
        else:
            report(OK, f"Пакет '{module}' доступен")


def check_models() -> None:
    whisper_dir = PROJECT_DIR / "whisper-large-v3-turbo-ct2"
    if (whisper_dir / "model.bin").exists():
        report(OK, "Локальная модель Whisper: whisper-large-v3-turbo-ct2/model.bin")
    else:
        report(
            WARN,
            "Нет локальной модели Whisper. Скачайте: python download_models.py "
            "(или будет скачана при первом запуске — нужен интернет).",
        )

    supertonic_dir = PROJECT_DIR / "supertonic-3-model"
    onnx_ok = (supertonic_dir / "onnx").is_dir() and any((supertonic_dir / "onnx").glob("*.onnx"))
    voices_ok = (supertonic_dir / "voice_styles").is_dir() and any((supertonic_dir / "voice_styles").glob("*.json"))
    if onnx_ok and voices_ok:
        report(OK, "Модель Supertonic TTS: supertonic-3-model/ (onnx + voice_styles)")
    else:
        report(
            ERR,
            "Нет модели Supertonic (supertonic-3-model/onnx + voice_styles). "
            "Скачайте: python download_models.py",
        )


def _read_lm_config() -> tuple[str, str]:
    """Достаёт base_url и model из app.py без его импорта (чтобы не тянуть зависимости)."""
    base_url = "http://127.0.0.1:1234"
    model = ""
    try:
        text = (PROJECT_DIR / "app.py").read_text(encoding="utf-8")
    except OSError:
        return base_url, model
    import re

    base_match = re.search(r'base_url\s*=\s*"([^"]+)"', text)
    model_match = re.search(r'model\s*=\s*"([^"]+)"', text)
    if base_match:
        base_url = base_match.group(1)
    if model_match:
        model = model_match.group(1)
    return base_url, model


def check_lm_studio() -> None:
    base_url, configured_model = _read_lm_config()
    api = base_url.rstrip("/")
    api = api if api.endswith("/v1") else api + "/v1"
    url = f"{api}/models"
    try:
        request = urllib.request.Request(url, headers={"Authorization": "Bearer lm-studio"})
        with urllib.request.urlopen(request, timeout=4) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        report(
            WARN,
            f"LM Studio не отвечает на {base_url}. Запустите LM Studio, загрузите модель и включите локальный сервер.",
        )
        return

    loaded = [str(m.get("id", "")) for m in data.get("data", []) if m.get("id")]
    report(OK, f"LM Studio отвечает ({base_url}); загружено моделей: {len(loaded)}")
    if configured_model and loaded and configured_model not in loaded:
        report(
            WARN,
            f"В app.py указана модель '{configured_model}', но в LM Studio её нет. "
            f"Доступны: {', '.join(loaded)}. Обновите LM_CONFIG.model в app.py или загрузите нужную модель.",
        )


def check_settings() -> None:
    settings = PROJECT_DIR / "assistant_settings.json"
    if not settings.exists():
        report(OK, "assistant_settings.json отсутствует — создастся при первом запуске с авто-выбором микрофона.")
        return
    try:
        raw = json.loads(settings.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        report(WARN, "assistant_settings.json не читается — удалите его, создастся заново.")
        return
    idx = raw.get("output_device_index")
    if isinstance(idx, int):
        report(
            WARN,
            f"output_device_index={idx} остался от прежней машины. Если устройства нет, приложение откатится на "
            "системный вывод; для чистоты удалите assistant_settings.json или сбросьте индексы в null.",
        )
    else:
        report(OK, "Настройки устройств не привязаны к чужому железу.")


def main() -> int:
    print("=== Проверка окружения ИИ-ассистента ===\n")
    check_python()
    check_dependencies()
    check_models()
    check_lm_studio()
    check_settings()

    print("\n=== Итог ===")
    if errors:
        print(f"{ERR} Блокирующих проблем: {errors}, предупреждений: {warnings}. Исправьте ошибки перед запуском.")
        return 1
    if warnings:
        print(f"{WARN} Ошибок нет, предупреждений: {warnings}. Запуск возможен, но проверьте замечания выше.")
        return 0
    print(f"{OK} Всё готово. Запускайте: run_app.bat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
