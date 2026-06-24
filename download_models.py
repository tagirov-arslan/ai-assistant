"""Скачивание локальных моделей ИИ-ассистента (Whisper + Supertonic).

Качает обе модели с HuggingFace Hub прямо в папки проекта, чтобы их не нужно
было переносить вручную:
    whisper-large-v3-turbo-ct2/   - STT (faster-whisper, CTranslate2)
    supertonic-3-model/           - TTS (Supertonic 3, ONNX)

Запуск:
    .venv-win\\Scripts\\python.exe download_models.py          # скачать недостающее
    .venv-win\\Scripts\\python.exe download_models.py --force  # перекачать заново

Нужен доступ в интернет и установленные зависимости (см. requirements.txt).
Скрипт идемпотентен: уже скачанные модели пропускаются.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

WHISPER_DIR = PROJECT_DIR / "whisper-large-v3-turbo-ct2"
WHISPER_MODEL_ID = "large-v3-turbo"

SUPERTONIC_DIR = PROJECT_DIR / "supertonic-3-model"
SUPERTONIC_MODEL_NAME = "supertonic-3"


def _whisper_present() -> bool:
    return (WHISPER_DIR / "model.bin").exists()


def _supertonic_present() -> bool:
    onnx_ok = (SUPERTONIC_DIR / "onnx").is_dir() and any((SUPERTONIC_DIR / "onnx").glob("*.onnx"))
    voices_ok = (SUPERTONIC_DIR / "voice_styles").is_dir() and any((SUPERTONIC_DIR / "voice_styles").glob("*.json"))
    return onnx_ok and voices_ok


def download_whisper(force: bool) -> bool:
    if _whisper_present() and not force:
        print(f"[ OK ] Whisper уже на месте: {WHISPER_DIR.name}")
        return True
    try:
        from faster_whisper import download_model
    except ImportError:
        print("[ОШИБ] Пакет faster-whisper не установлен. Сначала: pip install -r requirements.txt")
        return False

    print(f"[ .. ] Скачиваю Whisper '{WHISPER_MODEL_ID}' в {WHISPER_DIR.name}/ ...")
    try:
        download_model(WHISPER_MODEL_ID, output_dir=str(WHISPER_DIR))
    except Exception as error:
        print(f"[ОШИБ] Не удалось скачать Whisper: {error}")
        return False
    if not _whisper_present():
        print("[ОШИБ] Whisper скачан, но model.bin не найден — проверьте папку.")
        return False
    print(f"[ OK ] Whisper готов: {WHISPER_DIR.name}")
    return True


def download_supertonic(force: bool) -> bool:
    if _supertonic_present() and not force:
        print(f"[ OK ] Supertonic уже на месте: {SUPERTONIC_DIR.name}")
        return True
    try:
        from supertonic.loader import download_model
    except ImportError:
        print("[ОШИБ] Пакет supertonic не установлен. Сначала: pip install -r requirements.txt")
        return False

    print(f"[ .. ] Скачиваю Supertonic '{SUPERTONIC_MODEL_NAME}' в {SUPERTONIC_DIR.name}/ ...")
    try:
        download_model(str(SUPERTONIC_DIR), SUPERTONIC_MODEL_NAME)
    except Exception as error:
        print(f"[ОШИБ] Не удалось скачать Supertonic: {error}")
        return False
    if not _supertonic_present():
        print("[ОШИБ] Supertonic скачан, но onnx/voice_styles не найдены — проверьте папку.")
        return False
    print(f"[ OK ] Supertonic готов: {SUPERTONIC_DIR.name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Скачать модели Whisper и Supertonic в папки проекта.")
    parser.add_argument("--force", action="store_true", help="Перекачать, даже если модель уже есть.")
    parser.add_argument("--whisper-only", action="store_true", help="Скачать только Whisper.")
    parser.add_argument("--supertonic-only", action="store_true", help="Скачать только Supertonic.")
    args = parser.parse_args()

    print("=== Загрузка моделей ИИ-ассистента ===\n")
    ok = True
    if not args.supertonic_only:
        ok = download_whisper(args.force) and ok
    if not args.whisper_only:
        ok = download_supertonic(args.force) and ok

    print()
    if ok:
        print("[ OK ] Все модели на месте.")
        return 0
    print("[ОШИБ] Некоторые модели не скачались. Проверьте интернет и повторите: python download_models.py")
    return 1


if __name__ == "__main__":
    sys.exit(main())
