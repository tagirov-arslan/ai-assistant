"""Wake-word детектор для ИИ-ассистента.

Лёгкий режим прослушивания: периодически слушает короткие фрагменты с микрофона,
распознаёт их через уже существующий STT-сервис проекта и проверяет наличие
ключевой фразы. Пока wake word не найдено — ничего не отправляется в backend/LLM.
При обнаружении wake word вызывается callback `on_wake_detected`.

Модуль не зависит от конкретной реализации STT: `stt_service` — это любая
вызываемая функция, которая записывает короткое окно и возвращает текст ('' если
тишина). В проекте сюда передаётся ``SpeechModule.recognize_short_phrase``.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable


def normalize_text(text: str) -> str:
    """Нормализует текст для сравнения ключевых фраз.

    - нижний регистр;
    - "ё" → "е";
    - убирает запятые/точки/!/? и прочую пунктуацию;
    - схлопывает лишние пробелы.

    "Эй, Сталин!" → "эй сталин"; "Слушай, Сталин" → "слушай сталин".
    """
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


# Обратная совместимость / краткий внутренний алиас.
_normalize = normalize_text


class WakeWordListener:
    """Фоновый слушатель ключевой фразы.

    Параметры:
        wake_words: список ключевых фраз (например, ["ассистент", "эй ассистент"]).
        stt_service: callable() -> str — записывает короткое окно и возвращает текст.
                     Может бросать MicrophoneError / SttUnavailableError.
        on_wake_detected: callable() — вызывается при обнаружении wake word
                          (выполняется в потоке слушателя, блокирует цикл до конца).
        on_status: callable(str) | None — колбэк для отображения статуса в UI.
        on_log: callable(str) | None — дополнительный приёмник логов.
        should_listen: callable() -> bool | None — гейт: если вернёт False,
                       слушатель пропускает запись (ассистент занят).
        listen_interval: пауза в секундах между попытками прослушивания.
        language: язык wake word (информационно/для STT).
        enabled: стартовое состояние.
    """

    def __init__(
        self,
        wake_words: list[str],
        stt_service: Callable[[], str],
        on_wake_detected: Callable[[], None],
        *,
        on_status: Callable[[str], None] | None = None,
        on_log: Callable[[str], None] | None = None,
        should_listen: Callable[[], bool] | None = None,
        listen_interval: float = 3.0,
        language: str = "ru",
        enabled: bool = True,
    ) -> None:
        self.stt_service = stt_service
        self.on_wake_detected = on_wake_detected
        self.on_status = on_status
        self.on_log = on_log
        self.should_listen = should_listen
        self.listen_interval = float(listen_interval)
        self.language = language
        self.enabled = bool(enabled)

        self.set_wake_words(wake_words)

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # --- Конфигурация -----------------------------------------------------

    def set_wake_words(self, wake_words: list[str]) -> None:
        self.wake_words = [w for w in wake_words if isinstance(w, str) and w.strip()]
        self._normalized_wake_words = [_normalize(w) for w in self.wake_words]

    # --- Управление жизненным циклом -------------------------------------

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.listen_loop, name="wake-word", daemon=True)
        self._thread.start()
        self._log("Listening...")

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thread = None

    def set_enabled(self, enabled: bool) -> None:
        """Включает/выключает детектор. Сам поток продолжает жить, но простаивает."""
        self.enabled = bool(enabled)
        if self.enabled:
            self.start()
            self._set_status("Ожидание wake word")
        else:
            self._set_status("Wake word выключен")

    # --- Основной цикл ----------------------------------------------------

    def listen_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self.enabled or not self._can_listen():
                self._stop_event.wait(0.3)
                continue

            self._set_status("Ожидание wake word")
            self._log("Listening...")

            try:
                text = self.stt_service()
            except Exception as error:  # типизируем по имени класса, без жёсткого импорта
                error_name = type(error).__name__
                if error_name == "SttUnavailableError":
                    self._set_status("STT недоступен")
                    self._log(f"STT unavailable: {error}")
                else:
                    self._set_status("Ошибка микрофона")
                    self._log(f"Microphone error: {error}")
                self._stop_event.wait(max(1.0, self.listen_interval))
                continue

            if not text:
                # Тишина — это нормальный случай, просто слушаем дальше.
                continue

            self._log(f"Recognized text: {text}")

            if self.contains_wake_word(text):
                self._log(f"Wake word detected: {text.strip()}")
                self._set_status("Wake word обнаружено")
                self._log("Waiting for user command")
                try:
                    self.on_wake_detected()
                except Exception as error:
                    self._log(f"Command handler error: {error}")
                self._log("Returning to idle mode")
                self._set_status("Ожидание wake word")
            else:
                # Не wake word — выдерживаем интервал, чтобы не нагружать STT.
                self._stop_event.wait(self.listen_interval)

    def contains_wake_word(self, text: str) -> bool:
        normalized = _normalize(text)
        if not normalized:
            return False
        for wake in self._normalized_wake_words:
            if wake and wake in normalized:
                return True
        return False

    # --- Внутреннее -------------------------------------------------------

    def _can_listen(self) -> bool:
        if self.should_listen is None:
            return True
        try:
            return bool(self.should_listen())
        except Exception:
            return True

    def _set_status(self, status: str) -> None:
        if self.on_status is not None:
            try:
                self.on_status(status)
            except Exception:
                pass

    def _log(self, message: str) -> None:
        print(f"[WakeWord] {message}", flush=True)
        if self.on_log is not None:
            try:
                self.on_log(message)
            except Exception:
                pass
