"""Лёгкий слушатель прерывания во время озвучивания ответа ассистента.

Пока ассистент говорит (работает TTS), этот слушатель параллельно слушает
микрофон короткими окнами и распознаёт их через существующий STT. Если слышит
стоп-слово ("стоп", "остановись", "хватит") или повторный wake word ("Сталин"),
он сразу вызывает соответствующий callback, чтобы основной код прервал речь.

Слушатель НЕ отправляет ничего в LLM/backend — он только проверяет наличие
ключевых фраз остановки/активации.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from wake_word_listener import normalize_text


class InterruptListener:
    """Слушатель команды прерывания и повторного wake word во время TTS.

    Параметры:
        interrupt_words: стоп-слова (например, ["стоп", "остановись", "хватит"]).
        wake_words: фразы активации; при обнаружении считаются прерыванием с
                    последующим переходом к новой команде. Передайте [], чтобы
                    отключить прерывание по wake word.
        stt_service: callable() -> str — короткое окно записи + распознавание.
        on_stop_detected: callable() — вызывается при стоп-слове.
        on_wake_detected: callable() — вызывается при повторном wake word.
        on_status / on_log: колбэки для UI/логов.
        listen_interval: пауза между окнами прослушивания (сек).
    """

    def __init__(
        self,
        interrupt_words: list[str],
        wake_words: list[str],
        stt_service: Callable[[], str],
        on_stop_detected: Callable[[], None],
        on_wake_detected: Callable[[], None],
        *,
        on_status: Callable[[str], None] | None = None,
        on_log: Callable[[str], None] | None = None,
        listen_interval: float = 0.0,
    ) -> None:
        self.stt_service = stt_service
        self.on_stop_detected = on_stop_detected
        self.on_wake_detected = on_wake_detected
        self.on_status = on_status
        self.on_log = on_log
        self.listen_interval = float(listen_interval)

        self.set_words(interrupt_words, wake_words)

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def set_words(self, interrupt_words: list[str], wake_words: list[str]) -> None:
        self.interrupt_words = [w for w in interrupt_words if isinstance(w, str) and w.strip()]
        self.wake_words = [w for w in wake_words if isinstance(w, str) and w.strip()]
        self._norm_interrupt = [normalize_text(w) for w in self.interrupt_words]
        self._norm_wake = [normalize_text(w) for w in self.wake_words]

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start_during_tts(self) -> None:
        """Запускает фоновое прослушивание на время озвучивания ответа."""
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="interrupt-listen", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._log("Listening during TTS...")
            try:
                text = self.stt_service()
            except Exception as error:
                # Сбой STT/микрофона во время прерывания не должен ничего ломать —
                # просто продолжаем слушать.
                self._log(f"STT error during TTS: {error}")
                if self._stop_event.wait(max(0.3, self.listen_interval)):
                    return
                continue

            if self._stop_event.is_set():
                return
            if not text:
                if self.listen_interval and self._stop_event.wait(self.listen_interval):
                    return
                continue

            self._log(f"Recognized text: {text}")

            if self.contains_interrupt_word(text):
                self._log("Stop word detected")
                self._safe_call(self.on_stop_detected)
                return
            if self.contains_wake_word(text):
                self._log("Wake word detected during TTS")
                self._safe_call(self.on_wake_detected)
                return

            if self.listen_interval and self._stop_event.wait(self.listen_interval):
                return

    def contains_interrupt_word(self, text: str) -> bool:
        normalized = normalize_text(text)
        if not normalized:
            return False
        tokens = normalized.split()
        for stop_word in self._norm_interrupt:
            parts = stop_word.split()
            if len(parts) == 1:
                # короткое стоп-слово сверяем по целым словам,
                # чтобы не ловить "стоп" внутри других слов.
                if stop_word in tokens:
                    return True
            elif stop_word in normalized:
                return True
        return False

    def contains_wake_word(self, text: str) -> bool:
        normalized = normalize_text(text)
        if not normalized:
            return False
        for wake in self._norm_wake:
            if wake and wake in normalized:
                return True
        return False

    def _safe_call(self, callback: Callable[[], None]) -> None:
        try:
            callback()
        except Exception as error:
            self._log(f"Interrupt callback error: {error}")

    def _set_status(self, status: str) -> None:
        if self.on_status is not None:
            try:
                self.on_status(status)
            except Exception:
                pass

    def _log(self, message: str) -> None:
        print(f"[Interrupt] {message}", flush=True)
        if self.on_log is not None:
            try:
                self.on_log(message)
            except Exception:
                pass
