"""Мост между ИИ-ассистентом и MetaHuman в Unreal Engine.

Отправляет OSC-сообщения (UDP) на локальный OSC Server в UE.
Реализация OSC 1.0 на стандартной библиотеке — внешние зависимости не нужны.

Протокол (адрес → аргументы):
    /state        [str]            "idle" | "listening" | "thinking" | "speaking"
    /speak_start  []               начало воспроизведения речи
    /speak_end    []               конец/остановка речи
    /emotion      [str, float]     имя эмоции, интенсивность 0..1
    /gaze         [float, float]   yaw, pitch (-1..1)
    /gesture      [str]            имя жеста, напр. "nod"
    /curve        [str, float]     имя кривой AnimBP, значение 0..1
    /ping         [int]            heartbeat, номер тика
"""

from __future__ import annotations

import audioop
import socket
import struct
import threading
import time
from pathlib import Path

from unreal_audio_server import UnrealAudioServer


def _osc_pad(data: bytes) -> bytes:
    """OSC выравнивает строки и blob'ы до кратности 4 байт."""
    remainder = len(data) % 4
    if remainder:
        data += b"\x00" * (4 - remainder)
    return data


def _osc_string(value: str) -> bytes:
    return _osc_pad(value.encode("utf-8") + b"\x00")


def build_osc_message(address: str, *args: str | float | int) -> bytes:
    """Собирает бинарное OSC-сообщение: адрес, type tags, аргументы."""
    if not address.startswith("/"):
        raise ValueError(f"OSC-адрес должен начинаться с '/': {address}")

    type_tags = ","
    payload = b""
    for arg in args:
        if isinstance(arg, bool):
            raise TypeError("bool не поддерживается в этом протоколе")
        if isinstance(arg, str):
            type_tags += "s"
            payload += _osc_string(arg)
        elif isinstance(arg, int):
            type_tags += "i"
            payload += struct.pack(">i", arg)
        elif isinstance(arg, float):
            type_tags += "f"
            payload += struct.pack(">f", arg)
        else:
            raise TypeError(f"Неподдерживаемый тип OSC-аргумента: {type(arg)!r}")

    return _osc_string(address) + _osc_string(type_tags) + payload


class MetaHumanBridge:
    """OSC-клиент для управления MetaHuman. Все методы потокобезопасны и не блокируют.

    UDP — fire-and-forget: если UE не запущен, сообщения просто теряются,
    ассистент продолжает работать как обычно.
    """

    STATES = {"idle", "listening", "thinking", "speaking"}

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        enabled: bool = True,
        heartbeat_seconds: float = 2.0,
        websocket_host: str = "127.0.0.1",
        websocket_port: int = 8765,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.enabled = bool(enabled)
        self.heartbeat_seconds = float(heartbeat_seconds)

        self._lock = threading.Lock()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self._last_state = "idle"
        self._ping_counter = 0
        self._stop_event = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self.audio_server = UnrealAudioServer(websocket_host, websocket_port)
        self._audio_debug_path = Path(__file__).resolve().parent / "metahuman_audio_debug.log"
        self._audio_debug_counter = 0

        if self.enabled:
            self.audio_server.start()
            if self.heartbeat_seconds > 0:
                self._start_heartbeat()

    # --- Публичный API -------------------------------------------------

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if self.enabled:
            self.audio_server.start()
            if self._heartbeat_thread is None and self.heartbeat_seconds > 0:
                self._start_heartbeat()
            self.set_state(self._last_state, force=True)
        else:
            self.audio_server.stop()

    def set_state(self, state: str, force: bool = False) -> None:
        if state not in self.STATES:
            return
        if state == self._last_state and not force:
            return
        self._last_state = state
        self._send("/state", state)
        self.audio_server.send_state(state)

    def get_state(self) -> str:
        return self._last_state

    def speak_start(self) -> None:
        self.set_state("speaking")
        self._send("/speak_start")
        self.audio_server.send_event("speak_start")

    def speak_end(self) -> None:
        self._send("/speak_end")
        self.audio_server.send_event("speak_end")
        self.set_state("idle")

    def send_audio_chunk(self, pcm: bytes, sample_rate: int, channels: int) -> None:
        if self.enabled:
            self.audio_server.send_audio(pcm, sample_rate, channels, sample_width=2)
            self._audio_debug_counter += 1
            if self._audio_debug_counter == 1 or self._audio_debug_counter % 100 == 0:
                stats = self.audio_server.get_stats()
                usable = len(pcm) - (len(pcm) % 2)
                raw_rms = audioop.rms(pcm[:usable], 2) if usable else 0
                raw_min, raw_max = audioop.minmax(pcm[:usable], 2) if usable else (0, 0)
                norm_rms = raw_rms / 32768.0
                norm_peak = max(abs(raw_min), abs(raw_max)) / 32768.0
                try:
                    with self._audio_debug_path.open("a", encoding="utf-8") as log_file:
                        log_file.write(
                            f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                            f"callback={self._audio_debug_counter} pcm={len(pcm)} "
                            f"rate={sample_rate} channels={channels} "
                            f"raw_rms={raw_rms} raw_min={raw_min} raw_max={raw_max} "
                            f"norm_rms={norm_rms:.5f} norm_peak={norm_peak:.5f} "
                            f"first_bytes={pcm[:16].hex()} stats={stats}\n"
                        )
                except OSError:
                    pass

    def emotion(self, name: str, intensity: float = 0.5) -> None:
        self._send("/emotion", name, max(0.0, min(1.0, float(intensity))))

    def gaze(self, yaw: float, pitch: float) -> None:
        self._send("/gaze", float(yaw), float(pitch))

    def gesture(self, name: str) -> None:
        self._send("/gesture", name)

    def curve(self, name: str, value: float) -> None:
        self._send("/curve", name, max(0.0, min(1.0, float(value))))

    def shutdown(self) -> None:
        self._stop_event.set()
        try:
            self._send("/state", "idle")
        finally:
            self.audio_server.stop()
            try:
                self._socket.close()
            except OSError:
                pass

    # --- Внутреннее ------------------------------------------------------

    def _send(self, address: str, *args: str | float | int) -> None:
        if not self.enabled or self._stop_event.is_set():
            return
        try:
            packet = build_osc_message(address, *args)
            with self._lock:
                self._socket.sendto(packet, (self.host, self.port))
        except (OSError, ValueError, TypeError):
            # UDP fire-and-forget: ошибки отправки не должны ломать ассистента.
            pass

    def _start_heartbeat(self) -> None:
        def runner() -> None:
            while not self._stop_event.wait(self.heartbeat_seconds):
                self._ping_counter += 1
                self._send("/ping", self._ping_counter)

        self._heartbeat_thread = threading.Thread(target=runner, daemon=True)
        self._heartbeat_thread.start()


if __name__ == "__main__":
    # Смоук-тест: python metahuman_bridge.py — шлёт тестовую последовательность в UE.
    import time

    bridge = MetaHumanBridge()
    print(f"Шлю тестовые OSC-сообщения на {bridge.host}:{bridge.port} (Ctrl+C для выхода)")
    bridge.set_state("listening")
    time.sleep(1)
    bridge.set_state("thinking")
    time.sleep(1)
    bridge.speak_start()
    bridge.emotion("stern", 0.8)
    bridge.gaze(0.2, -0.1)
    time.sleep(2)
    bridge.speak_end()
    print("Готово. Проверь Output Log в UE.")
    bridge.shutdown()
