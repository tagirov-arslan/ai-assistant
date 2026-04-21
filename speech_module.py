from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pyaudio
import pyttsx3
from vosk import KaldiRecognizer, Model, SetLogLevel


def _env_int(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


@dataclass
class SpeechConfig:
    language: str = "ru-RU"
    tts_rate: int = 170
    sample_rate: int = 16000
    chunk_size: int = 4000
    energy_threshold: int = 80
    phrase_time_limit_seconds: float = 8.0
    silence_limit_seconds: float = 1.2
    start_timeout_seconds: float = 5.0
    input_device_index: int | None = field(default_factory=lambda: _env_int("AI_ASSISTANT_INPUT_DEVICE_INDEX"))
    vosk_model_path: str = field(default_factory=lambda: os.getenv("AI_ASSISTANT_VOSK_MODEL_PATH", "").strip())


class SpeechModule:
    """Offline speech-to-text through Vosk and text-to-speech through pyttsx3."""

    def __init__(
        self,
        config: SpeechConfig | None = None,
        *,
        language: str | None = None,
        rate: int | None = None,
    ) -> None:
        if config is None:
            config = SpeechConfig()
        if language is not None:
            config.language = language
        if rate is not None:
            config.tts_rate = rate

        self.config = config
        self.audio = pyaudio.PyAudio()
        self.tts_lock = threading.Lock()
        self.tts_engine = None
        self.tts_available = False

        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty("rate", self.config.tts_rate)
            self.tts_available = True
        except Exception:  # noqa: BLE001
            self.tts_engine = None
            self.tts_available = False

        SetLogLevel(-1)
        self.vosk_model_path = self._resolve_vosk_model_path(self.config.vosk_model_path)
        self.vosk_model = Model(str(self.vosk_model_path))
        self.input_device_index, self.input_device_name, self.active_sample_rate = self._resolve_input_device()

    def speech_to_text(self) -> str:
        recognizer = KaldiRecognizer(self.vosk_model, float(self.active_sample_rate))
        recognizer.SetWords(True)

        max_chunks = max(
            1,
            int(self.config.phrase_time_limit_seconds * self.active_sample_rate / self.config.chunk_size),
        )
        start_timeout_chunks = max(
            1,
            int(self.config.start_timeout_seconds * self.active_sample_rate / self.config.chunk_size),
        )
        silence_limit_chunks = max(
            1,
            int(self.config.silence_limit_seconds * self.active_sample_rate / self.config.chunk_size),
        )

        speech_started = False
        silence_chunks = 0
        collected_parts: list[str] = []
        last_partial = ""

        with self._open_input_stream(self.active_sample_rate, self.config.chunk_size) as stream:
            for index in range(max_chunks):
                data = stream.read(self.config.chunk_size, exception_on_overflow=False)
                energy = self._estimate_energy(data)

                if energy >= self.config.energy_threshold:
                    speech_started = True
                    silence_chunks = 0
                elif speech_started:
                    silence_chunks += 1

                if recognizer.AcceptWaveform(data):
                    text = self._extract_text(recognizer.Result())
                    if text:
                        self._append_unique_part(collected_parts, text)
                        speech_started = True
                else:
                    partial = self._extract_partial(recognizer.PartialResult())
                    if partial:
                        last_partial = partial
                        speech_started = True

                if not speech_started and index >= start_timeout_chunks:
                    raise RuntimeError("Речь не обнаружена. Проверьте микрофон и попробуйте ещё раз.")

                if speech_started and silence_chunks >= silence_limit_chunks:
                    break

        final_text = self._extract_text(recognizer.FinalResult())
        if final_text:
            self._append_unique_part(collected_parts, final_text)
        elif last_partial:
            self._append_unique_part(collected_parts, last_partial)

        result = " ".join(part.strip() for part in collected_parts if part.strip()).strip()
        result = self._collapse_repeated_phrase(result)
        if not result:
            raise RuntimeError("Не удалось распознать речь через Vosk.")
        return result

    def text_to_speech(self, text: str) -> None:
        if not text.strip() or self.tts_engine is None:
            return
        with self.tts_lock:
            self.tts_engine.stop()
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()

    def text_to_speech_async(self, text: str) -> None:
        if not text.strip() or self.tts_engine is None:
            return
        threading.Thread(target=self.text_to_speech, args=(text,), daemon=True).start()

    def stop_speaking(self) -> None:
        if self.tts_engine is None:
            return
        with self.tts_lock:
            self.tts_engine.stop()

    def shutdown(self) -> None:
        self.stop_speaking()
        self.audio.terminate()

    def set_input_device(self, device_index: int | None) -> None:
        self.config.input_device_index = device_index
        self.input_device_index, self.input_device_name, self.active_sample_rate = self._resolve_input_device()

    def get_input_device_index(self) -> int:
        return self.input_device_index

    def get_input_device_label(self) -> str:
        return f"{self.input_device_name} ({self.active_sample_rate} Hz)"

    def list_input_devices(self) -> list[dict[str, object]]:
        devices: list[dict[str, object]] = []
        for index in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(index)
            if int(info.get("maxInputChannels", 0)) <= 0:
                continue
            devices.append(
                {
                    "index": index,
                    "name": str(info.get("name", f"Device {index}")),
                    "max_input_channels": int(info.get("maxInputChannels", 0)),
                    "default_sample_rate": int(float(info.get("defaultSampleRate", self.config.sample_rate))),
                }
            )
        return devices

    @contextmanager
    def _open_input_stream(self, sample_rate: int, frames_per_buffer: int):
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            input_device_index=self.input_device_index,
            frames_per_buffer=frames_per_buffer,
        )
        try:
            yield stream
        finally:
            stream.stop_stream()
            stream.close()

    def _resolve_input_device(self) -> tuple[int, str, int]:
        devices = self.list_input_devices()
        if not devices:
            raise RuntimeError("В системе не найдено ни одного входного аудиоустройства.")

        if self.config.input_device_index is not None:
            for device in devices:
                if device["index"] == self.config.input_device_index:
                    rate = self._pick_sample_rate_for_device(
                        int(device["index"]),
                        int(device["default_sample_rate"]),
                    )
                    return int(device["index"]), str(device["name"]), rate
            raise RuntimeError(f"Микрофон с индексом {self.config.input_device_index} не найден.")

        default_info = self.audio.get_default_input_device_info()
        default_index = int(default_info["index"])
        default_rate = int(float(default_info.get("defaultSampleRate", self.config.sample_rate)))
        if self._can_open_device(default_index, default_rate):
            rate = self._pick_sample_rate_for_device(default_index, default_rate)
            return default_index, str(default_info.get("name", "Default input")), rate

        def score(device: dict[str, object]) -> int:
            name = str(device["name"]).lower()
            value = 0

            preferred = ["microphone", "микрофон", "mic input", "realtek", "array"]
            avoided = ["stereo mix", "speaker", "output", "hands-free", "переназначение"]

            for token in preferred:
                if token in name:
                    value += 10
            for token in avoided:
                if token in name:
                    value -= 12

            return value

        for device in sorted(devices, key=score, reverse=True):
            rate = self._pick_sample_rate_for_device(
                int(device["index"]),
                int(device["default_sample_rate"]),
            )
            if self._can_open_device(int(device["index"]), rate):
                return int(device["index"]), str(device["name"]), rate

        raise RuntimeError("Не удалось открыть ни один микрофон через PyAudio.")

    def _pick_sample_rate_for_device(self, device_index: int, default_sample_rate: int) -> int:
        preferred_rates = [self.config.sample_rate, 16000, 22050, 32000, default_sample_rate, 44100, 48000]
        checked: set[int] = set()
        for rate in preferred_rates:
            sample_rate = int(rate)
            if sample_rate in checked:
                continue
            checked.add(sample_rate)
            if self._can_open_device(device_index, sample_rate):
                return sample_rate
        return int(default_sample_rate)

    def _can_open_device(self, device_index: int, sample_rate: int) -> bool:
        try:
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=min(self.config.chunk_size, 2048),
            )
            stream.read(min(self.config.chunk_size, 2048), exception_on_overflow=False)
            stream.stop_stream()
            stream.close()
            return True
        except Exception:  # noqa: BLE001
            return False

    def _resolve_vosk_model_path(self, explicit_path: str) -> Path:
        candidates: list[Path] = []

        if explicit_path:
            candidates.append(Path(explicit_path))

        project_dir = Path(__file__).resolve().parent
        candidates.extend(
            [
                project_dir / "models" / "vosk-model-small-ru-0.22",
                project_dir / "resources" / "vosk" / "vosk-model-small-ru-0.22",
                Path(r"C:\Users\Macinery_knr\Desktop\jarvis-master\resources\vosk\vosk-model-small-ru-0.22"),
            ]
        )

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise RuntimeError("Vosk-модель не найдена. Укажите путь через AI_ASSISTANT_VOSK_MODEL_PATH.")

    def _estimate_energy(self, audio_bytes: bytes) -> int:
        samples = memoryview(audio_bytes).cast("h")
        if not samples:
            return 0
        return int(sum(abs(sample) for sample in samples) / len(samples))

    def _extract_text(self, payload: str) -> str:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return ""
        return str(data.get("text", "")).strip()

    def _extract_partial(self, payload: str) -> str:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return ""
        return str(data.get("partial", "")).strip()

    def _append_unique_part(self, parts: list[str], text: str) -> None:
        normalized = self._normalize_text(text)
        if not normalized:
            return
        if parts and self._normalize_text(parts[-1]) == normalized:
            return
        parts.append(text.strip())

    def _collapse_repeated_phrase(self, text: str) -> str:
        normalized = self._normalize_text(text)
        if not normalized:
            return ""

        words = normalized.split()
        if len(words) % 2 == 0:
            middle = len(words) // 2
            if words[:middle] == words[middle:]:
                return " ".join(words[:middle])

        return normalized

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.strip().split()).lower()
