from __future__ import annotations

import json
import os
import tempfile
import threading
import wave
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pyaudio
from vosk import KaldiRecognizer, Model, SetLogLevel

try:
    from supertonic import TTS as SupertonicTTS
except ImportError:  # optional local TTS backend
    SupertonicTTS = None


PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"


def _load_local_env() -> None:
    if not ENV_PATH.exists():
        return

    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip().lstrip("\ufeff")
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_local_env()


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
    sample_rate: int = 16000
    chunk_size: int = 2000
    energy_threshold: int = 80
    phrase_time_limit_seconds: float = 8.0
    silence_limit_seconds: float = 0.55
    start_timeout_seconds: float = 5.0
    input_device_index: int | None = field(default_factory=lambda: _env_int("AI_ASSISTANT_INPUT_DEVICE_INDEX"))
    vosk_model_path: str = field(default_factory=lambda: os.getenv("AI_ASSISTANT_VOSK_MODEL_PATH", "").strip())
    supertonic_model_dir: str = field(default_factory=lambda: os.getenv("SUPERTONIC_MODEL_DIR", str(PROJECT_DIR / "supertonic-3-model")).strip())
    supertonic_voice: str = field(default_factory=lambda: os.getenv("SUPERTONIC_VOICE", "F1").strip() or "F1")
    supertonic_lang: str = field(default_factory=lambda: os.getenv("SUPERTONIC_LANG", "ru").strip() or "ru")
    supertonic_steps: int = field(default_factory=lambda: _env_int("SUPERTONIC_STEPS") or 8)
    supertonic_speed: float = field(default_factory=lambda: float(os.getenv("SUPERTONIC_SPEED", "1.0") or "1.0"))


class SpeechModule:
    """Offline speech-to-text through Vosk and local text-to-speech through Supertonic 3."""

    def __init__(self, config: SpeechConfig | None = None) -> None:
        self.config = config or SpeechConfig()
        self.audio = pyaudio.PyAudio()
        self.tts_lock = threading.Lock()
        self.stop_playback_event = threading.Event()
        self.supertonic_tts = None
        self.supertonic_voice_style = None

        self.tts_available = self._is_tts_available()

        SetLogLevel(-1)
        self.vosk_model_path = self._resolve_vosk_model_path(self.config.vosk_model_path)
        self.vosk_model = self._load_vosk_model(self.vosk_model_path)
        self.stt_available = self.vosk_model is not None
        self.input_device_index, self.input_device_name, self.active_sample_rate = self._resolve_input_device()

    def speech_to_text(self) -> str:
        if self.vosk_model is None:
            raise RuntimeError("Vosk-модель не найдена. Укажите путь через AI_ASSISTANT_VOSK_MODEL_PATH.")
        if self.input_device_index is None:
            raise RuntimeError("Микрофон не найден. Выберите доступное входное устройство.")

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
        if not text.strip():
            return
        if not self.tts_available:
            raise RuntimeError("Supertonic 3 не настроен. Установите пакет supertonic.")

        with self.tts_lock:
            self.stop_playback_event.clear()
            self._speak_supertonic(text)

    def text_to_speech_async(self, text: str) -> None:
        if not text.strip() or not self.tts_available:
            return

        def runner() -> None:
            try:
                self.text_to_speech(text)
            except Exception:
                return

        threading.Thread(target=runner, daemon=True).start()

    def stop_speaking(self) -> None:
        self.stop_playback_event.set()

    def shutdown(self) -> None:
        self.stop_speaking()
        self.audio.terminate()

    def get_tts_status_label(self) -> str:
        if SupertonicTTS is None:
            return "Supertonic 3: пакет не установлен"
        return (
            f"Supertonic 3 | voice: {self.config.supertonic_voice} | "
            f"lang: {self.config.supertonic_lang} | steps: {self.config.supertonic_steps}"
        )

    def get_stt_status_label(self) -> str:
        if self.stt_available:
            return f"Vosk: {self.vosk_model_path}"
        return "Vosk: модель не найдена"

    def set_input_device(self, device_index: int | None) -> None:
        previous_config_index = self.config.input_device_index
        previous_input_index = self.input_device_index
        previous_input_name = self.input_device_name
        previous_sample_rate = self.active_sample_rate

        self.config.input_device_index = device_index
        try:
            (
                self.input_device_index,
                self.input_device_name,
                self.active_sample_rate,
            ) = self._resolve_input_device(require_selected=True)
        except Exception:
            self.config.input_device_index = previous_config_index
            self.input_device_index = previous_input_index
            self.input_device_name = previous_input_name
            self.active_sample_rate = previous_sample_rate
            raise

    def get_input_device_index(self) -> int | None:
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

    def _is_tts_available(self) -> bool:
        return SupertonicTTS is not None

    def _ensure_supertonic_engine(self) -> None:
        if SupertonicTTS is None:
            raise RuntimeError("Пакет supertonic не установлен. Выполните: pip install supertonic")
        if self.supertonic_tts is None:
            self.supertonic_tts = SupertonicTTS(model_dir=self.config.supertonic_model_dir, auto_download=False)
        if self.supertonic_voice_style is None:
            self.supertonic_voice_style = self.supertonic_tts.get_voice_style(
                voice_name=self.config.supertonic_voice
            )

    def _speak_supertonic(self, text: str) -> None:
        self._ensure_supertonic_engine()
        wav, _duration = self.supertonic_tts.synthesize(
            text=text,
            voice_style=self.supertonic_voice_style,
            total_steps=max(5, min(12, self.config.supertonic_steps)),
            speed=max(0.7, min(2.0, self.config.supertonic_speed)),
            max_chunk_length=300,
            silence_duration=0.25,
            lang=self.config.supertonic_lang,
            verbose=False,
        )

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
            self.supertonic_tts.save_audio(wav, str(temp_path))
            self._play_wav_audio(temp_path)
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

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

    def _play_pcm_audio(self, audio_bytes: bytes, sample_rate: int) -> None:
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            output=True,
            frames_per_buffer=2048,
        )
        try:
            chunk_size = 4096
            for start in range(0, len(audio_bytes), chunk_size):
                if self.stop_playback_event.is_set():
                    break
                chunk = audio_bytes[start : start + chunk_size]
                stream.write(chunk)
        finally:
            stream.stop_stream()
            stream.close()

    def _play_wav_audio(self, wav_path: Path) -> None:
        with wave.open(str(wav_path), "rb") as wav_file:
            stream = self.audio.open(
                format=self.audio.get_format_from_width(wav_file.getsampwidth()),
                channels=wav_file.getnchannels(),
                rate=wav_file.getframerate(),
                output=True,
                frames_per_buffer=1024,
            )
            try:
                while not self.stop_playback_event.is_set():
                    chunk = wav_file.readframes(1024)
                    if not chunk:
                        break
                    stream.write(chunk)
            finally:
                stream.stop_stream()
                stream.close()

    def _resolve_input_device(self, require_selected: bool = False) -> tuple[int | None, str, int]:
        devices = self.list_input_devices()
        if not devices:
            if require_selected:
                raise RuntimeError("В системе не найдено ни одного входного аудиоустройства.")
            return None, "Микрофон не найден", self.config.sample_rate

        if self.config.input_device_index is not None:
            for device in devices:
                if device["index"] == self.config.input_device_index:
                    rate = self._find_openable_sample_rate(
                        int(device["index"]),
                        int(device["default_sample_rate"]),
                    )
                    if rate is None:
                        if require_selected:
                            raise RuntimeError(
                                f"Микрофон с индексом {self.config.input_device_index} найден, "
                                "но его не удалось открыть."
                            )
                        self.config.input_device_index = None
                        break
                    return int(device["index"]), str(device["name"]), rate
            if require_selected:
                raise RuntimeError(f"Микрофон с индексом {self.config.input_device_index} не найден.")
            self.config.input_device_index = None

        try:
            default_info = self.audio.get_default_input_device_info()
        except OSError:
            default_info = None

        if default_info is not None:
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

        if require_selected:
            raise RuntimeError("Не удалось открыть ни один микрофон через PyAudio.")
        return None, "Микрофон не найден", self.config.sample_rate

    def _pick_sample_rate_for_device(self, device_index: int, default_sample_rate: int) -> int:
        rate = self._find_openable_sample_rate(device_index, default_sample_rate)
        if rate is None:
            return int(default_sample_rate)
        return rate

    def _find_openable_sample_rate(self, device_index: int, default_sample_rate: int) -> int | None:
        preferred_rates = [self.config.sample_rate, 16000, 22050, 32000, default_sample_rate, 44100, 48000]
        checked: set[int] = set()
        for rate in preferred_rates:
            sample_rate = int(rate)
            if sample_rate in checked:
                continue
            checked.add(sample_rate)
            if self._can_open_device(device_index, sample_rate):
                return sample_rate
        return None

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
        except Exception:
            return False

    def _resolve_vosk_model_path(self, explicit_path: str) -> Path | None:
        candidates: list[Path] = []

        if explicit_path:
            candidates.append(Path(explicit_path))

        candidates.extend(
            [
                PROJECT_DIR / "models" / "vosk-model-small-ru-0.22",
                PROJECT_DIR / "resources" / "vosk" / "vosk-model-small-ru-0.22",
                Path(r"C:\Users\Macinery_knr\Desktop\jarvis-master\resources\vosk\vosk-model-small-ru-0.22"),
            ]
        )

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return None

    def _load_vosk_model(self, model_path: Path | None) -> Model | None:
        if model_path is None:
            return None
        try:
            return Model(str(model_path))
        except Exception:
            # Fail-safe for fresh PCs: keep text chat usable when the optional local STT model is missing or corrupt.
            return None

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
