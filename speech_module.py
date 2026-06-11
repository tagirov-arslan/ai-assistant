from __future__ import annotations

import os
import tempfile
import threading
import wave
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pyaudio

try:
    from faster_whisper import WhisperModel
except ImportError:  # optional local STT backend
    WhisperModel = None

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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on", "да"}


def _default_whisper_model() -> str:
    """Local CTranslate2 build of whisper-large-v3-turbo, with hub fallback."""
    local_model = PROJECT_DIR / "whisper-large-v3-turbo-ct2"
    if (local_model / "model.bin").exists():
        return str(local_model)
    return "large-v3-turbo"


@dataclass
class SpeechConfig:
    language: str = "ru-RU"
    sample_rate: int = 16000
    chunk_size: int = 1024
    energy_threshold: int = 80
    phrase_time_limit_seconds: float = 8.0
    silence_limit_seconds: float = 0.45
    start_timeout_seconds: float = 5.0
    input_device_index: int | None = field(default_factory=lambda: _env_int("AI_ASSISTANT_INPUT_DEVICE_INDEX"))
    stt_backend: str = field(default_factory=lambda: os.getenv("AI_ASSISTANT_STT_BACKEND", "faster_whisper").strip().lower() or "faster_whisper")
    whisper_model: str = field(default_factory=lambda: os.getenv("AI_ASSISTANT_WHISPER_MODEL", "").strip() or _default_whisper_model())
    whisper_device: str = field(default_factory=lambda: os.getenv("AI_ASSISTANT_WHISPER_DEVICE", "cpu").strip() or "cpu")
    whisper_compute_type: str = field(default_factory=lambda: os.getenv("AI_ASSISTANT_WHISPER_COMPUTE_TYPE", "int8").strip() or "int8")
    whisper_language: str = field(default_factory=lambda: os.getenv("AI_ASSISTANT_WHISPER_LANGUAGE", "ru").strip() or "ru")
    whisper_beam_size: int = field(default_factory=lambda: _env_int("AI_ASSISTANT_WHISPER_BEAM_SIZE") or 1)
    whisper_vad_filter: bool = field(default_factory=lambda: _env_bool("AI_ASSISTANT_WHISPER_VAD_FILTER", True))
    whisper_initial_prompt: str = field(default_factory=lambda: os.getenv("AI_ASSISTANT_WHISPER_INITIAL_PROMPT", "").strip())
    supertonic_model_dir: str = field(default_factory=lambda: os.getenv("SUPERTONIC_MODEL_DIR", str(PROJECT_DIR / "supertonic-3-model")).strip())
    supertonic_voice: str = field(default_factory=lambda: os.getenv("SUPERTONIC_VOICE", "F1").strip() or "F1")
    supertonic_lang: str = field(default_factory=lambda: os.getenv("SUPERTONIC_LANG", "ru").strip() or "ru")
    supertonic_steps: int = field(default_factory=lambda: _env_int("SUPERTONIC_STEPS") or 8)
    supertonic_speed: float = field(default_factory=lambda: float(os.getenv("SUPERTONIC_SPEED", "1.0") or "1.0"))


class SpeechModule:
    """Offline speech-to-text and local text-to-speech through Supertonic 3."""

    def __init__(self, config: SpeechConfig | None = None) -> None:
        self.config = config or SpeechConfig()
        self.audio = pyaudio.PyAudio()
        self.tts_lock = threading.Lock()
        self.stop_playback_event = threading.Event()
        self.supertonic_tts = None
        self.supertonic_voice_style = None

        self.tts_available = self._is_tts_available()
        self.stt_backend = self.config.stt_backend
        self.whisper_model = None
        self.stt_available = self._is_stt_available()
        self.input_device_index, self.input_device_name, self.active_sample_rate = self._resolve_input_device()

    def speech_to_text(self) -> str:
        if self.stt_backend in {"faster_whisper", "whisper"}:
            return self._speech_to_text_whisper()
        raise RuntimeError(f"Неизвестный STT backend: {self.stt_backend}")

    def _speech_to_text_whisper(self) -> str:
        if WhisperModel is None:
            raise RuntimeError("Пакет faster-whisper не установлен. Выполните: pip install faster-whisper")
        if self.input_device_index is None:
            raise RuntimeError("Микрофон не найден. Выберите доступное входное устройство.")

        audio_bytes = self._record_phrase()
        if not audio_bytes:
            raise RuntimeError("Речь не обнаружена. Проверьте микрофон и попробуйте ещё раз.")

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
            self._write_wav(temp_path, audio_bytes, self.active_sample_rate)
            text = self._transcribe_whisper_file(temp_path)
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

        result = self._collapse_repeated_phrase(text)
        if not result:
            raise RuntimeError("Не удалось распознать речь через Whisper.")
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
        if self.stt_backend in {"faster_whisper", "whisper"}:
            if WhisperModel is None:
                return "Whisper: пакет faster-whisper не установлен"
            return (
                f"Whisper: {self.config.whisper_model} | "
                f"{self.config.whisper_device}/{self.config.whisper_compute_type}"
            )
        return f"STT: неизвестный backend {self.stt_backend}"

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

    def _is_stt_available(self) -> bool:
        if self.stt_backend in {"faster_whisper", "whisper"}:
            return WhisperModel is not None
        return False

    def _ensure_whisper_model(self) -> None:
        if WhisperModel is None:
            raise RuntimeError("Пакет faster-whisper не установлен. Выполните: pip install faster-whisper")
        if self.whisper_model is None:
            self.whisper_model = WhisperModel(
                self.config.whisper_model,
                device=self.config.whisper_device,
                compute_type=self.config.whisper_compute_type,
            )

    def _record_phrase(self) -> bytes:
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
        frames: list[bytes] = []
        pre_roll: list[bytes] = []
        pre_roll_limit = max(1, int(0.25 * self.active_sample_rate / self.config.chunk_size))

        with self._open_input_stream(self.active_sample_rate, self.config.chunk_size) as stream:
            for index in range(max_chunks):
                data = stream.read(self.config.chunk_size, exception_on_overflow=False)
                energy = self._estimate_energy(data)

                if not speech_started:
                    pre_roll.append(data)
                    if len(pre_roll) > pre_roll_limit:
                        pre_roll.pop(0)

                if energy >= self.config.energy_threshold:
                    if not speech_started:
                        speech_started = True
                        frames.extend(pre_roll)
                    frames.append(data)
                    silence_chunks = 0
                elif speech_started:
                    frames.append(data)
                    silence_chunks += 1

                if not speech_started and index >= start_timeout_chunks:
                    raise RuntimeError("Речь не обнаружена. Проверьте микрофон и попробуйте ещё раз.")

                if speech_started and silence_chunks >= silence_limit_chunks:
                    break

        return b"".join(frames)

    def _write_wav(self, wav_path: Path, audio_bytes: bytes, sample_rate: int) -> None:
        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)

    def _transcribe_whisper_file(self, wav_path: Path) -> str:
        self._ensure_whisper_model()
        segments, _info = self.whisper_model.transcribe(
            str(wav_path),
            language=self.config.whisper_language,
            beam_size=max(1, self.config.whisper_beam_size),
            vad_filter=self.config.whisper_vad_filter,
            condition_on_previous_text=False,
            initial_prompt=self.config.whisper_initial_prompt or None,
            temperature=0.0,
        )
        return " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()

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

            preferred = ["microphone", "микрофон", "mic input", "array"]
            avoided = [
                "line in",
                "line-in",
                "линейный",
                "лин. вход",
                "stereo mix",
                "стерео микшер",
                "speaker",
                "output",
                "hands-free",
                "переназначение",
            ]

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

    def _estimate_energy(self, audio_bytes: bytes) -> int:
        samples = memoryview(audio_bytes).cast("h")
        if not samples:
            return 0
        return int(sum(abs(sample) for sample in samples) / len(samples))

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
