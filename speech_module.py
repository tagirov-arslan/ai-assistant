from __future__ import annotations

import json
import os
import queue
import threading
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pyaudio
from vosk import KaldiRecognizer, Model, SetLogLevel


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
    chunk_size: int = 4000
    energy_threshold: int = 80
    phrase_time_limit_seconds: float = 8.0
    silence_limit_seconds: float = 1.2
    start_timeout_seconds: float = 5.0
    input_device_index: int | None = field(default_factory=lambda: _env_int("AI_ASSISTANT_INPUT_DEVICE_INDEX"))
    vosk_model_path: str = field(default_factory=lambda: os.getenv("AI_ASSISTANT_VOSK_MODEL_PATH", "").strip())
    elevenlabs_api_key: str = field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", "").strip())
    elevenlabs_voice_id: str = field(default_factory=lambda: os.getenv("ELEVENLABS_VOICE_ID", "").strip())
    elevenlabs_model_id: str = field(default_factory=lambda: os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5").strip() or "eleven_flash_v2_5")
    elevenlabs_output_format: str = field(default_factory=lambda: os.getenv("ELEVENLABS_OUTPUT_FORMAT", "pcm_16000").strip() or "pcm_16000")


class SpeechModule:
    """Offline speech-to-text through Vosk and online text-to-speech through ElevenLabs."""

    def __init__(self, config: SpeechConfig | None = None) -> None:
        self.config = config or SpeechConfig()
        self.audio = pyaudio.PyAudio()
        self.tts_lock = threading.Lock()
        self.stop_playback_event = threading.Event()

        self.tts_available = bool(self.config.elevenlabs_api_key and self.config.elevenlabs_voice_id)

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
        if not text.strip():
            return
        if not self.tts_available:
            raise RuntimeError("ElevenLabs не настроен. Укажите API key в .env и voice ID в assistant_settings.json.")

        with self.tts_lock:
            self.stop_playback_event.clear()
            sample_rate = self._resolve_output_sample_rate(self.config.elevenlabs_output_format)
            self._stream_elevenlabs_audio(text, sample_rate)

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
        if not self.config.elevenlabs_api_key:
            return "ElevenLabs: нет API key в .env"
        if not self.config.elevenlabs_voice_id:
            return "ElevenLabs: нет voice_id в assistant_settings.json"
        return (
            f"ElevenLabs | voice_id: {self.config.elevenlabs_voice_id} | "
            f"model: {self.config.elevenlabs_model_id}"
        )

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

    def _request_elevenlabs_audio(self, text: str) -> bytes:
        voice_id = urllib.parse.quote(self.config.elevenlabs_voice_id, safe="")
        output_format = urllib.parse.quote(self.config.elevenlabs_output_format, safe="")
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format={output_format}"

        payload = {
            "text": text,
            "model_id": self.config.elevenlabs_model_id,
            "voice_settings": {
                "stability": 0.4,
                "similarity_boost": 0.8,
                "style": 0.15,
                "use_speaker_boost": True,
                "speed": 1.0,
            },
        }

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "xi-api-key": self.config.elevenlabs_api_key,
                "Content-Type": "application/json",
                "Accept": "audio/pcm",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ошибка ElevenLabs ({error.code}): {body}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Не удалось подключиться к ElevenLabs: {error.reason}") from error

    def _stream_elevenlabs_audio(self, text: str, sample_rate: int) -> None:
        voice_id = urllib.parse.quote(self.config.elevenlabs_voice_id, safe="")
        output_format = urllib.parse.quote(self.config.elevenlabs_output_format, safe="")
        url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
            f"?output_format={output_format}&optimize_streaming_latency=3"
        )

        payload = {
            "text": text,
            "model_id": self.config.elevenlabs_model_id,
            "voice_settings": {
                "stability": 0.35,
                "similarity_boost": 0.8,
                "style": 0.1,
                "use_speaker_boost": True,
                "speed": 1.03,
            },
        }

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "xi-api-key": self.config.elevenlabs_api_key,
                "Content-Type": "application/json",
                "Accept": "audio/pcm",
            },
            method="POST",
        )

        audio_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=32)
        playback_error: list[Exception] = []

        def playback_worker() -> None:
            try:
                self._play_pcm_audio_stream(audio_queue, sample_rate)
            except Exception as error:
                playback_error.append(error)

        playback_thread = threading.Thread(target=playback_worker, daemon=True)
        playback_thread.start()

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                while not self.stop_playback_event.is_set():
                    chunk = response.read(4096)
                    if not chunk:
                        break
                    audio_queue.put(chunk)
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ошибка ElevenLabs ({error.code}): {body}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Не удалось подключиться к ElevenLabs: {error.reason}") from error
        finally:
            audio_queue.put(None)
            playback_thread.join()

        if playback_error:
            raise RuntimeError(f"Ошибка воспроизведения ElevenLabs: {playback_error[0]}") from playback_error[0]

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

    def _play_pcm_audio_stream(self, audio_queue: queue.Queue[bytes | None], sample_rate: int) -> None:
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            output=True,
            frames_per_buffer=1024,
        )
        try:
            while not self.stop_playback_event.is_set():
                chunk = audio_queue.get()
                if chunk is None:
                    break
                if chunk:
                    stream.write(chunk)
        finally:
            stream.stop_stream()
            stream.close()

    def _resolve_output_sample_rate(self, output_format: str) -> int:
        parts = output_format.lower().split("_")
        if len(parts) >= 2 and parts[0] == "pcm":
            try:
                return int(parts[1])
            except ValueError:
                pass
        return 24000

    def _resolve_input_device(self, require_selected: bool = False) -> tuple[int, str, int]:
        devices = self.list_input_devices()
        if not devices:
            raise RuntimeError("В системе не найдено ни одного входного аудиоустройства.")

        if self.config.input_device_index is not None:
            for device in devices:
                if device["index"] == self.config.input_device_index:
                    rate = self._find_openable_sample_rate(
                        int(device["index"]),
                        int(device["default_sample_rate"]),
                    )
                    if rate is None:
                        raise RuntimeError(
                            f"Микрофон с индексом {self.config.input_device_index} найден, "
                            "но его не удалось открыть."
                        )
                    return int(device["index"]), str(device["name"]), rate
            if require_selected:
                raise RuntimeError(f"Микрофон с индексом {self.config.input_device_index} не найден.")
            self.config.input_device_index = None

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

    def _resolve_vosk_model_path(self, explicit_path: str) -> Path:
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
