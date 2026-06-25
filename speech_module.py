from __future__ import annotations

import os
import audioop
import io
import queue
import re
import tempfile
import threading
import wave
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import perf
import pyaudio

try:
    from faster_whisper import WhisperModel
except ImportError:  # optional local STT backend
    WhisperModel = None

try:
    from supertonic import TTS as SupertonicTTS
except ImportError:  # optional local TTS backend
    SupertonicTTS = None


class SpeechError(RuntimeError):
    """Базовая ошибка речевого модуля."""


class MicrophoneError(SpeechError):
    """Микрофон недоступен или не удалось прочитать аудио."""


class SttUnavailableError(SpeechError):
    """STT-движок недоступен или не смог распознать речь."""


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


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


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
    # Длинные дефолты, чтобы НЕ обрывать речь пользователя: ассистент ждёт
    # до 2.5 c тишины после фразы и допускает команды до 20 c.
    # Влияет только на запись команды; wake/interrupt используют свои короткие окна.
    phrase_time_limit_seconds: float = field(default_factory=lambda: _env_float("STT_PHRASE_TIME_LIMIT", 20.0))
    silence_limit_seconds: float = field(default_factory=lambda: _env_float("STT_SILENCE_TIMEOUT", 2.5))
    start_timeout_seconds: float = 5.0
    input_device_index: int | None = field(default_factory=lambda: _env_int("AI_ASSISTANT_INPUT_DEVICE_INDEX"))
    output_device_index: int | None = field(default_factory=lambda: _env_int("AI_ASSISTANT_OUTPUT_DEVICE_INDEX"))
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
    # GPU для Supertonic (ONNX): требует onnxruntime-gpu + CUDA. При недоступности
    # CUDA движок сам откатывается на CPU.
    supertonic_gpu: bool = field(default_factory=lambda: _env_bool("SUPERTONIC_GPU", False))


class SentenceAssembler:
    """Накапливает стримящийся текст и отдаёт законченные предложения для озвучки."""

    _END_RE = re.compile(r"[.!?…]+[\"»')\]]*\s")

    def __init__(self, min_length: int = 24) -> None:
        self.min_length = min_length
        self._buffer = ""

    def feed(self, piece: str) -> list[str]:
        """Добавляет фрагмент текста, возвращает готовые предложения (возможно пустой список)."""
        self._buffer += piece
        ready: list[str] = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            sentence = self._buffer[:cut].strip()
            self._buffer = self._buffer[cut:]
            if sentence:
                ready.append(sentence)
        return ready

    def flush(self) -> str:
        """Возвращает остаток буфера (хвост ответа без финальной пунктуации)."""
        tail = self._buffer.strip()
        self._buffer = ""
        return tail

    def _find_cut(self) -> int | None:
        search_from = 0
        while True:
            newline = self._buffer.find("\n", search_from)
            match = self._END_RE.search(self._buffer, search_from)

            if newline == -1 and match is None:
                return None

            # Перенос строки — жёсткая граница (конец абзаца/пункта списка).
            if newline != -1 and (match is None or newline < match.end()):
                return newline + 1

            end = match.end()
            # Слишком короткие куски ("1.", "Да.") не отправляем отдельно —
            # ждём продолжения и озвучиваем вместе со следующим предложением.
            if len(self._buffer[:end].strip()) >= self.min_length:
                return end
            search_from = end


class TtsStream:
    """Конвейер озвучки: синтез следующего предложения идёт во время воспроизведения текущего."""

    def __init__(self, module: "SpeechModule") -> None:
        self._module = module
        self._text_queue: queue.Queue[str | None] = queue.Queue()
        self._wav_queue: queue.Queue[Path | None] = queue.Queue(maxsize=3)
        self.cancelled = threading.Event()
        self.finished = threading.Event()
        self._closed = False
        self._speak_notified = False
        threading.Thread(target=self._synth_worker, daemon=True).start()
        threading.Thread(target=self._play_worker, daemon=True).start()

    def wait(self, timeout: float | None = None) -> bool:
        """Блокируется до полного завершения воспроизведения очереди."""
        return self.finished.wait(timeout)

    def add(self, sentence: str) -> None:
        if sentence and sentence.strip() and not self.cancelled.is_set():
            self._text_queue.put(sentence.strip())

    def close(self) -> None:
        """Сообщает, что текста больше не будет; очередь доигрывает и завершается."""
        if not self._closed:
            self._closed = True
            self._text_queue.put(None)

    def cancel(self) -> None:
        """Останавливает озвучку: текущее воспроизведение и всю очередь."""
        self.cancelled.set()
        self.close()

    def _synth_worker(self) -> None:
        while True:
            text = self._text_queue.get()
            if text is None:
                self._wav_queue.put(None)
                return
            if self.cancelled.is_set():
                continue
            try:
                with self._module.tts_lock:
                    if self.cancelled.is_set():
                        continue
                    wav_path = self._module.synthesize_to_file(text)
            except Exception:
                continue
            self._wav_queue.put(wav_path)

    def _play_worker(self) -> None:
        while True:
            wav_path = self._wav_queue.get()
            if wav_path is None:
                if self._speak_notified:
                    self._speak_notified = False
                    self._module._notify_speak_end()
                self.finished.set()
                return
            try:
                if not self.cancelled.is_set():
                    if not self._speak_notified:
                        self._speak_notified = True
                        self._module._notify_speak_start()
                    self._module._play_wav_audio(wav_path, extra_stop=self.cancelled)
            except Exception:
                pass
            finally:
                try:
                    wav_path.unlink(missing_ok=True)
                except OSError:
                    pass


class SpeechModule:
    """Offline speech-to-text and local text-to-speech through Supertonic 3."""

    def __init__(self, config: SpeechConfig | None = None) -> None:
        self.config = config or SpeechConfig()
        self.audio = pyaudio.PyAudio()
        self.tts_lock = threading.Lock()
        self.whisper_lock = threading.Lock()
        self.stop_playback_event = threading.Event()
        self.supertonic_tts = None
        self.supertonic_voice_style = None
        self._active_tts_stream: TtsStream | None = None

        # Кэш заранее синтезированных коротких фраз ("Слушаю" и т.п.).
        self._phrase_cache: dict[str, Path] = {}
        # Замеры последней голосовой команды (заполняются в _speech_to_text_whisper).
        self.last_record_finished_monotonic: float | None = None
        self.last_record_seconds: float = 0.0
        self.last_stt_seconds: float = 0.0

        self.tts_available = self._is_tts_available()
        self.stt_backend = self.config.stt_backend
        self.whisper_model = None
        self.stt_available = self._is_stt_available()
        self.input_device_index, self.input_device_name, self.active_sample_rate = self._resolve_input_device()
        self.output_device_index, self.output_device_name = self._resolve_output_device()

        # Колбэки для внешних потребителей (например, MetaHuman-моста):
        # вызываются из рабочих потоков озвучки, не из потока UI.
        self.on_speak_start: Callable[[], None] | None = None
        self.on_speak_end: Callable[[], None] | None = None
        self.on_audio_chunk: Callable[[bytes, int, int], None] | None = None

    def speech_to_text(self, start_timeout: float | None = None) -> str:
        if self.stt_backend in {"faster_whisper", "whisper"}:
            return self._speech_to_text_whisper(start_timeout=start_timeout)
        raise RuntimeError(f"Неизвестный STT backend: {self.stt_backend}")

    def _speech_to_text_whisper(self, start_timeout: float | None = None) -> str:
        if WhisperModel is None:
            raise RuntimeError("Пакет faster-whisper не установлен. Выполните: pip install faster-whisper")
        if self.input_device_index is None:
            raise RuntimeError("Микрофон не найден. Выберите доступное входное устройство.")

        record_start = perf.now()
        audio_bytes = self._record_phrase(start_timeout=start_timeout)
        self.last_record_finished_monotonic = perf.now()
        self.last_record_seconds = self.last_record_finished_monotonic - record_start
        perf.log("Command recording finished", self.last_record_seconds * 1000)

        if not audio_bytes:
            raise RuntimeError("Речь не обнаружена. Проверьте микрофон и попробуйте ещё раз.")

        stt_start = perf.now()
        result = self._transcribe_audio_bytes(audio_bytes)
        self.last_stt_seconds = perf.now() - stt_start
        perf.log("STT recognition", self.last_stt_seconds * 1000)

        if not result:
            raise RuntimeError("Не удалось распознать речь через Whisper.")
        return result

    def recognize_short_phrase(
        self,
        *,
        start_timeout: float = 2.5,
        phrase_limit: float = 4.0,
        silence_limit: float = 0.4,
    ) -> str:
        """Слушает короткое окно и возвращает распознанный текст ('' если тишина).

        Используется wake-word детектором: на тишину не бросает исключение,
        а на проблемы с устройством/STT — типизированные ошибки для статусов UI.
        """
        if not self.stt_available:
            raise SttUnavailableError("STT-движок недоступен.")
        if self.input_device_index is None:
            raise MicrophoneError("Микрофон не найден.")

        try:
            audio_bytes = self._record_phrase(
                start_timeout=start_timeout,
                phrase_limit=phrase_limit,
                silence_limit=silence_limit,
                raise_on_silence=False,
            )
        except (MicrophoneError, SttUnavailableError):
            raise
        except Exception as error:
            raise MicrophoneError(str(error)) from error

        if not audio_bytes:
            return ""

        try:
            return self._transcribe_audio_bytes(audio_bytes)
        except Exception as error:
            raise SttUnavailableError(str(error)) from error

    def _transcribe_audio_bytes(self, audio_bytes: bytes) -> str:
        """Упаковывает PCM в WAV в памяти и прогоняет через Whisper (без диска)."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wav_file.setframerate(self.active_sample_rate)
            wav_file.writeframes(audio_bytes)
        buffer.seek(0)
        text = self._transcribe_whisper(buffer)
        return self._collapse_repeated_phrase(text)

    def is_busy(self) -> bool:
        """True, пока активный конвейер озвучки ещё не доиграл."""
        stream = self._active_tts_stream
        return stream is not None and not stream.finished.is_set()

    def wait_until_speech_done(self, timeout: float | None = None) -> None:
        """Блокируется, пока текущая озвучка не завершится (или не выйдет таймаут)."""
        stream = self._active_tts_stream
        if stream is not None:
            stream.wait(timeout)

    def text_to_speech(self, text: str) -> None:
        if not text.strip():
            return
        if not self.tts_available:
            raise RuntimeError("Supertonic 3 не настроен. Установите пакет supertonic.")

        with self.tts_lock:
            self.stop_playback_event.clear()
            self._notify_speak_start()
            try:
                self._speak_supertonic(text)
            finally:
                self._notify_speak_end()

    def text_to_speech_async(self, text: str) -> None:
        if not text.strip() or not self.tts_available:
            return

        def runner() -> None:
            try:
                self.text_to_speech(text)
            except Exception:
                return

        threading.Thread(target=runner, daemon=True).start()

    # --- Кэш коротких фраз ("Слушаю", "Готово" и т.п.) -----------------

    def _phrase_key(self, text: str) -> str:
        return (
            f"{self.config.supertonic_voice}|{self.config.supertonic_lang}|"
            f"{self.config.supertonic_speed}|{self.config.supertonic_steps}|{text.strip().lower()}"
        )

    def prime_phrase(self, text: str) -> None:
        """Заранее синтезирует короткую фразу в кэш, чтобы потом озвучить мгновенно."""
        if not self.tts_available or not text.strip():
            return
        key = self._phrase_key(text)
        with self.tts_lock:
            cached = self._phrase_cache.get(key)
            if cached is not None and cached.exists():
                return
            try:
                self._phrase_cache[key] = self.synthesize_to_file(text)
            except Exception:
                return

    def speak_cached(self, text: str) -> None:
        """Озвучивает фразу из кэша (мгновенно), синтезируя при первом обращении.

        Тем же голосом, что и обычные ответы (Supertonic, config.supertonic_voice).
        """
        if not text.strip():
            return
        if not self.tts_available:
            raise RuntimeError("Supertonic 3 не настроен. Установите пакет supertonic.")
        key = self._phrase_key(text)
        with self.tts_lock:
            self.stop_playback_event.clear()
            path = self._phrase_cache.get(key)
            if path is None or not path.exists():
                path = self.synthesize_to_file(text)
                self._phrase_cache[key] = path
            self._notify_speak_start()
            try:
                self._play_wav_audio(path)
            finally:
                self._notify_speak_end()

    def clear_phrase_cache(self) -> None:
        """Сбрасывает кэш фраз (например, при смене голоса)."""
        for path in self._phrase_cache.values():
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        self._phrase_cache.clear()

    def stop_speaking(self) -> None:
        self.stop_playback_event.set()
        stream = self._active_tts_stream
        if stream is not None:
            stream.cancel()

    def create_tts_stream(self) -> TtsStream:
        """Создаёт конвейер озвучки по предложениям, останавливая предыдущий."""
        previous = self._active_tts_stream
        if previous is not None:
            previous.cancel()
        self.stop_playback_event.clear()
        stream = TtsStream(self)
        self._active_tts_stream = stream
        return stream

    def warm_up_async(self, on_done: Callable[[], None] | None = None) -> None:
        """Загружает модели Whisper и Supertonic в фоне, чтобы первый запрос не ждал."""

        def runner() -> None:
            try:
                if self.stt_available:
                    self._ensure_whisper_model()
            except Exception:
                pass
            try:
                if self.tts_available:
                    with self.tts_lock:
                        self._ensure_supertonic_engine()
            except Exception:
                pass
            if on_done is not None:
                on_done()

        threading.Thread(target=runner, daemon=True).start()

    def shutdown(self) -> None:
        self.stop_speaking()
        self.clear_phrase_cache()
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

    def list_output_devices(self) -> list[dict[str, object]]:
        devices: list[dict[str, object]] = []
        for index in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(index)
            if int(info.get("maxOutputChannels", 0)) <= 0:
                continue
            devices.append(
                {
                    "index": index,
                    "name": str(info.get("name", f"Device {index}")),
                    "max_output_channels": int(info.get("maxOutputChannels", 0)),
                    "default_sample_rate": int(float(info.get("defaultSampleRate", self.config.sample_rate))),
                }
            )
        return devices

    def set_output_device(self, device_index: int | None) -> None:
        previous = self.config.output_device_index
        self.config.output_device_index = device_index
        try:
            self.output_device_index, self.output_device_name = self._resolve_output_device(require_selected=True)
        except Exception:
            self.config.output_device_index = previous
            self.output_device_index, self.output_device_name = self._resolve_output_device()
            raise

    def get_output_device_index(self) -> int | None:
        return self.output_device_index

    def get_output_device_label(self) -> str:
        return self.output_device_name

    def _resolve_output_device(self, require_selected: bool = False) -> tuple[int | None, str]:
        if self.config.output_device_index is None:
            return None, "Системное устройство по умолчанию"

        for device in self.list_output_devices():
            if device["index"] == self.config.output_device_index:
                return int(device["index"]), str(device["name"])

        if require_selected:
            raise RuntimeError(
                f"Устройство вывода с индексом {self.config.output_device_index} не найдено."
            )
        self.config.output_device_index = None
        return None, "Системное устройство по умолчанию"

    def _notify_speak_start(self) -> None:
        callback = self.on_speak_start
        if callback is not None:
            try:
                callback()
            except Exception:
                pass

    def _notify_speak_end(self) -> None:
        callback = self.on_speak_end
        if callback is not None:
            try:
                callback()
            except Exception:
                pass

    def _notify_audio_chunk(self, pcm: bytes, sample_rate: int, channels: int) -> None:
        callback = self.on_audio_chunk
        if callback is not None:
            try:
                callback(pcm, sample_rate, channels)
            except Exception:
                pass

    def _is_tts_available(self) -> bool:
        return SupertonicTTS is not None

    def _is_stt_available(self) -> bool:
        if self.stt_backend in {"faster_whisper", "whisper"}:
            return WhisperModel is not None
        return False

    def _register_cuda_dll_dirs(self) -> None:
        """Делает видимыми CUDA-DLL (cuBLAS/cuDNN) из pip-пакетов nvidia-*-cu12.

        ctranslate2 на Windows ищет cublas64_12.dll / cudnn*.dll в PATH. Если CUDA
        Toolkit не установлен глобально, но стоят колёса nvidia-cublas-cu12 /
        nvidia-cudnn-cu12 — добавляем их bin-папки, чтобы не править PATH вручную.
        """
        if os.name != "nt" or not hasattr(os, "add_dll_directory"):
            return
        site_packages = Path(__file__).resolve().parent / ".venv-win" / "Lib" / "site-packages"
        roots = [site_packages] if site_packages.exists() else []
        # На случай иного расположения venv — ищем nvidia/ в путях импорта.
        try:
            import nvidia  # type: ignore
            roots.extend(Path(p).parent for p in nvidia.__path__)
        except Exception:
            pass
        extra_paths: list[str] = []
        for root in roots:
            for bin_dir in root.glob("nvidia/*/bin"):
                path_str = str(bin_dir)
                extra_paths.append(path_str)
                try:
                    os.add_dll_directory(path_str)
                except (OSError, FileNotFoundError):
                    pass
        # ctranslate2 грузит cublas64_12.dll по имени — для этого нужен PATH,
        # одного add_dll_directory недостаточно.
        if extra_paths:
            os.environ["PATH"] = os.pathsep.join(extra_paths) + os.pathsep + os.environ.get("PATH", "")

    def _ensure_whisper_model(self) -> None:
        if WhisperModel is None:
            raise RuntimeError("Пакет faster-whisper не установлен. Выполните: pip install faster-whisper")
        with self.whisper_lock:
            if self.whisper_model is None:
                if "cuda" in self.config.whisper_device.lower():
                    self._register_cuda_dll_dirs()
                self.whisper_model = WhisperModel(
                    self.config.whisper_model,
                    device=self.config.whisper_device,
                    compute_type=self.config.whisper_compute_type,
                )

    def _record_phrase(
        self,
        *,
        start_timeout: float | None = None,
        phrase_limit: float | None = None,
        silence_limit: float | None = None,
        raise_on_silence: bool = True,
    ) -> bytes:
        phrase_time_limit = self.config.phrase_time_limit_seconds if phrase_limit is None else phrase_limit
        start_timeout_seconds = self.config.start_timeout_seconds if start_timeout is None else start_timeout
        silence_limit_seconds = self.config.silence_limit_seconds if silence_limit is None else silence_limit

        max_chunks = max(
            1,
            int(phrase_time_limit * self.active_sample_rate / self.config.chunk_size),
        )
        start_timeout_chunks = max(
            1,
            int(start_timeout_seconds * self.active_sample_rate / self.config.chunk_size),
        )
        silence_limit_chunks = max(
            1,
            int(silence_limit_seconds * self.active_sample_rate / self.config.chunk_size),
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
                    if raise_on_silence:
                        raise RuntimeError("Речь не обнаружена. Проверьте микрофон и попробуйте ещё раз.")
                    return b""

                if speech_started and silence_chunks >= silence_limit_chunks:
                    break

        return b"".join(frames)

    def _transcribe_whisper(self, audio) -> str:
        """audio: путь к WAV (str) или файлоподобный объект (io.BytesIO)."""
        self._ensure_whisper_model()
        source = str(audio) if isinstance(audio, (str, Path)) else audio
        segments, _info = self.whisper_model.transcribe(
            source,
            language=self.config.whisper_language,
            beam_size=max(1, self.config.whisper_beam_size),
            vad_filter=self.config.whisper_vad_filter,
            condition_on_previous_text=False,
            initial_prompt=self.config.whisper_initial_prompt or None,
            temperature=0.0,
        )
        return " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()

    def _enable_supertonic_gpu(self) -> None:
        """Включает CUDAExecutionProvider для Supertonic.

        SupertonicTTS не принимает device/providers, а список провайдеров в пакете
        захардкожен на CPU. Поэтому подменяем DEFAULT_ONNX_PROVIDERS в загрузчике
        до создания движка. Если CUDA-провайдер недоступен (нет onnxruntime-gpu),
        загрузчик сам отфильтрует его и останется на CPU.
        """
        try:
            import onnxruntime as ort
            from supertonic import loader as supertonic_loader
        except Exception as error:
            print(f"[TTS] GPU недоступен, остаюсь на CPU: {error}", flush=True)
            return
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            print(
                "[TTS] onnxruntime-gpu/CUDA не найден — Supertonic на CPU. "
                "Установите onnxruntime-gpu для ускорения.",
                flush=True,
            )
            return
        supertonic_loader.DEFAULT_ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("[TTS] Supertonic: включён CUDAExecutionProvider (GPU).", flush=True)

    def _ensure_supertonic_engine(self) -> None:
        if SupertonicTTS is None:
            raise RuntimeError("Пакет supertonic не установлен. Выполните: pip install supertonic")
        if self.supertonic_tts is None:
            if self.config.supertonic_gpu:
                self._enable_supertonic_gpu()
            self.supertonic_tts = SupertonicTTS(model_dir=self.config.supertonic_model_dir, auto_download=False)
        if self.supertonic_voice_style is None:
            self.supertonic_voice_style = self.supertonic_tts.get_voice_style(
                voice_name=self.config.supertonic_voice
            )

    def synthesize_to_file(self, text: str) -> Path:
        """Синтезирует речь в временный WAV-файл и возвращает путь к нему."""
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

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        self.supertonic_tts.save_audio(wav, str(temp_path))
        return temp_path

    def _speak_supertonic(self, text: str) -> None:
        temp_path = self.synthesize_to_file(text)
        try:
            self._play_wav_audio(temp_path)
        finally:
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

    def _open_output_stream(self, audio_format: int, channels: int, rate: int, frames_per_buffer: int):
        """Открывает выходной поток. Если устройство выбрано явно, не откатывается молча на системное."""
        if self.output_device_index is not None:
            try:
                return self.audio.open(
                    format=audio_format,
                    channels=channels,
                    rate=rate,
                    output=True,
                    output_device_index=self.output_device_index,
                    frames_per_buffer=frames_per_buffer,
                )
            except Exception as error:
                raise RuntimeError(
                    f"Не удалось открыть устройство вывода '{self.output_device_name}' "
                    f"(index={self.output_device_index}, rate={rate}, channels={channels}). "
                    f"Выберите другой CABLE Input в настройках озвучки. Ошибка PyAudio: {error}"
                ) from error
        return self.audio.open(
            format=audio_format,
            channels=channels,
            rate=rate,
            output=True,
            frames_per_buffer=frames_per_buffer,
        )

    def _play_pcm_audio(self, audio_bytes: bytes, sample_rate: int) -> None:
        stream = self._open_output_stream(pyaudio.paInt16, 1, sample_rate, 2048)
        try:
            chunk_size = 4096
            for start in range(0, len(audio_bytes), chunk_size):
                if self.stop_playback_event.is_set():
                    break
                chunk = audio_bytes[start : start + chunk_size]
                self._notify_audio_chunk(chunk, sample_rate, 1)
                stream.write(chunk)
        finally:
            stream.stop_stream()
            stream.close()

    def _play_wav_audio(self, wav_path: Path, extra_stop: threading.Event | None = None) -> None:
        with wave.open(str(wav_path), "rb") as wav_file:
            source_width = wav_file.getsampwidth()
            source_channels = wav_file.getnchannels()
            source_rate = wav_file.getframerate()

            if self.output_device_index is None:
                stream = self._open_output_stream(
                    self.audio.get_format_from_width(source_width),
                    source_channels,
                    source_rate,
                    1024,
                )
                try:
                    while not self.stop_playback_event.is_set():
                        if extra_stop is not None and extra_stop.is_set():
                            break
                        chunk = wav_file.readframes(1024)
                        if not chunk:
                            break
                        pcm_chunk = chunk if source_width == 2 else audioop.lin2lin(chunk, source_width, 2)
                        self._notify_audio_chunk(pcm_chunk, source_rate, source_channels)
                        stream.write(chunk)
                finally:
                    stream.stop_stream()
                    stream.close()
                return

            device_info = self.audio.get_device_info_by_index(self.output_device_index)
            target_rate = int(float(device_info.get("defaultSampleRate", source_rate)))
            target_channels = min(2, max(1, int(device_info.get("maxOutputChannels", source_channels))))

            pcm = wav_file.readframes(wav_file.getnframes())
            if source_width != 2:
                pcm = audioop.lin2lin(pcm, source_width, 2)
                source_width = 2

            if source_rate != target_rate:
                pcm, _state = audioop.ratecv(pcm, source_width, source_channels, source_rate, target_rate, None)
                source_rate = target_rate

            if source_channels == 1 and target_channels == 2:
                pcm = audioop.tostereo(pcm, source_width, 1.0, 1.0)
                source_channels = 2
            elif source_channels == 2 and target_channels == 1:
                pcm = audioop.tomono(pcm, source_width, 0.5, 0.5)
                source_channels = 1

            stream = self._open_output_stream(pyaudio.paInt16, source_channels, source_rate, 1024)
            try:
                frame_width = source_width * source_channels
                chunk_bytes = 1024 * frame_width
                for start in range(0, len(pcm), chunk_bytes):
                    if self.stop_playback_event.is_set():
                        break
                    if extra_stop is not None and extra_stop.is_set():
                        break
                    chunk = pcm[start : start + chunk_bytes]
                    self._notify_audio_chunk(chunk, source_rate, source_channels)
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


