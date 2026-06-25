from __future__ import annotations

import json
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

import perf
from lm_studio_module import LMStudioConfig, LMStudioModule
from metahuman_bridge import MetaHumanBridge
from speech_module import SentenceAssembler, SpeechConfig, SpeechModule
from interrupt_listener import InterruptListener
from wake_word_listener import WakeWordListener


DEFAULT_WAKE_WORDS = ["Сталин", "Эй, Сталин", "Слушай Сталин"]
DEFAULT_INTERRUPT_WORDS = ["стоп", "остановись", "хватит"]


LM_CONFIG = LMStudioConfig(
    base_url="http://127.0.0.1:1234",
    api_key="lm-studio",
    model="google/gemma-4-e2b",
    temperature=0.7,
    max_tokens=2000,
    system_prompt="Ты полезный ИИ-ассистент. Отвечай кратко, понятно и на русском языке.",
)

SETTINGS_PATH = Path(__file__).resolve().parent / "assistant_settings.json"


class AssistantApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ИИ-ассистент")
        self.root.geometry("1120x820")
        self.root.minsize(1040, 760)
        self.root.configure(bg="#dfe7f1")

        self._configure_styles()

        self.llm_module = LMStudioModule(LM_CONFIG)
        self.settings = self._load_settings()

        # Производительность: замеры этапов обработки после записи.
        perf.set_enabled(bool(self.settings.get("performance_log_enabled", True)))
        # LLM: ограничение истории, чтобы промпт не рос без предела.
        self.llm_max_history_messages = int(self.settings.get("llm_max_history_messages", 10))
        LM_CONFIG.max_tokens = int(self.settings.get("llm_max_tokens", LM_CONFIG.max_tokens))
        # Тайминг старта воспроизведения текущего ответа (для [Perf]).
        self._answer_perf_start: float | None = None
        self._answer_playback_logged = False
        self._answer_llm_first_token_at: float | None = None

        self.speech_module = SpeechModule(self._build_speech_config())
        active_input_index = self.speech_module.get_input_device_index()
        if self.settings.get("input_device_index") != active_input_index:
            self.settings["input_device_index"] = active_input_index
            self._save_settings()

        self.metahuman_bridge = MetaHumanBridge(
            host=str(self.settings.get("osc_host", "127.0.0.1")),
            port=int(self.settings.get("osc_port", 9000)),
            enabled=bool(self.settings.get("metahuman_enabled", False)),
        )
        # Колбэки приходят из потоков озвучки — UI обновляем через root.after.
        self.speech_module.on_speak_start = self._on_speak_start_hook
        self.speech_module.on_speak_end = lambda: (
            self.metahuman_bridge.speak_end(),
            self.root.after(0, self._refresh_metahuman_label),
        )
        self.speech_module.on_audio_chunk = self.metahuman_bridge.send_audio_chunk

        # Wake word: фоновый слушатель ключевой фразы поверх существующих STT/TTS/LLM.
        self._manual_capture_active = False
        self.wake_word_timeout = float(self.settings.get("wake_word_timeout", 5.0))
        self.command_listen_timeout = float(self.settings.get("command_listen_timeout", 8.0))
        self.wake_reply_enabled = bool(self.settings.get("wake_word_reply_enabled", True))
        self.wake_reply_text = str(self.settings.get("wake_word_reply_text", "Слушаю")) or "Слушаю"
        self.interrupt_enabled = bool(self.settings.get("interrupt_enabled", True))
        self.interrupt_by_wake_word = bool(self.settings.get("interrupt_by_wake_word", True))
        self.interrupt_listen_during_tts = bool(self.settings.get("interrupt_listen_during_tts", True))

        # Состояние текущего ответа для прерывания (создаётся на каждый ответ).
        self._answer_interrupt: str | None = None
        self._answer_cancel = threading.Event()
        self._answer_done = threading.Event()

        self.wake_status_text = (
            "Ожидание wake word" if bool(self.settings.get("wake_word_enabled", True)) else "Выключен"
        )
        self.wake_listener = WakeWordListener(
            wake_words=list(self.settings.get("wake_words", DEFAULT_WAKE_WORDS)),
            stt_service=self.speech_module.recognize_short_phrase,
            on_wake_detected=self._on_wake_detected,
            on_status=self._set_wake_status,
            should_listen=self._wake_should_listen,
            listen_interval=float(self.settings.get("wake_word_listen_interval", 3.0)),
            language=str(self.settings.get("wake_word_language", "ru")),
            enabled=bool(self.settings.get("wake_word_enabled", True)),
        )
        self.interrupt_listener = InterruptListener(
            interrupt_words=list(self.settings.get("interrupt_words", DEFAULT_INTERRUPT_WORDS)),
            wake_words=(
                list(self.settings.get("wake_words", DEFAULT_WAKE_WORDS))
                if self.interrupt_by_wake_word
                else []
            ),
            stt_service=self._recognize_interrupt_phrase,
            on_stop_detected=self._handle_interrupt_stop,
            on_wake_detected=self._handle_interrupt_wake,
        )

        self.conversation_history = [{"role": "system", "content": LM_CONFIG.system_prompt}]
        self.status_var = tk.StringVar(value="Готово к работе")
        self.mic_var = tk.StringVar(value=self.speech_module.get_input_device_label())
        self.voice_var = tk.StringVar(value=self.speech_module.get_tts_status_label())
        self.metahuman_var = tk.StringVar(value=self._metahuman_label_text())
        self.wake_var = tk.StringVar(value=self._wake_label_text())

        self._build_ui()
        self.root.update_idletasks()

        self.add_message("assistant", "Здравствуйте! Напишите сообщение или используйте голосовой ввод.")
        self.add_message("system", f"Активный микрофон: {self.speech_module.get_input_device_label()}")
        self.add_message("system", f"Распознавание: {self.speech_module.get_stt_status_label()}")
        self.add_message("system", f"Озвучивание: {self.speech_module.get_tts_status_label()}")
        if not self.speech_module.tts_available:
            self.add_message(
                "system",
                "Для Supertonic 3 установите пакет supertonic. Настройки голоса хранятся в assistant_settings.json.",
            )

        # Прогрев моделей речи в фоне, чтобы первый голосовой запрос и первая озвучка не ждали загрузки.
        self._set_status("Загружаю модели речи в фоне...")
        self.speech_module.warm_up_async(
            on_done=lambda: self.root.after(0, self._on_models_ready)
        )

    def _on_models_ready(self) -> None:
        self._set_status("Готово к работе")
        # Предсинтез фразы "Слушаю" в кэш, чтобы после wake word она звучала мгновенно.
        if self.wake_reply_enabled and self.wake_reply_text:
            threading.Thread(
                target=lambda: self.speech_module.prime_phrase(self.wake_reply_text),
                daemon=True,
            ).start()
        self._start_wake_word_if_enabled()

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure(".", font=("Segoe UI", 10))
        style.configure("Root.TFrame", background="#dfe7f1")
        style.configure("Sidebar.TFrame", background="#1d2733")
        style.configure("Main.TFrame", background="#f7f9fc")
        style.configure("Topbar.TFrame", background="#ffffff")
        style.configure("Composer.TFrame", background="#ffffff")
        style.configure("SidebarCard.TFrame", background="#243140")
        style.configure("DialogCard.TFrame", background="#ffffff")

        style.configure(
            "SidebarTitle.TLabel",
            background="#1d2733",
            foreground="#ffffff",
            font=("Segoe UI Semibold", 22),
        )
        style.configure(
            "SidebarSub.TLabel",
            background="#1d2733",
            foreground="#9eb0c3",
            font=("Segoe UI", 10),
        )
        style.configure(
            "SidebarCardTitle.TLabel",
            background="#243140",
            foreground="#ffffff",
            font=("Segoe UI Semibold", 10),
        )
        style.configure(
            "SidebarInfo.TLabel",
            background="#243140",
            foreground="#c9d6e3",
            font=("Segoe UI", 10),
        )
        style.configure(
            "TopTitle.TLabel",
            background="#ffffff",
            foreground="#17212b",
            font=("Segoe UI Semibold", 18),
        )
        style.configure(
            "TopSub.TLabel",
            background="#ffffff",
            foreground="#71808f",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Status.TLabel",
            background="#ffffff",
            foreground="#4d5b6a",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Primary.TButton",
            background="#2b6be6",
            foreground="#ffffff",
            padding=(14, 8),
            borderwidth=0,
            focusthickness=0,
            focuscolor="#2b6be6",
            font=("Segoe UI Semibold", 10),
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#245ecc"), ("pressed", "#1e51b1")],
        )
        style.configure(
            "Secondary.TButton",
            background="#eef3f8",
            foreground="#203142",
            padding=(12, 8),
            borderwidth=0,
            focusthickness=0,
            focuscolor="#eef3f8",
            font=("Segoe UI Semibold", 10),
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#e1eaf3"), ("pressed", "#d8e3ee")],
        )
        style.configure(
            "Ghost.TButton",
            background="#243140",
            foreground="#ffffff",
            padding=(12, 8),
            borderwidth=0,
            focusthickness=0,
            focuscolor="#243140",
            font=("Segoe UI Semibold", 10),
        )
        style.map(
            "Ghost.TButton",
            background=[("active", "#2a394b"), ("pressed", "#1b2633")],
        )

    def _default_settings(self) -> dict[str, object]:
        return {
            "input_device_index": None,
            "output_device_index": None,
            "supertonic_voice": "F1",
            "supertonic_lang": "ru",
            "supertonic_steps": 8,
            "supertonic_speed": 1.0,
            "metahuman_enabled": False,
            "osc_host": "127.0.0.1",
            "osc_port": 9000,
            "wake_word_enabled": True,
            "wake_words": list(DEFAULT_WAKE_WORDS),
            "wake_word_language": "ru",
            "wake_word_listen_interval": 3.0,
            "wake_word_timeout": 5.0,
            "wake_word_reply_enabled": True,
            "wake_word_reply_text": "Слушаю",
            "interrupt_enabled": True,
            "interrupt_words": list(DEFAULT_INTERRUPT_WORDS),
            "interrupt_by_wake_word": True,
            "interrupt_listen_during_tts": True,
            "command_listen_timeout": 15.0,
            "performance_log_enabled": True,
            "llm_max_history_messages": 10,
            "llm_max_tokens": LM_CONFIG.max_tokens,
            # Ускорители. Whisper: "cpu"/"cuda" + "int8"/"float16"/"int8_float16".
            # Supertonic GPU требует onnxruntime-gpu (иначе авто-откат на CPU).
            "whisper_device": "cpu",
            "whisper_compute_type": "int8",
            "supertonic_gpu": False,
        }

    def _load_settings(self) -> dict[str, object]:
        settings = self._default_settings()
        if not SETTINGS_PATH.exists():
            return settings

        try:
            raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return settings

        if isinstance(raw.get("input_device_index"), int):
            settings["input_device_index"] = raw["input_device_index"]
        if isinstance(raw.get("output_device_index"), int):
            settings["output_device_index"] = raw["output_device_index"]
        if isinstance(raw.get("metahuman_enabled"), bool):
            settings["metahuman_enabled"] = raw["metahuman_enabled"]
        if isinstance(raw.get("osc_host"), str) and raw["osc_host"].strip():
            settings["osc_host"] = raw["osc_host"].strip()
        if isinstance(raw.get("osc_port"), int):
            settings["osc_port"] = raw["osc_port"]
        for key in [
            "supertonic_voice",
            "supertonic_lang",
        ]:
            value = raw.get(key)
            if isinstance(value, str):
                settings[key] = value
        if isinstance(raw.get("supertonic_steps"), int):
            settings["supertonic_steps"] = raw["supertonic_steps"]
        if isinstance(raw.get("supertonic_speed"), (int, float)):
            settings["supertonic_speed"] = float(raw["supertonic_speed"])
        if isinstance(raw.get("wake_word_enabled"), bool):
            settings["wake_word_enabled"] = raw["wake_word_enabled"]
        if isinstance(raw.get("wake_words"), list):
            words = [w.strip() for w in raw["wake_words"] if isinstance(w, str) and w.strip()]
            if words:
                settings["wake_words"] = words
        if isinstance(raw.get("wake_word_language"), str) and raw["wake_word_language"].strip():
            settings["wake_word_language"] = raw["wake_word_language"].strip()
        if isinstance(raw.get("wake_word_listen_interval"), (int, float)) and raw["wake_word_listen_interval"] > 0:
            settings["wake_word_listen_interval"] = float(raw["wake_word_listen_interval"])
        if isinstance(raw.get("wake_word_timeout"), (int, float)) and raw["wake_word_timeout"] > 0:
            settings["wake_word_timeout"] = float(raw["wake_word_timeout"])
        if isinstance(raw.get("wake_word_reply_enabled"), bool):
            settings["wake_word_reply_enabled"] = raw["wake_word_reply_enabled"]
        if isinstance(raw.get("wake_word_reply_text"), str) and raw["wake_word_reply_text"].strip():
            settings["wake_word_reply_text"] = raw["wake_word_reply_text"].strip()
        if isinstance(raw.get("interrupt_enabled"), bool):
            settings["interrupt_enabled"] = raw["interrupt_enabled"]
        if isinstance(raw.get("interrupt_words"), list):
            stop_words = [w.strip() for w in raw["interrupt_words"] if isinstance(w, str) and w.strip()]
            if stop_words:
                settings["interrupt_words"] = stop_words
        if isinstance(raw.get("interrupt_by_wake_word"), bool):
            settings["interrupt_by_wake_word"] = raw["interrupt_by_wake_word"]
        if isinstance(raw.get("interrupt_listen_during_tts"), bool):
            settings["interrupt_listen_during_tts"] = raw["interrupt_listen_during_tts"]
        if isinstance(raw.get("command_listen_timeout"), (int, float)) and raw["command_listen_timeout"] > 0:
            settings["command_listen_timeout"] = float(raw["command_listen_timeout"])
        if isinstance(raw.get("performance_log_enabled"), bool):
            settings["performance_log_enabled"] = raw["performance_log_enabled"]
        if isinstance(raw.get("llm_max_history_messages"), int) and raw["llm_max_history_messages"] >= 0:
            settings["llm_max_history_messages"] = raw["llm_max_history_messages"]
        if isinstance(raw.get("llm_max_tokens"), int) and raw["llm_max_tokens"] > 0:
            settings["llm_max_tokens"] = raw["llm_max_tokens"]
        if isinstance(raw.get("whisper_device"), str) and raw["whisper_device"].strip():
            settings["whisper_device"] = raw["whisper_device"].strip()
        if isinstance(raw.get("whisper_compute_type"), str) and raw["whisper_compute_type"].strip():
            settings["whisper_compute_type"] = raw["whisper_compute_type"].strip()
        if isinstance(raw.get("supertonic_gpu"), bool):
            settings["supertonic_gpu"] = raw["supertonic_gpu"]
        return settings

    def _save_settings(self) -> None:
        SETTINGS_PATH.write_text(json.dumps(self.settings, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_speech_config(self) -> SpeechConfig:
        config = SpeechConfig(
            input_device_index=(
                self.settings.get("input_device_index")
                if isinstance(self.settings.get("input_device_index"), int)
                else None
            ),
            output_device_index=(
                self.settings.get("output_device_index")
                if isinstance(self.settings.get("output_device_index"), int)
                else None
            ),
            supertonic_voice=str(self.settings.get("supertonic_voice", "F1")) or "F1",
            supertonic_lang=str(self.settings.get("supertonic_lang", "ru")) or "ru",
            supertonic_steps=(
                int(self.settings.get("supertonic_steps", 8))
                if isinstance(self.settings.get("supertonic_steps"), int)
                else 8
            ),
            supertonic_speed=(
                float(self.settings.get("supertonic_speed", 1.0))
                if isinstance(self.settings.get("supertonic_speed"), (int, float))
                else 1.0
            ),
        )
        # Ускорители (GPU) — задаются из settings.json; если ключа нет, остаётся
        # значение из .env / дефолт (CPU). Так настройка .env не перетирается молча.
        device = self.settings.get("whisper_device")
        if isinstance(device, str) and device.strip():
            config.whisper_device = device.strip()
        compute = self.settings.get("whisper_compute_type")
        if isinstance(compute, str) and compute.strip():
            config.whisper_compute_type = compute.strip()
        if isinstance(self.settings.get("supertonic_gpu"), bool):
            config.supertonic_gpu = self.settings["supertonic_gpu"]
        return config

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, style="Root.TFrame", padding=18)
        outer.pack(fill=tk.BOTH, expand=True)

        sidebar = ttk.Frame(outer, style="Sidebar.TFrame", padding=20)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.configure(width=290)
        sidebar.pack_propagate(False)

        main = ttk.Frame(outer, style="Main.TFrame")
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(sidebar, text="ИИ-ассистент", style="SidebarTitle.TLabel").pack(anchor="w")
        ttk.Label(
            sidebar,
            text="Онлайн",
            style="SidebarSub.TLabel",
            wraplength=230,
        ).pack(anchor="w", pady=(6, 18))

        mic_card = ttk.Frame(sidebar, style="SidebarCard.TFrame", padding=14)
        mic_card.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(mic_card, text="Микрофон", style="SidebarCardTitle.TLabel").pack(anchor="w")
        ttk.Label(mic_card, textvariable=self.mic_var, style="SidebarInfo.TLabel", wraplength=220).pack(
            anchor="w",
            pady=(8, 0),
        )

        voice_card = ttk.Frame(sidebar, style="SidebarCard.TFrame", padding=14)
        voice_card.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(voice_card, text="Озвучивание", style="SidebarCardTitle.TLabel").pack(anchor="w")
        ttk.Label(voice_card, textvariable=self.voice_var, style="SidebarInfo.TLabel", wraplength=220).pack(
            anchor="w",
            pady=(8, 0),
        )

        metahuman_card = ttk.Frame(sidebar, style="SidebarCard.TFrame", padding=14)
        metahuman_card.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(metahuman_card, text="MetaHuman", style="SidebarCardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            metahuman_card,
            textvariable=self.metahuman_var,
            style="SidebarInfo.TLabel",
            wraplength=220,
        ).pack(anchor="w", pady=(8, 0))

        wake_card = ttk.Frame(sidebar, style="SidebarCard.TFrame", padding=14)
        wake_card.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(wake_card, text="Wake word", style="SidebarCardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            wake_card,
            textvariable=self.wake_var,
            style="SidebarInfo.TLabel",
            wraplength=220,
        ).pack(anchor="w", pady=(8, 0))

        ttk.Button(
            sidebar,
            text="Wake word вкл/выкл",
            style="Ghost.TButton",
            command=self.toggle_wake_word,
        ).pack(fill=tk.X, pady=(6, 0))
        ttk.Button(
            sidebar,
            text="MetaHuman вкл/выкл",
            style="Ghost.TButton",
            command=self.toggle_metahuman,
        ).pack(fill=tk.X, pady=(10, 0))
        ttk.Button(
            sidebar,
            text="Устройство вывода озвучки",
            style="Ghost.TButton",
            command=self.open_output_device_dialog,
        ).pack(fill=tk.X, pady=(10, 0))
        ttk.Button(sidebar, text="Выбрать голос", style="Ghost.TButton", command=self.open_voice_dialog).pack(
            fill=tk.X,
            pady=(10, 0),
        )
        ttk.Button(sidebar, text="Выбрать микрофон", style="Ghost.TButton", command=self.open_microphone_dialog).pack(
            fill=tk.X,
            pady=(10, 0),
        )
        ttk.Button(sidebar, text="Очистить диалог", style="Ghost.TButton", command=self.clear_chat).pack(
            fill=tk.X,
            pady=(10, 0),
        )

        topbar = ttk.Frame(main, style="Topbar.TFrame", padding=(22, 18))
        topbar.pack(fill=tk.X)
        ttk.Label(topbar, text="Диалог", style="TopTitle.TLabel").pack(anchor="w")

        dialog_shell = ttk.Frame(main, style="DialogCard.TFrame", padding=(18, 14))
        dialog_shell.pack(fill=tk.BOTH, expand=True, padx=18, pady=(16, 12))

        self.chat_area = scrolledtext.ScrolledText(
            dialog_shell,
            wrap=tk.WORD,
            state="disabled",
            bd=0,
            relief=tk.FLAT,
            padx=12,
            pady=12,
            font=("Segoe UI", 11),
            background="#ffffff",
            foreground="#1d2733",
            insertbackground="#1d2733",
            selectbackground="#cfe0ff",
        )
        self.chat_area.pack(fill=tk.BOTH, expand=True)
        self._configure_chat_tags()

        composer_shell = ttk.Frame(main, style="Composer.TFrame", padding=(18, 10))
        composer_shell.pack(fill=tk.X, padx=18, pady=(0, 12))

        composer_row = ttk.Frame(composer_shell, style="Composer.TFrame")
        composer_row.pack(fill=tk.X)

        self.input_entry = ttk.Entry(composer_row, font=("Segoe UI", 11))
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10), ipady=6)
        self.input_entry.bind("<Return>", lambda event: self.handle_send())
        self.input_entry.focus_set()

        actions = ttk.Frame(composer_row, style="Composer.TFrame")
        actions.pack(side=tk.RIGHT)

        ttk.Button(actions, text="Отправить", style="Primary.TButton", command=self.handle_send).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )
        ttk.Button(actions, text="Голос", style="Secondary.TButton", command=self.handle_voice_input).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )
        ttk.Button(actions, text="Микрофон", style="Secondary.TButton", command=self.open_microphone_dialog).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )
        ttk.Button(actions, text="Очистить", style="Secondary.TButton", command=self.clear_chat).pack(side=tk.LEFT)

        ttk.Label(composer_shell, textvariable=self.status_var, style="Status.TLabel").pack(anchor="w", pady=(8, 0))

    def _configure_chat_tags(self) -> None:
        self.chat_area.tag_configure(
            "assistant_header",
            foreground="#245ecc",
            font=("Segoe UI Semibold", 10),
            spacing1=12,
        )
        self.chat_area.tag_configure(
            "assistant_body",
            foreground="#1f2933",
            background="#eef4ff",
            lmargin1=18,
            lmargin2=18,
            rmargin=90,
            spacing1=2,
            spacing3=14,
        )
        self.chat_area.tag_configure(
            "user_header",
            foreground="#4a5b6d",
            font=("Segoe UI Semibold", 10),
            justify="right",
            spacing1=12,
        )
        self.chat_area.tag_configure(
            "user_body",
            foreground="#ffffff",
            background="#2b6be6",
            lmargin1=90,
            lmargin2=90,
            rmargin=18,
            justify="right",
            spacing1=2,
            spacing3=14,
        )
        self.chat_area.tag_configure(
            "system_header",
            foreground="#6f7f90",
            font=("Segoe UI Semibold", 9),
            spacing1=10,
        )
        self.chat_area.tag_configure(
            "system_body",
            foreground="#475464",
            background="#f2f5f8",
            lmargin1=48,
            lmargin2=48,
            rmargin=48,
            spacing1=2,
            spacing3=12,
        )
        self.chat_area.tag_configure(
            "time",
            foreground="#8a96a3",
            font=("Segoe UI", 9),
        )

    def _get_input_text(self) -> str:
        return self.input_entry.get().strip()

    def _clear_input(self) -> None:
        self.input_entry.delete(0, tk.END)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _refresh_device_labels(self) -> None:
        self.mic_var.set(self.speech_module.get_input_device_label())
        self.voice_var.set(self.speech_module.get_tts_status_label())
        self._refresh_metahuman_label()

    def _metahuman_label_text(self) -> str:
        if not self.metahuman_bridge.enabled:
            return "Выключен"
        state_names = {
            "idle": "ожидание",
            "listening": "слушает",
            "thinking": "думает",
            "speaking": "говорит",
        }
        state = state_names.get(self.metahuman_bridge.get_state(), "ожидание")
        return (
            f"OSC {self.metahuman_bridge.host}:{self.metahuman_bridge.port} | {state}\n"
            f"Вывод TTS: {self.speech_module.get_output_device_label()}"
        )

    def _refresh_metahuman_label(self) -> None:
        self.metahuman_var.set(self._metahuman_label_text())

    def _set_metahuman_state(self, state: str) -> None:
        """Шлёт состояние в UE и обновляет подпись. Безопасно из любого потока."""
        self.metahuman_bridge.set_state(state)
        self.root.after(0, self._refresh_metahuman_label)

    def toggle_metahuman(self) -> None:
        enabled = not self.metahuman_bridge.enabled
        self.metahuman_bridge.set_enabled(enabled)
        self.settings["metahuman_enabled"] = enabled
        self._save_settings()
        self._refresh_metahuman_label()
        self.add_message(
            "system",
            "MetaHuman-мост включён: OSC-команды идут на "
            f"{self.metahuman_bridge.host}:{self.metahuman_bridge.port}."
            if enabled
            else "MetaHuman-мост выключен.",
        )

    # --- Wake word -----------------------------------------------------

    def _wake_label_text(self) -> str:
        if not self.wake_listener.enabled:
            return "Выключен"
        words = " · ".join(self.wake_listener.wake_words[:3]) or "—"
        return f"{self.wake_status_text}\nФразы: {words}"

    def _refresh_wake_label(self) -> None:
        self.wake_var.set(self._wake_label_text())

    def _set_wake_status(self, status: str) -> None:
        """Обновляет статус wake word. Безопасно из любого потока."""
        self.wake_status_text = status
        self.root.after(0, self._refresh_wake_label)

    def _add_message_async(self, role: str, text: str) -> None:
        self.root.after(0, lambda: self.add_message(role, text))

    def _wake_should_listen(self) -> bool:
        """Слушатель не должен писать с микрофона, пока ассистент занят."""
        return not self._manual_capture_active and not self.speech_module.is_busy()

    def _start_wake_word_if_enabled(self) -> None:
        if not self.wake_listener.enabled:
            self._set_wake_status("Выключен")
            return
        if not self.speech_module.stt_available:
            self.wake_listener.enabled = False
            self._set_wake_status("STT недоступен")
            self.add_message("system", "Wake word отключён: STT-движок недоступен.")
            return
        self.wake_listener.start()
        self._set_wake_status("Ожидание wake word")
        self.add_message(
            "system",
            f"Wake word активен. Скажите «{self.wake_listener.wake_words[0]}», чтобы дать команду голосом.",
        )

    def toggle_wake_word(self) -> None:
        enabled = not self.wake_listener.enabled
        if enabled and not self.speech_module.stt_available:
            messagebox.showerror("Ошибка", "STT-движок недоступен — wake word работать не будет.")
            return
        self.wake_listener.set_enabled(enabled)
        self.settings["wake_word_enabled"] = enabled
        self._save_settings()
        self._refresh_wake_label()
        self.add_message("system", "Wake word включён." if enabled else "Wake word выключен.")

    def _recognize_interrupt_phrase(self) -> str:
        """Короткое окно прослушивания для детектора прерывания (снаппи)."""
        return self.speech_module.recognize_short_phrase(
            start_timeout=1.2,
            phrase_limit=2.2,
            silence_limit=0.3,
        )

    def _handle_interrupt_stop(self) -> None:
        """Колбэк прерывания по стоп-слову (из потока interrupt-слушателя)."""
        print("[Interrupt] Assistant speech interrupted", flush=True)
        self._answer_interrupt = "stop"
        self._answer_cancel.set()
        try:
            self.speech_module.stop_speaking()
        except Exception as error:
            print(f"[Interrupt] Failed to stop TTS: {error}", flush=True)
        self._answer_done.set()

    def _handle_interrupt_wake(self) -> None:
        """Колбэк прерывания по повторному wake word (из потока interrupt-слушателя)."""
        print("[Interrupt] Assistant speech interrupted", flush=True)
        self._answer_interrupt = "wake"
        self._answer_cancel.set()
        try:
            self.speech_module.stop_speaking()
        except Exception as error:
            print(f"[Interrupt] Failed to stop TTS: {error}", flush=True)
        self._answer_done.set()

    def _on_speak_start_hook(self) -> None:
        """Старт воспроизведения речи: MetaHuman + замер задержки ответа."""
        self.metahuman_bridge.speak_start()
        if self._answer_perf_start is not None and not self._answer_playback_logged:
            self._answer_playback_logged = True
            if self._answer_llm_first_token_at is not None:
                perf.log_since("Audio playback start (after first token)", self._answer_llm_first_token_at)
            perf.log_since("Full processing after recording", self._answer_perf_start)
        self.root.after(0, self._refresh_metahuman_label)

    def _speak_wake_reply(self) -> None:
        """Короткий ответ-подтверждение ('Слушаю') — из кэша TTS, мгновенно."""
        if not self.wake_reply_enabled:
            return
        reply = self.wake_reply_text
        self._set_wake_status("Слушаю")
        self.root.after(0, lambda: self._set_status("Слушаю"))
        print(f"[WakeWord] Reply: {reply}", flush=True)
        self._add_message_async("assistant", reply)
        if not self.speech_module.tts_available:
            return
        reply_start = perf.now()
        try:
            # speak_cached синтезирует один раз и далее проигрывает из кэша.
            self.speech_module.speak_cached(reply)
        except Exception as error:
            self._set_wake_status("Ошибка TTS")
            print(f"[WakeWord] Reply TTS error: {error}", flush=True)
            return
        perf.log_since("Wake reply spoken", reply_start)

    def _answer_with_interrupt(self, text: str) -> str:
        """Генерирует и озвучивает ответ, слушая прерывание. Возвращает 'wake'|'stop'|'done'."""
        self._answer_interrupt = None
        self._answer_cancel = threading.Event()
        self._answer_done = threading.Event()
        cancel_event = self._answer_cancel
        done_event = self._answer_done

        # Якорь замеров: конец записи команды (обработка считается от него).
        self._reset_answer_perf(self.speech_module.last_record_finished_monotonic)

        def worker() -> None:
            try:
                self._stream_llm_round(text, cancel_event=cancel_event)
                self.speech_module.wait_until_speech_done(timeout=300)
            finally:
                done_event.set()

        self.root.after(0, lambda: self._set_status("Озвучиваю ответ..."))
        self._set_wake_status("Озвучиваю ответ")

        watch = (
            self.interrupt_enabled
            and self.interrupt_listen_during_tts
            and self.speech_module.stt_available
            and self.interrupt_listener is not None
        )
        if watch:
            self.interrupt_listener.start_during_tts()

        threading.Thread(target=worker, args=(), daemon=True).start()
        done_event.wait(timeout=600)

        if self.interrupt_listener is not None:
            self.interrupt_listener.stop()

        outcome = self._answer_interrupt or "done"
        if outcome == "stop":
            self._set_wake_status("Ответ прерван")
            self._add_message_async("system", "Ответ прерван.")
            print("[Interrupt] Returning to idle mode", flush=True)
        elif outcome == "wake":
            self._set_wake_status("Ответ прерван")
            print("[Interrupt] Switching to new command mode", flush=True)
        return outcome

    def _on_wake_detected(self) -> None:
        """Полный голосовой цикл после ключевой фразы. Выполняется в потоке слушателя.

        Сценарий: ответить 'Слушаю' → принять команду → ответить с возможностью
        прерывания. Повторный wake word во время ответа запускает новый цикл,
        стоп-слово — возврат в ожидание.
        """
        if self._manual_capture_active:
            return
        self._manual_capture_active = True
        try:
            # На всякий случай прерываем текущую озвучку перед записью команды.
            self.speech_module.stop_speaking()
            self.metahuman_bridge.speak_end()

            need_command = True
            while need_command:
                need_command = False

                # 1. Короткий ответ-подтверждение "Слушаю".
                self._speak_wake_reply()
                print("[WakeWord] Waiting for command...", flush=True)

                # 2. Приём команды через полноценный STT.
                # Пользователю даём договорить полностью (длинные таймауты записи).
                self.root.after(0, lambda: self._set_status("Слушаю команду..."))
                self._set_wake_status("Пользователь говорит")
                self._set_metahuman_state("listening")
                try:
                    text = self.speech_module.speech_to_text(start_timeout=self.command_listen_timeout)
                except Exception as error:
                    self._set_metahuman_state("idle")
                    self.root.after(0, lambda: self._set_status("Готов"))
                    self._add_message_async("system", f"Команда не распознана: {error}")
                    self._set_wake_status("Ошибка STT")
                    break

                self._set_wake_status("Команда записана")
                print(f"[WakeWord] Command recognized: {text}", flush=True)
                self._add_message_async("user", text)
                self.root.after(0, lambda: self._set_status("Ассистент отвечает..."))
                self._set_wake_status("Генерирую ответ")

                # 3. Ответ с прослушиванием прерывания.
                outcome = self._answer_with_interrupt(text)
                if outcome == "wake":
                    # Повторный wake word во время ответа → новый цикл команды.
                    need_command = True
        finally:
            self._manual_capture_active = False
            self._set_metahuman_state("idle")
            self.root.after(0, lambda: self._set_status("Готово к работе"))
            # Статус "Ожидание wake word" выставит цикл слушателя на следующей итерации.
            self._set_wake_status("Возврат в режим ожидания")

    def open_output_device_dialog(self) -> None:
        devices = self.speech_module.list_output_devices()
        if not devices:
            messagebox.showerror("Ошибка", "В системе нет доступных устройств вывода.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Устройство вывода озвучки")
        dialog.geometry("740x280")
        dialog.configure(bg="#dfe7f1")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        shell = ttk.Frame(dialog, style="DialogCard.TFrame", padding=18)
        shell.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        ttk.Label(shell, text="Устройство вывода озвучки", style="TopTitle.TLabel").pack(anchor="w")
        ttk.Label(
            shell,
            text=(
                f"Текущее: {self.speech_module.get_output_device_label()}. "
                "Для lip sync в Unreal выберите «CABLE Input (VB-Audio Virtual Cable)»."
            ),
            style="TopSub.TLabel",
            wraplength=660,
        ).pack(anchor="w", pady=(6, 14))

        values = ["— | Системное устройство по умолчанию"] + [
            f"{device['index']} | {device['name']} | {device['default_sample_rate']} Hz"
            for device in devices
        ]
        selected_value = tk.StringVar()
        current_index = self.speech_module.get_output_device_index()
        selected_value.set(
            next((value for value in values if value.startswith(f"{current_index} |")), values[0])
        )

        combo = ttk.Combobox(shell, textvariable=selected_value, values=values, state="readonly")
        combo.pack(fill=tk.X)

        buttons = ttk.Frame(shell, style="DialogCard.TFrame")
        buttons.pack(fill=tk.X, pady=(18, 0))

        def save_selection() -> None:
            selection = selected_value.get().strip()
            if not selection:
                return
            try:
                head = selection.split("|", 1)[0].strip()
                device_index = None if head == "—" else int(head)
                self.speech_module.set_output_device(device_index)
                self.settings["output_device_index"] = device_index
                self._save_settings()
                self._refresh_device_labels()
            except Exception as error:
                messagebox.showerror("Ошибка", f"Не удалось переключить устройство вывода: {error}")
                return

            self.add_message(
                "system",
                f"Озвучка выводится в: {self.speech_module.get_output_device_label()}",
            )
            self._set_status("Устройство вывода обновлено")
            dialog.destroy()

        ttk.Button(buttons, text="Сохранить", style="Primary.TButton", command=save_selection).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Отмена", style="Secondary.TButton", command=dialog.destroy).pack(
            side=tk.RIGHT,
            padx=(0, 8),
        )

    def _available_supertonic_voices(self) -> list[str]:
        voices_dir = Path(__file__).resolve().parent / "supertonic-3-model" / "voice_styles"
        if voices_dir.exists():
            voices = sorted(path.stem for path in voices_dir.glob("*.json"))
            if voices:
                return voices
        return ["F1", "F2", "F3", "F4", "M1", "M2", "M3", "M4"]

    def open_voice_dialog(self) -> None:
        voices = self._available_supertonic_voices()
        dialog = tk.Toplevel(self.root)
        dialog.title("Выбор голоса")
        dialog.geometry("520x300")
        dialog.configure(bg="#dfe7f1")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        shell = ttk.Frame(dialog, style="DialogCard.TFrame", padding=18)
        shell.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        ttk.Label(shell, text="Выбор голоса", style="TopTitle.TLabel").pack(anchor="w")
        ttk.Label(
            shell,
            text=f"Текущий голос: {self.settings.get('supertonic_voice', 'F1')}",
            style="TopSub.TLabel",
            wraplength=460,
        ).pack(anchor="w", pady=(6, 14))

        selected_voice = tk.StringVar(value=str(self.settings.get("supertonic_voice", "F1")))
        combo = ttk.Combobox(shell, textvariable=selected_voice, values=voices, state="readonly")
        combo.pack(fill=tk.X)

        ttk.Label(
            shell,
            text="После сохранения этот голос будет использоваться для озвучивания ответов ассистента.",
            style="TopSub.TLabel",
            wraplength=460,
        ).pack(anchor="w", pady=(14, 0))

        buttons = ttk.Frame(shell, style="DialogCard.TFrame")
        buttons.pack(fill=tk.X, pady=(18, 0))

        def rebuild_speech_module(voice: str) -> None:
            self.speech_module.stop_speaking()
            self.settings["supertonic_voice"] = voice
            self.speech_module.config.supertonic_voice = voice
            self.speech_module.supertonic_voice_style = None
            # Кэш фраз привязан к голосу — сбрасываем и заново готовим "Слушаю".
            self.speech_module.clear_phrase_cache()
            if self.wake_reply_enabled and self.wake_reply_text:
                threading.Thread(
                    target=lambda: self.speech_module.prime_phrase(self.wake_reply_text),
                    daemon=True,
                ).start()


        def save_selection() -> None:
            voice = selected_voice.get().strip() or "F1"
            rebuild_speech_module(voice)
            self._save_settings()
            self._refresh_device_labels()
            self.add_message("system", f"Выбран голос озвучивания: {voice}")
            self._set_status("Голос обновлён")
            dialog.destroy()

        ttk.Button(buttons, text="Сохранить", style="Primary.TButton", command=save_selection).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Отмена", style="Secondary.TButton", command=dialog.destroy).pack(
            side=tk.RIGHT,
            padx=(0, 8),
        )

    def open_microphone_dialog(self) -> None:
        devices = self.speech_module.list_input_devices()
        if not devices:
            messagebox.showerror("Ошибка", "В системе нет доступных микрофонов.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Выбор микрофона")
        dialog.geometry("740x260")
        dialog.configure(bg="#dfe7f1")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        shell = ttk.Frame(dialog, style="DialogCard.TFrame", padding=18)
        shell.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        ttk.Label(shell, text="Выбор микрофона", style="TopTitle.TLabel").pack(anchor="w")
        ttk.Label(
            shell,
            text=f"Текущее устройство: {self.speech_module.get_input_device_label()}",
            style="TopSub.TLabel",
            wraplength=660,
        ).pack(anchor="w", pady=(6, 14))

        def device_score(device: dict[str, object]) -> int:
            name = str(device["name"]).lower()
            value = 0

            preferred = ["microphone", "микрофон", "mic input", "array"]
            avoided = ["line in", "line-in", "линейный", "лин. вход", "stereo mix", "стерео микшер", "speaker"]

            for token in preferred:
                if token in name:
                    value += 10
            for token in avoided:
                if token in name:
                    value -= 12
            return value

        devices = sorted(devices, key=device_score, reverse=True)
        values = [
            f"{device['index']} | {device['name']} | {device['default_sample_rate']} Hz"
            for device in devices
        ]
        selected_value = tk.StringVar()
        current_index = self.speech_module.get_input_device_index()
        selected_value.set(next((value for value in values if value.startswith(f"{current_index} |")), values[0]))

        combo = ttk.Combobox(shell, textvariable=selected_value, values=values, state="readonly")
        combo.pack(fill=tk.X)

        ttk.Label(
            shell,
            text="После сохранения приложение будет использовать выбранный индекс для голосового ввода.",
            style="TopSub.TLabel",
            wraplength=660,
        ).pack(anchor="w", pady=(14, 0))

        buttons = ttk.Frame(shell, style="DialogCard.TFrame")
        buttons.pack(fill=tk.X, pady=(18, 0))

        def save_selection() -> None:
            selection = selected_value.get().strip()
            if not selection:
                return

            try:
                device_index = int(selection.split("|", 1)[0].strip())
                self.speech_module.set_input_device(device_index)
                self.settings["input_device_index"] = device_index
                self._save_settings()
                self._refresh_device_labels()
            except Exception as error:
                messagebox.showerror("Ошибка", f"Не удалось переключить микрофон: {error}")
                return

            self.add_message("system", f"Выбран микрофон: {self.speech_module.get_input_device_label()}")
            self._set_status("Микрофон обновлён")
            dialog.destroy()

        ttk.Button(buttons, text="Сохранить", style="Primary.TButton", command=save_selection).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Отмена", style="Secondary.TButton", command=dialog.destroy).pack(
            side=tk.RIGHT,
            padx=(0, 8),
        )

    def add_message(self, role: str, text: str) -> None:
        timestamp = datetime.now().strftime("%H:%M")

        self.chat_area.configure(state="normal")
        if role == "user":
            self.chat_area.insert(tk.END, "Вы  ", "user_header")
            self.chat_area.insert(tk.END, f"{timestamp}\n", "time")
            self.chat_area.insert(tk.END, f"{text}\n\n", "user_body")
        elif role == "assistant":
            self.chat_area.insert(tk.END, "Ассистент  ", "assistant_header")
            self.chat_area.insert(tk.END, f"{timestamp}\n", "time")
            self.chat_area.insert(tk.END, f"{text}\n\n", "assistant_body")
        else:
            self.chat_area.insert(tk.END, "Система  ", "system_header")
            self.chat_area.insert(tk.END, f"{timestamp}\n", "time")
            self.chat_area.insert(tk.END, f"{text}\n\n", "system_body")
        self.chat_area.configure(state="disabled")
        self.chat_area.yview(tk.END)

    def _begin_assistant_message(self) -> None:
        timestamp = datetime.now().strftime("%H:%M")
        self.chat_area.configure(state="normal")
        self.chat_area.insert(tk.END, "Ассистент  ", "assistant_header")
        self.chat_area.insert(tk.END, f"{timestamp}\n", "time")
        self.chat_area.configure(state="disabled")
        self.chat_area.yview(tk.END)

    def _append_assistant_text(self, piece: str) -> None:
        self.chat_area.configure(state="normal")
        self.chat_area.insert(tk.END, piece, "assistant_body")
        self.chat_area.configure(state="disabled")
        self.chat_area.yview(tk.END)

    def _end_assistant_message(self) -> None:
        self._append_assistant_text("\n\n")
        self._set_status("Готово к работе")

    def _trimmed_history(self) -> list[dict[str, str]]:
        """История для LLM без локального system[0] и обрезанная до последних N сообщений."""
        turns = self.conversation_history[1:]
        if self.llm_max_history_messages > 0:
            turns = turns[-self.llm_max_history_messages:]
        return turns

    def _reset_answer_perf(self, anchor: float | None) -> None:
        """Готовит замеры под новый ответ. anchor — момент конца записи (или None)."""
        self._answer_perf_start = anchor
        self._answer_playback_logged = False
        self._answer_llm_first_token_at = None

    def _stream_llm_round(self, user_text: str, cancel_event: threading.Event | None = None) -> None:
        """Стримит ответ LLM в чат и параллельно озвучивает готовые предложения.

        Выполняется в рабочем потоке (не в потоке UI). Если передан cancel_event и
        он выставлен (прерывание пользователем) — генерация и озвучка прекращаются,
        ответ не дописывается и не попадает в историю.
        """
        self.conversation_history.append({"role": "user", "content": user_text})
        self._set_metahuman_state("thinking")

        tts_stream = self.speech_module.create_tts_stream() if self.speech_module.tts_available else None
        assembler = SentenceAssembler()
        chunks: list[str] = []
        error_text: str | None = None
        cancelled = False

        # Замеры (от конца записи) и ограничение истории, чтобы промпт не рос.
        proc_start = self._answer_perf_start if self._answer_perf_start is not None else perf.now()
        first_token = True
        first_sentence = True
        history = self._trimmed_history()

        self.root.after(0, self._begin_assistant_message)
        try:
            for piece in self.llm_module.ask_stream(history):
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break
                if first_token:
                    first_token = False
                    self._answer_llm_first_token_at = perf.now()
                    perf.log_since("LLM response generation (first token)", proc_start)
                chunks.append(piece)
                self.root.after(0, lambda p=piece: self._append_assistant_text(p))
                if tts_stream is not None:
                    for sentence in assembler.feed(piece):
                        if first_sentence:
                            first_sentence = False
                            perf.log_since("TTS first sentence queued", proc_start)
                        tts_stream.add(sentence)
        except Exception as error:
            error_text = f"Ошибка обращения к модели: {error}"

        if not cancelled and error_text is None:
            perf.log_since("LLM response generation (full)", proc_start)

        answer = "".join(chunks).strip()

        if tts_stream is not None:
            if not cancelled and error_text is None and answer:
                tail = assembler.flush()
                if tail:
                    tts_stream.add(tail)
                tts_stream.close()
            else:
                tts_stream.cancel()

        if cancelled:
            # Прерывание: ответ не пишем в историю и не показываем как ошибку.
            self._set_metahuman_state("idle")
            self.root.after(0, self._end_assistant_message)
            return

        if error_text is not None:
            prefix = "\n" if answer else ""
            self.root.after(0, lambda: self._append_assistant_text(f"{prefix}{error_text}"))
        elif not answer:
            self.root.after(0, lambda: self._append_assistant_text("Модель вернула пустой ответ."))
        else:
            self.conversation_history.append({"role": "assistant", "content": answer})

        # Если озвучки не будет, возвращаем аватара в idle сами;
        # иначе это сделает on_speak_end после конца воспроизведения.
        if tts_stream is None or error_text is not None or not answer:
            self._set_metahuman_state("idle")

        self.root.after(0, self._end_assistant_message)

    def handle_send(self) -> None:
        user_text = self._get_input_text()
        if not user_text:
            return

        self._clear_input()
        self.add_message("user", user_text)
        self._set_status("Ассистент отвечает...")

        # Текстовый ввод: записи нет — замеры "после записи" не считаем.
        self._reset_answer_perf(None)
        threading.Thread(target=self._stream_llm_round, args=(user_text,), daemon=True).start()

    def handle_voice_input(self) -> None:
        # Barge-in: голосовой ввод должен сразу прерывать текущую озвучку.
        self.speech_module.stop_speaking()
        self.metahuman_bridge.speak_end()
        self._set_status("Слушаю ваш голос...")
        self._set_metahuman_state("listening")

        def worker() -> None:
            # Пока идёт ручная запись, wake-слушатель не должен трогать микрофон.
            self._manual_capture_active = True
            try:
                text = self.speech_module.speech_to_text()
                # Замеры обработки считаем от момента окончания записи.
                self._reset_answer_perf(self.speech_module.last_record_finished_monotonic)

                def process_voice() -> None:
                    self._clear_input()
                    self.add_message("user", text)
                    self._set_status("Ассистент отвечает...")
                    threading.Thread(target=self._stream_llm_round, args=(text,), daemon=True).start()

                self.root.after(0, process_voice)

            except Exception as error:
                error_message = str(error)
                self._set_metahuman_state("idle")
                self.root.after(0, lambda: self._set_status("Готово к работе"))
                self.root.after(0, lambda: messagebox.showerror("Ошибка", error_message))
            finally:
                self._manual_capture_active = False

        threading.Thread(target=worker, daemon=True).start()

    def clear_chat(self) -> None:
        self.speech_module.stop_speaking()
        self.metahuman_bridge.speak_end()
        self.conversation_history = [{"role": "system", "content": LM_CONFIG.system_prompt}]
        self.chat_area.configure(state="normal")
        self.chat_area.delete("1.0", tk.END)
        self.chat_area.configure(state="disabled")
        self._set_status("Диалог очищен")

    def on_close(self) -> None:
        self.wake_listener.stop()
        if self.interrupt_listener is not None:
            self.interrupt_listener.stop()
        self.speech_module.shutdown()
        self.metahuman_bridge.shutdown()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = AssistantApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()


