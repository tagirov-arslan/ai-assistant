from __future__ import annotations

import json
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

from lm_studio_module import LMStudioConfig, LMStudioModule
from speech_module import SpeechConfig, SpeechModule


LM_CONFIG = LMStudioConfig(
    base_url="http://127.0.0.1:1234",
    api_key="lm-studio",
    model="qwen/qwen3.6-35b-a3b",
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
        self.speech_module = SpeechModule(self._build_speech_config())
        active_input_index = self.speech_module.get_input_device_index()
        if self.settings.get("input_device_index") != active_input_index:
            self.settings["input_device_index"] = active_input_index
            self._save_settings()

        self.conversation_history = [{"role": "system", "content": LM_CONFIG.system_prompt}]
        self.status_var = tk.StringVar(value="Готово к работе")
        self.mic_var = tk.StringVar(value=self.speech_module.get_input_device_label())
        self.voice_var = tk.StringVar(value=self.speech_module.get_tts_status_label())

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
            "supertonic_voice": "F1",
            "supertonic_lang": "ru",
            "supertonic_steps": 8,
            "supertonic_speed": 1.0,
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
        return settings

    def _save_settings(self) -> None:
        SETTINGS_PATH.write_text(json.dumps(self.settings, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_speech_config(self) -> SpeechConfig:
        return SpeechConfig(
            input_device_index=(
                self.settings.get("input_device_index")
                if isinstance(self.settings.get("input_device_index"), int)
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

        ttk.Button(sidebar, text="Выбрать голос", style="Ghost.TButton", command=self.open_voice_dialog).pack(
            fill=tk.X,
            pady=(6, 0),
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

    def ask_llm(self, user_text: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_text})
        try:
            answer = self.llm_module.ask(self.conversation_history[1:])
            self.conversation_history.append({"role": "assistant", "content": answer})
            return answer
        except Exception as error:
            return f"Ошибка обращения к модели: {error}"

    def handle_send(self) -> None:
        user_text = self._get_input_text()
        if not user_text:
            return

        self._clear_input()
        self.add_message("user", user_text)
        self._set_status("Ассистент обрабатывает запрос...")

        def worker() -> None:
            answer = self.ask_llm(user_text)

            def update_ui() -> None:
                self.add_message("assistant", answer)
                self._set_status("Готово к работе")

            self.root.after(0, update_ui)

            if not answer.startswith("Ошибка") and self.speech_module.tts_available:
                self.speech_module.text_to_speech_async(answer)

        threading.Thread(target=worker, daemon=True).start()

    def handle_voice_input(self) -> None:
        self._set_status("Слушаю ваш голос...")

        def worker() -> None:
            try:
                text = self.speech_module.speech_to_text()

                def process_voice() -> None:
                    self._clear_input()
                    self.add_message("user", text)
                    self._set_status("Обрабатываю голосовой запрос...")

                    def ask_worker() -> None:
                        answer = self.ask_llm(text)

                        def update_answer() -> None:
                            self.add_message("assistant", answer)
                            self._set_status("Готово к работе")

                        self.root.after(0, update_answer)

                        if not answer.startswith("Ошибка") and self.speech_module.tts_available:
                            self.speech_module.text_to_speech_async(answer)

                    threading.Thread(target=ask_worker, daemon=True).start()

                self.root.after(0, process_voice)

            except Exception as error:
                error_message = str(error)
                self.root.after(0, lambda: self._set_status("Готово к работе"))
                self.root.after(0, lambda: messagebox.showerror("Ошибка", error_message))

        threading.Thread(target=worker, daemon=True).start()

    def clear_chat(self) -> None:
        self.conversation_history = [{"role": "system", "content": LM_CONFIG.system_prompt}]
        self.chat_area.configure(state="normal")
        self.chat_area.delete("1.0", tk.END)
        self.chat_area.configure(state="disabled")
        self._set_status("Диалог очищен")

    def on_close(self) -> None:
        self.speech_module.shutdown()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = AssistantApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()






