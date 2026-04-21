from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

from lm_studio_module import LMStudioConfig, LMStudioModule
from speech_module import SpeechConfig, SpeechModule


LM_CONFIG = LMStudioConfig(
    base_url="http://127.0.0.1:1234",
    api_key="lm-studio",
    model="google/gemma-4-e2b",
    temperature=0.7,
    max_tokens=300,
    system_prompt="Ты полезный ИИ-ассистент. Отвечай кратко, понятно и на русском языке.",
)

SETTINGS_PATH = Path(__file__).resolve().parent / "assistant_settings.json"


class AssistantApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ИИ-ассистент")
        self.root.geometry("800x600")
        self.root.minsize(700, 500)

        self.llm_module = LMStudioModule(LM_CONFIG)
        self.settings = self._load_settings()
        self.speech_module = SpeechModule(
            SpeechConfig(input_device_index=self.settings.get("input_device_index"))
        )

        self.conversation_history = [
            {
                "role": "system",
                "content": LM_CONFIG.system_prompt,
            }
        ]

        self.status_var = tk.StringVar(value="Готово")
        self._build_ui()

        self.add_message("assistant", "Здравствуйте! Напишите сообщение или используйте голосовой ввод.")
        self.add_message("system", f"Активный микрофон: {self.speech_module.get_input_device_label()}")
        if not self.speech_module.tts_available:
            self.add_message(
                "system",
                "Озвучивание недоступно в этой среде, но чат и распознавание речи работают.",
            )

    def _build_ui(self) -> None:
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = ttk.Label(main_frame, text="ИИ-ассистент", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 10))

        self.chat_area = scrolledtext.ScrolledText(
            main_frame,
            wrap=tk.WORD,
            state="disabled",
            font=("Arial", 11),
        )
        self.chat_area.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X)

        self.input_entry = ttk.Entry(input_frame, font=("Arial", 11))
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.input_entry.bind("<Return>", lambda event: self.handle_send())

        self.send_button = ttk.Button(input_frame, text="Отправить", command=self.handle_send)
        self.send_button.pack(side=tk.LEFT, padx=(0, 5))

        self.voice_button = ttk.Button(input_frame, text="Голос", command=self.handle_voice_input)
        self.voice_button.pack(side=tk.LEFT, padx=(0, 5))

        self.mic_button = ttk.Button(input_frame, text="Микрофон", command=self.open_microphone_dialog)
        self.mic_button.pack(side=tk.LEFT, padx=(0, 5))

        self.clear_button = ttk.Button(input_frame, text="Очистить", command=self.clear_chat)
        self.clear_button.pack(side=tk.LEFT)

        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w")
        status_bar.pack(fill=tk.X, pady=(10, 0))

    def _load_settings(self) -> dict[str, int | None]:
        if not SETTINGS_PATH.exists():
            return {"input_device_index": None}

        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"input_device_index": None}

        input_device_index = data.get("input_device_index")
        if isinstance(input_device_index, int):
            return {"input_device_index": input_device_index}
        return {"input_device_index": None}

    def _save_settings(self) -> None:
        SETTINGS_PATH.write_text(
            json.dumps(self.settings, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def open_microphone_dialog(self) -> None:
        devices = self.speech_module.list_input_devices()
        if not devices:
            messagebox.showerror("Ошибка", "В системе нет доступных микрофонов.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Выбор микрофона")
        dialog.geometry("660x230")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(
            dialog,
            text=f"Текущий микрофон: {self.speech_module.get_input_device_label()}",
            wraplength=620,
        ).pack(fill=tk.X, padx=15, pady=(15, 10))

        ttk.Label(
            dialog,
            text="Выберите индекс микрофона из списка:",
        ).pack(anchor="w", padx=15)

        values = [
            f"{device['index']} | {device['name']} | {device['default_sample_rate']} Hz"
            for device in devices
        ]
        selected_value = tk.StringVar()
        current_index = self.speech_module.get_input_device_index()

        for value in values:
            if value.startswith(f"{current_index} |"):
                selected_value.set(value)
                break
        if not selected_value.get():
            selected_value.set(values[0])

        combo = ttk.Combobox(dialog, textvariable=selected_value, values=values, state="readonly")
        combo.pack(fill=tk.X, padx=15, pady=(8, 15))

        ttk.Label(
            dialog,
            text="После сохранения приложение будет использовать этот индекс для голосового ввода.",
            wraplength=620,
        ).pack(fill=tk.X, padx=15)

        buttons = ttk.Frame(dialog)
        buttons.pack(fill=tk.X, padx=15, pady=15)

        def save_selection() -> None:
            selection = selected_value.get().strip()
            if not selection:
                return

            try:
                device_index = int(selection.split("|", 1)[0].strip())
                self.speech_module.set_input_device(device_index)
                self.settings["input_device_index"] = device_index
                self._save_settings()
            except Exception as error:  # noqa: BLE001
                messagebox.showerror("Ошибка", f"Не удалось переключить микрофон: {error}")
                return

            self.add_message("system", f"Выбран микрофон: {self.speech_module.get_input_device_label()}")
            self.status_var.set("Микрофон обновлён")
            dialog.destroy()

        ttk.Button(buttons, text="Сохранить", command=save_selection).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT, padx=(0, 8))

    def add_message(self, role: str, text: str) -> None:
        self.chat_area.configure(state="normal")
        if role == "user":
            self.chat_area.insert(tk.END, f"Вы: {text}\n\n")
        elif role == "assistant":
            self.chat_area.insert(tk.END, f"Ассистент: {text}\n\n")
        else:
            self.chat_area.insert(tk.END, f"Система: {text}\n\n")
        self.chat_area.configure(state="disabled")
        self.chat_area.yview(tk.END)

    def ask_llm(self, user_text: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_text})
        try:
            answer = self.llm_module.ask(self.conversation_history[1:])
            self.conversation_history.append({"role": "assistant", "content": answer})
            return answer
        except Exception as error:  # noqa: BLE001
            return f"Ошибка обращения к модели: {error}"

    def handle_send(self) -> None:
        user_text = self.input_entry.get().strip()
        if not user_text:
            return

        self.input_entry.delete(0, tk.END)
        self.add_message("user", user_text)
        self.status_var.set("Обрабатываю запрос...")

        def worker() -> None:
            answer = self.ask_llm(user_text)

            def update_ui() -> None:
                self.add_message("assistant", answer)
                self.status_var.set("Готово")

            self.root.after(0, update_ui)

            if not answer.startswith("Ошибка"):
                self.speech_module.text_to_speech_async(answer)

        threading.Thread(target=worker, daemon=True).start()

    def handle_voice_input(self) -> None:
        self.status_var.set("Слушаю...")

        def worker() -> None:
            try:
                text = self.speech_module.speech_to_text()

                def process_voice() -> None:
                    self.input_entry.delete(0, tk.END)
                    self.input_entry.insert(0, text)
                    self.add_message("user", text)
                    self.status_var.set("Обрабатываю голосовой запрос...")

                    def ask_worker() -> None:
                        answer = self.ask_llm(text)

                        def update_answer() -> None:
                            self.add_message("assistant", answer)
                            self.status_var.set("Готово")

                        self.root.after(0, update_answer)

                        if not answer.startswith("Ошибка"):
                            self.speech_module.text_to_speech_async(answer)

                    threading.Thread(target=ask_worker, daemon=True).start()

                self.root.after(0, process_voice)

            except Exception as error:  # noqa: BLE001
                self.root.after(0, lambda: self.status_var.set("Готово"))
                self.root.after(0, lambda: messagebox.showerror("Ошибка", str(error)))

        threading.Thread(target=worker, daemon=True).start()

    def clear_chat(self) -> None:
        self.conversation_history = [
            {
                "role": "system",
                "content": LM_CONFIG.system_prompt,
            }
        ]
        self.chat_area.configure(state="normal")
        self.chat_area.delete(1.0, tk.END)
        self.chat_area.configure(state="disabled")
        self.status_var.set("Диалог очищен")

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
