from __future__ import annotations

import argparse
import time

from speech_module import SpeechConfig, SpeechModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-index", type=int, default=None, help="Принудительно выбрать индекс микрофона")
    parser.add_argument("--list-devices", action="store_true", help="Показать доступные входные устройства и выйти")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module = SpeechModule(SpeechConfig(input_device_index=args.device_index))

    try:
        if args.list_devices:
            print("Доступные входные устройства:")
            for device in module.list_input_devices():
                print(device)
            return

        print(f"Активный микрофон: {module.get_input_device_label()}")
        print(f"Распознавание: {module.get_stt_status_label()}")
        print("Диагностика запущена. После паузы распознавание завершится автоматически.")
        print("")

        started_at = time.monotonic()
        text = module.speech_to_text()
        elapsed = time.monotonic() - started_at

        print(f"Итоговый текст: {text!r}")
        print(f"Время записи и распознавания: {elapsed:.2f} c")
        print(f"Порог начала речи в модуле: {module.config.energy_threshold}")
        print("Если текст пустой или искажён, проверьте микрофон и модель в AI_ASSISTANT_WHISPER_MODEL.")
    finally:
        module.shutdown()


if __name__ == "__main__":
    main()
