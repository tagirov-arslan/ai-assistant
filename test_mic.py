from __future__ import annotations

import argparse
import time

from vosk import KaldiRecognizer

from speech_module import SpeechConfig, SpeechModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-index", type=int, default=None, help="Принудительно выбрать индекс микрофона")
    parser.add_argument("--list-devices", action="store_true", help="Показать доступные входные устройства и выйти")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module = SpeechModule(SpeechConfig(input_device_index=args.device_index))

    if args.list_devices:
        print("Доступные входные устройства:")
        for device in module.list_input_devices():
            print(device)
        module.shutdown()
        return

    recognizer = KaldiRecognizer(module.vosk_model, float(module.active_sample_rate))
    recognizer.SetWords(True)

    print(f"Активный микрофон: {module.get_input_device_label()}")
    print("Диагностика запущена. Говорите 8-10 секунд обычным голосом.")
    print("Будут показаны уровни сигнала и промежуточное распознавание Vosk.")
    print("")

    started_at = time.monotonic()
    last_partial = ""

    try:
        with module._open_input_stream(  # noqa: SLF001
            sample_rate=module.active_sample_rate,
            frames_per_buffer=module.config.chunk_size,
        ) as stream:
            while time.monotonic() - started_at < 10:
                data = stream.read(module.config.chunk_size, exception_on_overflow=False)
                energy = module._estimate_energy(data)  # noqa: SLF001

                if recognizer.AcceptWaveform(data):
                    text = module._extract_text(recognizer.Result())  # noqa: SLF001
                    print(f"[final] energy={energy:>4} text={text!r}")
                else:
                    partial = module._extract_partial(recognizer.PartialResult())  # noqa: SLF001
                    if partial and partial != last_partial:
                        last_partial = partial
                        print(f"[partial] energy={energy:>4} text={partial!r}")
                    else:
                        print(f"[level] energy={energy:>4}")

        final_text = module._extract_text(recognizer.FinalResult())  # noqa: SLF001
        print("")
        print(f"Итоговый текст: {final_text!r}")
        print(f"Порог начала речи в модуле: {module.config.energy_threshold}")
        print("Если energy почти всегда маленький, проблема в сигнале микрофона.")
        print("Если energy высокий, но text пустой, проблема уже в распознавании Vosk.")
    finally:
        module.shutdown()


if __name__ == "__main__":
    main()
