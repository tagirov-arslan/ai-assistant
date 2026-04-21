from speech_module import SpeechModule


def main() -> None:
    module = SpeechModule()

    print("Доступные микрофоны:\n")
    for device in module.list_input_devices():
        print(f"Индекс: {device['index']}")
        print(f"Название: {device['name']}")
        print(f"Частота по умолчанию: {device['default_sample_rate']} Hz")
        print("-" * 30)

    print(f"\nАктивный микрофон: {module.get_input_device_label()}")
    module.shutdown()


if __name__ == "__main__":
    main()
