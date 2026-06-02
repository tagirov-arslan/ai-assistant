# ИИ-ассистент для LM Studio

Настольный русскоязычный ИИ-ассистент на `tkinter`. Приложение подключается к локальному серверу LM Studio через OpenAI-compatible API, принимает текстовые и голосовые запросы, распознаёт речь офлайн через Vosk и озвучивает ответы локальной моделью Supertonic 3.

## Возможности

- чат с локальной LLM из LM Studio;
- текстовый ввод и отправка по `Enter`;
- голосовой ввод с микрофона;
- офлайн-распознавание русской речи через Vosk;
- локальная озвучка ответов через Supertonic 3;
- выбор микрофона из интерфейса;
- выбор голоса Supertonic из доступных `voice_styles`;
- сохранение пользовательских настроек в `assistant_settings.json`;
- диагностические скрипты для проверки микрофона и распознавания.

## Состав проекта

```text
app.py                    # tkinter-интерфейс ассистента
lm_studio_module.py       # HTTP-клиент для LM Studio /v1
speech_module.py          # Vosk STT, Supertonic TTS, работа с PyAudio
requirements.txt          # Python-зависимости
assistant_settings.json   # локальные настройки микрофона и голоса
.env.example              # пример локальных переменных окружения
check_mic.py              # список входных аудиоустройств
test_mic.py               # диагностика уровня сигнала и Vosk
models/                   # локальные модели Vosk
supertonic-3-model/       # локальная модель и стили голосов Supertonic 3
```

## Требования

- Windows;
- Python 3.12 или совместимая версия Python 3;
- рабочий микрофон;
- LM Studio с включённым локальным API-сервером;
- локальная модель Vosk, например `models/vosk-model-small-ru-0.22`;
- локальная модель Supertonic 3 в `supertonic-3-model`, если нужна озвучка.

## Установка

Создайте виртуальное окружение и установите зависимости:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Если `PyAudio` не устанавливается через `pip`, установите подходящее колесо для вашей версии Python или используйте окружение, где PortAudio уже доступен.

## Локальные модели

### Vosk

По умолчанию приложение ищет модель распознавания в таком порядке:

1. путь из переменной `AI_ASSISTANT_VOSK_MODEL_PATH`;
2. `models/vosk-model-small-ru-0.22`;
3. `resources/vosk/vosk-model-small-ru-0.22`;
4. старый совместимый путь из предыдущей сборки проекта.

Рекомендуемый вариант для этого репозитория:

```text
models/vosk-model-small-ru-0.22
```

### Supertonic 3

Озвучка использует пакет `supertonic` и локальную модель:

```text
supertonic-3-model
```

Если пакет или модель недоступны, чат продолжит работать без озвучивания.

## Настройки окружения

`speech_module.py` автоматически читает файл `.env` из корня проекта, если он существует. Скопируйте `.env.example` в `.env` и при необходимости измените значения:

```env
AI_ASSISTANT_INPUT_DEVICE_INDEX=1
AI_ASSISTANT_VOSK_MODEL_PATH=models/vosk-model-small-ru-0.22
SUPERTONIC_MODEL_DIR=supertonic-3-model
SUPERTONIC_VOICE=F1
SUPERTONIC_LANG=ru
SUPERTONIC_STEPS=8
SUPERTONIC_SPEED=1.0
```

Выбранные в интерфейсе микрофон и голос сохраняются в `assistant_settings.json` и применяются при следующем запуске.

`models/`, `supertonic-3-model/`, `.env` и `assistant_settings.json` считаются локальными файлами конкретного ПК и не должны попадать в Git.

## Запуск

1. Откройте LM Studio.
2. Загрузите модель.
3. Включите локальный сервер LM Studio.
4. Убедитесь, что сервер доступен по адресу `http://127.0.0.1:1234`.
5. Запустите приложение:

```powershell
.\.venv\Scripts\python.exe app.py
```

Текущая конфигурация по умолчанию в коде:

- адрес LM Studio: `http://127.0.0.1:1234`;
- API key: `lm-studio`;
- модель: `qwen/qwen3.6-35b-a3b`;
- язык ассистента: русский.

Если в LM Studio загружена другая модель, обновите `model` в `LM_CONFIG` внутри `app.py`.

## Использование

- Введите сообщение в поле внизу и нажмите `Отправить` или `Enter`.
- Нажмите `Голос`, чтобы записать короткую фразу с микрофона и отправить её модели.
- Нажмите `Микрофон`, чтобы выбрать входное устройство.
- Нажмите `Выбрать голос`, чтобы сменить голос Supertonic.
- Нажмите `Очистить`, чтобы сбросить текущую историю диалога.

Ответ модели отображается в диалоге. Если Supertonic 3 доступен, ответ также озвучивается.

## Диагностика

Показать доступные микрофоны:

```powershell
.\.venv\Scripts\python.exe check_mic.py
```

Проверить уровень сигнала и распознавание Vosk:

```powershell
.\.venv\Scripts\python.exe test_mic.py
```

Выбрать конкретный микрофон для диагностики:

```powershell
.\.venv\Scripts\python.exe test_mic.py --device-index 1
```

Показать устройства через диагностический скрипт:

```powershell
.\.venv\Scripts\python.exe test_mic.py --list-devices
```

## Частые проблемы

**LM Studio не отвечает**

Проверьте, что локальный сервер включён и endpoint `http://127.0.0.1:1234/v1/models` открывается.

**Vosk-модель не найдена**

Положите модель в `models/vosk-model-small-ru-0.22` или задайте `AI_ASSISTANT_VOSK_MODEL_PATH` в `.env`.

**Микрофон не распознаёт речь**

Запустите `check_mic.py`, выберите корректный индекс микрофона в интерфейсе или задайте `AI_ASSISTANT_INPUT_DEVICE_INDEX`.

**Озвучка не работает**

Проверьте, что установлен пакет `supertonic`, а каталог `supertonic-3-model` содержит ONNX-файлы и `voice_styles`.

## Примечания для разработки

- `LMStudioModule` работает напрямую через `urllib` и не требует отдельного SDK OpenAI.
- История диалога хранится в памяти процесса и сбрасывается кнопкой `Очистить`.
- Supertonic запускается с `auto_download=False`, поэтому модель должна быть доступна локально.
- Локальные модели, виртуальные окружения и кеши могут занимать много места и обычно не должны попадать в коммиты.
