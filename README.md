# ИИ-ассистент для LM Studio

Настольный русскоязычный ИИ-ассистент на `tkinter`. Приложение подключается к локальному серверу LM Studio через OpenAI-compatible API, принимает текстовые и голосовые запросы, распознаёт речь локально через Whisper и озвучивает ответы локальной моделью Supertonic 3.

## Возможности

- чат с локальной LLM из LM Studio;
- текстовый ввод и отправка по `Enter`;
- голосовой ввод с микрофона;
- локальное распознавание русской речи через faster-whisper (whisper-large-v3-turbo);
- локальная озвучка ответов через Supertonic 3;
- выбор микрофона из интерфейса;
- выбор голоса Supertonic из доступных `voice_styles`;
- сохранение пользовательских настроек в `assistant_settings.json`;
- диагностические скрипты для проверки микрофона и распознавания.

## Состав проекта

```text
app.py                    # tkinter-интерфейс ассистента
lm_studio_module.py       # HTTP-клиент для LM Studio /v1
speech_module.py          # Whisper STT, Supertonic TTS, работа с PyAudio
requirements.txt          # Python-зависимости
assistant_settings.json   # локальные настройки микрофона и голоса
.env.example              # пример локальных переменных окружения
check_mic.py              # список входных аудиоустройств
test_mic.py               # диагностика микрофона и локального STT
whisper-large-v3-turbo-ct2/  # локальная STT-модель (CTranslate2) для faster-whisper
supertonic-3-model/       # локальная модель и стили голосов Supertonic 3
```

## Требования

- Windows;
- Python 3.12 или совместимая версия Python 3;
- рабочий микрофон;
- LM Studio с включённым локальным API-сервером;
- локальная модель Whisper или доступ к уже скачанному кешу faster-whisper;
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

### Whisper

По умолчанию STT использует `faster-whisper` и локальную модель `whisper-large-v3-turbo-ct2` (CTranslate2-сборка whisper-large-v3-turbo, int8) на CPU. Если локальная папка с моделью отсутствует, используется имя `large-v3-turbo` — faster-whisper скачает модель в кеш при первом запуске.

```env
AI_ASSISTANT_STT_BACKEND=faster_whisper
AI_ASSISTANT_WHISPER_MODEL=whisper-large-v3-turbo-ct2
AI_ASSISTANT_WHISPER_DEVICE=cpu
AI_ASSISTANT_WHISPER_COMPUTE_TYPE=int8
AI_ASSISTANT_WHISPER_LANGUAGE=ru
AI_ASSISTANT_WHISPER_BEAM_SIZE=1
AI_ASSISTANT_WHISPER_VAD_FILTER=true
AI_ASSISTANT_WHISPER_INITIAL_PROMPT=
```

Если есть NVIDIA GPU, обычно быстрее использовать:

```env
AI_ASSISTANT_WHISPER_DEVICE=cuda
AI_ASSISTANT_WHISPER_COMPUTE_TYPE=float16
```

Если Whisper искажает имена, команды или термины, добавьте их через `AI_ASSISTANT_WHISPER_INITIAL_PROMPT`.

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
AI_ASSISTANT_STT_BACKEND=faster_whisper
AI_ASSISTANT_WHISPER_MODEL=small
AI_ASSISTANT_WHISPER_DEVICE=cpu
AI_ASSISTANT_WHISPER_COMPUTE_TYPE=int8
AI_ASSISTANT_WHISPER_LANGUAGE=ru
AI_ASSISTANT_WHISPER_BEAM_SIZE=1
AI_ASSISTANT_WHISPER_VAD_FILTER=true
AI_ASSISTANT_WHISPER_INITIAL_PROMPT=
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

На Windows также можно запустить готовый launcher:

```powershell
.\run_app.bat
```

Launcher использует проектное Windows-окружение `.venv-win`, а не случайный `python` из `PATH`.

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

Проверить микрофон и локальное распознавание:

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

**Whisper не установлен или модель не загрузилась**

Проверьте, что установлен `faster-whisper`, а `AI_ASSISTANT_WHISPER_MODEL` указывает на существующую локальную модель или допустимое имя модели.

**Микрофон не распознаёт речь**

Запустите `check_mic.py`, выберите корректный индекс микрофона в интерфейсе или задайте `AI_ASSISTANT_INPUT_DEVICE_INDEX`.

**Озвучка не работает**

Проверьте, что установлен пакет `supertonic`, а каталог `supertonic-3-model` содержит ONNX-файлы и `voice_styles`.

## Примечания для разработки

- `LMStudioModule` работает напрямую через `urllib` и не требует отдельного SDK OpenAI.
- История диалога хранится в памяти процесса и сбрасывается кнопкой `Очистить`.
- Supertonic запускается с `auto_download=False`, поэтому модель должна быть доступна локально.
- Локальные модели, виртуальные окружения и кеши могут занимать много места и обычно не должны попадать в коммиты.
