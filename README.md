# ИИ-ассистент для LM Studio

Настольный русскоязычный ИИ-ассистент на `tkinter`. Приложение подключается к локальному серверу LM Studio через OpenAI-compatible API, принимает текстовые и голосовые запросы, распознаёт речь локально через Whisper и озвучивает ответы локальной моделью Supertonic 3.

## Возможности

- чат с локальной LLM из LM Studio;
- текстовый ввод и отправка по `Enter`;
- голосовой ввод с микрофона;
- локальное распознавание русской речи через faster-whisper (whisper-large-v3-turbo);
- локальная озвучка ответов через Supertonic 3;
- выбор микрофона из интерфейса;
- выбор устройства вывода озвучки, включая `CABLE Input` для MetaHuman lip sync;
- MetaHuman-мост: OSC-состояния диалога для Unreal Engine;
- выбор голоса Supertonic из доступных `voice_styles`;
- сохранение пользовательских настроек в `assistant_settings.json`;
- диагностические скрипты для проверки микрофона и распознавания.

## Состав проекта

```text
app.py                    # tkinter-интерфейс ассистента
lm_studio_module.py       # HTTP-клиент для LM Studio /v1
speech_module.py          # Whisper STT, Supertonic TTS, работа с PyAudio
metahuman_bridge.py       # OSC-мост для MetaHuman / Unreal Engine
requirements.txt          # Python-зависимости
assistant_settings.json   # локальные настройки микрофона, вывода, голоса и MetaHuman
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

Требуется **Python 3.12** (не 3.13 — в нём удалён модуль `audioop`, который нужен для обработки звука).

Если `PyAudio` не устанавливается через `pip`, установите подходящее колесо для вашей версии Python или используйте окружение, где PortAudio уже доступен.

## Перенос на другой компьютер

В репозитории есть инструменты для быстрого переноса и развёртывания.

### Экспорт (на исходной машине)

```powershell
powershell -ExecutionPolicy Bypass -File export_assistant.ps1
```

Соберёт `export/ai-assistant-export-<дата>.zip` с исходниками, моделями
(Whisper + Supertonic), скриптами установки и шаблоном настроек. Исключает
виртуальные окружения, кеши, git, отладочные логи и личный
`assistant_settings.json`. Флаг `-NoModels` соберёт архив без папок моделей
(~1.2 ГБ) — тогда модели нужно скопировать отдельно.

### Развёртывание (на новой машине)

1. Распакуйте архив (или склонируйте репозиторий).
2. Установите **Python 3.12** (https://www.python.org/downloads/).
3. Запустите `setup.bat`. Он:
   - создаёт виртуальное окружение и ставит зависимости;
   - **скачивает модели Whisper и Supertonic** автоматически (`download_models.py`,
     ~1.2 ГБ, нужен интернет) — копировать их вручную больше не требуется;
   - создаёт `assistant_settings.json` из шаблона;
   - проверяет готовность окружения (`check_setup.py`).
4. Установите и запустите **LM Studio**, загрузите модель, включите локальный
   сервер. Имя модели должно совпадать с `model` в `LM_CONFIG` внутри `app.py`.
5. Запустите `run_app.bat`.

Скачать модели отдельно (если нужно перекачать или сделать это вручную):

```powershell
.venv-win\Scripts\python.exe download_models.py          # скачать недостающее
.venv-win\Scripts\python.exe download_models.py --force  # перекачать заново
```

Проверить окружение в любой момент:

```powershell
.venv-win\Scripts\python.exe check_setup.py
```

Модели (`whisper-large-v3-turbo-ct2/` ~780 МБ, `supertonic-3-model/` ~383 МБ) не
хранятся в git. Есть два пути их получить:
- **онлайн** — `setup.bat`/`download_models.py` скачают их с HuggingFace;
- **офлайн** — соберите полный архив `export_assistant.ps1` (без флага
  `-NoModels`), он включит модели внутрь.

> Индексы аудиоустройств в `assistant_settings.json` зависят от конкретного ПК.
> На новой машине шаблон `assistant_settings.example.json` ставит их в `null`,
> и микрофон выбирается автоматически. Если переносили свой
> `assistant_settings.json` — удалите его или сбросьте `input_device_index` и
> `output_device_index` в `null`.

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

## Ускорение на GPU (NVIDIA)

Распознавание (Whisper) и синтез (Supertonic) можно ускорить на видеокарте NVIDIA.
Настройки задаются в `assistant_settings.json`.

### 1. Whisper на GPU (самый большой выигрыш)

На CPU STT занимает секунды, на GPU — около **0.1 с**.

1. Установите CUDA-библиотеки (cuBLAS + cuDNN) поверх основных зависимостей:
   ```powershell
   .venv-win\Scripts\python.exe -m pip install -r requirements-gpu.txt
   ```
2. В `assistant_settings.json` укажите:
   ```json
   "whisper_device": "cuda",
   "whisper_compute_type": "float16"
   ```
   (для экономии памяти можно `"int8_float16"`).

Приложение само добавляет DLL из `nvidia-*-cu12` в `PATH` (см.
`speech_module._register_cuda_dll_dirs`) — вручную PATH править не нужно. Если CUDA
или библиотеки недоступны, будет ошибка загрузки — верните `"whisper_device": "cpu"`.
Проверить готовность: `python check_setup.py`.

### 2. Supertonic: меньше шагов + GPU

- **Шаги диффузии** — главный рычаг скорости синтеза. В `assistant_settings.json`:
  ```json
  "supertonic_steps": 6
  ```
  Допустимо 5–12; 5–6 заметно быстрее 8 при незначительной потере качества.
- **GPU для Supertonic (опционально)** требует GPU-сборку onnxruntime вместо
  обычной. Раскомментируйте строки в `requirements-gpu.txt`, установите его и задайте:
  ```json
  "supertonic_gpu": true
  ```
  Без `onnxruntime-gpu` Supertonic тихо остаётся на CPU (приложение это сообщит).

Требования: видеодрайвер NVIDIA и CUDA 12. Версии должны соответствовать вашей
карте (для новых GPU нужны свежие сборки `onnxruntime-gpu`).

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

Выбранные в интерфейсе микрофон, устройство вывода, голос и состояние MetaHuman-моста сохраняются в `assistant_settings.json` и применяются при следующем запуске.

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
- Нажмите `Устройство вывода озвучки`, чтобы направить TTS в системный выход или в `CABLE Input`.
- Нажмите `MetaHuman вкл/выкл`, чтобы включить отправку OSC-событий в Unreal Engine.
- Нажмите `Выбрать голос`, чтобы сменить голос Supertonic.
- Нажмите `Очистить`, чтобы сбросить текущую историю диалога.

Ответ модели отображается в диалоге. Если Supertonic 3 доступен, ответ также озвучивается.

## MetaHuman / Unreal Engine

Для lip sync в UE используется двухканальная схема:

- аудио: Supertonic TTS -> `CABLE Input (VB-Audio Virtual Cable)` -> `CABLE Output` -> MetaHuman Audio Source;
- управление: `metahuman_bridge.py` -> OSC UDP `127.0.0.1:9000` -> Blueprint OSC Server.

Настройка:

1. Убедитесь, что VB-Cable установлен и Windows видит `CABLE Input` / `CABLE Output`.
2. В приложении нажмите `Устройство вывода озвучки` и выберите `CABLE Input`.
3. Нажмите `MetaHuman вкл/выкл`; в карточке MetaHuman должен появиться адрес `OSC 127.0.0.1:9000`.
4. В UE включите плагины `OSC` и `MetaHuman Live Link`.
5. В Blueprint создайте MetaHuman Audio Source от устройства `CABLE Output`.
6. В Blueprint создайте OSC Server на `127.0.0.1:9000` и принимайте адреса:

```text
/state        string: idle | listening | thinking | speaking
/speak_start
/speak_end
/emotion      string, float
/gaze         float, float
/gesture      string
/curve        string, float
/ping         int
```

Быстрый тест OSC без запуска ассистента:

```powershell
.\.venv-win\Scripts\python.exe metahuman_bridge.py
```

В UE Output Log должны появиться тестовые сообщения `/state`, `/speak_start`, `/emotion`, `/gaze`, `/speak_end`.

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
