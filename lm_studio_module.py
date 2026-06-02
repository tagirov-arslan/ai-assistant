from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class LMStudioConfig:
    base_url: str = "http://127.0.0.1:1234"
    api_key: str = "lm-studio"
    model: str = "qwen/qwen3.6-35b-a3b"
    temperature: float = 0.7
    max_tokens: int = 2000
    system_prompt: str = (
        "Ты полезный ИИ-ассистент. Отвечай кратко, понятно и по делу. "
        "Если пользователь пишет на русском, отвечай на русском языке."
    )

    @property
    def api_base_url(self) -> str:
        base = self.base_url.strip().rstrip("/")
        return base if base.endswith("/v1") else f"{base}/v1"


class LMStudioModule:
    """Модуль подключения ИИ-ассистента к LM Studio через OpenAI-compatible HTTP API."""

    def __init__(self, config: LMStudioConfig | None = None) -> None:
        self.config = config or LMStudioConfig()

    def update_config(self, config: LMStudioConfig) -> None:
        self.config = config

    def get_models(self) -> list[str]:
        data = self._request_json("GET", "models")
        models = data.get("data", [])
        return sorted(str(model.get("id", "")) for model in models if model.get("id"))

    def ask(self, history: list[dict[str, str]]) -> str:
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            *history,
        ]
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        data = self._request_json("POST", "chat/completions", payload)
        choices = data.get("choices", [])
        if not choices:
            return "Модель вернула пустой ответ."

        message = choices[0].get("message", {})
        answer = str(message.get("content", "")).strip()
        if answer:
            return answer

        reasoning = str(message.get("reasoning_content", "")).strip()
        if reasoning:
            return "Модель ушла в режим рассуждения и не вернула финальный ответ. Попробуйте отправить запрос ещё раз."
        return "Модель вернула пустой ответ."


    def _request_json(self, method: str, endpoint: str, payload: dict[str, object] | None = None) -> dict[str, object]:
        url = f"{self.config.api_base_url}/{endpoint.lstrip('/')}"
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key or 'lm-studio'}",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as error:
            raw_error = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LM Studio HTTP {error.code}: {raw_error or error.reason}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Не удалось подключиться к LM Studio: {error.reason}") from error

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"LM Studio вернула не JSON: {raw[:500]}") from error
        if not isinstance(parsed, dict):
            raise RuntimeError("LM Studio вернула неожиданный JSON-ответ.")
        return parsed


