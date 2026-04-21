from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI


@dataclass
class LMStudioConfig:
    base_url: str = "http://127.0.0.1:1234"
    api_key: str = "lm-studio"
    model: str = "google/gemma-4-e2b"
    temperature: float = 0.7
    max_tokens: int = 500
    system_prompt: str = (
        "Ты полезный ИИ-ассистент. Отвечай кратко, понятно и по делу. "
        "Если пользователь пишет на русском, отвечай на русском языке."
    )

    @property
    def api_base_url(self) -> str:
        base = self.base_url.strip().rstrip("/")
        return base if base.endswith("/v1") else f"{base}/v1"


class LMStudioModule:
    """Модуль подключения ИИ-ассистента к LM Studio."""

    def __init__(self, config: LMStudioConfig | None = None) -> None:
        self.config = config or LMStudioConfig()

    def update_config(self, config: LMStudioConfig) -> None:
        self.config = config

    def _build_client(self) -> OpenAI:
        return OpenAI(
            base_url=self.config.api_base_url,
            api_key=self.config.api_key or "lm-studio",
        )

    def get_models(self) -> list[str]:
        client = self._build_client()
        models = client.models.list()
        return sorted(model.id for model in models.data)

    def ask(self, history: list[dict[str, str]]) -> str:
        client = self._build_client()
        messages = [{"role": "system", "content": self.config.system_prompt}, *history]
        response = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        answer = response.choices[0].message.content or ""
        return answer.strip() or "Модель вернула пустой ответ."
