from __future__ import annotations

import asyncio
from openai import OpenAI

from .base import Provider, ProviderResponse


class OllamaProvider(Provider):
    """Ollama — free local models via OpenAI-compatible API."""

    name = "ollama"
    default_model = "llama3.2"

    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None):
        # Ollama doesn't need an API key; accept a dummy one so the
        # build_providers flow works uniformly.
        super().__init__(api_key=api_key or "ollama", model=model)
        self.base_url = base_url or "http://localhost:11434/v1"

    async def query(self, prompt: str, system: str = "") -> ProviderResponse:
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = await asyncio.to_thread(
            client.chat.completions.create,
            model=self.model,
            messages=messages,
        )
        text = resp.choices[0].message.content or ""
        return ProviderResponse(provider=self.name, model=self.model, text=text)
