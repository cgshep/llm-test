from __future__ import annotations

import asyncio
from openai import OpenAI

from .base import Provider, ProviderResponse


class GroqProvider(Provider):
    """Groq — free tier, OpenAI-compatible API. Runs Llama, Mixtral, Gemma."""

    name = "groq"
    default_model = "llama-3.3-70b-versatile"

    async def query(self, prompt: str, system: str = "") -> ProviderResponse:
        client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1",
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
