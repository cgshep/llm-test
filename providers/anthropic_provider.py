from __future__ import annotations

import asyncio
from anthropic import Anthropic

from .base import Provider, ProviderResponse


class ClaudeProvider(Provider):
    name = "claude"
    default_model = "claude-sonnet-4-20250514"

    async def query(self, prompt: str, system: str = "") -> ProviderResponse:
        client = Anthropic(api_key=self.api_key)
        kwargs: dict = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        resp = await asyncio.to_thread(client.messages.create, **kwargs)
        text = resp.content[0].text if resp.content else ""
        return ProviderResponse(provider=self.name, model=self.model, text=text)
