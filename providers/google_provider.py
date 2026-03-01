from __future__ import annotations

import asyncio
import google.generativeai as genai

from .base import Provider, ProviderResponse


class GeminiProvider(Provider):
    name = "gemini"
    default_model = "gemini-2.0-flash"

    async def query(self, prompt: str, system: str = "") -> ProviderResponse:
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(
            self.model,
            system_instruction=system if system else None,
        )

        resp = await asyncio.to_thread(model.generate_content, prompt)
        text = resp.text or ""
        return ProviderResponse(provider=self.name, model=self.model, text=text)
