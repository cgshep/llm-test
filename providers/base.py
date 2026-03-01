from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class ProviderResponse:
    provider: str
    model: str
    text: str
    elapsed: float = 0.0
    error: str | None = None


class Provider(ABC):
    """Base class for AI providers."""

    name: str = "base"
    default_model: str = ""

    def __init__(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        self.model = model or self.default_model

    @abstractmethod
    async def query(self, prompt: str, system: str = "") -> ProviderResponse:
        ...

    async def safe_query(self, prompt: str, system: str = "") -> ProviderResponse:
        """Query with error handling and timing."""
        start = time.perf_counter()
        try:
            resp = await self.query(prompt, system)
            resp.elapsed = time.perf_counter() - start
            return resp
        except Exception as e:
            return ProviderResponse(
                provider=self.name,
                model=self.model,
                text="",
                elapsed=time.perf_counter() - start,
                error=str(e),
            )
