from .base import Provider, ProviderResponse
from .openai_provider import ChatGPTProvider
from .anthropic_provider import ClaudeProvider
from .google_provider import GeminiProvider
from .xai_provider import GrokProvider

PROVIDERS: dict[str, type[Provider]] = {
    "chatgpt": ChatGPTProvider,
    "claude": ClaudeProvider,
    "gemini": GeminiProvider,
    "grok": GrokProvider,
}

__all__ = [
    "Provider",
    "ProviderResponse",
    "PROVIDERS",
    "ChatGPTProvider",
    "ClaudeProvider",
    "GeminiProvider",
    "GrokProvider",
]
