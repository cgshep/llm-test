from .base import Provider, ProviderResponse
from .openai_provider import ChatGPTProvider
from .anthropic_provider import ClaudeProvider
from .google_provider import GeminiProvider
from .xai_provider import GrokProvider
from .groq_provider import GroqProvider
from .ollama_provider import OllamaProvider

PROVIDERS: dict[str, type[Provider]] = {
    "chatgpt": ChatGPTProvider,
    "claude": ClaudeProvider,
    "gemini": GeminiProvider,
    "grok": GrokProvider,
    "groq": GroqProvider,
    "ollama": OllamaProvider,
}

__all__ = [
    "Provider",
    "ProviderResponse",
    "PROVIDERS",
    "ChatGPTProvider",
    "ClaudeProvider",
    "GeminiProvider",
    "GrokProvider",
    "GroqProvider",
    "OllamaProvider",
]
