"""Aggregation engine: fans out a query to multiple providers, then synthesizes."""

from __future__ import annotations

import asyncio
from providers import PROVIDERS, Provider, ProviderResponse


AGGREGATION_PROMPT = """\
You have been given a user's original question and the responses from several \
different AI assistants. Your job is to produce a single, high-quality answer \
that synthesises the best parts of every response.

Guidelines:
- Where the responses agree, state the consensus confidently.
- Where they disagree, note the different perspectives and use your judgement.
- If one response has clearly better detail or accuracy, favour it.
- Be concise but thorough. Do not mention the source AIs by name unless it \
  adds value (e.g. "one model noted…").
- Use markdown formatting for readability.

--- ORIGINAL QUESTION ---
{question}

--- RESPONSES ---
{responses}

--- YOUR SYNTHESISED ANSWER ---
"""


def build_providers(
    keys: dict[str, str],
    models: dict[str, str] | None = None,
    enabled: set[str] | None = None,
) -> dict[str, Provider]:
    """Instantiate only the providers the user has keys for and has enabled."""
    models = models or {}
    key_map = {
        "chatgpt": "OPENAI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "grok": "XAI_API_KEY",
        "groq": "GROQ_API_KEY",
        "ollama": "OLLAMA_ENABLED",
    }
    result: dict[str, Provider] = {}
    for name, cls in PROVIDERS.items():
        env_key = key_map.get(name, "")
        api_key = keys.get(env_key, "")
        if not api_key:
            continue
        if enabled is not None and name not in enabled:
            continue
        if name == "ollama":
            base_url = keys.get("OLLAMA_BASE_URL") or None
            result[name] = cls(model=models.get(name), base_url=base_url)
        else:
            result[name] = cls(api_key=api_key, model=models.get(name))
    return result


async def fan_out(
    prompt: str,
    providers: dict[str, Provider],
    system: str = "",
) -> list[ProviderResponse]:
    """Send the prompt to all providers concurrently."""
    tasks = [p.safe_query(prompt, system) for p in providers.values()]
    return await asyncio.gather(*tasks)


async def aggregate(
    question: str,
    responses: list[ProviderResponse],
    aggregator_provider: Provider,
) -> ProviderResponse:
    """Use one provider to synthesise the collected responses."""
    successful = [r for r in responses if not r.error]
    if not successful:
        return ProviderResponse(
            provider="aggregator",
            model="n/a",
            text="All providers returned errors — nothing to aggregate.",
            error="no successful responses",
        )
    if len(successful) == 1:
        # Only one response — no need to aggregate, just return it.
        return ProviderResponse(
            provider="aggregator",
            model=successful[0].model,
            text=successful[0].text,
        )

    resp_block = "\n\n".join(
        f"### Response from {r.provider} ({r.model}):\n{r.text}" for r in successful
    )
    synth_prompt = AGGREGATION_PROMPT.format(question=question, responses=resp_block)
    return await aggregator_provider.safe_query(synth_prompt)
