#!/usr/bin/env python3
"""Multi-AI Aggregator — query multiple LLMs and get a synthesised answer."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from nicegui import ui, app

from providers import PROVIDERS, Provider
from aggregator import build_providers, fan_out, aggregate

load_dotenv()

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CONTEXT_FILE = Path("personal_context.txt")
DEFAULT_MODELS: dict[str, str] = {
    "chatgpt": "gpt-4o",
    "claude": "claude-sonnet-4-20250514",
    "gemini": "gemini-2.0-flash",
    "grok": "grok-3-latest",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3.2",
}
PROVIDER_LABELS: dict[str, str] = {
    "chatgpt": "ChatGPT",
    "claude": "Claude",
    "gemini": "Gemini",
    "grok": "Grok",
    "groq": "Groq",
    "ollama": "Ollama",
}
KEY_ENV_NAMES: dict[str, str] = {
    "chatgpt": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "grok": "XAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "ollama": "OLLAMA_ENABLED",
}


def _load_context() -> str:
    if CONTEXT_FILE.exists():
        return CONTEXT_FILE.read_text()
    return ""


def _save_context(text: str) -> None:
    CONTEXT_FILE.write_text(text)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
@ui.page("/")
async def index():
    # --- state -----------------------------------------------------------
    api_keys: dict[str, str] = {
        env: os.getenv(env, "") for env in KEY_ENV_NAMES.values()
    }
    # Also load Ollama base URL from env
    api_keys["OLLAMA_BASE_URL"] = os.getenv(
        "OLLAMA_BASE_URL", "http://localhost:11434/v1"
    )
    enabled: dict[str, bool] = {
        name: bool(api_keys.get(KEY_ENV_NAMES[name])) for name in PROVIDERS
    }
    models: dict[str, str] = dict(DEFAULT_MODELS)
    aggregator_choice: dict[str, str] = {
        "value": os.getenv("AGGREGATOR_PROVIDER", "claude")
    }

    # --- theme / header --------------------------------------------------
    ui.dark_mode(True)
    ui.add_head_html("""
    <style>
        .q-card { border-radius: 12px !important; }
        .nicegui-content { max-width: 1200px; margin: auto; }
        .provider-card { transition: box-shadow 0.2s; }
        .provider-card:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
    </style>
    """)

    with ui.header().classes("items-center justify-between bg-slate-900"):
        ui.label("Multi-AI Aggregator").classes(
            "text-2xl font-bold tracking-tight"
        )
        with ui.row().classes("gap-2"):
            dark_toggle = ui.dark_mode()
            ui.button(icon="settings", on_click=lambda: settings_dialog.open()).props(
                "flat round color=white"
            )

    # --- settings dialog -------------------------------------------------
    with ui.dialog() as settings_dialog, ui.card().classes("w-[600px]"):
        ui.label("Settings").classes("text-xl font-bold mb-2")
        ui.separator()

        ui.label("API Keys").classes("text-lg font-semibold mt-4")
        ui.label("Keys are loaded from your .env file by default. "
                 "Override them here for this session.").classes("text-xs text-gray-400")
        for pname, env_key in KEY_ENV_NAMES.items():
            if pname == "ollama":
                continue  # Ollama gets its own section below
            ui.input(
                label=f"{PROVIDER_LABELS[pname]} ({env_key})",
                value=api_keys.get(env_key, ""),
                password=True,
                password_toggle_button=True,
                on_change=lambda e, ek=env_key: api_keys.update({ek: e.value}),
            ).classes("w-full")

        ui.label("Ollama (Local)").classes("text-lg font-semibold mt-4")
        ui.label("Run 'ollama serve' locally, then enable here. No API key needed.").classes(
            "text-xs text-gray-400"
        )
        ui.switch(
            "Enable Ollama",
            value=bool(api_keys.get("OLLAMA_ENABLED")),
            on_change=lambda e: api_keys.update(
                {"OLLAMA_ENABLED": "enabled" if e.value else ""}
            ),
        )
        ui.input(
            label="Ollama Base URL",
            value=api_keys.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            on_change=lambda e: api_keys.update({"OLLAMA_BASE_URL": e.value}),
        ).classes("w-full")

        ui.label("Models").classes("text-lg font-semibold mt-4")
        for pname in PROVIDERS:
            ui.input(
                label=f"{PROVIDER_LABELS[pname]} model",
                value=models[pname],
                on_change=lambda e, p=pname: models.update({p: e.value}),
            ).classes("w-full")

        ui.label("Aggregator Provider").classes("text-lg font-semibold mt-4")
        ui.select(
            options={k: PROVIDER_LABELS[k] for k in PROVIDERS},
            value=aggregator_choice["value"],
            on_change=lambda e: aggregator_choice.update({"value": e.value}),
        ).classes("w-full")

        with ui.row().classes("w-full justify-end mt-4"):
            ui.button("Close", on_click=settings_dialog.close).props("flat")

    # --- personal context dialog -----------------------------------------
    with ui.dialog() as context_dialog, ui.card().classes("w-[600px]"):
        ui.label("Personal Context").classes("text-xl font-bold mb-2")
        ui.separator()
        ui.label(
            "This text is sent as the system prompt to every provider. "
            "Use it to tell the AIs about yourself, your preferences, "
            "or any standing instructions."
        ).classes("text-xs text-gray-400 mb-2")
        context_area = ui.textarea(
            value=_load_context(),
        ).classes("w-full min-h-[200px]")
        with ui.row().classes("w-full justify-end mt-4"):
            ui.button(
                "Save",
                on_click=lambda: (
                    _save_context(context_area.value),
                    ui.notify("Context saved", type="positive"),
                    context_dialog.close(),
                ),
            ).props("color=primary")
            ui.button("Cancel", on_click=context_dialog.close).props("flat")

    # --- main layout -----------------------------------------------------
    with ui.column().classes("w-full p-4 gap-4"):
        # Provider toggles
        with ui.row().classes("w-full items-center gap-4 flex-wrap"):
            ui.label("Providers:").classes("font-semibold")
            provider_switches: dict[str, ui.switch] = {}
            for pname in PROVIDERS:
                sw = ui.switch(
                    PROVIDER_LABELS[pname],
                    value=enabled[pname],
                    on_change=lambda e, p=pname: enabled.update({p: e.value}),
                )
                has_key = bool(api_keys.get(KEY_ENV_NAMES[pname]))
                if not has_key:
                    tip = ("Not enabled — enable it in Settings" if pname == "ollama"
                           else "No API key configured — add it in Settings")
                    sw.tooltip(tip)
                provider_switches[pname] = sw

            ui.space()
            ui.button(
                "Personal Context",
                icon="person",
                on_click=lambda: context_dialog.open(),
            ).props("outline")

        # Query input
        query_input = ui.textarea(
            label="Your question",
            placeholder="Ask anything — it will be sent to all enabled providers…",
        ).classes("w-full")

        submit_btn = ui.button("Ask All", icon="send").props(
            "color=primary size=lg"
        ).classes("self-end")

        # Results area
        results_container = ui.column().classes("w-full gap-4")

    # --- handler ---------------------------------------------------------
    async def on_submit():
        question = query_input.value.strip()
        if not question:
            ui.notify("Please enter a question", type="warning")
            return

        results_container.clear()
        submit_btn.disable()

        active = {
            name for name, on in enabled.items()
            if on and api_keys.get(KEY_ENV_NAMES[name])
        }
        if not active:
            ui.notify("Enable at least one provider with a valid API key", type="negative")
            submit_btn.enable()
            return

        providers = build_providers(api_keys, models, active)
        system = _load_context()

        # Show a spinner while waiting
        with results_container:
            spinner_row = ui.row().classes("w-full justify-center py-8")
            with spinner_row:
                ui.spinner("dots", size="xl")
                ui.label("Querying providers…").classes("ml-4 text-lg")

        # Fan out queries
        responses = await fan_out(question, providers, system)

        # Remove spinner
        results_container.clear()

        # Display individual responses
        with results_container:
            ui.label("Individual Responses").classes("text-xl font-bold")
            with ui.row().classes("w-full gap-4 flex-wrap"):
                for resp in responses:
                    with ui.card().classes(
                        "provider-card flex-1 min-w-[280px]"
                    ):
                        color = "red" if resp.error else "green"
                        with ui.row().classes("items-center gap-2 mb-2"):
                            ui.icon("smart_toy").classes(f"text-{color}")
                            ui.label(
                                f"{PROVIDER_LABELS.get(resp.provider, resp.provider)}"
                            ).classes("text-lg font-semibold")
                            ui.badge(resp.model).props("outline")
                            ui.badge(f"{resp.elapsed:.1f}s").props(
                                f"color={'red' if resp.error else 'grey'}"
                            )
                        ui.separator()
                        if resp.error:
                            ui.label(f"Error: {resp.error}").classes(
                                "text-red-400"
                            )
                        else:
                            ui.markdown(resp.text).classes("w-full")

            # Aggregation
            if len([r for r in responses if not r.error]) > 1:
                ui.separator()
                agg_spinner_row = ui.row().classes("w-full justify-center py-4")
                with agg_spinner_row:
                    ui.spinner("dots", size="lg")
                    ui.label("Synthesising…").classes("ml-4")

                agg_name = aggregator_choice["value"]
                agg_env = KEY_ENV_NAMES.get(agg_name)
                agg_key = api_keys.get(agg_env, "") if agg_env else ""
                if not agg_key:
                    # Fall back to first available
                    for name in active:
                        agg_name = name
                        agg_key = api_keys.get(KEY_ENV_NAMES[name], "")
                        if agg_key:
                            break

                if agg_key:
                    agg_provider = PROVIDERS[agg_name](
                        api_key=agg_key, model=models.get(agg_name)
                    )
                    agg_resp = await aggregate(question, responses, agg_provider)
                    agg_spinner_row.delete()

                    with results_container:
                        with ui.card().classes(
                            "w-full border-2 border-blue-500"
                        ):
                            with ui.row().classes("items-center gap-2 mb-2"):
                                ui.icon("auto_awesome").classes("text-blue-400")
                                ui.label("Synthesised Answer").classes(
                                    "text-xl font-bold"
                                )
                                ui.badge(
                                    f"via {PROVIDER_LABELS.get(agg_resp.provider, agg_resp.provider)}"
                                ).props("outline color=blue")
                                if agg_resp.elapsed:
                                    ui.badge(f"{agg_resp.elapsed:.1f}s").props(
                                        "color=grey"
                                    )
                            ui.separator()
                            if agg_resp.error:
                                ui.label(
                                    f"Aggregation error: {agg_resp.error}"
                                ).classes("text-red-400")
                            else:
                                ui.markdown(agg_resp.text).classes("w-full")
                else:
                    agg_spinner_row.delete()
                    with results_container:
                        ui.label(
                            "No API key available for the aggregator — "
                            "configure one in Settings."
                        ).classes("text-yellow-400")

        submit_btn.enable()

    submit_btn.on_click(on_submit)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
ui.run(
    title="Multi-AI Aggregator",
    favicon="🤖",
    port=8080,
    dark=True,
    reload=False,
)
