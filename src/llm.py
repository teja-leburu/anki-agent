"""Unified LLM client abstraction supporting both Anthropic and OpenAI.

All modules call through this interface so we can swap providers for experiments.
"""

import anthropic
import openai
from src.utils import parse_json_response


def create_client(provider: str):
    """Create an LLM client for the given provider."""
    if provider == "openai":
        return openai.OpenAI()
    return anthropic.Anthropic()


def infer_provider(model: str) -> str:
    """Infer the provider from the model name."""
    if model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4"):
        return "openai"
    return "anthropic"


def call_llm(client, model: str, system: str, user: str, max_tokens: int = 4096) -> str:
    """Call an LLM and return the raw response text.

    Works with both Anthropic and OpenAI clients.
    """
    if isinstance(client, openai.OpenAI):
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content
    else:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return message.content[0].text


def call_llm_json(client, model: str, system: str, user: str, max_tokens: int = 4096):
    """Call an LLM and parse the response as JSON."""
    raw = call_llm(client, model, system, user, max_tokens)
    return parse_json_response(raw)
