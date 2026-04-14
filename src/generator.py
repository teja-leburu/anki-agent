"""Baseline single-prompt flashcard generator using Claude."""

import json
import anthropic
from src.utils import parse_json_response

SYSTEM_PROMPT = """You are an expert flashcard creator following Piotr Wozniak's "20 Rules of Formulating Knowledge."

Your job is to generate high-quality Anki flashcards from source material.

Rules you MUST follow:
- Minimum information principle: each card tests exactly ONE atomic fact
- Cards must be self-contained — answerable without seeing the source text
- Front of card: under 30 words, clear question or cloze prompt
- Back of card: under 20 words, direct answer
- Prefer cloze deletions for definitions and key facts (use {{c1::...}} syntax)
- Never create cards for trivial or obvious information
- Avoid cards that test enumeration/lists — break into individual cards
- Each card should be answerable in under 8 seconds"""

USER_PROMPT_TEMPLATE = """Generate Anki flashcards from the following source material.

SOURCE MATERIAL:
{text}

Return a JSON array of flashcard objects. Each object must have:
- "type": either "basic" or "cloze"
- "front": the question or cloze text (use {{{{c1::...}}}} for cloze deletions)
- "back": the answer (for basic cards) or empty string (for cloze cards where the answer is in the cloze)
- "tags": list of 1-2 topic tags for the card

Return ONLY the JSON array, no other text."""


def generate_flashcards(text: str, model: str = "claude-sonnet-4-20250514") -> list[dict]:
    """Generate flashcards from a text chunk using Claude."""
    client = anthropic.Anthropic()

    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)}
        ],
    )

    return parse_json_response(message.content[0].text)
