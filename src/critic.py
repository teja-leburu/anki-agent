"""Step 3 of the pipeline — LLM-as-judge quality critique of generated flashcards."""

import json
import anthropic

SYSTEM_PROMPT = """You are a strict quality reviewer for Anki flashcards. You evaluate each card against established flashcard design principles.

Score each card on these dimensions (1-5 scale):
- truthfulness: Is the content factually accurate?
- atomicity: Does it test exactly ONE fact? (5 = perfectly atomic, 1 = tests multiple facts)
- self_containment: Can it be answered without external context? (5 = fully standalone)
- clarity: Is the question/answer unambiguous? (5 = crystal clear)
- relevance: Is this worth memorizing? (5 = essential, 1 = trivial)

A card PASSES if all scores are >= 3 and the average is >= 3.5.
A card FAILS otherwise and should be filtered out.

Be strict. Low-quality cards waste the learner's time."""

USER_PROMPT_TEMPLATE = """Review the following flashcards for quality.

FLASHCARDS:
{cards_json}

For each card, return a JSON object with:
- "card_index": the index of the card (0-based)
- "scores": {{"truthfulness": int, "atomicity": int, "self_containment": int, "clarity": int, "relevance": int}}
- "pass": boolean (true if all scores >= 3 AND average >= 3.5)
- "reason": brief explanation if the card fails (empty string if it passes)

Return ONLY a JSON array of review objects, no other text."""


def critique_cards(
    cards: list[dict], client: anthropic.Anthropic, model: str
) -> tuple[list[dict], list[dict]]:
    """Critique a batch of cards. Returns (passed_cards, reviews)."""
    cards_json = json.dumps(
        [{"index": i, **c} for i, c in enumerate(cards)], indent=2
    )

    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(cards_json=cards_json)}
        ],
    )

    response_text = message.content[0].text.strip()
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        response_text = "\n".join(lines)

    reviews = json.loads(response_text)

    passing_indices = {r["card_index"] for r in reviews if r.get("pass")}
    passed_cards = [c for i, c in enumerate(cards) if i in passing_indices]

    return passed_cards, reviews
