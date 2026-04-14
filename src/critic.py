"""Step 3 of the pipeline — LLM-as-judge quality critique of generated flashcards."""

import json
from src.llm import call_llm_json

SYSTEM_PROMPT_TEMPLATE = """You are a strict quality reviewer for Anki flashcards. You evaluate each card against established flashcard design principles.

Score each card on these dimensions (1-5 scale):
- truthfulness: Is the content factually accurate?
- atomicity: Does it test exactly ONE fact? (5 = perfectly atomic, 1 = tests multiple facts)
- self_containment: Can it be answered without external context? (5 = fully standalone)
- clarity: Is the question/answer unambiguous? (5 = crystal clear)
- relevance: Is this worth memorizing? (5 = essential, 1 = trivial)

A card PASSES if all scores are >= {min_score} and the average is >= {min_avg}.
A card FAILS otherwise and should be filtered out.

Be strict. Low-quality cards waste the learner's time."""

USER_PROMPT_TEMPLATE = """Review the following flashcards for quality.

FLASHCARDS:
{cards_json}

The passing criteria: all individual scores >= {min_score} AND average score >= {min_avg}.

For each card, return a JSON object with:
- "card_index": the index of the card (0-based)
- "scores": {{"truthfulness": int, "atomicity": int, "self_containment": int, "clarity": int, "relevance": int}}
- "pass": boolean (true if all scores >= {min_score} AND average >= {min_avg})
- "reason": brief explanation if the card fails (empty string if it passes)

Return ONLY a JSON array of review objects, no other text."""


def critique_cards(
    cards: list[dict], client, model: str,
    min_score: int = 3, min_avg: float = 3.5,
) -> tuple[list[dict], list[dict]]:
    """Critique a batch of cards. Returns (passed_cards, reviews).

    min_score: minimum individual dimension score to pass
    min_avg: minimum average across all dimensions to pass
    """
    cards_json = json.dumps(
        [{"index": i, **c} for i, c in enumerate(cards)], indent=2
    )

    system = SYSTEM_PROMPT_TEMPLATE.format(min_score=min_score, min_avg=min_avg)
    user = USER_PROMPT_TEMPLATE.format(
        cards_json=cards_json, min_score=min_score, min_avg=min_avg
    )

    reviews = call_llm_json(client, model, system, user)

    passing_indices = {r["card_index"] for r in reviews if r.get("pass")}
    passed_cards = [c for i, c in enumerate(cards) if i in passing_indices]

    return passed_cards, reviews
