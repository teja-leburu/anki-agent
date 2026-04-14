"""Automated evaluation framework for generated flashcards.

Combines heuristic checks, LLM-as-judge scoring, Bloom's taxonomy
classification, and source coverage measurement.
"""

import json
import re
from src.llm import call_llm_json


# ---------------------------------------------------------------------------
# 1. Heuristic checks (fast, no LLM needed)
# ---------------------------------------------------------------------------

def heuristic_check(card: dict) -> dict:
    """Run rule-based quality checks on a single card."""
    results = {}
    front = card.get("front", "")
    back = card.get("back", "")

    front_words = len(front.split())
    results["front_length"] = {
        "passed": front_words <= 30,
        "detail": f"{front_words} words (max 30)",
    }

    if card.get("type") != "cloze":
        back_words = len(back.split())
        results["back_length"] = {
            "passed": back_words <= 20,
            "detail": f"{back_words} words (max 20)",
        }

    results["front_nonempty"] = {
        "passed": len(front.strip()) > 0,
        "detail": "front is non-empty" if front.strip() else "front is empty",
    }

    if card.get("type") == "cloze":
        has_cloze = bool(re.search(r"\{\{c\d+::", front))
        results["cloze_format"] = {
            "passed": has_cloze,
            "detail": "valid cloze syntax" if has_cloze else "missing {{c1::...}} syntax",
        }

    if card.get("type") == "basic":
        q_count = front.count("?")
        results["single_question"] = {
            "passed": q_count <= 1,
            "detail": f"{q_count} question marks",
        }

    return results


def run_heuristics(cards: list[dict]) -> list[dict]:
    """Run heuristic checks on all cards."""
    return [{"card_index": i, "checks": heuristic_check(c)} for i, c in enumerate(cards)]


# ---------------------------------------------------------------------------
# 2. LLM-as-judge scoring
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """You are a strict flashcard quality evaluator. Score each card on these dimensions (1-5):

- truthfulness: Factual accuracy (5 = certainly correct, 1 = contains errors)
- atomicity: Tests exactly one fact (5 = perfectly atomic, 1 = tests 3+ facts)
- self_containment: Answerable without source text (5 = fully standalone, 1 = requires context)
- clarity: Unambiguous question and answer (5 = crystal clear, 1 = confusing)
- relevance: Worth memorizing (5 = essential knowledge, 1 = trivial)

Be strict and consistent. Provide brief justification for each score."""

JUDGE_USER = """Rate these flashcards on the rubric.

FLASHCARDS:
{cards_json}

Return a JSON array where each element has:
- "card_index": int
- "scores": {{"truthfulness": int, "atomicity": int, "self_containment": int, "clarity": int, "relevance": int}}
- "justification": one sentence explaining the weakest dimension

Return ONLY the JSON array."""


def llm_judge_score(cards: list[dict], client, model: str) -> list[dict]:
    """Score cards using an LLM judge."""
    cards_json = json.dumps(
        [{"index": i, **c} for i, c in enumerate(cards)], indent=2
    )
    return call_llm_json(client, model, JUDGE_SYSTEM,
                         JUDGE_USER.format(cards_json=cards_json))


# ---------------------------------------------------------------------------
# 3. Bloom's Taxonomy classification
# ---------------------------------------------------------------------------

BLOOM_SYSTEM = """Classify each flashcard by its Bloom's Taxonomy cognitive level:
- remember: Recall facts or basic concepts
- understand: Explain ideas or concepts
- apply: Use information in new situations
- analyze: Draw connections among ideas
- evaluate: Justify a stand or decision
- create: Produce new or original work

Most flashcards will be "remember" or "understand." Higher levels indicate deeper learning."""

BLOOM_USER = """Classify these flashcards by Bloom's Taxonomy level.

FLASHCARDS:
{cards_json}

Return a JSON array where each element has:
- "card_index": int
- "bloom_level": one of "remember", "understand", "apply", "analyze", "evaluate", "create"

Return ONLY the JSON array."""


def classify_blooms(cards: list[dict], client, model: str) -> list[dict]:
    """Classify each card by Bloom's Taxonomy level."""
    cards_json = json.dumps(
        [{"index": i, **c} for i, c in enumerate(cards)], indent=2
    )
    return call_llm_json(client, model, BLOOM_SYSTEM,
                         BLOOM_USER.format(cards_json=cards_json), max_tokens=2048)


# ---------------------------------------------------------------------------
# 4. Source coverage measurement
# ---------------------------------------------------------------------------

COVERAGE_SYSTEM = """You are evaluating how well a set of flashcards covers the key concepts in a source text.
Identify the key concepts in the source, then check which ones are covered by at least one flashcard."""

COVERAGE_USER = """Given this source text and these flashcards, evaluate coverage.

SOURCE TEXT:
{source_text}

FLASHCARDS:
{cards_json}

Return a JSON object with:
- "key_concepts": list of strings (the important concepts in the source)
- "covered": list of strings (concepts that at least one flashcard tests)
- "missing": list of strings (concepts with no corresponding flashcard)
- "coverage_pct": float (percentage of key concepts covered)

Return ONLY the JSON object."""


def measure_coverage(source_text: str, cards: list[dict], client, model: str) -> dict:
    """Measure what fraction of source concepts are covered by the cards."""
    cards_json = json.dumps(cards, indent=2)
    return call_llm_json(client, model, COVERAGE_SYSTEM,
                         COVERAGE_USER.format(source_text=source_text, cards_json=cards_json),
                         max_tokens=2048)


# ---------------------------------------------------------------------------
# 5. Aggregate evaluation
# ---------------------------------------------------------------------------

def evaluate_cards(cards: list[dict], source_text: str, client, model: str) -> dict:
    """Run the full evaluation suite on a set of cards."""
    heuristics = run_heuristics(cards)
    judge_scores = llm_judge_score(cards, client, model)

    bloom_classes = classify_blooms(cards, client, model)
    bloom_dist = {}
    for entry in bloom_classes:
        level = entry.get("bloom_level", "unknown")
        bloom_dist[level] = bloom_dist.get(level, 0) + 1

    coverage = measure_coverage(source_text, cards, client, model)

    all_scores = [s["scores"] for s in judge_scores if "scores" in s]
    dimensions = ["truthfulness", "atomicity", "self_containment", "clarity", "relevance"]
    avg_scores = {}
    for dim in dimensions:
        vals = [s[dim] for s in all_scores if dim in s]
        avg_scores[dim] = round(sum(vals) / len(vals), 2) if vals else 0

    heuristic_pass_rate = 0
    if heuristics:
        passed = sum(
            1 for h in heuristics
            if all(c["passed"] for c in h["checks"].values())
        )
        heuristic_pass_rate = round(passed / len(heuristics) * 100, 1)

    return {
        "card_count": len(cards),
        "heuristic_pass_rate": heuristic_pass_rate,
        "avg_judge_scores": avg_scores,
        "avg_overall": round(sum(avg_scores.values()) / len(avg_scores), 2) if avg_scores else 0,
        "bloom_distribution": bloom_dist,
        "coverage": coverage,
        "details": {
            "heuristics": heuristics,
            "judge_scores": judge_scores,
            "bloom_classes": bloom_classes,
        },
    }
