"""Prompt strategy variations for card generation.

Each strategy is a function that takes concepts + client + model and returns cards.
This lets us A/B test different prompt engineering approaches on the same concepts.
"""

import json
import anthropic


def _call_llm(client: anthropic.Anthropic, model: str, system: str, user: str) -> str:
    """Shared LLM call with response cleaning."""
    message = client.messages.create(
        model=model, max_tokens=4096, system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = message.content[0].text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text


# ---------------------------------------------------------------------------
# Strategy 1: Chain-of-Thought generation (Wei et al., 2022)
# ---------------------------------------------------------------------------

COT_SYSTEM = """You are an expert flashcard creator. You use step-by-step reasoning to produce high-quality Anki cards.

For each concept, think through:
1. What is the ONE most important thing to test?
2. Is this better as a basic Q&A or cloze deletion?
3. Is the front under 30 words and the back under 20 words?
4. Could someone answer this without seeing the source material?

Then produce the card."""

COT_USER = """Generate Anki flashcards for these concepts. For EACH concept, first write your reasoning inside <thinking> tags, then output the card(s).

CONCEPTS:
{concepts_json}

After your reasoning for all concepts, return a final JSON array of ALL flashcard objects with keys: type, front, back, tags.

The JSON array must appear after all reasoning, on its own, with no other text around it."""


def strategy_chain_of_thought(
    concepts: list[dict], client: anthropic.Anthropic, model: str
) -> list[dict]:
    """Use chain-of-thought prompting for card generation."""
    concepts_json = json.dumps(concepts, indent=2)
    response = _call_llm(client, model, COT_SYSTEM,
                         COT_USER.format(concepts_json=concepts_json))
    # Extract the JSON array from after the thinking blocks
    # Find the last [ ... ] block in the response
    last_bracket = response.rfind("[")
    if last_bracket != -1:
        json_str = response[last_bracket:]
        # Find matching closing bracket
        depth = 0
        for i, ch in enumerate(json_str):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    json_str = json_str[: i + 1]
                    break
        return json.loads(json_str)
    return json.loads(response)


# ---------------------------------------------------------------------------
# Strategy 2: Minimal few-shot (different example set — tests sensitivity)
# ---------------------------------------------------------------------------

MINIMAL_FEW_SHOT_SYSTEM = """You are an expert flashcard creator following the minimum information principle.
Each card must test exactly one atomic fact. Cards must be self-contained."""

MINIMAL_FEW_SHOT_USER = """Generate Anki flashcards for these concepts.

Example:
CONCEPT: "Mitochondria function" — "Mitochondria are the powerhouses of the cell, producing ATP through oxidative phosphorylation."
CARDS: [{{"type": "basic", "front": "What is the primary function of mitochondria?", "back": "Producing ATP via oxidative phosphorylation", "tags": ["biology"]}}]

CONCEPTS:
{concepts_json}

Return ONLY a JSON array of flashcard objects with keys: type, front, back, tags."""


def strategy_minimal_few_shot(
    concepts: list[dict], client: anthropic.Anthropic, model: str
) -> list[dict]:
    """Use a minimal single-example few-shot prompt."""
    concepts_json = json.dumps(concepts, indent=2)
    response = _call_llm(client, model, MINIMAL_FEW_SHOT_SYSTEM,
                         MINIMAL_FEW_SHOT_USER.format(concepts_json=concepts_json))
    return json.loads(response)


# ---------------------------------------------------------------------------
# Strategy 3: Source-type-specific prompts
# ---------------------------------------------------------------------------

SOURCE_TYPE_SYSTEMS = {
    "textbook": """You are creating Anki flashcards from a dense textbook.
Focus on: precise definitions, formulas, named theorems, and distinctions between similar terms.
Prefer cloze deletions for definitions. Use {{c1::...}} syntax.
Each card: one atomic fact, self-contained, front < 30 words, back < 20 words.""",

    "lecture": """You are creating Anki flashcards from lecture slides or notes.
Lecture content is often terse — expand abbreviations and add enough context for standalone cards.
Focus on: key takeaways, examples given by the instructor, and relationships between topics.
Each card: one atomic fact, self-contained, front < 30 words, back < 20 words.""",

    "paper": """You are creating Anki flashcards from a research paper.
Focus on: the paper's main contribution, methodology, key findings, and limitations.
Use basic Q&A cards for methodology and findings. Use cloze for specific metrics or thresholds.
Each card: one atomic fact, self-contained, front < 30 words, back < 20 words.""",
}

SOURCE_TYPE_USER = """Generate Anki flashcards for these concepts extracted from a {source_type}.

CONCEPTS:
{concepts_json}

Return ONLY a JSON array of flashcard objects with keys: type, front, back, tags."""


def strategy_source_specific(
    concepts: list[dict], client: anthropic.Anthropic, model: str,
    source_type: str = "textbook",
) -> list[dict]:
    """Use a source-type-specific system prompt."""
    system = SOURCE_TYPE_SYSTEMS.get(source_type, SOURCE_TYPE_SYSTEMS["textbook"])
    concepts_json = json.dumps(concepts, indent=2)
    response = _call_llm(client, model, system,
                         SOURCE_TYPE_USER.format(
                             source_type=source_type, concepts_json=concepts_json))
    return json.loads(response)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STRATEGIES = {
    "few_shot": None,  # default Phase 2 strategy (in card_generator.py)
    "chain_of_thought": strategy_chain_of_thought,
    "minimal_few_shot": strategy_minimal_few_shot,
    "source_textbook": lambda c, cl, m: strategy_source_specific(c, cl, m, "textbook"),
    "source_lecture": lambda c, cl, m: strategy_source_specific(c, cl, m, "lecture"),
    "source_paper": lambda c, cl, m: strategy_source_specific(c, cl, m, "paper"),
}
