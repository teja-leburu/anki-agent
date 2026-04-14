"""Step 2 of the pipeline — generate flashcards from extracted concepts using few-shot examples."""

import json
import anthropic

SYSTEM_PROMPT = """You are an expert flashcard creator following Piotr Wozniak's "20 Rules of Formulating Knowledge."

You will receive structured concepts and must generate high-quality Anki flashcards for each one.

Rules:
- Minimum information principle: ONE atomic fact per card
- Self-contained: answerable without the source text
- Front: under 30 words
- Back: under 20 words
- Prefer cloze deletions for definitions (use {{c1::...}} syntax)
- Each card answerable in under 8 seconds"""

FEW_SHOT_EXAMPLES = """Here are examples of well-crafted cards for different concept types:

CONCEPT: {"label": "Photosynthesis equation", "fact": "Photosynthesis converts CO2 and H2O into glucose and oxygen using light energy.", "concept_type": "definition"}
CARDS:
[
  {"type": "cloze", "front": "Photosynthesis converts {{c1::CO2}} and {{c2::H2O}} into {{c3::glucose}} and {{c4::oxygen}} using light energy.", "back": "", "tags": ["biology", "photosynthesis"]},
  {"type": "basic", "front": "What energy source drives photosynthesis?", "back": "Light energy (solar radiation)", "tags": ["biology", "photosynthesis"]}
]

CONCEPT: {"label": "TCP vs UDP", "fact": "TCP provides reliable, ordered delivery with error checking while UDP provides fast, connectionless delivery without guarantees.", "concept_type": "distinction"}
CARDS:
[
  {"type": "basic", "front": "Which transport protocol guarantees ordered, reliable delivery?", "back": "TCP (Transmission Control Protocol)", "tags": ["networking"]},
  {"type": "basic", "front": "Which transport protocol is connectionless and prioritizes speed over reliability?", "back": "UDP (User Datagram Protocol)", "tags": ["networking"]}
]

CONCEPT: {"label": "p-value threshold", "fact": "A p-value below 0.05 is conventionally used to reject the null hypothesis, indicating statistical significance.", "concept_type": "fact"}
CARDS:
[
  {"type": "cloze", "front": "A p-value below {{c1::0.05}} is the conventional threshold for {{c2::statistical significance}}.", "back": "", "tags": ["statistics"]}
]"""

USER_PROMPT_TEMPLATE = """Generate Anki flashcards for each of the following extracted concepts.

CONCEPTS:
{concepts_json}

{few_shot}

Now generate cards for ALL the concepts above. For each concept, generate 1-3 cards depending on complexity. Prefer cloze deletions for definitions and facts. Use basic cards for relationships and distinctions.

Return ONLY a JSON array of flashcard objects with keys: type, front, back, tags."""


def generate_cards_from_concepts(
    concepts: list[dict], client: anthropic.Anthropic, model: str
) -> list[dict]:
    """Generate flashcards from a list of extracted concepts using few-shot prompting."""
    concepts_json = json.dumps(concepts, indent=2)

    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    concepts_json=concepts_json, few_shot=FEW_SHOT_EXAMPLES
                ),
            }
        ],
    )

    response_text = message.content[0].text.strip()
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        response_text = "\n".join(lines)

    return json.loads(response_text)
