"""Step 1 of the pipeline — extract key concepts from a text chunk."""

from src.llm import call_llm_json

SYSTEM_PROMPT = """You are an expert at identifying the most important concepts in educational material.

Your job is to extract key concepts that are worth memorizing. For each concept, provide:
- A clear label/name
- The core fact, definition, or relationship
- Why it matters (one sentence of context)

Focus on:
- Definitions and terminology
- Key relationships and causal links
- Important facts, figures, or thresholds
- Distinctions between similar concepts
- Core principles and rules

Skip:
- Trivial or obvious statements
- Filler content, transitions, or meta-commentary
- Information that requires extensive context to understand"""

USER_PROMPT_TEMPLATE = """Extract the key concepts worth memorizing from this text.

SOURCE MATERIAL:
{text}

Return a JSON array of concept objects. Each object must have:
- "label": short name for the concept (2-5 words)
- "fact": the core fact, definition, or relationship (1-2 sentences)
- "context": why this matters or how it connects to the broader topic (1 sentence)
- "concept_type": one of "definition", "relationship", "fact", "distinction", "principle"

Return ONLY the JSON array, no other text."""


def extract_concepts(text: str, client, model: str) -> list[dict]:
    """Extract key concepts from a text chunk."""
    return call_llm_json(client, model, SYSTEM_PROMPT,
                         USER_PROMPT_TEMPLATE.format(text=text))
