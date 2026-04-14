"""Shared utilities for LLM response parsing."""

import json
import re


def parse_json_response(text: str):
    """Robustly parse a JSON array or object from an LLM response.

    Handles: markdown fences, leading/trailing text, multiple JSON blocks.
    """
    text = text.strip()

    # Strip markdown code fences
    if "```" in text:
        # Extract content between first pair of fences
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first [ or { and try to parse from there
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    raise ValueError(f"Could not parse JSON from response: {text[:200]}...")
