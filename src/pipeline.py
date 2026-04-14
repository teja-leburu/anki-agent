"""Multi-prompt pipeline orchestrator.

Pipeline: Chunks → Extract Concepts → Generate Cards → Critique → Deduplicate
"""

import json
import sys
from pathlib import Path

import anthropic

from src.extractor import extract_concepts
from src.card_generator import generate_cards_from_concepts
from src.critic import critique_cards
from src.dedup import deduplicate_cards


def run_pipeline(
    chunks: list[dict],
    gen_model: str,
    judge_model: str,
    output_dir: str = "data/outputs",
) -> tuple[list[dict], dict]:
    """Run the full multi-prompt pipeline on a list of text chunks.

    Uses gen_model for extraction and card generation, judge_model for critique.
    Returns (final_cards, stats) where stats tracks counts at each stage.
    """
    client = anthropic.Anthropic()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {
        "chunks": len(chunks),
        "gen_model": gen_model,
        "judge_model": judge_model,
        "concepts_extracted": 0,
        "cards_generated": 0,
        "cards_passed_critique": 0,
        "cards_after_dedup": 0,
        "reviews": [],
    }

    all_cards = []

    for i, chunk in enumerate(chunks):
        label = f"chunk {i + 1}/{len(chunks)} (pages {chunk['source_pages']})"

        # Step 1: Extract concepts (gen_model)
        print(f"  [{label}] Extracting concepts...")
        try:
            concepts = extract_concepts(chunk["text"], client, gen_model)
            print(f"    → {len(concepts)} concepts extracted.")
            stats["concepts_extracted"] += len(concepts)
        except Exception as e:
            print(f"    ✗ Extraction error: {e}", file=sys.stderr)
            continue

        # Step 2: Generate cards from concepts (gen_model)
        print(f"  [{label}] Generating cards from concepts...")
        try:
            cards = generate_cards_from_concepts(concepts, client, gen_model)
            print(f"    → {len(cards)} cards generated.")
            stats["cards_generated"] += len(cards)
        except Exception as e:
            print(f"    ✗ Generation error: {e}", file=sys.stderr)
            continue

        # Step 3: Critique cards (judge_model)
        print(f"  [{label}] Running quality critique...")
        try:
            passed, reviews = critique_cards(cards, client, judge_model)
            failed_count = len(cards) - len(passed)
            print(f"    → {len(passed)} passed, {failed_count} filtered out.")
            stats["cards_passed_critique"] += len(passed)
            stats["reviews"].extend(reviews)
            all_cards.extend(passed)
        except Exception as e:
            print(f"    ✗ Critique error: {e}", file=sys.stderr)
            # On critique failure, keep all cards rather than losing them
            all_cards.extend(cards)
            stats["cards_passed_critique"] += len(cards)

    # Step 4: Deduplicate across all chunks
    print(f"\n  Deduplicating {len(all_cards)} cards across chunks...")
    final_cards = deduplicate_cards(all_cards)
    dupes_removed = len(all_cards) - len(final_cards)
    print(f"    → {dupes_removed} duplicates removed, {len(final_cards)} cards remaining.")
    stats["cards_after_dedup"] = len(final_cards)

    # Save pipeline stats
    stats_path = output_path / "pipeline_stats.json"
    serializable_stats = {k: v for k, v in stats.items()}
    with open(stats_path, "w") as f:
        json.dump(serializable_stats, f, indent=2)
    print(f"\n  Pipeline stats saved to {stats_path}")

    return final_cards, stats


def print_stats(stats: dict):
    """Print a summary of pipeline statistics."""
    print("\n=== Pipeline Summary ===")
    print(f"  Chunks processed:      {stats['chunks']}")
    print(f"  Concepts extracted:    {stats['concepts_extracted']}")
    print(f"  Cards generated:       {stats['cards_generated']}")
    print(f"  Cards passed critique: {stats['cards_passed_critique']}")
    print(f"  Cards after dedup:     {stats['cards_after_dedup']}")

    if stats["cards_generated"] > 0:
        pass_rate = stats["cards_passed_critique"] / stats["cards_generated"] * 100
        print(f"  Critique pass rate:    {pass_rate:.1f}%")

    if stats["cards_passed_critique"] > 0:
        dedup_rate = (
            (stats["cards_passed_critique"] - stats["cards_after_dedup"])
            / stats["cards_passed_critique"]
            * 100
        )
        print(f"  Dedup removal rate:    {dedup_rate:.1f}%")
