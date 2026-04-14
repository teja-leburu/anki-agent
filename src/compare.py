"""Compare prompt strategies on the same source material.

Runs each strategy through the pipeline and evaluates the results,
producing a side-by-side comparison table.
"""

import json
import sys
from pathlib import Path

import anthropic

from src.parser import extract_text_from_pdf, chunk_pages
from src.extractor import extract_concepts
from src.card_generator import generate_cards_from_concepts
from src.strategies import STRATEGIES, strategy_chain_of_thought
from src.critic import critique_cards
from src.dedup import deduplicate_cards
from src.evaluator import evaluate_cards


def run_strategy_comparison(
    pdf_path: str,
    strategies: list[str],
    gen_model: str = "claude-haiku-4-5-20251001",
    judge_model: str = "claude-opus-4-20250514",
    output_dir: str = "data/outputs/comparison",
):
    """Run multiple strategies on the same PDF and compare results.

    Uses gen_model for extraction/generation, judge_model for critique/evaluation.
    """
    client = anthropic.Anthropic()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Parse and chunk once
    print(f"Parsing {pdf_path}...")
    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_pages(pages)
    print(f"  {len(pages)} pages → {len(chunks)} chunks\n")

    print(f"  Generator model: {gen_model}")
    print(f"  Judge model:     {judge_model}\n")

    # Extract concepts once (shared across strategies)
    print("Extracting concepts (shared across all strategies)...")
    all_concepts = []
    chunk_texts = []
    for i, chunk in enumerate(chunks):
        try:
            concepts = extract_concepts(chunk["text"], client, gen_model)
            all_concepts.append(concepts)
            chunk_texts.append(chunk["text"])
            print(f"  chunk {i + 1}: {len(concepts)} concepts")
        except Exception as e:
            print(f"  chunk {i + 1}: error — {e}", file=sys.stderr)
            all_concepts.append([])
            chunk_texts.append(chunk["text"])

    source_text = "\n\n".join(chunk_texts)
    results = {}

    for strategy_name in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy_name}")
        print(f"{'='*60}")

        strategy_cards = []
        for i, concepts in enumerate(all_concepts):
            if not concepts:
                continue
            print(f"  Generating cards for chunk {i + 1}...")
            try:
                if strategy_name == "few_shot":
                    # Default Phase 2 generator
                    cards = generate_cards_from_concepts(concepts, client, gen_model)
                else:
                    gen_fn = STRATEGIES[strategy_name]
                    cards = gen_fn(concepts, client, gen_model)
                print(f"    → {len(cards)} cards")
                strategy_cards.extend(cards)
            except Exception as e:
                print(f"    ✗ Error: {e}", file=sys.stderr)

        # Critique (judge model)
        print(f"  Critiquing {len(strategy_cards)} cards...")
        try:
            passed, reviews = critique_cards(strategy_cards, client, judge_model)
            print(f"    → {len(passed)} passed critique")
        except Exception as e:
            print(f"    ✗ Critique error: {e}", file=sys.stderr)
            passed = strategy_cards

        # Dedup
        final = deduplicate_cards(passed)
        print(f"    → {len(final)} after dedup")

        # Evaluate
        print(f"  Evaluating {len(final)} cards...")
        try:
            eval_result = evaluate_cards(final, source_text, client, judge_model)
        except Exception as e:
            print(f"    ✗ Evaluation error: {e}", file=sys.stderr)
            eval_result = {"error": str(e)}

        results[strategy_name] = {
            "cards": final,
            "evaluation": eval_result,
        }

        # Save per-strategy results
        strategy_path = out / f"{strategy_name}.json"
        with open(strategy_path, "w") as f:
            json.dump(results[strategy_name], f, indent=2)

    # Print comparison table
    print_comparison_table(results)

    # Save full comparison
    summary = {
        name: {
            "card_count": r["evaluation"].get("card_count", 0),
            "heuristic_pass_rate": r["evaluation"].get("heuristic_pass_rate", 0),
            "avg_judge_scores": r["evaluation"].get("avg_judge_scores", {}),
            "avg_overall": r["evaluation"].get("avg_overall", 0),
            "bloom_distribution": r["evaluation"].get("bloom_distribution", {}),
            "coverage_pct": r["evaluation"].get("coverage", {}).get("coverage_pct", 0),
        }
        for name, r in results.items()
        if "error" not in r.get("evaluation", {})
    }
    with open(out / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull results saved to {out}/")

    return results


def print_comparison_table(results: dict):
    """Print a formatted comparison table."""
    print(f"\n{'='*80}")
    print("STRATEGY COMPARISON")
    print(f"{'='*80}")

    header = f"{'Strategy':<20} {'Cards':>6} {'Heur%':>6} {'Truth':>6} {'Atom':>6} {'Clear':>6} {'Avg':>6} {'Cov%':>6}"
    print(header)
    print("-" * len(header))

    for name, r in results.items():
        ev = r.get("evaluation", {})
        if "error" in ev:
            print(f"{name:<20} {'ERROR':>6}")
            continue
        scores = ev.get("avg_judge_scores", {})
        print(
            f"{name:<20} "
            f"{ev.get('card_count', 0):>6} "
            f"{ev.get('heuristic_pass_rate', 0):>5.1f}% "
            f"{scores.get('truthfulness', 0):>5.1f} "
            f"{scores.get('atomicity', 0):>5.1f} "
            f"{scores.get('clarity', 0):>5.1f} "
            f"{ev.get('avg_overall', 0):>5.1f} "
            f"{ev.get('coverage', {}).get('coverage_pct', 0):>5.1f}%"
        )
    print()
