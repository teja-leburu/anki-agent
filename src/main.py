"""CLI entry point — supports baseline, pipeline, comparison, and evaluation modes."""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.parser import extract_text_from_pdf, chunk_pages
from src.generator import generate_flashcards
from src.pipeline import run_pipeline, print_stats
from src.exporter import export_to_apkg

load_dotenv()

GEN_MODEL = "claude-sonnet-4-20250514"
JUDGE_MODEL = "claude-opus-4-20250514"


def run_baseline(chunks: list[dict], model: str) -> list[dict]:
    """Phase 1 baseline: single-prompt generation per chunk."""
    all_cards = []
    for i, chunk in enumerate(chunks):
        print(f"  Generating cards for chunk {i + 1}/{len(chunks)} "
              f"(pages {chunk['source_pages']})...")
        try:
            cards = generate_flashcards(chunk["text"], model=model)
            print(f"    → {len(cards)} cards generated.")
            all_cards.extend(cards)
        except Exception as e:
            print(f"    ✗ Error on chunk {i + 1}: {e}", file=sys.stderr)
    return all_cards


def run(pdf_path: str, output_path: str, deck_name: str, gen_model: str,
        judge_model: str, mode: str, evaluate: bool = False, max_pages: int = 0):
    print(f"Parsing {pdf_path}...")
    print(f"  Generator model: {gen_model}")
    print(f"  Judge model:     {judge_model}")
    pages = extract_text_from_pdf(pdf_path)
    if max_pages:
        pages = pages[:max_pages]
        print(f"  Extracted {len(pages)} pages (limited to first {max_pages}).")
    else:
        print(f"  Extracted {len(pages)} pages with text.")

    chunks = chunk_pages(pages)
    print(f"  Split into {len(chunks)} chunks.")

    if mode == "baseline":
        print("\n--- Running Phase 1: Baseline (single prompt) ---\n")
        all_cards = run_baseline(chunks, gen_model)
    else:
        print("\n--- Running Phase 2: Multi-Prompt Pipeline ---\n")
        all_cards, stats = run_pipeline(
            chunks, gen_model, judge_model, str(Path(output_path).parent)
        )
        print_stats(stats)

    print(f"\nTotal cards: {len(all_cards)}")

    if not all_cards:
        print("No cards generated. Exiting.")
        return

    # Save raw JSON
    json_path = Path(output_path).with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(all_cards, f, indent=2)
    print(f"Raw cards saved to {json_path}")

    # Export .apkg
    export_to_apkg(all_cards, deck_name, output_path)
    print(f"Anki deck exported to {output_path}")

    # Optional evaluation
    if evaluate:
        print("\n--- Running Evaluation ---\n")
        import anthropic
        from src.evaluator import evaluate_cards
        client = anthropic.Anthropic()
        source_text = "\n\n".join(c["text"] for c in chunks)
        eval_result = evaluate_cards(all_cards, source_text, client, judge_model)

        eval_path = Path(output_path).with_suffix(".eval.json")
        with open(eval_path, "w") as f:
            json.dump(eval_result, f, indent=2)

        print(f"  Heuristic pass rate: {eval_result['heuristic_pass_rate']}%")
        print(f"  Avg judge scores:    {eval_result['avg_judge_scores']}")
        print(f"  Avg overall:         {eval_result['avg_overall']}")
        print(f"  Bloom's distribution: {eval_result['bloom_distribution']}")
        cov = eval_result.get("coverage", {})
        print(f"  Coverage:            {cov.get('coverage_pct', 'N/A')}%")
        print(f"  Evaluation saved to {eval_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Anki flashcards from a PDF.")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Default generate command
    gen = subparsers.add_parser("generate", help="Generate flashcards from a PDF")
    gen.add_argument("pdf", help="Path to the input PDF file")
    gen.add_argument("-o", "--output", default="data/outputs/deck.apkg",
                     help="Output .apkg file path")
    gen.add_argument("-n", "--name", default="AnkiAgent Deck",
                     help="Anki deck name")
    gen.add_argument("-m", "--model", default=GEN_MODEL,
                     help=f"Generator model (default: {GEN_MODEL})")
    gen.add_argument("-j", "--judge-model", default=JUDGE_MODEL,
                     help=f"Judge/evaluation model (default: {JUDGE_MODEL})")
    gen.add_argument("--mode", choices=["baseline", "pipeline"], default="pipeline",
                     help="Generation mode: baseline (Phase 1) or pipeline (Phase 2)")
    gen.add_argument("--evaluate", action="store_true",
                     help="Run evaluation after generation")
    gen.add_argument("--max-pages", type=int, default=0,
                     help="Limit to first N pages (0 = all pages)")

    # Compare command
    cmp = subparsers.add_parser("compare", help="Compare prompt strategies")
    cmp.add_argument("pdf", help="Path to the input PDF file")
    cmp.add_argument("-s", "--strategies", nargs="+",
                     default=["few_shot", "chain_of_thought", "minimal_few_shot"],
                     help="Strategies to compare")
    cmp.add_argument("-m", "--model", default=GEN_MODEL,
                     help=f"Generator model (default: {GEN_MODEL})")
    cmp.add_argument("-j", "--judge-model", default=JUDGE_MODEL,
                     help=f"Judge/evaluation model (default: {JUDGE_MODEL})")
    cmp.add_argument("-o", "--output-dir", default="data/outputs/comparison",
                     help="Output directory for comparison results")

    args = parser.parse_args()

    if args.command == "compare":
        from src.compare import run_strategy_comparison
        run_strategy_comparison(
            args.pdf, args.strategies, args.model, args.judge_model, args.output_dir
        )
    elif args.command == "generate":
        run(args.pdf, args.output, args.name, args.model, args.judge_model,
            args.mode, args.evaluate, args.max_pages)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
