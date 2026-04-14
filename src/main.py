"""CLI entry point — supports both baseline (Phase 1) and pipeline (Phase 2) modes."""

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


def run(pdf_path: str, output_path: str, deck_name: str, model: str, mode: str):
    print(f"Parsing {pdf_path}...")
    pages = extract_text_from_pdf(pdf_path)
    print(f"  Extracted {len(pages)} pages with text.")

    chunks = chunk_pages(pages)
    print(f"  Split into {len(chunks)} chunks.")

    if mode == "baseline":
        print("\n--- Running Phase 1: Baseline (single prompt) ---\n")
        all_cards = run_baseline(chunks, model)
    else:
        print("\n--- Running Phase 2: Multi-Prompt Pipeline ---\n")
        all_cards, stats = run_pipeline(chunks, model, str(Path(output_path).parent))
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


def main():
    parser = argparse.ArgumentParser(description="Generate Anki flashcards from a PDF.")
    parser.add_argument("pdf", help="Path to the input PDF file")
    parser.add_argument("-o", "--output", default="data/outputs/deck.apkg",
                        help="Output .apkg file path (default: data/outputs/deck.apkg)")
    parser.add_argument("-n", "--name", default="AnkiAgent Deck",
                        help="Anki deck name (default: AnkiAgent Deck)")
    parser.add_argument("-m", "--model", default="claude-sonnet-4-20250514",
                        help="Claude model to use")
    parser.add_argument("--mode", choices=["baseline", "pipeline"], default="pipeline",
                        help="Generation mode: baseline (Phase 1) or pipeline (Phase 2, default)")
    args = parser.parse_args()

    run(args.pdf, args.output, args.name, args.model, args.mode)


if __name__ == "__main__":
    main()
