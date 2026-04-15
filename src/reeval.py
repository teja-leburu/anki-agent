"""Re-evaluate failed experiment runs by batching large card sets."""

import json
import sys
from pathlib import Path
from dotenv import load_dotenv

from src.llm import create_client, infer_provider
from src.evaluator import (
    run_heuristics, llm_judge_score, classify_blooms, measure_coverage,
)

load_dotenv()

JUDGE_MODEL = "claude-opus-4-20250514"
BATCH_SIZE = 20  # evaluate this many cards at a time


def batched_judge_score(cards, client, model):
    """Score cards in batches to avoid overwhelming the LLM."""
    all_scores = []
    for i in range(0, len(cards), BATCH_SIZE):
        batch = cards[i:i + BATCH_SIZE]
        print(f"      Judging batch {i//BATCH_SIZE + 1} ({len(batch)} cards)...")
        try:
            scores = llm_judge_score(batch, client, model)
            # Re-index to global card indices
            for s in scores:
                s["card_index"] = s["card_index"] + i
            all_scores.extend(scores)
        except Exception as e:
            print(f"      ✗ Batch error: {e}", file=sys.stderr)
    return all_scores


def batched_blooms(cards, client, model):
    """Classify cards in batches."""
    all_classes = []
    for i in range(0, len(cards), BATCH_SIZE):
        batch = cards[i:i + BATCH_SIZE]
        print(f"      Bloom's batch {i//BATCH_SIZE + 1} ({len(batch)} cards)...")
        try:
            classes = classify_blooms(batch, client, model)
            for c in classes:
                c["card_index"] = c["card_index"] + i
            all_classes.extend(classes)
        except Exception as e:
            print(f"      ✗ Batch error: {e}", file=sys.stderr)
    return all_classes


def evaluate_cards_batched(cards, source_text, client, model):
    """Full evaluation with batching for large card sets."""
    heuristics = run_heuristics(cards)

    print("    Running batched judge scoring...")
    judge_scores = batched_judge_score(cards, client, model)

    print("    Running batched Bloom's classification...")
    bloom_classes = batched_blooms(cards, client, model)
    bloom_dist = {}
    for entry in bloom_classes:
        level = entry.get("bloom_level", "unknown")
        bloom_dist[level] = bloom_dist.get(level, 0) + 1

    print("    Measuring coverage...")
    try:
        # Truncate source text if very long to avoid token limits
        truncated = source_text[:8000] if len(source_text) > 8000 else source_text
        coverage = measure_coverage(truncated, cards, client, model)
    except Exception as e:
        print(f"    ✗ Coverage error: {e}", file=sys.stderr)
        coverage = {"coverage_pct": 0, "error": str(e)}

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


def reeval_failed():
    """Find and re-evaluate all failed experiment result files."""
    client = create_client(infer_provider(JUDGE_MODEL))

    # We need source text per dataset for coverage — reconstruct from the PDFs
    from src.parser import extract_text_from_pdf, chunk_pages
    source_texts = {}
    pdf_map = {
        "cns_path": "/Users/teja/Library/Messages/Attachments/89/09/0B537A63-0AD1-4B17-A0E8-BF586452B4B6/CNS Path.pdf",
        "growth_adaptations": "/Users/teja/Library/Messages/Attachments/0b/11/F8B34975-A8B6-4526-BECE-DB39C9AA9E5C/Chp 1 Growth Adaptations.pdf",
        "gi_path": "/Users/teja/Library/Messages/Attachments/b8/08/B3DD12B3-9175-440A-A3B9-D89A726EEFCE/GI Path.pdf",
    }
    for ds, pdf in pdf_map.items():
        pages = extract_text_from_pdf(pdf)
        chunks = chunk_pages(pages)
        source_texts[ds] = "\n\n".join(c["text"] for c in chunks)

    # Find failed files
    failed = []
    for f in sorted(Path("data/outputs/experiments").rglob("*.json")):
        if "summary" in f.name:
            continue
        try:
            d = json.load(open(f))
            ev = d.get("evaluation", {})
            if "error" in ev:
                # Figure out which dataset this belongs to
                parts = str(f.relative_to("data/outputs/experiments")).split("/")
                ds = parts[0]
                cards = d.get("cards", d.get("cards_list", []))
                card_count = d.get("cards_after_dedup", d.get("card_count", 0))
                if card_count > 0:
                    failed.append({"path": str(f), "ds": ds, "card_count": card_count})
        except:
            pass

    print(f"Found {len(failed)} files needing re-evaluation\n")

    for item in failed:
        path = item["path"]
        ds = item["ds"]
        print(f"  Re-evaluating: {Path(path).relative_to('data/outputs/experiments')}")

        d = json.load(open(path))

        # Extract cards from the result file
        cards = d.get("cards", [])
        if not cards:
            # For some result structures, cards aren't stored in the file
            # We need to look at card_count but can't re-evaluate without cards
            print(f"    ✗ No cards stored in file (card_count={item['card_count']}), skipping")
            continue

        print(f"    {len(cards)} cards, evaluating with batched Opus...")
        source = source_texts.get(ds, "")
        eval_result = evaluate_cards_batched(cards, source, client, JUDGE_MODEL)

        d["evaluation"] = eval_result
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        print(f"    ✓ Done — avg quality: {eval_result['avg_overall']}\n")


if __name__ == "__main__":
    reeval_failed()
