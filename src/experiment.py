"""Experiment runner for the three core experiments.

Experiment 1: Coverage-Quality Pareto Frontier (sweep critique thresholds)
Experiment 2: Technique Ablation (isolate impact of each pipeline stage)
Experiment 3: Claude vs GPT (same prompts, different models)

All experiments use Opus as the judge for consistent evaluation.
"""

import json
import sys
import time
from pathlib import Path

from src.llm import create_client, infer_provider
from src.parser import extract_text_from_pdf, chunk_pages
from src.extractor import extract_concepts
from src.card_generator import generate_cards_from_concepts
from src.generator import generate_flashcards
from src.strategies import STRATEGIES
from src.critic import critique_cards
from src.dedup import deduplicate_cards
from src.evaluator import evaluate_cards


def _extract_all_concepts(chunks, client, model):
    """Extract concepts from all chunks, returning parallel lists."""
    all_concepts = []
    for i, chunk in enumerate(chunks):
        try:
            concepts = extract_concepts(chunk["text"], client, model)
            all_concepts.append(concepts)
            print(f"    chunk {i + 1}: {len(concepts)} concepts")
        except Exception as e:
            print(f"    chunk {i + 1}: error — {e}", file=sys.stderr)
            all_concepts.append([])
    return all_concepts


def _generate_all_cards(chunks, all_concepts, client, model, strategy_name="few_shot"):
    """Generate cards from concepts using a given strategy."""
    all_cards = []
    gen_fn = STRATEGIES.get(strategy_name)
    for i, concepts in enumerate(all_concepts):
        if not concepts:
            continue
        try:
            if strategy_name == "few_shot" or gen_fn is None:
                cards = generate_cards_from_concepts(concepts, client, model)
            else:
                cards = gen_fn(concepts, client, model)
            all_cards.extend(cards)
        except Exception as e:
            print(f"    chunk {i + 1} generation error: {e}", file=sys.stderr)
    return all_cards


def _evaluate_with_judge(cards, source_text, judge_client, judge_model):
    """Run full evaluation using the judge model."""
    if not cards:
        return {"card_count": 0, "error": "no cards"}
    try:
        return evaluate_cards(cards, source_text, judge_client, judge_model)
    except Exception as e:
        return {"card_count": len(cards), "error": str(e)}


# ---------------------------------------------------------------------------
# Experiment 1: Coverage-Quality Pareto Frontier
# ---------------------------------------------------------------------------

def experiment_pareto(
    chunks: list[dict],
    source_text: str,
    gen_model: str = "claude-sonnet-4-20250514",
    judge_model: str = "claude-opus-4-20250514",
    thresholds: list[dict] = None,
    output_dir: str = "data/outputs/experiments/pareto",
):
    """Sweep critique thresholds to map the coverage-quality trade-off."""
    if thresholds is None:
        thresholds = [
            {"min_score": 1, "min_avg": 1.0, "label": "no_filter"},
            {"min_score": 2, "min_avg": 2.0, "label": "lenient"},
            {"min_score": 2, "min_avg": 3.0, "label": "moderate"},
            {"min_score": 3, "min_avg": 3.5, "label": "strict"},
            {"min_score": 4, "min_avg": 4.0, "label": "very_strict"},
        ]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gen_client = create_client(infer_provider(gen_model))
    judge_client = create_client(infer_provider(judge_model))

    # Generate cards once (shared across all threshold levels)
    print("  Extracting concepts...")
    all_concepts = _extract_all_concepts(chunks, gen_client, gen_model)

    print("  Generating cards (shared across thresholds)...")
    raw_cards = _generate_all_cards(chunks, all_concepts, gen_client, gen_model)
    print(f"    → {len(raw_cards)} total cards generated\n")

    results = {}
    for t in thresholds:
        label = t["label"]
        print(f"  Threshold: {label} (min_score={t['min_score']}, min_avg={t['min_avg']})")

        if label == "no_filter":
            passed = raw_cards
            print(f"    → {len(passed)} cards (no filtering)")
        else:
            try:
                passed, reviews = critique_cards(
                    raw_cards, judge_client, judge_model,
                    min_score=t["min_score"], min_avg=t["min_avg"],
                )
                print(f"    → {len(passed)}/{len(raw_cards)} cards passed")
            except Exception as e:
                print(f"    ✗ Critique error: {e}", file=sys.stderr)
                passed = raw_cards

        final = deduplicate_cards(passed)
        print(f"    → {len(final)} after dedup")

        print(f"    Evaluating...")
        eval_result = _evaluate_with_judge(final, source_text, judge_client, judge_model)

        results[label] = {
            "threshold": t,
            "cards_before_filter": len(raw_cards),
            "cards_after_filter": len(passed),
            "cards_after_dedup": len(final),
            "evaluation": eval_result,
        }

        with open(out / f"{label}.json", "w") as f:
            json.dump(results[label], f, indent=2)
        print()

    _save_and_print_pareto(results, out)
    return results


def _save_and_print_pareto(results, out):
    """Print and save the Pareto frontier summary."""
    print(f"\n{'='*75}")
    print("PARETO FRONTIER: Coverage vs Quality")
    print(f"{'='*75}")
    header = f"{'Threshold':<15} {'Cards':>6} {'Avg Q':>6} {'Cov%':>6} {'Atom':>6} {'Truth':>6}"
    print(header)
    print("-" * len(header))

    summary = {}
    for label, r in results.items():
        ev = r.get("evaluation", {})
        scores = ev.get("avg_judge_scores", {})
        cov = ev.get("coverage", {}).get("coverage_pct", 0)
        summary[label] = {
            "cards": r["cards_after_dedup"],
            "avg_quality": ev.get("avg_overall", 0),
            "coverage_pct": cov,
            "atomicity": scores.get("atomicity", 0),
            "truthfulness": scores.get("truthfulness", 0),
        }
        print(
            f"{label:<15} "
            f"{r['cards_after_dedup']:>6} "
            f"{ev.get('avg_overall', 0):>5.2f} "
            f"{cov:>5.1f}% "
            f"{scores.get('atomicity', 0):>5.2f} "
            f"{scores.get('truthfulness', 0):>5.2f}"
        )

    with open(out / "pareto_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out}/")


# ---------------------------------------------------------------------------
# Experiment 2: Technique Ablation
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = [
    {"name": "baseline", "desc": "Single prompt, no pipeline"},
    {"name": "pipeline_no_critique", "desc": "Extract → Generate → Dedup (no critique)"},
    {"name": "pipeline_full", "desc": "Extract → Generate → Critique → Dedup"},
    {"name": "pipeline_cot", "desc": "Full pipeline + chain-of-thought generation"},
    {"name": "pipeline_few_shot_3", "desc": "Full pipeline + 3-example few-shot (default)"},
    {"name": "pipeline_few_shot_1", "desc": "Full pipeline + 1-example few-shot"},
]


def experiment_ablation(
    chunks: list[dict],
    source_text: str,
    gen_model: str = "claude-sonnet-4-20250514",
    judge_model: str = "claude-opus-4-20250514",
    output_dir: str = "data/outputs/experiments/ablation",
):
    """Run technique ablation — test each pipeline component in isolation."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gen_client = create_client(infer_provider(gen_model))
    judge_client = create_client(infer_provider(judge_model))

    # Pre-extract concepts (shared for pipeline configs)
    print("  Extracting concepts (shared)...")
    all_concepts = _extract_all_concepts(chunks, gen_client, gen_model)

    results = {}

    for config in ABLATION_CONFIGS:
        name = config["name"]
        print(f"\n  {'='*50}")
        print(f"  Config: {name} — {config['desc']}")
        print(f"  {'='*50}")

        try:
            if name == "baseline":
                # Single prompt, no pipeline
                cards = []
                for chunk in chunks:
                    try:
                        c = generate_flashcards(chunk["text"], gen_client, gen_model)
                        cards.extend(c)
                    except Exception as e:
                        print(f"    baseline error: {e}", file=sys.stderr)
                final = deduplicate_cards(cards)

            elif name == "pipeline_no_critique":
                cards = _generate_all_cards(chunks, all_concepts, gen_client, gen_model)
                final = deduplicate_cards(cards)

            elif name == "pipeline_full" or name == "pipeline_few_shot_3":
                cards = _generate_all_cards(chunks, all_concepts, gen_client, gen_model)
                passed, _ = critique_cards(cards, judge_client, judge_model)
                final = deduplicate_cards(passed)

            elif name == "pipeline_cot":
                cards = _generate_all_cards(
                    chunks, all_concepts, gen_client, gen_model,
                    strategy_name="chain_of_thought"
                )
                passed, _ = critique_cards(cards, judge_client, judge_model)
                final = deduplicate_cards(passed)

            elif name == "pipeline_few_shot_1":
                cards = _generate_all_cards(
                    chunks, all_concepts, gen_client, gen_model,
                    strategy_name="minimal_few_shot"
                )
                passed, _ = critique_cards(cards, judge_client, judge_model)
                final = deduplicate_cards(passed)

            else:
                continue

            print(f"    → {len(final)} cards after processing")
            print(f"    Evaluating...")
            eval_result = _evaluate_with_judge(final, source_text, judge_client, judge_model)

        except Exception as e:
            print(f"    ✗ Config error: {e}", file=sys.stderr)
            final = []
            eval_result = {"error": str(e)}

        results[name] = {
            "config": config,
            "card_count": len(final),
            "evaluation": eval_result,
        }

        with open(out / f"{name}.json", "w") as f:
            json.dump(results[name], f, indent=2)

    _save_and_print_ablation(results, out)
    return results


def _save_and_print_ablation(results, out):
    """Print and save ablation results."""
    print(f"\n{'='*85}")
    print("TECHNIQUE ABLATION")
    print(f"{'='*85}")
    header = f"{'Config':<25} {'Cards':>6} {'Avg Q':>6} {'Atom':>6} {'Clear':>6} {'Cov%':>6}"
    print(header)
    print("-" * len(header))

    summary = {}
    for name, r in results.items():
        ev = r.get("evaluation", {})
        if "error" in ev:
            print(f"{name:<25} {'ERROR':>6}")
            continue
        scores = ev.get("avg_judge_scores", {})
        cov = ev.get("coverage", {}).get("coverage_pct", 0)
        summary[name] = {
            "cards": r["card_count"],
            "avg_quality": ev.get("avg_overall", 0),
            "coverage_pct": cov,
            "scores": scores,
        }
        print(
            f"{name:<25} "
            f"{r['card_count']:>6} "
            f"{ev.get('avg_overall', 0):>5.2f} "
            f"{scores.get('atomicity', 0):>5.2f} "
            f"{scores.get('clarity', 0):>5.2f} "
            f"{cov:>5.1f}%"
        )

    with open(out / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out}/")


# ---------------------------------------------------------------------------
# Experiment 3: Claude vs GPT
# ---------------------------------------------------------------------------

def experiment_model_comparison(
    chunks: list[dict],
    source_text: str,
    models: list[dict] = None,
    judge_model: str = "claude-opus-4-20250514",
    output_dir: str = "data/outputs/experiments/models",
):
    """Compare different generator models using the same prompts, judged by Opus."""
    if models is None:
        models = [
            {"name": "claude-sonnet", "model": "claude-sonnet-4-20250514"},
            {"name": "gpt-4o", "model": "gpt-4o"},
        ]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    judge_client = create_client(infer_provider(judge_model))
    results = {}

    for m in models:
        name = m["name"]
        model = m["model"]
        print(f"\n  {'='*50}")
        print(f"  Model: {name} ({model})")
        print(f"  {'='*50}")

        gen_client = create_client(infer_provider(model))

        # Extract concepts with this model
        print("    Extracting concepts...")
        all_concepts = _extract_all_concepts(chunks, gen_client, model)

        # Generate cards with this model
        print("    Generating cards...")
        cards = _generate_all_cards(chunks, all_concepts, gen_client, model)
        print(f"    → {len(cards)} cards generated")

        # Critique with the JUDGE model (Opus) — neutral arbiter
        print("    Critiquing with judge model...")
        try:
            passed, reviews = critique_cards(cards, judge_client, judge_model)
            print(f"    → {len(passed)}/{len(cards)} passed critique")
        except Exception as e:
            print(f"    ✗ Critique error: {e}", file=sys.stderr)
            passed = cards

        final = deduplicate_cards(passed)
        print(f"    → {len(final)} after dedup")

        # Evaluate with judge model
        print("    Evaluating...")
        eval_result = _evaluate_with_judge(final, source_text, judge_client, judge_model)

        results[name] = {
            "model": model,
            "cards_generated": len(cards),
            "cards_after_critique": len(passed),
            "cards_after_dedup": len(final),
            "evaluation": eval_result,
        }

        with open(out / f"{name}.json", "w") as f:
            json.dump(results[name], f, indent=2)

    _save_and_print_models(results, out)
    return results


def _save_and_print_models(results, out):
    """Print and save model comparison results."""
    print(f"\n{'='*85}")
    print("MODEL COMPARISON (same prompts, Opus judge)")
    print(f"{'='*85}")
    header = f"{'Model':<20} {'Gen':>5} {'Pass':>5} {'Final':>5} {'Avg Q':>6} {'Atom':>6} {'Truth':>6} {'Cov%':>6}"
    print(header)
    print("-" * len(header))

    summary = {}
    for name, r in results.items():
        ev = r.get("evaluation", {})
        if "error" in ev:
            print(f"{name:<20} {'ERROR':>5}")
            continue
        scores = ev.get("avg_judge_scores", {})
        cov = ev.get("coverage", {}).get("coverage_pct", 0)
        summary[name] = {
            "cards_generated": r["cards_generated"],
            "pass_rate": round(r["cards_after_critique"] / r["cards_generated"] * 100, 1) if r["cards_generated"] else 0,
            "final_cards": r["cards_after_dedup"],
            "avg_quality": ev.get("avg_overall", 0),
            "coverage_pct": cov,
            "scores": scores,
        }
        print(
            f"{name:<20} "
            f"{r['cards_generated']:>5} "
            f"{r['cards_after_critique']:>5} "
            f"{r['cards_after_dedup']:>5} "
            f"{ev.get('avg_overall', 0):>5.2f} "
            f"{scores.get('atomicity', 0):>5.2f} "
            f"{scores.get('truthfulness', 0):>5.2f} "
            f"{cov:>5.1f}%"
        )

    with open(out / "model_comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out}/")


# ---------------------------------------------------------------------------
# Run all experiments
# ---------------------------------------------------------------------------

def run_all_experiments(
    pdf_path: str,
    gen_model: str = "claude-sonnet-4-20250514",
    judge_model: str = "claude-opus-4-20250514",
    max_pages: int = 0,
    output_dir: str = "data/outputs/experiments",
):
    """Run all three experiments on a given PDF."""
    print(f"Parsing {pdf_path}...")
    pages = extract_text_from_pdf(pdf_path)
    if max_pages:
        pages = pages[:max_pages]
    chunks = chunk_pages(pages)
    source_text = "\n\n".join(c["text"] for c in chunks)
    print(f"  {len(pages)} pages → {len(chunks)} chunks\n")

    print("=" * 60)
    print("EXPERIMENT 1: Coverage-Quality Pareto Frontier")
    print("=" * 60)
    pareto = experiment_pareto(chunks, source_text, gen_model, judge_model,
                               output_dir=f"{output_dir}/pareto")

    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Technique Ablation")
    print("=" * 60)
    ablation = experiment_ablation(chunks, source_text, gen_model, judge_model,
                                   output_dir=f"{output_dir}/ablation")

    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Claude vs GPT")
    print("=" * 60)
    models = experiment_model_comparison(chunks, source_text,
                                         judge_model=judge_model,
                                         output_dir=f"{output_dir}/models")

    return {"pareto": pareto, "ablation": ablation, "models": models}
