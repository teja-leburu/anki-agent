# anki-agent

AI-powered Anki flashcard generation from source documents using Claude and prompt engineering techniques.

**Carnegie Mellon 17-630 Prompt Engineering — Final Project**

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```bash
# Generate flashcards (Phase 2 pipeline, default)
python -m src.main generate path/to/document.pdf -o output.apkg -n "My Deck"

# Generate with Phase 1 baseline
python -m src.main generate path/to/document.pdf --mode baseline

# Generate with evaluation metrics
python -m src.main generate path/to/document.pdf --evaluate

# Compare prompt strategies side-by-side
python -m src.main compare path/to/document.pdf -s few_shot chain_of_thought minimal_few_shot
```

### Available Strategies

| Strategy | Description |
|---|---|
| `few_shot` | Default Phase 2 with rich few-shot examples (3 concept types) |
| `chain_of_thought` | CoT reasoning before card generation (Wei et al., 2022) |
| `minimal_few_shot` | Single-example few-shot (tests prompt sensitivity) |
| `source_textbook` | Optimized for dense textbook material |
| `source_lecture` | Optimized for terse lecture slides |
| `source_paper` | Optimized for research papers |

## Architecture

**Phase 1 (Baseline):** Single-prompt generation

```
PDF → Parser → Chunks → Claude (single prompt) → JSON → genanki → .apkg
```

**Phase 2 (Pipeline):** Multi-prompt pipeline with quality control

```
PDF → Parser → Chunks
                 ↓
         1. Concept Extraction  (identify key facts, definitions, relationships)
                 ↓
         2. Card Generation     (few-shot prompted, Wozniak's 20 Rules)
                 ↓
         3. Quality Critique    (LLM-as-judge: truthfulness, atomicity, clarity)
                 ↓
         4. Deduplication       (embedding similarity across chunks)
                 ↓
              genanki → .apkg
```

## Evaluation Framework

The `--evaluate` flag and `compare` command run a multi-dimensional evaluation:

- **Heuristic checks** — card length, cloze syntax, single-question detection
- **LLM-as-judge** — scores on truthfulness, atomicity, self-containment, clarity, relevance (1-5)
- **Bloom's Taxonomy** — classifies cards by cognitive level (remember/understand/apply/analyze)
- **Source coverage** — measures what % of key concepts from the source are represented

## Project Structure

```
src/
  parser.py         # PDF text extraction and chunking
  generator.py      # Phase 1 baseline single-prompt generator
  extractor.py      # Phase 2 Step 1: concept extraction
  card_generator.py # Phase 2 Step 2: few-shot card generation
  critic.py         # Phase 2 Step 3: LLM-as-judge quality critique
  dedup.py          # Phase 2 Step 4: embedding deduplication
  pipeline.py       # Phase 2 orchestrator
  strategies.py     # Phase 3: prompt strategy variations (CoT, source-specific, etc.)
  evaluator.py      # Phase 3: evaluation framework (heuristics, judge, Bloom's, coverage)
  compare.py        # Phase 3: strategy comparison runner
  exporter.py       # .apkg export via genanki
  main.py           # CLI entry point
data/
  sample_inputs/    # Test PDFs
  outputs/          # Generated decks, stats, and evaluations (gitignored)
tests/              # Unit tests
```
