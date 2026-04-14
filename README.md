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
# Phase 2 multi-prompt pipeline (default)
python -m src.main path/to/document.pdf -o output.apkg -n "My Deck"

# Phase 1 baseline for comparison
python -m src.main path/to/document.pdf --mode baseline
```

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
  exporter.py       # .apkg export via genanki
  main.py           # CLI entry point
data/
  sample_inputs/    # Test PDFs
  outputs/          # Generated decks and stats (gitignored)
tests/              # Unit tests
```
