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
python -m src.main path/to/document.pdf -o output.apkg -n "My Deck"
```

## Architecture

**Phase 1 (Baseline):** Single-prompt pipeline

```
PDF → Parser → Chunks → Claude (single prompt) → JSON → genanki → .apkg
```

## Project Structure

```
src/
  parser.py      # PDF text extraction and chunking
  generator.py   # LLM prompt and flashcard generation
  exporter.py    # .apkg export via genanki
  main.py        # CLI entry point
data/
  sample_inputs/ # Test PDFs
  outputs/       # Generated decks (gitignored)
tests/           # Unit tests
```
