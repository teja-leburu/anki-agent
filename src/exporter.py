"""Export generated flashcards to Anki .apkg format using genanki."""

import random
import genanki


# Stable IDs so re-exports update the same model/deck rather than duplicating
BASIC_MODEL_ID = 1607392319
CLOZE_MODEL_ID = 1607392320

BASIC_MODEL = genanki.Model(
    BASIC_MODEL_ID,
    "AnkiAgent Basic",
    fields=[
        {"name": "Front"},
        {"name": "Back"},
    ],
    templates=[
        {
            "name": "Card 1",
            "qfmt": "{{Front}}",
            "afmt": '{{FrontSide}}<hr id="answer">{{Back}}',
        },
    ],
)

CLOZE_MODEL = genanki.Model(
    CLOZE_MODEL_ID,
    "AnkiAgent Cloze",
    fields=[
        {"name": "Text"},
        {"name": "Extra"},
    ],
    templates=[
        {
            "name": "Cloze",
            "qfmt": "{{cloze:Text}}",
            "afmt": "{{cloze:Text}}<br>{{Extra}}",
        },
    ],
    model_type=genanki.Model.CLOZE,
)


def export_to_apkg(cards: list[dict], deck_name: str, output_path: str) -> str:
    """Export a list of flashcard dicts to an .apkg file.

    Each card dict must have: type, front, back, tags.
    Returns the output file path.
    """
    deck_id = random.randrange(1 << 30, 1 << 31)
    deck = genanki.Deck(deck_id, deck_name)

    for card in cards:
        tags = card.get("tags", [])
        if card["type"] == "cloze":
            note = genanki.Note(
                model=CLOZE_MODEL,
                fields=[card["front"], ""],
                tags=tags,
            )
        else:
            note = genanki.Note(
                model=BASIC_MODEL,
                fields=[card["front"], card["back"]],
                tags=tags,
            )
        deck.add_note(note)

    package = genanki.Package(deck)
    package.write_to_file(output_path)
    return output_path
