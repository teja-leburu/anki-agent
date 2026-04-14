"""Step 4 of the pipeline — deduplicate flashcards using embedding similarity."""

from sentence_transformers import SentenceTransformer
import numpy as np


def _card_text(card: dict) -> str:
    """Combine front and back into a single string for embedding."""
    return f"{card['front']} {card['back']}".strip()


def deduplicate_cards(
    cards: list[dict], similarity_threshold: float = 0.85, model_name: str = "all-MiniLM-L6-v2"
) -> list[dict]:
    """Remove near-duplicate cards based on embedding cosine similarity.

    Keeps the first occurrence when duplicates are found.
    """
    if len(cards) <= 1:
        return cards

    model = SentenceTransformer(model_name)
    texts = [_card_text(c) for c in cards]
    embeddings = model.encode(texts, normalize_embeddings=True)

    # Cosine similarity matrix (embeddings are already normalized)
    sim_matrix = np.dot(embeddings, embeddings.T)

    keep = []
    removed = set()
    for i in range(len(cards)):
        if i in removed:
            continue
        keep.append(i)
        # Mark all subsequent cards too similar to this one
        for j in range(i + 1, len(cards)):
            if j not in removed and sim_matrix[i][j] > similarity_threshold:
                removed.add(j)

    return [cards[i] for i in keep]
