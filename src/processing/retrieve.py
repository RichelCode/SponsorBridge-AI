from pathlib import Path
import json
import numpy as np

from sentence_transformers import SentenceTransformer

EMBEDDINGS_DIR = Path("data/embeddings")
MODEL_NAME = "all-MiniLM-L6-v2"


def load_all_embeddings():
    all_items = []

    files = list(EMBEDDINGS_DIR.glob("*.json"))
    print(f"Loading {len(files)} embedding files")

    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        all_items.extend(data)

    print(f"Total chunks loaded: {len(all_items)}")
    return all_items


def embed_query(query: str, model):
    return model.encode(query)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search(query: str, items, model, top_k=5):
    print(f"\nQuery: {query}")

    query_embedding = embed_query(query, model)

    scored_results = []

    for item in items:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored_results.append((score, item))

    scored_results.sort(key=lambda x: x[0], reverse=True)

    return scored_results[:top_k]


def main():
    model = SentenceTransformer(MODEL_NAME)

    items = load_all_embeddings()

    query = input("\nEnter your question: ")

    results = search(query, items, model)

    print("\nTop Results:\n")

    for score, item in results:
        print(f"Score: {score:.4f}")
        print(f"Company: {item['company']}")
        print(f"Page: {item['page_type']}")
        print(f"Text: {item['text'][:200]}...")
        print("-" * 50)


if __name__ == "__main__":
    main()