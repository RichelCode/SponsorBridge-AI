from pathlib import Path
import json

from sentence_transformers import SentenceTransformer

INPUT_DIR = Path("data/chunks")
OUTPUT_DIR = Path("data/embeddings")

MODEL_NAME = "all-MiniLM-L6-v2"


def load_model() -> SentenceTransformer:
    print(f"Loading embedding model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


def process_file(file_path: Path, model: SentenceTransformer) -> None:
    print(f"\nProcessing file: {file_path.name}")

    data = json.loads(file_path.read_text(encoding="utf-8"))

    texts = [item["text"] for item in data]
    embeddings = model.encode(texts, show_progress_bar=False)

    output_data = []

    for item, embedding in zip(data, embeddings):
        output_data.append({
            "company": item.get("company"),
            "page_type": item.get("page_type"),
            "url": item.get("url"),
            "chunk_id": item.get("chunk_id"),
            "text": item.get("text"),
            "embedding": embedding.tolist()
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / file_path.name
    output_file.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"Saved embeddings: {output_file}")


def main() -> None:
    files = list(INPUT_DIR.glob("*.json"))
    print(f"Found {len(files)} chunk files")

    model = load_model()

    for file in files:
        process_file(file, model)


if __name__ == "__main__":
    main()