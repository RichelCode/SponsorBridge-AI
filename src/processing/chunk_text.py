from pathlib import Path
import json

INPUT_DIR = Path("data/clean")
OUTPUT_DIR = Path("data/chunks")


def split_into_chunks(text: str, chunk_size: int = 500) -> list:
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def process_file(file_path: Path) -> None:
    print(f"\nProcessing file: {file_path.name}")

    data = json.loads(file_path.read_text(encoding="utf-8"))

    company = data.get("company")
    page_type = data.get("page_type")
    url = data.get("url")
    text = data.get("text", "")

    chunks = split_into_chunks(text)

    output_data = []

    for i, chunk in enumerate(chunks):
        output_data.append({
            "company": company,
            "page_type": page_type,
            "url": url,
            "chunk_id": i,
            "text": chunk
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / file_path.name
    output_file.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"Saved chunks: {output_file}")


def main():
    files = list(INPUT_DIR.glob("*.json"))

    print(f"Found {len(files)} files to process")

    for file in files:
        process_file(file)


if __name__ == "__main__":
    main()