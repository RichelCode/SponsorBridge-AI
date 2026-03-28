from pathlib import Path
import json
import re

INPUT_DIR = Path("data/clean")
OUTPUT_DIR = Path("data/chunks")


NOISE_PATTERNS = [
    r"Skip to main content",
    r"Privacy and security",
    r"Terms and conditions",
    r"Cookies",
    r"Accessibility",
    r"All rights reserved",
    r"Equal Opportunity Employer",
    r"Search JPMorganChase",
    r"Search",
    r"Join our team",
]


def clean_text_for_chunking(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()

    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_chunks(text: str, chunk_size: int = 120, overlap: int = 30) -> list:
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def process_file(file_path: Path) -> None:
    print(f"\nProcessing file: {file_path.name}")

    data = json.loads(file_path.read_text(encoding="utf-8"))

    company = data.get("company")
    page_type = data.get("page_type")
    url = data.get("url")
    text = data.get("text", "")

    cleaned_text = clean_text_for_chunking(text)
    chunks = split_into_chunks(cleaned_text)

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