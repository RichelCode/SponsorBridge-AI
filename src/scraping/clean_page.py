from pathlib import Path
import json

from bs4 import BeautifulSoup

RAW_FILE = Path("data/raw/example_com.html")
CLEAN_FILE = Path("data/clean/example_com.json")


def load_html(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def clean_html(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    # remove script and style content
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else "No title"

    text = soup.get_text(separator=" ", strip=True)
    text = " ".join(text.split())

    return {
        "title": title,
        "text": text
    }


def save_json(data: dict, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def main() -> None:
    print("Loading raw HTML...")
    html = load_html(RAW_FILE)

    print("Cleaning HTML...")
    cleaned = clean_html(html)

    save_json(cleaned, CLEAN_FILE)

    print(f"Saved cleaned data to: {CLEAN_FILE}")


if __name__ == "__main__":
    main()