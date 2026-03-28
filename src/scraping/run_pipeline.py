from pathlib import Path
import json

import requests
from bs4 import BeautifulSoup
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


URLS = [
    "https://example.com/",
    "https://www.iana.org/domains/example"
]


def fetch_html(url: str) -> str:
    response = requests.get(url, timeout=20, verify=False)
    response.raise_for_status()
    return response.text


def clean_html(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else "No title"

    text = soup.get_text(separator=" ", strip=True)
    text = " ".join(text.split())

    return {
        "title": title,
        "text": text
    }


def save_json(data: dict, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def process_url(url: str) -> None:
    print(f"\nProcessing: {url}")

    html = fetch_html(url)
    cleaned = clean_html(html)

    filename = url.replace("https://", "").replace("/", "_") + ".json"
    output_path = Path("data/clean") / filename

    save_json(cleaned, output_path)

    print(f"Saved: {output_path}")


def main() -> None:
    for url in URLS:
        process_url(url)


if __name__ == "__main__":
    main()