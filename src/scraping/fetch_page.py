from pathlib import Path
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

URL = "https://example.com/"
OUTPUT_FILE = Path("data/raw/example_com.html")


def fetch_html(url: str) -> str:
    response = requests.get(url, timeout=20, verify=False)
    response.raise_for_status()
    return response.text


def save_html(html: str, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html, encoding="utf-8")


def main() -> None:
    print(f"Fetching page from: {URL}")
    html = fetch_html(URL)
    save_html(html, OUTPUT_FILE)
    print(f"Saved raw HTML to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()