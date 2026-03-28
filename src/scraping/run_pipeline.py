from pathlib import Path
import json
import csv

import requests
from bs4 import BeautifulSoup
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CSV_FILE = Path("data/company_targets.csv")


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


def make_safe_filename(company: str, page_type: str) -> str:
    safe_company = company.lower().replace(" ", "_").replace("&", "and")
    safe_page_type = page_type.lower().replace(" ", "_")
    return f"{safe_company}_{safe_page_type}.json"


def load_targets_from_csv() -> list:
    targets = []

    print(f"Reading CSV from: {CSV_FILE}")

    with CSV_FILE.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        print("CSV columns:", reader.fieldnames)

        for row in reader:
            status = (row.get("status") or "").strip()
            company = (row.get("company") or "").strip()

            if status == "to_scrape":
                about = (row.get("about_url") or "").strip()
                careers = (row.get("careers_url") or "").strip()
                students = (row.get("students_url") or "").strip()

                if about:
                    targets.append({
                        "company": company,
                        "page_type": "about",
                        "url": about
                    })

                if careers:
                    targets.append({
                        "company": company,
                        "page_type": "careers",
                        "url": careers
                    })

                if students:
                    targets.append({
                        "company": company,
                        "page_type": "students",
                        "url": students
                    })

    print("Loaded targets:", len(targets))
    return targets


def process_target(target: dict) -> None:
    company = target["company"]
    page_type = target["page_type"]
    url = target["url"]

    print(f"\nProcessing {company} | {page_type} | {url}")

    html = fetch_html(url)
    cleaned = clean_html(html)

    enriched_data = {
        "company": company,
        "page_type": page_type,
        "url": url,
        "title": cleaned["title"],
        "text": cleaned["text"]
    }

    filename = make_safe_filename(company, page_type)
    output_path = Path("data/clean") / filename

    save_json(enriched_data, output_path)

    print(f"Saved: {output_path}")


def main() -> None:
    targets = load_targets_from_csv()

    print(f"Total targets to process: {len(targets)}")

    for target in targets:
        process_target(target)


if __name__ == "__main__":
    main()