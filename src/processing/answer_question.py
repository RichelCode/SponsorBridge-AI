from pathlib import Path
import json
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

EMBEDDINGS_DIR = Path("data/embeddings")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-small"


def load_all_embeddings():
    all_items = []

    files = list(EMBEDDINGS_DIR.glob("*.json"))
    print(f"Loading {len(files)} embedding files")

    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        all_items.extend(data)

    print(f"Total chunks loaded: {len(all_items)}")
    return all_items


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search(query: str, items, embed_model, top_k=5):
    query_embedding = embed_model.encode(query)

    scored_results = []

    for item in items:
        if not item.get("company") or not item.get("page_type") or not item.get("url"):
            continue

        score = cosine_similarity(query_embedding, item["embedding"])
        scored_results.append((score, item))

    scored_results.sort(key=lambda x: x[0], reverse=True)

    unique_results = []
    seen_keys = set()

    for score, item in scored_results:
        text_key = " ".join(item["text"].split()[:40]).strip().lower()
        dedupe_key = (item["company"], item["page_type"], text_key)

        if dedupe_key in seen_keys:
            continue

        seen_keys.add(dedupe_key)
        unique_results.append((score, item))

        if len(unique_results) == top_k:
            break

    return unique_results


def build_context(results):
    context_parts = []

    for score, item in results:
        context_parts.append(
            f"Company: {item['company']}\n"
            f"Page Type: {item['page_type']}\n"
            f"Content: {item['text']}\n"
        )

    return "\n\n".join(context_parts)


def build_prompt(question: str, context: str) -> str:
    return f"""Question: {question}

Context:
{context}

Write a short answer in 2 to 3 sentences using only the context.
Do not copy instructions.
Do not copy headings or menu text.
If the answer is unclear, say: I do not have enough information.

Answer:"""

def generate_answer(prompt: str, tokenizer, model) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        temperature=0.0
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return answer


def main():
    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Loading generation tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)

    print("Loading generation model...")
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)

    items = load_all_embeddings()

    question = input("\nEnter your question: ")

    results = search(question, items, embed_model, top_k=3)

    print("\nTop retrieved chunks:\n")
    for score, item in results:
        print(f"Score: {score:.4f} | {item['company']} | {item['page_type']}")
        print(f"Snippet: {item['text'][:250]}")
        print("-" * 50)

    context = build_context(results)
    prompt = build_prompt(question, context)

    answer = generate_answer(prompt, tokenizer, gen_model)

    print("\nGenerated Answer:\n")
    print(answer)

    print("\nSources used:\n")
    for _, item in results:
        print(f"- {item['company']} | {item['page_type']} | {item['url']}")


if __name__ == "__main__":
    main()