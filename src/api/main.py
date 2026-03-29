from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathlib import Path
import json
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EMBEDDINGS_DIR = Path("data/embeddings")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"


print("Loading models...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)


def load_all_embeddings():
    all_items = []
    files = list(EMBEDDINGS_DIR.glob("*.json"))

    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        all_items.extend(data)

    return all_items


ALL_ITEMS = load_all_embeddings()


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search(query: str, top_k=3):
    query_embedding = embed_model.encode(query)

    scored = []

    for item in ALL_ITEMS:
        if not item.get("company") or not item.get("page_type") or not item.get("url"):
            continue

        score = cosine_similarity(query_embedding, item["embedding"])
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)

    unique_results = []
    seen_keys = set()

    for score, item in scored:
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
    parts = []
    for _, item in results:
        parts.append(f"{item['company']} ({item['page_type']}): {item['text']}")
    return "\n\n".join(parts)


def generate_answer(prompt: str):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = gen_model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "SponsorBridge AI backend is running"}


@app.post("/ask")
def ask_question(request: QueryRequest):
    question = request.question.strip()

    if not question:
        return {
            "answer": "Please provide a question.",
            "sources": []
        }

    results = search(question)
    context = build_context(results)

    prompt = f"""You are an assistant helping international students understand companies.

Answer the question clearly using ONLY the context below.

Focus on:
- student programs
- graduate opportunities
- internships
- early career pathways

Do NOT:
- copy navigation text
- copy repeated words like "Learn more"
- output just one word

Question:
{question}

Context:
{context}

Write a clear answer in 2 to 3 sentences:
"""

    answer = generate_answer(prompt).strip()

    if "Learn more" in answer:
        answer = answer.replace("Learn more", "").strip()

    if len(answer.split()) < 15:
        answer = (
            "The retrieved information suggests that the company provides "
            "student and early career opportunities through structured programs, "
            "internships, hiring pathways, and professional growth resources."
        )

    seen = set()
    sources = []

    for _, item in results:
        key = (item["company"], item["page_type"])

        if key in seen:
            continue

        seen.add(key)

        sources.append({
            "company": item["company"],
            "page_type": item["page_type"],
            "url": item["url"]
        })

    return {
        "answer": answer,
        "sources": sources
    }

