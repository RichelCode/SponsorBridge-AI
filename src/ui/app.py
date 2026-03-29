from pathlib import Path
import json
import numpy as np
import gradio as gr

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

EMBEDDINGS_DIR = Path("data/embeddings")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"
ASR_MODEL_NAME = "openai/whisper-base"

print("Loading ASR model...")
asr_model = pipeline(
    "automatic-speech-recognition",
    model=ASR_MODEL_NAME
)

print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

print("Loading generation tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)

print("Loading generation model...")
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)

def transcribe_audio(audio_path: str) -> str:
    if not audio_path:
        return ""

    result = asr_model(audio_path)
    return result["text"].strip()

def load_all_embeddings():
    all_items = []

    files = list(EMBEDDINGS_DIR.glob("*.json"))
    print(f"Loading {len(files)} embedding files")

    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        all_items.extend(data)

    print(f"Total chunks loaded: {len(all_items)}")
    return all_items


ALL_ITEMS = load_all_embeddings()


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search(query: str, items, top_k=3):
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

    for _, item in results:
        context_parts.append(
            f"{item['company']} ({item['page_type']}): {item['text']}"
        )

    return "\n\n".join(context_parts)


def build_prompt(question: str, context: str) -> str:
    return f"""You are an assistant helping international students understand companies.

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

Write a clear answer in 2–3 sentences:
"""


def generate_answer(prompt: str) -> str:
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

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return answer


def format_sources(results) -> str:
    lines = []

    for score, item in results:
        lines.append(
            f"- {item['company']} | {item['page_type']} | score={score:.4f}\n  {item['url']}"
        )

    return "\n".join(lines)


def ask_question(question: str, audio):
    typed_question = (question or "").strip()
    spoken_question = transcribe_audio(audio) if audio else ""

    final_question = typed_question if typed_question else spoken_question

    if not final_question:
        return "Please enter a question or record audio.", "", ""

    results = search(final_question, ALL_ITEMS, top_k=3)
    context = build_context(results)
    prompt = build_prompt(final_question, context)

    answer = generate_answer(prompt)

    if len(answer.strip()) < 20:
        answer = "The retrieved information suggests that companies provide student and early career opportunities, but more detailed context is needed."

    sources = format_sources(results)

    return final_question, answer, sources

demo = gr.Interface(
    fn=ask_question,
    inputs=[
        gr.Textbox(
            lines=3,
            placeholder="Ask something like: What information is available about JPMorgan Chase student opportunities?",
            label="Your Question"
        ),
        gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Optional Voice Question"
        )
    ],
    outputs=[
        gr.Textbox(label="Question Used"),
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Sources")
    ],
    title="SponsorBridge AI",
    description="A simple RAG assistant for exploring company information relevant to international students."
)
if __name__ == "__main__":
    demo.launch()