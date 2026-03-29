# Learning Log

## Day 1 — Setup

- Installed VS Code
- Installed Python
- Installed Git
- Created project structure

### What I understand
VS Code is my coding workspace.  
Python runs my code.  
Git helps me track and save my progress.

---

## Day 2 — Web Scraping Pipeline

### What I built

I built a pipeline that:
1. Fetches web pages using Python
2. Saves raw HTML
3. Cleans HTML into readable text
4. Stores structured JSON data

### Key concepts I learned

#### 1. HTTP Requests
- Used `requests.get()` to fetch web pages
- Learned about SSL errors and how to debug them

#### 2. Raw vs Clean Data
- `data/raw/` stores original HTML
- `data/clean/` stores processed readable text
- Important for debugging and reproducibility

#### 3. HTML Parsing
- Used BeautifulSoup to parse HTML
- Removed `<script>`, `<style>`, and `<noscript>` tags
- Extracted visible text and titles

#### 4. Data Pipeline Thinking
- Broke process into steps:
  fetch → clean → save
- Each function does one clear job

---

## Day 3 — Scaling with Multiple Companies

### What I built

- Created a CSV file (`company_targets.csv`) to control scraping
- Built a system that reads URLs from CSV instead of hardcoding
- Processed multiple company pages automatically

### Key concepts

#### 1. CSV as Control Layer
- CSV acts as a configuration file
- Controls which companies and pages to scrape
- Enables scalability

#### 2. Automation
- Instead of scraping one page manually
- The system loops through multiple companies

#### 3. Metadata
- Added:
  - company name
  - page type (about, careers, students)
  - URL
- Makes data more structured and meaningful

---

## Day 4 — Preparing Data for AI (Chunking)

### What I built

- Split long text into smaller chunks
- Saved chunked data into `data/chunks/`

### Key concepts

#### 1. Why Chunking?
- Large text is hard for models to process
- Smaller chunks improve retrieval accuracy

#### 2. Chunking Strategy
- Split text into fixed-size word groups
- Each chunk gets:
  - company
  - page type
  - chunk_id

#### 3. AI Readiness
- Chunking is required for:
  - embeddings
  - retrieval systems (RAG)

---

## Day 5 — Embeddings (Turning Text into Meaning)

### What I built

- Used `sentence-transformers` model
- Converted text chunks into embeddings (vectors)
- Saved embeddings into `data/embeddings/`

### Key concepts

#### 1. What is an Embedding?
- A numerical representation of text
- Captures semantic meaning (not just keywords)

#### 2. Why Embeddings Matter
- Allows similarity search
- Enables the system to find relevant answers

#### 3. Pipeline Extension
- chunk → embedding → saved vector

---

## Big Picture Understanding

### Full Pipeline

```text
Website → HTML → Clean Text → JSON → Chunks → Embeddings

---

## Day 6 — Retrieval and Answer Generation

### What I built

- Built a semantic retrieval system using embeddings and cosine similarity
- Allowed a user question to be converted into an embedding and compared against stored chunk embeddings
- Returned the most relevant chunks as context
- Added an answer generation step using a transformer model

### Key concepts I learned

#### 1. Retrieval
- Retrieval is the process of finding the most relevant text chunks for a user question
- I used cosine similarity to compare the embedding of the question with stored chunk embeddings
- Higher similarity means the chunk is more relevant

#### 2. RAG pipeline structure
My system now follows this flow:

```text
Website → Clean Text → Chunks → Embeddings → Retrieval → Generated Answer

---

## Day 7 — Full-Stack Integration and UI

### What I built

- Built a Gradio-based UI for asking questions and viewing sources
- Upgraded the project to use a stronger Hugging Face generation model
- Added support for audio input preparation using a Hugging Face speech-to-text model
- Created a FastAPI backend with a real `/ask` endpoint
- Connected a Lovable-generated frontend UI to the FastAPI backend
- Began debugging frontend-to-backend rendering and state mapping

### Key concepts I learned

#### 1. Full-stack AI architecture
My system is no longer just a script. It now has:

```text
Frontend UI → FastAPI backend → retrieval pipeline → answer generation