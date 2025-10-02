# MindTheGap – Research Gap Finder (RAG System)

MindTheGap is a RAG-powered system that reads research papers, retrieves the most relevant passages, and uses an LLM to surface gaps, limitations, and future directions. It ingests PDFs/URLs, performs OCR if needed, chunks & embeds content in ChromaDB, and runs targeted prompts to produce clear, actionable gap summaries. Designed for literature reviews, it highlights what’s known vs. what’s missing, cites evidence spans, and helps researchers prioritize novel, testable questions. Use it to accelerate scoping studies, thesis/problem definition, and to monitor emerging open problems across a field.

This project is a **Retrieval-Augmented Generation (RAG)** pipeline with a FastAPI backend.  
It allows you to:

1. **Ingest PDFs** (uploaded or from a URL) → OCR (Tesseract) → Chunk → Embed (Sentence Transformers) → Store in **ChromaDB**.
2. **Ask questions** about the ingested research using **Groq LLMs**.
3. **Find research gaps** by prompting the LLM to highlight limitations, open problems, and future work.
4. Monitor the pipeline with **structured logging** for ingestion, retrieval, and generation.

---

## Purpose of the Project
The goal is to build an AI-powered **Research Assistant** that can:
- Read & understand research papers.
- Answer questions in **simple, accessible language**.
- Detect **gaps, limitations, and future directions** from academic works.
- Scale to support **research discovery, analysis, and alerts**.

---

## Technologies Used
- **FastAPI** → REST API framework.
- **ChromaDB** → Vector database for embeddings.
- **Sentence Transformers** → Embedding model (`all-mpnet-base-v2`).
- **Groq API** → LLM for question answering and gap detection.
- **LangChain** → Text splitting (`RecursiveCharacterTextSplitter`).
- **Tesseract OCR** → Extract text from scanned PDFs.
- **pdfplumber** → Extract text & metadata from PDFs.
- **Logging** → Structured logs for ingestion, retrieval, and LLM calls.

---

##  Endpoints

### Health Check
`GET /health`  
Returns `{ "ok": true }` if the service is running.

---

### Ingest PDF (Upload)
`POST /ingest/pdf`  
Upload a PDF → OCR → Chunk → Embed → Store.

- **Request (form-data):**
  - `pdf`: (File) Your PDF.
  - `collection_name` (optional).
  - `ocr_lang` (default: `"eng"`).
  - `chunk_size` (default: `500`).
  - `chunk_overlap` (default: `50`).
  - `dpi` (default: `300`).

---

### Ingest PDF (URL)
`POST /ingest/url`  
Provide a URL → Downloads PDF → OCR → Embeds → Stores.

- **Request (JSON):**
```json
{
  "url": "https://arxiv.org/pdf/1234.5678.pdf",
  "collection_name": "ai_papers",
  "chunk_size": 500,
  "chunk_overlap": 50,
  "dpi": 300
}
```

---

### Ask a Question
`POST /ask`  
Ask a plain question → Retrieve context → Generate answer.

- **Request (JSON):**
```json
{
  "question": "What is gradient descent?",
  "top_k": 3
}
```

- **Response (simplified):**
```json
{
  "question": "What is gradient descent?",
  "answer": "Gradient descent is an optimization algorithm ..."
}
```

---

### 🔍 Find Research Gaps
`POST /gaps`  
Special endpoint for **gap detection**.  
Uses a separate LLM prompt that explains the paper in simple terms and **identifies open problems or missing areas**.

- **Request (JSON):**
```json
{
  "question": "What are the gaps in micro-paper research?",
  "top_k": 3
}
```

- **Response (simplified):**
```json
{
  "question": "What are the gaps in micro-paper research?",
  "answer": "1) Lack of formal evaluation ... 2) Limited adoption in academia ... 3) Unclear archival standards ..."
}
```

Only the **final answer** is returned in API responses.  
Retrieved chunks, chain-of-thought, and metadata are logged internally.

---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Setup
Set environment variables:
```bash
export GROQ_API_KEY="gsk_********"
export CHROMA_PATH="chroma_store"
export COLLECTION_NAME="edu_books"
export EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2"
export STORAGE_PDF_DIR="storage/pdfs"
# Optional for OCR
export TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### 3. Run the server
```bash
uvicorn src.main:app --reload --port 8000
```

---

## Example Usage (Postman)

**Upload & ingest a PDF**  
- `POST http://localhost:8000/ingest/pdf` → form-data → file.

**Ingest by URL**  
- `POST http://localhost:8000/ingest/url` → JSON with `url`.

**Ask a question**  
- `POST http://localhost:8000/ask` → JSON with `question`.

**Find research gaps**  
- `POST http://localhost:8000/gaps` → JSON with `question`.

---

##  Future Roadmap
- **Hybrid retrieval**: combine dense embeddings + BM25 + rerankers.  
- **Cross-paper gap analysis**: compare gaps across multiple related works.  
- **Knowledge graphs**: build networks of research topics, methods, and datasets.  
- **Alerts**: notify when new papers in a topic have unexplored gaps.  
- **Export**: generate structured reports in Markdown, Docx, or Notion.
