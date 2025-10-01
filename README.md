# 📚 RAG PDF Ingestion & QA System

This project is a **Retrieval-Augmented Generation (RAG)** pipeline with a FastAPI backend.  
It allows you to:

1. **Ingest any PDF** → OCR (Tesseract) → Chunk → Embed (Sentence Transformers) → Store in **ChromaDB**.
2. **Ask questions** in plain text → Retrieve relevant chunks → Generate answers with **Groq LLM**.
3. Serve endpoints that work easily with **Postman** or any frontend.

---

## 🚀 Purpose of the Project
The goal is to build an **AI tutor** that can:
- Read & understand educational PDFs (via OCR).
- Store them in a vector database for semantic search.
- Answer user questions by combining **retrieved context** with a **large language model** (LLM).
- Provide an easy-to-use API for integration into apps, dashboards, or chatbots.

---

## 🛠️ Technologies Used
- **FastAPI** → REST API framework.
- **ChromaDB** → Vector database for embeddings.
- **Sentence Transformers** → Embedding model (`all-mpnet-base-v2`).
- **Groq API** → LLM & Chat Completions.
- **LangChain** → Text splitting (`RecursiveCharacterTextSplitter`).
- **Tesseract OCR** → Extract text from scanned PDFs.
- **pdfplumber** → Extract PDF pages as images for OCR.
- **Pillow (PIL)** → Image processing.

---

## ⚙️ Endpoints

### ✅ Health Check
`GET /health`  
Returns `{ "ok": true }` if the service is running.

---

### 📥 Ingest PDF
`POST /ingest/pdf`  
Accepts a PDF file, runs OCR, splits into chunks, embeds, and stores in ChromaDB.

- **Request (form-data):**
  - `pdf`: (File) Your PDF file.
  - `collection_name` (optional): Custom ChromaDB collection.
  - `ocr_lang` (default: `"eng"`): OCR language(s) e.g., `"eng+urd"`.
  - `chunk_size` (default: `500`): Chunk size for splitting.
  - `chunk_overlap` (default: `50`): Overlap between chunks.
  - `dpi` (default: `300`): Resolution for PDF to image.

- **Response Example:**
```json
{
  "ok": true,
  "pdf": "test.pdf",
  "doc_id": "test_12ab34cd",
  "pages": 12,
  "chunks": 138,
  "embedded": 138,
  "ids": ["test_12ab34cd_chunk_0", "test_12ab34cd_chunk_1", "..."],
  "collection": "edu_books"
}
```

---

### ❓ Ask a Question
`POST /ask`  
Takes a text query, retrieves top chunks, and generates an LLM-powered answer.

- **Request (JSON):**
```json
{
  "question": "What is gradient descent?",
  "top_k": 3
}
```

- **Response Example:**
```json
{
  "question": "What is gradient descent?",
  "answer": "Gradient descent is an optimization algorithm ...",
  "sources": [
    {
      "id": "test_12ab34cd_chunk_0",
      "document": "Gradient descent is a method to minimize cost function ...",
      "metadata": {"source_file": "test.pdf", "doc_id": "test_12ab34cd", "chunk_index": 0},
      "distance": 0.18
    }
  ]
}
```

---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install fastapi uvicorn pdfplumber pytesseract pillow chromadb sentence-transformers langchain groq
# if needed:
# pip install langchain-text-splitters
```

### 2. Environment Setup
Set environment variables:

```bash
export GROQ_API_KEY="gsk_********"     # replace with your Groq API key
export CHROMA_PATH="chroma_store"
export COLLECTION_NAME="edu_books"
export EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2"

# Optional (Windows Tesseract path)
export TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### 3. Run the server
```bash
uvicorn api:app --reload --port 8000
```

---

## 🧪 Example Usage with Postman

### Upload and ingest a PDF
- **POST** → `http://localhost:8000/ingest/pdf`  
- Body → `form-data` → Key=`pdf` → Choose File → your `file.pdf`.

### Ask a question
- **POST** → `http://localhost:8000/ask`  
- Body → `raw JSON`:
```json
{
  "question": "Summarize chapter 1",
  "top_k": 3
}
```
