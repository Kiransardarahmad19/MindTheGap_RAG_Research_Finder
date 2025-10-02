import os
import tempfile
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl

from src.config import AppConfig
from src.logging_setup import setup_logging
from src.vectorstore import ChromaVectorStore
from src.embeddings import STEmbedder
from src.llm_groq import GroqClient
from src.rag_service import RAGService
from src.ocr import PDFIngestor

import logging
logger = logging.getLogger(__name__)

# --- Boot ---
cfg = AppConfig()
setup_logging(cfg.log_dir, cfg.log_level)

app = FastAPI(title="Research Assistant API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Singletons / Services ---
vs = ChromaVectorStore(cfg.chroma_path, cfg.collection_name)
embedder = STEmbedder(cfg.embedding_model)
llm = GroqClient(cfg.groq_api_key, model="qwen/qwen3-32b")
rag = RAGService(vs, embedder, llm)
ingestor = PDFIngestor(
    vector_store=vs,
    embedder=embedder,
    ocr_lang=cfg.ocr_lang,
    tesseract_cmd=cfg.tesseract_cmd,
    save_dir=cfg.storage_pdf_dir,
)

# ---------- Schemas ----------
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=10)

class AskResponse(BaseModel):
    question: str
    answer: str

@app.get("/health")
def health():
    logger.info("Health check")
    return {"ok": True}

# -------- Q&A endpoint --------
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    logger.info("/ask | question='%s' top_k=%d", req.question, req.top_k)
    result = rag.ask_qa(req.question, top_k=req.top_k)
    return AskResponse(**result)

# -------- GAP endpoint --------
@app.post("/gaps", response_model=AskResponse)
def gaps(req: AskRequest):
    logger.info("/gaps | question='%s' top_k=%d", req.question, req.top_k)
    result = rag.find_gaps(req.question, top_k=req.top_k)
    return AskResponse(**result)

# ---------- Ingest (Upload) ----------
class IngestResponse(BaseModel):
    ok: bool
    pdf: str
    doc_id: str
    pages: int
    chunks: int
    embedded: int
    ids: List[str]
    collection: str
    saved_pdf_path: Optional[str] = None

@app.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf_endpoint(
    pdf: UploadFile = File(..., description="PDF file to OCR & index"),
    collection_name: Optional[str] = Form(None),
    ocr_lang: str = Form(None),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
    dpi: int = Form(300),
):
    logger.info("/ingest/pdf | filename=%s size~=%s", pdf.filename, getattr(pdf, "size", "unknown"))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await pdf.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        result = ingestor.ingest_path(
            tmp_path,
            collection_name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            dpi=dpi,
        )
        return IngestResponse(**result)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            logger.warning("Failed to delete temp file | %s", tmp_path)

# ---------- Ingest (URL) ----------
class IngestURLRequest(BaseModel):
    url: HttpUrl
    collection_name: Optional[str] = None
    chunk_size: int = 500
    chunk_overlap: int = 50
    dpi: int = 300

@app.post("/ingest/url", response_model=IngestResponse)
def ingest_pdf_url(req: IngestURLRequest):
    logger.info("/ingest/url | url=%s", req.url)
    result = ingestor.ingest_url(
        url=str(req.url),
        collection_name=req.collection_name,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
        dpi=req.dpi,
    )
    return IngestResponse(**result)
