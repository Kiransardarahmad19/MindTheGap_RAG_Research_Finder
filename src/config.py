from dataclasses import dataclass
import os
from pathlib import Path

@dataclass(frozen=True)
class AppConfig:
    # Vector / embeddings
    chroma_path: str = os.getenv("CHROMA_PATH", "chroma_store")
    collection_name: str = os.getenv("COLLECTION_NAME", "edu_books")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")

    # LLM
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")

    # OCR
    tesseract_cmd: str = os.getenv("TESSERACT_CMD", "")  # e.g., r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    ocr_lang: str = os.getenv("OCR_LANG", "eng")

    # Storage for original PDFs
    storage_pdf_dir: str = os.getenv("STORAGE_PDF_DIR", str(Path("storage") / "pdfs"))

    # Logging
    log_dir: str = os.getenv("LOG_DIR", "logs")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
