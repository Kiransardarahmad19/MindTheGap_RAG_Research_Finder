import os
import re
import uuid
import shutil
import tempfile
import logging
from typing import List, Tuple, Optional, Dict

import requests
import pdfplumber
import pytesseract
from PIL import Image

from .embeddings import STEmbedder
from .vectorstore import ChromaVectorStore

logger = logging.getLogger(__name__)

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

_REFERENCES_CUTOFF = re.compile(r"^\s*(references|bibliography|works\s+cited|acknowledg(e)ments?)\b", re.I)
_SECTION_PATTERNS = [
    ("abstract",      r"^\s*abstract\b[:\.\s]*", re.I),
    ("introduction",  r"^\s*introduction\b[:\.\s]*", re.I),
    ("background",    r"^\s*background\b[:\.\s]*", re.I),
    ("related work",  r"^\s*related\s+work\b[:\.\s]*", re.I),
    ("methods",       r"^\s*(methods?|methodology)\b[:\.\s]*", re.I),
    ("results",       r"^\s*results?\b[:\.\s]*", re.I),
    ("discussion",    r"^\s*discussion\b[:\.\s]*", re.I),
    ("conclusion",    r"^\s*conclusions?\b[:\.\s]*", re.I),
    ("limitations",   r"^\s*limitations?\b[:\.\s]*", re.I),
    ("future work",   r"^\s*(future\s+work|future\s+directions|further\s+work)\b[:\.\s]*", re.I),
]

class PDFIngestor:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedder: STEmbedder,
        ocr_lang: str = "eng",
        tesseract_cmd: str = "",
        save_dir: Optional[str] = None,
    ):
        self.vs = vector_store
        self.embedder = embedder
        self.ocr_lang = ocr_lang
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.save_dir = save_dir
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        logger.info("PDFIngestor ready | ocr_lang=%s save_dir=%s", ocr_lang, save_dir)

    def ingest_path(
        self,
        pdf_path: str,
        collection_name: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        dpi: int = 300,
    ) -> dict:
        logger.info("Ingesting PDF from path | path=%s", pdf_path)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pages_text, meta = self._extract_pages(pdf_path, dpi=dpi)
        sections = self._split_sections(pages_text)
        filtered_sections = self._filter_sections(sections)
        chunks, per_md = self._chunk_sections(filtered_sections, chunk_size, chunk_overlap)

        doc_id = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_{uuid.uuid4().hex[:8]}"
        metadata_common = {
            "source_file": os.path.basename(pdf_path),
            "doc_id": doc_id,
            **meta,
        }

        self._embed_upsert(chunks, metadata_common, per_md, collection_name)

        saved_pdf_path = self._save_original(pdf_path, doc_id)
        result = {
            "ok": True,
            "pdf": os.path.basename(pdf_path),
            "doc_id": doc_id,
            "pages": len(pages_text),
            "sections_detected": [s["name"] for s in sections],
            "sections_indexed": [s["name"] for s in filtered_sections],
            "chunks": len(chunks),
            "embedded": len(chunks),
            "ids": [f"{doc_id}_chunk_{i}" for i in range(len(chunks))][:50],
            "collection": collection_name if collection_name else "default",
            "meta": metadata_common,
            "saved_pdf_path": saved_pdf_path,
        }
        logger.info("Ingest result | doc_id=%s chunks=%d", doc_id, len(chunks))
        return result

    def ingest_url(
        self,
        url: str,
        collection_name: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        dpi: int = 300,
        timeout: int = 30,
    ) -> dict:
        logger.info("Ingesting PDF from URL | url=%s", url)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp_path = tmp.name
        try:
            r = requests.get(url, stream=True, timeout=timeout)
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return self.ingest_path(tmp_path, collection_name, chunk_size, chunk_overlap, dpi)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    def _save_original(self, pdf_path: str, doc_id: str) -> Optional[str]:
        if not self.save_dir:
            return None
        os.makedirs(self.save_dir, exist_ok=True)
        out_path = os.path.join(self.save_dir, f"{doc_id}.pdf")
        try:
            shutil.copy2(pdf_path, out_path)
            logger.info("Saved original PDF | %s", out_path)
            return out_path
        except Exception as e:
            logger.warning("Failed saving original PDF | %s | %s", pdf_path, e)
            return None

    def _extract_pages(self, pdf_path: str, dpi: int = 300) -> Tuple[List[str], Dict[str, str]]:
        import pdfplumber
        pages_text: List[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            meta = self._extract_meta(pdf)
            for i, page in enumerate(pdf.pages):
                txt = self._extract_page_text_or_ocr(page, dpi=dpi)
                logger.debug("Page extracted | page=%d chars=%d", i+1, len(txt))
                pages_text.append(txt)
        return pages_text, meta

    def _extract_meta(self, pdf) -> dict:
        meta = pdf.metadata or {}
        title = (meta.get("Title") or "").strip()
        authors = (meta.get("Author") or "").strip()
        subject = (meta.get("Subject") or "").strip()
        keywords = (meta.get("Keywords") or "").strip()

        if not title or not authors:
            try:
                first = (pdf.pages[0].extract_text() or "")
                lines = [l.strip() for l in first.splitlines() if l.strip()]
                if not title and lines:
                    title = lines[0][:200]
                if not authors and len(lines) > 1 and re.search(r"(,| and )", lines[1], re.I):
                    authors = lines[1][:200]
            except Exception:
                pass

        year = ""
        for k in ["ModDate", "CreationDate", "Producer", "Creator"]:
            v = str(meta.get(k) or "")
            m = re.search(r"(19|20)\d{2}", v)
            if m:
                year = m.group(0)
                break
        if not year:
            try:
                m = re.search(r"(19|20)\d{2}", pdf.pages[0].extract_text() or "")
                if m: year = m.group(0)
            except Exception:
                pass

        clean = {"title": title, "authors": authors, "subject": subject, "keywords": keywords, "year": year}
        logger.debug("PDF meta extracted | %s", clean)
        return clean

    def _extract_page_text_or_ocr(self, page, dpi: int = 300) -> str:
        txt = (page.extract_text() or "").strip()
        if len(txt) < 30:
            try:
                image: Image.Image = page.to_image(resolution=dpi).original
                txt = pytesseract.image_to_string(image, lang=self.ocr_lang).strip()
            except Exception:
                pass
        return txt

    def _split_sections(self, pages_text: List[str]) -> List[dict]:
        full_lines: List[Tuple[int, str]] = []
        for i, page_txt in enumerate(pages_text, start=1):
            for line in (page_txt or "").splitlines():
                full_lines.append((i, line))

        cut_idx = None
        for idx, (_, line) in enumerate(full_lines):
            if _REFERENCES_CUTOFF.match(line):
                cut_idx = idx
                break
        if cut_idx is not None:
            full_lines = full_lines[:cut_idx]

        headers = []
        for idx, (pg, line) in enumerate(full_lines):
            for name, pattern, flags in _SECTION_PATTERNS:
                if re.match(pattern, line, flags):
                    headers.append((idx, name))
                    break

        if not headers:
            joined = "\n".join([ln for _, ln in full_lines]).strip()
            sections = [{"name": "body", "text": joined, "page_start": 1, "page_end": len(pages_text)}] if joined else []
            logger.debug("Sections parsed | no headers found, body_len=%d", len(joined))
            return sections

        sections: List[dict] = []
        for k, (start_idx, name) in enumerate(headers):
            end_idx = headers[k + 1][0] if k + 1 < len(headers) else len(full_lines)
            chunk_lines = full_lines[start_idx:end_idx]
            text = "\n".join([ln for _, ln in chunk_lines]).strip()
            if not text:
                continue
            pages = [pg for pg, _ in chunk_lines]
            sections.append({
                "name": name,
                "text": text,
                "page_start": min(pages),
                "page_end": max(pages),
            })
        logger.debug("Sections parsed | count=%d", len(sections))
        return sections

    def _filter_sections(self, sections: List[dict]) -> List[dict]:
        keep = {"abstract", "introduction", "conclusion", "future work", "limitations", "discussion"}
        out = [s for s in sections if s["name"] in keep] or [s for s in sections if s["name"] == "body"]
        logger.info("Sections kept | %s", [s["name"] for s in out])
        return out

    def _chunk_sections(self, sections: List[dict], chunk_size: int, chunk_overlap: int):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
        )
        chunks: List[str] = []
        per_md: List[dict] = []
        for sec in sections:
            sec_chunks = splitter.split_text(sec["text"] or "")
            sec_chunks = [c.strip() for c in sec_chunks if c and c.strip()]
            chunks.extend(sec_chunks)
            per_md.extend([{"section": sec["name"], "page_start": sec["page_start"], "page_end": sec["page_end"]} for _ in sec_chunks])
        logger.info("Chunked sections | total_chunks=%d", len(chunks))
        return chunks, per_md

    def _embed_upsert(self, chunks: List[str], common_md: dict, per_md: List[dict], collection_name: Optional[str]):
        if not chunks:
            logger.warning("No chunks to embed")
            return
        ids = [f"{common_md['doc_id']}_chunk_{i}" for i in range(len(chunks))]
        vectors = self.embedder.encode(chunks)
        # Light log only (to avoid secrets/PII in logs)
        logger.info("Embedding complete | dims=%d count=%d", len(vectors[0]) if vectors else 0, len(vectors))
        metadatas = []
        for i, md in enumerate(per_md):
            m = {**common_md, **md, "chunk_index": i}
            metadatas.append(m)
        self.vs.upsert(ids=ids, documents=chunks, embeddings=vectors, metadatas=metadatas)
