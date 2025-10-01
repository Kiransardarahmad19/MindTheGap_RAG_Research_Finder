import logging
from typing import Dict, Any, List

from .embeddings import STEmbedder
from .vectorstore import ChromaVectorStore
from .llm_groq import GroqClient

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, vs: ChromaVectorStore, embedder: STEmbedder, llm: GroqClient):
        self.vs = vs
        self.embedder = embedder
        self.llm = llm
        logger.info("RAGService initialized")

    def retrieve(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        logger.info("Retrieve | question_len=%d top_k=%d", len(question), top_k)
        q_emb = self.embedder.encode_one(question)
        res = self.vs.query(q_emb, n_results=top_k)

        # Flatten lists
        docs  = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        ids   = (res.get("ids") or [[]])[0]

        n = min(len(docs), len(metas), len(dists), len(ids)) if ids else min(len(docs), len(metas), len(dists))
        docs, metas, dists = docs[:n], metas[:n], dists[:n]
        ids = ids[:n] if ids else []

        # Log a compact view
        preview = [{"id": ids[i] if i < len(ids) else None, "distance": dists[i], "section": metas[i].get("section") if i < len(metas) and isinstance(metas[i], dict) else None} for i in range(n)]
        logger.info("Retrieve hits | count=%d preview=%s", n, preview[:5])

        return {"documents": docs, "metadatas": metas, "distances": dists, "ids": ids}

    def _build_context(self, docs: List[str]) -> str:
        if not docs:
            return "No context available."
        parts = []
        for i, d in enumerate(docs):
            snippet = d if len(d) <= 4000 else d[:4000] + " ..."
            parts.append(f"[Source {i+1}]\n{snippet}")
        ctx = "\n\n".join(parts)
        logger.debug("Context built | length=%d", len(ctx))
        return ctx

    def ask(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        hits = self.retrieve(question, top_k=top_k)
        context = self._build_context(hits["documents"])
        logger.info("Prompt to LLM | question='%s' ctx_chars=%d", question[:120], len(context))
        answer = self.llm.answer_from_context(question, context)

        sources = []
        for i, doc in enumerate(hits["documents"]):
            sources.append({
                "id": hits["ids"][i] if i < len(hits.get("ids", [])) else None,
                "document": doc,
                "metadata": hits["metadatas"][i] if i < len(hits["metadatas"]) else None,
                "distance": hits["distances"][i] if i < len(hits["distances"]) else None,
            })

        logger.info("Ask complete | answer_len=%d sources=%d", len(answer), len(sources))
        return {"question": question, "answer": answer, "sources": sources}
