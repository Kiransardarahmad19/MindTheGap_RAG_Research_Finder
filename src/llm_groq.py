import logging
import re
from groq import Groq

logger = logging.getLogger(__name__)

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)

def _sanitize_answer(text: str) -> str:
    if not text:
        return text
    cleaned = _THINK_TAG_RE.sub("", text).strip()
    return cleaned

class GroqClient:
    def __init__(self, api_key: str, model: str = "qwen/qwen3-32b"):
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        self._client = Groq(api_key=api_key)
        self._model = model
        logger.info("GroqClient initialized | model=%s", model)

    # ---------- Q&A prompt ----------
    def answer_qa_from_context(self, question: str, context: str, temperature: float = 0.2) -> str:
        logger.info("LLM[QA] | q_len=%d ctx_len=%d", len(question or ""), len(context or ""))

        system = (
            "You are a precise academic assistant. Use ONLY the provided context. "
            "If the answer is not in the context, say you don't have enough information. "
            "Do NOT include chain-of-thought or hidden reasoning—return the final answer only."
        )
        user = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer concisely. If applicable, reference sources as [Source N]."

        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            top_p=0.95,
            max_completion_tokens=1024,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        text = _sanitize_answer(raw)
        logger.info("LLM[QA] resp | raw_len=%d ans_len=%d", len(raw), len(text))
        logger.debug("LLM[QA] RAW ===\n%s\n=== END", raw)
        logger.debug("LLM[QA] CLEAN ===\n%s\n=== END", text)
        return text

    # ---------- GAP prompt ----------
    def answer_gap_from_context(self, question: str, context: str, temperature: float = 0.2) -> str:
        logger.info("LLM[GAP] | q_len=%d ctx_len=%d", len(question or ""), len(context or ""))

        system = (
            "You are a diligent Researcher.\n"
            "Tasks:\n"
            "1) Explain the research content simply.\n"
            "2) Identify potential research gaps, limitations, and future directions explicitly from the context.\n"
            "Rules: Use ONLY the provided context; if info is missing, say so. "
            "Do NOT include chain-of-thought—return final answer only. "
            "Be specific and, when helpful, point to [Source N]."
        )
        user = (
            f"Context:\n{context}\n\nQuestion:\n{question}\n\n"
            "Format:\n"
            "1) Plain-English Explanation (2–5 bullets)\n"
            "2) Potential Research Gaps (bullets)\n"
            "3) If information is insufficient, list what’s missing\n"
        )

        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            top_p=0.95,
            max_completion_tokens=1024,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        text = _sanitize_answer(raw)
        logger.info("LLM[GAP] resp | raw_len=%d ans_len=%d", len(raw), len(text))
        logger.debug("LLM[GAP] RAW ===\n%s\n=== END", raw)
        logger.debug("LLM[GAP] CLEAN ===\n%s\n=== END", text)
        return text
