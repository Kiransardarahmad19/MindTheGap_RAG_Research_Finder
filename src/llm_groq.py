import logging
import re
from groq import Groq

logger = logging.getLogger(__name__)

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)

def _sanitize_answer(text: str) -> str:
    if not text:
        return text
    cleaned = _THINK_TAG_RE.sub("", text)         
    cleaned = cleaned.strip()
    return cleaned

class GroqClient:
    def __init__(self, api_key: str, model: str = "qwen/qwen3-32b"):
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        self._client = Groq(api_key=api_key)
        self._model = model
        logger.info("GroqClient initialized | model=%s", model)

    def answer_from_context(self, question: str, context: str, temperature: float = 0.2) -> str:
        """
        Generate an answer that:
        - explains the paper(s) simply
        - identifies research gaps
        - uses ONLY the provided context
        - does NOT include chain-of-thought in the final output
        """
        logger.info("LLM call | question_len=%d context_len=%d", len(question or ""), len(context or ""))

        system = (
           
            "You are a diligent Researcher. Your tasks:\n"
            "1) Explain the provided research content in clear, simple language.\n"
            "2) Identify potential research gaps, limitations, and future directions explicitly from the context.\n"
            "Rules:\n"
            "- Use ONLY the provided context; if needed info is missing, say you don't have enough information.\n"
            "- Do NOT include your chain-of-thought, hidden analysis, or step-by-step reasoning. "
            "Return the final answer only.\n"
            "- Be concise, specific, and cite which parts of the context you're using (e.g., 'From Source 2')."
        )

        user = (
            "Context:\n"
            f"{context}\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Required format (adapt as needed):\n"
            "1) Plain-English Explanation:\n"
            "- <2–5 short bullets that summarize the key points relevant to the question>\n"
            "2) Potential Research Gaps (based on the context):\n"
            "- <bullet list of concrete, testable gaps or open directions>\n"
            "3) If information is insufficient:\n"
            "- State explicitly what is missing from the context to answer.\n"
        )

        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            top_p=0.95,
            max_completion_tokens=1024,
            stream=False,
        )

        raw = (completion.choices[0].message.content or "").strip()
        text = _sanitize_answer(raw)

        # Logging (avoid printing full content by default)
        logger.info("LLM response | raw_len=%d answer_len=%d", len(raw), len(text))
        logger.debug("LLM RAW ===\n%s\n=== END RAW", raw)
        logger.debug("LLM CLEAN ===\n%s\n=== END CLEAN", text)

        # Optional: short preview at INFO
        preview = text[:200].replace("\n", " ")
        logger.info("LLM preview | %s%s", preview, "..." if len(text) > 200 else "")

        return text
