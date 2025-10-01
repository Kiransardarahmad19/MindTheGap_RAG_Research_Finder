import logging
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class STEmbedder:
    def __init__(self, model_name: str):
        logger.info("Loading SentenceTransformer | model=%s", model_name)
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> List[List[float]]:
        logger.debug("Encoding batch | count=%d", len(texts))
        vecs = self._model.encode(texts)
        return vecs.tolist() if hasattr(vecs, "tolist") else vecs

    def encode_one(self, text: str) -> List[float]:
        logger.debug("Encoding single text | length=%d", len(text))
        vec = self._model.encode(text)
        return vec.tolist() if hasattr(vec, "tolist") else vec
