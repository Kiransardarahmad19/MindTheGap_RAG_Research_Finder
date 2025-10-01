import logging
from typing import List, Dict, Any, Optional
import chromadb

logger = logging.getLogger(__name__)

class ChromaVectorStore:
    def __init__(self, path: str, collection_name: str):
        logger.info("Initializing ChromaVectorStore | path=%s collection=%s", path, collection_name)
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(collection_name)

    def upsert(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
    ) -> None:
        logger.info("Upserting into Chroma | count=%d", len(ids))
        self._collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def query(self, query_embedding: List[float], n_results: int = 3) -> Dict[str, Any]:
        logger.info("Querying Chroma | n_results=%d", n_results)
        res = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],  # ids come back automatically
        )
        # Basic shape logging (avoid logging full docs)
        logger.debug("Chroma raw keys: %s", list(res.keys()))
        return res
