"""Cross-encoder re-ranker for improving retrieval precision."""

from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
from loguru import logger


class CrossEncoderReranker:
    """Re-ranks retrieved documents using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder re-ranker.

        Args:
            model_name: Name of the cross-encoder model to use
        """
        self.model_name = model_name
        self._model: Optional[CrossEncoder] = None
        logger.info(f"CrossEncoderReranker initialized with model: {model_name}")

    @property
    def model(self) -> CrossEncoder:
        """Lazy load the cross-encoder model."""
        if self._model is None:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder model loaded successfully")
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        text_field: str = "text"
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents using cross-encoder.

        Args:
            query: Search query
            documents: List of retrieved documents
            top_k: Number of top results to return (None = return all)
            text_field: Field name containing document text

        Returns:
            Re-ranked documents with cross-encoder scores
        """
        if not documents:
            return []

        logger.info(f"Re-ranking {len(documents)} documents for query: '{query[:50]}...'")

        pairs = []
        for doc in documents:
            text = doc.get(text_field, "")
            if isinstance(text, str):
                pairs.append([query, text])
            else:
                pairs.append([query, str(text)])

        scores = self.model.predict(pairs)

        reranked = []
        for doc, score in zip(documents, scores):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(score)
            if "score" in doc:
                doc_copy["original_score"] = doc["score"]
            doc_copy["score"] = float(score)
            reranked.append(doc_copy)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        if top_k is not None:
            reranked = reranked[:top_k]

        logger.info(f"Re-ranking complete. Top score: {reranked[0]['rerank_score']:.4f}")
        return reranked

    def score_pairs(self, query: str, texts: List[str]) -> List[float]:
        """
        Score query-text pairs.

        Args:
            query: Search query
            texts: List of text passages

        Returns:
            List of scores
        """
        pairs = [[query, text] for text in texts]
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]
