"""Hybrid retrieval combining BM25 and dense vector search with rank fusion."""

from typing import List, Dict, Any, Optional
from loguru import logger
from .bm25_retriever import BM25Retriever


class HybridRetriever:
    """Hybrid retriever combining BM25 keyword search and dense vector search."""

    def __init__(self, bm25_weight: float = 0.5, rrf_k: int = 60):
        """
        Initialize hybrid retriever.

        Args:
            bm25_weight: Weight for BM25 scores in linear combination (0-1)
            rrf_k: K parameter for Reciprocal Rank Fusion (default: 60)
        """
        self.bm25_weight = bm25_weight
        self.dense_weight = 1.0 - bm25_weight
        self.rrf_k = rrf_k
        self.bm25_retriever: Optional[BM25Retriever] = None

    def build_bm25_index(self, documents: List[Dict[str, Any]], text_field: str = "text"):
        """
        Build BM25 index from documents.

        Args:
            documents: List of document dictionaries with 'id' and text field
            text_field: Field name containing text to index
        """
        self.bm25_retriever = BM25Retriever()
        self.bm25_retriever.build_index(documents, text_field=text_field)
        logger.info(f"BM25 index built for {len(documents)} documents")

    def reciprocal_rank_fusion(
        self,
        bm25_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Combine BM25 and dense results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) across all result lists

        Args:
            bm25_results: Results from BM25 search
            dense_results: Results from dense vector search

        Returns:
            Fused and re-ranked results
        """
        # Create mapping of document IDs to RRF scores
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        # Process BM25 results
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result["document"]["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (self.rrf_k + rank))
            doc_map[doc_id] = result["document"]

        # Process dense results
        for rank, result in enumerate(dense_results, start=1):
            doc_id = result["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (self.rrf_k + rank))
            doc_map[doc_id] = result

        # Create final results sorted by RRF score
        fused_results = []
        for doc_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            fused_results.append({
                **doc_map[doc_id],
                "rrf_score": score
            })

        return fused_results

    def linear_combination(
        self,
        bm25_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Combine BM25 and dense results using weighted linear combination of normalized scores.

        Args:
            bm25_results: Results from BM25 search with scores
            dense_results: Results from dense vector search with scores

        Returns:
            Combined and re-ranked results
        """
        # Normalize BM25 scores
        bm25_scores = [r["score"] for r in bm25_results]
        max_bm25 = max(bm25_scores) if bm25_scores else 1.0
        min_bm25 = min(bm25_scores) if bm25_scores else 0.0
        bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1.0

        # Normalize dense scores
        dense_scores = [r.get("score", 0) for r in dense_results]
        max_dense = max(dense_scores) if dense_scores else 1.0
        min_dense = min(dense_scores) if dense_scores else 0.0
        dense_range = max_dense - min_dense if max_dense != min_dense else 1.0

        # Create score maps
        bm25_score_map: Dict[str, float] = {}
        for result in bm25_results:
            doc_id = result["document"]["id"]
            normalized_score = (result["score"] - min_bm25) / bm25_range
            bm25_score_map[doc_id] = normalized_score

        dense_score_map: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}
        for result in dense_results:
            doc_id = result["id"]
            normalized_score = (result.get("score", 0) - min_dense) / dense_range
            dense_score_map[doc_id] = normalized_score
            doc_map[doc_id] = result

        # Also add BM25-only documents to doc_map
        for result in bm25_results:
            doc_id = result["document"]["id"]
            if doc_id not in doc_map:
                doc_map[doc_id] = result["document"]

        # Combine scores
        combined_scores: Dict[str, float] = {}
        all_doc_ids = set(bm25_score_map.keys()) | set(dense_score_map.keys())

        for doc_id in all_doc_ids:
            bm25_score = bm25_score_map.get(doc_id, 0.0)
            dense_score = dense_score_map.get(doc_id, 0.0)
            combined_scores[doc_id] = (
                self.bm25_weight * bm25_score +
                self.dense_weight * dense_score
            )

        # Create final results sorted by combined score
        combined_results = []
        for doc_id, score in sorted(combined_scores.items(), key=lambda x: x[1], reverse=True):
            combined_results.append({
                **doc_map[doc_id],
                "hybrid_score": score,
                "bm25_score": bm25_score_map.get(doc_id, 0.0),
                "dense_score": dense_score_map.get(doc_id, 0.0)
            })

        return combined_results

    def search(
        self,
        query: str,
        dense_results: List[Dict[str, Any]],
        top_k: int = 10,
        fusion_method: str = "rrf"
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining BM25 and dense vector results.

        Args:
            query: Search query
            dense_results: Results from dense vector search
            top_k: Number of top results to return
            fusion_method: Fusion method - "rrf" (Reciprocal Rank Fusion) or "linear"

        Returns:
            Hybrid search results
        """
        if not self.bm25_retriever:
            logger.warning("BM25 index not built, returning dense results only")
            return dense_results[:top_k]

        # Get BM25 results
        bm25_results = self.bm25_retriever.search(query, top_k=top_k * 2)

        if not bm25_results:
            logger.info("No BM25 results, returning dense results only")
            return dense_results[:top_k]

        if not dense_results:
            logger.info("No dense results, returning BM25 results only")
            return [r["document"] for r in bm25_results[:top_k]]

        # Combine results using specified fusion method
        if fusion_method == "rrf":
            logger.info(f"Using Reciprocal Rank Fusion (k={self.rrf_k})")
            fused_results = self.reciprocal_rank_fusion(bm25_results, dense_results)
        else:  # linear
            logger.info(f"Using Linear Combination (BM25 weight={self.bm25_weight})")
            fused_results = self.linear_combination(bm25_results, dense_results)

        return fused_results[:top_k]
