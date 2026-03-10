"""Retrieval evaluation metrics for RAG systems.

Implements Precision@K, Recall@K, MRR, and Hit Rate metrics.
"""

from typing import List, Set, Any, Dict
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """Container for retrieval evaluation results."""
    precision_at_k: float
    recall_at_k: float
    mrr: float
    hit_rate: float
    k: int
    num_retrieved: int
    num_relevant: int
    num_relevant_retrieved: int


def precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int
) -> float:
    """
    Calculate Precision@K.
    
    Precision@K = (# relevant items in top-K) / K
    
    Args:
        retrieved_ids: List of retrieved document/chunk IDs (ordered by relevance)
        relevant_ids: Set of ground truth relevant IDs
        k: Number of top results to consider
        
    Returns:
        Precision score between 0 and 1
    """
    if k <= 0:
        return 0.0
    
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    
    return relevant_in_top_k / k


def recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int
) -> float:
    """
    Calculate Recall@K.
    
    Recall@K = (# relevant items in top-K) / (total # relevant items)
    
    Args:
        retrieved_ids: List of retrieved document/chunk IDs (ordered by relevance)
        relevant_ids: Set of ground truth relevant IDs
        k: Number of top results to consider
        
    Returns:
        Recall score between 0 and 1
    """
    if not relevant_ids:
        return 0.0
    
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    
    return relevant_in_top_k / len(relevant_ids)


def mean_reciprocal_rank(
    retrieved_ids: List[str],
    relevant_ids: Set[str]
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    MRR = 1 / rank of first relevant item
    
    Args:
        retrieved_ids: List of retrieved document/chunk IDs (ordered by relevance)
        relevant_ids: Set of ground truth relevant IDs
        
    Returns:
        MRR score between 0 and 1 (0 if no relevant items found)
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    
    return 0.0


def hit_rate(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int = None
) -> float:
    """
    Calculate Hit Rate (success rate).
    
    Hit Rate = 1 if any relevant item is in top-K, else 0
    
    Args:
        retrieved_ids: List of retrieved document/chunk IDs
        relevant_ids: Set of ground truth relevant IDs
        k: Number of top results to consider (None = all)
        
    Returns:
        1.0 if hit, 0.0 otherwise
    """
    if k is not None:
        retrieved_ids = retrieved_ids[:k]
    
    for doc_id in retrieved_ids:
        if doc_id in relevant_ids:
            return 1.0
    
    return 0.0


def ndcg_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.
    
    Uses binary relevance (relevant = 1, not relevant = 0).
    
    Args:
        retrieved_ids: List of retrieved document/chunk IDs
        relevant_ids: Set of ground truth relevant IDs
        k: Number of top results to consider
        
    Returns:
        NDCG score between 0 and 1
    """
    import math
    
    def dcg(relevances: List[int], k: int) -> float:
        """Calculate DCG."""
        dcg_sum = 0.0
        for i, rel in enumerate(relevances[:k]):
            dcg_sum += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
        return dcg_sum
    
    # Get binary relevance for retrieved items
    retrieved_relevances = [1 if doc_id in relevant_ids else 0 for doc_id in retrieved_ids]
    
    # Calculate DCG
    actual_dcg = dcg(retrieved_relevances, k)
    
    # Calculate ideal DCG (all relevant items first)
    ideal_relevances = [1] * min(len(relevant_ids), k) + [0] * (k - min(len(relevant_ids), k))
    ideal_dcg = dcg(ideal_relevances, k)
    
    if ideal_dcg == 0:
        return 0.0
    
    return actual_dcg / ideal_dcg


def evaluate_retrieval(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int = 5
) -> RetrievalResult:
    """
    Run all retrieval metrics and return combined results.
    
    Args:
        retrieved_ids: List of retrieved document/chunk IDs
        relevant_ids: Set of ground truth relevant IDs
        k: Number of top results for @K metrics
        
    Returns:
        RetrievalResult with all metrics
    """
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    
    return RetrievalResult(
        precision_at_k=precision_at_k(retrieved_ids, relevant_ids, k),
        recall_at_k=recall_at_k(retrieved_ids, relevant_ids, k),
        mrr=mean_reciprocal_rank(retrieved_ids, relevant_ids),
        hit_rate=hit_rate(retrieved_ids, relevant_ids, k),
        k=k,
        num_retrieved=len(retrieved_ids),
        num_relevant=len(relevant_ids),
        num_relevant_retrieved=relevant_in_top_k
    )


def aggregate_retrieval_metrics(results: List[RetrievalResult]) -> Dict[str, float]:
    """
    Aggregate retrieval metrics across multiple queries.
    
    Args:
        results: List of RetrievalResult objects
        
    Returns:
        Dictionary with mean metrics
    """
    if not results:
        return {}
    
    n = len(results)
    return {
        "mean_precision_at_k": sum(r.precision_at_k for r in results) / n,
        "mean_recall_at_k": sum(r.recall_at_k for r in results) / n,
        "mean_mrr": sum(r.mrr for r in results) / n,
        "mean_hit_rate": sum(r.hit_rate for r in results) / n,
        "k": results[0].k,
        "num_queries": n
    }


# Convenience aliases
mrr = mean_reciprocal_rank
