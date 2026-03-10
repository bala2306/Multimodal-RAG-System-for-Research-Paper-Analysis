"""Unit tests for evaluation metrics."""

import pytest
from evaluation.metrics.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    hit_rate,
    ndcg_at_k,
    evaluate_retrieval
)
from evaluation.metrics.generation_metrics import (
    GenerationMetrics,
    calibration_error
)


class TestRetrievalMetrics:
    """Test retrieval metrics."""
    
    def test_precision_at_k_perfect(self):
        """Test precision when all retrieved are relevant."""
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "b", "c", "d", "e"}
        
        assert precision_at_k(retrieved, relevant, 5) == 1.0
        assert precision_at_k(retrieved, relevant, 3) == 1.0
    
    def test_precision_at_k_partial(self):
        """Test precision with partially relevant results."""
        retrieved = ["a", "b", "x", "y", "z"]
        relevant = {"a", "b", "c"}
        
        # 2 out of 5 are relevant
        assert precision_at_k(retrieved, relevant, 5) == 0.4
        # 2 out of 3 are relevant
        assert precision_at_k(retrieved, relevant, 3) == pytest.approx(0.666, rel=0.01)
    
    def test_precision_at_k_none(self):
        """Test precision when no results are relevant."""
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b", "c"}
        
        assert precision_at_k(retrieved, relevant, 3) == 0.0
    
    def test_recall_at_k_perfect(self):
        """Test recall when all relevant are retrieved."""
        retrieved = ["a", "b", "c", "x", "y"]
        relevant = {"a", "b", "c"}
        
        assert recall_at_k(retrieved, relevant, 5) == 1.0
    
    def test_recall_at_k_partial(self):
        """Test recall with partial coverage."""
        retrieved = ["a", "b", "x", "y", "z"]
        relevant = {"a", "b", "c", "d"}
        
        # Retrieved 2 out of 4 relevant
        assert recall_at_k(retrieved, relevant, 5) == 0.5
    
    def test_recall_at_k_empty_relevant(self):
        """Test recall with no relevant items."""
        retrieved = ["a", "b", "c"]
        relevant = set()
        
        assert recall_at_k(retrieved, relevant, 3) == 0.0
    
    def test_mrr_first_position(self):
        """Test MRR when first result is relevant."""
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        
        assert mean_reciprocal_rank(retrieved, relevant) == 1.0
    
    def test_mrr_third_position(self):
        """Test MRR when third result is relevant."""
        retrieved = ["x", "y", "a", "b"]
        relevant = {"a", "b"}
        
        # First relevant at position 3
        assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(0.333, rel=0.01)
    
    def test_mrr_no_relevant(self):
        """Test MRR with no relevant results."""
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        
        assert mean_reciprocal_rank(retrieved, relevant) == 0.0
    
    def test_hit_rate_hit(self):
        """Test hit rate when there is a hit."""
        retrieved = ["x", "y", "a"]
        relevant = {"a"}
        
        assert hit_rate(retrieved, relevant) == 1.0
    
    def test_hit_rate_miss(self):
        """Test hit rate when there is no hit."""
        retrieved = ["x", "y", "z"]
        relevant = {"a"}
        
        assert hit_rate(retrieved, relevant) == 0.0
    
    def test_hit_rate_with_k(self):
        """Test hit rate with K limit."""
        retrieved = ["x", "y", "a", "b", "c"]
        relevant = {"a"}
        
        # a is at position 3, so K=2 should miss, K=3 should hit
        assert hit_rate(retrieved, relevant, k=2) == 0.0
        assert hit_rate(retrieved, relevant, k=3) == 1.0
    
    def test_ndcg_perfect(self):
        """Test NDCG with perfect ranking."""
        retrieved = ["a", "b", "c", "x", "y"]
        relevant = {"a", "b", "c"}
        
        # All relevant items first = NDCG 1.0
        assert ndcg_at_k(retrieved, relevant, 5) == 1.0
    
    def test_ndcg_suboptimal(self):
        """Test NDCG with suboptimal ranking."""
        retrieved = ["x", "a", "b", "c", "y"]
        relevant = {"a", "b", "c"}
        
        # Relevant items not in optimal order
        result = ndcg_at_k(retrieved, relevant, 5)
        assert 0 < result < 1
    
    def test_evaluate_retrieval_combined(self):
        """Test combined retrieval evaluation."""
        retrieved = ["a", "b", "x", "c", "y"]
        relevant = {"a", "b", "c", "d"}
        
        result = evaluate_retrieval(retrieved, relevant, k=5)
        
        assert result.precision_at_k == 0.6  # 3/5
        assert result.recall_at_k == 0.75    # 3/4
        assert result.mrr == 1.0             # First is relevant
        assert result.hit_rate == 1.0        # At least one hit
        assert result.k == 5
        assert result.num_relevant_retrieved == 3


class TestGenerationMetrics:
    """Test generation metrics."""
    
    def test_accuracy_exact_match(self):
        """Test exact match accuracy."""
        metrics = GenerationMetrics()
        
        assert metrics.accuracy("Hello World", "Hello World", method="exact") == 1.0
        assert metrics.accuracy("Hello World", "hello world", method="exact") == 1.0
        assert metrics.accuracy("Hello", "World", method="exact") == 0.0
    
    def test_accuracy_fuzzy(self):
        """Test fuzzy accuracy (Jaccard)."""
        metrics = GenerationMetrics()
        
        # Same tokens
        result = metrics.accuracy(
            "RAG combines retrieval and generation",
            "RAG combines retrieval and generation",
            method="fuzzy"
        )
        assert result == 1.0
        
        # Partial overlap
        result = metrics.accuracy(
            "RAG uses retrieval for generation",
            "RAG combines retrieval and generation",
            method="fuzzy"
        )
        assert 0 < result < 1
    
    def test_keyword_overlap_full(self):
        """Test keyword overlap with full match."""
        metrics = GenerationMetrics()
        
        answer = "RAG uses retrieval to find relevant documents and generates answers."
        keywords = ["retrieval", "documents", "generates"]
        
        result = metrics.keyword_overlap(answer, keywords)
        assert result == 1.0
    
    def test_keyword_overlap_partial(self):
        """Test keyword overlap with partial match."""
        metrics = GenerationMetrics()
        
        answer = "RAG uses retrieval to find documents."
        keywords = ["retrieval", "documents", "embedding", "vector"]
        
        result = metrics.keyword_overlap(answer, keywords)
        assert result == 0.5  # 2 out of 4
    
    def test_keyword_overlap_empty_keywords(self):
        """Test keyword overlap with no keywords."""
        metrics = GenerationMetrics()
        
        result = metrics.keyword_overlap("Some answer", [])
        assert result == 0.0


class TestCalibrationError:
    """Test calibration error calculation."""
    
    def test_calibration_perfect(self):
        """Test calibration with perfectly calibrated model."""
        # 80% confident and 80% correct
        confidences = [0.8] * 10
        correctness = [True, True, True, True, True, True, True, True, False, False]
        
        result = calibration_error(confidences, correctness)
        assert result["ece"] == pytest.approx(0.0, abs=0.01)
    
    def test_calibration_overconfident(self):
        """Test calibration with overconfident model."""
        # 90% confident but 50% correct
        confidences = [0.9] * 10
        correctness = [True, True, True, True, True, False, False, False, False, False]
        
        result = calibration_error(confidences, correctness)
        # ECE should be around 0.4 (0.9 - 0.5)
        assert result["ece"] > 0.3
    
    def test_calibration_empty(self):
        """Test calibration with empty inputs."""
        result = calibration_error([], [])
        assert result["ece"] == 0.0
        assert result["num_samples"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
