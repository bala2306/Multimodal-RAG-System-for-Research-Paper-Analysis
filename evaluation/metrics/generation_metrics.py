"""Generation evaluation metrics for RAG systems.

Implements Accuracy, Faithfulness, Answer Relevancy, and Calibration metrics.
Uses LLM-as-Judge approach for subjective metrics.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from loguru import logger
import re

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.config import settings

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


@dataclass
class GenerationResult:
    """Container for generation evaluation results."""
    accuracy: float
    faithfulness: Optional[float]
    answer_relevancy: Optional[float]
    keyword_overlap: float
    answer_length: int


class GenerationMetrics:
    """Metrics for evaluating generated answers."""
    
    def __init__(self, model: Optional[str] = None):
        """
        Initialize generation metrics with LLM for judging.
        
        Args:
            model: Groq model for LLM-as-judge (defaults to settings)
        """
        self.llm_available = GROQ_AVAILABLE and settings.groq_api_key
        if self.llm_available:
            self.client = Groq(api_key=settings.groq_api_key)
            self.model = model or settings.default_llm_model
    
    def accuracy(
        self,
        predicted: str,
        expected: str,
        method: str = "fuzzy"
    ) -> float:
        """
        Calculate accuracy between predicted and expected answers.
        
        Args:
            predicted: Generated answer
            expected: Ground truth answer
            method: "exact", "fuzzy", or "semantic"
            
        Returns:
            Accuracy score between 0 and 1
        """
        if method == "exact":
            return 1.0 if predicted.strip().lower() == expected.strip().lower() else 0.0
        
        elif method == "fuzzy":
            # Token overlap based similarity
            pred_tokens = set(predicted.lower().split())
            exp_tokens = set(expected.lower().split())
            
            if not exp_tokens:
                return 0.0
            
            intersection = pred_tokens & exp_tokens
            union = pred_tokens | exp_tokens
            
            # Jaccard similarity
            return len(intersection) / len(union) if union else 0.0
        
        elif method == "semantic":
            # Use LLM to judge semantic similarity
            if not self.llm_available:
                return self.accuracy(predicted, expected, method="fuzzy")
            
            return self._llm_judge_accuracy(predicted, expected)
        
        else:
            raise ValueError(f"Unknown accuracy method: {method}")
    
    def _llm_judge_accuracy(self, predicted: str, expected: str) -> float:
        """Use LLM to judge semantic accuracy."""
        prompt = f"""Compare the following two answers and rate their semantic similarity on a scale of 0 to 1.

Expected Answer: {expected}

Predicted Answer: {predicted}

Instructions:
- Score 1.0 if they convey the same meaning
- Score 0.5 if they partially overlap
- Score 0.0 if they are completely different
- Only return a single number between 0 and 1

Score:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(re.search(r"[\d.]+", score_text).group())
            return min(max(score, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"LLM accuracy judge failed: {e}")
            return self.accuracy(predicted, expected, method="fuzzy")
    
    def faithfulness(
        self,
        answer: str,
        context: str
    ) -> float:
        """
        Evaluate faithfulness - is the answer grounded in the context?
        
        Uses LLM-as-judge to check if claims in the answer are supported.
        
        Args:
            answer: Generated answer
            context: Retrieved context chunks
            
        Returns:
            Faithfulness score between 0 and 1
        """
        if not self.llm_available:
            logger.warning("LLM not available for faithfulness evaluation")
            return None
        
        prompt = f"""Rate faithfulness on a scale of 0 to 1.

Context: {context[:2000]}

Answer: {answer[:500]}

Score 1.0 = fully faithful, 0.5 = partial, 0.0 = unfaithful.
Respond with ONLY a number like 0.8 or 0.5, nothing else.

Score:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            match = re.search(r"[\d.]+", score_text)
            if match:
                score = float(match.group())
                return min(max(score, 0.0), 1.0)
            else:
                logger.warning(f"Could not parse faithfulness score from: {score_text}")
                return 0.5  # Default to middle score
            
        except Exception as e:
            logger.error(f"Faithfulness evaluation failed: {e}")
            return None
    
    def answer_relevancy(
        self,
        answer: str,
        question: str
    ) -> float:
        """
        Evaluate answer relevancy - does the answer address the question?
        
        Args:
            answer: Generated answer
            question: Original question
            
        Returns:
            Relevancy score between 0 and 1
        """
        if not self.llm_available:
            logger.warning("LLM not available for relevancy evaluation")
            return None
        
        prompt = f"""Rate answer relevancy on a scale of 0 to 1.

Question: {question}

Answer: {answer[:500]}

Score 1.0 = directly addresses question, 0.5 = partial, 0.0 = off-topic.
Respond with ONLY a number like 0.8 or 0.5, nothing else.

Score:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            match = re.search(r"[\d.]+", score_text)
            if match:
                score = float(match.group())
                return min(max(score, 0.0), 1.0)
            else:
                logger.warning(f"Could not parse relevancy score from: {score_text}")
                return 0.5  # Default to middle score
            
        except Exception as e:
            logger.error(f"Relevancy evaluation failed: {e}")
            return None
    
    def keyword_overlap(
        self,
        answer: str,
        keywords: List[str]
    ) -> float:
        """
        Calculate keyword overlap between answer and expected keywords.
        
        Args:
            answer: Generated answer
            keywords: List of expected keywords
            
        Returns:
            Overlap score between 0 and 1
        """
        if not keywords:
            return 0.0
        
        answer_lower = answer.lower()
        found = sum(1 for kw in keywords if kw.lower() in answer_lower)
        
        return found / len(keywords)
    
    def evaluate(
        self,
        predicted: str,
        expected: str,
        question: str,
        context: Optional[str] = None,
        keywords: Optional[List[str]] = None
    ) -> GenerationResult:
        """
        Run all generation metrics.
        
        Args:
            predicted: Generated answer
            expected: Ground truth answer
            question: Original question
            context: Retrieved context (optional, for faithfulness)
            keywords: Expected keywords (optional)
            
        Returns:
            GenerationResult with all metrics
        """
        # Calculate accuracy
        accuracy_score = self.accuracy(predicted, expected, method="fuzzy")
        
        # Calculate faithfulness if context provided
        faithfulness_score = None
        if context and self.llm_available:
            faithfulness_score = self.faithfulness(predicted, context)
        
        # Calculate answer relevancy
        relevancy_score = None
        if self.llm_available:
            relevancy_score = self.answer_relevancy(predicted, question)
        
        # Calculate keyword overlap
        kw_score = self.keyword_overlap(predicted, keywords or [])
        
        return GenerationResult(
            accuracy=accuracy_score,
            faithfulness=faithfulness_score,
            answer_relevancy=relevancy_score,
            keyword_overlap=kw_score,
            answer_length=len(predicted)
        )


def calibration_error(
    confidences: List[float],
    correctness: List[bool],
    num_bins: int = 10
) -> Dict[str, float]:
    """
    Calculate Expected Calibration Error (ECE).
    
    Measures gap between model confidence and actual accuracy.
    
    Args:
        confidences: List of model confidence scores (0-1)
        correctness: List of whether predictions were correct
        num_bins: Number of bins for calibration
        
    Returns:
        Dictionary with ECE and per-bin stats
    """
    if len(confidences) != len(correctness):
        raise ValueError("Confidences and correctness must have same length")
    
    if not confidences:
        return {"ece": 0.0, "num_samples": 0}
    
    # Create bins
    bin_boundaries = [i / num_bins for i in range(num_bins + 1)]
    bin_correct = [0] * num_bins
    bin_conf = [0.0] * num_bins
    bin_count = [0] * num_bins
    
    for conf, correct in zip(confidences, correctness):
        # Find bin
        bin_idx = min(int(conf * num_bins), num_bins - 1)
        bin_count[bin_idx] += 1
        bin_correct[bin_idx] += int(correct)
        bin_conf[bin_idx] += conf
    
    # Calculate ECE
    ece = 0.0
    n = len(confidences)
    
    for i in range(num_bins):
        if bin_count[i] > 0:
            avg_accuracy = bin_correct[i] / bin_count[i]
            avg_confidence = bin_conf[i] / bin_count[i]
            ece += (bin_count[i] / n) * abs(avg_accuracy - avg_confidence)
    
    return {
        "ece": ece,
        "num_samples": len(confidences),
        "mean_confidence": sum(confidences) / len(confidences),
        "mean_accuracy": sum(correctness) / len(correctness)
    }


def aggregate_generation_metrics(results: List[GenerationResult]) -> Dict[str, float]:
    """
    Aggregate generation metrics across multiple queries.
    
    Args:
        results: List of GenerationResult objects
        
    Returns:
        Dictionary with mean metrics
    """
    if not results:
        return {}
    
    n = len(results)
    
    # Filter None values for optional metrics
    faithfulness_scores = [r.faithfulness for r in results if r.faithfulness is not None]
    relevancy_scores = [r.answer_relevancy for r in results if r.answer_relevancy is not None]
    
    metrics = {
        "mean_accuracy": sum(r.accuracy for r in results) / n,
        "mean_keyword_overlap": sum(r.keyword_overlap for r in results) / n,
        "mean_answer_length": sum(r.answer_length for r in results) / n,
        "num_queries": n
    }
    
    if faithfulness_scores:
        metrics["mean_faithfulness"] = sum(faithfulness_scores) / len(faithfulness_scores)
    
    if relevancy_scores:
        metrics["mean_answer_relevancy"] = sum(relevancy_scores) / len(relevancy_scores)
    
    return metrics
