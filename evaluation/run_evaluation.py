"""Main evaluation runner for RAG Pipeline.

Runs evaluation comparing No-RAG baseline vs Basic RAG vs Advanced RAG.
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.baselines.no_rag_baseline import NoRAGBaseline
from evaluation.metrics.retrieval_metrics import evaluate_retrieval, aggregate_retrieval_metrics
from evaluation.metrics.generation_metrics import (
    GenerationMetrics, 
    GenerationResult,
    aggregate_generation_metrics,
    calibration_error
)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation run."""
    test_dataset_path: str = "evaluation/datasets/test_queries.json"
    output_dir: str = "evaluation/results"
    top_k: int = 10
    limit: Optional[int] = None  # Limit number of questions (for testing)
    methods: List[str] = None  # Which methods to evaluate
    
    def __post_init__(self):
        if self.methods is None:
            self.methods = ["no_rag", "basic_rag", "advanced_rag"]


@dataclass
class QueryResult:
    """Results for a single query."""
    question_id: str
    question: str
    expected_answer: str
    method: str
    generated_answer: str
    accuracy: float
    faithfulness: Optional[float]
    answer_relevancy: Optional[float]
    keyword_overlap: float
    confidence: Optional[float]
    latency_ms: float
    error: Optional[str] = None


class RAGEvaluator:
    """Main evaluator for RAG pipeline."""
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.results: List[QueryResult] = []
        self.generation_metrics = GenerationMetrics()
        
        # Load test dataset
        self.dataset = self._load_dataset()
        logger.info(f"Loaded {len(self.dataset['questions'])} questions")
    
    def _load_dataset(self) -> Dict[str, Any]:
        """Load test dataset from JSON."""
        dataset_path = Path(self.config.test_dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        with open(dataset_path, "r") as f:
            return json.load(f)
    
    def _get_no_rag_answer(self, question: str) -> Dict[str, Any]:
        """Get answer from No-RAG baseline."""
        try:
            baseline = NoRAGBaseline()
            start_time = time.time()
            result = baseline.answer(question, include_confidence=True)
            latency = (time.time() - start_time) * 1000
            
            return {
                "answer": result["answer"],
                "confidence": result.get("confidence"),
                "latency_ms": latency,
                "context": None,
                "error": result.get("error")
            }
        except Exception as e:
            logger.error(f"No-RAG baseline failed: {e}")
            return {
                "answer": "",
                "confidence": None,
                "latency_ms": 0,
                "context": None,
                "error": str(e)
            }
    
    def _get_basic_rag_answer(self, question: str) -> Dict[str, Any]:
        """Get answer from Basic RAG pipeline."""
        try:
            import requests
            
            start_time = time.time()
            response = requests.post(
                "http://localhost:8000/api/v1/basic/query",
                json={"query": question, "top_k": self.config.top_k},
                timeout=60
            )
            latency = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "answer": data.get("answer", ""),
                    "confidence": None,
                    "latency_ms": latency,
                    "context": "\n".join([c.get("text", "") for c in data.get("sources", [])]),
                    "error": None
                }
            else:
                return {
                    "answer": "",
                    "confidence": None,
                    "latency_ms": latency,
                    "context": None,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except requests.exceptions.ConnectionError:
            return {
                "answer": "",
                "confidence": None,
                "latency_ms": 0,
                "context": None,
                "error": "API server not running. Start with: uvicorn app.main:app"
            }
        except Exception as e:
            logger.error(f"Basic RAG query failed: {e}")
            return {
                "answer": "",
                "confidence": None,
                "latency_ms": 0,
                "context": None,
                "error": str(e)
            }
    
    def _get_advanced_rag_answer(self, question: str) -> Dict[str, Any]:
        """Get answer from Advanced RAG pipeline."""
        try:
            import requests
            
            start_time = time.time()
            response = requests.post(
                "http://localhost:8000/api/v1/advanced/query",
                json={"query": question, "top_k": self.config.top_k},
                timeout=90
            )
            latency = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                context_parts = [c.get("text", "") for c in data.get("sources", [])]
                # Include visual element descriptions
                for ve in data.get("visual_elements", []):
                    context_parts.append(ve.get("description", ""))
                
                return {
                    "answer": data.get("answer", ""),
                    "confidence": None,
                    "latency_ms": latency,
                    "context": "\n".join(context_parts),
                    "error": None
                }
            else:
                return {
                    "answer": "",
                    "confidence": None,
                    "latency_ms": latency,
                    "context": None,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except requests.exceptions.ConnectionError:
            return {
                "answer": "",
                "confidence": None,
                "latency_ms": 0,
                "context": None,
                "error": "API server not running. Start with: uvicorn app.main:app"
            }
        except Exception as e:
            logger.error(f"Advanced RAG query failed: {e}")
            return {
                "answer": "",
                "confidence": None,
                "latency_ms": 0,
                "context": None,
                "error": str(e)
            }
    
    def evaluate_question(
        self,
        question_data: Dict[str, Any],
        method: str
    ) -> QueryResult:
        """
        Evaluate a single question with specified method.
        
        Args:
            question_data: Question data from dataset
            method: "no_rag", "basic_rag", or "advanced_rag"
            
        Returns:
            QueryResult with metrics
        """
        question = question_data["question"]
        expected = question_data["expected_answer"]
        keywords = question_data.get("keywords", [])
        
        # Get answer based on method
        if method == "no_rag":
            result = self._get_no_rag_answer(question)
        elif method == "basic_rag":
            result = self._get_basic_rag_answer(question)
        elif method == "advanced_rag":
            result = self._get_advanced_rag_answer(question)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Calculate metrics if we got an answer
        if result["error"]:
            return QueryResult(
                question_id=question_data["id"],
                question=question,
                expected_answer=expected,
                method=method,
                generated_answer="",
                accuracy=0.0,
                faithfulness=None,
                answer_relevancy=None,
                keyword_overlap=0.0,
                confidence=None,
                latency_ms=result["latency_ms"],
                error=result["error"]
            )
        
        # Evaluate generation metrics
        gen_result = self.generation_metrics.evaluate(
            predicted=result["answer"],
            expected=expected,
            question=question,
            context=result["context"],
            keywords=keywords
        )
        
        return QueryResult(
            question_id=question_data["id"],
            question=question,
            expected_answer=expected,
            method=method,
            generated_answer=result["answer"],
            accuracy=gen_result.accuracy,
            faithfulness=gen_result.faithfulness,
            answer_relevancy=gen_result.answer_relevancy,
            keyword_overlap=gen_result.keyword_overlap,
            confidence=result["confidence"],
            latency_ms=result["latency_ms"]
        )
    
    def run(self) -> Dict[str, Any]:
        """
        Run full evaluation.
        
        Returns:
            Dictionary with all results and aggregated metrics
        """
        questions = self.dataset["questions"]
        
        # Apply limit if specified
        if self.config.limit:
            questions = questions[:self.config.limit]
        
        logger.info(f"Running evaluation on {len(questions)} questions")
        logger.info(f"Methods: {self.config.methods}")
        
        all_results = []
        
        for method in self.config.methods:
            logger.info(f"\n{'='*50}")
            logger.info(f"Evaluating: {method}")
            logger.info(f"{'='*50}")
            
            method_results = []
            
            for i, q in enumerate(questions):
                logger.info(f"  [{i+1}/{len(questions)}] {q['id']}: {q['question'][:50]}...")
                result = self.evaluate_question(q, method)
                method_results.append(result)
                all_results.append(result)
                
                if result.error:
                    logger.warning(f"    Error: {result.error}")
                else:
                    logger.info(f"    Accuracy: {result.accuracy:.2f}, Relevancy: {result.answer_relevancy}")
        
        # Aggregate results by method
        summary = self._aggregate_results(all_results)
        
        # Save results
        output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_questions": len(questions),
                "methods": self.config.methods,
                "top_k": self.config.top_k
            },
            "summary": summary,
            "detailed_results": [asdict(r) for r in all_results]
        }
        
        self._save_results(output)
        
        return output
    
    def _aggregate_results(self, results: List[QueryResult]) -> Dict[str, Any]:
        """Aggregate results by method."""
        summary = {}
        
        for method in self.config.methods:
            method_results = [r for r in results if r.method == method]
            
            if not method_results:
                continue
            
            # Calculate means
            n = len(method_results)
            valid_results = [r for r in method_results if r.error is None]
            nv = len(valid_results)
            
            if nv == 0:
                summary[method] = {"error": "All queries failed"}
                continue
            
            summary[method] = {
                "num_queries": n,
                "num_successful": nv,
                "success_rate": nv / n,
                "mean_accuracy": sum(r.accuracy for r in valid_results) / nv,
                "mean_keyword_overlap": sum(r.keyword_overlap for r in valid_results) / nv,
                "mean_latency_ms": sum(r.latency_ms for r in valid_results) / nv
            }
            
            # Add optional metrics if available
            faithfulness = [r.faithfulness for r in valid_results if r.faithfulness is not None]
            if faithfulness:
                summary[method]["mean_faithfulness"] = sum(faithfulness) / len(faithfulness)
            
            relevancy = [r.answer_relevancy for r in valid_results if r.answer_relevancy is not None]
            if relevancy:
                summary[method]["mean_answer_relevancy"] = sum(relevancy) / len(relevancy)
            
            # Calibration for No-RAG (has confidence scores)
            if method == "no_rag":
                confidences = [r.confidence for r in valid_results if r.confidence is not None]
                if confidences:
                    # Consider "correct" if accuracy > 0.5
                    correctness = [r.accuracy > 0.5 for r in valid_results if r.confidence is not None]
                    cal = calibration_error(confidences, correctness)
                    summary[method]["calibration_error"] = cal["ece"]
        
        return summary
    
    def _save_results(self, output: Dict[str, Any]):
        """Save results to JSON file."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"evaluation_{timestamp}.json"
        
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.success(f"Results saved to: {output_path}")
        
        # Also save summary separately
        summary_path = output_dir / f"summary_{timestamp}.json"
        with open(summary_path, "w") as f:
            json.dump(output["summary"], f, indent=2)
        
        # Print summary
        self._print_summary(output["summary"])
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print summary to console."""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        for method, metrics in summary.items():
            print(f"\n{method.upper()}")
            print("-" * 40)
            
            if "error" in metrics:
                print(f"  Error: {metrics['error']}")
                continue
            
            print(f"  Success Rate:       {metrics['success_rate']:.1%}")
            
            # Primary RAG metrics (most important)
            if "mean_faithfulness" in metrics:
                print(f"  Mean Faithfulness:  {metrics['mean_faithfulness']:.3f}")
            
            if "mean_answer_relevancy" in metrics:
                print(f"  Mean Relevancy:     {metrics['mean_answer_relevancy']:.3f}")
            
            print(f"  Mean Keyword Match: {metrics['mean_keyword_overlap']:.3f}")
            print(f"  Mean Latency:       {metrics['mean_latency_ms']:.0f}ms")
            
            # Secondary metrics
            print(f"  Mean Accuracy:      {metrics['mean_accuracy']:.3f}  (token overlap - less reliable)")
            
            if "calibration_error" in metrics:
                print(f"  Calibration Error:  {metrics['calibration_error']:.3f}")
        
        print("\n" + "="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run RAG Pipeline Evaluation")
    parser.add_argument(
        "--dataset", "-d",
        default="evaluation/datasets/test_queries.json",
        help="Path to test dataset JSON"
    )
    parser.add_argument(
        "--output", "-o",
        default="evaluation/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Top-K for retrieval"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of questions (for testing)"
    )
    parser.add_argument(
        "--methods", "-m",
        nargs="+",
        default=["no_rag", "basic_rag", "advanced_rag"],
        help="Methods to evaluate"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run on first 3 questions only"
    )
    
    args = parser.parse_args()
    
    # Handle dry run
    if args.dry_run:
        args.limit = 3
        logger.info("Dry run mode: evaluating 3 questions only")
    
    config = EvaluationConfig(
        test_dataset_path=args.dataset,
        output_dir=args.output,
        top_k=args.top_k,
        limit=args.limit,
        methods=args.methods
    )
    
    evaluator = RAGEvaluator(config)
    results = evaluator.run()
    
    return results


if __name__ == "__main__":
    main()
