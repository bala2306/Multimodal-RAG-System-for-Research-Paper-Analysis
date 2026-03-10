"""Error Analysis Module for RAG Pipeline Evaluation.

Analyzes evaluation results to identify failure patterns and generate
qualitative insights for the research report.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from loguru import logger


@dataclass
class ErrorCategory:
    """Categorization of an error."""
    category: str
    subcategory: str
    description: str
    count: int = 0
    examples: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.examples is None:
            self.examples = []


class ErrorAnalyzer:
    """Analyzes RAG evaluation results for error patterns."""
    
    # Error category definitions
    ERROR_CATEGORIES = {
        "retrieval_failure": {
            "description": "System failed to retrieve relevant documents",
            "indicators": ["no relevant information", "couldn't find", "not found in"]
        },
        "generation_hallucination": {
            "description": "Generated answer contains information not in context",
            "indicators": []  # Detected via low faithfulness + high relevancy
        },
        "answer_incomplete": {
            "description": "Answer is correct but missing key details",
            "indicators": []  # Detected via high relevancy + low keyword match
        },
        "wrong_focus": {
            "description": "Answer addresses wrong aspect of question",
            "indicators": []  # Detected via low relevancy
        },
        "api_error": {
            "description": "System error during processing",
            "indicators": ["HTTP 500", "error", "failed"]
        },
        "verbose_correct": {
            "description": "Answer is correct but overly verbose (penalized by Jaccard)",
            "indicators": []  # Detected via high relevancy + low accuracy
        }
    }
    
    def __init__(self, results_path: Optional[str] = None):
        """
        Initialize error analyzer.
        
        Args:
            results_path: Path to evaluation results JSON
        """
        self.results_path = results_path
        self.results = None
        self.error_analysis = {}
        
    def load_results(self, path: str = None) -> Dict[str, Any]:
        """Load evaluation results from JSON file."""
        path = path or self.results_path
        if not path:
            # Find most recent results file
            results_dir = Path("evaluation/results")
            result_files = sorted(results_dir.glob("evaluation_*.json"), reverse=True)
            if not result_files:
                raise FileNotFoundError("No evaluation results found")
            path = result_files[0]
            
        with open(path, "r") as f:
            self.results = json.load(f)
        
        logger.info(f"Loaded results from {path}")
        return self.results
    
    def analyze(self) -> Dict[str, Any]:
        """
        Run full error analysis on loaded results.
        
        Returns:
            Dictionary with error analysis by method
        """
        if not self.results:
            self.load_results()
            
        detailed_results = self.results.get("detailed_results", [])
        
        # Group by method
        by_method = {}
        for result in detailed_results:
            method = result["method"]
            if method not in by_method:
                by_method[method] = []
            by_method[method].append(result)
        
        # Analyze each method
        analysis = {}
        for method, results in by_method.items():
            analysis[method] = self._analyze_method(method, results)
        
        self.error_analysis = analysis
        return analysis
    
    def _analyze_method(self, method: str, results: List[Dict]) -> Dict[str, Any]:
        """Analyze errors for a specific method."""
        total = len(results)
        successful = [r for r in results if not r.get("error")]
        failed = [r for r in results if r.get("error")]
        
        # Categorize errors
        categories = {
            "api_errors": [],
            "low_faithfulness": [],
            "low_relevancy": [],
            "low_keyword_match": [],
            "verbose_correct": [],
            "retrieval_issues": []
        }
        
        for result in results:
            if result.get("error"):
                categories["api_errors"].append({
                    "question_id": result["question_id"],
                    "question": result["question"][:100],
                    "error": result["error"]
                })
                continue
                
            # Analyze metric patterns
            faithfulness = result.get("faithfulness")
            relevancy = result.get("answer_relevancy")
            accuracy = result.get("accuracy", 0)
            keyword = result.get("keyword_overlap", 0)
            
            # Low faithfulness = potential hallucination
            if faithfulness is not None and faithfulness < 0.5:
                categories["low_faithfulness"].append({
                    "question_id": result["question_id"],
                    "question": result["question"][:80],
                    "expected": result["expected_answer"][:80],
                    "generated": result["generated_answer"][:150],
                    "faithfulness": faithfulness,
                    "issue": "Answer may not be grounded in retrieved context"
                })
            
            # Low relevancy = wrong focus
            if relevancy is not None and relevancy < 0.5:
                categories["low_relevancy"].append({
                    "question_id": result["question_id"],
                    "question": result["question"][:80],
                    "generated": result["generated_answer"][:150],
                    "relevancy": relevancy,
                    "issue": "Answer may not address the question directly"
                })
            
            # Low keyword match
            if keyword < 0.3:
                categories["low_keyword_match"].append({
                    "question_id": result["question_id"],
                    "question": result["question"][:80],
                    "expected_keywords": result.get("expected_answer", "")[:50],
                    "generated": result["generated_answer"][:100],
                    "keyword_overlap": keyword,
                    "issue": "Missing expected key terms"
                })
            
            # Verbose but correct (high relevancy, low accuracy)
            if relevancy is not None and relevancy >= 0.7 and accuracy < 0.3:
                categories["verbose_correct"].append({
                    "question_id": result["question_id"],
                    "question": result["question"][:80],
                    "accuracy": accuracy,
                    "relevancy": relevancy,
                    "issue": "Answer is relevant but verbose (Jaccard penalized)"
                })
        
        # Calculate statistics
        successful_results = [r for r in results if not r.get("error")]
        
        return {
            "total_queries": total,
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / total if total > 0 else 0,
            "error_categories": {
                cat: {
                    "count": len(items),
                    "percentage": len(items) / total * 100 if total > 0 else 0,
                    "examples": items[:3]  # Top 3 examples
                }
                for cat, items in categories.items() if items
            },
            "recommendations": self._generate_recommendations(categories, method)
        }
    
    def _generate_recommendations(self, categories: Dict, method: str) -> List[str]:
        """Generate actionable recommendations based on error patterns."""
        recommendations = []
        
        if categories["api_errors"]:
            recommendations.append(
                "Fix API errors - check rate limits, server stability, and error handling"
            )
        
        if categories["low_faithfulness"]:
            n = len(categories["low_faithfulness"])
            recommendations.append(
                f"Improve grounding: {n} answers showed low faithfulness. "
                "Consider: (1) retrieving more documents, (2) improving chunk quality, "
                "(3) adding source attribution prompts"
            )
        
        if categories["low_relevancy"]:
            n = len(categories["low_relevancy"])
            recommendations.append(
                f"Improve relevancy: {n} answers didn't address the question. "
                "Consider: (1) better query understanding, (2) re-ranking retrieved chunks, "
                "(3) question-focused generation prompts"
            )
        
        if categories["low_keyword_match"]:
            n = len(categories["low_keyword_match"])
            recommendations.append(
                f"Key terms missing in {n} answers. "
                "Consider: (1) extractive + abstractive hybrid, (2) keyword-aware prompting"
            )
        
        if categories["verbose_correct"]:
            n = len(categories["verbose_correct"])
            recommendations.append(
                f"{n} answers were verbose but correct (penalized by Jaccard). "
                "This is a metric issue, not a system issue - consider semantic accuracy instead"
            )
        
        if not recommendations:
            recommendations.append("No major error patterns detected")
        
        return recommendations
    
    def generate_report(self, output_path: str = None) -> str:
        """Generate a markdown error analysis report."""
        if not self.error_analysis:
            self.analyze()
        
        report_lines = [
            "# RAG Pipeline Error Analysis Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nSource: {self.results_path or 'Most recent evaluation'}",
            "\n---\n"
        ]
        
        for method, analysis in self.error_analysis.items():
            report_lines.append(f"## {method.upper().replace('_', ' ')}")
            report_lines.append(f"\n**Success Rate:** {analysis['success_rate']:.1%} "
                              f"({analysis['successful']}/{analysis['total_queries']} queries)")
            
            # Error categories
            if analysis["error_categories"]:
                report_lines.append("\n### Error Breakdown\n")
                
                for cat_name, cat_data in analysis["error_categories"].items():
                    report_lines.append(f"#### {cat_name.replace('_', ' ').title()} "
                                      f"({cat_data['count']} cases, {cat_data['percentage']:.1f}%)")
                    
                    if cat_data["examples"]:
                        report_lines.append("\n**Examples:**\n")
                        for i, ex in enumerate(cat_data["examples"][:2], 1):
                            report_lines.append(f"{i}. Q: *{ex.get('question', 'N/A')}*")
                            if "generated" in ex:
                                report_lines.append(f"   - Generated: {ex['generated'][:100]}...")
                            if "issue" in ex:
                                report_lines.append(f"   - Issue: {ex['issue']}")
                            report_lines.append("")
            
            # Recommendations
            report_lines.append("\n### Recommendations\n")
            for rec in analysis["recommendations"]:
                report_lines.append(f"- {rec}")
            
            report_lines.append("\n---\n")
        
        report = "\n".join(report_lines)
        
        # Save report
        if output_path is None:
            output_path = f"evaluation/results/error_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        
        logger.success(f"Error analysis report saved to: {output_path}")
        return report
    
    def print_summary(self):
        """Print error analysis summary to console."""
        if not self.error_analysis:
            self.analyze()
        
        print("\n" + "=" * 60)
        print("ERROR ANALYSIS SUMMARY")
        print("=" * 60)
        
        for method, analysis in self.error_analysis.items():
            print(f"\n{method.upper()}")
            print("-" * 40)
            print(f"  Success Rate: {analysis['success_rate']:.1%}")
            
            if analysis["error_categories"]:
                print("\n  Error Categories:")
                for cat, data in analysis["error_categories"].items():
                    print(f"    - {cat}: {data['count']} ({data['percentage']:.1f}%)")
            
            print("\n  Recommendations:")
            for rec in analysis["recommendations"][:3]:
                print(f"    → {rec[:80]}{'...' if len(rec) > 80 else ''}")
        
        print("\n" + "=" * 60)


def main():
    """Run error analysis on most recent evaluation results."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze RAG evaluation errors")
    parser.add_argument(
        "--results", "-r",
        help="Path to evaluation results JSON"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output path for error analysis report"
    )
    
    args = parser.parse_args()
    
    analyzer = ErrorAnalyzer(args.results)
    analyzer.load_results()
    analyzer.analyze()
    analyzer.print_summary()
    analyzer.generate_report(args.output)


if __name__ == "__main__":
    main()
