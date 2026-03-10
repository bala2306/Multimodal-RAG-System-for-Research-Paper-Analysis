"""No-RAG baseline for evaluation.

This baseline sends questions directly to the LLM without any document retrieval,
to measure the value added by the RAG system.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.config import settings

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class NoRAGBaseline:
    """Baseline that answers questions using only LLM knowledge, no retrieval."""
    
    def __init__(self, model: Optional[str] = None):
        """
        Initialize No-RAG baseline.
        
        Args:
            model: Groq model name (defaults to settings.default_llm_model)
        """
        if not GROQ_AVAILABLE:
            raise RuntimeError("Groq not installed. Install with: pip install groq")
        
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY not set in environment")
        
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = model or settings.default_llm_model
        logger.info(f"No-RAG baseline initialized with model: {self.model}")
    
    def answer(
        self,
        question: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        include_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a question using only LLM knowledge (no retrieval).
        
        Args:
            question: The question to answer
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            include_confidence: Whether to ask model for confidence score
            
        Returns:
            Dictionary with answer and metadata
        """
        if include_confidence:
            prompt = f"""Answer the following question based on your knowledge.

Question: {question}

Instructions:
- Provide a clear, concise answer
- At the end of your answer, on a new line, provide your confidence as a percentage (0-100%)
- Format: "Confidence: X%"

Answer:"""
        else:
            prompt = f"""Answer the following question based on your knowledge.

Question: {question}

Instructions:
- Provide a clear, concise answer
- Be accurate and informative

Answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a knowledgeable assistant answering questions accurately and concisely."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            answer_text = response.choices[0].message.content
            
            # Parse confidence if requested
            confidence = None
            if include_confidence and "Confidence:" in answer_text:
                try:
                    conf_line = answer_text.split("Confidence:")[-1].strip()
                    conf_value = conf_line.replace("%", "").strip()
                    confidence = float(conf_value) / 100.0
                    # Remove confidence line from answer
                    answer_text = answer_text.rsplit("Confidence:", 1)[0].strip()
                except (ValueError, IndexError):
                    pass
            
            return {
                "answer": answer_text,
                "confidence": confidence,
                "model": self.model,
                "method": "no_rag_baseline",
                "retrieval_used": False,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"No-RAG baseline failed: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "confidence": None,
                "model": self.model,
                "method": "no_rag_baseline",
                "retrieval_used": False,
                "error": str(e)
            }
    
    def batch_answer(
        self,
        questions: list,
        **kwargs
    ) -> list:
        """
        Answer multiple questions.
        
        Args:
            questions: List of questions
            **kwargs: Arguments passed to answer()
            
        Returns:
            List of answer dictionaries
        """
        results = []
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}")
            result = self.answer(question, **kwargs)
            result["question_index"] = i
            results.append(result)
        return results


# Convenience function
def get_baseline() -> NoRAGBaseline:
    """Get a No-RAG baseline instance."""
    return NoRAGBaseline()


if __name__ == "__main__":
    # Quick test
    baseline = NoRAGBaseline()
    result = baseline.answer("What is retrieval-augmented generation?")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']}")
