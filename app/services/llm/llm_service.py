"""LLM service for answer generation using Groq and Gemini."""

from typing import List, Dict, Any, Optional
from loguru import logger
from app.core.config import settings

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq not available")


class LLMService:
    """Service for LLM-based answer generation using Groq."""

    def __init__(self, model: Optional[str] = None):
        """
        Initialize LLM service with Groq.

        Args:
            model: Model name (defaults to settings.default_llm_model)
        """
        # Initialize Groq for text generation
        if not GROQ_AVAILABLE:
            raise RuntimeError(
                "Groq not installed. Install with: pip install groq"
            )

        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY not set in environment")

        self.groq_client = Groq(api_key=settings.groq_api_key)
        self.model_name = model or settings.default_llm_model

        logger.info(f"LLM service initialized: Groq - {self.model_name}")
    
    def generate_answer(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        visual_elements: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Generate answer based on question and retrieved context.
        
        Args:
            question: User question
            context_chunks: Retrieved text chunks
            visual_elements: Retrieved visual elements (tables/images)
            
        Returns:
            Generated answer
        """
        # Build context
        context = self._build_context(context_chunks, visual_elements)
        
        # Create prompt (use multimodal prompt if visual elements are present)
        has_visual_elements = visual_elements is not None and len(visual_elements) > 0
        prompt = self._create_prompt(question, context, has_visual_elements=has_visual_elements)
        
        # Generate answer using Groq
        try:
            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert data analyst and research assistant. "
                            "Answer questions by PRIORITIZING the provided document context (text, tables, figures), "
                            "but you may also use your general knowledge when:\n"
                            "  - The context is incomplete or doesn't fully answer the question\n"
                            "  - The question asks for well-known facts that aren't in the context\n"
                            "  - Additional context from your knowledge enhances understanding\n\n"
                            "When using context: Extract specific values accurately, cite sources with [Source N].\n"
                            "When using general knowledge: Clearly state 'Based on general knowledge' or similar.\n"
                            "Combine both intelligently for comprehensive answers."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
            
            answer = response.choices[0].message.content
            logger.debug(f"Generated answer using Groq ({len(answer)} chars)")
            return answer
            
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            raise
    
    def _build_context(
        self,
        chunks: List[Dict[str, Any]],
        visual_elements: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Build context string from chunks and visual elements."""
        context_parts = []
        
        # Add text chunks
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            page = chunk.get("page", "unknown")
            doc_name = chunk.get("document_name", "document")
            
            context_parts.append(
                f"[Source {i}] (from {doc_name}, page {page}):\n{text}"
            )
        
        # Add visual elements if present
        if visual_elements:
            for i, element in enumerate(visual_elements, len(chunks) + 1):
                element_type = element.get("element_type", "visual")
                description = element.get("description", "")
                page = element.get("page_number", "unknown")
                
                # Add table markdown if available
                if element_type == "table" and element.get("table_markdown"):
                    context_parts.append(
                        f"[Source {i}] ({element_type} on page {page}):\n"
                        f"{element.get('table_markdown')}\n"
                        f"Description: {description}"
                    )
                else:
                    context_parts.append(
                        f"[Source {i}] ({element_type} on page {page}):\n{description}"
                    )
        
        return "\n\n".join(context_parts)
    
    def _create_prompt(self, question: str, context: str, has_visual_elements: bool = False) -> str:
        """Create enhanced prompt for answer generation with multimodal awareness."""
        
        # For multimodal queries (Advanced RAG), use a prompt that emphasizes grounding
        if has_visual_elements:
            prompt = f"""Document Context (including tables, images, charts, and figures):
{context}

Question: {question}

ANSWER INSTRUCTIONS FOR MULTIMODAL CONTENT:

**CRITICAL: GROUND YOUR ANSWER IN THE PROVIDED CONTEXT**
- ONLY use information from the [Source N] entries above
- If quoting tables/figures, cite the exact source: [Source N]
- Answer in 2-4 sentences, being specific about what the sources show

1. TABLE/FIGURE PRIORITY:
   - When sources contain tables or visual descriptions, extract data directly
   - Quote exact values, percentages, or findings from the sources
   - If a table shows the answer, state: "According to [Source N], ..."

2. DO NOT MAKE UP INFORMATION:
   - If the answer is not in the context, say "The provided context does not contain this information"
   - Do not use general knowledge for specific facts

3. CITATION FORMAT:
   - Always cite: [Source N] where information came from
   - Be specific about whether it's from text, table, or figure

Answer based ONLY on the provided sources:"""
        else:
            # For text-only queries (Basic RAG), use concise prompt
            prompt = f"""Document Context:
{context}

Question: {question}

ANSWER INSTRUCTIONS:

**CRITICAL: BE CONCISE AND DIRECT**
- Answer in 1-3 sentences maximum
- Give the specific answer first, then brief supporting context if needed
- Do NOT repeat the question or use preambles like "Based on the context..."
- Do NOT say "I couldn't find" - just answer with what you know

1. CITATION REQUIREMENTS:
   - Cite sources briefly: [Source N]

2. ACCURACY FIRST:
   - Extract EXACT values from the context
   - Don't approximate unless explicitly stated

3. HANDLING INCOMPLETE CONTEXT:
   - If the context doesn't fully answer the question, use your general knowledge
   - Provide direct answers, not explanations of what you couldn't find

Answer CONCISELY and DIRECTLY:"""

        return prompt
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate text from prompt (generic method).
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        try:
            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature or settings.llm_temperature,
                max_tokens=max_tokens or settings.llm_max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            raise
