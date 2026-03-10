"""Query expansion service for improving retrieval recall."""

import re
from typing import List, Dict
from loguru import logger


class QueryExpander:
    """Expands queries with synonyms, acronyms, and related terms."""

    # Common NLP/ML acronym expansions
    ACRONYM_MAP = {
        # Transformer models
        "BERT": "Bidirectional Encoder Representations from Transformers",
        "GPT": "Generative Pre-trained Transformer",
        "ELMo": "Embeddings from Language Models",
        "RoBERTa": "Robustly Optimized BERT Pretraining Approach",
        "ALBERT": "A Lite BERT",
        "DistilBERT": "Distilled BERT",
        "ELECTRA": "Efficiently Learning an Encoder that Classifies Token Replacements Accurately",

        # Retrieval models
        "DPR": "Dense Passage Retrieval",
        "ColBERT": "Contextualized Late Interaction over BERT",
        "SBERT": "Sentence-BERT",
        "RAG": "Retrieval-Augmented Generation",

        # Embeddings
        "GloVe": "Global Vectors for Word Representation",
        "Word2Vec": "Word to Vector",

        # Architectures
        "CNN": "Convolutional Neural Network",
        "RNN": "Recurrent Neural Network",
        "LSTM": "Long Short-Term Memory",
        "GRU": "Gated Recurrent Unit",
        "BiLSTM": "Bidirectional Long Short-Term Memory",

        # Training methods
        "MLM": "Masked Language Modeling",
        "NSP": "Next Sentence Prediction",
        "SOP": "Sentence Order Prediction",
        "CoT": "Chain of Thought",

        # Metrics
        "BLEU": "Bilingual Evaluation Understudy",
        "ROUGE": "Recall-Oriented Understudy for Gisting Evaluation",
        "RAGAS": "Retrieval-Augmented Generation Assessment",

        # Other
        "NLP": "Natural Language Processing",
        "LLM": "Large Language Model",
        "MLM": "Masked Language Model",
        "SLM": "Statistical Language Model",
        "NER": "Named Entity Recognition",
        "POS": "Part of Speech",
    }

    def __init__(self):
        """Initialize query expander."""
        # Build reverse mapping (expansion -> acronym)
        self.expansion_to_acronym = {
            v.lower(): k for k, v in self.ACRONYM_MAP.items()
        }

    def detect_acronyms(self, text: str) -> List[str]:
        """
        Detect potential acronyms in text.

        Args:
            text: Input text

        Returns:
            List of detected acronyms
        """
        # Match sequences of 2+ uppercase letters possibly with digits
        pattern = r'\b[A-Z][A-Z0-9]{1,}\b'
        acronyms = re.findall(pattern, text)
        return list(set(acronyms))

    def expand_query(self, query: str, max_expansions: int = 3) -> str:
        """
        Expand query with acronym expansions and variations.

        Args:
            query: Original query
            max_expansions: Maximum number of expansions to add

        Returns:
            Expanded query
        """
        # Detect acronyms in query
        acronyms = self.detect_acronyms(query)

        expansions = []
        for acronym in acronyms[:max_expansions]:
            if acronym in self.ACRONYM_MAP:
                expansion = self.ACRONYM_MAP[acronym]
                expansions.append(expansion)
                logger.debug(f"Expanding {acronym} -> {expansion}")

        # Combine original query with expansions
        if expansions:
            expanded = f"{query} {' '.join(expansions)}"
            logger.info(f"Query expanded: '{query}' -> '{expanded[:100]}...'")
            return expanded

        return query

    def generate_query_variations(self, query: str) -> List[str]:
        """
        Generate multiple variations of a query for comprehensive search.

        Args:
            query: Original query

        Returns:
            List of query variations
        """
        variations = [query]

        # Add expanded version
        expanded = self.expand_query(query)
        if expanded != query:
            variations.append(expanded)
        
        stand_for_match = re.search(r'what does (\w+) stand for', query, re.IGNORECASE)
        if stand_for_match:
            acronym = stand_for_match.group(1).upper()
            if acronym in self.ACRONYM_MAP:
                expansion = self.ACRONYM_MAP[acronym]
                variations.append(f"What is {expansion}?")
                variations.append(f"{acronym} {expansion}")

        if re.search(r'key (innovation|contribution|idea)', query, re.IGNORECASE):
            subject_match = re.search(r'(of|in) (\w+)', query, re.IGNORECASE)
            if subject_match:
                subject = subject_match.group(2)
                variations.append(f"{subject} innovation")
                variations.append(f"{subject} contribution")

        logger.debug(f"Generated {len(variations)} query variations")
        return list(set(variations))  # Remove duplicates

    def add_custom_acronym(self, acronym: str, expansion: str):
        """
        Add custom acronym mapping.

        Args:
            acronym: Acronym (e.g., "BERT")
            expansion: Full expansion
        """
        self.ACRONYM_MAP[acronym] = expansion
        self.expansion_to_acronym[expansion.lower()] = acronym
        logger.info(f"Added custom acronym: {acronym} -> {expansion}")
