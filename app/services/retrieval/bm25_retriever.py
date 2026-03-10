"""BM25 retriever for keyword-based search."""

from typing import List, Dict, Any, Optional
from collections import defaultdict
import math
import re
from loguru import logger


class BM25Retriever:
    """BM25 retriever for keyword-based document ranking."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever.

        Args:
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.corpus: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.idf: Dict[str, float] = {}
        self.avg_doc_len: float = 0.0
        self.doc_lens: List[int] = []

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text

        Returns:
            List of lowercase tokens
        """
        # Convert to lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def build_index(self, documents: List[Dict[str, Any]], text_field: str = "text"):
        """
        Build BM25 index from documents.

        Args:
            documents: List of document dictionaries
            text_field: Field name containing text to index
        """
        logger.info(f"Building BM25 index for {len(documents)} documents")

        self.corpus = documents
        self.tokenized_corpus = []
        self.doc_freqs = defaultdict(int)
        self.doc_lens = []

        # Tokenize all documents
        for doc in documents:
            text = doc.get(text_field, "")
            tokens = self.tokenize(text)
            self.tokenized_corpus.append(tokens)
            self.doc_lens.append(len(tokens))

            # Count document frequency for each term
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1

        # Calculate average document length
        self.avg_doc_len = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0

        # Calculate IDF for each term
        num_docs = len(self.corpus)
        for term, freq in self.doc_freqs.items():
            # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
            self.idf[term] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1)

        logger.info(f"BM25 index built: {len(self.idf)} unique terms, avg doc len: {self.avg_doc_len:.1f}")

    def get_scores(self, query: str) -> List[float]:
        """
        Calculate BM25 scores for all documents given a query.

        Args:
            query: Search query

        Returns:
            List of BM25 scores for each document
        """
        query_tokens = self.tokenize(query)
        scores = [0.0] * len(self.corpus)

        for i, doc_tokens in enumerate(self.tokenized_corpus):
            score = 0.0
            doc_len = self.doc_lens[i]

            # Count term frequencies in document
            term_freqs = defaultdict(int)
            for token in doc_tokens:
                term_freqs[token] += 1

            # Calculate BM25 score
            for token in query_tokens:
                if token not in self.idf:
                    continue

                tf = term_freqs.get(token, 0)
                idf = self.idf[token]

                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))

                score += idf * (numerator / denominator)

            scores[i] = score

        return scores

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search documents using BM25.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of (document, score) tuples, sorted by score descending
        """
        if not self.corpus:
            logger.warning("BM25 index is empty, returning no results")
            return []

        scores = self.get_scores(query)

        results = []
        for i, score in enumerate(scores):
            if score > 0:
                results.append({
                    "document": self.corpus[i],
                    "score": score,
                    "index": i
                })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        # Return top-k
        return results[:top_k]
