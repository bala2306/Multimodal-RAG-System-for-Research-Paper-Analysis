"""Embedding service using sentence-transformers."""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger
from app.core.config import settings


class EmbeddingService:
    """Service for generating text embeddings."""
    
    _instance: 'EmbeddingService' = None
    _model: SentenceTransformer = None
    
    def __new__(cls):
        """Singleton pattern for model loading."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize embedding model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            self._model = SentenceTransformer(settings.embedding_model)
            logger.success(f"Embedding model loaded successfully")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * settings.embedding_dimension
        
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Filter out empty texts but maintain indices
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
        
        if not valid_texts:
            logger.warning("No valid texts provided for batch embedding")
            return [[0.0] * settings.embedding_dimension] * len(texts)
        
        # Generate embeddings for valid texts
        logger.debug(f"Generating embeddings for {len(valid_texts)} texts")
        embeddings = self._model.encode(
            valid_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(valid_texts) > 100
        )
        
        # Create result list with zeros for invalid texts
        result = [[0.0] * settings.embedding_dimension] * len(texts)
        result_list = [list(result[i]) for i in range(len(texts))]
        
        # Fill in valid embeddings
        for idx, embedding in zip(valid_indices, embeddings):
            result_list[idx] = embedding.tolist()
        
        logger.debug(f"Generated {len(result_list)} embeddings")
        return result_list
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return settings.embedding_dimension
    
    def get_model_name(self) -> str:
        """Get the name of the embedding model."""
        return settings.embedding_model


# Global embedding service instance
embedding_service = EmbeddingService()
