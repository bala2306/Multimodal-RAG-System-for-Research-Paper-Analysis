"""Basic RAG Qdrant operations."""

from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
import uuid
from loguru import logger
from app.core.config import settings


class QdrantBasicService:
    """Service for basic RAG collection operations."""
    
    COLLECTION_NAME = "basic_rag_collection"
    
    def __init__(self):
        """Initialize Qdrant client."""
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key if settings.qdrant_api_key else None
        )
        logger.info(f"Qdrant client initialized for {self.COLLECTION_NAME}")
    
    def insert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        document_id: str
    ) -> List[str]:
        """
        Insert chunks with embeddings into basic collection.
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: List of embedding vectors
            document_id: Document ID
            
        Returns:
            List of vector IDs
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        points = []
        vector_ids = []
        
        for chunk, embedding in zip(chunks, embeddings):
            vector_id = str(uuid.uuid4())
            vector_ids.append(vector_id)
            
            point = PointStruct(
                id=vector_id,
                vector=embedding,
                payload={
                    "chunk_id": chunk.get("chunk_id"),
                    "document_id": document_id,
                    "text": chunk.get("text", ""),
                    "page": chunk.get("page_number", 1),
                    "ingestion_type": "basic",
                    "chunk_index": chunk.get("chunk_index", 0),
                    "metadata": chunk.get("metadata", {})
                }
            )
            points.append(point)
        
        # Insert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=batch
            )
        
        logger.info(f"Inserted {len(points)} points into {self.COLLECTION_NAME}")
        return vector_ids
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            document_ids: Optional filter by document IDs
            
        Returns:
            List of search results with scores
        """
        query_filter = None
        
        if document_ids:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(any=document_ids)
                    )
                ]
            )
        
        results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_embedding,
            limit=top_k,
            query_filter=query_filter
        ).points
        
        formatted_results = []
        for result in results:
            formatted_results.append({
                "vector_id": result.id,
                "score": result.score,
                "chunk_id": result.payload.get("chunk_id"),
                "document_id": result.payload.get("document_id"),
                "text": result.payload.get("text"),
                "page": result.payload.get("page"),
                "chunk_index": result.payload.get("chunk_index"),
                "metadata": result.payload.get("metadata", {})
            })
        
        logger.debug(f"Found {len(formatted_results)} results in basic collection")
        return formatted_results
    
    def delete_by_document(self, document_id: str) -> None:
        """
        Delete all vectors for a document.
        
        Args:
            document_id: Document ID
        """
        self.client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
        )
        logger.info(f"Deleted vectors for document {document_id} from {self.COLLECTION_NAME}")
