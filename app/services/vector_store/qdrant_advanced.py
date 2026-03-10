"""Advanced RAG Qdrant operations."""

from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue, MatchAny
import uuid
from loguru import logger
from app.core.config import settings


class QdrantAdvancedService:
    """Service for advanced RAG collection operations."""
    
    TEXT_COLLECTION = "advanced_text_collection"
    VISUAL_COLLECTION = "advanced_visual_collection"
    
    def __init__(self):
        """Initialize Qdrant client."""
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key if settings.qdrant_api_key else None
        )
        logger.info("Qdrant client initialized for advanced collections")
    
    def insert_text_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        document_id: str
    ) -> List[str]:
        """
        Insert text chunks into advanced text collection.
        
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
                    "chunk_type": chunk.get("chunk_type", "text"),
                    "ingestion_type": "advanced",
                    "has_visuals": bool(chunk.get("visual_element_ids")),
                    "visual_element_ids": chunk.get("visual_element_ids", []),
                    "metadata": chunk.get("metadata", {})
                }
            )
            points.append(point)
        
        # Insert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.TEXT_COLLECTION,
                points=batch
            )
        
        logger.info(f"Inserted {len(points)} text chunks into {self.TEXT_COLLECTION}")
        return vector_ids
    
    def insert_visual_elements(
        self,
        visual_elements: List[Dict[str, Any]],
        embeddings: List[List[float]],
        document_id: str
    ) -> List[str]:
        """
        Insert visual elements into advanced visual collection.
        
        Args:
            visual_elements: List of visual element dictionaries
            embeddings: List of embedding vectors
            document_id: Document ID
            
        Returns:
            List of vector IDs
        """
        if len(visual_elements) != len(embeddings):
            raise ValueError("Number of elements must match number of embeddings")
        
        points = []
        vector_ids = []
        
        for element, embedding in zip(visual_elements, embeddings):
            vector_id = str(uuid.uuid4())
            vector_ids.append(vector_id)
            
            point = PointStruct(
                id=vector_id,
                vector=embedding,
                payload={
                    "element_id": element.get("element_id"),
                    "document_id": document_id,
                    "element_type": element.get("element_type", "unknown"),
                    "text_annotation": element.get("text_annotation", ""),
                    "page": element.get("page_number", 1),
                    "file_path": element.get("file_path", ""),
                    "ingestion_type": "advanced",
                    "metadata": element.get("metadata", {})
                }
            )
            points.append(point)
        
        # Insert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.VISUAL_COLLECTION,
                points=batch
            )
        
        logger.info(f"Inserted {len(points)} visual elements into {self.VISUAL_COLLECTION}")
        return vector_ids
    
    def search_text(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search in text collection."""
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
            collection_name=self.TEXT_COLLECTION,
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
                "chunk_type": result.payload.get("chunk_type"),
                "visual_element_ids": result.payload.get("visual_element_ids", []),
                "metadata": result.payload.get("metadata", {})
            })
        
        logger.debug(f"Found {len(formatted_results)} text results")
        return formatted_results
    
    def search_visual(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        element_types: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search in visual collection."""
        filters = []
        
        if document_ids:
            filters.append(
                FieldCondition(
                    key="document_id",
                    match=MatchValue(any=document_ids)
                )
            )
        
        if element_types:
            filters.append(
                FieldCondition(
                    key="element_type",
                    match=MatchAny(any=element_types)
                )
            )
        
        query_filter = Filter(must=filters) if filters else None
        
        results = self.client.query_points(
            collection_name=self.VISUAL_COLLECTION,
            query=query_embedding,
            limit=top_k,
            query_filter=query_filter
        ).points
        
        formatted_results = []
        for result in results:
            formatted_results.append({
                "vector_id": result.id,
                "score": result.score,
                "element_id": result.payload.get("element_id"),
                "document_id": result.payload.get("document_id"),
                "element_type": result.payload.get("element_type"),
                "text_annotation": result.payload.get("text_annotation"),
                "page": result.payload.get("page"),
                "file_path": result.payload.get("file_path"),
                "metadata": result.payload.get("metadata", {})
            })
        
        logger.debug(f"Found {len(formatted_results)} visual results")
        return formatted_results
    
    def delete_by_document(self, document_id: str) -> None:
        """Delete all vectors for a document from both collections."""
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id)
                )
            ]
        )
        
        # Delete from text collection
        self.client.delete(
            collection_name=self.TEXT_COLLECTION,
            points_selector=filter_condition
        )
        
        # Delete from visual collection
        self.client.delete(
            collection_name=self.VISUAL_COLLECTION,
            points_selector=filter_condition
        )
        
        logger.info(f"Deleted vectors for document {document_id} from advanced collections")
