"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime


# Upload Schemas
class UploadResponse(BaseModel):
    """Response for document upload."""
    document_id: str
    filename: str
    total_pages: int
    chunks_created: int
    ingestion_type: Literal["basic", "advanced"]
    processing_status: str
    message: str


class AdvancedUploadResponse(UploadResponse):
    """Extended response for advanced upload with visual elements."""
    visual_elements_count: int
    tables_extracted: int
    images_extracted: int


# Query Schemas
class QueryRequest(BaseModel):
    """Request for querying the RAG system."""
    query: str = Field(..., min_length=1, description="User question")
    top_k: Optional[int] = Field(default=None, description="Number of results to retrieve")


class SourceReference(BaseModel):
    """Reference to a source chunk or visual element."""
    chunk_id: Optional[str] = None
    document_id: str
    document_name: str
    page_number: int
    text_snippet: str
    relevance_score: float
    chunk_type: Optional[str] = None


class VisualReference(BaseModel):
    """Reference to a visual element (table or image)."""
    element_id: str
    element_type: Literal["table", "image", "chart", "figure"]
    document_id: str
    page_number: int
    description: str
    file_path: Optional[str] = None
    image_url: Optional[str] = None
    table_markdown: Optional[str] = None
    relevance_score: float


class QueryResponse(BaseModel):
    """Response from RAG query."""
    answer: str
    sources: List[SourceReference]
    ingestion_type: Literal["basic", "advanced"]
    query_time_ms: int


class AdvancedQueryResponse(QueryResponse):
    """Extended response for advanced queries with visual elements."""
    visual_elements: List[VisualReference] = []


# Document Schemas
class DocumentInfo(BaseModel):
    """Document information."""
    id: str
    filename: str
    upload_date: datetime
    total_pages: int
    processing_status: str
    ingestion_type: Literal["basic", "advanced"]
    chunks_count: Optional[int] = None
    visual_elements_count: Optional[int] = None
    tables_extracted: Optional[int] = None
    images_extracted: Optional[int] = None


class DocumentListResponse(BaseModel):
    """List of documents with insights."""
    documents: List[DocumentInfo]
    total: int
    insights: Optional[dict] = None


# Chunk Schemas
class ChunkData(BaseModel):
    """Chunk data structure."""
    text: str
    page_number: int
    chunk_index: int
    metadata: Optional[dict] = None


class VisualElementData(BaseModel):
    """Visual element data structure."""
    element_type: Literal["table", "image", "chart", "figure"]
    page_number: int
    text_annotation: str
    file_path: Optional[str] = None
    table_data: Optional[dict] = None
    table_markdown: Optional[str] = None
    bounding_box: Optional[dict] = None
    metadata: Optional[dict] = None
