"""Database models representing table structures."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class Document:
    """Document database model."""
    id: str
    filename: str
    file_path: Optional[str]
    upload_date: datetime
    total_pages: int
    processing_status: str
    ingestion_type: str
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class Chunk:
    """Chunk database model."""
    id: str
    document_id: str
    chunk_text: str
    chunk_index: int
    page_number: int
    parent_chunk_id: Optional[str] = None
    chunk_type: Optional[str] = None
    ingestion_type: str = 'basic'
    visual_element_ids: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None


@dataclass
class VisualElement:
    """Visual element database model."""
    id: str
    document_id: str
    element_type: str
    page_number: int
    text_annotation: str
    bounding_box: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None
    table_data: Optional[Dict[str, Any]] = None
    table_markdown: Optional[str] = None
    related_chunk_ids: Optional[List[str]] = None
    ingestion_type: str = 'advanced'
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None


@dataclass
class Embedding:
    """Embedding database model."""
    id: str
    chunk_id: Optional[str]
    visual_element_id: Optional[str]
    collection_name: str
    vector_id: str
    embedding_model: str
    ingestion_type: str
    created_at: Optional[datetime] = None


@dataclass
class QueryLog:
    """Query log database model."""
    id: str
    query_text: str
    ingestion_type: str
    collection_searched: List[str]
    retrieved_chunk_ids: List[str]
    response_time_ms: int
    created_at: Optional[datetime] = None
