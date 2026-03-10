"""API dependencies."""

from app.core.database import get_supabase
from app.services.embeddings.embedding_service import embedding_service
from app.services.llm.llm_service import LLMService
from app.services.vector_store.qdrant_basic import QdrantBasicService
from app.services.vector_store.qdrant_advanced import QdrantAdvancedService
from app.services.pdf.basic_processor import BasicPDFProcessor
from app.services.pdf.advanced_processor import AdvancedPDFProcessor
from app.services.visual.table_processor import TableProcessor
from app.services.visual.image_processor import ImageProcessor


def get_embedding_service():
    """Get embedding service instance."""
    return embedding_service


def get_llm_service():
    """Get LLM service instance."""
    return LLMService()


def get_qdrant_basic():
    """Get Qdrant basic service instance."""
    return QdrantBasicService()


def get_qdrant_advanced():
    """Get Qdrant advanced service instance."""
    return QdrantAdvancedService()


def get_basic_pdf_processor():
    """Get basic PDF processor instance."""
    return BasicPDFProcessor()


def get_advanced_pdf_processor():
    """Get advanced PDF processor instance."""
    return AdvancedPDFProcessor()


def get_table_processor():
    """Get table processor instance."""
    return TableProcessor()


def get_image_processor():
    """Get image processor instance."""
    return ImageProcessor()
