"""
Initialize Qdrant collections for RAG pipeline.
Run this script once to set up all required collections.
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType
from loguru import logger

load_dotenv()

def init_qdrant_collections():
    """Initialize all Qdrant collections for the RAG pipeline."""
    
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    # Initialize client
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key if qdrant_api_key else None
    )
    
    embedding_dim = int(os.getenv("EMBEDDING_DIMENSION", "768"))
    
    collections = [
        {
            "name": "basic_rag_collection",
            "description": "Basic RAG - simple text chunks"
        },
        {
            "name": "advanced_text_collection",
            "description": "Advanced RAG - text chunks with visual references"
        },
        {
            "name": "advanced_visual_collection",
            "description": "Advanced RAG - tables and images"
        }
    ]
    
    for collection_info in collections:
        collection_name = collection_info["name"]
        
        try:
            # Check if collection exists
            existing_collections = client.get_collections().collections
            exists = any(col.name == collection_name for col in existing_collections)
            
            if exists:
                logger.info(f"Collection '{collection_name}' already exists. Skipping.")
                continue
            
            # Create collection
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE
                )
            )
            
            logger.success(f"✓ Created collection: {collection_name}")
            logger.info(f"  Description: {collection_info['description']}")
            logger.info(f"  Vector dimension: {embedding_dim}")
            logger.info(f"  Distance metric: COSINE")
            
        except Exception as e:
            logger.error(f"✗ Failed to create collection '{collection_name}': {e}")
            raise
    
    # Create payload indexes for filtering
    try:
        logger.info("Creating payload indexes for advanced_visual_collection...")
        client.create_payload_index(
            collection_name="advanced_visual_collection",
            field_name="element_type",
            field_schema=PayloadSchemaType.KEYWORD
        )
        logger.success("✓ Created index on element_type field")
    except Exception as e:
        logger.warning(f"Index creation skipped (may already exist): {e}")
    
    logger.success("\n🎉 All Qdrant collections initialized successfully!")
    
    # Display summary
    logger.info("\nCollection Summary:")
    collections_list = client.get_collections().collections
    for col in collections_list:
        logger.info(f"  - {col.name}")

if __name__ == "__main__":
    logger.info("Initializing Qdrant collections for RAG pipeline...")
    init_qdrant_collections()

