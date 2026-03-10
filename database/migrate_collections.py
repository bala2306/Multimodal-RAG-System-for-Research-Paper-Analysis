"""
Migrate Qdrant collections from old embedding dimensions to new dimensions.
This script recreates collections with new vector dimensions and provides instructions for re-indexing.
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from loguru import logger
import sys

load_dotenv()


def migrate_qdrant_collections(
    old_dimension: int = 384,
    new_dimension: int = 768,
    backup_old: bool = True
):
    """
    Migrate Qdrant collections to new embedding dimensions.

    Args:
        old_dimension: Old vector dimension (default: 384 for all-MiniLM-L6-v2)
        new_dimension: New vector dimension (default: 768 for all-mpnet-base-v2)
        backup_old: Whether to backup old collections before deleting
    """
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    # Initialize client
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key if qdrant_api_key else None
    )

    collections_to_migrate = [
        "basic_rag_collection",
        "advanced_text_collection",
        "advanced_visual_collection"
    ]

    logger.info(f"Old embedding dimension: {old_dimension}")
    logger.info(f"New embedding dimension: {new_dimension}")
    logger.info(f"Collections to migrate: {', '.join(collections_to_migrate)}")

    # Get existing collections
    existing_collections = client.get_collections().collections
    existing_names = {col.name for col in existing_collections}

    for collection_name in collections_to_migrate:
        logger.info(f"\nProcessing collection: {collection_name}")

        # Check if collection exists
        if collection_name not in existing_names:
            logger.warning(f"  Collection '{collection_name}' does not exist. Creating new...")
            # Create new collection
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=new_dimension,
                    distance=Distance.COSINE
                )
            )
            logger.success(f"  ✓ Created new collection: {collection_name}")
            continue

        # Get collection info
        collection_info = client.get_collection(collection_name)
        current_dim = collection_info.config.params.vectors.size

        if current_dim == new_dimension:
            logger.info(f"  Collection '{collection_name}' already has dimension {new_dimension}. Skipping.")
            continue

        logger.info(f"  Current dimension: {current_dim}")
        logger.info(f"  Target dimension: {new_dimension}")

        # Get collection stats
        point_count = collection_info.points_count
        logger.info(f"  Current point count: {point_count}")

        if point_count == 0:
            logger.info("  Collection is empty. Deleting and recreating...")
            client.delete_collection(collection_name)
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=new_dimension,
                    distance=Distance.COSINE
                )
            )
            logger.success(f"  ✓ Recreated collection: {collection_name}")
            continue

        # Collection has data - requires manual intervention
        logger.warning(f"  ⚠ Collection has {point_count} points - data migration required!")

        if backup_old:
            backup_name = f"{collection_name}_backup_{old_dimension}d"
            logger.info(f"  Creating backup: {backup_name}")

            try:
                # Create backup collection
                client.create_collection(
                    collection_name=backup_name,
                    vectors_config=VectorParams(
                        size=current_dim,
                        distance=Distance.COSINE
                    )
                )

                logger.warning(f"Backup collection created, but you need to manually copy data!")
                logger.info(f"Use Qdrant's scroll API to copy vectors from {collection_name} to {backup_name}")

            except Exception as e:
                logger.error(f"Failed to create backup: {e}")

        # Ask for confirmation to delete
        logger.warning(f"\n  ATTENTION: To migrate '{collection_name}':")
        logger.warning(f"  1. The existing collection with {point_count} points will be DELETED")
        logger.warning(f"  2. A new empty collection with {new_dimension}-dim vectors will be created")
        logger.warning(f"  3. You must RE-INDEX all documents using the new embedding model")

    logger.warning("\nTo complete the migration, choose one of these options:")
    logger.info("\n1. AUTOMATIC RECREATION (deletes data):")
    logger.info("   Run this script with --force flag to delete old collections")
    logger.info("   Then re-upload all documents via the API")
    logger.info("\n2. MANUAL MIGRATION (preserves data):")
    logger.info("   a) Use Qdrant's snapshot/restore feature")
    logger.info("   b) Re-embed all chunks with new model")
    logger.info("   c) Insert into new collections")
    logger.info("\n3. KEEP BOTH (recommended for testing):")
    logger.info("   a) Create new collections with different names")
    logger.info("   b) Upload documents to new collections")
    logger.info("   c) Compare performance before deleting old collections")


def force_recreate_collections(new_dimension: int = 768):
    """
    Force recreate all collections (DELETES EXISTING DATA).

    Args:
        new_dimension: New vector dimension
    """
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key if qdrant_api_key else None
    )

    collections = ["basic_rag_collection", "advanced_text_collection", "advanced_visual_collection"]

    logger.warning("⚠ FORCE RECREATE MODE - ALL DATA WILL BE DELETED!")

    for collection_name in collections:
        try:
            logger.info(f"Deleting collection: {collection_name}")
            client.delete_collection(collection_name)
            logger.success(f"  ✓ Deleted")
        except Exception as e:
            logger.info(f"  Collection doesn't exist or already deleted: {e}")

        logger.info(f"Creating new collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=new_dimension,
                distance=Distance.COSINE
            )
        )
        logger.success(f"  ✓ Created with {new_dimension}-dimensional vectors")

    logger.success("\n✓ All collections recreated. You must now re-upload all documents.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        logger.warning("FORCE MODE: All collections will be deleted and recreated!")
        confirm = input("Type 'DELETE ALL DATA' to confirm: ")
        if confirm == "DELETE ALL DATA":
            force_recreate_collections()
        else:
            logger.info("Migration cancelled.")
    else:
        migrate_qdrant_collections()
        logger.info("\n💡 TIP: Run with --force flag to automatically recreate collections")
