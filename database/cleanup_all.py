"""
Cleanup script to delete all data from Supabase and Qdrant.
WARNING: This will permanently delete all documents, chunks, embeddings, and vector data.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList
from loguru import logger

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Qdrant configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Collection names
COLLECTIONS = [
    "basic_rag_collection",
    "advanced_text_collection",
    "advanced_visual_collection"
]


def cleanup_supabase():
    """Delete all data from Supabase tables."""
    try:
        logger.info("Connecting to Supabase...")
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Delete in order (respecting foreign key constraints)
        tables = ["embeddings", "visual_elements", "chunks", "query_logs", "documents"]
        
        for table in tables:
            logger.info(f"Deleting all records from {table}...")
            try:
                # Delete all records (Supabase doesn't have a simple truncate)
                result = supabase.table(table).delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
                logger.success(f"Deleted records from {table}")
            except Exception as e:
                logger.warning(f"Error deleting from {table}: {e}")
        
        logger.success("✅ Supabase cleanup completed")
        
    except Exception as e:
        logger.error(f"Failed to cleanup Supabase: {e}")
        raise


def cleanup_qdrant():
    """Delete all points from Qdrant collections."""
    try:
        logger.info("Connecting to Qdrant...")
        
        # Initialize Qdrant client
        if QDRANT_API_KEY:
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            client = QdrantClient(url=QDRANT_URL)
        
        # Delete all points from each collection
        for collection_name in COLLECTIONS:
            try:
                # Check if collection exists
                collections = client.get_collections().collections
                collection_names = [c.name for c in collections]
                
                if collection_name not in collection_names:
                    logger.warning(f"Collection {collection_name} does not exist, skipping...")
                    continue
                
                # Get count before deletion
                count_before = client.count(collection_name).count
                logger.info(f"Collection {collection_name} has {count_before} points")
                
                if count_before == 0:
                    logger.info(f"Collection {collection_name} is already empty")
                    continue
                
                logger.info(f"Deleting all points from collection: {collection_name}...")
                
                all_points = []
                offset = None
                
                while True:
                    scroll_result = client.scroll(
                        collection_name=collection_name,
                        limit=100,
                        offset=offset,
                        with_payload=False,
                        with_vectors=False
                    )
                    
                    points, next_offset = scroll_result
                    
                    if points:
                        all_points.extend([point.id for point in points])
                    
                    if next_offset is None:
                        break
                    
                    offset = next_offset
                
                # Delete all points
                if all_points:
                    client.delete(
                        collection_name=collection_name,
                        points_selector=PointIdsList(
                            points=all_points
                        )
                    )
                    logger.success(f"Deleted {len(all_points)} points from {collection_name}")
                else:
                    logger.info(f"No points found in {collection_name}")
                    
            except Exception as e:
                logger.error(f"Error cleaning collection {collection_name}: {e}")
                import traceback
                traceback.print_exc()
        
        logger.success("✅ Qdrant cleanup completed")
        
    except Exception as e:
        logger.error(f"Failed to cleanup Qdrant: {e}")
        raise


def main():
    """Main cleanup function."""
    logger.info("=" * 60)
    logger.warning("⚠️  WARNING: This will DELETE ALL DATA!")
    logger.info("=" * 60)
    
    # Ask for confirmation
    response = input("\nAre you sure you want to delete all data? (yes/no): ")
    
    if response.lower() != "yes":
        logger.info("Cleanup cancelled.")
        return
    
    logger.info("\nStarting cleanup process...\n")
    
    # Cleanup Supabase
    try:
        cleanup_supabase()
    except Exception as e:
        logger.error(f"Supabase cleanup failed: {e}")
    
    # Cleanup Qdrant
    try:
        cleanup_qdrant()
    except Exception as e:
        logger.error(f"Qdrant cleanup failed: {e}")
    
    # Cleanup Supabase Storage
    logger.info("\n" + "=" * 60)
    logger.info("Cleaning Supabase Storage...")
    logger.info("=" * 60)
    
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        bucket_name = "rag-images"
        logger.info(f"Deleting all files from bucket '{bucket_name}'...")
        
        def delete_folder_recursively(folder_path=""):
            """Recursively delete all files in a folder."""
            file_count = 0
            try:
                # List items in current folder
                items = supabase.storage.from_(bucket_name).list(folder_path)
                
                if not items:
                    return 0
                
                for item in items:
                    if not item.get('name'):
                        continue
                    
                    item_name = item['name']
                    full_path = f"{folder_path}/{item_name}" if folder_path else item_name
                    
                    if item.get('id') is None:
                        logger.info(f"Exploring folder: {full_path}")
                        file_count += delete_folder_recursively(full_path)
                    else:
                        # It's a file, delete it
                        try:
                            supabase.storage.from_(bucket_name).remove([full_path])
                            file_count += 1
                            logger.debug(f"Deleted file: {full_path}")
                        except Exception as e:
                            logger.warning(f"Error deleting file {full_path}: {e}")
                
                return file_count
                
            except Exception as e:
                logger.warning(f"Error listing folder {folder_path}: {e}")
                return file_count
        
        # Start recursive deletion from root
        total_deleted = delete_folder_recursively()
        
        if total_deleted > 0:
            logger.success(f"Deleted {total_deleted} total files from Supabase Storage")
        else:
            logger.info("No files found in storage bucket")
            
    except Exception as e:
        logger.error(f"Error cleaning Supabase Storage: {e}")
    
    logger.success("Cleanup process completed!")    


if __name__ == "__main__":
    main()

