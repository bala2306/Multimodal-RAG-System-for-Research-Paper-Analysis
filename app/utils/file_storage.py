"""File storage utilities for images and tables using Supabase Storage."""

import os
from pathlib import Path
from typing import Optional
from loguru import logger
from app.core.config import settings
from supabase import create_client, Client


class FileStorage:
    """Manages Supabase Storage for visual elements."""
    
    def __init__(self):
        """Initialize Supabase client for storage."""
        self.supabase: Client = create_client(
            settings.supabase_url,
            settings.supabase_key
        )
        self.bucket_name = "rag-images"  # Supabase storage bucket name
        
        # Ensure bucket exists
        try:
            self.supabase.storage.get_bucket(self.bucket_name)
            logger.info(f"Connected to Supabase storage bucket: {self.bucket_name}")
        except Exception as e:
            logger.warning(f"Bucket {self.bucket_name} might not exist: {e}")
            # Try to create bucket if it doesn't exist
            try:
                self.supabase.storage.create_bucket(
                    self.bucket_name,
                    options={"public": False}
                )
                logger.info(f"Created Supabase storage bucket: {self.bucket_name}")
            except Exception as create_error:
                logger.error(f"Failed to create bucket: {create_error}")
    
    def save_image(
        self,
        document_id: str,
        element_id: str,
        image_data: bytes,
        extension: str = "png"
    ) -> str:
        """
        Save image to Supabase Storage.
        
        Args:
            document_id: Document ID
            element_id: Visual element ID
            image_data: Image binary data
            extension: File extension (without dot)
            
        Returns:
            Storage path (public URL or path)
        """
        try:
            # Create path in storage
            file_path = f"{document_id}/images/{element_id}.{extension}"
            
            # Upload to Supabase Storage
            self.supabase.storage.from_(self.bucket_name).upload(
                path=file_path,
                file=image_data,
                file_options={"content-type": f"image/{extension}"}
            )
            
            logger.info(f"Uploaded image to Supabase Storage: {file_path}")
            
            # Return the storage path
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to upload image to Supabase Storage: {e}")
            raise
    
    def save_table_json(
        self,
        document_id: str,
        element_id: str,
        table_data: dict
    ) -> str:
        """
        Save table data as JSON to Supabase Storage.
        
        Args:
            document_id: Document ID
            element_id: Visual element ID
            table_data: Table data dictionary
            
        Returns:
            Storage path
        """
        import json
        
        try:
            # Convert to JSON bytes
            json_data = json.dumps(table_data, indent=2).encode('utf-8')
            
            # Create path in storage
            file_path = f"{document_id}/tables/{element_id}.json"
            
            # Upload to Supabase Storage
            self.supabase.storage.from_(self.bucket_name).upload(
                path=file_path,
                file=json_data,
                file_options={"content-type": "application/json"}
            )
            
            logger.info(f"Uploaded table JSON to Supabase Storage: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to upload table to Supabase Storage: {e}")
            raise
    
    def save_pdf(
        self,
        document_id: str,
        filename: str,
        pdf_data: bytes
    ) -> str:
        """
        Save PDF file to Supabase Storage.
        
        Args:
            document_id: Document ID
            filename: Original filename
            pdf_data: PDF binary data
            
        Returns:
            Storage path
        """
        try:
            # Create path in storage
            file_path = f"{document_id}/pdf/{filename}"
            
            # Upload to Supabase Storage
            self.supabase.storage.from_(self.bucket_name).upload(
                path=file_path,
                file=pdf_data,
                file_options={"content-type": "application/pdf"}
            )
            
            logger.info(f"Uploaded PDF to Supabase Storage: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to upload PDF to Supabase Storage: {e}")
            raise
    
    def get_public_url(self, file_path: str) -> str:
        """
        Get public URL for a file in Supabase Storage.
        
        Args:
            file_path: Path in storage bucket
            
        Returns:
            Public URL
        """
        try:
            response = self.supabase.storage.from_(self.bucket_name).get_public_url(file_path)
            return response
        except Exception as e:
            logger.error(f"Failed to get public URL: {e}")
            return file_path
    
    def get_signed_url(self, file_path: str, expires_in: int = 3600) -> str:
        """
        Get signed URL for a file in Supabase Storage (for private buckets).
        
        Args:
            file_path: Path in storage bucket
            expires_in: URL expiration time in seconds (default 1 hour)
            
        Returns:
            Signed URL
        """
        try:
            response = self.supabase.storage.from_(self.bucket_name).create_signed_url(
                file_path, 
                expires_in
            )
            if response and 'signedURL' in response:
                return response['signedURL']
            elif response and 'signedUrl' in response:
                return response['signedUrl']
            else:
                logger.error(f"Unexpected response format: {response}")
                return file_path
        except Exception as e:
            logger.error(f"Failed to get signed URL: {e}")
            # Fallback to public URL
            return self.get_public_url(file_path)
    
    def download_file(self, file_path: str) -> bytes:
        """
        Download file from Supabase Storage.
        
        Args:
            file_path: Path in storage bucket
            
        Returns:
            File contents as bytes
        """
        try:
            response = self.supabase.storage.from_(self.bucket_name).download(file_path)
            return response
        except Exception as e:
            logger.error(f"Failed to download file: {e}")
            raise
    
    def delete_document_files(self, document_id: str) -> None:
        """
        Delete all files for a document from Supabase Storage.
        
        Args:
            document_id: Document ID
        """
        try:
            # List all files for this document
            files = self.supabase.storage.from_(self.bucket_name).list(document_id)
            
            if files:
                file_paths = [f"{document_id}/{f['name']}" for f in files if 'name' in f]
                if file_paths:
                    self.supabase.storage.from_(self.bucket_name).remove(file_paths)
                    logger.info(f"Deleted {len(file_paths)} files for document {document_id}")
                else:
                    logger.info(f"No files found to delete for document {document_id}")
            else:
                logger.info(f"No files found to delete for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to delete files for document {document_id}: {e}")
    
    def get_file_path(self, relative_path: str) -> Path:
        """
        Legacy method - returns path for compatibility.
        With Supabase Storage, files are accessed via download/URL.
        """
        logger.warning("get_file_path called but files are in Supabase Storage")
        return Path(relative_path)
    
    def file_exists(self, file_path: str) -> bool:
        """Check if file exists in Supabase Storage."""
        try:
            self.supabase.storage.from_(self.bucket_name).download(file_path)
            return True
        except Exception as e:
            if "The resource was not found" in str(e):
                return False
            logger.error(f"Error checking file existence for {file_path}: {e}")
            return False


# Global storage instance
storage = FileStorage()
