"""File validation and sanitization utilities."""

import re
from pathlib import Path
from typing import Tuple
from fastapi import UploadFile, HTTPException
from loguru import logger


class FileValidator:
    """Validates uploaded files."""
    
    ALLOWED_EXTENSIONS = {'.pdf'}
    MAX_FILENAME_LENGTH = 255
    
    @staticmethod
    def validate_pdf(file: UploadFile, max_size_mb: int = 50) -> None:
        """
        Validate PDF file upload.
        
        Args:
            file: Uploaded file
            max_size_mb: Maximum file size in MB
            
        Raises:
            HTTPException: If validation fails
        """
        # Check content type
        if file.content_type not in ['application/pdf']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Only PDF files are allowed. Got: {file.content_type}"
            )
        
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in FileValidator.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file extension. Only {FileValidator.ALLOWED_EXTENSIONS} are allowed."
            )
        
        # Check filename length
        if len(file.filename) > FileValidator.MAX_FILENAME_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Filename too long. Maximum {FileValidator.MAX_FILENAME_LENGTH} characters."
            )
        
        logger.debug(f"File validation passed for: {file.filename}")
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename to prevent path traversal and other issues.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Get just the filename without path
        filename = Path(filename).name
        
        # Remove or replace dangerous characters
        filename = re.sub(r'[^\w\s\-\.]', '_', filename)
        
        # Remove multiple dots (except the extension)
        parts = filename.rsplit('.', 1)
        if len(parts) == 2:
            name, ext = parts
            name = name.replace('.', '_')
            filename = f"{name}.{ext}"
        
        # Limit length
        if len(filename) > FileValidator.MAX_FILENAME_LENGTH:
            name, ext = filename.rsplit('.', 1)
            max_name_len = FileValidator.MAX_FILENAME_LENGTH - len(ext) - 1
            filename = f"{name[:max_name_len]}.{ext}"
        
        return filename
    
    @staticmethod
    def validate_and_sanitize(file: UploadFile, max_size_mb: int = 50) -> str:
        """
        Validate and sanitize uploaded file.
        
        Args:
            file: Uploaded file
            max_size_mb: Maximum file size in MB
            
        Returns:
            Sanitized filename
        """
        FileValidator.validate_pdf(file, max_size_mb)
        return FileValidator.sanitize_filename(file.filename)
