"""Basic PDF processor using PyMuPDF."""

import fitz  # PyMuPDF
from typing import Dict, List, Any
from pathlib import Path
from loguru import logger


class BasicPDFProcessor:
    """Basic PDF text extraction using PyMuPDF."""
    
    def extract_text(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from PDF file using PyMuPDF.
        Falls back to OCR if minimal text is extracted.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Extracting text from PDF: {pdf_path.name}")
        
        try:
            doc = fitz.open(pdf_path)
            
            pages_data = []
            total_text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                pages_data.append({
                    "page_number": page_num + 1,
                    "text": text,
                    "char_count": len(text)
                })
                
                total_text += text + "\n\n"
            
            doc.close()
            
            meaningful_chars = len(total_text.strip())
            
            if meaningful_chars < 100:
                logger.warning(
                    f"PyMuPDF extracted only {meaningful_chars} characters. "
                    "Attempting OCR for text extraction..."
                )
                return self._extract_with_ocr(pdf_path)
            
            result = {
                "filename": pdf_path.name,
                "total_pages": len(pages_data),
                "pages": pages_data,
                "full_text": total_text,
                "metadata": {
                    "total_characters": len(total_text),
                    "extraction_method": "PyMuPDF"
                }
            }
            
            logger.success(
                f"Extracted text from {len(pages_data)} pages "
                f"({len(total_text)} characters)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            raise
    
    def _extract_with_ocr(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract text using OCR for scanned PDFs.
        Only extracts text - no images or tables.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            from docling.document_converter import DocumentConverter
            
            logger.info(f"Using OCR for text extraction: {pdf_path.name}")
            
            converter = DocumentConverter()
            result = converter.convert(str(pdf_path))
            doc = result.document
            
            markdown_text = doc.export_to_markdown()
            
            total_pages_attr = getattr(doc, 'num_pages', None)
            if total_pages_attr is not None and callable(total_pages_attr):
                total_pages = total_pages_attr()
            elif total_pages_attr is not None:
                total_pages = int(total_pages_attr)
            else:
                total_pages = max(1, len(markdown_text) // 3000)
            
            pages_data = []
            chars_per_page = len(markdown_text) // total_pages if total_pages > 0 else len(markdown_text)
            
            for page_num in range(total_pages):
                start = page_num * chars_per_page
                end = start + chars_per_page if page_num < total_pages - 1 else len(markdown_text)
                page_text = markdown_text[start:end]
                
                pages_data.append({
                    "page_number": page_num + 1,
                    "text": page_text,
                    "char_count": len(page_text)
                })
            
            result_data = {
                "filename": pdf_path.name,
                "total_pages": total_pages,
                "pages": pages_data,
                "full_text": markdown_text,
                "metadata": {
                    "total_characters": len(markdown_text),
                    "extraction_method": "OCR (text only)"
                }
            }
            
            logger.success(
                f"OCR extracted text from {total_pages} pages "
                f"({len(markdown_text)} characters)"
            )
            
            return result_data
            
        except ImportError:
            logger.error("Docling not available for OCR. Install with: pip install docling")
            raise RuntimeError(
                "Cannot process scanned PDF: Docling not installed. "
                "Install with: pip install docling"
            )
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            raise
    
    def extract_page_text(self, pdf_path: str, page_number: int) -> str:
        """
        Extract text from a specific page.
        
        Args:
            pdf_path: Path to PDF file
            page_number: Page number (1-indexed)
            
        Returns:
            Extracted text
        """
        try:
            doc = fitz.open(pdf_path)
            
            if page_number < 1 or page_number > len(doc):
                raise ValueError(f"Invalid page number: {page_number}")
            
            page = doc[page_number - 1]
            text = page.get_text()
            doc.close()
            
            return text
            
        except Exception as e:
            logger.error(f"Failed to extract text from page {page_number}: {e}")
            raise
    
    def get_pdf_info(self, pdf_path: str) -> Dict[str, Any]:
        """
        Get PDF metadata without extracting text.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            PDF metadata
        """
        try:
            doc = fitz.open(pdf_path)
            
            metadata = {
                "filename": Path(pdf_path).name,
                "total_pages": len(doc),
                "pdf_metadata": doc.metadata,
                "file_size_bytes": Path(pdf_path).stat().st_size
            }
            
            doc.close()
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get PDF info: {e}")
            raise
