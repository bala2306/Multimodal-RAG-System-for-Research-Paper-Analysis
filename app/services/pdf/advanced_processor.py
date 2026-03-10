"""Advanced PDF processor using Docling."""

from typing import Dict, List, Any, Optional
from pathlib import Path
from loguru import logger

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logger.warning("Docling not available. Advanced PDF processing will be limited.")


class AdvancedPDFProcessor:
    """Advanced PDF processor using Docling for multimodal extraction."""
    
    def __init__(self):
        """Initialize advanced PDF processor."""
        if not DOCLING_AVAILABLE:
            logger.warning(
                "Docling is not installed. Install with: pip install docling docling-core"
            )
        self.converter = None
    
    def _ensure_converter(self):
        """Initialize Docling converter if not already done."""
        if not DOCLING_AVAILABLE:
            raise RuntimeError(
                "Docling is not available. Please install: pip install docling docling-core"
            )
        if self.converter is None:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            
            # Configure pipeline to extract images
            pipeline_options = PdfPipelineOptions()
            pipeline_options.generate_picture_images = True
            pipeline_options.images_scale = 2.0
            
            self.converter = DocumentConverter(
                format_options={
                    "pdf": PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            logger.info("Docling converter initialized with image extraction enabled")
    
    def extract_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract document with structure, tables, and images.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extracted content and structure
        """
        self._ensure_converter()
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Processing PDF with Docling: {pdf_path.name}")
        
        try:
            result = self.converter.convert(str(pdf_path))
            doc = result.document
            
            sections = []
            tables = []
            images = []
            
            total_pages_attr = getattr(doc, 'num_pages', None)
            if total_pages_attr is not None and callable(total_pages_attr):
                total_pages = total_pages_attr()
            elif total_pages_attr is not None:
                total_pages = int(total_pages_attr)
            else:
                total_pages = 1
            
            try:
                from docling_core.types.doc import DocItemLabel, TableItem, PictureItem, TextItem
                
                for element, _level in doc.iterate_items():
                    if isinstance(element, TableItem):
                        try:
                            table_markdown = element.export_to_markdown(doc) if hasattr(element, 'export_to_markdown') else ""
                            
                            bbox = None
                            if hasattr(element, 'prov') and element.prov:
                                prov = element.prov[0]
                                if hasattr(prov, 'bbox'):
                                    bbox = {
                                        'l': prov.bbox.l,
                                        't': prov.bbox.t,
                                        'r': prov.bbox.r,
                                        'b': prov.bbox.b,
                                        'coord_origin': str(prov.bbox.coord_origin) if hasattr(prov.bbox, 'coord_origin') else 'BOTTOMLEFT'
                                    }
                            
                            table_data = {
                                "type": "table",
                                "page": element.prov[0].page_no if element.prov else 1,
                                "text": table_markdown,
                                "markdown": table_markdown,
                                "bbox": bbox,
                                "metadata": {
                                    "label": str(element.label) if hasattr(element, 'label') else "table"
                                }
                            }
                            tables.append(table_data)
                            logger.debug(f"Extracted table from page {table_data['page']}, markdown length: {len(table_markdown)}")
                        except Exception as e:
                            logger.warning(f"Failed to extract table: {e}")
                    
                    elif isinstance(element, TextItem):
                        try:
                            text_content = element.export_to_text(doc) if hasattr(element, 'export_to_text') else str(element.text if hasattr(element, 'text') else element)
                            if text_content and text_content.strip():
                                section_data = {
                                    "type": "paragraph",
                                    "text": text_content.strip(),
                                    "page": element.prov[0].page_no if element.prov else 1,
                                    "metadata": {
                                        "label": str(element.label) if hasattr(element, 'label') else "text"
                                    }
                                }
                                sections.append(section_data)
                        except Exception as e:
                            logger.warning(f"Failed to extract text section: {e}")
                
                logger.debug(f"Found {len(doc.pictures)} pictures in document")
                for pic_item in doc.pictures:
                    try:
                        image_pil = pic_item.get_image(doc)
                        
                        if image_pil:
                            image_data = {
                                "type": "image",
                                "page": pic_item.prov[0].page_no if pic_item.prov else 1,
                                "caption": pic_item.caption_text(doc) if hasattr(pic_item, 'caption_text') else "",
                                "text": pic_item.caption_text(doc) if hasattr(pic_item, 'caption_text') else "Image",
                                "image_pil": image_pil,  # PIL Image object
                                "metadata": {
                                    "label": str(pic_item.label) if hasattr(pic_item, 'label') else "picture",
                                    "size": image_pil.size
                                }
                            }
                            images.append(image_data)
                            logger.debug(f"Extracted image from page {image_data['page']}, size: {image_pil.size}")
                        else:
                            logger.warning(f"Could not get image data for picture on page {pic_item.prov[0].page_no if pic_item.prov else '?'}")
                    except Exception as e:
                        logger.warning(f"Failed to extract image: {e}")
                
            except ImportError:
                logger.warning("Could not import Docling types, falling back to markdown export")
                markdown_text = doc.export_to_markdown()
                parts = markdown_text.split('\n\n')
                page_num = 1
                
                for i, part in enumerate(parts):
                    if part.strip():
                        sections.append({
                            "type": "paragraph",
                            "text": part.strip(),
                            "page": page_num,
                            "metadata": {}
                        })
                        if (i + 1) % 10 == 0:
                            page_num += 1
            
            if not sections:
                logger.warning("No sections extracted, using markdown fallback")
                markdown_text = doc.export_to_markdown()
                if markdown_text.strip():
                    parts = markdown_text.split('\n\n')
                    page_num = 1
                    
                    for i, part in enumerate(parts):
                        if part.strip():
                            sections.append({
                                "type": "paragraph",
                                "text": part.strip(),
                                "page": page_num,
                                "metadata": {}
                            })
                            if (i + 1) % 10 == 0:
                                page_num += 1
            
            result_data = {
                "filename": pdf_path.name,
                "total_pages": int(total_pages),
                "sections": sections,
                "tables": tables,
                "images": images,
                "metadata": {
                    "extraction_method": "Docling",
                    "sections_count": int(len(sections)),
                    "tables_count": int(len(tables)),
                    "images_count": int(len(images)),
                    "total_characters": sum(len(s.get("text", "")) for s in sections)
                }
            }
            
            logger.success(
                f"Extracted {len(sections)} sections, "
                f"{len(tables)} tables, {len(images)} images"
            )
            
            return result_data
            
        except Exception as e:
            logger.error(f"Failed to process PDF with Docling: {e}")
            logger.info("Falling back to basic text extraction")
            return self._fallback_extraction(pdf_path)
    
    def _extract_table(self, table_element) -> Dict[str, Any]:
        """Extract table data from Docling table element."""
        try:
            table_data = {
                "page": getattr(table_element, "page", 1),
                "bbox": getattr(table_element, "bbox", None),
                "cells": [],
                "markdown": str(table_element) if hasattr(table_element, "__str__") else ""
            }
            
            if hasattr(table_element, "data"):
                table_data["cells"] = table_element.data
            
            return table_data
            
        except Exception as e:
            logger.warning(f"Failed to extract table details: {e}")
            return {
                "page": 1,
                "cells": [],
                "markdown": ""
            }
    
    def _extract_image(self, image_element) -> Dict[str, Any]:
        """Extract image data from Docling image element."""
        try:
            image_data = {
                "page": getattr(image_element, "page", 1),
                "bbox": getattr(image_element, "bbox", None),
                "caption": getattr(image_element, "caption", ""),
                "image_data": getattr(image_element, "image", None)
            }
            
            return image_data
            
        except Exception as e:
            logger.warning(f"Failed to extract image details: {e}")
            return {
                "page": 1,
                "bbox": None,
                "caption": "",
                "image_data": None
            }
    
    def _fallback_extraction(self, pdf_path: Path) -> Dict[str, Any]:
        """Fallback to basic extraction if Docling fails."""
        import fitz
        
        doc = fitz.open(pdf_path)
        sections = []
        total_pages = len(doc)
        
        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text()
            
            if text.strip():
                sections.append({
                    "type": "paragraph",
                    "text": text,
                    "page": page_num + 1,
                    "metadata": {}
                })
        
        doc.close()
        
        return {
            "filename": pdf_path.name,
            "total_pages": total_pages,
            "sections": sections,
            "tables": [],
            "images": [],
            "metadata": {
                "extraction_method": "PyMuPDF (fallback)",
                "sections_count": len(sections),
                "tables_count": 0,
                "images_count": 0
            }
        }
