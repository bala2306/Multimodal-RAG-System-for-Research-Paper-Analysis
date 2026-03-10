"""Table processing and description generation."""

from typing import Dict, Any, List, Optional
import json
from loguru import logger
from app.utils.file_storage import storage


class TableProcessor:
    """Process tables and generate descriptions."""
    
    def __init__(self):
        """Initialize table processor."""
        self.storage = storage
    
    def process_table(
        self,
        table_data: dict,
        document_id: str,
        element_id: str,
        page_number: int,
        pdf_path: Optional[str] = None
    ) -> dict:
        """
        Process table data and save to storage.
        
        Args:
            table_data: Dictionary with table information
            document_id: Document ID
            element_id: Visual element ID
            page_number: Page number
            pdf_path: Optional path to PDF for image extraction
            
        Returns:
            Processed table metadata
        """
        file_path = None
                
        with open("/tmp/table_debug.txt", "a") as f:
            f.write(f"process_table called: pdf_path={pdf_path is not None}, bbox={table_data.get('bbox') is not None}\n")
        
        logger.debug(f"Table processing - pdf_path: {pdf_path is not None}, bbox: {table_data.get('bbox') is not None}")
        if pdf_path and table_data.get('bbox'):
            try:
                logger.info(f"Attempting to extract table image with bbox: {table_data['bbox']}")
                file_path = self._extract_table_image(
                    pdf_path=pdf_path,
                    bbox=table_data['bbox'],
                    page_number=page_number,
                    document_id=document_id,
                    element_id=element_id
                )
                logger.info(f"Extracted table image: {file_path}")
            except Exception as e:
                import traceback
                logger.error(f"Failed to extract table image: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
        else:
            if not pdf_path:
                logger.warning("No PDF path provided for table image extraction")
            if not table_data.get('bbox'):
                logger.warning(f"No bbox in table_data. Keys: {list(table_data.keys())}")
                
        markdown = table_data.get("markdown", table_data.get("text", ""))
                
        table_json_path = self.storage.save_table_json(
            document_id=document_id,
            element_id=element_id,
            table_data={
                "markdown": markdown,
                "page": page_number,
                "metadata": table_data.get("metadata", {})
            }
        )
        
        description = self._generate_description(table_data)
        
        return {
            "element_type": "table",
            "page_number": page_number,
            "file_path": file_path,
            "table_json_path": table_json_path,
            "table_markdown": markdown,
            "text_annotation": description,
            "metadata": table_data.get("metadata", {})
        }
    
    def _extract_table_image(
        self,
        pdf_path: str,
        bbox: dict,
        page_number: int,
        document_id: str,
        element_id: str
    ) -> str:
        """
        Extract table region from PDF as an image.
        
        Args:
            pdf_path: Path to PDF file
            bbox: Bounding box with l, t, r, b, coord_origin
            page_number: Page number (1-indexed)
            document_id: Document ID
            element_id: Element ID
            
        Returns:
            Storage path to saved image
        """
        import fitz 
        
        pdf_doc = fitz.open(pdf_path)
        page = pdf_doc[page_number - 1]  
        page_height = page.rect.height
        
        coord_origin = str(bbox.get('coord_origin', 'BOTTOMLEFT'))
        if 'BOTTOMLEFT' in coord_origin:
            y0 = page_height - bbox['t'] 
            y1 = page_height - bbox['b'] 
        else:
            y0 = bbox['t']
            y1 = bbox['b']
        
        rect = fitz.Rect(bbox['l'], y0, bbox['r'], y1)
        
        pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(2, 2))
        
        img_bytes = pix.tobytes("png")
        
        pdf_doc.close()
        
        file_path = self.storage.save_image(
            document_id=document_id,
            element_id=element_id,
            image_data=img_bytes,
            extension="png"
        )
        
        return file_path
    
    def _generate_description(self, table_data: dict) -> str:
        """
        Generate rich, searchable text description of table with full content extraction.

        Extracts headers, row labels, and statistical keywords to make tables fully searchable.
        """
        markdown = table_data.get("markdown", table_data.get("text", ""))

        if not markdown or markdown == "| Table data unavailable |":
            return "Table with 0 rows"

        lines = [l.strip() for l in markdown.split('\n') if l.strip() and l.strip().startswith('|')]

        data_lines = [l for l in lines if not all(c in '|-: ' for c in l)]

        if not data_lines:
            return "Empty table"

        headers = []
        if data_lines:
            first_line = data_lines[0]
            headers = [h.strip() for h in first_line.split('|') if h.strip()]

        row_labels = []
        if len(data_lines) > 1:
            for line in data_lines[1:]:
                cells = [c.strip() for c in line.split('|') if c.strip()]
                if cells:
                    row_labels.append(cells[0])

        all_values = []
        for line in data_lines:
            cells = [c.strip() for c in line.split('|') if c.strip()]
            all_values.extend(cells)

        statistical_terms = []
        stat_keywords = ['p-value', 'p.value', 'pvalue', 'coefficient', 'coef', 'std', 'stderr',
                        'conf', 'interval', 't-stat', 'z-score', 'chi-square', 'df', 'mean',
                        'median', 'correlation', 'r-squared', 'beta', 'odds', 'ratio',
                        'significance', 'alpha', 'estimate']

        for term in stat_keywords:
            if any(term.lower() in str(v).lower() for v in all_values):
                if term not in statistical_terms:
                    statistical_terms.append(term)

        description_parts = []

        row_count = len(data_lines) - 1  
        description_parts.append(f"Statistical table with {row_count} data rows")

        if headers:
            description_parts.append(f"Columns: {', '.join(headers[:5])}")

        if row_labels:
            description_parts.append(f"Variables: {', '.join(row_labels[:10])}")

        if statistical_terms:
            description_parts.append(f"Contains: {', '.join(statistical_terms)}")

        plain_text_rows = self._extract_plain_text_rows(markdown, headers, row_labels)
        if plain_text_rows:
            description_parts.append(f"Data: {plain_text_rows}")

        description = ". ".join(description_parts)

        return description

    def _extract_plain_text_rows(
        self,
        markdown: str,
        headers: List[str],
        row_labels: List[str]
    ) -> str:
        """Extract first few rows as natural language for better searchability."""
        if not markdown or not headers:
            return ""

        lines = [l.strip() for l in markdown.split('\n') if l.strip() and l.strip().startswith('|')]
        data_lines = [l for l in lines if not all(c in '|-: ' for c in l)]

        if len(data_lines) < 2:
            return ""

        plain_text_parts = []
        for row_line in data_lines[1:4]:
            cells = [c.strip() for c in row_line.split('|') if c.strip()]

            if not cells or len(cells) != len(headers):
                continue

            row_label = cells[0]

            value_pairs = []
            for i, (header, value) in enumerate(zip(headers, cells)):
                if i == 0:
                    continue
                if any(keyword in header.lower() for keyword in ['p.value', 'p-value', 'pvalue', 'coefficient', 'coef', 'estimate']):
                    if value and value not in ['-', 'NA', 'N/A', '']:
                        value_pairs.append(f"{row_label} {header} {value}")

            plain_text_parts.extend(value_pairs)

        return "; ".join(plain_text_parts[:5]) 
