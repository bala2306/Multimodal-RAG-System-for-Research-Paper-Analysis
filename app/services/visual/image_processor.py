"""Image processing and description generation."""

from typing import Dict, Any
from loguru import logger
from app.utils.file_storage import storage


class ImageProcessor:
    """Process images and generate descriptions."""
    
    def process_image(
        self,
        image_data: Dict[str, Any],
        document_id: str,
        element_id: str,
        page_number: int
    ) -> Dict[str, Any]:
        """
        Process image and save to storage.
        
        Args:
            image_data: Raw image data from Docling
            document_id: Document ID
            element_id: Visual element ID
            page_number: Page number
            
        Returns:
            Processed image metadata
        """
        try:
            # Save image if PIL Image is available
            file_path = None
            if image_data.get("image_pil"):
                try:
                    # Convert PIL Image to bytes
                    from io import BytesIO
                    pil_image = image_data["image_pil"]
                    
                    logger.debug(f"Processing PIL image: {type(pil_image)}, size: {pil_image.size if pil_image else 'None'}")
                    
                    # Save as PNG
                    img_bytes = BytesIO()
                    pil_image.save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    
                    file_path = storage.save_image(
                        document_id=document_id,
                        element_id=element_id,
                        image_data=img_bytes.read(),
                        extension="png"
                    )
                    logger.info(f"Saved image to {file_path}")
                except Exception as img_error:
                    logger.error(f"Failed to save image: {img_error}")
            else:
                logger.warning(f"No PIL image data available for element {element_id}")
            
            # Get caption or generate description
            caption = image_data.get("caption", "")
            description = self._generate_description(caption, image_data)
            
            result = {
                "page_number": page_number,
                "element_type": self._classify_image_type(image_data),
                "file_path": file_path,
                "text_annotation": description,
                "bounding_box": image_data.get("bbox"),
                "metadata": {
                    "has_caption": bool(caption),
                    "original_caption": caption
                }
            }
            
            logger.debug(f"Processed image on page {page_number}, file_path: {file_path}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            return {
                "page_number": page_number,
                "element_type": "image",
                "file_path": None,
                "text_annotation": "Image (processing failed)",
                "metadata": {}
            }
    
    def _classify_image_type(self, image_data: Dict[str, Any]) -> str:
        """Classify image type based on available data."""
        caption = image_data.get("caption", "").lower()
        
        if "chart" in caption or "graph" in caption:
            return "chart"
        elif "figure" in caption or "fig" in caption:
            return "figure"
        else:
            return "image"
    
    def _generate_description(
        self,
        caption: str,
        image_data: Dict[str, Any]
    ) -> str:
        """
        Generate semantically rich, searchable text description for research paper figures.

        Creates comprehensive descriptions with enhanced keywords and context
        to improve semantic matching during vector search.
        """
        parts = []

        # Classify element type
        element_type = self._classify_image_type(image_data)

        # ENHANCED: Add element type with semantic keywords for better matching
        if element_type == "figure":
            # Add rich semantic context for figures
            parts.append("Figure")
            if caption and any(kw in caption.lower() for kw in ["overview", "architecture", "system", "model", "framework"]):
                parts.append("visual diagram illustrating system overview and architecture")
            elif caption and any(kw in caption.lower() for kw in ["flow", "process", "pipeline"]):
                parts.append("flowchart showing process flow")
            elif caption and any(kw in caption.lower() for kw in ["comparison", "vs", "versus"]):
                parts.append("comparison diagram")
            else:
                parts.append("visual illustration")
        elif element_type == "chart":
            parts.append("Chart or Graph")
            parts.append("data visualization showing trends and patterns")
        else:
            parts.append("Image")
            parts.append("visual representation")

        # Add caption - this is the KEY content for semantic matching!
        # Research paper captions like "Figure 1: Overview of SELF-RAG" are crucial
        if caption:
            # Keep full caption text for semantic matching
            parts.append(caption)

            # ENHANCED: Extract and emphasize key terms from caption for better matching
            caption_lower = caption.lower()

            # Add extracted key concepts as additional searchable text
            key_concepts = []

            # Extract figure number patterns (Figure 1, Fig. 2, etc.)
            import re
            fig_num_match = re.search(r'(figure|fig\.?)\s*(\d+)', caption_lower)
            if fig_num_match:
                key_concepts.append(f"figure {fig_num_match.group(2)}")

            # Extract title/description after colon
            if ':' in caption:
                title_part = caption.split(':', 1)[1].strip()
                if title_part:
                    # This is the main concept (e.g., "Overview of SELF-RAG")
                    key_concepts.append(f"Depicts: {title_part}")

            # Add domain-specific keywords for common research paper figures
            if any(kw in caption_lower for kw in ["overview", "architecture", "diagram"]):
                key_concepts.append("system architecture diagram")
            if any(kw in caption_lower for kw in ["training", "learning", "model"]):
                key_concepts.append("model training visualization")
            if any(kw in caption_lower for kw in ["results", "performance", "evaluation"]):
                key_concepts.append("experimental results")
            if any(kw in caption_lower for kw in ["example", "sample", "instance"]):
                key_concepts.append("illustrative example")

            # Add extracted concepts to description
            if key_concepts:
                parts.append(". ".join(key_concepts))
        else:
            # No caption available - add generic description
            parts.append("Research paper visual element")

        # Add page context
        page = image_data.get("page", 1)
        parts.append(f"Located on page {page}")

        # Join with periods for clear sentence structure
        description = ". ".join(parts)

        logger.debug(f"Generated enriched description for {element_type}: {description[:150]}...")
        return description

    def _generate_enhanced_description(
        self,
        caption: str,
        image_data: Dict[str, Any]
    ) -> str:
        """
        Generate enhanced text description when vision LLM fails.

        Creates richer descriptions by analyzing metadata and context.
        """
        element_type = self._classify_image_type(image_data)

        # Start with element type
        if element_type == "chart":
            description = "Chart or graph visualization"
        elif element_type == "figure":
            description = "Figure or diagram"
        else:
            description = "Image"

        # Add caption if available
        if caption:
            description += f": {caption}"

        # Add metadata insights
        metadata = image_data.get("metadata", {})
        if metadata.get("original_size"):
            width, height = metadata["original_size"]
            description += f" ({width}x{height} pixels)"

        # Add page context
        page = image_data.get("page", 1)
        description += f" from page {page}"

        return description
