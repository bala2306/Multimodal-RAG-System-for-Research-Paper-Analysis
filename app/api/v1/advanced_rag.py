"""Advanced RAG API endpoints."""

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from typing import List, Dict, Any
import tempfile
import time
from pathlib import Path
from loguru import logger
from supabase import Client
import uuid

from app.models.schemas import (
    AdvancedUploadResponse,
    QueryRequest,
    AdvancedQueryResponse,
    SourceReference,
    VisualReference,
    DocumentInfo,
    DocumentListResponse
)
from app.core.database import get_supabase
from app.api.dependencies import (
    get_embedding_service,
    get_llm_service,
    get_qdrant_advanced,
    get_advanced_pdf_processor,
    get_table_processor,
    get_image_processor
)
from app.utils.validators import FileValidator
from app.utils.chunking import SemanticChunker
from app.services.embeddings.embedding_service import EmbeddingService
from app.services.llm.llm_service import LLMService
from app.services.vector_store.qdrant_advanced import QdrantAdvancedService
from app.services.pdf.advanced_processor import AdvancedPDFProcessor
from app.services.visual.table_processor import TableProcessor
from app.services.visual.image_processor import ImageProcessor
from app.utils.file_storage import storage

router = APIRouter(prefix="/advanced", tags=["Advanced RAG"])


def _extract_query_terms(query: str) -> list:
    """
    UNIVERSAL query term extraction

    Extracts ALL meaningful terms from the query including:
    - Function calls: log(population), sqrt(x), etc.
    - Multi-word phrases: "market share", "customer satisfaction"
    - Quoted terms: "exact phrase"
    - Compound terms with punctuation: p.value, t-test, etc.
    - Individual significant words

    Returns list of terms to search for in tables.
    """
    import re

    terms = []
    query_lower = query.lower()
    
    quoted = re.findall(r'["\']([^"\']+)["\']', query)
    terms.extend(quoted)

    function_patterns = re.findall(r'\b([a-z_]+)\s*\(\s*([^)]+)\s*\)', query_lower)
    for func, arg in function_patterns:
        terms.append(f"{func}({arg.strip()})")
        terms.append(f"{func} ({arg.strip()})")
        terms.append(arg.strip())

    compound_terms = re.findall(r'\b[a-z0-9]+[._-][a-z0-9._-]+\b', query_lower)
    for term in compound_terms:
        terms.append(term)
        terms.append(term.replace('.', ' '))
        terms.append(term.replace('-', ' '))
        terms.append(term.replace('_', ' '))
        terms.append(term.replace('.', '').replace('-', '').replace('_', ''))

    words = re.findall(r'\b[a-z][a-z0-9_]*\b', query_lower)
    stop_words = {'what', 'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                  'to', 'for', 'of', 'with', 'by', 'from', 'as', 'this', 'that',
                  'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do',
                  'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                  'can', 'show', 'find', 'get', 'give', 'tell', 'me', 'you', 'it'}

    filtered_words = [w for w in words if w not in stop_words and len(w) > 2]

    for i in range(len(filtered_words) - 1):
        bigram = f"{filtered_words[i]} {filtered_words[i+1]}"
        terms.append(bigram)
    for i in range(len(filtered_words) - 2):
        trigram = f"{filtered_words[i]} {filtered_words[i+1]} {filtered_words[i+2]}"
        terms.append(trigram)

    terms.extend(filtered_words)

    numbers = re.findall(r'\b\d+\.?\d*%?\b', query)
    terms.extend(numbers)

    unique_terms = []
    seen = set()
    for term in terms:
        term_clean = term.strip().lower()
        if term_clean and term_clean not in seen and len(term_clean) > 1:
            unique_terms.append(term_clean)
            seen.add(term_clean)

    return unique_terms


def _search_tables_by_terms(
    supabase,
    document_ids: list,
    query_terms: list
) -> list:
    """
    UNIVERSAL table search

    Uses fuzzy matching and intelligent scoring to find relevant tables
    for ANY query type (not just statistical).

    Args:
        supabase: Supabase client
        document_ids: List of document IDs to search within
        query_terms: List of terms extracted from the query

    Returns:
        List of matched tables with relevance scores
    """
    try:
        tables_result = supabase.table("visual_elements").select(
            "id, element_type, page_number, file_path, text_annotation, table_markdown, metadata, document_id"
        ).eq("element_type", "table").in_("document_id", document_ids).execute()

        if not tables_result.data:
            return []

        matched_tables = []

        for table in tables_result.data:
            table_markdown = table.get("table_markdown", "")

            if not table_markdown:
                continue

            markdown_lower = table_markdown.lower()
            text_annotation_lower = table.get("text_annotation", "").lower()

            matches = []
            match_scores = []

            for term in query_terms:
                term_clean = term.strip().lower()

                if not term_clean or len(term_clean) < 2:
                    continue

                if term_clean in markdown_lower:
                    matches.append(term_clean)
                    match_scores.append(len(term_clean))
                    continue

                term_parts = term_clean.split()
                if len(term_parts) > 1:
                    if all(part in markdown_lower or part in text_annotation_lower for part in term_parts):
                        matches.append(term_clean)
                        match_scores.append(len(term_clean) * 0.8)

            if matches:
                total_match_score = sum(match_scores)
                match_count = len(matches)

                relevance_score = min(0.90 + (match_count * 0.02) + (total_match_score / 1000), 0.99)

                matched_tables.append({
                    "vector_id": f"term_match_{table['id']}",
                    "score": relevance_score,
                    "element_id": table["id"],
                    "document_id": table["document_id"],
                    "element_type": "table",
                    "text_annotation": table.get("text_annotation", ""),
                    "page": table.get("page_number", 1),
                    "file_path": table.get("file_path"),
                    "metadata": {
                        **table.get("metadata", {}),
                        "matched_terms": matches,
                        "match_count": match_count,
                        "match_method": "hybrid_term_search"
                    }
                })

                logger.info(f"Table on page {table.get('page_number')} matches {match_count} terms: {matches[:5]}")

        matched_tables.sort(key=lambda x: x["score"], reverse=True)

        return matched_tables[:8]

    except Exception as e:
        logger.error(f"Universal table search failed: {e}")
        return []


def _is_visual_query(query: str) -> bool:
    """
    Enhanced detection of visual queries using comprehensive keyword analysis.

    Detects queries asking for visual elements (graphs, charts, figures, images)
    with expanded keyword coverage and semantic analysis.

    Args:
        query: User query string

    Returns:
        True if query is requesting a visual element
    """
    query_lower = query.lower()

    # Comprehensive visual keywords organized by category
    visual_keywords = [
        # Direct visual terms
        'graph', 'chart', 'plot', 'figure', 'diagram', 'image',
        'visualization', 'visual', 'picture', 'illustration',
        'graphic', 'infographic', 'schematic',

        # Chart types
        'bar chart', 'line graph', 'pie chart', 'scatter plot',
        'histogram', 'heatmap', 'box plot', 'violin plot',
        'area chart', 'bubble chart', 'radar chart', 'treemap',
        'sankey diagram', 'network graph', 'flowchart',

        # Visual actions
        'show me', 'display', 'illustrate', 'visualize',
        'plot', 'chart', 'graph', 'draw', 'sketch',

        # Visual references
        'give me the graph', 'give me the chart', 'give me the figure',
        'show the plot', 'display the chart', 'view the graph',

        # Data visualization
        'trend', 'pattern', 'distribution', 'correlation',
        'comparison', 'relationship', 'progression',

        # Visual context
        'looks like', 'appears as', 'represented by',
        'depicted in', 'shown in', 'displayed as'
    ]

    # Check for exact keyword matches
    for keyword in visual_keywords:
        if keyword in query_lower:
            return True

    # Check for visual question patterns
    visual_question_patterns = [
        'what does.*look like',
        'how does.*appear',
        'can you show',
        'is there a.*showing',
        'does the.*display',
        'where is the.*graph',
        'find the.*chart'
    ]

    import re
    for pattern in visual_question_patterns:
        if re.search(pattern, query_lower):
            return True

    if any(ext in query_lower for ext in ['.png', '.jpg', '.jpeg', '.svg', '.pdf']):
        return True

    return False


def _is_statistical_query(query: str) -> bool:
    """
    Detect if a query is asking for statistical/numerical data from tables.

    Args:
        query: User query string

    Returns:
        True if query contains statistical keywords
    """
    query_lower = query.lower()

    # Statistical keywords that indicate the query needs table data
    statistical_keywords = [
        'p-value', 'p value', 'pvalue', 'p.value',
        'coefficient', 'coef', 'beta',
        'correlation', 'r-squared', 'r2', 'r squared',
        'mean', 'median', 'average', 'std', 'standard deviation',
        'confidence interval', 'ci', 'conf',
        't-statistic', 't-stat', 't value',
        'z-score', 'chi-square',
        'odds ratio', 'hazard ratio',
        'regression', 'model',
        'estimate', 'estimator',
        'significance', 'significant',
        'log(', 'ln(', 'exp(',
        'table', 'row', 'column',
        'value of', 'value for',
        'what is the', 'what are the',
        'find',
        'number', 'percentage', 'rate'
    ]

    # Check if query contains any statistical keywords
    for keyword in statistical_keywords:
        if keyword in query_lower:
            return True

    # Check for numeric patterns (e.g., "0.05", "95%")
    import re
    if re.search(r'\d+\.?\d*%?', query_lower):
        return True

    return False


def _wants_figure_or_image(query: str) -> bool:
    """
    Detect if query specifically wants figures/images (not tables).
    
    Args:
        query: User query string
        
    Returns:
        True if query is asking for figures, images, charts, or visualizations
    """
    query_lower = query.lower()
    
    # Keywords that indicate user wants images/figures, NOT tables
    figure_keywords = [
        'figure', 'image', 'picture', 'diagram', 'illustration',
        'chart', 'graph', 'plot', 'visualization', 'visual',
        'schematic', 'flowchart', 'overview diagram', 'architecture'
    ]
    
    return any(kw in query_lower for kw in figure_keywords)


@router.post("/upload", response_model=AdvancedUploadResponse)
async def upload_pdf_advanced(
    file: UploadFile = File(...),
    supabase: Client = Depends(get_supabase),
    pdf_processor: AdvancedPDFProcessor = Depends(get_advanced_pdf_processor),
    table_processor: TableProcessor = Depends(get_table_processor),
    image_processor: ImageProcessor = Depends(get_image_processor),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    qdrant_service: QdrantAdvancedService = Depends(get_qdrant_advanced),
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Upload and process PDF for advanced RAG with multimodal support.
    
    - Extracts text, tables, and images using Docling
    - Creates semantic chunks
    - Processes visual elements with LLM-generated descriptions
    - Stores in advanced collections
    """
    try:
        sanitized_filename = FileValidator.validate_and_sanitize(file)
        logger.info(f"Processing advanced upload: {sanitized_filename}")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Extract document with structure
            logger.info("Extracting document with Docling...")
            extracted_data = pdf_processor.extract_document(tmp_path)
            
            # Create document record
            doc_data = {
                "filename": sanitized_filename,
                "file_path": None,
                "total_pages": extracted_data["total_pages"],
                "processing_status": "processing",
                "ingestion_type": "advanced",
                "metadata": extracted_data["metadata"]
            }
            
            result = supabase.table("documents").insert(doc_data).execute()
            document_id = result.data[0]["id"]
            logger.info(f"Created document record: {document_id}")
            
            # Store PDF in Supabase Storage
            logger.info("Storing PDF in Supabase Storage...")
            pdf_storage_path = storage.save_pdf(
                document_id=document_id,
                filename=sanitized_filename,
                pdf_data=content
            )
            
            # Update document with file path
            supabase.table("documents").update({
                "file_path": pdf_storage_path
            }).eq("id", document_id).execute()
            
            # Process visual elements
            visual_elements_data = []
            tables_count = 0
            images_count = 0
            
            # Process tables with image extraction
            for table_data in extracted_data.get("tables", []):
                element_id = str(uuid.uuid4())
                processed_table = table_processor.process_table(
                    table_data=table_data,
                    document_id=document_id,
                    element_id=element_id,
                    page_number=table_data.get("page", 1),
                    pdf_path=str(tmp_path)  # Pass PDF path for image extraction
                )

                visual_element = {
                    "id": element_id,
                    "document_id": document_id,
                    "element_type": "table",
                    "page_number": processed_table["page_number"],
                    "bounding_box": None,
                    "file_path": processed_table.get("file_path"),  # Table image path
                    "table_markdown": processed_table.get("table_markdown"),
                    "text_annotation": processed_table["text_annotation"],
                    "ingestion_type": "advanced",
                    "metadata": processed_table.get("metadata", {})
                }
                visual_elements_data.append(visual_element)
                tables_count += 1
            
            # Process images with caption-based descriptions
            for image_data in extracted_data.get("images", []):
                element_id = str(uuid.uuid4())
                processed_image = image_processor.process_image(
                    image_data=image_data,
                    document_id=document_id,
                    element_id=element_id,
                    page_number=image_data.get("page", 1)
                )

                visual_element = {
                    "id": element_id,
                    "document_id": document_id,
                    "element_type": processed_image["element_type"],
                    "page_number": processed_image["page_number"],
                    "bounding_box": processed_image.get("bounding_box"),
                    "file_path": processed_image.get("file_path"),
                    "table_markdown": None,
                    "text_annotation": processed_image["text_annotation"],
                    "ingestion_type": "advanced",
                    "metadata": processed_image.get("metadata", {})
                }
                visual_elements_data.append(visual_element)
                images_count += 1
            
            # Store visual elements in database
            if visual_elements_data:
                supabase.table("visual_elements").insert(visual_elements_data).execute()
                logger.info(f"Stored {len(visual_elements_data)} visual elements")
            
            # Create semantic chunks with table enrichment
            logger.info("Creating semantic chunks with table enrichment...")
            chunker = SemanticChunker(
                min_chunk_size=500,
                max_chunk_size=1500,
                table_context_window=500
            )
            chunks = chunker.chunk_document(
                sections=extracted_data.get("sections", []),
                tables=extracted_data.get("tables", []),
                preserve_structure=True
            )
            
            if not chunks:
                raise ValueError("No chunks created from document")
            
            # Store chunks in database
            chunk_records = []
            for chunk in chunks:
                # Link to visual elements if referenced
                visual_refs = chunk.get("metadata", {}).get("visual_refs", [])
                
                chunk_record = {
                    "document_id": document_id,
                    "chunk_text": chunk["text"],
                    "chunk_index": chunk["chunk_index"],
                    "page_number": chunk["page_number"],
                    "chunk_type": chunk.get("chunk_type", "text"),
                    "ingestion_type": "advanced",
                    "visual_element_ids": visual_refs,
                    "metadata": chunk.get("metadata", {})
                }
                chunk_records.append(chunk_record)
            
            chunks_result = supabase.table("chunks").insert(chunk_records).execute()
            stored_chunks = chunks_result.data
            logger.info(f"Stored {len(stored_chunks)} chunks")
            
            # Generate embeddings for text chunks
            logger.info("Generating text embeddings...")
            texts = [chunk["text"] for chunk in chunks]
            text_embeddings = embedding_service.embed_batch(texts)
            
            # Prepare chunks with IDs
            chunks_with_ids = []
            for chunk, stored_chunk in zip(chunks, stored_chunks):
                chunk_copy = chunk.copy()
                chunk_copy["chunk_id"] = stored_chunk["id"]
                chunk_copy["visual_element_ids"] = stored_chunk.get("visual_element_ids", [])
                chunks_with_ids.append(chunk_copy)
            
            # Store text chunks in Qdrant
            logger.info("Storing text vectors...")
            text_vector_ids = qdrant_service.insert_text_chunks(
                chunks=chunks_with_ids,
                embeddings=text_embeddings,
                document_id=document_id
            )
            
            # Store text embedding records
            text_embedding_records = []
            for chunk_id, vector_id in zip([c["id"] for c in stored_chunks], text_vector_ids):
                text_embedding_records.append({
                    "chunk_id": chunk_id,
                    "visual_element_id": None,
                    "collection_name": "advanced_text_collection",
                    "vector_id": vector_id,
                    "embedding_model": embedding_service.get_model_name(),
                    "ingestion_type": "advanced"
                })
            
            supabase.table("embeddings").insert(text_embedding_records).execute()
            
            # Generate embeddings for visual elements
            if visual_elements_data:
                logger.info("Generating visual embeddings...")
                visual_texts = [ve["text_annotation"] for ve in visual_elements_data]
                visual_embeddings = embedding_service.embed_batch(visual_texts)
                
                # Prepare visual elements with IDs
                visual_with_ids = []
                for ve in visual_elements_data:
                    visual_with_ids.append({
                        "element_id": ve["id"],
                        "element_type": ve["element_type"],
                        "text_annotation": ve["text_annotation"],
                        "page_number": ve["page_number"],
                        "file_path": ve.get("file_path", ""),
                        "metadata": ve.get("metadata", {})
                    })
                
                # Store visual vectors
                logger.info("Storing visual vectors...")
                visual_vector_ids = qdrant_service.insert_visual_elements(
                    visual_elements=visual_with_ids,
                    embeddings=visual_embeddings,
                    document_id=document_id
                )
                
                # Store visual embedding records
                visual_embedding_records = []
                for ve_id, vector_id in zip([ve["id"] for ve in visual_elements_data], visual_vector_ids):
                    visual_embedding_records.append({
                        "chunk_id": None,
                        "visual_element_id": ve_id,
                        "collection_name": "advanced_visual_collection",
                        "vector_id": vector_id,
                        "embedding_model": embedding_service.get_model_name(),
                        "ingestion_type": "advanced"
                    })
                
                supabase.table("embeddings").insert(visual_embedding_records).execute()
            
            # Update document status
            supabase.table("documents").update({
                "processing_status": "completed"
            }).eq("id", document_id).execute()
            
            logger.success(f"Successfully processed advanced document {document_id}")
            
            return AdvancedUploadResponse(
                document_id=document_id,
                filename=sanitized_filename,
                total_pages=extracted_data["total_pages"],
                chunks_created=len(stored_chunks),
                visual_elements_count=len(visual_elements_data),
                tables_extracted=tables_count,
                images_extracted=images_count,
                ingestion_type="advanced",
                processing_status="completed",
                message="Document processed successfully with multimodal extraction"
            )
            
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
    
    except Exception as e:
        logger.error(f"Advanced upload failed: {e}")
        
        # Update document status if created
        if 'document_id' in locals():
            try:
                supabase.table("documents").update({
                    "processing_status": "failed",
                    "error_message": str(e)
                }).eq("id", document_id).execute()
            except:
                pass
        
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=AdvancedQueryResponse)
async def query_advanced(
    request: QueryRequest,
    supabase: Client = Depends(get_supabase),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    qdrant_service: QdrantAdvancedService = Depends(get_qdrant_advanced),
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Query the advanced RAG system with multimodal support.
    
    - Searches both text and visual collections
    - Retrieves tables and images
    - Generates comprehensive answer
    """
    try:
        start_time = time.time()
        
        logger.info(f"Processing advanced query: {request.query[:100]}...")
        
        # Embed query
        query_embedding = embedding_service.embed_text(request.query)

        # Detect query type to optimize retrieval
        is_visual_query = _is_visual_query(request.query)
        is_statistical_query = _is_statistical_query(request.query)

        top_k = request.top_k or 10

        if is_visual_query:
            text_k = min(top_k, 5)
            visual_k = min(top_k * 2, 20)
            logger.info(f"Visual query detected (graph/chart/image), retrieving {visual_k} visual candidates")
        elif is_statistical_query:
            text_k = min(top_k * 3, 30)
            visual_k = max(int(top_k * 0.6), 8)
            logger.info(f"Statistical query detected, retrieving {text_k} text and {visual_k} visual candidates")
        else:
            text_k = min(top_k * 2, 20)
            visual_k = max(int(top_k * 0.4), 5)

        text_results = qdrant_service.search_text(
            query_embedding=query_embedding,
            top_k=text_k
        )

        # Determine element types to search based on query
        visual_element_types = None
        if _wants_figure_or_image(request.query):
            visual_element_types = ["image", "figure", "chart"]
            logger.info("Filtering visual search to images/figures/charts only (excluding tables)")

        # Search visual collection
        visual_results = qdrant_service.search_visual(
            query_embedding=query_embedding,
            top_k=visual_k,
            element_types=visual_element_types
        )

        boost_factor = 1.5 if is_statistical_query else 1.2

        for result in text_results:
            chunk_id = result.get("chunk_id")
            if chunk_id:
                try:
                    chunk_data = supabase.table("chunks").select("chunk_type, metadata").eq("id", chunk_id).execute()
                    if chunk_data.data and len(chunk_data.data) > 0:
                        chunk_type = chunk_data.data[0].get("chunk_type", "")
                        if "table" in chunk_type:
                            result["score"] = result["score"] * boost_factor
                            logger.debug(f"Boosted table chunk score by {boost_factor}x to {result['score']}")
                except Exception as e:
                    logger.warning(f"Failed to fetch chunk type: {e}")

        text_results = sorted(text_results, key=lambda x: x["score"], reverse=True)[:top_k]

        if visual_element_types is None:
            logger.info("Applying universal hybrid table search")
            query_terms = _extract_query_terms(request.query)
            logger.info(f"Extracted {len(query_terms)} terms from query: {query_terms[:10]}")

            top_doc_ids = list(set([r["document_id"] for r in text_results[:5]]))

            if top_doc_ids and query_terms:
                term_matched_tables = _search_tables_by_terms(
                    supabase=supabase,
                    document_ids=top_doc_ids,
                    query_terms=query_terms
                )

                if term_matched_tables:
                    logger.success(f"Found {len(term_matched_tables)} tables via hybrid term search!")
                    for table in term_matched_tables:
                        existing_ids = {v.get("element_id") for v in visual_results}
                        if table["element_id"] not in existing_ids:
                            visual_results.insert(0, table)
        else:
            logger.info(f"Skipping table hybrid search - user specifically requested: {visual_element_types}")

        if visual_results:
            visual_results.sort(key=lambda x: x["score"], reverse=True)
            logger.info(f"Sorted {len(visual_results)} visual results by relevance score (top score: {visual_results[0]['score']:.3f})")

        if not text_results and not visual_results:
            return AdvancedQueryResponse(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                visual_elements=[],
                ingestion_type="advanced",
                query_time_ms=int((time.time() - start_time) * 1000)
            )
        
        all_doc_ids = list(set(
            [r["document_id"] for r in text_results] +
            [r["document_id"] for r in visual_results]
        ))
        docs_result = supabase.table("documents").select("id, filename").in_("id", all_doc_ids).execute()
        doc_map = {doc["id"]: doc["filename"] for doc in docs_result.data}
        
        visual_element_ids = [r["element_id"] for r in visual_results]
        visual_details = {}
        if visual_element_ids:
            ve_result = supabase.table("visual_elements").select("*").in_("id", visual_element_ids).execute()
            visual_details = {ve["id"]: ve for ve in ve_result.data}
        
        context_chunks = []
        for idx, result in enumerate(text_results, 1):
            chunk_text = result["text"]

            source_prefix = f"[Source {idx}, Page {result['page']}, {doc_map.get(result['document_id'], 'Unknown')}]\n"

            context_chunks.append({
                "text": source_prefix + chunk_text,
                "page": result["page"],
                "document_name": doc_map.get(result["document_id"], "Unknown"),
                "score": result.get("score", 0)
            })
        
        visual_elements_for_llm = []
        for result in visual_results:
            ve_detail = visual_details.get(result["element_id"], {})
            visual_elements_for_llm.append({
                "element_type": result["element_type"],
                "description": result["text_annotation"],
                "page_number": result["page"],
                "table_markdown": ve_detail.get("table_markdown"),
                "document_name": doc_map.get(result["document_id"], "Unknown")
            })
        
        logger.info("Generating multimodal answer...")
        answer = llm_service.generate_answer(
            question=request.query,
            context_chunks=context_chunks,
            visual_elements=visual_elements_for_llm
        )

        sources = []
        for result in text_results:
            sources.append(SourceReference(
                chunk_id=result["chunk_id"],
                document_id=result["document_id"],
                document_name=doc_map.get(result["document_id"], "Unknown"),
                page_number=result["page"],
                text_snippet=result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                relevance_score=result["score"],
                chunk_type=result.get("chunk_type")
            ))
        
        # Filter by minimum relevance score - only return truly relevant visuals
        MIN_VISUAL_RELEVANCE_SCORE = 0.3  # Configurable threshold
        filtered_visual_results = [
            r for r in visual_results 
            if r["score"] >= MIN_VISUAL_RELEVANCE_SCORE
        ]
        
        # Limit to top 5 relevant results (or 0 if none are relevant)
        MAX_VISUAL_RESULTS = 5
        filtered_visual_results = filtered_visual_results[:MAX_VISUAL_RESULTS]
        
        logger.info(f"Filtered from {len(visual_results)} to {len(filtered_visual_results)} visual elements (min score: {MIN_VISUAL_RELEVANCE_SCORE})")
        
        visual_refs = []
        for result in filtered_visual_results:
            ve_detail = visual_details.get(result["element_id"], {})
            
            image_url = None
            file_path = ve_detail.get("file_path")
            if file_path:
                if result["element_type"] in ["image", "figure", "chart"]:
                    image_url = storage.get_signed_url(file_path, expires_in=3600)
                elif result["element_type"] == "table":
                    metadata = ve_detail.get("metadata", {})
                    if metadata.get("image_path"):
                        image_url = storage.get_signed_url(metadata["image_path"], expires_in=3600)
                    elif file_path.endswith(('.png', '.jpg', '.jpeg')):
                        image_url = storage.get_signed_url(file_path, expires_in=3600)
            
            visual_refs.append(VisualReference(
                element_id=result["element_id"],
                element_type=result["element_type"],
                document_id=result["document_id"],
                page_number=result["page"],
                description=result["text_annotation"],
                file_path=file_path,
                image_url=image_url,
                table_markdown=ve_detail.get("table_markdown"),
                relevance_score=result["score"]
            ))
        
        query_time_ms = int((time.time() - start_time) * 1000)
        
        try:
            supabase.table("query_logs").insert({
                "query_text": request.query,
                "ingestion_type": "advanced",
                "collection_searched": ["advanced_text_collection", "advanced_visual_collection"],
                "retrieved_chunk_ids": [r["chunk_id"] for r in text_results],
                "response_time_ms": query_time_ms
            }).execute()
        except Exception as log_error:
            logger.warning(f"Failed to log query: {log_error}")
        
        logger.success(f"Advanced query completed in {query_time_ms}ms")

        return AdvancedQueryResponse(
            answer=answer,
            sources=sources,
            visual_elements=visual_refs,
            ingestion_type="advanced",
            query_time_ms=query_time_ms
        )
    
    except Exception as e:
        logger.error(f"Advanced query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents", response_model=DocumentListResponse)
async def list_advanced_documents(
    supabase: Client = Depends(get_supabase)
):
    """
    List all documents in the advanced collection with multimodal statistics.
    
    Returns document metadata and collection-level insights including visual elements.
    """
    try:
        logger.info("Fetching advanced collection documents")
        
        docs_result = supabase.table("documents").select(
            "id, filename, created_at, total_pages, processing_status, ingestion_type"
        ).eq("ingestion_type", "advanced").order("created_at", desc=True).execute()
        
        documents = []
        total_chunks = 0
        total_pages = 0
        total_tables = 0
        total_images = 0
        total_visual_elements = 0
        
        for doc in docs_result.data:
            chunks_result = supabase.table("chunks").select(
                "id", count="exact"
            ).eq("document_id", doc["id"]).eq("ingestion_type", "advanced").execute()
            
            visual_result = supabase.table("visual_elements").select(
                "element_type"
            ).eq("document_id", doc["id"]).execute()
            
            chunk_count = chunks_result.count or 0
            tables_count = sum(1 for ve in visual_result.data if ve["element_type"] == "table")
            images_count = len(visual_result.data) - tables_count
            visual_count = len(visual_result.data)
            
            total_chunks += chunk_count
            total_pages += doc["total_pages"]
            total_tables += tables_count
            total_images += images_count
            total_visual_elements += visual_count
            
            documents.append(DocumentInfo(
                id=doc["id"],
                filename=doc["filename"],
                upload_date=doc["created_at"],
                total_pages=doc["total_pages"],
                processing_status=doc["processing_status"],
                ingestion_type=doc["ingestion_type"],
                chunks_count=chunk_count,
                visual_elements_count=visual_count,
                tables_extracted=tables_count,
                images_extracted=images_count
            ))
        
        insights = {
            "total_documents": len(documents),
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "total_visual_elements": total_visual_elements,
            "total_tables": total_tables,
            "total_images": total_images,
            "avg_visual_per_doc": round(total_visual_elements / len(documents), 1) if documents else 0
        }
        
        logger.success(f"Retrieved {len(documents)} advanced documents")
        
        return DocumentListResponse(
            documents=documents,
            total=len(documents),
            insights=insights
        )
    
    except Exception as e:
        logger.error(f"Failed to list advanced documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/pdf")
async def get_document_pdf(
    document_id: str,
    supabase: Client = Depends(get_supabase)
):
    """
    Get PDF URL for an advanced document.
    
    Returns the public URL to view/download the PDF.
    """
    try:
        logger.info(f"Fetching PDF for advanced document: {document_id}")
        
        doc_result = supabase.table("documents").select(
            "id, filename, file_path, ingestion_type"
        ).eq("id", document_id).eq("ingestion_type", "advanced").execute()
        
        if not doc_result.data:
            raise HTTPException(status_code=404, detail="Document not found")
        
        document = doc_result.data[0]
        file_path = document.get("file_path")
        
        if not file_path:
            raise HTTPException(status_code=404, detail="PDF file not found for this document")
        
        pdf_url = storage.get_signed_url(file_path, expires_in=3600)
        
        return {
            "document_id": document_id,
            "filename": document["filename"],
            "pdf_url": pdf_url,
            "file_path": file_path
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get PDF URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/images")
async def get_document_images(
    document_id: str,
    supabase: Client = Depends(get_supabase)
):
    """
    Get all image elements for an advanced document.
    
    Returns list of images with URLs and descriptions.
    """
    try:
        logger.info(f"Fetching images for document: {document_id}")
        
        doc_result = supabase.table("documents").select(
            "id, filename, ingestion_type"
        ).eq("id", document_id).eq("ingestion_type", "advanced").execute()
        
        if not doc_result.data:
            raise HTTPException(status_code=404, detail="Document not found")
        
        visual_result = supabase.table("visual_elements").select(
            "id, element_type, page_number, file_path, text_annotation, metadata"
        ).eq("document_id", document_id).in_("element_type", ["image", "figure", "chart"]).execute()
        
        images = []
        for ve in visual_result.data:
            image_url = None
            if ve.get("file_path"):
                image_url = storage.get_signed_url(ve["file_path"], expires_in=3600)
            
            images.append({
                "id": ve["id"],
                "element_type": ve["element_type"],
                "page_number": ve["page_number"],
                "image_url": image_url,
                "description": ve.get("text_annotation", "No description available"),
                "metadata": ve.get("metadata", {})
            })
        
        return {
            "document_id": document_id,
            "filename": doc_result.data[0]["filename"],
            "total_images": len(images),
            "images": images
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get images: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/tables")
async def get_document_tables(
    document_id: str,
    supabase: Client = Depends(get_supabase)
):
    """
    Get all table elements for an advanced document.
    
    Returns list of tables with data and descriptions.
    """
    try:
        logger.info(f"Fetching tables for document: {document_id}")
        
        doc_result = supabase.table("documents").select(
            "id, filename, ingestion_type"
        ).eq("id", document_id).eq("ingestion_type", "advanced").execute()
        
        if not doc_result.data:
            raise HTTPException(status_code=404, detail="Document not found")
        
        visual_result = supabase.table("visual_elements").select(
            "id, element_type, page_number, file_path, text_annotation, metadata"
        ).eq("document_id", document_id).eq("element_type", "table").execute()
        
        tables = []
        for ve in visual_result.data:
            table_image_url = None
            metadata = ve.get("metadata", {})
            
            if metadata.get("image_path"):
                table_image_url = storage.get_signed_url(metadata["image_path"], expires_in=3600)
            elif ve.get("file_path") and ve["file_path"].endswith(('.png', '.jpg', '.jpeg')):
                table_image_url = storage.get_signed_url(ve["file_path"], expires_in=3600)
            
            tables.append({
                "id": ve["id"],
                "page_number": ve["page_number"],
                "description": ve.get("text_annotation", "No description available"),
                "image_url": table_image_url,
                "metadata": metadata
            })
        
        return {
            "document_id": document_id,
            "filename": doc_result.data[0]["filename"],
            "total_tables": len(tables),
            "tables": tables
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get tables: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_id}")
async def delete_advanced_document(
    document_id: str,
    supabase: Client = Depends(get_supabase),
    qdrant_service: QdrantAdvancedService = Depends(get_qdrant_advanced)
):
    """
    Delete a document and all associated data from advanced RAG system.

    Removes:
    - Document record from database
    - All chunks and embeddings
    - All visual elements (tables, images)
    - All vectors from Qdrant (both text and visual collections)
    - All files from Supabase Storage
    """
    try:
        logger.info(f"Deleting advanced document: {document_id}")

        doc_result = supabase.table("documents").select(
            "id, filename, ingestion_type"
        ).eq("id", document_id).eq("ingestion_type", "advanced").execute()

        if not doc_result.data:
            raise HTTPException(status_code=404, detail="Document not found")

        document = doc_result.data[0]
        filename = document["filename"]
        
        logger.info(f"Deleting vectors from Qdrant for document {document_id}")
        try:
            qdrant_service.delete_by_document(document_id)
        except Exception as e:
            logger.warning(f"Failed to delete from Qdrant (may not exist): {e}")
        
        logger.info(f"Deleting files from storage for document {document_id}")
        try:
            storage.delete_document_files(document_id)
        except Exception as e:
            logger.warning(f"Failed to delete storage files (may not exist): {e}")
        
        logger.info(f"Deleting embeddings for document {document_id}")
        try:
            # Get chunk IDs
            chunks_result = supabase.table("chunks").select("id").eq(
                "document_id", document_id
            ).eq("ingestion_type", "advanced").execute()
            chunk_ids = [c["id"] for c in chunks_result.data]

            # Get visual element IDs
            visual_result = supabase.table("visual_elements").select("id").eq(
                "document_id", document_id
            ).execute()
            visual_element_ids = [v["id"] for v in visual_result.data]

            # Delete chunk embeddings
            if chunk_ids:
                supabase.table("embeddings").delete().in_("chunk_id", chunk_ids).execute()

            # Delete visual embeddings
            if visual_element_ids:
                supabase.table("embeddings").delete().in_("visual_element_id", visual_element_ids).execute()
        except Exception as e:
            logger.warning(f"Failed to delete embeddings: {e}")
        
        logger.info(f"Deleting visual elements for document {document_id}")
        try:
            supabase.table("visual_elements").delete().eq("document_id", document_id).execute()
        except Exception as e:
            logger.warning(f"Failed to delete visual elements: {e}")
        
        logger.info(f"Deleting chunks for document {document_id}")
        try:
            supabase.table("chunks").delete().eq("document_id", document_id).eq(
                "ingestion_type", "advanced"
            ).execute()
        except Exception as e:
            logger.warning(f"Failed to delete chunks: {e}")
        
        logger.info(f"Deleting document record {document_id}")
        supabase.table("documents").delete().eq("id", document_id).eq(
            "ingestion_type", "advanced"
        ).execute()

        logger.success(f"Successfully deleted advanced document {document_id} ({filename})")

        return {
            "success": True,
            "message": f"Document '{filename}' deleted successfully",
            "document_id": document_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")
