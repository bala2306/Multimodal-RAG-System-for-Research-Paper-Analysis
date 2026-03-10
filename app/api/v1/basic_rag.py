"""Basic RAG API endpoints."""

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from typing import List
import tempfile
import time
from pathlib import Path
from loguru import logger
from supabase import Client

from app.models.schemas import (
    UploadResponse,
    QueryRequest,
    QueryResponse,
    SourceReference,
    DocumentInfo,
    DocumentListResponse
)
from app.core.database import get_supabase
from app.api.dependencies import (
    get_embedding_service,
    get_llm_service,
    get_qdrant_basic,
    get_basic_pdf_processor
)
from app.utils.validators import FileValidator
from app.utils.chunking import FixedSizeChunker
from app.services.embeddings.embedding_service import EmbeddingService
from app.services.llm.llm_service import LLMService
from app.services.vector_store.qdrant_basic import QdrantBasicService
from app.services.pdf.basic_processor import BasicPDFProcessor
from app.utils.file_storage import storage

router = APIRouter(prefix="/basic", tags=["Basic RAG"])


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    supabase: Client = Depends(get_supabase),
    pdf_processor: BasicPDFProcessor = Depends(get_basic_pdf_processor),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    qdrant_service: QdrantBasicService = Depends(get_qdrant_basic)
):
    """
    Upload and process PDF for basic RAG.
    
    - Extracts text using PyMuPDF
    - Creates fixed-size chunks
    - Generates embeddings
    - Stores in basic_rag_collection
    """
    try:
        sanitized_filename = FileValidator.validate_and_sanitize(file)
        logger.info(f"Processing upload: {sanitized_filename}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            logger.info("Extracting text from PDF...")
            extracted_data = pdf_processor.extract_text(tmp_path)
            
            doc_data = {
                "filename": sanitized_filename,
                "file_path": None,
                "total_pages": extracted_data["total_pages"],
                "processing_status": "processing",
                "ingestion_type": "basic",
                "metadata": extracted_data["metadata"]
            }
            
            result = supabase.table("documents").insert(doc_data).execute()
            document_id = result.data[0]["id"]
            logger.info(f"Created document record: {document_id}")
            
            logger.info("Storing PDF in Supabase Storage...")
            pdf_storage_path = storage.save_pdf(
                document_id=document_id,
                filename=sanitized_filename,
                pdf_data=content
            )
            
            supabase.table("documents").update({
                "file_path": pdf_storage_path
            }).eq("id", document_id).execute()
            
            logger.info("Creating chunks...")
            chunker = FixedSizeChunker(chunk_size=1000, overlap=200)
            all_chunks = []
            
            for page_data in extracted_data["pages"]:
                page_chunks = chunker.chunk_text(
                    text=page_data["text"],
                    page_number=page_data["page_number"]
                )
                all_chunks.extend(page_chunks)
            
            if not all_chunks:
                raise ValueError("No text chunks created from PDF")
            
            chunk_records = []
            for chunk in all_chunks:
                chunk_record = {
                    "document_id": document_id,
                    "chunk_text": chunk["text"],
                    "chunk_index": chunk["chunk_index"],
                    "page_number": chunk["page_number"],
                    "chunk_type": "text",
                    "ingestion_type": "basic",
                    "metadata": chunk.get("metadata", {})
                }
                chunk_records.append(chunk_record)
            
            chunks_result = supabase.table("chunks").insert(chunk_records).execute()
            stored_chunks = chunks_result.data
            logger.info(f"Stored {len(stored_chunks)} chunks in database")
            
            logger.info("Generating embeddings...")
            texts = [chunk["text"] for chunk in all_chunks]
            embeddings = embedding_service.embed_batch(texts)
            
            chunks_with_ids = []
            for chunk, stored_chunk in zip(all_chunks, stored_chunks):
                chunk_copy = chunk.copy()
                chunk_copy["chunk_id"] = stored_chunk["id"]
                chunks_with_ids.append(chunk_copy)
            
            logger.info("Storing vectors in Qdrant...")
            vector_ids = qdrant_service.insert_chunks(
                chunks=chunks_with_ids,
                embeddings=embeddings,
                document_id=document_id
            )
            
            embedding_records = []
            for chunk_id, vector_id in zip([c["id"] for c in stored_chunks], vector_ids):
                embedding_records.append({
                    "chunk_id": chunk_id,
                    "collection_name": "basic_rag_collection",
                    "vector_id": vector_id,
                    "embedding_model": embedding_service.get_model_name(),
                    "ingestion_type": "basic"
                })
            
            supabase.table("embeddings").insert(embedding_records).execute()
            
            supabase.table("documents").update({
                "processing_status": "completed"
            }).eq("id", document_id).execute()
            
            logger.success(f"Successfully processed document {document_id}")
            
            return UploadResponse(
                document_id=document_id,
                filename=sanitized_filename,
                total_pages=extracted_data["total_pages"],
                chunks_created=len(stored_chunks),
                ingestion_type="basic",
                processing_status="completed",
                message="Document processed successfully"
            )
            
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
    
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        
        if 'document_id' in locals():
            try:
                supabase.table("documents").update({
                    "processing_status": "failed",
                    "error_message": str(e)
                }).eq("id", document_id).execute()
            except:
                pass
        
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query_basic(
    request: QueryRequest,
    supabase: Client = Depends(get_supabase),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    qdrant_service: QdrantBasicService = Depends(get_qdrant_basic),
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Query the basic RAG system.
    
    - Embeds the question
    - Searches basic_rag_collection
    - Generates answer with LLM
    """
    try:
        start_time = time.time()
        
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Embed query
        query_embedding = embedding_service.embed_text(request.query)
        
        # Search Qdrant basic collection
        top_k = request.top_k or 5
        search_results = qdrant_service.search(
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        if not search_results:
            return QueryResponse(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                ingestion_type="basic",
                query_time_ms=int((time.time() - start_time) * 1000)
            )
        
        # Get document names
        doc_ids = list(set(r["document_id"] for r in search_results))
        docs_result = supabase.table("documents").select("id, filename").in_("id", doc_ids).execute()
        doc_map = {doc["id"]: doc["filename"] for doc in docs_result.data}
        
        # Prepare context for LLM
        context_chunks = []
        for result in search_results:
            context_chunks.append({
                "text": result["text"],
                "page": result["page"],
                "document_name": doc_map.get(result["document_id"], "Unknown")
            })
        
        # Generate answer
        logger.info("Generating answer...")
        answer = llm_service.generate_answer(
            question=request.query,
            context_chunks=context_chunks
        )
        
        # Prepare source references
        sources = []
        for result in search_results:
            sources.append(SourceReference(
                chunk_id=result["chunk_id"],
                document_id=result["document_id"],
                document_name=doc_map.get(result["document_id"], "Unknown"),
                page_number=result["page"],
                text_snippet=result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                relevance_score=result["score"]
            ))
        
        query_time_ms = int((time.time() - start_time) * 1000)
        
        # Log query
        try:
            supabase.table("query_logs").insert({
                "query_text": request.query,
                "ingestion_type": "basic",
                "collection_searched": ["basic_rag_collection"],
                "retrieved_chunk_ids": [r["chunk_id"] for r in search_results],
                "response_time_ms": query_time_ms
            }).execute()
        except Exception as log_error:
            logger.warning(f"Failed to log query: {log_error}")
        
        logger.success(f"Query completed in {query_time_ms}ms")
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            ingestion_type="basic",
            query_time_ms=query_time_ms
        )
    
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents", response_model=DocumentListResponse)
async def list_basic_documents(
    supabase: Client = Depends(get_supabase)
):
    """
    List all documents in the basic collection with statistics.
    
    Returns document metadata and collection-level insights.
    """
    try:
        logger.info("Fetching basic collection documents")
        
        # Get all basic documents
        docs_result = supabase.table("documents").select(
            "id, filename, created_at, total_pages, processing_status, ingestion_type"
        ).eq("ingestion_type", "basic").order("created_at", desc=True).execute()
        
        documents = []
        total_chunks = 0
        total_pages = 0
        
        for doc in docs_result.data:
            # Get chunk count for this document
            chunks_result = supabase.table("chunks").select(
                "id", count="exact"
            ).eq("document_id", doc["id"]).eq("ingestion_type", "basic").execute()
            
            chunk_count = chunks_result.count or 0
            total_chunks += chunk_count
            total_pages += doc["total_pages"]
            
            documents.append(DocumentInfo(
                id=doc["id"],
                filename=doc["filename"],
                upload_date=doc["created_at"],
                total_pages=doc["total_pages"],
                processing_status=doc["processing_status"],
                ingestion_type=doc["ingestion_type"],
                chunks_count=chunk_count
            ))
        
        # Calculate insights
        insights = {
            "total_documents": len(documents),
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "avg_chunks_per_doc": round(total_chunks / len(documents), 1) if documents else 0
        }
        
        logger.success(f"Retrieved {len(documents)} basic documents")
        
        return DocumentListResponse(
            documents=documents,
            total=len(documents),
            insights=insights
        )
    
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/pdf")
async def get_document_pdf(
    document_id: str,
    supabase: Client = Depends(get_supabase)
):
    """
    Get PDF URL for a basic document.

    Returns the public URL to view/download the PDF.
    """
    try:
        logger.info(f"Fetching PDF for document: {document_id}")

        # Get document record
        doc_result = supabase.table("documents").select(
            "id, filename, file_path, ingestion_type"
        ).eq("id", document_id).eq("ingestion_type", "basic").execute()

        if not doc_result.data:
            raise HTTPException(status_code=404, detail="Document not found")

        document = doc_result.data[0]
        file_path = document.get("file_path")

        if not file_path:
            raise HTTPException(status_code=404, detail="PDF file not found for this document")

        # Get signed URL from storage (valid for 1 hour)
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


@router.delete("/documents/{document_id}")
async def delete_basic_document(
    document_id: str,
    supabase: Client = Depends(get_supabase),
    qdrant_service: QdrantBasicService = Depends(get_qdrant_basic)
):
    """
    Delete a document and all associated data from basic RAG system.

    Removes:
    - Document record from database
    - All chunks and embeddings
    - All vectors from Qdrant
    - All files from Supabase Storage
    """
    try:
        logger.info(f"Deleting basic document: {document_id}")
        
        doc_result = supabase.table("documents").select(
            "id, filename, ingestion_type"
        ).eq("id", document_id).eq("ingestion_type", "basic").execute()

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
            chunks_result = supabase.table("chunks").select("id").eq(
                "document_id", document_id
            ).eq("ingestion_type", "basic").execute()

            chunk_ids = [c["id"] for c in chunks_result.data]

            if chunk_ids:
                supabase.table("embeddings").delete().in_("chunk_id", chunk_ids).execute()
        except Exception as e:
            logger.warning(f"Failed to delete embeddings: {e}")

        logger.info(f"Deleting chunks for document {document_id}")
        try:
            supabase.table("chunks").delete().eq("document_id", document_id).eq(
                "ingestion_type", "basic"
            ).execute()
        except Exception as e:
            logger.warning(f"Failed to delete chunks: {e}")

        logger.info(f"Deleting document record {document_id}")
        supabase.table("documents").delete().eq("id", document_id).eq(
            "ingestion_type", "basic"
        ).execute()

        logger.success(f"Successfully deleted document {document_id} ({filename})")

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
