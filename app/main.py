"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys

from app.core.config import settings
from app.api.v1 import basic_rag, advanced_rag

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level=settings.log_level
)

# Create FastAPI app
app = FastAPI(
    title="RAG Pipeline API",
    description="Dual-mode RAG system with basic and advanced document processing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(basic_rag.router, prefix="/api/v1")
app.include_router(advanced_rag.router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("=" * 60)
    logger.info("RAG Pipeline API Starting")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"LLM Model: {settings.default_llm_model}")
    logger.info(f"Embedding Model: {settings.embedding_model}")
    logger.info("=" * 60)
    
    # Warm up embedding model
    try:
        from app.services.embeddings.embedding_service import embedding_service
        embedding_service.embed_text("warmup")
        logger.success("Embedding model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load embedding model: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("RAG Pipeline API Shutting Down")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Pipeline API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "basic_rag": {
                "upload": "/api/v1/basic/upload",
                "query": "/api/v1/basic/query"
            },
            "advanced_rag": {
                "upload": "/api/v1/advanced/upload",
                "query": "/api/v1/advanced/query"
            }
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "environment": settings.environment
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development"
    )
