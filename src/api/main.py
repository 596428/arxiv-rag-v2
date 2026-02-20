"""
arXiv RAG v1 - FastAPI Main Application

REST API for hybrid vector search using Qdrant + Supabase.

Endpoints:
- POST /api/v1/search      - Hybrid vector search
- POST /api/v1/chat        - RAG chat with LLM
- GET  /api/v1/papers      - List papers
- GET  /api/v1/papers/{id} - Get paper details
- GET  /api/v1/health      - Health check
"""

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..utils.logging import get_logger
from ..storage.qdrant_client import get_qdrant_client

logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting arXiv RAG API...")

    # Initialize Qdrant client
    qdrant = get_qdrant_client()
    if qdrant.health_check():
        logger.info("Qdrant connection healthy")
    else:
        logger.warning("Qdrant connection failed - some features may be unavailable")

    yield

    # Shutdown
    logger.info("Shutting down arXiv RAG API...")
    qdrant.close()


# Create FastAPI app
app = FastAPI(
    title="arXiv RAG API",
    description="Hybrid vector search API for LLM research papers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS configuration
# Allow GitHub Pages and local development
ALLOWED_ORIGINS = [
    "https://ajh428.github.io",  # GitHub Pages
    "https://acacia.chat",       # Custom domain
    "https://api.acacia.chat",   # API domain
    "http://localhost:3000",     # Local dev
    "http://localhost:8080",     # Local dev
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
]

# In development, allow all origins
if os.getenv("ENV", "production") == "development":
    ALLOWED_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time"],
)


# Request timing middleware
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add response timing header."""
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Response-Time"] = f"{process_time:.0f}ms"
    return response


# Health check endpoint
@app.get("/api/v1/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        Service health status
    """
    qdrant = get_qdrant_client()
    qdrant_healthy = qdrant.health_check()

    status = "healthy" if qdrant_healthy else "degraded"

    return {
        "status": status,
        "version": "1.0.0",
        "services": {
            "qdrant": "healthy" if qdrant_healthy else "unhealthy",
        }
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "arXiv RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


# Import and include routers
from .routes import search, papers, chat

app.include_router(search.router, prefix="/api/v1", tags=["Search"])
app.include_router(papers.router, prefix="/api/v1", tags=["Papers"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("ENV") == "development" else None,
        }
    )
