"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from forma.config import get_settings
from forma.extractor import Extractor
from forma.proxy import OpenAIProxy
from forma.storage import Storage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global instances
proxy: OpenAIProxy
extractor: Extractor
storage: Storage


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global proxy, extractor, storage
    settings = get_settings()
    proxy = OpenAIProxy(settings)
    extractor = Extractor(settings)
    storage = Storage(
        host=settings.chromadb_host,
        port=settings.chromadb_port,
        persist_directory=settings.chromadb_persist_directory,
    )
    logger.info(f"Forma proxy starting - upstream: {settings.upstream_base_url}")
    if settings.embedding_base_url:
        logger.info(
            f"Embedding endpoint: {settings.embedding_base_url} (model: {settings.embedding_model_name})"
        )
    else:
        logger.info("Embeddings will use upstream endpoint")
    if settings.extractor_base_url:
        logger.info(
            f"Extraction endpoint: {settings.extractor_base_url} (model: {settings.extractor_model_name})"
        )
    else:
        logger.info("Extraction will use upstream endpoint")
    # Log storage stats
    stats = storage.get_collection_stats()
    logger.info(f"ChromaDB collections: facts={stats['facts']}, recipes={stats['recipes']}")
    yield
    logger.info("Forma proxy shutting down")


app = FastAPI(
    title="Forma",
    description="Autonomous Cognitive Proxy - OpenAI-compatible API proxy",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check
@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


# OpenAI-compatible endpoints
@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """List available models."""
    return await proxy.list_models()


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> dict[str, Any] | StreamingResponse:
    """Create chat completion with synchronous extraction."""
    payload = await request.json()
    messages = payload.get("messages", [])

    # Synchronous extraction before forwarding
    # This allows us to modify the prompt based on extracted data later
    if messages and extractor.settings.extractor_model_name:
        try:
            logger.info(f"Extracting from {len(messages)} messages...")
            result = await extractor.extract_from_messages_async(messages)
            if result.is_valid():
                logger.info(
                    f"Extraction complete: {len(result.entities)} entities, "
                    f"{len(result.relationships)} relationships, "
                    f"{len(result.facts)} facts, {len(result.recipes)} recipes"
                )
                # Store facts and recipes in ChromaDB
                if result.facts or result.recipes:
                    facts_count, recipes_count = storage.store_extraction(
                        result.facts, result.recipes
                    )
                    logger.info(f"Stored: {facts_count} facts, {recipes_count} recipes")
            elif result.parse_error:
                logger.warning(f"Extraction parse error: {result.parse_error}")
        except Exception as e:
            # Don't fail the request if extraction fails
            logger.error(f"Extraction error: {e}")

    # Forward to upstream
    return await proxy.chat_completions(payload)


@app.post("/v1/completions", response_model=None)
async def completions(request: Request) -> dict[str, Any] | StreamingResponse:
    """Create legacy completion."""
    payload = await request.json()
    return await proxy.completions(payload)


@app.post("/v1/embeddings")
async def embeddings(request: Request) -> dict[str, Any]:
    """Create embeddings."""
    payload = await request.json()
    return await proxy.embeddings(payload)


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": "internal_error"}},
    )


def run_server() -> None:
    """Run the server (entry point for CLI)."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "forma.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    run_server()
