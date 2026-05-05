"""FastAPI application entry point."""

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
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

# Logs directory
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"

# Global instances
proxy: OpenAIProxy
extractor: Extractor
storage: Storage


def _log_context_retrieval(
    entities_queries: list[str],
    fact_query: str | None,
    recipe_query: str | None,
    context: dict[str, Any],
    augmented_prompt: str | None,
) -> None:
    """Log context retrieval to file for debugging."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "queries": {
            "entities_queries": entities_queries,
            "fact_query": fact_query,
            "recipe_query": recipe_query,
        },
        "retrieved": {
            "relationships": context.get("relationships", []),
            "facts": context.get("facts", []),
            "recipes": context.get("recipes", []),
        },
        "tokens_used": context.get("tokens_used", 0),
        "scores": context.get("scores", {}),
        "augmented_prompt": augmented_prompt,
    }
    log_file = LOGS_DIR / "retrievals.jsonl"
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
        logger.debug(
            f"Logged retrieval: {len(context.get('relationships', []))} relationships, "
            f"{len(context.get('facts', []))} facts, {len(context.get('recipes', []))} recipes, "
            f"{context.get('tokens_used', 0)} tokens"
        )
    except Exception as e:
        logger.error(f"Failed to log retrieval: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global proxy, extractor, storage
    settings = get_settings()
    proxy = OpenAIProxy(settings)
    extractor = Extractor(settings)
    storage = Storage(
        chromadb_host=settings.chromadb_host,
        chromadb_port=settings.chromadb_port,
        chromadb_persist_directory=settings.chromadb_persist_directory,
        cogdb_home=settings.cogdb_home,
        cogdb_path_prefix=settings.cogdb_path_prefix,
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
    stats = storage.get_stats()
    logger.info(
        f"Storage: ChromaDB facts={stats['chromadb']['facts']}, recipes={stats['chromadb']['recipes']}, "
        f"CogDB entities={stats['cogdb']['entities']}"
    )
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
    """Create chat completion with synchronous extraction and RAG context."""
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
                # Store all extracted data
                if result.entities or result.relationships or result.facts or result.recipes:
                    entities_count, relationships_count, facts_count, recipes_count = (
                        storage.store_extraction(
                            result.entities, result.relationships, result.facts, result.recipes
                        )
                    )
                    logger.info(
                        f"Stored: {entities_count} entities, {relationships_count} relationships, "
                        f"{facts_count} facts, {recipes_count} recipes"
                    )

                # Retrieve context if extraction has queries
                if result.has_queries():
                    context = storage.retrieve_context(
                        entities_queries=result.entities_queries,
                        fact_query=result.fact_query,
                        recipe_query=result.recipe_query,
                    )
                    if (
                        context.get("relationships")
                        or context.get("facts")
                        or context.get("recipes")
                    ):
                        context_str = storage.format_context_for_prompt(context)
                        logger.info(
                            f"Retrieved context: {len(context['relationships'])} relationships, "
                            f"{len(context['facts'])} facts, {len(context['recipes'])} recipes"
                        )
                        # Augment first user message with context
                        augmented_prompt = None
                        for msg in messages:
                            if msg.get("role") == "user":
                                content = msg.get("content", "")
                                if isinstance(content, str):
                                    augmented_prompt = context_str + content
                                    msg["content"] = augmented_prompt
                                elif isinstance(content, list):
                                    # Handle multi-modal content - prepend to first text part
                                    for part in content:
                                        if isinstance(part, dict) and part.get("type") == "text":
                                            augmented_prompt = context_str + part.get("text", "")
                                            part["text"] = augmented_prompt
                                            break
                                break

                        # Log the retrieval for debugging
                        _log_context_retrieval(
                            result.entities_queries,
                            result.fact_query,
                            result.recipe_query,
                            context,
                            augmented_prompt,
                        )
                    else:
                        # Log empty retrieval for debugging
                        _log_context_retrieval(
                            result.entities_queries,
                            result.fact_query,
                            result.recipe_query,
                            context,
                            None,
                        )

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


# Admin endpoints
@app.post("/admin/clear")
async def clear_storage() -> dict[str, Any]:
    """Clear all stored data from ChromaDB and CogDB."""
    result = storage.clear_all()
    logger.info(
        f"Storage cleared: facts={result['cleared']['facts']}, recipes={result['cleared']['recipes']}, entities={result['cleared']['entities']}"
    )
    return result


@app.get("/admin/stats")
async def get_storage_stats() -> dict[str, Any]:
    """Get storage statistics for ChromaDB and CogDB."""
    return storage.get_stats()


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
