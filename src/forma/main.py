"""FastAPI application entry point."""

import asyncio
import contextlib
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from forma.api import router as ui_router
from forma.config import get_settings
from forma.extractor import Extractor
from forma.proxy import OpenAIProxy
from forma.storage import Storage
from forma.tracker import RequestTracker, get_tracker

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
tracker: RequestTracker
_storage_lock = asyncio.Lock()


_retrieval_log_file: Any = None


def _ensure_retrieval_log() -> Any:
    """Open persistent retrieval log file handle."""
    global _retrieval_log_file
    if _retrieval_log_file is None:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        _retrieval_log_file = open(  # noqa: SIM115
            LOGS_DIR / "retrievals.jsonl", "a", encoding="utf-8"
        )
    return _retrieval_log_file


def _log_context_retrieval(
    entities_queries: list[str],
    fact_query: str | None,
    recipe_query: str | None,
    context: dict[str, Any],
    augmented_prompt: str | None,
) -> None:
    """Log context retrieval to file for debugging."""
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
    try:
        f = _ensure_retrieval_log()
        f.write(json.dumps(log_entry) + "\n")
        f.flush()
    except Exception as e:
        logger.error(f"Failed to log retrieval: {e}")


async def _store_extraction_background(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    recipes: list[dict[str, Any]],
) -> None:
    """Background task to store extracted data."""
    async with _storage_lock:
        try:
            if entities or relationships or facts or recipes:
                storage.store_extraction(entities, relationships, facts, recipes)
                logger.info(
                    f"Stored (background): {len(entities)} entities, "
                    f"{len(relationships)} relationships, "
                    f"{len(facts)} facts, {len(recipes)} recipes"
                )
        except Exception as e:
            logger.error(f"Background storage error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Manage application lifespan."""
    global proxy, extractor, storage, tracker
    settings = get_settings()
    proxy = OpenAIProxy(settings)
    extractor = Extractor(settings, proxy=proxy)
    storage = Storage(
        grafitodb_path=settings.grafitodb_path,
        grafitodb_embedding_model=settings.grafitodb_embedding_model,
        grafitodb_vector_dim=settings.grafitodb_vector_dim,
        grafitodb_model_cache_path=settings.grafitodb_model_cache_path,
    )

    # Initialize tracker for web UI
    if settings.tracker_enabled:
        tracker = RequestTracker(
            db_path=settings.tracker_db_path,
            max_records=settings.tracker_max_records,
        )
    else:
        tracker = None

    logger.info(f"Forma proxy starting - upstream: {settings.upstream_base_url}")
    if settings.extractor_base_url:
        logger.info(
            f"Extraction endpoint: {settings.extractor_base_url} "
            f"(model: {settings.extractor_model_name})"
        )

    if tracker:
        logger.info(f"Request tracking enabled - max records: {settings.tracker_max_records}")

    # Log storage stats
    stats = storage.get_stats()
    logger.info(
        f"Storage: GrafitoDB entities={stats['grafitodb']['entities']}, "
        f"relationships={stats['grafitodb']['relationships']}, "
        f"facts={stats['grafitodb']['facts']}, "
        f"recipes={stats['grafitodb']['recipes']}"
    )
    yield
    logger.info("Forma proxy shutting down")
    await proxy.close()
    extractor.close()
    storage.close()
    if tracker:
        tracker.close()
    if _retrieval_log_file is not None:
        with contextlib.suppress(Exception):
            _retrieval_log_file.close()


app = FastAPI(
    title="Forma",
    description="Autonomous Cognitive Proxy - OpenAI-compatible API proxy",
    version="0.1.0",
    lifespan=lifespan,
)

# Include UI API router
app.include_router(ui_router)

# CORS middleware for browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve SPA static files (if webui_dist exists)
WEBUI_DIST = Path(__file__).parent.parent.parent / "webui_dist"
if WEBUI_DIST.exists():
    assets_dir = WEBUI_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    favicon_path = WEBUI_DIST / "favicon.svg"
    if favicon_path.exists():

        @app.get("/favicon.svg")
        async def favicon() -> FileResponse:
            """Serve favicon."""
            return FileResponse(str(favicon_path))


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


def _get_user_prompt(messages: list[dict]) -> str:
    """Extract the user prompt from messages."""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "")
    return ""


def _get_agent_response(response: dict) -> str:
    """Extract the agent response from the API response."""
    for choice in response.get("choices", []):
        msg = choice.get("message", {})
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
    return ""


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> dict[str, Any] | StreamingResponse:
    """
    Create chat completion with extraction, retrieval, and RAG context.

    Pipeline:
    1. Extract entities/facts/recipes/queries from messages
    2. Retrieve context from storage using extracted queries
    3. Augment prompt with retrieved context
    4. Forward to upstream model
    5. Store extracted data in background (async, fire-and-forget)
    6. Track request for web UI
    """
    payload = await request.json()
    messages = payload.get("messages", [])
    model = payload.get("model", "")

    start_time = time.time()

    extraction_result = None
    retrieval_context = None
    augmented_prompt = ""
    extraction_response = ""
    extraction_latency = 0.0
    retrieval_results = []

    # Get user prompt for tracking
    user_prompt = _get_user_prompt(messages)

    # Step 1: Extract entities, facts, recipes, and queries
    if messages and extractor.settings.extractor_model_name:
        try:
            extraction_start = time.time()
            logger.info(f"Extracting from {len(messages)} messages...")
            result = await extractor.extract_from_messages_async(messages)
            extraction_latency = (time.time() - extraction_start) * 1000
            extraction_response = result.raw_response

            if result.is_valid():
                logger.info(
                    f"Extraction complete: {len(result.entities)} entities, "
                    f"{len(result.relationships)} relationships, "
                    f"{len(result.facts)} facts, {len(result.recipes)} recipes"
                )
                extraction_result = result
            elif result.parse_error:
                logger.warning(f"Extraction parse error: {result.parse_error}")
        except Exception as e:
            extraction_latency = (time.time() - extraction_start) * 1000
            logger.error(f"Extraction error: {e}")

    # Step 2: Retrieve context from storage using extracted queries
    if extraction_result and extraction_result.has_queries():
        try:
            retrieval_start = time.time()
            context = storage.retrieve_context(
                entities_queries=extraction_result.entities_queries,
                fact_query=extraction_result.fact_query,
                recipe_query=extraction_result.recipe_query,
            )
            retrieval_latency = (time.time() - retrieval_start) * 1000
            retrieval_context = context

            # Step 3: Augment prompt with retrieved context
            if context.get("relationships") or context.get("facts") or context.get("recipes"):
                context_str = storage.format_context_for_prompt(context)
                logger.info(
                    f"Retrieved context: {len(context['relationships'])} relationships, "
                    f"{len(context['facts'])} facts, {len(context['recipes'])} recipes"
                )

                # Augment first user message with context
                for msg in messages:
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            augmented_prompt = context_str + content
                            msg["content"] = augmented_prompt
                        elif isinstance(content, list):
                            for part in content:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    original_text = part.get("text", "")
                                    augmented_prompt = context_str + original_text
                                    part["text"] = augmented_prompt
                                    break
                        break

                # Build retrieval results for tracking
                for r in context.get("relationships", [])[:10]:
                    retrieval_results.append(
                        {
                            "type": "relationship",
                            "data": r,
                        }
                    )
                for f in context.get("facts", [])[:10]:
                    retrieval_results.append(
                        {
                            "type": "fact",
                            "data": f,
                        }
                    )
                for r in context.get("recipes", [])[:10]:
                    retrieval_results.append(
                        {
                            "type": "recipe",
                            "data": r,
                        }
                    )

                # Log the retrieval for debugging
                _log_context_retrieval(
                    extraction_result.entities_queries,
                    extraction_result.fact_query,
                    extraction_result.recipe_query,
                    context,
                    augmented_prompt,
                )
            else:
                _log_context_retrieval(
                    extraction_result.entities_queries,
                    extraction_result.fact_query,
                    extraction_result.recipe_query,
                    context,
                    None,
                )
                logger.debug("No relevant context retrieved")
        except Exception as e:
            logger.error(f"Retrieval error: {e}")

    # Step 4: Forward to upstream
    response = await proxy.chat_completions(payload)

    # Get agent response for tracking
    agent_response = ""
    if isinstance(response, dict):
        agent_response = _get_agent_response(response)

    # Step 5: Extract facts from assistant response (non-streaming only)
    assistant_facts: list[dict[str, Any]] = []
    if isinstance(response, dict) and agent_response and extractor.settings.extractor_model_name:
        try:
            logger.info("Extracting facts from assistant response...")
            assistant_result = await extractor.extract_from_text_async(agent_response)
            if assistant_result.facts:
                logger.info(
                    f"Extracted {len(assistant_result.facts)} facts from assistant response"
                )
                assistant_facts = assistant_result.facts
        except Exception as e:
            logger.error(f"Assistant response extraction error: {e}")

    # Step 6: Track request for web UI (if tracker enabled)
    if tracker:
        try:
            request_id = tracker.record_request(
                model=model,
                user_prompt=user_prompt,
                messages=messages,
                extraction_response=extraction_response,
                extraction_ms=extraction_latency,
                augmented_prompt=augmented_prompt,
                agent_response=agent_response,
            )

            # Record extractions
            if extraction_result:
                tracker.record_extractions_batch(
                    request_id=request_id,
                    entities=extraction_result.entities,
                    relationships=extraction_result.relationships,
                    facts=extraction_result.facts + assistant_facts,
                    recipes=extraction_result.recipes,
                )

            # Record retrievals
            if retrieval_results:
                tracker.record_retrievals_batch(
                    request_id=request_id,
                    results=retrieval_results,
                )
        except Exception as e:
            logger.error(f"Tracking error: {e}")

    # Step 7: Store all extracted data in background (fire-and-forget)
    entities = extraction_result.entities if extraction_result else []
    relationships = extraction_result.relationships if extraction_result else []
    facts = (extraction_result.facts if extraction_result else []) + assistant_facts
    recipes = extraction_result.recipes if extraction_result else []
    if entities or relationships or facts or recipes:
        asyncio.create_task(
            _store_extraction_background(
                entities,
                relationships,
                facts,
                recipes,
            )
        )

    return response


@app.post("/v1/completions", response_model=None)
async def completions(request: Request) -> dict[str, Any] | StreamingResponse:
    """Create legacy completion."""
    payload = await request.json()
    return await proxy.completions(payload)


# Admin endpoints
@app.post("/admin/clear")
async def clear_storage() -> dict[str, Any]:
    """Clear all stored data from GrafitoDB."""
    result = storage.clear_all()
    logger.info(f"Storage cleared: {result}")
    return result


@app.get("/admin/stats")
async def get_storage_stats() -> dict[str, Any]:
    """Get storage statistics for GrafitoDB."""
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


# SPA catch-all route (must be after all API routes)
if WEBUI_DIST.exists():

    @app.get("/", response_class=HTMLResponse)
    async def serve_root() -> HTMLResponse:
        """Serve SPA root."""
        index_path = WEBUI_DIST / "index.html"
        if index_path.exists():
            return HTMLResponse(content=index_path.read_text(), status_code=200)
        return HTMLResponse(content="<h1>SPA not built</h1>", status_code=404)

    @app.get("/{path:path}", response_model=None)
    async def serve_spa(path: str) -> HTMLResponse | FileResponse:
        """Serve SPA for client-side routes, or static files."""
        file_path = WEBUI_DIST / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))

        index_path = WEBUI_DIST / "index.html"
        if index_path.exists():
            return HTMLResponse(content=index_path.read_text(), status_code=200)

        return HTMLResponse(content="<h1>SPA not built</h1>", status_code=404)


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
