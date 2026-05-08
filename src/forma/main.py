"""FastAPI application entry point."""

import asyncio
import contextlib
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
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
from forma.forma_db import FormaDatabase, get_db
from forma.upstream_manager import UpstreamManager
from forma.tools import ToolExecutor, get_registry
from forma.tools.executor import ToolExecutionEvent

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
db: FormaDatabase
upstream_manager: UpstreamManager
tool_executor: ToolExecutor | None = None
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
        "timestamp": datetime.now(UTC).isoformat(),
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
    relationships: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    recipes: list[dict[str, Any]],
) -> None:
    """Background task to store extracted data."""
    async with _storage_lock:
        try:
            if relationships or facts or recipes:
                storage.store_extraction(relationships, facts, recipes)
                logger.info(
                    f"Stored (background): {len(relationships)} relationships, "
                    f"{len(facts)} facts, {len(recipes)} recipes"
                )
        except Exception as e:
            logger.error(f"Background storage error: {e}")


def _fire_and_forget(coro) -> None:
    """Schedule a background task with error logging."""
    task = asyncio.create_task(coro)
    task.add_done_callback(_background_task_done)


def _background_task_done(task: asyncio.Task) -> None:
    """Callback for fire-and-forget tasks to log exceptions."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.error(f"Background task failed: {exc}", exc_info=exc)


def _record_request_to_db(
    *,
    model: str,
    user_prompt: str,
    messages: list[dict],
    extraction_response: str,
    extraction_result: Any,
    extraction_latency: float,
    augmented_prompt: str,
    agent_response: str,
    retrieval_results: list[dict],
) -> None:
    """Record a request and its extractions/retrievals to the database."""
    if not db:
        return
    try:
        extraction_prompt_text = (
            extraction_result.extraction_prompt if extraction_result else ""
        )
        request_id = db.record_request(
            model=model,
            user_prompt=user_prompt,
            messages=messages,
            extraction_response=extraction_response,
            extraction_prompt=extraction_prompt_text,
            extraction_ms=extraction_latency,
            augmented_prompt=augmented_prompt,
            agent_response=agent_response,
        )
        if extraction_result:
            db.record_extractions_batch(
                request_id=request_id,
                relationships=extraction_result.relationships,
                facts=extraction_result.facts,
                recipes=extraction_result.recipes,
            )
        if retrieval_results:
            db.record_retrievals_batch(
                request_id=request_id,
                results=retrieval_results,
            )
    except Exception as e:
        logger.error(f"Database recording error: {e}")


async def _stream_with_realtime_events(
    event_queue: asyncio.Queue,
    model: str,
) -> Any:
    """Stream tool execution events in real-time.

    This generator:
    1. Emits tool execution events from queue immediately as they arrive
    2. Stops when it receives a "tool_loop_complete" event

    Args:
        event_queue: Queue receiving tool events in real-time
        model: Model name for SSE chunks

    Yields:
        SSE formatted chunks (OpenAI-compatible format)
    """
    tool_loop_complete_received = False
    events_streamed = 0

    # Stream events from queue in real-time until tool_loop_complete event
    while not tool_loop_complete_received:
        # Try to get event immediately (non-blocking)
        try:
            event = event_queue.get_nowait()
            logger.debug(f"Streaming event {event.event_type} at {time.time() * 1000:.0f}ms")
            events_streamed += 1
        except asyncio.QueueEmpty:
            # Queue is empty - yield control briefly to let tool executor run
            await asyncio.sleep(0.001)  # 1ms
            continue

        # Check if this is the completion event
        if event.event_type == "tool_loop_complete":
            tool_loop_complete_received = True
            logger.info(f"Streamed {events_streamed} tool events in real-time")

        content = event.to_content_delta()
        if content:
            # Format as OpenAI SSE chunk
            chunk = {
                "id": f"tool_event_{event.event_type}",
                "object": "chat.completion.chunk",
                "created": int(event.timestamp / 1000),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()

            # Send SSE comment to force flush (comments are ignored by clients)
            # This prevents HTTP buffering from coalescing events
            yield ": flush\n\n".encode()

            # Force flush by yielding control to event loop
            # This ensures the SSE chunk is sent immediately, not buffered
            await asyncio.sleep(0)


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Manage application lifespan."""
    global proxy, extractor, storage, db, upstream_manager, tool_executor
    settings = get_settings()

    # Initialize database for web UI and upstreams
    if settings.history_enabled:
        db = FormaDatabase(
            db_path=settings.forma_db_path,
            max_records=settings.history_max_records,
        )
    else:
        db = None

    # Initialize upstream manager (loads upstreams from database)
    upstream_manager = UpstreamManager(db)

    # Initialize proxy with upstream manager
    proxy = OpenAIProxy(settings, upstream_manager)
    extractor = Extractor(settings, proxy=proxy)
    storage = Storage(
        grafitodb_path=settings.grafitodb_path,
        grafitodb_embedding_model=settings.grafitodb_embedding_model,
        grafitodb_vector_dim=settings.grafitodb_vector_dim,
        grafitodb_model_cache_path=settings.grafitodb_model_cache_path,
    )

    # Initialize tool executor if tools enabled
    if settings.tools_enabled:
        registry = get_registry(storage=storage)
        tool_executor = ToolExecutor(
            registry=registry,
            max_iterations=settings.tools_max_iterations,
            timeout=settings.tools_timeout,
        )
        logger.info(
            f"Tool execution enabled - max iterations: {settings.tools_max_iterations}, "
            f"available tools: {registry.get_tool_names()}"
        )
    else:
        tool_executor = None

    if db:
        logger.info(f"Request history enabled - max records: {settings.history_max_records}")

    # Log upstream configuration
    upstreams = upstream_manager.get_all_upstreams()
    logger.info(f"Upstream configurations: {len(upstreams)}")
    for u in upstreams:
        status_str = "enabled" if u.is_enabled else "disabled"
        logger.info(f"  - {u.name}: {u.base_url} ({status_str})")

    # Log storage stats
    stats = storage.get_stats()
    logger.info(
        f"Storage: GrafitoDB nodes={stats['grafitodb']['nodes']}, "
        f"relationships={stats['grafitodb']['relationships']}, "
        f"facts={stats['grafitodb']['facts']}, "
        f"recipes={stats['grafitodb']['recipes']}"
    )
    yield
    logger.info("Forma proxy shutting down")
    await proxy.close()
    extractor.close()
    storage.close()
    if db:
        db.close()
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
    """Extract the LAST user prompt from messages."""
    last_user_content = ""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                last_user_content = content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        last_user_content = part.get("text", "")
    return last_user_content


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
    Create chat completion with extraction, retrieval, RAG context, and tool execution.

    Pipeline:
    1. Extract relationships/facts/recipes/queries from messages
    2. Retrieve context from storage using extracted queries
    3. Augment prompt with retrieved context
    4. Execute tool loop if tools are provided (server-side tool calling)
    5. Forward to upstream model (or get final response from tool loop)
    6. Store extracted data in background (async, fire-and-forget)
    7. Track request for web UI
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

    # Step 1: Extract relationships, facts, recipes, and queries
    if messages and extractor.settings.extractor_model_name:
        try:
            extraction_start = time.time()
            logger.info(f"Extracting from {len(messages)} messages...")
            result = await extractor.extract_from_messages_async(messages)
            extraction_latency = (time.time() - extraction_start) * 1000
            extraction_response = result.raw_response

            if result.is_valid():
                logger.info(
                    f"Extraction complete: {len(result.relationships)} relationships, "
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
                # Get available tools if tools enabled
                available_tools = None
                if tool_executor:
                    available_tools = tool_executor.registry.get_openai_tools()

                context_str = storage.format_context_for_prompt(context, available_tools)
                logger.info(
                    f"Retrieved context: {len(context['relationships'])} relationships, "
                    f"{len(context['facts'])} facts, {len(context['recipes'])} recipes"
                )
                logger.info(f"Context string length: {len(context_str)} chars")

                # Augment LAST user message with context
                for msg in reversed(messages):
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

    # Step 4: Automatically add available tools to request if tools enabled
    # This allows the upstream model to see and use available tools
    if tool_executor:
        registry_tools = tool_executor.registry.get_openai_tools()
        if registry_tools:
            # Add tools to payload so upstream model can decide to use them
            payload["tools"] = registry_tools
            # Set tool_choice to "auto" if not specified - model decides when to use tools
            if "tool_choice" not in payload:
                payload["tool_choice"] = "auto"
            logger.info(f"Added {len(registry_tools)} available tools to request")

    # Step 5: Execute tool loop or forward to upstream
    tools = payload.get("tools", [])
    tool_choice = payload.get("tool_choice", "auto")
    stream = payload.get("stream", False)
    stream_requested = stream  # Track original streaming request

    # Tool execution loop (if tools provided and tools enabled)
    if tools and tool_executor:
        # Remove tools from payload for upstream calls (we handle tool execution)
        payload_without_tools = {
            k: v for k, v in payload.items() if k not in ("tools", "tool_choice", "stream")
        }

        # For real-time streaming: use async queue for events
        if stream_requested:
            # Create queue for real-time streaming
            event_queue: asyncio.Queue = asyncio.Queue()

            def realtime_event_callback(event: ToolExecutionEvent) -> None:
                """Callback to put events into queue for real-time streaming."""
                event_queue.put_nowait(event)
                logger.debug(f"Tool event queued: {event.event_type}")

            # Define forward function for tool executor (always non-streaming for tool calls)
            async def forward_with_tools(
                messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None,
                tool_choice: str | dict[str, Any] = "auto",
            ) -> dict[str, Any]:
                """Forward request to upstream, optionally with tools (non-streaming)."""
                forward_payload = payload_without_tools.copy()
                forward_payload["messages"] = messages
                forward_payload["stream"] = False  # Always non-streaming for tool calls

                if tools:
                    forward_payload["tools"] = tools
                    forward_payload["tool_choice"] = tool_choice
                    # Log tool names being sent to upstream
                    tool_names = [
                        t.get("function", {}).get("name", "")
                        for t in tools
                        if t.get("type") == "function"
                    ]
                    logger.info(f"Forwarding to upstream with tools: {tool_names}")
                    logger.info(f"Forwarding {len(messages)} messages to upstream")

                    # Log first user message preview
                    if messages:
                        for msg in messages:
                            if msg.get("role") == "user":
                                content = msg.get("content", "")
                                preview = content[:200] if len(content) > 200 else content
                                logger.info(f"First user message preview: {preview}")
                                break

                return await proxy.chat_completions(forward_payload)

            # Create streaming response generator first (will wait for events from queue)
            final_payload_template = payload_without_tools.copy()
            model_name = final_payload_template.get("model", "unknown")

            async def stream_generator_wrapper():
                """Wrapper that executes tool loop and streams events in parallel."""
                # Start tool execution in parallel
                tool_task = asyncio.create_task(
                    tool_executor.execute_loop(
                        messages=messages,
                        tools=tools,
                        forward_request=forward_with_tools,
                        tool_choice=tool_choice,
                        event_callback=realtime_event_callback,
                    )
                )

                # Stream events from the real-time generator
                async for chunk in _stream_with_realtime_events(
                    event_queue=event_queue,
                    model=model_name,
                ):
                    yield chunk

                # Wait for tool execution to complete and get final messages
                tool_result = await tool_task

                # Log tool execution
                if tool_result.has_tool_calls():
                    logger.info(
                        f"Tool execution complete: {len(tool_result.tool_calls)} calls, "
                        f"{tool_result.iterations} iterations, "
                        f"{tool_result.total_tool_time_ms:.1f}ms total"
                    )
                    if tool_result.max_iterations_reached:
                        logger.warning("Tool execution reached max iterations limit")

                # Now stream the final response with the actual messages
                if tool_result.final_messages:
                    final_payload = final_payload_template.copy()
                    final_payload["messages"] = tool_result.final_messages
                    final_payload["stream"] = True

                    response = await proxy.chat_completions(final_payload)
                    if isinstance(response, StreamingResponse):
                        async for chunk in response.body_iterator:
                            yield chunk
                    elif isinstance(response, dict):
                        # Convert to SSE format
                        sse_response = response.copy()
                        if "choices" in sse_response:
                            for choice in sse_response["choices"]:
                                if "message" in choice:
                                    message = choice.pop("message", {})
                                    choice["delta"] = {"content": message.get("content", "")}
                                    if message.get("role"):
                                        choice["delta"]["role"] = message.get("role")
                        yield f"data: {json.dumps(sse_response)}\n\n".encode()
                        yield "data: [DONE]\n\n".encode()

                # Store extraction data in background
                relationships = extraction_result.relationships if extraction_result else []
                facts = extraction_result.facts if extraction_result else []
                recipes = extraction_result.recipes if extraction_result else []
                if relationships or facts or recipes:
                    _fire_and_forget(_store_extraction_background(relationships, facts, recipes))

                # Record request in database
                _record_request_to_db(
                    model=model,
                    user_prompt=user_prompt,
                    messages=messages,
                    extraction_response=extraction_response,
                    extraction_result=extraction_result,
                    extraction_latency=extraction_latency,
                    augmented_prompt=augmented_prompt,
                    agent_response="",
                    retrieval_results=retrieval_results,
                )

            # Return streaming response with headers to disable buffering
            return StreamingResponse(
                stream_generator_wrapper(),
                media_type="text/event-stream",
                headers={
                    "X-Accel-Buffering": "no",  # Disable nginx buffering
                    "Cache-Control": "no-cache",  # Prevent caching
                    "Connection": "keep-alive",  # Maintain connection
                },
            )

        else:
            # Non-streaming: collect events normally
            tool_events: list[ToolExecutionEvent] = []

            def event_callback(event: ToolExecutionEvent) -> None:
                """Callback to collect tool execution events."""
                tool_events.append(event)
                logger.debug(f"Tool event: {event.event_type}")

            # Define forward function for tool executor
            async def forward_with_tools(
                messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None,
                tool_choice: str | dict[str, Any] = "auto",
            ) -> dict[str, Any]:
                """Forward request to upstream, optionally with tools (non-streaming)."""
                forward_payload = payload_without_tools.copy()
                forward_payload["messages"] = messages
                forward_payload["stream"] = False

                if tools:
                    forward_payload["tools"] = tools
                    forward_payload["tool_choice"] = tool_choice
                    tool_names = [
                        t.get("function", {}).get("name", "")
                        for t in tools
                        if t.get("type") == "function"
                    ]
                    logger.info(f"Forwarding to upstream with tools: {tool_names}")
                    logger.info(f"Forwarding {len(messages)} messages to upstream")

                return await proxy.chat_completions(forward_payload)

            # Execute tool loop
            logger.info(f"Executing tool loop (streaming=False)")
            tool_result = await tool_executor.execute_loop(
                messages=messages,
                tools=tools,
                forward_request=forward_with_tools,
                tool_choice=tool_choice,
                event_callback=event_callback,
            )

            # Log tool execution
            if tool_result.has_tool_calls():
                logger.info(
                    f"Tool execution complete: {len(tool_result.tool_calls)} calls, "
                    f"{tool_result.iterations} iterations, "
                    f"{tool_result.total_tool_time_ms:.1f}ms total"
                )
                if tool_result.max_iterations_reached:
                    logger.warning("Tool execution reached max iterations limit")

            # Non-streaming response - continue to Step 6 for recording
            response = tool_result.response
    else:
        # Normal forward without tool execution
        response = await proxy.chat_completions(payload)
        # If streaming and no tools, response is StreamingResponse - return directly
        if isinstance(response, StreamingResponse):
            # Record request without agent_response for streaming
            _record_request_to_db(
                model=model,
                user_prompt=user_prompt,
                messages=messages,
                extraction_response=extraction_response,
                extraction_result=extraction_result,
                extraction_latency=extraction_latency,
                augmented_prompt=augmented_prompt,
                agent_response="",
                retrieval_results=retrieval_results,
            )

            relationships = extraction_result.relationships if extraction_result else []
            facts = extraction_result.facts if extraction_result else []
            recipes = extraction_result.recipes if extraction_result else []
            if relationships or facts or recipes:
                _fire_and_forget(
                    _store_extraction_background(
                        relationships,
                        facts,
                        recipes,
                    )
                )

            return response  # Return StreamingResponse directly

    # Get agent response for tracking (only for non-streaming responses)
    agent_response = ""
    if isinstance(response, dict):
        agent_response = _get_agent_response(response)

    # Step 6: Record request for web UI (if database enabled)
    _record_request_to_db(
        model=model,
        user_prompt=user_prompt,
        messages=messages,
        extraction_response=extraction_response,
        extraction_result=extraction_result,
        extraction_latency=extraction_latency,
        augmented_prompt=augmented_prompt,
        agent_response=agent_response,
        retrieval_results=retrieval_results,
    )

    # Step 7: Store all extracted data in background (fire-and-forget)
    relationships = extraction_result.relationships if extraction_result else []
    facts = extraction_result.facts if extraction_result else []
    recipes = extraction_result.recipes if extraction_result else []
    if relationships or facts or recipes:
        _fire_and_forget(
            _store_extraction_background(
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
