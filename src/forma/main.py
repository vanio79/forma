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
from forma.agents import (
    AgentRegistry,
    AgentRouter,
    format_agent_discovery_context,
    format_agent_system_prompt,
    get_agent_tools_config,
    load_agents_from_config,
)

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
agent_registry: AgentRegistry | None = None
agent_router: AgentRouter | None = None
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
        extraction_prompt_text = extraction_result.extraction_prompt if extraction_result else ""
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
    global \
        proxy, \
        extractor, \
        storage, \
        db, \
        upstream_manager, \
        tool_executor, \
        agent_registry, \
        agent_router
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

    # Initialize agent system if enabled
    if settings.agents_enabled:
        agent_registry = AgentRegistry(db)

        # Load agents from config file
        agents_config_path = Path(settings.agents_config_path)
        if agents_config_path.exists():
            load_result = load_agents_from_config(str(agents_config_path), agent_registry)
            logger.info(f"Loaded agents from config: {agents_config_path} - {load_result}")

        # Initialize agent router
        agent_router = AgentRouter(agent_registry)

        enabled_agents = agent_registry.get_enabled_agents()
        logger.info(f"Agent system enabled - {len(enabled_agents)} agents available")
        for a in enabled_agents:
            logger.info(f"  - @{a['name']}: {a['purpose']}")
    else:
        agent_registry = None
        agent_router = None
        logger.info("Agent system disabled")

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


async def _execute_agent_request(
    agent: dict[str, Any],
    messages: list[dict[str, Any]],
    user_message: str,
    original_payload: dict[str, Any],
) -> dict[str, Any]:
    """Execute a request for a specific agent with tool execution.

    Uses agent's configuration: upstream, instruction prompt, tool settings.
    Executes tool loop if tools are enabled for the agent.

    Args:
        agent: Agent configuration dict
        messages: Original messages array
        user_message: The message to send to this agent (extracted from user role)
        original_payload: Original request payload

    Returns:
        Response dict from upstream (after tool execution if applicable)
    """
    from forma.tools.executor import ToolExecutor

    settings = get_settings()

    # Get agent's system prompt (with discovery context)
    discovery_context = ""
    if agent_registry and settings.agents_discovery_enabled:
        other_agents = [a for a in agent_registry.get_enabled_agents() if a["id"] != agent["id"]]
        discovery_context = format_agent_discovery_context(other_agents)

    system_prompt = format_agent_system_prompt(agent, discovery_context)

    # Get agent's tool configuration
    tools_config = get_agent_tools_config(
        agent, tool_executor.registry.get_tool_names() if tool_executor else None
    )

    # Agent-specific RAG context retrieval
    rag_config = agent.get("rag_config", {})
    rag_enabled = rag_config.get("enabled", False)
    augmented_user_message = user_message

    if rag_enabled and storage:
        # Get RAG config parameters
        token_budget = rag_config.get("token_budget", 1500)
        min_confidence = rag_config.get("min_confidence", 0.5)
        max_distance = rag_config.get("max_distance", 0.7)

        # Simple query extraction from user message
        # For more sophisticated extraction, could use extractor here
        entities_queries = []
        fact_query = user_message
        recipe_query = user_message

        try:
            agent_context = storage.retrieve_context(
                entities_queries=entities_queries,
                fact_query=fact_query,
                recipe_query=recipe_query,
                token_budget=token_budget,
                min_confidence=min_confidence,
                max_distance=max_distance,
            )

            if agent_context.get("facts") or agent_context.get("recipes"):
                # Format context for prompt
                available_tools = None
                if tool_executor and tools_config["tools_enabled"]:
                    agent_tools_list = tool_executor.registry.get_openai_tools()
                    if tools_config["allowed_tools"]:
                        available_tools = [
                            t
                            for t in agent_tools_list
                            if t.get("function", {}).get("name") in tools_config["allowed_tools"]
                        ]
                    else:
                        available_tools = agent_tools_list

                context_str = storage.format_context_for_prompt(
                    agent_context, available_tools, discovery_context
                )

                augmented_user_message = context_str + user_message

                logger.info(
                    f"Agent @{agent.get('name')} RAG: "
                    f"{len(agent_context.get('facts', []))} facts, "
                    f"{len(agent_context.get('recipes', []))} recipes, "
                    f"{agent_context.get('tokens_used', 0)} tokens"
                )
        except Exception as e:
            logger.warning(f"Agent RAG retrieval error for @{agent.get('name')}: {e}")

    # Build messages for this agent
    agent_messages = [{"role": "system", "content": system_prompt}]

    # Check if this is a delegation (last message is assistant's response)
    if messages and messages[-1].get("role") == "assistant":
        # Delegation case: preserve full conversation history and add continuation prompt
        agent_messages.extend(messages)  # Include user request + assistant's delegation message

        # Add delegation context to help the agent understand what was requested
        delegation_msg = messages[-1].get("content", "")

        delegation_context = (
            f"\n\n--- Delegation Context ---\n"
            f'Previous agent delegated: "{delegation_msg}"\n'
            f"---\n\n"
            f"Continue with this task using your specialized capabilities."
        )
        agent_messages.append(
            {"role": "user", "content": augmented_user_message + delegation_context}
        )
        logger.info(f"Agent @{agent.get('name')} handling delegation from previous agent")
    else:
        # Direct request case: replace last user message with augmented version
        agent_messages.extend(messages[:-1])  # All messages except last
        agent_messages.append({"role": "user", "content": augmented_user_message})

    # Build payload for agent
    agent_payload = original_payload.copy()

    # Determine upstream for this agent
    agent_upstream_id = agent.get("upstream_id")
    if agent_upstream_id:
        agent_upstream = upstream_manager.get_upstream_by_id(agent_upstream_id)
        if agent_upstream:
            agent_payload["model"] = agent_upstream.name
    else:
        # If agent has no specific upstream, use the first enabled upstream
        enabled_upstreams = upstream_manager.get_all_upstreams()
        if enabled_upstreams:
            agent_payload["model"] = enabled_upstreams[0].name
            logger.info(
                f"Agent @{agent.get('name')} using default upstream: {enabled_upstreams[0].name}"
            )

    # Prepare tools for this agent
    agent_tools = None
    if tools_config["tools_enabled"] and tool_executor:
        all_tools = tool_executor.registry.get_openai_tools()
        if tools_config["allowed_tools"]:
            agent_tools = [
                t
                for t in all_tools
                if t.get("function", {}).get("name") in tools_config["allowed_tools"]
            ]
        else:
            agent_tools = all_tools

    # If no tools or tool_executor not available, just forward without tool execution
    if not agent_tools or not tool_executor:
        agent_payload["messages"] = agent_messages
        agent_payload["stream"] = False
        return await proxy.chat_completions(agent_payload)

    # Create a ToolExecutor with agent-specific max_iterations
    agent_max_iterations = agent.get("max_iterations", settings.tools_max_iterations)
    agent_tool_executor = ToolExecutor(
        registry=tool_executor.registry,
        max_iterations=agent_max_iterations,
        timeout=settings.tools_timeout,
    )
    logger.info(
        f"Agent @{agent.get('name')} tool execution: max_iterations={agent_max_iterations}, "
        f"allowed_tools={tools_config.get('allowed_tools', [])}"
    )

    # Remove tools/stream from payload for forward calls
    payload_without_tools = {
        k: v for k, v in agent_payload.items() if k not in ("tools", "tool_choice", "stream")
    }

    # Collect tool events for logging
    tool_events: list[ToolExecutionEvent] = []

    def event_callback(event: ToolExecutionEvent) -> None:
        """Callback to collect tool execution events."""
        tool_events.append(event)
        logger.debug(f"Agent tool event: {event.event_type}")

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
                t.get("function", {}).get("name", "") for t in tools if t.get("type") == "function"
            ]
            logger.info(f"Agent forwarding to upstream with tools: {tool_names}")

        return await proxy.chat_completions(forward_payload)

    # Execute tool loop
    logger.info(f"Executing tool loop for agent @{agent.get('name')}")
    tool_result = await agent_tool_executor.execute_loop(
        messages=agent_messages,
        tools=agent_tools,
        forward_request=forward_with_tools,
        tool_choice="auto",
        event_callback=event_callback,
    )

    # Log tool execution
    if tool_result.has_tool_calls():
        logger.info(
            f"Agent @{agent.get('name')} tool execution complete: "
            f"{len(tool_result.tool_calls)} calls, {tool_result.iterations} iterations, "
            f"{tool_result.total_tool_time_ms:.1f}ms total"
        )
        if tool_result.max_iterations_reached:
            logger.warning(f"Agent @{agent.get('name')} reached max iterations limit")

    # Return final response from tool execution
    return tool_result.response


async def _execute_agent_request_streaming(
    agent: dict[str, Any],
    messages: list[dict[str, Any]],
    user_message: str,
    original_payload: dict[str, Any],
) -> StreamingResponse:
    """Execute a streaming request for a specific agent with tool execution.

    Uses agent's configuration: upstream, instruction prompt, tool settings.
    Executes tool loop if tools are enabled for the agent and streams tool events.

    Args:
        agent: Agent configuration dict
        messages: Original messages array
        user_message: The message to send to this agent (extracted from user role)
        original_payload: Original request payload

    Returns:
        StreamingResponse from upstream (after tool execution if applicable)
    """
    from forma.tools.executor import ToolExecutor

    settings = get_settings()

    # Get agent's system prompt (with discovery context)
    discovery_context = ""
    if agent_registry and settings.agents_discovery_enabled:
        other_agents = [a for a in agent_registry.get_enabled_agents() if a["id"] != agent["id"]]
        discovery_context = format_agent_discovery_context(other_agents)

    system_prompt = format_agent_system_prompt(agent, discovery_context)

    # Get agent's tool configuration
    tools_config = get_agent_tools_config(
        agent, tool_executor.registry.get_tool_names() if tool_executor else None
    )

    # Agent-specific RAG context retrieval
    rag_config = agent.get("rag_config", {})
    rag_enabled = rag_config.get("enabled", False)
    augmented_user_message = user_message

    if rag_enabled and storage:
        # Get RAG config parameters
        token_budget = rag_config.get("token_budget", 1500)
        min_confidence = rag_config.get("min_confidence", 0.5)
        max_distance = rag_config.get("max_distance", 0.7)

        # Simple query extraction from user message
        # For more sophisticated extraction, could use extractor here
        entities_queries = []
        fact_query = user_message
        recipe_query = user_message

        try:
            agent_context = storage.retrieve_context(
                entities_queries=entities_queries,
                fact_query=fact_query,
                recipe_query=recipe_query,
                token_budget=token_budget,
                min_confidence=min_confidence,
                max_distance=max_distance,
            )

            if agent_context.get("facts") or agent_context.get("recipes"):
                # Format context for prompt
                available_tools = None
                if tool_executor and tools_config["tools_enabled"]:
                    agent_tools_list = tool_executor.registry.get_openai_tools()
                    if tools_config["allowed_tools"]:
                        available_tools = [
                            t
                            for t in agent_tools_list
                            if t.get("function", {}).get("name") in tools_config["allowed_tools"]
                        ]
                    else:
                        available_tools = agent_tools_list

                context_str = storage.format_context_for_prompt(
                    agent_context, available_tools, discovery_context
                )

                augmented_user_message = context_str + user_message

                logger.info(
                    f"Agent @{agent.get('name')} RAG: "
                    f"{len(agent_context.get('facts', []))} facts, "
                    f"{len(agent_context.get('recipes', []))} recipes, "
                    f"{agent_context.get('tokens_used', 0)} tokens"
                )
        except Exception as e:
            logger.warning(f"Agent RAG retrieval error for @{agent.get('name')}: {e}")

    # Build messages for this agent
    agent_messages = [{"role": "system", "content": system_prompt}]

    # Check if this is a delegation (last message is assistant's response)
    if messages and messages[-1].get("role") == "assistant":
        # Delegation case: preserve full conversation history and add continuation prompt
        agent_messages.extend(messages)  # Include user request + assistant's delegation message

        # Add delegation context to help the agent understand what was requested
        delegation_msg = messages[-1].get("content", "")

        delegation_context = (
            f"\n\n--- Delegation Context ---\n"
            f'Previous agent delegated: "{delegation_msg}"\n'
            f"---\n\n"
            f"Continue with this task using your specialized capabilities."
        )
        agent_messages.append(
            {"role": "user", "content": augmented_user_message + delegation_context}
        )
        logger.info(f"Agent @{agent.get('name')} handling delegation from previous agent")
    else:
        # Direct request case: replace last user message with augmented version
        agent_messages.extend(messages[:-1])  # All messages except last
        agent_messages.append({"role": "user", "content": augmented_user_message})

    # Build payload for agent
    agent_payload = original_payload.copy()

    # Determine upstream for this agent
    agent_upstream_id = agent.get("upstream_id")
    if agent_upstream_id:
        agent_upstream = upstream_manager.get_upstream_by_id(agent_upstream_id)
        if agent_upstream:
            agent_payload["model"] = agent_upstream.name
    else:
        # If agent has no specific upstream, use the first enabled upstream
        enabled_upstreams = upstream_manager.get_all_upstreams()
        if enabled_upstreams:
            agent_payload["model"] = enabled_upstreams[0].name
            logger.info(
                f"Agent @{agent.get('name')} using default upstream: {enabled_upstreams[0].name}"
            )

    # Prepare tools for this agent
    agent_tools = None
    if tools_config["tools_enabled"] and tool_executor:
        all_tools = tool_executor.registry.get_openai_tools()
        if tools_config["allowed_tools"]:
            agent_tools = [
                t
                for t in all_tools
                if t.get("function", {}).get("name") in tools_config["allowed_tools"]
            ]
        else:
            agent_tools = all_tools

    # If no tools or tool_executor not available, just stream without tool execution
    if not agent_tools or not tool_executor:
        agent_payload["messages"] = agent_messages
        agent_payload["stream"] = True
        return await proxy.chat_completions(agent_payload)

    # Create a ToolExecutor with agent-specific max_iterations
    agent_max_iterations = agent.get("max_iterations", settings.tools_max_iterations)
    agent_tool_executor = ToolExecutor(
        registry=tool_executor.registry,
        max_iterations=agent_max_iterations,
        timeout=settings.tools_timeout,
    )
    logger.info(
        f"Agent @{agent.get('name')} tool execution: max_iterations={agent_max_iterations}, "
        f"allowed_tools={tools_config.get('allowed_tools', [])}"
    )

    # Remove tools/stream from payload for forward calls (we handle tool execution)
    payload_without_tools = {
        k: v for k, v in agent_payload.items() if k not in ("tools", "tool_choice", "stream")
    }

    # Create queue for real-time streaming events
    event_queue: asyncio.Queue = asyncio.Queue()

    def realtime_event_callback(event: ToolExecutionEvent) -> None:
        """Callback to put events into queue for real-time streaming."""
        event_queue.put_nowait(event)
        logger.debug(f"Agent tool event queued: {event.event_type}")

    # Define forward function for tool executor (always non-streaming for tool calls)
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
                t.get("function", {}).get("name", "") for t in tools if t.get("type") == "function"
            ]
            logger.info(f"Agent forwarding to upstream with tools: {tool_names}")

        return await proxy.chat_completions(forward_payload)

    # Create streaming response generator
    final_payload_template = payload_without_tools.copy()
    model_name = final_payload_template.get("model", "unknown")

    async def stream_generator_wrapper():
        """Wrapper that executes tool loop and streams events in parallel."""
        # Start tool execution in parallel
        tool_task = asyncio.create_task(
            agent_tool_executor.execute_loop(
                messages=agent_messages,
                tools=agent_tools,
                forward_request=forward_with_tools,
                tool_choice="auto",
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
                f"Agent @{agent.get('name')} tool execution complete: "
                f"{len(tool_result.tool_calls)} calls, {tool_result.iterations} iterations, "
                f"{tool_result.total_tool_time_ms:.1f}ms total"
            )
            if tool_result.max_iterations_reached:
                logger.warning(f"Agent @{agent.get('name')} reached max iterations limit")

        # Stream the final response with the actual messages
        if tool_result.final_messages:
            final_payload = final_payload_template.copy()
            final_payload["messages"] = tool_result.final_messages
            final_payload["stream"] = True

            response = await proxy.chat_completions(final_payload)
            if isinstance(response, StreamingResponse):
                async for chunk in response.body_iterator:
                    # Skip [DONE] markers - caller will send final [DONE]
                    if isinstance(chunk, bytes):
                        chunk_str = chunk.decode("utf-8")
                        if chunk_str.strip() == "data: [DONE]\n" or "data: [DONE]" in chunk_str:
                            continue  # Skip [DONE] markers from upstream
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
                # Note: Do NOT send [DONE] here - the caller (_stream_route_to_agents) handles it

    # Return streaming response with headers to disable buffering
    return StreamingResponse(
        stream_generator_wrapper(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


def _check_response_for_mentions(response_content: str) -> list[str]:
    """Check agent response for mentions of other agents.

    Args:
        response_content: The text content from an agent's response

    Returns:
        List of UNIQUE agent names mentioned in the response (e.g., ["coder", "researcher"])
    """
    if not agent_router or not agent_registry:
        return []

    mentions = agent_router.parse_mentions(response_content)
    # Return just the agent names (deduplicated)
    seen = set()
    unique_names = []
    for m in mentions:
        if m.agent_name not in seen:
            seen.add(m.agent_name)
            unique_names.append(m.agent_name)
    return unique_names


async def _route_to_agents(
    routing_info: dict[str, Any],
    messages: list[dict[str, Any]],
    original_payload: dict[str, Any],
    depth: int = 0,
    max_depth: int = 3,
    agent_chain: list[str] | None = None,
) -> dict[str, Any]:
    """Route message to mentioned agents sequentially with agent-to-agent support.

    Each agent executes one after another. After each agent completes, checks
    for mentions of other agents. If mentions found and depth < max_depth,
    recursively routes to those agents.

    Args:
        routing_info: Dict with 'agents' (list of agent configs) and 'routing_type'
        messages: Original messages array
        original_payload: Original request payload
        depth: Current recursion depth (for agent-to-agent)
        max_depth: Maximum recursion depth to prevent infinite loops
        agent_chain: List of agent names showing delegation chain (e.g., ["assistant", "researcher"])

    Returns:
        Combined response dict with all agent results
    """
    agents = routing_info.get("agents", [])
    routing_type = routing_info.get("routing_type", "mention")

    # Initialize agent chain if not provided
    if agent_chain is None:
        agent_chain = []

    results: list[dict[str, Any]] = []

    for agent in agents:
        agent_name = agent.get("name", "unknown")

        # Add this agent to the chain
        current_chain = agent_chain + [agent_name]

        logger.info(f"Routing to agent @{agent_name} (depth={depth}, chain={current_chain})")

        # Get the actual user message for RAG query (not the last message which might be assistant)
        actual_user_msg = _get_user_prompt(messages)

        # Execute agent request (non-streaming)
        response = await _execute_agent_request(
            agent=agent,
            messages=messages,
            user_message=actual_user_msg,
            original_payload=original_payload,
        )

        # Extract response content
        response_content = _get_agent_response(response)

        # Check for agent-to-agent mentions
        mentioned_agents = _check_response_for_mentions(response_content)

        if mentioned_agents and depth < max_depth:
            logger.info(
                f"Agent @{agent_name} mentioned other agents: {mentioned_agents}, "
                f"routing recursively (depth={depth + 1})"
            )

            # Build new routing info for mentioned agents
            sub_agents = []
            for mentioned_name in mentioned_agents:
                # Prevent self-delegation - agent cannot delegate to itself
                if mentioned_name == agent_name:
                    logger.warning(
                        f"Agent @{agent_name} attempted self-delegation to @{mentioned_name}, skipping"
                    )
                    continue

                mentioned_agent = agent_registry.get_agent_by_name(mentioned_name)
                if mentioned_agent and mentioned_agent.get("is_enabled"):
                    sub_agents.append(mentioned_agent)

            if sub_agents:
                # Create sub-messages with the agent's response as context
                sub_messages = messages.copy()
                sub_messages.append({"role": "assistant", "content": response_content})

                # Recursively route to mentioned agents with updated chain
                sub_response = await _route_to_agents(
                    routing_info=sub_routing_info,
                    messages=sub_messages,
                    original_payload=original_payload,
                    depth=depth + 1,
                    max_depth=max_depth,
                    agent_chain=current_chain,  # Pass the current chain to sub-agents
                )

                # Combine results
                results.append(
                    {
                        "agent": agent,
                        "content": response_content,
                        "sub_agents": sub_response.get("results", []),
                    }
                )
            else:
                results.append(
                    {
                        "agent": agent,
                        "content": response_content,
                    }
                )
        else:
            results.append(
                {
                    "agent": agent,
                    "content": response_content,
                }
            )

    # Build combined response
    combined_content = ""
    for i, result in enumerate(results):
        agent_name = result.get("agent", {}).get("name", "unknown")
        content = result.get("content", "")

        if i > 0:
            combined_content += "\n\n"
        combined_content += f"[ @{agent_name} ]\n{content}"

        # Add sub-agent results
        sub_results = result.get("sub_agents", [])
        for sub_result in sub_results:
            sub_agent_name = sub_result.get("agent", {}).get("name", "unknown")
            sub_content = sub_result.get("content", "")
            combined_content += f"\n\n[ @{sub_agent_name} ]\n{sub_content}"

    # Return in OpenAI response format
    return {
        "id": f"agent-route-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": original_payload.get("model", "unknown"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": combined_content,
                },
                "finish_reason": "stop",
            }
        ],
        "results": results,  # Include detailed results for debugging
    }


async def _stream_route_to_agents(
    routing_info: dict[str, Any],
    messages: list[dict[str, Any]],
    original_payload: dict[str, Any],
    depth: int = 0,
    max_depth: int = 3,
    agent_chain: list[str] | None = None,
) -> StreamingResponse:
    """Route message to mentioned agents sequentially with streaming and agent-to-agent support.

    Each agent's response is streamed with markers:
    __AGENT_START__{"agent": "name", "depth": N, "chain": ["agent1", "agent2"]}__END__
    [streaming content]
    __AGENT_END__{"agent": "name", "depth": N, "chain": ["agent1", "agent2"]}__END__

    After each agent completes, checks for mentions of other agents.
    If mentions found and depth < max_depth, recursively streams sub-agent responses.

    Args:
        routing_info: Dict with 'agents' (list of agent configs) and 'routing_type'
        messages: Original messages array
        original_payload: Original request payload
        depth: Current recursion depth (for agent-to-agent)
        max_depth: Maximum recursion depth to prevent infinite loops
        agent_chain: List of agent names showing delegation chain (e.g., ["assistant", "researcher"])

    Returns:
        StreamingResponse with agent markers and content
    """
    agents = routing_info.get("agents", [])
    routing_type = routing_info.get("routing_type", "mention")

    # Initialize agent chain if not provided
    if agent_chain is None:
        agent_chain = []

    async def stream_generator():
        """Generate streaming response with agent markers."""
        streamed_content_for_mention_check = ""

        for agent in agents:
            agent_name = agent.get("name", "unknown")
            logger.info(f"Streaming to agent @{agent_name} (depth={depth}, chain={agent_chain})")

            # Add this agent to the chain
            current_chain = agent_chain + [agent_name]

            # Send agent start marker with chain
            chain_json = json.dumps(current_chain)
            start_marker = f'__AGENT_START__{{"agent": "{agent_name}", "depth": {depth}, "chain": {chain_json}}}__END__\n'
            yield start_marker.encode()

            # Get the actual user message for RAG query (not the last message which might be assistant)
            actual_user_msg = _get_user_prompt(messages)

            # Execute agent request (streaming)
            response = await _execute_agent_request_streaming(
                agent=agent,
                messages=messages,
                user_message=actual_user_msg,
                original_payload=original_payload,
            )

            # Stream the agent's response
            if isinstance(response, StreamingResponse):
                async for chunk in response.body_iterator:
                    # Skip [DONE] markers - we'll send our own at the end
                    if isinstance(chunk, bytes):
                        chunk_str = chunk.decode("utf-8")
                        if chunk_str.strip() == "data: [DONE]\n" or "data: [DONE]" in chunk_str:
                            continue  # Skip [DONE] markers from agent streams

                    # Collect content for mention checking
                    if isinstance(chunk, bytes):
                        chunk_str = chunk.decode("utf-8")
                        # Extract content from SSE chunks
                        try:
                            if chunk_str.startswith("data: ") and not chunk_str.startswith(
                                "data: [DONE]"
                            ):
                                data_str = chunk_str[6:].strip()
                                if data_str:
                                    chunk_json = json.loads(data_str)
                                    for choice in chunk_json.get("choices", []):
                                        delta = choice.get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            streamed_content_for_mention_check += content
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            pass
                    yield chunk

            # Send agent end marker with chain
            chain_json = json.dumps(current_chain)
            end_marker = f'__AGENT_END__{{"agent": "{agent_name}", "depth": {depth}, "chain": {chain_json}}}__END__\n'
            yield end_marker.encode()

            # Check for agent-to-agent mentions in the streamed content
            mentioned_agents = _check_response_for_mentions(streamed_content_for_mention_check)

            if mentioned_agents and depth < max_depth:
                logger.info(
                    f"Agent @{agent_name} mentioned other agents: {mentioned_agents}, "
                    f"streaming recursively (depth={depth + 1})"
                )

                # Build new routing info for mentioned agents
                sub_agents = []
                for mentioned_name in mentioned_agents:
                    # Prevent self-delegation - agent cannot delegate to itself
                    if mentioned_name == agent_name:
                        logger.warning(
                            f"Agent @{agent_name} attempted self-delegation to @{mentioned_name}, skipping"
                        )
                        continue

                    mentioned_agent = agent_registry.get_agent_by_name(mentioned_name)
                    if mentioned_agent and mentioned_agent.get("is_enabled"):
                        sub_agents.append(mentioned_agent)

                if sub_agents:
                    # Create sub-messages with the agent's response as context
                    sub_messages = messages.copy()
                    sub_messages.append(
                        {"role": "assistant", "content": streamed_content_for_mention_check}
                    )

                    # Build routing info for mentioned agents
                    sub_routing_info = {
                        "agents": sub_agents,
                        "routing_type": "mention",
                    }

                    # Recursively stream to mentioned agents with updated chain
                    sub_response = await _stream_route_to_agents(
                        routing_info=sub_routing_info,
                        messages=sub_messages,
                        original_payload=original_payload,
                        depth=depth + 1,
                        max_depth=max_depth,
                        agent_chain=current_chain,  # Pass the current chain to sub-agents
                    )

                    # Stream sub-agent responses
                    if isinstance(sub_response, StreamingResponse):
                        async for chunk in sub_response.body_iterator:
                            # Skip [DONE] markers from sub-agent streams
                            if isinstance(chunk, bytes):
                                chunk_str = chunk.decode("utf-8")
                                if (
                                    chunk_str.strip() == "data: [DONE]\n"
                                    or "data: [DONE]" in chunk_str
                                ):
                                    continue  # Skip [DONE] markers from sub-agent streams
                            yield chunk

        # Send final [DONE] marker only at depth 0
        if depth == 0:
            yield "data: [DONE]\n\n".encode()

    # Return streaming response with headers to disable buffering
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> dict[str, Any] | StreamingResponse:
    """
    Create chat completion with extraction, retrieval, RAG context, tool execution, and agent routing.

    Pipeline:
    0. Check for agent mentions - if found, route to those agents
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

    # Step 0: Check for agent mentions - route to specific agents if found
    stream_requested = payload.get("stream", False)
    if agent_router and agent_registry:
        mentions = agent_router.parse_mentions(user_prompt)

        if mentions:
            # Extract complete routing info
            routing_info_full = agent_router.extract_routing_info(user_prompt)

            # Convert agents dict to list
            agents_list = list(routing_info_full.get("agents", {}).values())

            if agents_list:
                routing_info = {
                    "agents": agents_list,
                    "routing_type": "mention",
                }

                logger.info(
                    f"Agent mentions found in user prompt: {[a.get('name') for a in agents_list]}"
                )

                # Route to mentioned agents
                if stream_requested:
                    # Streaming routing
                    return await _stream_route_to_agents(
                        routing_info=routing_info,
                        messages=messages,
                        original_payload=payload,
                        depth=0,
                        max_depth=3,
                    )
                else:
                    # Non-streaming routing
                    routed_response = await _route_to_agents(
                        routing_info=routing_info,
                        messages=messages,
                        original_payload=payload,
                        depth=0,
                        max_depth=3,
                    )

                    # Record routed request to database
                    _record_request_to_db(
                        model=model,
                        user_prompt=user_prompt,
                        messages=messages,
                        extraction_response="",
                        extraction_result=None,
                        extraction_latency=0.0,
                        augmented_prompt="",
                        agent_response=_get_agent_response(routed_response),
                        retrieval_results=[],
                    )

                    return routed_response

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


# Agent management endpoints
@app.get("/api/agents")
async def list_agents() -> list[dict[str, Any]]:
    """List all agents with their configurations including RAG config."""
    if not agent_registry:
        return []
    return agent_registry.get_all_agents()


@app.get("/api/agents/{agent_name}")
async def get_agent(agent_name: str) -> dict[str, Any] | None:
    """Get a specific agent by name."""
    if not agent_registry:
        return None
    return agent_registry.get_agent_by_name(agent_name)


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
