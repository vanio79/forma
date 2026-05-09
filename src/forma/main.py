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

from forma.agents import (
    AgentRegistry,
    AgentRouter,
    format_agent_discovery_context,
    format_agent_system_prompt,
    get_agent_tools_config,
    load_agents_from_config,
)
from forma.agents.meta_evaluation import (
    EvaluationResult,
    build_compaction_input,
    create_isolated_context,
    create_retry_context,
    estimate_messages_tokens,
    extract_summary,
    format_evaluator_input,
    format_summarizer_input,
    parse_evaluator_response,
    should_compact_context,
)
from forma.api import router as ui_router
from forma.config import get_settings
from forma.extractor import Extractor
from forma.forma_db import FormaDatabase
from forma.proxy import OpenAIProxy
from forma.storage import Storage
from forma.tools import ToolExecutor, get_registry
from forma.tools.executor import ToolExecutionEvent
from forma.upstream_manager import UpstreamManager

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
            yield b": flush\n\n"

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

    # Auto-compact context if needed (prevent overflow during agent chains)
    # This happens before agent execution, not after
    # All agents (including meta-agents) now receive full history and rely on automatic compaction
    settings_compact = get_settings()
    if should_compact_context(
        agent_messages,
        settings_compact.context_window_size,
        settings_compact.context_compaction_threshold,
        settings_compact.context_chars_per_token,
    ):
        logger.info(f"Auto-compacting context before agent @{agent.get('name')} execution")
        agent_messages = await _compact_conversation_context(
            messages=agent_messages,
            original_payload=original_payload,
            keep_recent=settings_compact.context_keep_recent_messages,
        )

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

    # Auto-compact context if needed (prevent overflow during agent chains)
    # This happens before agent execution, not after
    # All agents (including meta-agents) now receive full history and rely on automatic compaction
    settings_compact = get_settings()
    if should_compact_context(
        agent_messages,
        settings_compact.context_window_size,
        settings_compact.context_compaction_threshold,
        settings_compact.context_chars_per_token,
    ):
        logger.info(f"Auto-compacting context before agent @{agent.get('name')} execution")
        agent_messages = await _compact_conversation_context(
            messages=agent_messages,
            original_payload=original_payload,
            keep_recent=settings_compact.context_keep_recent_messages,
        )

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


# Meta-agent names (these agents don't get evaluated)
META_AGENTS = {"evaluator", "summarizer"}


def _is_meta_agent(agent_name: str) -> bool:
    """Check if agent is a meta-agent (internal system agent).

    Meta-agents (evaluator, summarizer) are trusted and don't get
    evaluated since they're part of the quality control system.

    Args:
        agent_name: Name of the agent

    Returns:
        True if agent is a meta-agent
    """
    return agent_name in META_AGENTS


async def _compact_conversation_context(
    messages: list[dict[str, Any]],
    original_payload: dict[str, Any],
    keep_recent: int = 4,
) -> list[dict[str, Any]]:
    """Compact conversation context using @summarizer agent.

    Replaces older messages with a summary to prevent context overflow
    during multi-agent conversations.

    Args:
        messages: Current conversation messages
        original_payload: Original request payload
        keep_recent: Number of recent messages to keep (default: 4 = 2 exchanges)

    Returns:
        Compacted messages array with summary replacing old messages
    """
    from forma.agents.meta_evaluation import (
        build_compaction_input,
        estimate_messages_tokens,
        extract_summary,
        should_compact_context,
    )

    settings = get_settings()

    # Check if compaction is needed
    context_window = settings.context_window_size
    threshold = settings.context_compaction_threshold
    chars_per_token = settings.context_chars_per_token

    if not should_compact_context(messages, context_window, threshold, chars_per_token):
        return messages  # No compaction needed

    # Calculate current token count and threshold
    old_tokens = estimate_messages_tokens(messages, chars_per_token)
    threshold_tokens = int(context_window * threshold)

    logger.info(
        f"Context compaction needed: {old_tokens} tokens "
        f"(threshold: {threshold_tokens}, window: {context_window})"
    )

    # Ensure we have enough messages to compact
    if len(messages) <= keep_recent:
        # Can't compact by removing messages, but we can iteratively summarize large content
        logger.warning(
            f"Context exceeds threshold but only {len(messages)} messages exist. "
            f"Using iterative summarization to compact large messages."
        )

        # Get summarizer agent
        if not agent_registry:
            logger.warning("Agent registry unavailable, cannot compact context")
            return messages

        summarizer_agent = agent_registry.get_agent_by_name("summarizer")
        if not summarizer_agent or not summarizer_agent.get("is_enabled"):
            logger.warning("Summarizer agent unavailable, cannot compact context")
            return messages

        # Split messages into chunks that fit within summarizer's context window
        # Leave room for summarizer prompt and response (~4000 tokens buffer)
        chunk_target_tokens = int(context_window * 0.7)  # 70% of context for each chunk
        max_summary_tokens = 2000  # Target summary size

        # Combine all messages into one large text for chunking
        full_text_parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str):
                full_text_parts.append(f"{role.upper()}: {content}")
            elif isinstance(content, list):
                # Handle multi-modal content
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                if text_parts:
                    full_text_parts.append(f"{role.upper()}: {' '.join(text_parts)}")

        full_text = "\n\n".join(full_text_parts)

        # Split into chunks based on character estimation
        chars_per_chunk = chunk_target_tokens * chars_per_token
        chunks = []
        current_pos = 0

        while current_pos < len(full_text):
            chunk_end = min(current_pos + chars_per_chunk, len(full_text))
            # Try to find a good break point (double newline)
            if chunk_end < len(full_text):
                # Look for break point within last 500 chars of chunk
                search_start = max(chunk_end - 500, current_pos)
                break_pos = full_text.find("\n\n", search_start, chunk_end + 500)
                if break_pos > current_pos:
                    chunk_end = break_pos

            chunks.append(full_text[current_pos:chunk_end])
            current_pos = chunk_end

        logger.info(
            f"Split large context ({old_tokens} tokens) into {len(chunks)} chunks "
            f"for iterative summarization"
        )

        # Iteratively summarize chunks
        accumulated_summary = ""

        try:
            for i, chunk in enumerate(chunks):
                # Build prompt for this chunk
                if i == 0:
                    prompt = (
                        f"Summarize this conversation segment concisely:\n\n{chunk}\n\n"
                        f"Provide a summary (target {max_summary_tokens} tokens) capturing "
                        f"key information, decisions, and context needed for continuing work."
                    )
                else:
                    # Include previous summary + new chunk
                    prompt = (
                        f"Previous summary:\n{accumulated_summary}\n\n"
                        f"New conversation segment:\n{chunk}\n\n"
                        f"Update and expand the summary to include information from the new segment. "
                        f"Keep it concise (target {max_summary_tokens} tokens total)."
                    )

                summarizer_messages = [{"role": "user", "content": prompt}]

                logger.info(f"Summarizing chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)")

                # Execute summarizer
                summarizer_response = await _execute_agent_request(
                    agent=summarizer_agent,
                    messages=summarizer_messages,
                    user_message=prompt,
                    original_payload=original_payload,
                )

                summarizer_content = _get_agent_response(summarizer_response)
                accumulated_summary = extract_summary(summarizer_content)

                logger.info(f"Chunk {i + 1} summary: {accumulated_summary[:100]}...")

            # Final summary message
            summary_message = {
                "role": "system",
                "content": f"[CONTEXT SUMMARY]\n{accumulated_summary}",
            }

            # Return just the summary (no original messages since they're compacted)
            result_messages = [summary_message]

            # Add keep_recent most recent messages if they exist
            if keep_recent > 0 and len(messages) >= keep_recent:
                result_messages.extend(messages[-keep_recent:])

            # Calculate reduction
            new_tokens = estimate_messages_tokens(result_messages, chars_per_token)
            reduction_pct = (1 - new_tokens / old_tokens) * 100 if old_tokens > 0 else 0

            logger.info(
                f"Iterative summarization complete: {old_tokens} -> {new_tokens} tokens "
                f"({reduction_pct:.1f}% reduction)"
            )

            return result_messages

        except Exception as e:
            logger.error(f"Iterative summarization failed: {e}")
            # Fallback: truncate aggressively
            logger.warning("Falling back to truncation")
            target_tokens = int(threshold_tokens * 0.6)
            max_chars = target_tokens * chars_per_token

            truncated_messages = []
            for msg in messages:
                truncated_msg = msg.copy()
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) > max_chars:
                    truncated_msg["content"] = content[:max_chars] + "\n[...TRUNCATED...]"
                truncated_messages.append(truncated_msg)

            return truncated_messages

    # Messages to summarize (everything except recent)
    messages_to_summarize = messages[:-keep_recent]
    recent_messages = messages[-keep_recent:]

    logger.info(
        f"Compacting context: summarizing {len(messages_to_summarize)} messages, "
        f"keeping {len(recent_messages)} recent"
    )

    # Get summarizer agent
    if not agent_registry:
        logger.warning("Agent registry unavailable, cannot compact context")
        return messages

    summarizer_agent = agent_registry.get_agent_by_name("summarizer")
    if not summarizer_agent or not summarizer_agent.get("is_enabled"):
        logger.warning("Summarizer agent unavailable, cannot compact context")
        return messages

    # Build compaction prompt
    compaction_prompt = build_compaction_input(messages_to_summarize)

    # Create messages for summarizer
    summarizer_messages = [{"role": "user", "content": compaction_prompt}]

    try:
        # Execute summarizer (non-streaming, no tools)
        # Pass full payload - automatic compaction will handle context size
        logger.info("Calling @summarizer for context compaction")
        summarizer_response = await _execute_agent_request(
            agent=summarizer_agent,
            messages=summarizer_messages,
            user_message=compaction_prompt,
            original_payload=original_payload,
        )

        summarizer_content = _get_agent_response(summarizer_response)

        # Extract clean summary
        summary = extract_summary(summarizer_content)

        logger.info(f"Context compaction summary: {summary[:100]}...")

        # Create summary message
        summary_message = {
            "role": "system",
            "content": f"[CONTEXT SUMMARY]\n{summary}",
        }

        # Return: summary + recent messages
        compacted_messages = [summary_message] + recent_messages

        # Log the compaction result
        old_tokens = estimate_messages_tokens(messages, chars_per_token)
        new_tokens = estimate_messages_tokens(compacted_messages, chars_per_token)
        reduction_pct = (1 - new_tokens / old_tokens) * 100 if old_tokens > 0 else 0

        logger.info(
            f"Context compacted: {old_tokens} -> {new_tokens} tokens "
            f"({reduction_pct:.1f}% reduction)"
        )

        return compacted_messages

    except Exception as e:
        logger.error(f"Context compaction failed: {e}")
        # Return original messages if compaction fails
        return messages


def _should_evaluate_subagent(
    calling_agent_name: str,
    subagent_name: str,
    depth: int,
) -> bool:
    """Determine if subagent's work should be evaluated.

    Args:
        calling_agent_name: Agent that delegated
        subagent_name: Agent that was delegated to
        depth: Current delegation depth

    Returns:
        True if evaluation should occur
    """
    # Don't evaluate meta-agents
    if _is_meta_agent(subagent_name):
        return False

    # Don't evaluate if calling agent is meta-agent (they're trusted)
    if _is_meta_agent(calling_agent_name):
        return False

    # Don't evaluate at depth 0 (direct user request, not delegation)
    return depth != 0


async def _evaluate_subagent_response(
    task_description: str,
    subagent_name: str,
    subagent_response: str,
    calling_agent_name: str,
    original_payload: dict[str, Any],
) -> EvaluationResult:
    """Call evaluator agent to assess subagent's work.

    Args:
        task_description: What the subagent was asked to do
        subagent_name: Name of the subagent
        subagent_response: The subagent's output
        calling_agent_name: Name of agent that delegated
        original_payload: Original request payload (full payload for context)

    Returns:
        EvaluationResult with status and instructions
    """
    if not agent_registry:
        return EvaluationResult(
            status="failed",
            reason="Agent registry not available",
            is_valid=False,
        )

    evaluator_agent = agent_registry.get_agent_by_name("evaluator")
    if not evaluator_agent or not evaluator_agent.get("is_enabled"):
        logger.warning("Evaluator agent not available, assuming task complete")
        return EvaluationResult(
            status="complete",
            reason="Evaluator unavailable, assuming success",
            confidence=0.5,
        )

    # Format input for evaluator
    evaluator_prompt = format_evaluator_input(
        task_description=task_description,
        subagent_name=subagent_name,
        subagent_response=subagent_response,
    )

    # Create messages for evaluator
    evaluator_messages = [{"role": "user", "content": evaluator_prompt}]

    logger.info(f"Calling evaluator to assess @{subagent_name}'s work")

    try:
        # Execute evaluator (non-streaming, no tools)
        # Pass full payload - automatic compaction will handle context size
        evaluator_response = await _execute_agent_request(
            agent=evaluator_agent,
            messages=evaluator_messages,
            user_message=evaluator_prompt,
            original_payload=original_payload,
        )

        evaluator_content = _get_agent_response(evaluator_response)

        # Parse evaluator's JSON output
        evaluation = parse_evaluator_response(evaluator_content)

        logger.info(
            f"Evaluator assessment: status={evaluation.status}, "
            f"confidence={evaluation.confidence}, reason={evaluation.reason[:50]}"
        )

        return evaluation

    except Exception as e:
        logger.error(f"Evaluator execution failed: {e}")
        return EvaluationResult(
            status="failed",
            reason=f"Evaluator error: {e}",
            is_valid=False,
        )


async def _summarize_subagent_work(
    task_description: str,
    subagent_name: str,
    full_context: str,
    evaluation: EvaluationResult,
    original_payload: dict[str, Any],
) -> str:
    """Call summarizer agent to compact subagent's work.

    Args:
        task_description: What the task was
        subagent_name: Name of subagent
        full_context: Full context from subagent execution
        evaluation: Evaluator's assessment
        original_payload: Original request payload (full payload - summarizer needs full history)

    Returns:
        Concise summary for calling agent
    """
    if not agent_registry:
        return f"Summary: @{subagent_name} completed task (registry unavailable)"

    summarizer_agent = agent_registry.get_agent_by_name("summarizer")
    if not summarizer_agent or not summarizer_agent.get("is_enabled"):
        logger.warning("Summarizer agent not available, using raw response")
        # Fallback: truncate the response
        return extract_summary(full_context[:500])

    # Format input for summarizer
    summarizer_prompt = format_summarizer_input(
        task_description=task_description,
        subagent_name=subagent_name,
        full_context=full_context,
        evaluator_assessment=evaluation,
    )

    # Create messages for summarizer
    summarizer_messages = [{"role": "user", "content": summarizer_prompt}]

    logger.info(f"Calling summarizer to compact @{subagent_name}'s work")

    try:
        # Execute summarizer (non-streaming, no tools)
        # Pass full payload - summarizer needs full history for proper summarization
        # Automatic compaction will handle context size
        summarizer_response = await _execute_agent_request(
            agent=summarizer_agent,
            messages=summarizer_messages,
            user_message=summarizer_prompt,
            original_payload=original_payload,
        )

        summarizer_content = _get_agent_response(summarizer_response)

        # Extract clean summary
        summary = extract_summary(summarizer_content)

        logger.info(f"Summarizer output: {summary[:100]}...")

        return summary

    except Exception as e:
        logger.error(f"Summarizer execution failed: {e}")
        # Fallback: use truncated context
        return f"Summary: @{subagent_name} worked on task (error: {e})"


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
    max_evaluation_retries: int = 10,
) -> dict[str, Any]:
    """Route message to mentioned agents sequentially with agent-to-agent support and evaluation.

    Each agent executes one after another. After each agent completes, checks
    for mentions of other agents. If mentions found and depth < max_depth,
    recursively routes to those agents.

    For subagents (depth > 0), evaluates task completion and optionally retries
    with evaluator guidance. Summarizes results to prevent context pollution.

    Args:
        routing_info: Dict with 'agents' (list of agent configs) and 'routing_type'
        messages: Original messages array
        original_payload: Original request payload
        depth: Current recursion depth (for agent-to-agent)
        max_depth: Maximum recursion depth to prevent infinite loops
        agent_chain: List of agent names showing delegation chain (e.g., ["assistant", "researcher"])
        max_evaluation_retries: Maximum retries when evaluator says task incomplete

    Returns:
        Combined response dict with all agent results
    """
    agents = routing_info.get("agents", [])

    # Initialize agent chain if not provided
    if agent_chain is None:
        agent_chain = []

    results: list[dict[str, Any]] = []

    # Determine calling agent for evaluation context
    calling_agent_name = agent_chain[-1] if agent_chain else "user"

    for agent in agents:
        agent_name = agent.get("name", "unknown")

        # Add this agent to the chain
        current_chain = agent_chain + [agent_name]

        logger.info(f"Routing to agent @{agent_name} (depth={depth}, chain={current_chain})")

        # Determine if this is a subagent delegation (depth > 0)
        is_subagent_delegation = depth > 0

        # Get task description for evaluation
        task_description = _get_user_prompt(messages)

        # For subagents, use isolated context to prevent pollution
        if is_subagent_delegation:
            # Create isolated context with just the delegation message
            isolated_messages = create_isolated_context(
                original_task=task_description,
                calling_agent_name=calling_agent_name,
                delegation_message=messages[-1].get("content", "") if messages else "",
            )
            execution_messages = isolated_messages
            logger.info(f"Using isolated context for subagent @{agent_name}")
        else:
            execution_messages = messages

        # Execute agent with retry loop for evaluation
        final_response_content = ""
        evaluation_attempts = 0
        evaluation_result = None

        while True:
            # Get the actual user message for RAG query
            actual_user_msg = _get_user_prompt(execution_messages)

            # Execute agent request (non-streaming)
            response = await _execute_agent_request(
                agent=agent,
                messages=execution_messages,
                user_message=actual_user_msg,
                original_payload=original_payload,
            )

            # Extract response content
            response_content = _get_agent_response(response)

            # Check if we should evaluate this subagent's work
            should_eval = _should_evaluate_subagent(
                calling_agent_name=calling_agent_name,
                subagent_name=agent_name,
                depth=depth,
            )

            if should_eval and evaluation_attempts < max_evaluation_retries:
                # Evaluate the response
                evaluation_result = await _evaluate_subagent_response(
                    task_description=task_description,
                    subagent_name=agent_name,
                    subagent_response=response_content,
                    calling_agent_name=calling_agent_name,
                    original_payload=original_payload,
                )

                if (
                    evaluation_result.status == "incomplete"
                    and evaluation_result.retry_instructions
                ):
                    # Retry with evaluator guidance
                    evaluation_attempts += 1
                    logger.info(
                        f"Evaluator says @{agent_name} incomplete (attempt {evaluation_attempts}), "
                        f"retrying with guidance"
                    )

                    # Create retry context with evaluator instructions
                    execution_messages = create_retry_context(
                        original_task=task_description,
                        previous_response=response_content,
                        evaluator_instructions=evaluation_result.retry_instructions,
                        attempt_number=evaluation_attempts,
                    )
                    continue  # Retry loop

            # Either no evaluation needed, or evaluation complete, or max retries reached
            final_response_content = response_content
            break  # Exit retry loop

        # After execution (with or without evaluation), check for agent-to-agent mentions
        mentioned_agents = _check_response_for_mentions(final_response_content)

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
                # Build routing info for sub-agents (fix bug: was missing)
                sub_routing_info = {
                    "agents": sub_agents,
                    "routing_type": "mention",
                }

                # Create sub-messages with the agent's response as context
                # For subagent delegations, use summary if available, otherwise use response
                content_for_next_agent = final_response_content
                if is_subagent_delegation and evaluation_result:
                    # Summarize the work before passing to next agent
                    summary = await _summarize_subagent_work(
                        task_description=task_description,
                        subagent_name=agent_name,
                        full_context=response_content,  # Use the actual response as context
                        evaluation=evaluation_result,
                        original_payload=original_payload,
                    )
                    content_for_next_agent = summary
                    logger.info(
                        f"Summarized @{agent_name}'s work for downstream: {summary[:100]}..."
                    )

                sub_messages = messages.copy()
                sub_messages.append({"role": "assistant", "content": content_for_next_agent})

                # Recursively route to mentioned agents with updated chain
                sub_response = await _route_to_agents(
                    routing_info=sub_routing_info,
                    messages=sub_messages,
                    original_payload=original_payload,
                    depth=depth + 1,
                    max_depth=max_depth,
                    agent_chain=current_chain,  # Pass the current chain to sub-agents
                    max_evaluation_retries=max_evaluation_retries,
                )

                # Combine results
                results.append(
                    {
                        "agent": agent,
                        "content": final_response_content,
                        "evaluation": evaluation_result,
                        "sub_agents": sub_response.get("results", []),
                    }
                )
            else:
                # No sub_agents found, just record result
                # For subagent delegations, summarize before returning
                content_to_return = final_response_content
                if is_subagent_delegation and evaluation_result:
                    content_to_return = await _summarize_subagent_work(
                        task_description=task_description,
                        subagent_name=agent_name,
                        full_context=response_content,
                        evaluation=evaluation_result,
                        original_payload=original_payload,
                    )
                    logger.info(f"Summarized @{agent_name}'s work: {content_to_return[:100]}...")

                results.append(
                    {
                        "agent": agent,
                        "content": content_to_return,
                        "evaluation": evaluation_result,
                    }
                )
        else:
            # No mentions or max depth reached
            # For subagent delegations, summarize before returning
            content_to_return = final_response_content
            if is_subagent_delegation and evaluation_result:
                content_to_return = await _summarize_subagent_work(
                    task_description=task_description,
                    subagent_name=agent_name,
                    full_context=response_content,
                    evaluation=evaluation_result,
                    original_payload=original_payload,
                )
                logger.info(f"Summarized @{agent_name}'s work: {content_to_return[:100]}...")

            results.append(
                {
                    "agent": agent,
                    "content": content_to_return,
                    "evaluation": evaluation_result,
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
    max_evaluation_retries: int = 10,
) -> StreamingResponse:
    """Route message to mentioned agents sequentially with streaming and agent-to-agent support.

    Each agent's response is streamed with markers:
    __AGENT_START__{"agent": "name", "depth": N, "chain": ["agent1", "agent2"]}__END__
    [streaming content]
    __AGENT_END__{"agent": "name", "depth": N, "chain": ["agent1", "agent2"]}__END__

    For subagents (depth > 0), after streaming completes:
    __EVALUATION__{"status": "...", "reason": "..."}__END__
    __SUMMARY__{"content": "..."}__END__

    After each agent completes, checks for mentions of other agents.
    If mentions found and depth < max_depth, recursively streams sub-agent responses.

    Note: Streaming doesn support retry loop. Evaluation happens after stream ends.
    If evaluator says incomplete, a marker is sent but retry happens on next request.

    Args:
        routing_info: Dict with 'agents' (list of agent configs) and 'routing_type'
        messages: Original messages array
        original_payload: Original request payload
        depth: Current recursion depth (for agent-to-agent)
        max_depth: Maximum recursion depth to prevent infinite loops
        agent_chain: List of agent names showing delegation chain (e.g., ["assistant", "researcher"])
        max_evaluation_retries: Maximum retries for evaluation (used for non-streaming fallback)

    Returns:
        StreamingResponse with agent markers and content
    """
    agents = routing_info.get("agents", [])

    # Initialize agent chain if not provided
    if agent_chain is None:
        agent_chain = []

    # Determine calling agent for evaluation context
    calling_agent_name = agent_chain[-1] if agent_chain else "user"

    async def stream_generator():
        """Generate streaming response with agent markers."""
        # We need to track content for each agent separately for evaluation
        agent_streams_data: dict[str, str] = {}

        for agent in agents:
            agent_name = agent.get("name", "unknown")
            logger.info(f"Streaming to agent @{agent_name} (depth={depth}, chain={agent_chain})")

            # Add this agent to the chain
            current_chain = agent_chain + [agent_name]

            # Determine if this is a subagent delegation
            is_subagent_delegation = depth > 0

            # Get task description for evaluation
            task_description = _get_user_prompt(messages)

            # Retry loop for evaluation
            evaluation_attempts = 0
            evaluation_result = None
            final_streamed_content = ""

            while evaluation_attempts <= max_evaluation_retries:
                # Send agent start marker with chain (only on first attempt)
                if evaluation_attempts == 0:
                    chain_json = json.dumps(current_chain)
                    start_marker = f'__AGENT_START__{{"agent": "{agent_name}", "depth": {depth}, "chain": {chain_json}}}__END__\n'
                    yield start_marker.encode()

                # Prepare messages for execution
                if is_subagent_delegation:
                    if evaluation_attempts == 0:
                        # First attempt: use isolated context
                        isolated_messages = create_isolated_context(
                            original_task=task_description,
                            calling_agent_name=calling_agent_name,
                            delegation_message=messages[-1].get("content", "") if messages else "",
                        )
                        execution_messages = isolated_messages
                        logger.info(
                            f"Using isolated context for subagent @{agent_name} (streaming)"
                        )
                    else:
                        # Retry: use retry context with evaluator guidance
                        retry_messages = create_retry_context(
                            original_task=task_description,
                            previous_response=streamed_content,
                            evaluator_instructions=evaluation_result.retry_instructions
                            if evaluation_result
                            else "",
                            attempt_number=evaluation_attempts,
                        )
                        execution_messages = retry_messages

                        # Send retry indicator to user
                        retry_msg = (
                            f"\n🔄 **Retry attempt {evaluation_attempts}** for @{agent_name}\n"
                        )
                        retry_msg += f"Guidance from evaluator: {evaluation_result.retry_instructions if evaluation_result else 'N/A'}\n"
                        retry_sse = {
                            "choices": [
                                {
                                    "delta": {"role": "assistant", "content": retry_msg},
                                    "finish_reason": None,
                                    "index": 0,
                                }
                            ]
                        }
                        yield f"data: {json.dumps(retry_sse)}\n\n".encode()

                        logger.info(
                            f"Created retry context (attempt #{evaluation_attempts}) for @{agent_name}"
                        )
                else:
                    execution_messages = messages

                # Get the actual user message for RAG query
                actual_user_msg = _get_user_prompt(execution_messages)

                # Execute agent request (streaming)
                response = await _execute_agent_request_streaming(
                    agent=agent,
                    messages=execution_messages,
                    user_message=actual_user_msg,
                    original_payload=original_payload,
                )

                # Stream the agent's response and collect content
                streamed_content = ""
                if isinstance(response, StreamingResponse):
                    async for chunk in response.body_iterator:
                        # Skip [DONE] markers - we'll send our own at the end
                        if isinstance(chunk, bytes):
                            chunk_str = chunk.decode("utf-8")
                            if chunk_str.strip() == "data: [DONE]\n" or "data: [DONE]" in chunk_str:
                                continue  # Skip [DONE] markers from agent streams

                        # Collect content for evaluation and mention checking
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
                                                streamed_content += content
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                pass
                        yield chunk

                # Store streamed content for this agent
                final_streamed_content = streamed_content
                agent_streams_data[agent_name] = streamed_content

                # DEBUG: Log stream completion
                logger.info(
                    f"DEBUG: Stream completed for @{agent_name}, "
                    f"streamed_content length: {len(streamed_content)}, "
                    f"is_subagent_delegation: {is_subagent_delegation}, "
                    f"depth: {depth}"
                )

                # For subagent delegations, evaluate after stream ends
                if is_subagent_delegation:
                    should_eval = _should_evaluate_subagent(
                        calling_agent_name=calling_agent_name,
                        subagent_name=agent_name,
                        depth=depth,
                    )

                    # DEBUG: Log evaluation decision
                    logger.info(
                        f"DEBUG: Evaluation decision for @{agent_name}: "
                        f"should_eval={should_eval}, "
                        f"calling_agent={calling_agent_name}, "
                        f"subagent={agent_name}, "
                        f"depth={depth}"
                    )

                    if should_eval:
                        # Evaluate the streamed response
                        evaluation_result = await _evaluate_subagent_response(
                            task_description=task_description,
                            subagent_name=agent_name,
                            subagent_response=streamed_content,
                            calling_agent_name=calling_agent_name,
                            original_payload=original_payload,
                        )

                        # Send evaluation as visible text in the chat (SSE format)
                        status_emoji = {"complete": "✅", "incomplete": "⚠️", "failed": "❌"}.get(
                            evaluation_result.status, "📋"
                        )

                        eval_visible = f"\n\n---\n**{status_emoji} EVALUATION of @{agent_name}** (attempt {evaluation_attempts + 1})\n"
                        eval_visible += f"**Status:** {evaluation_result.status} ({int(evaluation_result.confidence * 100)}% confidence)\n"
                        eval_visible += f"**Reason:** {evaluation_result.reason}\n"
                        if evaluation_result.retry_instructions:
                            eval_visible += (
                                f"**Retry Instructions:** {evaluation_result.retry_instructions}\n"
                            )
                        if evaluation_result.summary_focus:
                            eval_visible += (
                                f"**Summary Focus:** {evaluation_result.summary_focus}\n"
                            )
                        eval_visible += "---\n"

                        # Wrap in SSE format for UI to display
                        eval_sse = {
                            "choices": [
                                {
                                    "delta": {"role": "assistant", "content": eval_visible},
                                    "finish_reason": None,
                                    "index": 0,
                                }
                            ]
                        }
                        yield f"data: {json.dumps(eval_sse)}\n\n".encode()

                        # Also send the marker for internal processing
                        eval_marker_data = {
                            "status": evaluation_result.status,
                            "reason": evaluation_result.reason,
                            "confidence": evaluation_result.confidence,
                        }
                        if evaluation_result.retry_instructions:
                            eval_marker_data["retry_instructions"] = (
                                evaluation_result.retry_instructions
                            )

                        eval_marker = f"__EVALUATION__{json.dumps(eval_marker_data)}__END__\n"
                        yield eval_marker.encode()

                        # Log full evaluation details
                        logger.info(
                            f"Full evaluation for @{agent_name}: "
                            f"status={evaluation_result.status}, "
                            f"confidence={evaluation_result.confidence}, "
                            f"reason={evaluation_result.reason}"
                        )
                        if evaluation_result.retry_instructions:
                            logger.info(
                                f"Retry instructions: {evaluation_result.retry_instructions}"
                            )
                        if evaluation_result.summary_focus:
                            logger.info(f"Summary focus: {evaluation_result.summary_focus}")

                        # Check if we should retry
                        if (
                            evaluation_result.status == "incomplete"
                            and evaluation_result.retry_instructions
                            and evaluation_attempts < max_evaluation_retries
                        ):
                            evaluation_attempts += 1

                            # Send retry attempt message to UI immediately
                            retry_message = f"\n\n🔄 **Retrying @{agent_name}** (attempt {evaluation_attempts}/{max_evaluation_retries})...\n\n"
                            retry_sse = {
                                "choices": [
                                    {
                                        "delta": {"role": "assistant", "content": retry_message},
                                        "finish_reason": None,
                                        "index": 0,
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(retry_sse)}\n\n".encode()

                            logger.info(
                                f"Evaluator says @{agent_name} incomplete (attempt {evaluation_attempts}), "
                                f"retrying with guidance"
                            )

                            continue  # Retry loop - will re-execute with retry context
                        else:
                            # Complete, failed, or max retries reached
                            # Summarize and break
                            break
                    else:
                        # No evaluation needed
                        break
                else:
                    # Not subagent delegation, no retry
                    break

            # After retry loop completes, summarize if needed
            if is_subagent_delegation and evaluation_result and should_eval:
                # Summarize the work for the calling agent
                summary = await _summarize_subagent_work(
                    task_description=task_description,
                    subagent_name=agent_name,
                    full_context=final_streamed_content,
                    evaluation=evaluation_result,
                    original_payload=original_payload,
                )

                summary_visible = (
                    f"\n**📝 SUMMARY for calling agent @{calling_agent_name}:** {summary}\n"
                )
                # Wrap in SSE format for UI to display
                summary_sse = {
                    "choices": [
                        {
                            "delta": {"role": "assistant", "content": summary_visible},
                            "finish_reason": None,
                            "index": 0,
                        }
                    ]
                }
                yield f"data: {json.dumps(summary_sse)}\n\n".encode()

                # Also send marker for internal processing
                summary_marker = f"__SUMMARY__{json.dumps({'content': summary})}__END__\n"
                yield summary_marker.encode()

                logger.info(f"Full summary for @{agent_name}: {summary}")
                logger.info(f"Sent evaluation and summary markers for @{agent_name}")

                # Store summary for potential use by calling agent
                agent_streams_data[f"{agent_name}_summary"] = summary

            # Send agent end marker with chain
            chain_json = json.dumps(current_chain)
            end_marker = f'__AGENT_END__{{"agent": "{agent_name}", "depth": {depth}, "chain": {chain_json}}}__END__\n'
            yield end_marker.encode()

            # Check for agent-to-agent mentions in the streamed content
            mentioned_agents = _check_response_for_mentions(final_streamed_content)

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
                    # For subagent delegations, use summary if available
                    content_for_next_agent = final_streamed_content
                    if is_subagent_delegation:
                        # Get the summary we just sent
                        # The summary should be used for the next agent's context
                        # But in streaming, we already sent the full content
                        # For now, pass the full content (streaming limitation)
                        pass

                    # Create sub-messages with the agent's response as context
                    sub_messages = messages.copy()
                    sub_messages.append({"role": "assistant", "content": content_for_next_agent})

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
                        max_evaluation_retries=max_evaluation_retries,
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
            yield b"data: [DONE]\n\n"

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
                        yield b"data: [DONE]\n\n"

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
            logger.info("Executing tool loop (streaming=False)")
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
