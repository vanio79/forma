"""Web UI API endpoints for Forma.

Provides REST endpoints for the SPA frontend to retrieve request history and manage upstreams.
"""

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ui", tags=["ui"])


def _get_proxy():
    """Get the proxy instance (lazy loading to avoid circular imports)."""
    from forma.main import proxy

    return proxy


def _get_db():
    """Get the database instance from main (lazy loading to avoid circular imports)."""
    from forma.main import db

    return db


def _get_agent_registry():
    """Get the agent registry instance (lazy loading to avoid circular imports)."""
    from forma.main import agent_registry

    return agent_registry


@router.get("/stats")
async def get_stats():
    """Get summary statistics for the dashboard."""
    db = _get_db()
    stats = db.get_stats()
    return JSONResponse(stats)


@router.get("/requests")
async def get_requests(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Get list of recent requests."""
    db = _get_db()
    requests = db.get_requests(limit=limit, offset=offset)

    # Format for frontend
    return JSONResponse(
        {
            "requests": [
                {
                    "id": r["id"],
                    "model": r["model"],
                    "user_prompt": r["user_prompt"],
                    "timestamp": r["timestamp"],
                    "timestamp_formatted": datetime.fromtimestamp(r["timestamp"]).isoformat(),
                    "extraction_ms": r["extraction_ms"],
                    "has_extraction": bool(
                        r["extraction_response"] and len(r["extraction_response"]) > 0
                    ),
                    "has_augmentation": bool(
                        r["augmented_prompt"] and len(r["augmented_prompt"]) > 0
                    ),
                }
                for r in requests
            ],
            "limit": limit,
            "offset": offset,
        }
    )


@router.get("/requests/{request_id}")
async def get_request_detail(request_id: str):
    """Get detailed information for a specific request."""
    db = _get_db()
    detail = db.get_request_detail(request_id)

    if not detail:
        raise HTTPException(status_code=404, detail="Request not found")

    # Format request
    request = detail["request"]
    formatted_request = {
        "id": request["id"],
        "model": request["model"],
        "user_prompt": request["user_prompt"],
        "history": request["history"],
        "extraction_response": request["extraction_response"],
        "extraction_prompt": request.get("extraction_prompt", ""),
        "extraction_ms": request["extraction_ms"],
        "augmented_prompt": request["augmented_prompt"],
        "agent_response": request["agent_response"],
        "timestamp": request["timestamp"],
        "timestamp_formatted": datetime.fromtimestamp(request["timestamp"]).isoformat(),
    }

    # Group extractions by type
    extractions_by_type = {}
    for e in detail.get("extractions", []):
        ext_type = e["extraction_type"]
        if ext_type not in extractions_by_type:
            extractions_by_type[ext_type] = []
        extractions_by_type[ext_type].append(
            {
                "id": e["id"],
                "data": e["data"],
                "confidence": e["confidence"],
            }
        )

    # Group retrievals by type
    retrievals_by_type = {}
    for r in detail.get("retrievals", []):
        ret_type = r["retrieval_type"]
        if ret_type not in retrievals_by_type:
            retrievals_by_type[ret_type] = []
        retrievals_by_type[ret_type].append(
            {
                "id": r["id"],
                "data": r["data"],
                "confidence": r["confidence"],
                "score": r["score"],
            }
        )

    return JSONResponse(
        {
            "request": formatted_request,
            "extractions": extractions_by_type,
            "retrievals": retrievals_by_type,
        }
    )


@router.delete("/clear")
async def clear_data():
    """Clear all request history data."""
    db = _get_db()
    db.clear_all()
    return JSONResponse({"status": "ok", "message": "All request history cleared"})


# === Upstream Management ===


@router.get("/upstreams")
async def get_upstreams():
    """Get all upstream configurations."""
    db = _get_db()
    upstreams = db.get_upstreams()
    return JSONResponse({"upstreams": upstreams})


@router.post("/upstreams")
async def create_upstream(
    name: str = Query(..., min_length=1, max_length=100),
    upstream_model: str = Query(default="", max_length=100),
    base_url: str = Query(..., min_length=1),
    api_key: str = Query(default=""),
    timeout: float = Query(default=300.0, ge=1.0, le=600.0),
    is_enabled: bool = Query(default=True),
):
    """Create a new upstream configuration.

    Args:
        name: Local model name (routing key - client requests with this model go to this upstream)
        upstream_model: Model name to send to upstream API (if empty, uses name)
        base_url: Upstream API base URL
        api_key: API key for authentication
        timeout: Request timeout in seconds
        is_enabled: Whether this upstream is enabled
    """
    db = _get_db()

    # Check if name already exists
    existing = db.get_upstream_by_name(name)
    if existing:
        raise HTTPException(status_code=400, detail="Upstream name already exists")

    upstream_id = str(uuid.uuid4())
    db.create_upstream(
        id=upstream_id,
        name=name,
        upstream_model=upstream_model,
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        timeout=timeout,
        is_enabled=is_enabled,
    )

    # Reload upstreams in proxy
    proxy = _get_proxy()
    if proxy:
        proxy.reload_upstreams()

    return JSONResponse(
        {
            "status": "ok",
            "message": "Upstream created",
            "upstream": db.get_upstream_by_id(upstream_id),
        }
    )


@router.get("/upstreams/{upstream_id}")
async def get_upstream(upstream_id: str):
    """Get a specific upstream configuration."""
    db = _get_db()
    upstream = db.get_upstream_by_id(upstream_id)

    if not upstream:
        raise HTTPException(status_code=404, detail="Upstream not found")

    return JSONResponse({"upstream": upstream})


@router.put("/upstreams/{upstream_id}")
async def update_upstream(
    upstream_id: str,
    name: str = Query(default=None, min_length=1, max_length=100),
    upstream_model: str = Query(default=None, max_length=100),
    base_url: str = Query(default=None, min_length=1),
    api_key: str = Query(default=None),
    timeout: float = Query(default=None, ge=1.0, le=600.0),
    is_enabled: bool = Query(default=None),
):
    """Update an upstream configuration.

    Args:
        name: Local model name (routing key)
        upstream_model: Model name to send to upstream API (if empty, uses name)
        base_url: Upstream API base URL
        api_key: API key for authentication
        timeout: Request timeout in seconds
        is_enabled: Whether this upstream is enabled
    """
    db = _get_db()

    upstream = db.get_upstream_by_id(upstream_id)
    if not upstream:
        raise HTTPException(status_code=404, detail="Upstream not found")

    # Check name uniqueness if changing
    if name and name != upstream["name"]:
        existing = db.get_upstream_by_name(name)
        if existing:
            raise HTTPException(status_code=400, detail="Upstream name already exists")

    # Update with provided values, keeping existing for None
    db.update_upstream(
        id=upstream_id,
        name=name or upstream["name"],
        upstream_model=upstream_model
        if upstream_model is not None
        else upstream.get("upstream_model", ""),
        base_url=(base_url or upstream["base_url"]).rstrip("/"),
        api_key=api_key if api_key is not None else upstream["api_key"],
        timeout=timeout if timeout is not None else upstream["timeout"],
        is_enabled=is_enabled if is_enabled is not None else upstream["is_enabled"],
    )

    # Reload upstreams in proxy
    proxy = _get_proxy()
    if proxy:
        proxy.reload_upstreams()

    return JSONResponse(
        {
            "status": "ok",
            "message": "Upstream updated",
            "upstream": db.get_upstream_by_id(upstream_id),
        }
    )


@router.delete("/upstreams/{upstream_id}")
async def delete_upstream(upstream_id: str):
    """Delete an upstream configuration."""
    db = _get_db()

    upstream = db.get_upstream_by_id(upstream_id)
    if not upstream:
        raise HTTPException(status_code=404, detail="Upstream not found")

    db.delete_upstream(upstream_id)

    # Reload upstreams in proxy
    proxy = _get_proxy()
    if proxy:
        proxy.reload_upstreams()

    return JSONResponse({"status": "ok", "message": "Upstream deleted"})


@router.post("/upstreams/reload")
async def reload_upstreams():
    """Reload upstream configurations from database."""
    proxy = _get_proxy()
    if proxy:
        proxy.reload_upstreams()
        return JSONResponse({"status": "ok", "message": "Upstreams reloaded"})
    return JSONResponse({"status": "error", "message": "Proxy not initialized"})


# === Agent Management ===


@router.get("/agents")
async def get_agents():
    """Get all agent configurations."""
    agent_registry = _get_agent_registry()
    if not agent_registry:
        return JSONResponse({"agents": [], "message": "Agent system disabled"})

    agents = agent_registry.get_all_agents()

    # Format agents for frontend
    formatted_agents = []
    for agent in agents:
        import json

        tool_whitelist_str = agent.get("tool_whitelist", "[]")
        try:
            tool_whitelist = (
                json.loads(tool_whitelist_str)
                if isinstance(tool_whitelist_str, str)
                else tool_whitelist_str
            )
        except (json.JSONDecodeError, TypeError):
            tool_whitelist = []

        formatted_agents.append(
            {
                "id": agent["id"],
                "name": agent["name"],
                "purpose": agent["purpose"],
                "instruction_prompt": agent["instruction_prompt"],
                "upstream_id": agent["upstream_id"],
                "tools_enabled": agent["tools_enabled"],
                "tool_whitelist": tool_whitelist,
                "max_iterations": agent["max_iterations"],
                "is_enabled": agent["is_enabled"],
                "created_at": agent["created_at"],
                "updated_at": agent["updated_at"],
            }
        )

    return JSONResponse({"agents": formatted_agents})


@router.post("/agents")
async def create_agent(
    name: str = Query(..., min_length=1, max_length=100),
    purpose: str = Query(..., min_length=1),
    instruction_prompt: str = Query(..., min_length=1),
    upstream_id: str | None = Query(default=None),
    tools_enabled: bool = Query(default=True),
    tool_whitelist: str = Query(default="[]"),  # JSON array as string
    max_iterations: int = Query(default=5, ge=1, le=20),
    is_enabled: bool = Query(default=True),
):
    """Create a new agent configuration.

    Args:
        name: Unique agent name (used for @agent_name mentions)
        purpose: Brief description of agent's role
        instruction_prompt: System prompt / instruction for this agent
        upstream_id: Reference to upstream config (null = use default)
        tools_enabled: Whether tools are enabled for this agent
        tool_whitelist: JSON array of allowed tool names (empty = all tools)
        max_iterations: Max tool iterations for this agent
        is_enabled: Whether agent is active
    """
    agent_registry = _get_agent_registry()
    if not agent_registry:
        raise HTTPException(status_code=503, detail="Agent system disabled")

    try:
        import json

        whitelist = json.loads(tool_whitelist) if tool_whitelist else []
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid tool_whitelist JSON")

    try:
        agent_id = agent_registry.register_agent(
            name=name,
            purpose=purpose,
            instruction_prompt=instruction_prompt,
            upstream_id=upstream_id,
            tools_enabled=tools_enabled,
            tool_whitelist=whitelist,
            max_iterations=max_iterations,
            is_enabled=is_enabled,
        )

        return JSONResponse(
            {
                "status": "ok",
                "message": "Agent created",
                "agent": agent_registry.get_agent(agent_id),
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get a specific agent configuration."""
    agent_registry = _get_agent_registry()
    if not agent_registry:
        raise HTTPException(status_code=503, detail="Agent system disabled")

    agent = agent_registry.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    import json

    tool_whitelist_str = agent.get("tool_whitelist", "[]")
    try:
        tool_whitelist = (
            json.loads(tool_whitelist_str)
            if isinstance(tool_whitelist_str, str)
            else tool_whitelist_str
        )
    except (json.JSONDecodeError, TypeError):
        tool_whitelist = []

    return JSONResponse(
        {
            "agent": {
                "id": agent["id"],
                "name": agent["name"],
                "purpose": agent["purpose"],
                "instruction_prompt": agent["instruction_prompt"],
                "upstream_id": agent["upstream_id"],
                "tools_enabled": agent["tools_enabled"],
                "tool_whitelist": tool_whitelist,
                "max_iterations": agent["max_iterations"],
                "is_enabled": agent["is_enabled"],
                "created_at": agent["created_at"],
                "updated_at": agent["updated_at"],
            }
        }
    )


@router.put("/agents/{agent_id}")
async def update_agent(
    agent_id: str,
    name: str | None = Query(default=None, min_length=1, max_length=100),
    purpose: str | None = Query(default=None, min_length=1),
    instruction_prompt: str | None = Query(default=None, min_length=1),
    upstream_id: str | None = Query(default=None),
    tools_enabled: bool | None = Query(default=None),
    tool_whitelist: str | None = Query(default=None),  # JSON array as string
    max_iterations: int | None = Query(default=None, ge=1, le=20),
    is_enabled: bool | None = Query(default=None),
):
    """Update an agent configuration.

    Args:
        name: Unique agent name
        purpose: Brief description of agent's role
        instruction_prompt: System prompt / instruction for this agent
        upstream_id: Reference to upstream config (null = use default)
        tools_enabled: Whether tools are enabled for this agent
        tool_whitelist: JSON array of allowed tool names (empty = all tools)
        max_iterations: Max tool iterations for this agent
        is_enabled: Whether agent is active
    """
    agent_registry = _get_agent_registry()
    if not agent_registry:
        raise HTTPException(status_code=503, detail="Agent system disabled")

    # Check if agent exists
    existing = agent_registry.get_agent(agent_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Build updates dict
    updates = {}
    if name is not None:
        updates["name"] = name
    if purpose is not None:
        updates["purpose"] = purpose
    if instruction_prompt is not None:
        updates["instruction_prompt"] = instruction_prompt
    if upstream_id is not None:
        updates["upstream_id"] = upstream_id
    if tools_enabled is not None:
        updates["tools_enabled"] = tools_enabled
    if tool_whitelist is not None:
        try:
            import json

            updates["tool_whitelist"] = json.loads(tool_whitelist)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid tool_whitelist JSON")
    if max_iterations is not None:
        updates["max_iterations"] = max_iterations
    if is_enabled is not None:
        updates["is_enabled"] = is_enabled

    try:
        agent_registry.update_agent(agent_id, **updates)

        return JSONResponse(
            {
                "status": "ok",
                "message": "Agent updated",
                "agent": agent_registry.get_agent(agent_id),
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str):
    """Delete an agent configuration."""
    agent_registry = _get_agent_registry()
    if not agent_registry:
        raise HTTPException(status_code=503, detail="Agent system disabled")

    # Check if agent exists
    existing = agent_registry.get_agent(agent_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent_registry.delete_agent(agent_id)

    return JSONResponse({"status": "ok", "message": "Agent deleted"})
