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
