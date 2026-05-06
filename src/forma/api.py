"""Web UI API endpoints for Forma.

Provides REST endpoints for the SPA frontend to retrieve tracking data.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from forma.tracker import get_tracker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ui", tags=["ui"])


@router.get("/stats")
async def get_stats():
    """Get summary statistics for the dashboard."""
    tracker = get_tracker()
    stats = tracker.get_stats()
    return JSONResponse(stats)


@router.get("/requests")
async def get_requests(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Get list of recent requests."""
    tracker = get_tracker()
    requests = tracker.get_requests(limit=limit, offset=offset)

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
    tracker = get_tracker()
    detail = tracker.get_request_detail(request_id)

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
async def clear_tracking_data():
    """Clear all tracking data."""
    tracker = get_tracker()
    tracker.clear_all()
    return JSONResponse({"status": "ok", "message": "All tracking data cleared"})
