"""Web search tool using DuckDuckGo."""

import time
from typing import Any

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS  # Fallback for older installations

from forma.tools.base import SyncTool, ToolResult


class WebSearchTool(SyncTool):
    """Search the web using DuckDuckGo.

    Returns search results with titles, URLs, and snippets.
    Useful for finding current information, research, and facts.
    """

    name = "search_web"
    description = "Search the web for information. Returns relevant search results with titles, URLs, and content snippets. Use this to find current information, research topics, or verify facts."

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Be specific and use natural language.",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (default: 10, max: 50)",
                "default": 10,
            },
        },
        "required": ["query"],
    }

    timeout = 30.0  # Increased timeout for more results
    max_results = 50  # Increased from 10 to 50

    def execute_sync(self, **kwargs: Any) -> ToolResult:
        """Execute web search."""
        query = kwargs.get("query", "")
        num_results = min(kwargs.get("num_results", 10), self.max_results)

        if not query.strip():
            return ToolResult(
                success=False,
                error="Search query cannot be empty",
            )

        start_time = time.time()

        try:
            results = []
            with DDGS() as ddgs:
                # Use text search with backend set for reliability
                search_results = list(ddgs.text(query, max_results=num_results))

                for r in search_results:
                    results.append(
                        {
                            "title": r.get("title", ""),
                            "url": r.get("href", ""),
                            "snippet": r.get("body", ""),
                        }
                    )

            duration_ms = (time.time() - start_time) * 1000

            if not results:
                return ToolResult(
                    success=True,
                    output={"results": [], "message": "No results found"},
                    duration_ms=duration_ms,
                    metadata={"query": query, "num_requested": num_results},
                )

            return ToolResult(
                success=True,
                output={
                    "results": results,
                    "query": query,
                    "count": len(results),
                },
                duration_ms=duration_ms,
                metadata={"query": query, "num_requested": num_results},
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return ToolResult(
                success=False,
                error=f"Web search failed: {str(e)}",
                duration_ms=duration_ms,
            )
