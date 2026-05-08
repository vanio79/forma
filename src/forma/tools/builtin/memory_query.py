"""Query Forma's memory (GrafitoDB) directly."""

import time
from typing import Any

from forma.tools.base import Tool, ToolResult


class QueryMemoryTool(Tool):
    """Query Forma's memory storage directly.

    Allows direct access to GrafitoDB for retrieving relationships,
    facts, and recipes. Useful when RAG retrieval doesn't find
    specific information or for targeted queries.
    """

    name = "query_memory"
    description = "Query Forma's memory storage directly. Search for relationships (entity connections), facts, or procedural knowledge (recipes). Use this when you need specific information from past conversations that may not be retrieved through normal context augmentation."

    parameters = {
        "type": "object",
        "properties": {
            "query_type": {
                "type": "string",
                "description": "Type of data to query: 'relationships', 'facts', 'recipes', or 'all'",
                "enum": ["relationships", "facts", "recipes", "all"],
                "default": "all",
            },
            "query": {
                "type": "string",
                "description": "Search query. For relationships, use entity names. For facts/recipes, use natural language.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 10, max: 20)",
                "default": 10,
            },
        },
        "required": ["query"],
    }

    timeout = 10.0
    max_limit = 20

    # Storage reference will be set by registry
    _storage: Any = None

    def set_storage(self, storage: Any) -> None:
        """Set the storage reference for querying GrafitoDB."""
        self._storage = storage

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute memory query."""
        query = kwargs.get("query", "")
        query_type = kwargs.get("query_type", "all")
        limit = min(kwargs.get("limit", 10), self.max_limit)

        if not query.strip():
            return ToolResult(
                success=False,
                error="Query cannot be empty",
            )

        if self._storage is None:
            return ToolResult(
                success=False,
                error="Memory storage not available. Ensure Forma is running with storage initialized.",
            )

        start_time = time.time()

        try:
            results: dict[str, Any] = {
                "query": query,
                "query_type": query_type,
                "items": [],
            }

            # Query relationships (graph traversal)
            if query_type in ("relationships", "all"):
                try:
                    # Extract entity names from query for relationship matching
                    entity_names = self._extract_entity_names(query)
                    if entity_names:
                        for entity_name in entity_names:
                            rels = self._storage.query_relationships(
                                subject=entity_name,
                                n_results=limit,
                            )
                            for r in rels:
                                results["items"].append(
                                    {
                                        "type": "relationship",
                                        "data": {
                                            "subject": r.get("subject", ""),
                                            "predicate": r.get("predicate", ""),
                                            "object": r.get("object", ""),
                                            "confidence": r.get("confidence", 0.9),
                                        },
                                    }
                                )
                except Exception as e:
                    results["relationship_error"] = str(e)

            # Query facts (semantic similarity)
            if query_type in ("facts", "all"):
                try:
                    facts = self._storage.query_facts(query, n_results=limit)
                    for f in facts:
                        results["items"].append(
                            {
                                "type": "fact",
                                "data": {
                                    "content": f.get("document", ""),
                                    "confidence": f.get("metadata", {}).get("confidence", 0.9),
                                    "score": 1.0 - f.get("distance", 1.0),
                                },
                            }
                        )
                except Exception as e:
                    results["fact_error"] = str(e)

            # Query recipes (semantic similarity)
            if query_type in ("recipes", "all"):
                try:
                    recipes = self._storage.query_recipes(query, n_results=limit)
                    for r in recipes:
                        results["items"].append(
                            {
                                "type": "recipe",
                                "data": {
                                    "content": r.get("document", ""),
                                    "confidence": r.get("metadata", {}).get("confidence", 0.9),
                                    "score": 1.0 - r.get("distance", 1.0),
                                },
                            }
                        )
                except Exception as e:
                    results["recipe_error"] = str(e)

            duration_ms = (time.time() - start_time) * 1000
            results["count"] = len(results["items"])
            results["duration_ms"] = duration_ms

            return ToolResult(
                success=True,
                output=results,
                duration_ms=duration_ms,
                metadata={"query_type": query_type, "limit": limit},
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return ToolResult(
                success=False,
                error=f"Memory query failed: {str(e)}",
                duration_ms=duration_ms,
            )

    def _extract_entity_names(self, query: str) -> list[str]:
        """Extract potential entity names from a query.

        Simple heuristic: split by common words and extract capitalized terms,
        quoted strings, and specific patterns.
        """
        import re

        entities = []

        # Find quoted strings
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)

        # Find capitalized words (potential entity names)
        capitalized = re.findall(r"\b[A-Z][a-zA-Z]+\b", query)
        entities.extend(capitalized)

        # Find "The user" specifically
        if "the user" in query.lower() or "user" in query.lower():
            entities.append("The user")

        # Deduplicate and clean
        entities = list(set(e.strip() for e in entities if e.strip()))

        return entities[:5]  # Limit to avoid overloading
