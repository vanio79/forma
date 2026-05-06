"""Upstream manager for model-based routing.

Manages multiple upstream API configurations and routes requests
based on the model name in the request.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from forma.forma_db import FormaDatabase

logger = logging.getLogger(__name__)


@dataclass
class UpstreamConfig:
    """Configuration for an upstream API."""

    id: str
    name: str  # Local model name used for routing
    upstream_model: str  # Model name to send to upstream API
    base_url: str
    api_key: str = ""
    timeout: float = 300.0
    is_enabled: bool = True


class UpstreamManager:
    """Manages upstream configurations for model-based routing."""

    def __init__(self, db: Optional[FormaDatabase] = None) -> None:
        """Initialize the upstream manager.

        Args:
            db: FormaDatabase instance for loading upstreams
        """
        self._db = db
        self._upstreams: dict[str, UpstreamConfig] = {}  # name (model) -> upstream
        self._upstreams_by_id: dict[str, UpstreamConfig] = {}

        # Load upstreams
        self.reload()

    def reload(self) -> None:
        """Reload upstream configurations from database."""
        self._upstreams = {}
        self._upstreams_by_id = {}

        # Load from database
        if self._db:
            try:
                upstreams_data = self._db.get_upstreams()
                for upstream_dict in upstreams_data:
                    upstream = UpstreamConfig(
                        id=upstream_dict["id"],
                        name=upstream_dict["name"],
                        upstream_model=upstream_dict.get("upstream_model", upstream_dict["name"]),
                        base_url=upstream_dict["base_url"].rstrip("/"),
                        api_key=upstream_dict["api_key"],
                        timeout=upstream_dict["timeout"],
                        is_enabled=bool(upstream_dict["is_enabled"]),
                    )

                    self._upstreams_by_id[upstream.id] = upstream

                    if upstream.is_enabled:
                        # Map name (model) to upstream
                        self._upstreams[upstream.name] = upstream
                        logger.debug(
                            f"Mapped model '{upstream.name}' -> '{upstream.upstream_model}' to upstream '{upstream.base_url}'"
                        )

                logger.info(f"Loaded {len(self._upstreams)} enabled upstream configurations")
            except Exception as e:
                logger.warning(f"Failed to load upstreams from database: {e}")

    def get_upstream_for_model(self, model: str) -> Optional[UpstreamConfig]:
        """Get the upstream configuration for a specific model.

        Args:
            model: The model name from the request

        Returns:
            UpstreamConfig if found, None otherwise
        """
        return self._upstreams.get(model)

    def get_all_upstreams(self) -> list[UpstreamConfig]:
        """Get all upstream configurations."""
        return list(self._upstreams_by_id.values())

    def get_upstream_by_id(self, upstream_id: str) -> Optional[UpstreamConfig]:
        """Get an upstream by its ID."""
        return self._upstreams_by_id.get(upstream_id)

    def has_upstream(self, model: str) -> bool:
        """Check if there's an upstream for a model."""
        return model in self._upstreams
