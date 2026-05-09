"""Agent Registry for managing agent configurations.

The AgentRegistry provides CRUD operations for agent configurations stored
in the FormaDatabase. It also handles default agent creation and validation.
"""

from __future__ import annotations

import uuid
from datetime import datetime, UTC
from typing import Any

from forma.forma_db import FormaDatabase


class AgentRegistry:
    """Registry for managing agent configurations.

    Provides CRUD operations for agents and handles default agent creation.
    All agents are stored in the FormaDatabase agents table.
    """

    def __init__(self, db: FormaDatabase):
        """Initialize the agent registry.

        Args:
            db: FormaDatabase instance for persistence
        """
        self._db = db

    def register_agent(
        self,
        name: str,
        purpose: str,
        instruction_prompt: str,
        upstream_id: str | None = None,
        tools_enabled: bool = True,
        tool_whitelist: list[str] | None = None,
        max_iterations: int = 5,
        is_enabled: bool = True,
        rag_config: dict | None = None,
    ) -> str:
        """Register a new agent configuration.

        Args:
            name: Unique human-readable agent name (e.g., "researcher")
            purpose: Brief description of agent's role
            instruction_prompt: System prompt / instruction for this agent
            upstream_id: Reference to upstream config (None = use default upstream)
            tools_enabled: Whether tools are enabled for this agent
            tool_whitelist: Allowed tools (empty list or None = all tools)
            max_iterations: Max tool iterations for this agent
            is_enabled: Whether agent is active
            rag_config: RAG configuration dict (enabled, token_budget, min_confidence, max_distance)

        Returns:
            The agent ID (UUID string)

        Raises:
            ValueError: If agent name already exists
        """
        # Check if name already exists
        existing = self._db.get_agent_by_name(name)
        if existing:
            raise ValueError(f"Agent with name '{name}' already exists")

        # Generate UUID
        agent_id = str(uuid.uuid4())

        # Create agent in database
        self._db.create_agent(
            id=agent_id,
            name=name,
            purpose=purpose,
            instruction_prompt=instruction_prompt,
            upstream_id=upstream_id,
            tools_enabled=tools_enabled,
            tool_whitelist=tool_whitelist or [],
            max_iterations=max_iterations,
            is_enabled=is_enabled,
            rag_config=rag_config or {},
        )

        return agent_id

    def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        """Get agent by ID.

        Args:
            agent_id: The agent UUID

        Returns:
            Agent configuration dict or None if not found
        """
        return self._db.get_agent_by_id(agent_id)

    def get_agent_by_name(self, name: str) -> dict[str, Any] | None:
        """Get agent by name.

        Args:
            name: The agent name

        Returns:
            Agent configuration dict or None if not found
        """
        return self._db.get_agent_by_name(name)

    def get_all_agents(self) -> list[dict[str, Any]]:
        """Get all registered agents.

        Returns:
            List of all agent configuration dicts
        """
        return self._db.get_agents()

    def get_enabled_agents(self) -> list[dict[str, Any]]:
        """Get all enabled agents (for discovery).

        Returns:
            List of enabled agent configuration dicts
        """
        return self._db.get_enabled_agents()

    def update_agent(self, agent_id: str, **updates) -> bool:
        """Update agent configuration.

        Args:
            agent_id: The agent UUID
            **updates: Fields to update (name, purpose, instruction_prompt, rag_config, etc.)

        Returns:
            True if update successful, False if agent not found
        """
        # Check if agent exists
        existing = self._db.get_agent_by_id(agent_id)
        if not existing:
            return False

        # If updating name, check it doesn't conflict
        if "name" in updates:
            name_conflict = self._db.get_agent_by_name(updates["name"])
            if name_conflict and name_conflict["id"] != agent_id:
                raise ValueError(f"Agent with name '{updates['name']}' already exists")

        # Handle tool_whitelist JSON serialization
        if "tool_whitelist" in updates and isinstance(updates["tool_whitelist"], list):
            import json

            updates["tool_whitelist"] = json.dumps(updates["tool_whitelist"])

        # Handle rag_config JSON serialization
        if "rag_config" in updates and isinstance(updates["rag_config"], dict):
            import json

            updates["rag_config"] = json.dumps(updates["rag_config"])

        # Perform update - need to provide all required fields
        # Get existing values to merge with updates
        merged_data = {
            "name": updates.get("name", existing["name"]),
            "purpose": updates.get("purpose", existing["purpose"]),
            "instruction_prompt": updates.get("instruction_prompt", existing["instruction_prompt"]),
            "upstream_id": updates.get("upstream_id", existing["upstream_id"]),
            "tools_enabled": updates.get("tools_enabled", existing["tools_enabled"]),
            "tool_whitelist": json.loads(updates.get("tool_whitelist", existing["tool_whitelist"]))
            if isinstance(updates.get("tool_whitelist", existing["tool_whitelist"]), str)
            else updates.get("tool_whitelist", existing["tool_whitelist"]),
            "max_iterations": updates.get("max_iterations", existing["max_iterations"]),
            "is_enabled": updates.get("is_enabled", existing["is_enabled"]),
            "rag_config": json.loads(
                updates.get("rag_config", json.dumps(existing.get("rag_config", {})))
            )
            if isinstance(
                updates.get("rag_config", json.dumps(existing.get("rag_config", {}))), str
            )
            else updates.get("rag_config", existing.get("rag_config", {})),
        }

        self._db.update_agent(agent_id, **merged_data)
        return True

    def delete_agent(self, agent_id: str) -> bool:
        """Delete agent configuration.

        Args:
            agent_id: The agent UUID

        Returns:
            True if deletion successful, False if agent not found
        """
        # Check if agent exists
        existing = self._db.get_agent_by_id(agent_id)
        if not existing:
            return False

        self._db.delete_agent(agent_id)
        return True

    def create_default_agent_if_needed(
        self,
        default_name: str = "assistant",
        default_purpose: str = "General assistant for all tasks",
        default_instruction: str = "You are a helpful AI assistant. Provide clear, accurate responses. Use available tools when helpful.",
    ) -> str | None:
        """Create default agent if no agents exist.

        Args:
            default_name: Name for the default agent
            default_purpose: Purpose description for default agent
            default_instruction: Instruction prompt for default agent

        Returns:
            Agent ID if created, None if agents already exist
        """
        # Check if any agents exist
        existing_agents = self.get_all_agents()
        if existing_agents:
            return None

        # Create default agent
        agent_id = self.register_agent(
            name=default_name,
            purpose=default_purpose,
            instruction_prompt=default_instruction,
            upstream_id=None,  # Use default upstream
            tools_enabled=True,
            tool_whitelist=None,  # All tools
            max_iterations=5,
            is_enabled=True,
        )

        return agent_id
