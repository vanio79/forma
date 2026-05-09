"""Agent Configuration File Loader.

Loads agent configurations from JSON files on startup and syncs them
with the database.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from forma.agents.registry import AgentRegistry

logger = logging.getLogger(__name__)


def load_agents_from_config(
    config_path: str,
    registry: AgentRegistry,
    replace_existing: bool = False,
) -> dict[str, Any]:
    """Load agent configurations from a JSON file.

    Reads agent definitions from a JSON config file and registers them
    in the database. Can optionally replace existing agents.

    Args:
        config_path: Path to the agents.json config file
        registry: AgentRegistry instance for registration
        replace_existing: If True, delete existing agents before loading

    Returns:
        Dict with:
        - loaded: Number of agents loaded
        - created: Number of agents created
        - updated: Number of agents updated
        - skipped: Number of agents skipped (already existed)
        - errors: List of error messages
    """
    result = {
        "loaded": 0,
        "created": 0,
        "updated": 0,
        "skipped": 0,
        "errors": [],
    }

    # Check if config file exists
    config_file = Path(config_path)
    if not config_file.exists():
        logger.info(f"Agent config file not found: {config_path}")
        return result

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in agent config file: {e}"
        logger.error(error_msg)
        result["errors"].append(error_msg)
        return result
    except Exception as e:
        error_msg = f"Error reading agent config file: {e}"
        logger.error(error_msg)
        result["errors"].append(error_msg)
        return result

    # Get agents list from config
    agents = config.get("agents", [])
    if not agents:
        logger.info("No agents defined in config file")
        return result

    result["loaded"] = len(agents)

    # Optionally replace existing agents
    if replace_existing:
        existing = registry.get_all_agents()
        for agent in existing:
            try:
                registry.delete_agent(agent["id"])
                logger.info(f"Deleted existing agent: {agent['name']}")
            except Exception as e:
                error_msg = f"Failed to delete agent {agent['name']}: {e}"
                logger.error(error_msg)
                result["errors"].append(error_msg)

    # Load each agent
    for agent_config in agents:
        name = agent_config.get("name")
        if not name:
            error_msg = "Agent config missing 'name' field"
            logger.warning(error_msg)
            result["errors"].append(error_msg)
            continue

        try:
            # Check if agent already exists
            existing = registry.get_agent_by_name(name)

            if existing:
                # Update existing agent
                updates = {}

                if "purpose" in agent_config:
                    updates["purpose"] = agent_config["purpose"]
                if "instruction_prompt" in agent_config:
                    updates["instruction_prompt"] = agent_config["instruction_prompt"]

                # Handle upstream reference
                if "upstream" in agent_config:
                    # Note: upstream is a name/ID reference, need to resolve
                    # For now, store as-is; the main system should resolve
                    updates["upstream_id"] = agent_config["upstream"]

                if "tools_enabled" in agent_config:
                    updates["tools_enabled"] = agent_config["tools_enabled"]
                if "tool_whitelist" in agent_config:
                    updates["tool_whitelist"] = agent_config["tool_whitelist"]
                if "max_iterations" in agent_config:
                    updates["max_iterations"] = agent_config["max_iterations"]
                if "is_enabled" in agent_config:
                    updates["is_enabled"] = agent_config["is_enabled"]
                if "rag_config" in agent_config:
                    updates["rag_config"] = agent_config["rag_config"]

                if updates:
                    registry.update_agent(existing["id"], **updates)
                    result["updated"] += 1
                    logger.info(f"Updated agent: {name}")
                else:
                    result["skipped"] += 1
                    logger.debug(f"Skipped agent (no changes): {name}")
            else:
                # Create new agent
                registry.register_agent(
                    name=name,
                    purpose=agent_config.get("purpose", ""),
                    instruction_prompt=agent_config.get("instruction_prompt", ""),
                    upstream_id=agent_config.get("upstream"),
                    tools_enabled=agent_config.get("tools_enabled", True),
                    tool_whitelist=agent_config.get("tool_whitelist"),
                    max_iterations=agent_config.get("max_iterations", 5),
                    is_enabled=agent_config.get("is_enabled", True),
                    rag_config=agent_config.get("rag_config"),
                )
                result["created"] += 1
                logger.info(f"Created agent: {name}")

        except ValueError as e:
            error_msg = f"Failed to create/update agent {name}: {e}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error with agent {name}: {e}"
            logger.error(error_msg)
            result["errors"].append(error_msg)

    return result


def create_default_config(
    config_path: str,
    default_name: str = "assistant",
) -> bool:
    """Create a default agent configuration file.

    Creates a minimal agents.json with a default assistant agent.

    Args:
        config_path: Path where to create the config file
        default_name: Name for the default agent

    Returns:
        True if file created successfully
    """
    config_file = Path(config_path)

    # Create parent directories if needed
    config_file.parent.mkdir(parents=True, exist_ok=True)

    default_config = {
        "agents": [
            {
                "name": default_name,
                "purpose": "General assistant for all tasks",
                "instruction_prompt": "You are a helpful AI assistant. Provide clear, accurate responses. Use available tools when helpful.",
                "upstream": None,
                "tools_enabled": True,
                "tool_whitelist": [],
                "max_iterations": 5,
                "is_enabled": True,
            }
        ]
    }

    try:
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=2)
        logger.info(f"Created default agent config at: {config_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create default config: {e}")
        return False
