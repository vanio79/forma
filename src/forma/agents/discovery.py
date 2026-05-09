"""Agent Discovery Context Formatting.

Formats agent discovery information for prompt augmentation.
This context is added to messages so agents can discover and communicate
with other agents.
"""

from __future__ import annotations

from typing import Any


def format_agent_discovery_context(
    agents: list[dict[str, Any]],
    include_routing_help: bool = True,
) -> str:
    """Format agent discovery information for prompt augmentation.

    Creates a formatted context block that lists all available agents
    and explains how to route messages to them.

    Args:
        agents: List of agent configuration dicts (from get_enabled_agents())
        include_routing_help: Whether to include routing syntax help text

    Returns:
        Formatted discovery context string
    """
    if not agents:
        return ""

    lines = ["Available agents you can communicate with:"]
    lines.append("")

    for agent in agents:
        name = agent.get("name", "unknown")
        purpose = agent.get("purpose", "No description available")
        lines.append(f"- @{name}: {purpose}")

    if include_routing_help:
        lines.append("")
        lines.append("To send a message to another agent:")
        lines.append("- Mention the agent by name: '@researcher please search for...'")
        lines.append("- Use explicit routing: '>>> researcher: search for...'")

    lines.append("")

    return "\n".join(lines)


def format_agent_system_prompt(
    agent: dict[str, Any],
    discovery_context: str | None = None,
) -> str:
    """Format the complete system prompt for an agent.

    Combines agent discovery context (if available) with the agent's
    instruction prompt.

    Args:
        agent: Agent configuration dict
        discovery_context: Pre-formatted discovery context (optional)

    Returns:
        Complete system prompt string
    """
    parts = []

    # Add discovery context first (other agents)
    if discovery_context:
        parts.append(discovery_context)

    # Add agent's own instruction prompt
    instruction = agent.get("instruction_prompt", "")
    if instruction:
        parts.append(instruction)

    return "\n\n".join(parts)


def get_agent_tools_config(
    agent: dict[str, Any],
    all_tools: list[str] | None = None,
) -> dict[str, Any]:
    """Get tool configuration for an agent.

    Returns a dict with tools_enabled, allowed_tools, and max_iterations
    suitable for use in request processing.

    Args:
        agent: Agent configuration dict
        all_tools: List of all available tool names (optional)

    Returns:
        Dict with tools_enabled, allowed_tools, max_iterations
    """
    import json

    tools_enabled = agent.get("tools_enabled", False)

    # Parse tool whitelist
    tool_whitelist_json = agent.get("tool_whitelist", "[]")
    try:
        tool_whitelist = (
            json.loads(tool_whitelist_json)
            if isinstance(tool_whitelist_json, str)
            else tool_whitelist_json
        )
    except (json.JSONDecodeError, TypeError):
        tool_whitelist = []

    # Determine allowed tools
    if not tools_enabled:
        allowed_tools = []
    elif tool_whitelist:
        # Use whitelist if specified
        allowed_tools = tool_whitelist
    else:
        # All tools if whitelist is empty
        allowed_tools = all_tools or []

    max_iterations = agent.get("max_iterations", 5)

    return {
        "tools_enabled": tools_enabled,
        "allowed_tools": allowed_tools,
        "max_iterations": max_iterations,
    }
