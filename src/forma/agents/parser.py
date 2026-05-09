"""Agent Mention Parser.

Parses agent mentions from message content using patterns:
- @agent_name: Direct mention
- >>> agent_name: Explicit routing
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class RoutingType(Enum):
    """Type of agent routing."""

    MENTION = "mention"  # @agent_name
    EXPLICIT = "explicit"  # >>> agent_name:


@dataclass
class AgentMention:
    """Represents a parsed agent mention."""

    agent_name: str
    routing_type: RoutingType
    message_content: str
    start_pos: int
    end_pos: int


# Regex patterns for agent mentions
MENTION_PATTERN = re.compile(r"@([a-zA-Z0-9_-]+)(?:\s+(.*?)(?=@|$))?")
EXPLICIT_PATTERN = re.compile(r">>>\s*([a-zA-Z0-9_-]+)\s*:\s*(.*?)(?=>>>|@|$)", re.DOTALL)


def parse_agent_mentions(content: str) -> list[AgentMention]:
    """Parse all agent mentions from message content.

    Detects two patterns:
    1. @agent_name - Direct mention
    2. >>> agent_name: message - Explicit routing

    Args:
        content: The message content to parse

    Returns:
        List of AgentMention objects in order of appearance
    """
    mentions: list[AgentMention] = []

    # Parse explicit routing first (>>> agent_name: message)
    for match in EXPLICIT_PATTERN.finditer(content):
        agent_name = match.group(1).strip()
        message = match.group(2).strip()

        mentions.append(
            AgentMention(
                agent_name=agent_name,
                routing_type=RoutingType.EXPLICIT,
                message_content=message,
                start_pos=match.start(),
                end_pos=match.end(),
            )
        )

    # Parse direct mentions (@agent_name message)
    for match in MENTION_PATTERN.finditer(content):
        name = match.group(1).strip()

        # Check if this mention overlaps with an explicit routing
        overlaps = False
        for existing in mentions:
            if match.start() >= existing.start_pos and match.start() < existing.end_pos:
                overlaps = True
                break

        if overlaps:
            continue

        message = match.group(2).strip() if match.group(2) else ""

        mentions.append(
            AgentMention(
                agent_name=name,
                routing_type=RoutingType.MENTION,
                message_content=message,
                start_pos=match.start(),
                end_pos=match.end(),
            )
        )

    # Sort by position
    mentions.sort(key=lambda m: m.start_pos)

    return mentions


def extract_message_for_agent(
    content: str,
    mention: AgentMention,
) -> str:
    """Extract the message portion intended for a specific agent.

    For explicit routing (>>> agent_name: message), returns the message.
    For mentions (@agent_name message), returns the message after the mention.

    Args:
        content: Original message content
        mention: The agent mention

    Returns:
        Message content intended for the agent
    """
    if mention.routing_type == RoutingType.EXPLICIT:
        # Explicit routing already has the message extracted
        return mention.message_content

    if mention.routing_type == RoutingType.MENTION:
        if mention.message_content:
            return mention.message_content
        # If no message after mention, use full content
        return content

    return content


def remove_agent_mentions(content: str, mentions: list[AgentMention]) -> str:
    """Remove agent mention syntax from content, leaving just the message.

    Useful for cleaning up messages before routing to agents.

    Args:
        content: Original message content
        mentions: List of agent mentions to remove

    Returns:
        Cleaned content without mention syntax
    """
    if not mentions:
        return content

    # Sort mentions by position (descending) for removal
    sorted_mentions = sorted(mentions, key=lambda m: m.start_pos, reverse=True)

    result = content
    for mention in sorted_mentions:
        if mention.routing_type == RoutingType.EXPLICIT:
            # Remove ">>> agent_name: " prefix but keep message
            prefix_len = content.find(":", mention.start_pos) - mention.start_pos + 1
            result = result[: mention.start_pos] + result[mention.start_pos + prefix_len :].lstrip()
        elif mention.routing_type == RoutingType.MENTION:
            # Remove "@agent_name " prefix but keep message
            prefix_len = len(f"@{mention.agent_name}")
            if (
                mention.start_pos + prefix_len < len(result)
                and result[mention.start_pos + prefix_len] == " "
            ):
                prefix_len += 1
            result = result[: mention.start_pos] + result[mention.start_pos + prefix_len :]

    return result.strip()


def group_mentions_by_agent(mentions: list[AgentMention]) -> dict[str, list[AgentMention]]:
    """Group mentions by agent name.

    Args:
        mentions: List of agent mentions

    Returns:
        Dict mapping agent name to list of mentions
    """
    grouped: dict[str, list[AgentMention]] = {}

    for mention in mentions:
        name = mention.agent_name
        if name not in grouped:
            grouped[name] = []
        grouped[name].append(mention)

    return grouped


def has_agent_mentions(content: str) -> bool:
    """Check if content has any agent mentions.

    Args:
        content: Message content to check

    Returns:
        True if any agent mentions found
    """
    return len(parse_agent_mentions(content)) > 0
