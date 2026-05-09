"""Agent Router for routing messages between agents.

Handles message routing, agent lookup, and response tagging.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from forma.agents.registry import AgentRegistry
from forma.agents.parser import (
    AgentMention,
    RoutingType,
    parse_agent_mentions,
    extract_message_for_agent,
    group_mentions_by_agent,
)

logger = logging.getLogger(__name__)


class AgentRouter:
    """Routes messages between agents in the multi-agent system.

    Responsibilities:
    - Parse agent mentions from messages
    - Look up agent configurations
    - Route messages to specific agents
    - Tag responses with agent names
    """

    def __init__(
        self,
        registry: AgentRegistry,
    ):
        """Initialize the agent router.

        Args:
            registry: AgentRegistry for agent lookups
        """
        self._registry = registry

    def parse_mentions(self, content: str) -> list[AgentMention]:
        """Parse agent mentions from message content.

        Args:
            content: Message content to parse

        Returns:
            List of AgentMention objects
        """
        return parse_agent_mentions(content)

    def resolve_agent(self, agent_name: str) -> dict[str, Any] | None:
        """Look up agent by name.

        Args:
            agent_name: Agent name to look up

        Returns:
            Agent configuration dict or None if not found
        """
        return self._registry.get_agent_by_name(agent_name)

    def resolve_agents_for_mentions(
        self,
        mentions: list[AgentMention],
    ) -> dict[str, dict[str, Any]]:
        """Resolve all mentioned agents to their configurations.

        Args:
            mentions: List of agent mentions

        Returns:
            Dict mapping agent name to agent configuration
        """
        resolved: dict[str, dict[str, Any]] = {}

        for mention in mentions:
            name = mention.agent_name

            # Only resolve each agent once
            if name in resolved:
                continue

            agent = self.resolve_agent(name)
            if agent:
                resolved[name] = agent
            else:
                logger.warning(f"Agent '{name}' mentioned but not found in registry")

        return resolved

    def has_routing(self, content: str) -> bool:
        """Check if content contains agent routing.

        Args:
            content: Message content to check

        Returns:
            True if content has agent mentions
        """
        return len(self.parse_mentions(content)) > 0

    def extract_routing_info(
        self,
        content: str,
    ) -> dict[str, Any]:
        """Extract complete routing information from content.

        Args:
            content: Message content to analyze

        Returns:
            Dict with:
            - mentions: List of AgentMention objects
            - agents: Dict mapping agent name to agent config
            - is_multi_agent: Whether multiple agents were mentioned
        """
        mentions = self.parse_mentions(content)
        agents = self.resolve_agents_for_mentions(mentions)

        return {
            "mentions": mentions,
            "agents": agents,
            "is_multi_agent": len(agents) > 1,
        }

    def prepare_message_for_agent(
        self,
        agent: dict[str, Any],
        mention: AgentMention | None,
        original_content: str,
    ) -> str:
        """Prepare the message content for routing to an agent.

        Args:
            agent: Target agent configuration
            mention: The mention that triggered this routing (may be None)
            original_content: The original user message

        Returns:
            Message content to send to the agent
        """
        if mention:
            return extract_message_for_agent(original_content, mention)
        return original_content

    def tag_response(
        self,
        agent: dict[str, Any],
        response_content: str,
        mention: AgentMention | None = None,
    ) -> str:
        """Tag an agent's response with its name.

        Args:
            agent: Agent configuration
            response_content: The response content
            mention: The mention that triggered this routing (optional)

        Returns:
            Tagged response string
        """
        agent_name = agent.get("name", "unknown")

        # Tag the response with agent name
        tagged = f"[{agent_name}] {response_content}"

        return tagged

    def format_multi_agent_response(
        self,
        responses: list[dict[str, Any]],
    ) -> str:
        """Format multiple agent responses into a single message.

        Args:
            responses: List of dicts with 'agent' and 'content' keys

        Returns:
            Formatted multi-agent response string
        """
        if not responses:
            return ""

        if len(responses) == 1:
            agent = responses[0].get("agent", {})
            content = responses[0].get("content", "")
            return self.tag_response(agent, content)

        # Multiple responses - format with separators
        parts = []
        for resp in responses:
            agent = resp.get("agent", {})
            content = resp.get("content", "")
            agent_name = agent.get("name", "unknown")
            parts.append(f"--- [@{agent_name}]\n{content}")

        return "\n\n".join(parts)

    def get_routing_targets(
        self,
        content: str,
    ) -> list[dict[str, Any]]:
        """Get list of agents that should receive this message.

        Determines the routing targets based on mentions:
        - Specific agents mentioned: route to those agents
        - @all: route to all enabled agents
        - No mentions: return empty list

        Args:
            content: Message content to analyze

        Returns:
            List of agent configurations that should receive the message
        """
        mentions = self.parse_mentions(content)

        if not mentions:
            return []

        # Route to specific mentioned agents
        agents = self.resolve_agents_for_mentions(mentions)
        return list(agents.values())

    async def route_to_agents(
        self,
        content: str,
        router_func: Any,  # Callable that takes (agent, message) and returns response
    ) -> dict[str, Any]:
        """Route message to appropriate agents and return combined response.

        Args:
            content: Original message content
            router_func: Async function that routes to a single agent
                        Signature: async def router(agent: dict, message: str) -> str

        Returns:
            Dict with:
            - mentions: List of mentions found
            - responses: List of {agent, content} dicts
            - formatted_response: Combined response string
            - is_multi_agent: Whether multiple agents responded
        """
        routing_info = self.extract_routing_info(content)
        mentions = routing_info["mentions"]
        agents = routing_info["agents"]

        if not mentions:
            return {
                "mentions": [],
                "responses": [],
                "formatted_response": "",
                "is_multi_agent": False,
            }

        responses: list[dict[str, Any]] = []

        # Route to specific agents
        # Group mentions by agent
        grouped = group_mentions_by_agent(mentions)

        tasks = []
        task_agents = []

        for agent_name, agent_mentions in grouped.items():
            agent = agents.get(agent_name)
            if not agent:
                continue

            # Use first mention's message for this agent
            mention = agent_mentions[0]
            message = self.prepare_message_for_agent(agent, mention, content)
            tasks.append(router_func(agent, message))
            task_agents.append(agent)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for agent, result in zip(task_agents, results):
            if isinstance(result, Exception):
                logger.error(f"Error routing to agent {agent.get('name')}: {result}")
                continue

            responses.append(
                {
                    "agent": agent,
                    "content": str(result),
                }
            )

        formatted = self.format_multi_agent_response(responses)

        return {
            "mentions": mentions,
            "responses": responses,
            "formatted_response": formatted,
            "is_multi_agent": len(responses) > 1,
        }
