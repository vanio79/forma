"""Multi-Agent Orchestrator for coordinating multiple agent executions.

Handles sequential orchestration of multi-agent conversations.
Each agent executes one after another, with optional context passing.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Awaitable

from forma.agents.registry import AgentRegistry
from forma.agents.router import AgentRouter

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Orchestrates multi-agent conversations sequentially.

    Supports:
    - Sequential execution: agents run one after another
    - Chain execution: output of one agent feeds into next (context passing)
    """

    def __init__(
        self,
        registry: AgentRegistry,
        router: AgentRouter,
    ):
        """Initialize the orchestrator.

        Args:
            registry: AgentRegistry for agent lookups
            router: AgentRouter for message routing
        """
        self._registry = registry
        self._router = router

    async def execute_sequential(
        self,
        agents: list[dict[str, Any]],
        messages: list[str],
        executor: Callable[[dict[str, Any], str], Awaitable[str]],
    ) -> list[dict[str, Any]]:
        """Execute agents sequentially, one after another.

        Each agent receives its designated message and executes independently.

        Args:
            agents: List of agent configurations
            messages: Message for each agent (same length as agents)
            executor: Async function to execute for each agent
                     Signature: async def executor(agent: dict, message: str) -> str

        Returns:
            List of {agent, content, error} dicts in execution order
        """
        results: list[dict[str, Any]] = []

        for i, (agent, message) in enumerate(zip(agents, messages)):
            agent_name = agent.get("name", "unknown")
            logger.info(f"Executing agent {i + 1}/{len(agents)}: @{agent_name}")

            try:
                response = await executor(agent, message)
                results.append(
                    {
                        "agent": agent,
                        "content": response,
                        "error": None,
                    }
                )
                logger.info(f"Agent @{agent_name} completed successfully")
            except Exception as e:
                logger.error(f"Error executing agent @{agent_name}: {e}")
                results.append(
                    {
                        "agent": agent,
                        "content": None,
                        "error": str(e),
                    }
                )
                # Continue to next agent on error

        return results

    async def execute_chain(
        self,
        agents: list[dict[str, Any]],
        initial_message: str,
        executor: Callable[[dict[str, Any], str], Awaitable[str]],
        chain_format: str = "previous",
    ) -> list[dict[str, Any]]:
        """Execute agents in a chain, where each agent receives previous output.

        This enables agent-to-agent context passing where the output of one
        agent becomes the input for the next agent.

        Args:
            agents: List of agent configurations
            initial_message: Starting message for first agent
            executor: Async function to execute for each agent
            chain_format: How to format the chain message:
                         "previous" - just previous agent's output
                         "context" - previous output with context marker

        Returns:
            List of {agent, content, error} dicts in chain order
        """
        results: list[dict[str, Any]] = []
        current_message = initial_message

        for i, agent in enumerate(agents):
            agent_name = agent.get("name", "unknown")
            logger.info(f"Chain execution step {i + 1}/{len(agents)}: @{agent_name}")

            try:
                # Prepare message based on chain format
                if i == 0:
                    message = current_message
                elif chain_format == "previous":
                    message = current_message
                elif chain_format == "context":
                    message = f"[Context from previous agent]\n{current_message}"
                else:
                    message = current_message

                response = await executor(agent, message)

                results.append(
                    {
                        "agent": agent,
                        "content": response,
                        "error": None,
                    }
                )

                logger.info(f"Chain step @{agent_name} completed")

                # Update current message for next agent
                current_message = response

            except Exception as e:
                logger.error(f"Error in chain execution for @{agent_name}: {e}")
                results.append(
                    {
                        "agent": agent,
                        "content": None,
                        "error": str(e),
                    }
                )
                # Stop chain on error
                break

        return results

    async def orchestrate(
        self,
        content: str,
        executor: Callable[[dict[str, Any], str], Awaitable[str]],
        strategy: str = "sequential",
    ) -> dict[str, Any]:
        """Orchestrate multi-agent execution based on message content.

        Executes agents sequentially. Chain execution is used when explicit
        routing syntax (>>> agent1: >>> agent2:) indicates context passing.

        Args:
            content: Message content to analyze
            executor: Async function to execute for each agent
            strategy: Orchestration strategy: "sequential" or "chain"

        Returns:
            Dict with:
            - strategy: Strategy used
            - results: List of {agent, content, error} dicts
            - formatted_response: Combined response string
        """
        routing_info = self._router.extract_routing_info(content)
        mentions = routing_info["mentions"]
        agents_dict = routing_info["agents"]

        if not mentions:
            return {
                "strategy": "none",
                "results": [],
                "formatted_response": "",
            }

        # Get target agents
        target_agents = list(agents_dict.values())

        if not target_agents:
            return {
                "strategy": "none",
                "results": [],
                "formatted_response": "",
            }

        # Prepare messages for each agent
        messages = []
        from forma.agents.parser import group_mentions_by_agent, extract_message_for_agent

        grouped = group_mentions_by_agent(mentions)

        for agent in target_agents:
            agent_name = agent.get("name")
            agent_mentions = grouped.get(agent_name, [])

            if agent_mentions:
                # Use first mention's message
                message = extract_message_for_agent(content, agent_mentions[0])
            else:
                # Broadcast or fallback - use full content
                message = content

            messages.append(message)

        # Execute sequentially
        results = await self.execute_sequential(target_agents, messages, executor)

        # Format response
        formatted = self._router.format_multi_agent_response(
            [{"agent": r["agent"], "content": r["content"]} for r in results if r.get("content")]
        )

        return {
            "strategy": strategy,
            "results": results,
            "formatted_response": formatted,
        }
