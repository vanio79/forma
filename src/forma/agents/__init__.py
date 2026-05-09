"""Multi-Agent system for Forma.

This module provides agent discovery, routing, and orchestration capabilities
for the Forma multi-agent architecture.
"""

from forma.agents.registry import AgentRegistry
from forma.agents.discovery import (
    format_agent_discovery_context,
    format_agent_system_prompt,
    get_agent_tools_config,
)
from forma.agents.parser import parse_agent_mentions, AgentMention, RoutingType
from forma.agents.router import AgentRouter
from forma.agents.orchestrator import AgentOrchestrator
from forma.agents.config_loader import load_agents_from_config, create_default_config

__all__ = [
    "AgentRegistry",
    "format_agent_discovery_context",
    "format_agent_system_prompt",
    "get_agent_tools_config",
    "parse_agent_mentions",
    "AgentMention",
    "RoutingType",
    "AgentRouter",
    "AgentOrchestrator",
    "load_agents_from_config",
    "create_default_config",
]
