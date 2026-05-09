"""Multi-Agent system for Forma.

This module provides agent discovery, routing, and orchestration capabilities
for the Forma multi-agent architecture.
"""

from forma.agents.config_loader import create_default_config, load_agents_from_config
from forma.agents.discovery import (
    format_agent_discovery_context,
    format_agent_system_prompt,
    get_agent_tools_config,
)
from forma.agents.meta_evaluation import (
    EvaluationResult,
    create_isolated_context,
    create_retry_context,
    extract_summary,
    format_evaluator_input,
    format_summarizer_input,
    parse_evaluator_response,
)
from forma.agents.orchestrator import AgentOrchestrator
from forma.agents.parser import AgentMention, RoutingType, parse_agent_mentions
from forma.agents.registry import AgentRegistry
from forma.agents.router import AgentRouter

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
    # Meta-evaluation exports
    "EvaluationResult",
    "parse_evaluator_response",
    "create_isolated_context",
    "create_retry_context",
    "format_evaluator_input",
    "format_summarizer_input",
    "extract_summary",
]
