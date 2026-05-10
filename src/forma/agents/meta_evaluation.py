"""Meta-Agent Evaluation System.

Handles evaluation of subagent task completion and summarization of results.
This ensures quality control and prevents context pollution.

Also handles automatic context compaction for agent-to-agent conversations
to prevent context overflow during multi-agent interactions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result from evaluator agent assessment."""

    status: str  # "complete", "incomplete", "failed"
    reason: str
    retry_instructions: str | None = None
    summary_focus: str | None = None
    confidence: float = 0.0
    is_valid: bool = True  # Whether JSON was parsed correctly


def parse_evaluator_response(response_content: str) -> EvaluationResult:
    """Parse evaluator agent's JSON response.

    Args:
        response_content: Raw text from evaluator agent

    Returns:
        EvaluationResult with parsed fields
    """
    # Try to extract JSON from response
    # Evaluator might wrap JSON in markdown or add extra text
    json_str = response_content.strip()

    # Try direct parse
    try:
        data = json.loads(json_str)
        return EvaluationResult(
            status=data.get("status", "failed"),
            reason=data.get("reason", "No reason provided"),
            retry_instructions=data.get("retry_instructions"),
            summary_focus=data.get("summary_focus"),
            confidence=data.get("confidence", 0.5),
            is_valid=True,
        )
    except json.JSONDecodeError:
        pass

    # Try to find JSON block in markdown
    if "```json" in json_str:
        start = json_str.find("```json") + 7
        end = json_str.find("```", start)
        if end > start:
            json_block = json_str[start:end].strip()
            try:
                data = json.loads(json_block)
                return EvaluationResult(
                    status=data.get("status", "failed"),
                    reason=data.get("reason", "No reason provided"),
                    retry_instructions=data.get("retry_instructions"),
                    summary_focus=data.get("summary_focus"),
                    confidence=data.get("confidence", 0.5),
                    is_valid=True,
                )
            except json.JSONDecodeError:
                pass

    # Try to find JSON object anywhere in text
    start = json_str.find("{")
    end = json_str.rfind("}") + 1
    if start >= 0 and end > start:
        json_block = json_str[start:end]
        try:
            data = json.loads(json_block)
            return EvaluationResult(
                status=data.get("status", "failed"),
                reason=data.get("reason", "No reason provided"),
                retry_instructions=data.get("retry_instructions"),
                summary_focus=data.get("summary_focus"),
                confidence=data.get("confidence", 0.5),
                is_valid=True,
            )
        except json.JSONDecodeError:
            pass

    # Fallback: couldn't parse JSON
    logger.warning(f"Failed to parse evaluator response as JSON: {response_content[:200]}")
    return EvaluationResult(
        status="failed",
        reason="Evaluator response was not valid JSON",
        retry_instructions=None,
        summary_focus="Include what was attempted",
        confidence=0.0,
        is_valid=False,
    )


def create_isolated_context(
    original_task: str,
    calling_agent_name: str,
    delegation_message: str,
    max_context_tokens: int = 500,
) -> list[dict[str, str]]:
    """Create isolated context for subagent.

    Subagents should work with minimal context to avoid pollution.
    Only essential information is passed.

    Args:
        original_task: The user's original request
        calling_agent_name: Name of the delegating agent
        delegation_message: The delegation instruction from calling agent
        max_context_tokens: Maximum context to pass (approximate)

    Returns:
        Minimal message array for subagent
    """
    # Create minimal context
    # System prompt will be added by _execute_agent_request
    isolated_messages = [
        {
            "role": "user",
            "content": f"Task delegated by @{calling_agent_name}:\n\n"
            f"Original request: {original_task[:max_context_tokens]}\n\n"
            f"Delegation instruction: {delegation_message}",
        }
    ]

    logger.info(
        f"Created isolated context for subagent: "
        f"calling_agent={calling_agent_name}, "
        f"task_preview={original_task[:50]}..."
    )

    return isolated_messages


def create_retry_context(
    original_task: str,
    previous_response: str,
    evaluator_instructions: str,
    attempt_number: int,
) -> list[dict[str, str]]:
    """Create context for retry attempt with evaluator guidance.

    Args:
        original_task: The original task description
        previous_response: What the subagent produced last time
        evaluator_instructions: Specific guidance from evaluator
        attempt_number: Which retry attempt (for context)

    Returns:
        Message array for retry
    """
    retry_messages = [
        {
            "role": "user",
            "content": f"Retry attempt #{attempt_number}\n\n"
            f"Original task: {original_task}\n\n"
            f"Your previous response:\n{previous_response}\n\n"
            f"Evaluator feedback:\n{evaluator_instructions}\n\n"
            f"Please continue with improved approach based on this feedback.",
        }
    ]

    logger.info(f"Created retry context (attempt #{attempt_number})")

    return retry_messages


def format_evaluator_input(
    task_description: str,
    subagent_name: str,
    subagent_response: str,
    tool_calls_summary: str | None = None,
) -> str:
    """Format input for evaluator agent.

    Args:
        task_description: What the subagent was asked to do
        subagent_name: Name of the subagent
        subagent_response: The subagent's output
        tool_calls_summary: Summary of tools used (optional)

    Returns:
        Formatted prompt for evaluator
    """
    parts = [
        "Evaluate this subagent task completion:\n\n",
        f"Task: {task_description}\n\n",
        f"Subagent: @{subagent_name}\n\n",
    ]

    if tool_calls_summary:
        parts.append(f"Tools used: {tool_calls_summary}\n\n")

    parts.extend(
        [
            f"Response:\n{subagent_response}\n\n",
            "Provide your assessment in JSON format with fields: "
            "status, reason, retry_instructions (if incomplete), "
            "summary_focus (if complete/failed), confidence.",
        ]
    )

    return "".join(parts)


def format_summarizer_input(
    task_description: str,
    subagent_name: str,
    full_context: str,
    evaluator_assessment: EvaluationResult,
) -> str:
    """Format input for summarizer agent.

    Args:
        task_description: What the task was
        subagent_name: Name of subagent
        full_context: Complete context from subagent execution
        evaluator_assessment: The evaluator's verdict

    Returns:
        Formatted prompt for summarizer
    """
    return (
        f"Summarize this subagent's work. Preserve ALL information relevant to the original task.\n\n"
        f"Original task: {task_description}\n\n"
        f"Subagent: @{subagent_name}\n\n"
        f"Evaluator assessment: {evaluator_assessment.status} "
        f"(reason: {evaluator_assessment.reason})\n\n"
        f"Full context:\n{full_context}\n\n"
        f"Instructions:\n"
        f"1. Start by restating the original task so the calling agent knows what was requested.\n"
        f"2. Preserve ALL facts, data, findings, and conclusions relevant to the original task.\n"
        f"3. Keep specific details: names, dates, numbers, URLs, results. Do NOT omit them.\n"
        f"4. Use bullet lists and short statements. Do NOT write flowing paragraphs.\n"
        f"5. If the task was incomplete, say what was accomplished and what remains.\n"
        f"6. Prioritize completeness over brevity.\n\n"
        f"Return your summary using these sections (skip any section that has no content):\n\n"
        f"### Original Task\n"
        f"[Write the original task here]\n\n"
        f"### What Was Done\n"
        f"- [List each step the subagent took]\n"
        f"- [...]\n\n"
        f"### Key Findings / Results\n"
        f"- [List each finding with all specific details preserved]\n"
        f"- [...]\n\n"
        f"### Data / Evidence\n"
        f"- [List specific data: numbers, dates, names, URLs, quotes]\n"
        f"- [...]\n\n"
        f"### Conclusions / Answers\n"
        f"- [Direct answer to the original task]\n"
        f"- [...]\n\n"
        f"### Incomplete Items\n"
        f"- [Only include this section if something remains undone]\n"
        f"- [...]"
    )


def extract_summary(response_content: str) -> str:
    """Extract summary from summarizer response.

    Args:
        response_content: Raw response from summarizer

    Returns:
        Clean summary text
    """
    content = response_content.strip()

    # Remove legacy "Summary:" prefix if present
    if content.lower().startswith("summary:"):
        content = content[8:].strip()

    return content


def estimate_messages_tokens(
    messages: list[dict[str, Any]],
    chars_per_token: int = 4,
) -> int:
    """Estimate token count for a messages array.

    Uses character-based estimation (4 chars per token is a reasonable default).

    Args:
        messages: Array of message dicts with 'role' and 'content'
        chars_per_token: Characters per token ratio (default: 4)

    Returns:
        Estimated token count
    """
    total_chars = 0
    for msg in messages:
        # Count role overhead (system/user/assistant/tool)
        total_chars += len(msg.get("role", "")) + 10  # role + JSON overhead

        # Count content
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            # Multi-modal content (text + images)
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total_chars += len(part.get("text", ""))

    return total_chars // chars_per_token


def should_compact_context(
    messages: list[dict[str, Any]],
    context_window_size: int,
    compaction_threshold: float = 0.90,
    chars_per_token: int = 4,
) -> bool:
    """Check if context should be compacted.

    Args:
        messages: Current messages array
        context_window_size: Maximum context window in tokens
        compaction_threshold: Trigger compaction at this percentage (default: 90%)
        chars_per_token: Characters per token ratio

    Returns:
        True if compaction should be triggered
    """
    estimated_tokens = estimate_messages_tokens(messages, chars_per_token)
    threshold_tokens = int(context_window_size * compaction_threshold)

    should_compact = estimated_tokens >= threshold_tokens

    if should_compact:
        logger.info(
            f"Context compaction needed: {estimated_tokens} tokens "
            f"(threshold: {threshold_tokens}, window: {context_window_size})"
        )

    return should_compact


def build_compaction_input(messages_to_summarize: list[dict[str, Any]]) -> str:
    """Build input for summarizer agent to compact conversation history.

    Args:
        messages_to_summarize: Messages to be summarized

    Returns:
        Formatted prompt for summarizer
    """
    # Format messages as conversation transcript
    transcript_parts = []
    for msg in messages_to_summarize:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, str):
            transcript_parts.append(f"{role.upper()}: {content}")
        elif isinstance(content, list):
            # Handle multi-modal content (just extract text parts)
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            if text_parts:
                transcript_parts.append(f"{role.upper()}: {' '.join(text_parts)}")

    transcript = "\n\n".join(transcript_parts)

    return (
        f"Summarize this conversation to compact it for future context:\n\n"
        f"{transcript}\n\n"
        f"Provide a concise summary (50-200 tokens) that:\n"
        f"- Captures key information and decisions\n"
        f"- Preserves essential context for continuing work\n"
        f"- Lists important files/code entities mentioned\n"
        f"- Notes current state and next steps\n\n"
        f"Format: 'Summary: [key points in 1-3 sentences]'"
    )
