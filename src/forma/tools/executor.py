"""Tool execution loop for server-side tool calling.

Uses semantic block format for tool events:
[TOOL_START: tool_name]
id: call_id
status: running
args: {"key": "value"}
[/TOOL_START]

[TOOL_END: tool_name]
id: call_id
status: success/failed
duration: 123ms
result: preview...
[/TOOL_END]

This format is human-readable and self-documenting for API consumers.
"""

import asyncio
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from forma.tools.base import ToolCall, ToolResult
from forma.tools.registry import ToolRegistry, get_registry

logger = logging.getLogger(__name__)


@dataclass
class ToolExecutionEvent:
    """Event emitted during tool execution for streaming.

    Attributes:
        event_type: Type of event (tool_call_start, tool_call_end, tool_loop_progress)
        data: Event data payload
        timestamp: Event timestamp in ms
    """

    event_type: str
    data: dict[str, Any]
    timestamp: float = field(default_factory=lambda: time.time() * 1000)

    def to_sse(self) -> str:
        """Convert event to SSE format string."""
        return f"data: {json.dumps({'type': self.event_type, **self.data, 'timestamp': self.timestamp})}\n\n"

    def to_content_delta(self) -> str:
        """Convert event to OpenAI content delta format (for standard OpenAI clients).

        Returns content that can be embedded in choices[0].delta.content
        Uses semantic block format: [TOOL_START/END: name]...[/TOOL_START/END]
        so the UI can detect and parse tool events separately from regular content.
        This format is human-readable and self-documenting.
        """
        event_data = {"type": self.event_type, "timestamp": self.timestamp, **self.data}

        if self.event_type == "tool_call_start":
            # Tool call start
            tool_name = self.data.get("name", "unknown")
            tool_id = self.data.get("id", "")
            args = self.data.get("arguments", {})
            args_str = json.dumps(args, ensure_ascii=False) if args else "{}"
            return f"[TOOL_START: {tool_name}]\nid: {tool_id}\nstatus: running\nargs: {args_str}\n[/TOOL_START]"

        elif self.event_type == "tool_call_end":
            # Tool call end
            tool_name = self.data.get("name", "unknown")
            tool_id = self.data.get("id", "")
            success = self.data.get("success", False)
            duration_ms = self.data.get("duration_ms", 0)
            result_preview = self.data.get("result_preview", "")
            status = "success" if success else "failed"
            # Truncate result preview for readability
            preview = result_preview[:200] if result_preview else ""
            return f"[TOOL_END: {tool_name}]\nid: {tool_id}\nstatus: {status}\nduration: {duration_ms:.1f}ms\nresult: {preview}\n[/TOOL_END]"

        elif self.event_type == "tool_loop_complete":
            # Tool loop complete - emit as a simple marker
            total_calls = self.data.get("total_tool_calls", 0)
            total_time = self.data.get("total_tool_time_ms", 0)
            iterations = self.data.get("iterations", 0)
            return f"[TOOL_LOOP_COMPLETE]\niterations: {iterations}\ntotal_calls: {total_calls}\ntotal_time: {total_time:.1f}ms\n[/TOOL_LOOP_COMPLETE]"

        elif self.event_type == "tool_loop_progress":
            # Tool loop progress
            iteration = self.data.get("iteration", 0)
            max_iterations = self.data.get("max_iterations", 0)
            return f"[TOOL_PROGRESS]\niteration: {iteration}/{max_iterations}\n[/TOOL_PROGRESS]"

        elif self.event_type == "tool_calls_received":
            # Tool calls received
            count = self.data.get("count", 0)
            tools = self.data.get("tools", [])
            tools_str = ", ".join(t.get("name", "?") for t in tools) if tools else ""
            return (
                f"[TOOL_CALLS_RECEIVED]\ncount: {count}\ntools: {tools_str}\n[/TOOL_CALLS_RECEIVED]"
            )

        else:
            # Generic fallback for unknown event types
            return f"[TOOL_EVENT]\ntype: {self.event_type}\ndata: {json.dumps(event_data)}\n[/TOOL_EVENT]"


@dataclass
class ToolExecutionRecord:
    """Record of a single tool execution.

    Attributes:
        id: Tool call ID
        name: Tool name
        arguments: Arguments passed to tool
        result: Execution result
        error: Error message if failed
    """

    id: str
    name: str
    arguments: dict[str, Any]
    result: ToolResult | None = None
    error: str | None = None


@dataclass
class ToolExecutionResult:
    """Result of the complete tool execution loop.

    Attributes:
        response: Final response from upstream (after all tool iterations)
        final_messages: Final message history including tool results (for streaming)
        iterations: Number of iterations executed
        tool_calls: List of all tool execution records
        max_iterations_reached: Whether max iterations was hit
        total_tool_time_ms: Total time spent executing tools
    """

    response: dict[str, Any]
    final_messages: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    tool_calls: list[ToolExecutionRecord] = field(default_factory=list)
    max_iterations_reached: bool = False
    total_tool_time_ms: float = 0.0

    def has_tool_calls(self) -> bool:
        """Check if any tools were executed."""
        return len(self.tool_calls) > 0


class ToolExecutor:
    """Executes the tool calling loop.

    The executor manages the complete tool execution flow:
    1. Send request to upstream with tools defined
    2. Check response for tool_calls (OpenAI format or natural markup)
    3. Execute each tool call
    4. Append tool results to messages
    5. Repeat until no tool_calls or max iterations reached
    6. Return final response

    The executor does NOT handle:
    - Extraction (handled by Extractor)
    - Retrieval (handled by Storage)
    - Message augmentation (handled by main.py)

    It only handles the tool execution loop after messages are prepared.
    """

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        max_iterations: int = 5,
        timeout: float = 30.0,
    ) -> None:
        """Initialize tool executor.

        Args:
            registry: Tool registry to use (defaults to global registry)
            max_iterations: Maximum number of tool iterations
            timeout: Default timeout for tool execution
        """
        self.registry = registry or get_registry()
        self.max_iterations = max_iterations
        self.timeout = timeout

    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call.

        Args:
            tool_call: Tool call to execute

        Returns:
            ToolResult from execution
        """
        tool = self.registry.get_tool(tool_call.name)

        if tool is None:
            logger.warning(f"Tool '{tool_call.name}' not found in registry")
            return ToolResult(
                success=False,
                error=f"Tool '{tool_call.name}' not found",
            )

        if not self.registry.is_enabled(tool_call.name):
            logger.warning(f"Tool '{tool_call.name}' is disabled")
            return ToolResult(
                success=False,
                error=f"Tool '{tool_call.name}' is disabled",
            )

        # Validate arguments
        is_valid, error_msg = tool.validate_arguments(tool_call.arguments)
        if not is_valid:
            logger.warning(f"Tool '{tool_call.name}' argument validation failed: {error_msg}")
            return ToolResult(
                success=False,
                error=error_msg or "Invalid arguments",
            )

        # Execute tool
        start_time = time.time()
        try:
            logger.info(f"Executing tool '{tool_call.name}' with args: {tool_call.arguments}")
            result = await tool.execute(**tool_call.arguments)
            result.duration_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Tool '{tool_call.name}' completed in {result.duration_ms:.1f}ms "
                f"(success={result.success})"
            )
            return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Tool '{tool_call.name}' execution failed: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )

    def extract_tool_calls(self, response: dict[str, Any]) -> list[ToolCall]:
        """Extract tool calls from an OpenAI API response.

        Args:
            response: OpenAI API response dict

        Returns:
            List of ToolCall objects (empty if no tool calls)
        """
        tool_calls: list[ToolCall] = []

        choices = response.get("choices", [])
        if not choices:
            return tool_calls

        message = choices[0].get("message", {})
        raw_tool_calls = message.get("tool_calls", [])

        for raw_call in raw_tool_calls:
            try:
                tool_call = ToolCall.from_openai_format(raw_call)
                tool_calls.append(tool_call)
            except Exception as e:
                logger.warning(f"Failed to parse tool call: {e}")

        return tool_calls

    def extract_tool_calls_from_lmstudio_format(self, content: str) -> list[ToolCall]:
        """Extract tool calls from Gemma 4/LM Studio format.

        Parses format: <|tool_call>call:tool_name{key:<|"|>value<|"|>}<tool_call|>
        Optionally followed by <|tool_response>

        Args:
            content: Response content string

        Returns:
            List of ToolCall objects (empty if no tool calls found)
        """
        tool_calls: list[ToolCall] = []

        # Pattern: <|tool_call>call:tool_name{args}<tool_call|>
        # The args can contain <|"|> token markers around string values
        # <|tool_response> is optional (LM Studio sometimes omits it)
        # Args can be empty {} or contain key:value pairs
        pattern = r"<\|tool_call\>call:(\w+)\s*\{([^}]*)\}<tool_call\|>(?:<\|tool_response\>)?"

        matches = re.finditer(pattern, content)
        for match in matches:
            tool_name = match.group(1)
            args_str = match.group(2).strip()

            # Skip if not a known tool
            if not self.registry.has_tool(tool_name):
                continue

            # Parse arguments - format: key:<|"|>value<|"|>
            arguments = {}
            if args_str:
                # Clean up LM Studio token markers
                # Replace <|"|> with just quotes
                args_str_clean = re.sub(r'<\|["\']?\|>', '"', args_str)

                # Parse key:value pairs (may have multiple)
                try:
                    # Split by comma if multiple args, but be careful with nested values
                    # Simple approach: look for key:"value" or key:value patterns
                    pairs_pattern = r'(\w+)\s*:\s*"([^"]*)"|(\w+)\s*:\s*([^",]+)'
                    pair_matches = re.finditer(pairs_pattern, args_str_clean)

                    # Get tool schema for type conversion
                    tool = self.registry.get_tool(tool_name)
                    param_schema = tool.parameters.get("properties", {}) if tool else {}

                    for pm in pair_matches:
                        # Group 1-2 is key:"value", Group 3-4 is key:value
                        key = pm.group(1) if pm.group(1) else pm.group(3)
                        value = pm.group(2) if pm.group(2) else pm.group(4)
                        if key and value:
                            key = key.strip()
                            value = value.strip()

                            # Convert value type based on parameter schema
                            param_type = param_schema.get(key, {}).get("type", "string")
                            if param_type == "integer":
                                try:
                                    value = int(value)
                                except ValueError:
                                    # Keep as string if conversion fails
                                    pass
                            elif param_type == "number":
                                try:
                                    value = float(value)
                                except ValueError:
                                    # Keep as string if conversion fails
                                    pass
                            elif param_type == "boolean":
                                value = value.lower() in ("true", "yes", "1")

                            arguments[key] = value
                except Exception as e:
                    logger.warning(f"Failed to parse Gemma 4 args '{args_str}': {e}")

            # Create tool call
            tool_call = ToolCall(
                id=f"call_{tool_name}_{len(tool_calls)}",
                name=tool_name,
                arguments=arguments,
            )
            tool_calls.append(tool_call)
            logger.info(f"Extracted Gemma 4 tool call: {tool_name} -> args: {arguments}")

        return tool_calls

    def append_tool_results(
        self,
        messages: list[dict[str, Any]],
        assistant_message: dict[str, Any],
        tool_results: list[tuple[ToolCall, ToolResult]],
    ) -> None:
        """Append tool results to messages.

        Modifies messages list in-place, adding:
        1. Assistant message with tool_calls
        2. Tool result messages for each execution

        Args:
            messages: Message list to modify
            assistant_message: Original assistant message (may have content)
            tool_results: List of (ToolCall, ToolResult) tuples
        """
        # Append assistant message with tool_calls
        messages.append(assistant_message)

        # Append tool result messages
        for tool_call, result in tool_results:
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result.to_content(),
            }
            messages.append(tool_message)

    async def execute_loop(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        forward_request: Callable[..., Any],
        tool_choice: str | dict[str, Any] = "auto",
        event_callback: Callable[[ToolExecutionEvent], None] | None = None,
    ) -> ToolExecutionResult:
        """Execute the complete tool calling loop.

        Args:
            messages: Chat messages (may be augmented with RAG context)
            tools: Tool definitions in OpenAI format (from client request)
            forward_request: Async function to forward request to upstream
                Signature: async forward_request(messages, tools, tool_choice) -> dict
            tool_choice: Tool choice mode ("auto", "none", "required", or specific tool)
            event_callback: Optional callback to emit events during execution (for streaming)

        Returns:
            ToolExecutionResult with final response and execution history
        """
        iteration = 0
        accumulated_messages = messages.copy()
        tool_execution_records: list[ToolExecutionRecord] = []
        total_tool_time_ms = 0.0

        def emit_event(event_type: str, data: dict[str, Any]) -> None:
            """Helper to emit event if callback provided."""
            if event_callback:
                event = ToolExecutionEvent(event_type=event_type, data=data)
                logger.debug(f"Emitting event {event_type} at {time.time() * 1000:.0f}ms")
                event_callback(event)

        # Get tool names from request for validation
        requested_tool_names = set()
        for tool_def in tools:
            if tool_def.get("type") == "function":
                func = tool_def.get("function", {})
                name = func.get("name", "")
                if name:
                    requested_tool_names.add(name)

        # Verify requested tools are available
        missing_tools = requested_tool_names - set(self.registry.get_tool_names())
        if missing_tools:
            logger.warning(f"Requested tools not available: {missing_tools}")

        while iteration < self.max_iterations:
            logger.info(f"Tool execution iteration {iteration + 1}/{self.max_iterations}")

            # Forward request with tools
            try:
                response = await forward_request(
                    messages=accumulated_messages,
                    tools=tools,
                    tool_choice=tool_choice,
                )
                # Log response structure
                choices = response.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    has_tool_calls = bool(msg.get("tool_calls"))
                    content_preview = msg.get("content", "")[:50] if msg.get("content") else None
                    logger.info(
                        f"Upstream response: has_tool_calls={has_tool_calls}, "
                        f"content_preview={content_preview}, "
                        f"finish_reason={choices[0].get('finish_reason')}"
                    )
            except Exception as e:
                logger.error(f"Upstream request failed in tool loop: {e}")
                # Return partial result with error indicator
                return ToolExecutionResult(
                    response={"error": str(e)},
                    final_messages=accumulated_messages,
                    iterations=iteration,
                    tool_calls=tool_execution_records,
                    max_iterations_reached=False,
                    total_tool_time_ms=total_tool_time_ms,
                )

            # Check for tool calls in response (OpenAI format)
            tool_calls = self.extract_tool_calls(response)

            # If no OpenAI-format tool calls, check for LM Studio/Gemma 4 format
            if not tool_calls:
                choices = response.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    if content:
                        tool_calls = self.extract_tool_calls_from_lmstudio_format(content)
                        if tool_calls:
                            logger.info(
                                f"Extracted {len(tool_calls)} tool calls from LM Studio format"
                            )

            if not tool_calls:
                # No tool calls - final response
                logger.info(f"Tool loop complete after {iteration} iterations")

                # Always emit tool_loop_complete as control signal for streaming
                # (even when no tools were used - needed to unblock streaming generator)
                emit_event(
                    "tool_loop_complete",
                    {
                        "iterations": iteration,
                        "total_tool_calls": len(tool_execution_records),
                        "total_tool_time_ms": total_tool_time_ms,
                    },
                )

                return ToolExecutionResult(
                    response=response,
                    final_messages=accumulated_messages,
                    iterations=iteration,
                    tool_calls=tool_execution_records,
                    max_iterations_reached=False,
                    total_tool_time_ms=total_tool_time_ms,
                )

            logger.info(f"Received {len(tool_calls)} tool calls")

            # Emit iteration progress event (only when tool calls are found)
            emit_event(
                "tool_loop_progress",
                {
                    "iteration": iteration + 1,
                    "max_iterations": self.max_iterations,
                },
            )

            # Emit event for tool calls received
            emit_event(
                "tool_calls_received",
                {
                    "count": len(tool_calls),
                    "tools": [{"name": tc.name, "arguments": tc.arguments} for tc in tool_calls],
                },
            )

            # Build assistant message with tool_calls
            choices = response.get("choices", [])
            if choices:
                assistant_message = choices[0].get("message", {})
                # If we extracted from markup, build proper tool_calls array
                if tool_calls and not assistant_message.get("tool_calls"):
                    # Strip markup from content if present
                    content = assistant_message.get("content", "")
                    if content:
                        # Remove markup from content
                        content_clean = re.sub(r"\w+\s*\([^)]*\)", "", content)
                        content_clean = content_clean.strip()
                        assistant_message["content"] = content_clean if content_clean else None

                    # Add proper tool_calls array
                    assistant_message["tool_calls"] = [
                        {
                            "type": "function",
                            "id": tc.id,
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in tool_calls
                    ]
            else:
                assistant_message = {"role": "assistant", "content": None}

            # Execute each tool call
            tool_results: list[tuple[ToolCall, ToolResult]] = []
            for tool_call in tool_calls:
                # Emit tool call start event
                emit_event(
                    "tool_call_start",
                    {
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    },
                )

                # Small delay to allow streaming generator to process the start event
                # before tool execution begins (ensures real-time delivery)
                await asyncio.sleep(0.05)  # 50ms

                start_time = time.time()

                # Execute tool
                result = await self.execute_tool(tool_call)

                # Track time
                tool_time = (time.time() - start_time) * 1000
                total_tool_time_ms += tool_time

                # Emit tool call end event
                emit_event(
                    "tool_call_end",
                    {
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "success": result.success,
                        "duration_ms": tool_time,
                        "result_preview": str(result.output) if result.success else result.error,
                    },
                )

                # Record execution
                record = ToolExecutionRecord(
                    id=tool_call.id,
                    name=tool_call.name,
                    arguments=tool_call.arguments,
                    result=result,
                    error=result.error if not result.success else None,
                )
                tool_execution_records.append(record)

                tool_results.append((tool_call, result))

            # Append assistant message and tool results to messages
            self.append_tool_results(accumulated_messages, assistant_message, tool_results)

            iteration += 1

        # Max iterations reached
        logger.warning(f"Max iterations ({self.max_iterations}) reached")
        return ToolExecutionResult(
            response=response,
            final_messages=accumulated_messages,
            iterations=iteration,
            tool_calls=tool_execution_records,
            max_iterations_reached=True,
            total_tool_time_ms=total_tool_time_ms,
        )
