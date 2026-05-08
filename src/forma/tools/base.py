"""Base classes for tool definitions."""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Result of a tool execution.

    Attributes:
        success: Whether the tool execution succeeded
        output: The output content (string or JSON-serializable dict)
        error: Error message if execution failed
        duration_ms: Execution duration in milliseconds
        metadata: Additional metadata about the execution
    """

    success: bool
    output: str | dict[str, Any] = ""
    error: str | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> str:
        """Convert result to content string for API response.

        Returns JSON string if output is dict, otherwise string.
        Errors are returned as JSON error objects.
        """
        if not self.success:
            return json.dumps(
                {
                    "error": self.error or "Tool execution failed",
                    "success": False,
                }
            )

        if isinstance(self.output, dict):
            return json.dumps(self.output)
        return str(self.output)


@dataclass
class ToolCall:
    """Represents a tool call from the model.

    Attributes:
        id: Unique identifier for this tool call
        name: Name of the tool to execute
        arguments: Arguments passed to the tool (parsed from JSON string)
    """

    id: str
    name: str
    arguments: dict[str, Any]

    @classmethod
    def from_openai_format(cls, tool_call: dict[str, Any]) -> "ToolCall":
        """Parse tool call from OpenAI API response format.

        Args:
            tool_call: Tool call dict from OpenAI response:
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Berlin"}'
                    }
                }

        Returns:
            ToolCall instance
        """
        function = tool_call.get("function", {})
        name = function.get("name", "")
        arguments_str = function.get("arguments", "{}")

        try:
            arguments = json.loads(arguments_str) if arguments_str else {}
        except json.JSONDecodeError:
            arguments = {"_raw_arguments": arguments_str}

        return cls(
            id=tool_call.get("id", ""),
            name=name,
            arguments=arguments,
        )


class Tool(ABC):
    """Base class for defining a tool.

    Tools must define:
    - name: Unique identifier for the tool
    - description: Human-readable description for the model
    - parameters: JSON schema for parameters (OpenAI format)
    - execute: Implementation of the tool logic

    Example:
        class SearchWebTool(Tool):
            name = "search_web"
            description = "Search the web for information"

            parameters = {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }

            async def execute(self, query: str) -> ToolResult:
                # Implementation here
                results = await search(query)
                return ToolResult(success=True, output=results)
    """

    name: str
    description: str
    parameters: dict[str, Any]

    # Optional settings
    timeout: float = 30.0  # Default timeout in seconds
    dangerous: bool = False  # Requires explicit permission
    enabled: bool = True  # Whether tool is enabled by default

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given arguments.

        Args:
            **kwargs: Arguments matching the parameter schema

        Returns:
            ToolResult with success/failure and output
        """
        pass

    def to_openai_format(self) -> dict[str, Any]:
        """Convert tool definition to OpenAI function format.

        Returns:
            Dict in OpenAI tools format:
            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": {...}
                }
            }
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def validate_arguments(self, arguments: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate arguments against parameter schema.

        Basic validation for required parameters.
        More sophisticated validation can be added later.

        Args:
            arguments: Arguments to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.parameters:
            return True, None

        required = self.parameters.get("required", [])
        properties = self.parameters.get("properties", {})

        # Check required parameters
        for param in required:
            if param not in arguments:
                return False, f"Missing required parameter: {param}"

        # Check parameter types (basic check)
        for param, value in arguments.items():
            if param in properties:
                expected_type = properties[param].get("type")
                if expected_type:
                    if not self._check_type(value, expected_type):
                        return (
                            False,
                            f"Parameter '{param}' has wrong type, expected {expected_type}",
                        )

        return True, None

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Basic type checking."""
        type_map: dict[str, type | tuple[type, ...]] = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        expected_python_type = type_map.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        return True  # Unknown type, allow it


class SyncTool(Tool):
    """Base class for synchronous tools.

    For tools that don't need async execution.
    Override execute_sync instead of execute.
    """

    def execute_sync(self, **kwargs: Any) -> ToolResult:
        """Synchronous execution implementation.

        Override this method to implement synchronous tool logic.
        """
        raise NotImplementedError("Subclasses must implement execute_sync")

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Wrap sync execution in async thread to avoid blocking the event loop."""
        return await asyncio.to_thread(self.execute_sync, **kwargs)


class EchoTool(SyncTool):
    """Simple echo tool for testing.

    Returns the input arguments as output.
    Useful for testing the tool execution pipeline.
    """

    name = "echo"
    description = "Echo back the input arguments (for testing)"
    parameters = {
        "type": "object",
        "properties": {"message": {"type": "string", "description": "Message to echo back"}},
        "required": ["message"],
    }

    def execute_sync(self, **kwargs: Any) -> ToolResult:
        """Echo the message back."""
        message = kwargs.get("message", "")
        start = time.time()
        result = ToolResult(
            success=True,
            output={"echoed": message, "arguments": kwargs},
            duration_ms=(time.time() - start) * 1000,
        )
        return result
