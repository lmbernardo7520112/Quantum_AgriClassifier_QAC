"""
QAC — Tool Registry.

Manages tool registration, schema validation, precondition checking,
and isolated tool execution. Tools cannot depend on each other directly.

Supports:
- Invariant 4 (Isolation): tools only access context_manager
- Test C (Contract Violation): structured error on precondition failure
"""

from __future__ import annotations

import time
from typing import Any, Callable, Awaitable

from mcp_server.schemas import SchemaRegistry


class ToolError(Exception):
    """Structured error for tool contract violations."""

    def __init__(self, tool_name: str, error_type: str, message: str, details: dict | None = None):
        self.tool_name = tool_name
        self.error_type = error_type
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": True,
            "tool_name": self.tool_name,
            "error_type": self.error_type,
            "message": self.message,
            "details": self.details,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }


# Type alias for tool functions
ToolFunction = Callable[..., Awaitable[dict[str, Any]] | dict[str, Any]]


class ToolRegistry:
    """
    Registry for tool functions with schema validation and precondition enforcement.

    Tools are fully isolated — they receive context through the execution engine,
    never directly from each other (Invariant 4).
    """

    def __init__(self, schema_registry: SchemaRegistry) -> None:
        self._schema_registry = schema_registry
        self._tools: dict[str, ToolFunction] = {}
        self._precondition_checkers: dict[str, Callable[[], bool]] = {}

    def register(
        self,
        tool_name: str,
        func: ToolFunction,
        precondition_checker: Callable[[], bool] | None = None,
    ) -> None:
        """Register a tool function with optional precondition checker."""
        # Verify schema exists
        schema = self._schema_registry.get_schema(tool_name)
        if schema is None:
            raise ValueError(f"No schema found for tool: {tool_name}. Register schema first.")
        self._tools[tool_name] = func
        if precondition_checker:
            self._precondition_checkers[tool_name] = precondition_checker

    def get(self, tool_name: str) -> ToolFunction | None:
        """Get a registered tool function."""
        return self._tools.get(tool_name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def validate_and_check(self, tool_name: str, input_data: dict[str, Any]) -> None:
        """
        Validate input schema and check preconditions.
        Raises ToolError on failure (Test C — structured error, no crash).
        """
        # Check tool exists
        if tool_name not in self._tools:
            raise ToolError(
                tool_name=tool_name,
                error_type="TOOL_NOT_FOUND",
                message=f"Tool '{tool_name}' is not registered.",
                details={"available_tools": self.list_tools()},
            )

        # Validate input schema
        errors = self._schema_registry.validate_input(tool_name, input_data)
        if errors:
            raise ToolError(
                tool_name=tool_name,
                error_type="SCHEMA_VALIDATION_ERROR",
                message=f"Input validation failed for '{tool_name}'.",
                details={"validation_errors": errors},
            )

        # Check preconditions
        checker = self._precondition_checkers.get(tool_name)
        if checker and not checker():
            preconditions = self._schema_registry.get_preconditions(tool_name)
            raise ToolError(
                tool_name=tool_name,
                error_type="PRECONDITION_FAILED",
                message=f"Preconditions not met for '{tool_name}'.",
                details={"required_preconditions": preconditions},
            )

    def validate_output(self, tool_name: str, output_data: dict[str, Any]) -> list[str]:
        """Validate output against schema. Returns list of errors."""
        return self._schema_registry.validate_output(tool_name, output_data)

    def is_registered(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self._tools
