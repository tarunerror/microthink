"""
Tool registry for MicroThink.

Provides a decorator-based system for registering functions
that can be called by the model during generation.
"""

import inspect
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolInfo:
    """Information about a registered tool."""

    name: str
    description: str
    func: Callable
    parameters: Dict[str, Any]


class ToolRegistry:
    """
    Registry for tools that can be called by the model.

    Example:
        >>> registry = ToolRegistry()
        >>> @registry.register
        ... def greet(name: str) -> str:
        ...     '''Generate a greeting message.'''
        ...     return f"Hello, {name}!"
        >>> registry.execute("greet", {"name": "World"})
        'Hello, World!'
    """

    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        self.tools: Dict[str, ToolInfo] = {}

    def register(self, func: Callable) -> Callable:
        """
        Register a function as a tool.

        Args:
            func: The function to register.

        Returns:
            The original function unchanged.
        """
        name = func.__name__
        description = func.__doc__ or ""
        parameters = self._extract_parameters(func)

        self.tools[name] = ToolInfo(
            name=name,
            description=description.strip(),
            func=func,
            parameters=parameters,
        )

        return func

    def _extract_parameters(self, func: Callable) -> Dict[str, Any]:
        """Extract parameter info from function signature."""
        sig = inspect.signature(func)
        params = {}

        for param_name, param in sig.parameters.items():
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"

            params[param_name] = {"type": param_type}

        return params

    def execute(self, name: str, args: Dict[str, Any]) -> Any:
        """
        Execute a tool by name.

        Args:
            name: The tool name.
            args: Arguments to pass to the tool.

        Returns:
            The tool's return value.

        Raises:
            KeyError: If tool not found.
        """
        if name not in self.tools:
            raise KeyError(f"Tool not found: {name}")

        return self.tools[name].func(**args)

    def get_tools_prompt(self) -> str:
        """
        Generate a prompt describing available tools.

        Returns:
            A string describing tools for the model.
        """
        if not self.tools:
            return ""

        lines = ["Available tools:"]
        for name, info in self.tools.items():
            params_str = ", ".join(
                f"{p}: {param_info['type']}"
                for p, param_info in info.parameters.items()
            )
            lines.append(f"- {name}({params_str}): {info.description}")

        lines.append("")
        lines.append("To use a tool, output:")
        lines.append(
            '<tool_call>{"name": "tool_name", "args": {"arg": "value"}}</tool_call>'
        )

        return "\n".join(lines)


def parse_tool_call(content: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single tool call from content.

    Args:
        content: The model output to parse.

    Returns:
        Dict with 'name' and 'args' if found, None otherwise.
    """
    match = re.search(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
    if not match:
        return None

    try:
        data = json.loads(match.group(1))
        if isinstance(data, dict):
            return data
        return None
    except json.JSONDecodeError:
        return None


def parse_all_tool_calls(content: str) -> List[Dict[str, Any]]:
    """
    Parse all tool calls from content.

    Args:
        content: The model output to parse.

    Returns:
        List of dicts with 'name' and 'args'.
    """
    matches = re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
    results: List[Dict[str, Any]] = []

    for match in matches:
        try:
            data = json.loads(match)
            if isinstance(data, dict):
                results.append(data)
        except json.JSONDecodeError:
            continue

    return results


# Global default registry
_default_registry = ToolRegistry()


def tool(func: Callable) -> Callable:
    """
    Decorator to register a function as a tool.

    Example:
        >>> @tool
        ... def greet(name: str) -> str:
        ...     '''Generate a greeting message.'''
        ...     return f"Hello, {name}!"
    """
    return _default_registry.register(func)
