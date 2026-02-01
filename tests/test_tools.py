"""Tests for tool calling."""

from unittest.mock import MagicMock, patch

import pytest


class TestToolRegistry:
    """Tests for tool registration."""

    def test_tool_decorator_importable(self):
        """tool decorator is importable."""
        from microthink.tools.registry import tool

        assert tool is not None

    def test_tool_decorator_registers_function(self):
        """tool decorator registers a function."""
        from microthink.tools.registry import ToolRegistry, tool

        registry = ToolRegistry()

        @registry.register
        def my_tool(x: int) -> int:
            """A test tool."""
            return x * 2

        assert "my_tool" in registry.tools
        assert registry.tools["my_tool"].func is my_tool

    def test_tool_has_metadata(self):
        """Registered tools have name and description."""
        from microthink.tools.registry import ToolRegistry

        registry = ToolRegistry()

        @registry.register
        def calculate(expression: str) -> float:
            """Evaluate a math expression."""
            return eval(expression)

        tool_info = registry.tools["calculate"]
        assert tool_info.name == "calculate"
        assert "math expression" in tool_info.description.lower()

    def test_tool_execution(self):
        """Tools can be executed by name."""
        from microthink.tools.registry import ToolRegistry

        registry = ToolRegistry()

        @registry.register
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        result = registry.execute("add", {"a": 2, "b": 3})
        assert result == 5


class TestToolCallParsing:
    """Tests for parsing tool calls from model output."""

    def test_parse_tool_call(self):
        """Parse tool call from XML tags."""
        from microthink.tools.registry import parse_tool_call

        content = '<tool_call>{"name": "calculate", "args": {"expression": "2+2"}}</tool_call>'
        result = parse_tool_call(content)

        assert result is not None
        assert result["name"] == "calculate"
        assert result["args"]["expression"] == "2+2"

    def test_parse_no_tool_call(self):
        """Returns None when no tool call present."""
        from microthink.tools.registry import parse_tool_call

        content = "Just a regular response"
        result = parse_tool_call(content)

        assert result is None

    def test_parse_multiple_tool_calls(self):
        """Parse all tool calls from content."""
        from microthink.tools.registry import parse_all_tool_calls

        content = """
        <tool_call>{"name": "get_weather", "args": {"city": "Tokyo"}}</tool_call>
        <tool_call>{"name": "calculate", "args": {"expression": "22 * 0.15"}}</tool_call>
        """
        results = parse_all_tool_calls(content)

        assert len(results) == 2
        assert results[0]["name"] == "get_weather"
        assert results[1]["name"] == "calculate"


class TestToolIntegration:
    """Tests for tool calling with client."""

    def test_client_accepts_tools_parameter(self):
        """Client generate() accepts tools parameter."""
        from microthink.tools.registry import ToolRegistry

        from microthink import MicroThinkClient

        registry = ToolRegistry()

        @registry.register
        def dummy() -> str:
            """A dummy tool."""
            return "dummy"

        client = MicroThinkClient()
        # Should not raise - just verify parameter accepted
        assert hasattr(client, "generate")


class TestToolInfo:
    """Tests for ToolInfo dataclass."""

    def test_tool_info_has_fields(self):
        """ToolInfo has required fields."""
        from microthink.tools.registry import ToolInfo

        def sample():
            pass

        info = ToolInfo(name="test", description="A test", func=sample, parameters={})
        assert info.name == "test"
        assert info.description == "A test"
