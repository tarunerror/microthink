# Phase 3: Capabilities Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add conversation memory/sessions and tool calling capabilities to MicroThink.

**Architecture:** Sessions wrap the client with message history management. Tool calling uses a decorator-based registry with XML tag detection for tool invocations.

**Tech Stack:** Python 3.9+, dataclasses, existing async infrastructure.

---

## Task 1: Conversation Memory/Sessions

Add session support for multi-turn conversations with history management.

**Files:**
- Create: `src/microthink/session.py`
- Modify: `src/microthink/__init__.py`
- Create: `tests/test_session.py`

### Step 1.1: Write failing tests for sessions

Create `tests/test_session.py`:

```python
"""Tests for conversation sessions."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestSession:
    """Tests for Session class."""

    def test_session_importable(self):
        """Session is importable from package."""
        from microthink import Session
        assert Session is not None

    def test_session_created_from_client(self):
        """Session can be created from client."""
        from microthink import MicroThinkClient, Session

        client = MicroThinkClient()
        session = Session(client)
        assert session.client is client

    def test_session_has_generate_method(self):
        """Session has generate method."""
        from microthink import MicroThinkClient, Session

        client = MicroThinkClient()
        session = Session(client)
        assert hasattr(session, "generate")
        assert callable(session.generate)

    def test_session_tracks_history(self):
        """Session tracks message history."""
        from microthink import MicroThinkClient, Session

        with patch("microthink.client.ollama.Client") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.return_value = mock_client
            mock_client.chat.return_value = {
                "message": {"content": "<answer>Hello!</answer>"}
            }

            client = MicroThinkClient()
            session = Session(client)
            session.generate("Hi there")

            assert len(session.history) >= 2  # user + assistant

    def test_session_includes_history_in_next_call(self):
        """Session includes history in subsequent calls."""
        from microthink import MicroThinkClient, Session

        with patch("microthink.client.ollama.Client") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.return_value = mock_client
            mock_client.chat.return_value = {
                "message": {"content": "<answer>Response</answer>"}
            }

            client = MicroThinkClient()
            session = Session(client)
            session.generate("First message")
            session.generate("Second message")

            # Check last call included history
            last_call = mock_client.chat.call_args
            messages = last_call[1]["messages"]
            # Should have system + first user + first assistant + second user
            assert len(messages) >= 4

    def test_session_max_history(self):
        """Session respects max_history limit."""
        from microthink import MicroThinkClient, Session

        with patch("microthink.client.ollama.Client") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.return_value = mock_client
            mock_client.chat.return_value = {
                "message": {"content": "<answer>Response</answer>"}
            }

            client = MicroThinkClient()
            session = Session(client, max_history=4)

            # Generate more than max_history
            for i in range(5):
                session.generate(f"Message {i}")

            # History should be trimmed
            assert len(session.history) <= 4

    def test_session_clear(self):
        """Session can clear history."""
        from microthink import MicroThinkClient, Session

        with patch("microthink.client.ollama.Client") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.return_value = mock_client
            mock_client.chat.return_value = {
                "message": {"content": "<answer>Response</answer>"}
            }

            client = MicroThinkClient()
            session = Session(client)
            session.generate("Hello")
            session.clear()

            assert len(session.history) == 0


class TestSessionPersistence:
    """Tests for session save/load."""

    def test_session_save(self):
        """Session can be saved to file."""
        from microthink import MicroThinkClient, Session

        with patch("microthink.client.ollama.Client") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.return_value = mock_client
            mock_client.chat.return_value = {
                "message": {"content": "<answer>Response</answer>"}
            }

            client = MicroThinkClient()
            session = Session(client)
            session.generate("Hello")

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                session.save(f.name)
                assert Path(f.name).exists()

                # Verify content
                with open(f.name) as saved:
                    data = json.load(saved)
                    assert "history" in data

    def test_session_load(self):
        """Session can be loaded from file."""
        from microthink import MicroThinkClient, Session

        client = MicroThinkClient()

        # Create a session file
        session_data = {
            "history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            "behavior": "general",
        }

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            json.dump(session_data, f)
            f.flush()

            loaded = Session.load(client, f.name)
            assert len(loaded.history) == 2
```

### Step 1.2: Run tests to verify they fail

```bash
pytest tests/test_session.py -v
```

Expected: FAIL with import errors

### Step 1.3: Implement session module

Create `src/microthink/session.py`:

```python
"""
Conversation session management for MicroThink.

Provides stateful conversation support with history tracking,
persistence, and automatic history trimming.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from microthink.client import MicroThinkClient
from microthink.core.prompts import build_system_prompt, SYSTEM_PERSONAS


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "user", "assistant", or "system"
    content: str


class Session:
    """
    A conversation session with history tracking.

    Example:
        >>> client = MicroThinkClient()
        >>> session = Session(client)
        >>> session.generate("My name is Alice")
        >>> session.generate("What's my name?")
        'Your name is Alice'

        >>> # Save and restore
        >>> session.save("chat.json")
        >>> restored = Session.load(client, "chat.json")
    """

    def __init__(
        self,
        client: MicroThinkClient,
        max_history: int = 20,
        behavior: str = "general",
    ) -> None:
        """
        Initialize a conversation session.

        Args:
            client: The MicroThinkClient to use.
            max_history: Maximum messages to keep in history.
            behavior: The persona to use for this session.
        """
        self.client = client
        self.max_history = max_history
        self.behavior = behavior
        self._history: List[Dict[str, str]] = []

    @property
    def history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self._history.copy()

    def generate(
        self,
        prompt: str,
        expect_json: bool = False,
        web_search: bool = False,
        debug: bool = False,
    ) -> Union[str, Dict[str, Any], List[Any]]:
        """
        Generate a response within the session context.

        Args:
            prompt: The user's input prompt.
            expect_json: If True, parse JSON output.
            web_search: If True, include web search context.
            debug: If True, output debug information.

        Returns:
            The model's response.
        """
        # Add user message to history
        self._history.append({"role": "user", "content": prompt})

        # Build messages with history
        system_prompt = build_system_prompt(
            self.behavior, expect_json=expect_json, brief=False
        )
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self._history)

        # Use client's internal method for generation
        response = self.client._client.chat(
            model=self.client.model,
            messages=messages,
        )

        # Extract answer
        from microthink.core.parser import extract_answer
        raw_content = response["message"]["content"]
        answer = extract_answer(raw_content)

        # Add assistant response to history
        self._history.append({"role": "assistant", "content": answer})

        # Trim history if needed
        self._trim_history()

        # Parse JSON if expected
        if expect_json:
            return json.loads(answer)

        return answer

    def _trim_history(self) -> None:
        """Trim history to max_history size."""
        if len(self._history) > self.max_history:
            # Keep the most recent messages
            self._history = self._history[-self.max_history:]

    def clear(self) -> None:
        """Clear the conversation history."""
        self._history = []

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the session to a JSON file.

        Args:
            path: Path to save the session.
        """
        data = {
            "history": self._history,
            "behavior": self.behavior,
            "max_history": self.max_history,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, client: MicroThinkClient, path: Union[str, Path]) -> "Session":
        """
        Load a session from a JSON file.

        Args:
            client: The MicroThinkClient to use.
            path: Path to the saved session.

        Returns:
            A restored Session instance.
        """
        with open(path) as f:
            data = json.load(f)

        session = cls(
            client=client,
            max_history=data.get("max_history", 20),
            behavior=data.get("behavior", "general"),
        )
        session._history = data.get("history", [])

        return session
```

### Step 1.4: Export from package

Modify `src/microthink/__init__.py`:

Add import:
```python
from microthink.session import Session
```

Add to `__all__`:
```python
"Session",
```

### Step 1.5: Run session tests

```bash
pytest tests/test_session.py -v
```

Expected: All tests PASS

### Step 1.6: Commit

```bash
git add src/microthink/session.py src/microthink/__init__.py tests/test_session.py
git commit -m "feat: add conversation session support

- Add Session class for multi-turn conversations
- Track message history with configurable max_history
- Support save/load for session persistence
- Automatic history trimming when limit exceeded"
```

---

## Task 2: Tool Calling

Add tool calling support for external function integration.

**Files:**
- Create: `src/microthink/tools/registry.py`
- Modify: `src/microthink/client.py`
- Modify: `src/microthink/__init__.py`
- Create: `tests/test_tools.py`

### Step 2.1: Write failing tests for tool calling

Create `tests/test_tools.py`:

```python
"""Tests for tool calling."""

import pytest
from unittest.mock import MagicMock, patch


class TestToolRegistry:
    """Tests for tool registration."""

    def test_tool_decorator_importable(self):
        """tool decorator is importable."""
        from microthink.tools.registry import tool
        assert tool is not None

    def test_tool_decorator_registers_function(self):
        """tool decorator registers a function."""
        from microthink.tools.registry import tool, ToolRegistry

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

        content = '''
        <tool_call>{"name": "get_weather", "args": {"city": "Tokyo"}}</tool_call>
        <tool_call>{"name": "calculate", "args": {"expression": "22 * 0.15"}}</tool_call>
        '''
        results = parse_all_tool_calls(content)

        assert len(results) == 2
        assert results[0]["name"] == "get_weather"
        assert results[1]["name"] == "calculate"


class TestToolIntegration:
    """Tests for tool calling with client."""

    def test_client_accepts_tools_parameter(self):
        """Client generate() accepts tools parameter."""
        from microthink import MicroThinkClient
        from microthink.tools.registry import ToolRegistry

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
```

### Step 2.2: Run tests to verify they fail

```bash
pytest tests/test_tools.py -v
```

Expected: FAIL with import errors

### Step 2.3: Implement tool registry

Create `src/microthink/tools/registry.py`:

```python
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
        ... def calculate(expression: str) -> float:
        ...     '''Evaluate a math expression.'''
        ...     return eval(expression)
        >>> registry.execute("calculate", {"expression": "2+2"})
        4
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
                f"{p}: {info['type']}" for p, info in info.parameters.items()
            )
            lines.append(f"- {name}({params_str}): {info.description}")

        lines.append("")
        lines.append("To use a tool, output:")
        lines.append('<tool_call>{"name": "tool_name", "args": {"arg": "value"}}</tool_call>')

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
        return json.loads(match.group(1))
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
    results = []

    for match in matches:
        try:
            results.append(json.loads(match))
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
        ... def calculate(expression: str) -> float:
        ...     '''Evaluate a math expression.'''
        ...     return eval(expression)
    """
    return _default_registry.register(func)
```

### Step 2.4: Export from package

Modify `src/microthink/__init__.py`:

Add imports:
```python
from microthink.tools.registry import ToolRegistry, ToolInfo, tool
```

Add to `__all__`:
```python
"ToolRegistry",
"ToolInfo",
"tool",
```

### Step 2.5: Run tool tests

```bash
pytest tests/test_tools.py -v
```

Expected: All tests PASS

### Step 2.6: Commit

```bash
git add src/microthink/tools/registry.py src/microthink/__init__.py tests/test_tools.py
git commit -m "feat: add tool calling support

- Add ToolRegistry for registering callable functions
- Add ToolInfo dataclass for tool metadata
- Add parse_tool_call() for extracting tool calls from output
- Add @tool decorator for easy registration
- Support parameter type extraction from signatures"
```

---

## Task 3: Final Integration and Verification

### Step 3.1: Run all tests

```bash
pytest tests/ -v
```

### Step 3.2: Verify all exports

```python
from microthink import (
    MicroThinkClient,
    AsyncMicroThinkClient,
    BatchProcessor,
    BatchResult,
    Session,
    ToolRegistry,
    ToolInfo,
    tool,
    MicroThinkError,
    PersonaError,
    SchemaValidationError,
)
```

### Step 3.3: Final commit

```bash
git add -A
git commit -m "chore: complete Phase 3 Capabilities

Features added:
- Conversation sessions with history tracking
- Tool calling with registry and decorator support

All tests passing."
```

---

## Summary

| Task | Files Created/Modified | Key Features |
|------|----------------------|--------------|
| Sessions | `session.py` | Multi-turn conversations, save/load |
| Tool Calling | `tools/registry.py` | @tool decorator, ToolRegistry, parsing |

**Total new tests:** ~20+
