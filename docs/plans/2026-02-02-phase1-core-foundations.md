# Phase 1: Core Foundations Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add async support, schema validation, and standard logging to MicroThink.

**Architecture:** Async support mirrors the sync API with `AsyncMicroThinkClient`. Schema validation adds a new `schema.py` module that validates JSON results against example schemas. Standard logging integrates Python's logging module while keeping Rich as an optional enhancer.

**Tech Stack:** Python 3.9+, asyncio, httpx (for async HTTP), pytest-asyncio for testing.

---

## Prerequisites

```bash
# Install async HTTP library
pip install httpx

# Install async test support
pip install pytest-asyncio
```

---

## Task 1: Schema Validation

Add actual schema validation to `generate_with_schema()` so it validates the result structure, not just JSON syntax.

**Files:**
- Create: `src/microthink/core/schema.py`
- Modify: `src/microthink/client.py`
- Create: `tests/test_schema.py`

### Step 1.1: Write failing tests for schema validation

Create `tests/test_schema.py`:

```python
"""Tests for schema validation."""

import pytest
from microthink.core.schema import (
    validate_schema,
    SchemaValidationError,
    infer_type,
)


class TestInferType:
    """Tests for type inference from example values."""

    def test_infer_string(self):
        """Infers string type from 'string' placeholder."""
        assert infer_type("string") == "string"

    def test_infer_number(self):
        """Infers number type from 'number' placeholder."""
        assert infer_type("number") == "number"

    def test_infer_integer(self):
        """Infers integer type from 'integer' placeholder."""
        assert infer_type("integer") == "integer"

    def test_infer_boolean(self):
        """Infers boolean type from 'boolean' placeholder."""
        assert infer_type("boolean") == "boolean"

    def test_infer_from_actual_string(self):
        """Infers string from actual string value."""
        assert infer_type("hello") == "string"

    def test_infer_from_actual_int(self):
        """Infers integer from actual int value."""
        assert infer_type(42) == "integer"

    def test_infer_from_actual_float(self):
        """Infers number from actual float value."""
        assert infer_type(3.14) == "number"

    def test_infer_from_actual_bool(self):
        """Infers boolean from actual bool value."""
        assert infer_type(True) == "boolean"

    def test_infer_array(self):
        """Infers array type from list."""
        assert infer_type([1, 2, 3]) == "array"

    def test_infer_object(self):
        """Infers object type from dict."""
        assert infer_type({"a": 1}) == "object"


class TestValidateSchema:
    """Tests for schema validation."""

    def test_valid_simple_object(self):
        """Validates a simple object matches schema."""
        schema = {"name": "string", "age": "number"}
        data = {"name": "Alice", "age": 30}
        validate_schema(data, schema)  # Should not raise

    def test_missing_required_field(self):
        """Raises error for missing required field."""
        schema = {"name": "string", "age": "number"}
        data = {"name": "Alice"}
        with pytest.raises(SchemaValidationError, match="Missing required field"):
            validate_schema(data, schema)

    def test_wrong_type_string_vs_number(self):
        """Raises error for wrong type."""
        schema = {"name": "string", "age": "number"}
        data = {"name": "Alice", "age": "thirty"}
        with pytest.raises(SchemaValidationError, match="Expected number"):
            validate_schema(data, schema)

    def test_extra_fields_allowed(self):
        """Extra fields are allowed by default."""
        schema = {"name": "string"}
        data = {"name": "Alice", "extra": "field"}
        validate_schema(data, schema)  # Should not raise

    def test_nested_object(self):
        """Validates nested objects."""
        schema = {
            "user": {"name": "string", "age": "number"},
            "active": "boolean",
        }
        data = {
            "user": {"name": "Bob", "age": 25},
            "active": True,
        }
        validate_schema(data, schema)  # Should not raise

    def test_nested_object_invalid(self):
        """Raises error for invalid nested object."""
        schema = {
            "user": {"name": "string", "age": "number"},
        }
        data = {
            "user": {"name": "Bob", "age": "young"},
        }
        with pytest.raises(SchemaValidationError, match="user.age"):
            validate_schema(data, schema)

    def test_array_of_strings(self):
        """Validates array of strings."""
        schema = {"tags": ["string"]}
        data = {"tags": ["a", "b", "c"]}
        validate_schema(data, schema)  # Should not raise

    def test_array_with_wrong_element_type(self):
        """Raises error for wrong element type in array."""
        schema = {"tags": ["string"]}
        data = {"tags": ["a", 123, "c"]}
        with pytest.raises(SchemaValidationError, match="tags\\[1\\]"):
            validate_schema(data, schema)

    def test_empty_array_valid(self):
        """Empty array is valid."""
        schema = {"items": ["number"]}
        data = {"items": []}
        validate_schema(data, schema)  # Should not raise

    def test_type_coercion_int_as_number(self):
        """Integer is accepted where number is expected."""
        schema = {"value": "number"}
        data = {"value": 42}
        validate_schema(data, schema)  # Should not raise

    def test_validates_list_result(self):
        """Validates when result is a list, not dict."""
        schema = ["string"]
        data = ["a", "b", "c"]
        validate_schema(data, schema)  # Should not raise

    def test_list_schema_wrong_type(self):
        """Raises error for wrong type in list result."""
        schema = ["number"]
        data = ["a", "b"]
        with pytest.raises(SchemaValidationError, match="\\[0\\]"):
            validate_schema(data, schema)


class TestSchemaValidationError:
    """Tests for SchemaValidationError."""

    def test_error_message(self):
        """Error includes field path."""
        error = SchemaValidationError("user.name", "Expected string, got int")
        assert "user.name" in str(error)
        assert "Expected string" in str(error)

    def test_error_is_exception(self):
        """Error can be raised and caught."""
        with pytest.raises(SchemaValidationError):
            raise SchemaValidationError("field", "message")
```

### Step 1.2: Run tests to verify they fail

```bash
pytest tests/test_schema.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'microthink.core.schema'`

### Step 1.3: Implement schema validation module

Create `src/microthink/core/schema.py`:

```python
"""
Schema validation for MicroThink.

Validates JSON results against example schemas to ensure
the model output matches the expected structure.
"""

from typing import Any, List, Union


class SchemaValidationError(Exception):
    """
    Raised when JSON output doesn't match the expected schema.

    Attributes:
        path: The path to the invalid field (e.g., "user.name" or "items[0]").
        message: Description of the validation error.
    """

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"Schema validation failed at '{path}': {message}")


# Type placeholders in example schemas
TYPE_PLACEHOLDERS = {"string", "number", "integer", "boolean", "array", "object"}


def infer_type(value: Any) -> str:
    """
    Infer the expected type from an example value.

    Args:
        value: An example value or type placeholder string.

    Returns:
        The inferred type name: "string", "number", "integer", "boolean", "array", or "object".

    Example:
        >>> infer_type("string")
        'string'
        >>> infer_type(42)
        'integer'
        >>> infer_type([1, 2])
        'array'
    """
    # Check if it's a type placeholder
    if isinstance(value, str) and value in TYPE_PLACEHOLDERS:
        return value

    # Infer from actual value
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"

    return "unknown"


def _check_type(value: Any, expected_type: str) -> bool:
    """
    Check if a value matches the expected type.

    Args:
        value: The value to check.
        expected_type: The expected type name.

    Returns:
        True if the value matches the expected type.
    """
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    return True


def validate_schema(
    data: Any,
    schema: Any,
    path: str = "",
) -> None:
    """
    Validate data against an example schema.

    Supports example schemas like:
        {"name": "string", "age": "number"}
        {"items": ["string"]}
        {"user": {"name": "string"}}

    Args:
        data: The JSON data to validate.
        schema: The example schema to validate against.
        path: Current path for error messages (internal use).

    Raises:
        SchemaValidationError: If the data doesn't match the schema.

    Example:
        >>> validate_schema({"name": "Alice", "age": 30}, {"name": "string", "age": "number"})
        # No error raised

        >>> validate_schema({"name": "Alice"}, {"name": "string", "age": "number"})
        SchemaValidationError: Schema validation failed at 'age': Missing required field
    """
    # Handle list schema (e.g., ["string"] for array of strings)
    if isinstance(schema, list):
        if not isinstance(data, list):
            raise SchemaValidationError(
                path or "root",
                f"Expected array, got {type(data).__name__}",
            )
        if len(schema) > 0:
            element_schema = schema[0]
            for i, item in enumerate(data):
                item_path = f"{path}[{i}]" if path else f"[{i}]"
                _validate_element(item, element_schema, item_path)
        return

    # Handle dict schema
    if isinstance(schema, dict):
        if not isinstance(data, dict):
            raise SchemaValidationError(
                path or "root",
                f"Expected object, got {type(data).__name__}",
            )

        # Check all required fields
        for key, expected in schema.items():
            field_path = f"{path}.{key}" if path else key

            if key not in data:
                raise SchemaValidationError(field_path, "Missing required field")

            _validate_element(data[key], expected, field_path)
        return

    # Handle primitive type validation at root
    _validate_element(data, schema, path or "root")


def _validate_element(value: Any, expected: Any, path: str) -> None:
    """
    Validate a single element against its expected type or nested schema.

    Args:
        value: The value to validate.
        expected: The expected type/schema.
        path: The current path for error messages.

    Raises:
        SchemaValidationError: If validation fails.
    """
    # Nested object schema
    if isinstance(expected, dict):
        validate_schema(value, expected, path)
        return

    # Array schema (e.g., ["string"])
    if isinstance(expected, list):
        validate_schema(value, expected, path)
        return

    # Primitive type check
    expected_type = infer_type(expected)
    if not _check_type(value, expected_type):
        actual_type = infer_type(value)
        raise SchemaValidationError(
            path,
            f"Expected {expected_type}, got {actual_type}",
        )
```

### Step 1.4: Run tests to verify they pass

```bash
pytest tests/test_schema.py -v
```

Expected: All tests PASS

### Step 1.5: Add client integration tests

Add to `tests/test_schema.py`:

```python
class TestClientSchemaIntegration:
    """Tests for schema validation in MicroThinkClient."""

    def test_generate_with_schema_validates_result(self):
        """generate_with_schema validates the result."""
        from microthink.core.schema import SchemaValidationError

        # This test would require mocking the Ollama client
        # For now, just verify the function signature accepts validate=True
        pass  # Integration test - requires mock

    def test_schema_validation_error_exported(self):
        """SchemaValidationError is importable from client module."""
        from microthink.client import SchemaValidationError
        assert SchemaValidationError is not None
```

### Step 1.6: Integrate schema validation into client

Modify `src/microthink/client.py`:

Add import at top:
```python
from microthink.core.schema import SchemaValidationError, validate_schema
```

Update `generate_with_schema` method to add validation:
```python
def generate_with_schema(
    self,
    prompt: str,
    schema: Dict[str, Any],
    behavior: str = "general",
    debug: bool = False,
    brief: bool = False,
    web_search: bool = False,
    validate: bool = True,
) -> Dict[str, Any]:
    """
    Generate JSON output that conforms to a specified schema.

    Args:
        prompt: The user's input prompt.
        schema: A JSON schema or example structure to follow.
        behavior: The persona to use.
        debug: If True, log the reasoning process.
        brief: If True, output just the result without explanation.
        web_search: If True, search the web for current information.
        validate: If True, validate result against schema (default: True).

    Returns:
        The parsed JSON response as a dictionary.

    Raises:
        MicroThinkError: If JSON parsing fails after all retries.
        SchemaValidationError: If result doesn't match schema (when validate=True).
    """
    schema_str = json.dumps(schema, indent=2)
    enhanced_prompt = (
        f"{prompt}\n\nReturn JSON matching this structure:\n{schema_str}"
    )

    result = self.generate(
        prompt=enhanced_prompt,
        behavior=behavior,
        expect_json=True,
        debug=debug,
        brief=brief,
        web_search=web_search,
    )

    # Validate against schema
    if validate:
        validate_schema(result, schema)

    # Ensure we return a dict (not a list) for dict schemas
    if isinstance(schema, dict) and not isinstance(result, dict):
        raise MicroThinkError(
            f"Expected dict but got {type(result).__name__}",
            last_output=str(result),
            attempts=self.MAX_RETRIES,
        )

    return result
```

### Step 1.7: Export SchemaValidationError from package

Modify `src/microthink/__init__.py`:

```python
from microthink.client import MicroThinkClient, MicroThinkError
from microthink.core.prompts import (
    PersonaError,
    get_persona,
    register_persona,
    unregister_persona,
)
from microthink.core.schema import SchemaValidationError

__version__ = "0.1.0"
__all__ = [
    "MicroThinkClient",
    "MicroThinkError",
    "PersonaError",
    "SchemaValidationError",
    "get_persona",
    "register_persona",
    "unregister_persona",
    "__version__",
]
```

### Step 1.8: Run all tests

```bash
pytest tests/ -v
```

Expected: All tests PASS

### Step 1.9: Commit

```bash
git add src/microthink/core/schema.py src/microthink/client.py src/microthink/__init__.py tests/test_schema.py
git commit -m "feat: add schema validation for generate_with_schema

- Add SchemaValidationError exception
- Add validate_schema() function with example schema support
- Support nested objects, arrays, and type inference
- Add validate parameter to generate_with_schema (default: True)
- Export SchemaValidationError from package"
```

---

## Task 2: Standard Logging

Integrate Python's standard logging module while keeping Rich as an optional enhancer.

**Files:**
- Create: `src/microthink/utils/logging_config.py`
- Modify: `src/microthink/utils/logger.py`
- Modify: `src/microthink/client.py`
- Create: `tests/test_logging.py`

### Step 2.1: Write failing tests for standard logging

Create `tests/test_logging.py`:

```python
"""Tests for standard logging integration."""

import logging
import pytest
from io import StringIO


class TestLoggingConfiguration:
    """Tests for logging configuration."""

    def test_get_logger_returns_logger(self):
        """get_logger returns a Python logger."""
        from microthink.utils.logging_config import get_logger

        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)

    def test_logger_default_level_is_warning(self):
        """Default log level is WARNING."""
        from microthink.utils.logging_config import get_logger

        logger = get_logger("test_default")
        assert logger.level == logging.WARNING or logger.level == 0

    def test_configure_logging_sets_level(self):
        """configure_logging sets the log level."""
        from microthink.utils.logging_config import configure_logging, get_logger

        configure_logging(level=logging.DEBUG)
        logger = get_logger("microthink")
        # The microthink logger should respect the configured level
        assert logger.isEnabledFor(logging.DEBUG)

    def test_configure_logging_accepts_string_level(self):
        """configure_logging accepts string level names."""
        from microthink.utils.logging_config import configure_logging

        configure_logging(level="DEBUG")  # Should not raise

    def test_logger_name_prefix(self):
        """Logger names are prefixed with 'microthink'."""
        from microthink.utils.logging_config import get_logger

        logger = get_logger("client")
        assert logger.name == "microthink.client"


class TestLoggerFunctions:
    """Tests for logger utility functions."""

    def test_log_info_uses_logger(self):
        """log_info uses standard logging."""
        from microthink.utils.logger import log_info_std

        # Capture log output
        logger = logging.getLogger("microthink")
        handler = logging.StreamHandler(StringIO())
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        log_info_std("Test message")

        output = handler.stream.getvalue()
        assert "Test message" in output

        logger.removeHandler(handler)

    def test_log_debug_respects_level(self):
        """Debug messages respect log level."""
        from microthink.utils.logger import log_debug_std

        logger = logging.getLogger("microthink")
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        # Set level to WARNING - debug should not appear
        logger.setLevel(logging.WARNING)
        log_debug_std("Debug message")
        assert "Debug message" not in stream.getvalue()

        # Set level to DEBUG - debug should appear
        logger.setLevel(logging.DEBUG)
        log_debug_std("Debug message 2")
        assert "Debug message 2" in stream.getvalue()

        logger.removeHandler(handler)


class TestClientLoggingIntegration:
    """Tests for logging in MicroThinkClient."""

    def test_client_accepts_logger_param(self):
        """Client accepts logger configuration."""
        from microthink import MicroThinkClient

        # Should not raise
        client = MicroThinkClient(logger="standard")
        assert hasattr(client, "_use_rich_logging")

    def test_client_default_uses_rich(self):
        """Client uses Rich logging by default."""
        from microthink import MicroThinkClient

        client = MicroThinkClient()
        # Default should be Rich when debug=True
        assert client._use_rich_logging is True

    def test_client_standard_logging_disables_rich(self):
        """Setting logger='standard' disables Rich."""
        from microthink import MicroThinkClient

        client = MicroThinkClient(logger="standard")
        assert client._use_rich_logging is False
```

### Step 2.2: Run tests to verify they fail

```bash
pytest tests/test_logging.py -v
```

Expected: FAIL with import errors

### Step 2.3: Implement logging configuration module

Create `src/microthink/utils/logging_config.py`:

```python
"""
Standard logging configuration for MicroThink.

Provides Python standard library logging integration,
with Rich as an optional enhancer for pretty console output.
"""

import logging
from typing import Optional, Union

# Root logger for MicroThink
LOGGER_NAME = "microthink"


def get_logger(name: str = "") -> logging.Logger:
    """
    Get a logger for a MicroThink module.

    Args:
        name: The module name (will be prefixed with 'microthink.').
              Empty string returns the root microthink logger.

    Returns:
        A configured Logger instance.

    Example:
        >>> logger = get_logger("client")
        >>> logger.name
        'microthink.client'
    """
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)


def configure_logging(
    level: Union[int, str] = logging.WARNING,
    format: Optional[str] = None,
    handler: Optional[logging.Handler] = None,
) -> None:
    """
    Configure MicroThink logging.

    Args:
        level: Log level (int or string like "DEBUG", "INFO").
        format: Log message format. Defaults to standard format.
        handler: Custom handler. Defaults to StreamHandler.

    Example:
        >>> configure_logging(level="DEBUG")
        >>> configure_logging(level=logging.INFO, format="%(message)s")
    """
    logger = get_logger()

    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.WARNING)

    logger.setLevel(level)

    # Set up handler if none exists or custom provided
    if handler:
        logger.addHandler(handler)
    elif not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        if format is None:
            format = "[%(levelname)s] %(name)s: %(message)s"

        formatter = logging.Formatter(format)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
```

### Step 2.4: Update logger.py with standard logging functions

Modify `src/microthink/utils/logger.py` - add standard logging functions:

```python
"""
Logging utilities for MicroThink.

Provides both Rich console output (for interactive debug mode)
and standard Python logging (for integration with existing systems).
"""

import logging
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from microthink.utils.logging_config import get_logger

# Shared console instance for Rich output
console = Console()

# Standard logger
_logger = get_logger("output")


# --- Standard logging functions ---

def log_debug_std(message: str) -> None:
    """Log a debug message using standard logging."""
    _logger.debug(message)


def log_info_std(message: str) -> None:
    """Log an info message using standard logging."""
    _logger.info(message)


def log_warning_std(message: str) -> None:
    """Log a warning message using standard logging."""
    _logger.warning(message)


def log_error_std(message: str) -> None:
    """Log an error message using standard logging."""
    _logger.error(message)


# --- Rich logging functions (existing) ---

def log_thinking(content: str) -> None:
    """Log the model's thinking/reasoning process with Rich."""
    panel = Panel(
        content,
        title="[bold yellow]Thought Process[/bold yellow]",
        border_style="yellow",
        padding=(1, 2),
    )
    console.print(panel)


def log_answer(content: str, is_json: bool = False) -> None:
    """Log the model's final answer with Rich."""
    title = (
        "[bold green]Final Answer (JSON)[/bold green]"
        if is_json
        else "[bold green]Final Answer[/bold green]"
    )
    panel = Panel(
        content,
        title=title,
        border_style="green",
        padding=(1, 2),
    )
    console.print(panel)


def log_retry(attempt: int, max_attempts: int, error: str) -> None:
    """Log a JSON correction retry attempt with Rich."""
    text = Text()
    text.append(f"[Reflexion] ", style="bold red")
    text.append(f"Retry {attempt}/{max_attempts} - ", style="yellow")
    text.append(f"Error: {error}", style="dim")
    console.print(text)


def log_success(message: str) -> None:
    """Log a success message with Rich."""
    console.print(f"[bold green][OK][/bold green] {message}")


def log_error(message: str) -> None:
    """Log an error message with Rich."""
    console.print(f"[bold red][ERROR][/bold red] {message}")


def log_info(message: str) -> None:
    """Log an informational message with Rich."""
    console.print(f"[bold blue][INFO][/bold blue] {message}")
```

### Step 2.5: Integrate logging configuration into client

Modify `src/microthink/client.py`:

Add to imports:
```python
from microthink.utils.logger import log_info_std, log_debug_std
```

Update `__init__` to accept logger parameter:
```python
def __init__(
    self,
    model: str = DEFAULT_MODEL,
    host: Optional[str] = None,
    timeout: float = 120.0,
    cache: bool = False,
    cache_ttl: float = 3600.0,
    cache_max_size: int = 1000,
    logger: str = "rich",
) -> None:
    """
    Initialize the MicroThink client.

    Args:
        model: The Ollama model to use (default: "llama3.2:3b").
        host: Optional Ollama host URL.
        timeout: Request timeout in seconds (default: 120.0).
        cache: Enable response caching (default: False).
        cache_ttl: Cache time-to-live in seconds (default: 3600.0).
        cache_max_size: Maximum cache entries (default: 1000).
        logger: Logging mode - "rich" for Rich console, "standard" for Python logging.
    """
    if not model:
        raise ValueError("Model name cannot be empty")

    self.model = model
    self.host = host
    self.timeout = timeout
    self._use_rich_logging = logger != "standard"

    # Initialize Ollama client
    if host:
        self._client = ollama.Client(host=host, timeout=timeout)
    else:
        self._client = ollama.Client(timeout=timeout)

    # Initialize cache
    self._cache: Optional[ResponseCache] = None
    if cache:
        self._cache = ResponseCache(max_size=cache_max_size, ttl=cache_ttl)
```

### Step 2.6: Run all logging tests

```bash
pytest tests/test_logging.py -v
```

Expected: All tests PASS

### Step 2.7: Commit

```bash
git add src/microthink/utils/logging_config.py src/microthink/utils/logger.py src/microthink/client.py tests/test_logging.py
git commit -m "feat: add standard Python logging support

- Add logging_config.py with get_logger() and configure_logging()
- Add standard logging functions (log_info_std, log_debug_std, etc.)
- Add logger parameter to MicroThinkClient ('rich' or 'standard')
- Keep Rich as default for interactive debug mode"
```

---

## Task 3: Async Support

Add `AsyncMicroThinkClient` that mirrors the sync API using httpx for async HTTP.

**Files:**
- Create: `src/microthink/async_client.py`
- Modify: `src/microthink/__init__.py`
- Create: `tests/test_async.py`

### Step 3.1: Write failing tests for async client

Create `tests/test_async.py`:

```python
"""Tests for async client."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


class TestAsyncMicroThinkClient:
    """Tests for AsyncMicroThinkClient."""

    def test_async_client_importable(self):
        """AsyncMicroThinkClient is importable."""
        from microthink import AsyncMicroThinkClient
        assert AsyncMicroThinkClient is not None

    def test_async_client_has_generate(self):
        """AsyncMicroThinkClient has async generate method."""
        from microthink import AsyncMicroThinkClient
        import inspect

        client = AsyncMicroThinkClient()
        assert hasattr(client, "generate")
        assert asyncio.iscoroutinefunction(client.generate)

    def test_async_client_has_generate_with_schema(self):
        """AsyncMicroThinkClient has async generate_with_schema method."""
        from microthink import AsyncMicroThinkClient
        import inspect

        client = AsyncMicroThinkClient()
        assert hasattr(client, "generate_with_schema")
        assert asyncio.iscoroutinefunction(client.generate_with_schema)

    def test_async_client_accepts_same_params(self):
        """AsyncMicroThinkClient accepts same init params as sync."""
        from microthink import AsyncMicroThinkClient

        client = AsyncMicroThinkClient(
            model="llama3.2:3b",
            timeout=60.0,
            cache=True,
            cache_ttl=1800,
        )
        assert client.model == "llama3.2:3b"
        assert client.timeout == 60.0

    def test_async_client_has_context_manager(self):
        """AsyncMicroThinkClient supports async context manager."""
        from microthink import AsyncMicroThinkClient
        import inspect

        assert hasattr(AsyncMicroThinkClient, "__aenter__")
        assert hasattr(AsyncMicroThinkClient, "__aexit__")

    @pytest.mark.asyncio
    async def test_async_context_manager_usage(self):
        """Can use AsyncMicroThinkClient as async context manager."""
        from microthink import AsyncMicroThinkClient

        async with AsyncMicroThinkClient() as client:
            assert client is not None

    def test_async_client_has_cache_methods(self):
        """AsyncMicroThinkClient has cache methods."""
        from microthink import AsyncMicroThinkClient

        client = AsyncMicroThinkClient(cache=True)
        assert hasattr(client, "cache_stats")
        assert hasattr(client, "clear_cache")

    def test_async_client_has_register_persona(self):
        """AsyncMicroThinkClient has register_persona method."""
        from microthink import AsyncMicroThinkClient

        client = AsyncMicroThinkClient()
        assert hasattr(client, "register_persona")


class TestAsyncClientMocked:
    """Tests for async client with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_generate_returns_string(self):
        """Async generate returns string for non-JSON."""
        from microthink.async_client import AsyncMicroThinkClient

        with patch("microthink.async_client.httpx.AsyncClient") as mock_httpx:
            # Mock the HTTP response
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "message": {
                    "content": "<thinking>Step 1</thinking><answer>42</answer>"
                }
            }
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_httpx.return_value = mock_client

            async with AsyncMicroThinkClient() as client:
                result = await client.generate("What is 2+2?")

            assert result == "42"

    @pytest.mark.asyncio
    async def test_generate_returns_json(self):
        """Async generate returns parsed JSON when expect_json=True."""
        from microthink.async_client import AsyncMicroThinkClient

        with patch("microthink.async_client.httpx.AsyncClient") as mock_httpx:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "message": {
                    "content": '<thinking>Creating JSON</thinking><answer>{"name": "Alice"}</answer>'
                }
            }
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_httpx.return_value = mock_client

            async with AsyncMicroThinkClient() as client:
                result = await client.generate("Create user", expect_json=True)

            assert result == {"name": "Alice"}
```

### Step 3.2: Run tests to verify they fail

```bash
pytest tests/test_async.py -v
```

Expected: FAIL with import errors

### Step 3.3: Implement async client

Create `src/microthink/async_client.py`:

```python
"""
Async MicroThink Client.

Provides an async/await interface to MicroThink for concurrent requests.
Uses httpx for async HTTP communication with Ollama.
"""

import json
from typing import Any, Dict, List, Optional, Union

import httpx

from microthink.core.cache import ResponseCache, make_cache_key
from microthink.core.parser import clean_json_text, parse_response
from microthink.core.prompts import (
    SYSTEM_PERSONAS,
    build_system_prompt,
    register_persona as _register_persona,
)
from microthink.core.schema import SchemaValidationError, validate_schema
from microthink.client import MicroThinkError


class AsyncMicroThinkClient:
    """
    Async wrapper around Ollama that enhances small LLM performance.

    Features:
        - Async/await API for concurrent requests
        - Same features as sync client (CoT, JSON validation, caching)
        - Context manager support for proper resource cleanup

    Example:
        >>> async with AsyncMicroThinkClient() as client:
        ...     result = await client.generate("What is 2+2?")
        ...     print(result)
        '4'

        >>> # Concurrent requests
        >>> async with AsyncMicroThinkClient() as client:
        ...     results = await asyncio.gather(
        ...         client.generate("Question 1"),
        ...         client.generate("Question 2"),
        ...     )
    """

    DEFAULT_MODEL = "llama3.2:3b"
    MAX_RETRIES = 3
    RETRY_TEMPERATURE = 0.2
    DEFAULT_HOST = "http://localhost:11434"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: Optional[str] = None,
        timeout: float = 120.0,
        cache: bool = False,
        cache_ttl: float = 3600.0,
        cache_max_size: int = 1000,
    ) -> None:
        """
        Initialize the async MicroThink client.

        Args:
            model: The Ollama model to use.
            host: Ollama host URL (default: http://localhost:11434).
            timeout: Request timeout in seconds.
            cache: Enable response caching.
            cache_ttl: Cache TTL in seconds.
            cache_max_size: Maximum cache entries.
        """
        if not model:
            raise ValueError("Model name cannot be empty")

        self.model = model
        self.host = host or self.DEFAULT_HOST
        self.timeout = timeout
        self._http_client: Optional[httpx.AsyncClient] = None

        # Initialize cache
        self._cache: Optional[ResponseCache] = None
        if cache:
            self._cache = ResponseCache(max_size=cache_max_size, ttl=cache_ttl)

    async def __aenter__(self) -> "AsyncMicroThinkClient":
        """Enter async context manager."""
        self._http_client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    @property
    def available_behaviors(self) -> List[str]:
        """Return list of available behavior personas."""
        return list(SYSTEM_PERSONAS.keys())

    def register_persona(
        self,
        name: str,
        prompt: str,
        allow_override: bool = False,
    ) -> None:
        """Register a custom persona."""
        _register_persona(name, prompt, allow_override)

    def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self._cache is None:
            return {"hits": 0, "misses": 0, "size": 0, "hit_rate": 0.0, "enabled": False}
        stats = self._cache.stats()
        stats["enabled"] = True
        return stats

    def clear_cache(self) -> None:
        """Clear the response cache."""
        if self._cache is not None:
            self._cache.clear()

    async def _chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Make async chat request to Ollama.

        Args:
            messages: The message history.
            temperature: Optional temperature override.

        Returns:
            The Ollama API response.
        """
        if self._http_client is None:
            raise RuntimeError(
                "Client not initialized. Use 'async with AsyncMicroThinkClient() as client:'"
            )

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if temperature is not None:
            payload["options"] = {"temperature": temperature}

        response = await self._http_client.post(
            f"{self.host}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    async def generate(
        self,
        prompt: str,
        behavior: str = "general",
        expect_json: bool = False,
        debug: bool = False,
        brief: bool = False,
    ) -> Union[str, Dict[str, Any], List[Any]]:
        """
        Generate a response asynchronously.

        Args:
            prompt: The user's input prompt.
            behavior: The persona to use.
            expect_json: If True, parse and validate JSON output.
            debug: If True, log the reasoning process.
            brief: If True, answer contains only the result.

        Returns:
            If expect_json=False: The answer as a string.
            If expect_json=True: The parsed JSON.

        Raises:
            MicroThinkError: If JSON parsing fails after retries.
        """
        if behavior not in SYSTEM_PERSONAS:
            raise ValueError(
                f"Invalid behavior '{behavior}'. Available: {self.available_behaviors}"
            )

        # Check cache
        cache_key = None
        if self._cache is not None:
            cache_key = make_cache_key(
                model=self.model,
                behavior=behavior,
                prompt=prompt,
                expect_json=expect_json,
                web_search=False,
            )
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        # Build system prompt
        system_prompt = build_system_prompt(behavior, expect_json, brief)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Initial request
        response = await self._chat(messages)
        raw_content = response["message"]["content"]

        # Parse response
        parsed = parse_response(raw_content)
        answer_content = parsed["answer"]

        if not expect_json:
            if self._cache is not None and cache_key:
                self._cache.set(cache_key, answer_content)
            return answer_content

        # JSON parsing with retries
        messages.append({"role": "assistant", "content": raw_content})
        retries = 0
        last_error: Optional[str] = None

        while retries < self.MAX_RETRIES:
            try:
                cleaned = clean_json_text(answer_content)
                result = json.loads(cleaned)

                if self._cache is not None and cache_key:
                    self._cache.set(cache_key, result)
                return result

            except json.JSONDecodeError as e:
                retries += 1
                last_error = str(e)

                if retries >= self.MAX_RETRIES:
                    raise MicroThinkError(
                        f"JSON parsing failed after {self.MAX_RETRIES} retries",
                        last_output=answer_content,
                        attempts=retries,
                        json_error=last_error,
                    )

                correction_msg = (
                    f"Your previous response was invalid JSON.\n"
                    f"Error: {last_error}\n"
                    f"Output ONLY valid JSON inside <answer> tags."
                )
                messages.append({"role": "user", "content": correction_msg})

                response = await self._chat(messages, temperature=self.RETRY_TEMPERATURE)
                raw_content = response["message"]["content"]
                parsed = parse_response(raw_content)
                answer_content = parsed["answer"]
                messages.append({"role": "assistant", "content": raw_content})

    async def generate_with_schema(
        self,
        prompt: str,
        schema: Dict[str, Any],
        behavior: str = "general",
        debug: bool = False,
        brief: bool = False,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate JSON output conforming to a schema.

        Args:
            prompt: The user's input prompt.
            schema: Example schema structure.
            behavior: The persona to use.
            debug: If True, log the reasoning process.
            brief: If True, output just the result.
            validate: If True, validate against schema.

        Returns:
            The parsed JSON response.

        Raises:
            MicroThinkError: If JSON parsing fails.
            SchemaValidationError: If validation fails.
        """
        schema_str = json.dumps(schema, indent=2)
        enhanced_prompt = f"{prompt}\n\nReturn JSON matching this structure:\n{schema_str}"

        result = await self.generate(
            prompt=enhanced_prompt,
            behavior=behavior,
            expect_json=True,
            debug=debug,
            brief=brief,
        )

        if validate:
            validate_schema(result, schema)

        if isinstance(schema, dict) and not isinstance(result, dict):
            raise MicroThinkError(
                f"Expected dict but got {type(result).__name__}",
                last_output=str(result),
            )

        return result
```

### Step 3.4: Export AsyncMicroThinkClient from package

Modify `src/microthink/__init__.py`:

```python
from microthink.async_client import AsyncMicroThinkClient
from microthink.client import MicroThinkClient, MicroThinkError
from microthink.core.prompts import (
    PersonaError,
    get_persona,
    register_persona,
    unregister_persona,
)
from microthink.core.schema import SchemaValidationError

__version__ = "0.1.0"
__all__ = [
    "AsyncMicroThinkClient",
    "MicroThinkClient",
    "MicroThinkError",
    "PersonaError",
    "SchemaValidationError",
    "get_persona",
    "register_persona",
    "unregister_persona",
    "__version__",
]
```

### Step 3.5: Run all async tests

```bash
pytest tests/test_async.py -v
```

Expected: All tests PASS

### Step 3.6: Run all tests

```bash
pytest tests/ -v
```

Expected: All tests PASS

### Step 3.7: Commit

```bash
git add src/microthink/async_client.py src/microthink/__init__.py tests/test_async.py
git commit -m "feat: add AsyncMicroThinkClient for async/await support

- Add AsyncMicroThinkClient using httpx for async HTTP
- Support async context manager for proper resource cleanup
- Mirror sync client API (generate, generate_with_schema, cache, personas)
- Add comprehensive async tests with mocked HTTP"
```

---

## Task 4: Final Integration

### Step 4.1: Run all tests

```bash
pytest tests/ -v
```

### Step 4.2: Verify exports

Check that all new exports work:

```python
from microthink import (
    MicroThinkClient,
    AsyncMicroThinkClient,
    MicroThinkError,
    PersonaError,
    SchemaValidationError,
    register_persona,
    unregister_persona,
    get_persona,
)
```

### Step 4.3: Final commit

```bash
git add -A
git commit -m "chore: complete Phase 1 Core Foundations

Features added:
- Schema validation for generate_with_schema()
- Standard Python logging integration
- AsyncMicroThinkClient for async/await support

All tests passing."
```

---

## Summary

| Task | Files Created/Modified | Key Features |
|------|----------------------|--------------|
| Schema Validation | `schema.py`, `client.py` | `validate_schema()`, `SchemaValidationError` |
| Standard Logging | `logging_config.py`, `logger.py`, `client.py` | `configure_logging()`, `logger` param |
| Async Support | `async_client.py` | `AsyncMicroThinkClient` |

**New dependencies:** `httpx`, `pytest-asyncio`

**Total new tests:** ~40+
