# Quick Wins Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement three high-value, independent features: Custom Personas, Response Caching, and Better Error Messages.

**Architecture:** Each feature is self-contained. Custom Personas extends `prompts.py` and `client.py`. Response Caching adds a new `cache.py` module. Better Error Messages enhances `MicroThinkError` with diagnostic context.

**Tech Stack:** Python 3.9+, pytest for testing, no new dependencies required.

---

## Prerequisites

Before starting, set up the test infrastructure:

```bash
# Create tests directory
mkdir -p tests

# Install dev dependencies
pip install -e ".[dev]"
```

---

## Task 1: Custom Personas

Allow users to register their own personas beyond the 4 built-in ones.

**Files:**
- Modify: `src/microthink/core/prompts.py`
- Modify: `src/microthink/client.py`
- Create: `tests/test_personas.py`

### Step 1.1: Write the failing test for register_persona

Create `tests/test_personas.py`:

```python
"""Tests for custom persona registration."""

import pytest
from microthink.core.prompts import (
    SYSTEM_PERSONAS,
    register_persona,
    get_persona,
    unregister_persona,
    PersonaError,
)


class TestRegisterPersona:
    """Tests for registering custom personas."""

    def test_register_new_persona(self):
        """Can register a new custom persona."""
        register_persona(
            name="sql_expert",
            prompt="You are an expert SQL developer.",
        )
        assert "sql_expert" in SYSTEM_PERSONAS
        assert SYSTEM_PERSONAS["sql_expert"] == "You are an expert SQL developer."
        # Cleanup
        unregister_persona("sql_expert")

    def test_register_persona_rejects_empty_name(self):
        """Cannot register persona with empty name."""
        with pytest.raises(PersonaError, match="name cannot be empty"):
            register_persona(name="", prompt="Some prompt")

    def test_register_persona_rejects_empty_prompt(self):
        """Cannot register persona with empty prompt."""
        with pytest.raises(PersonaError, match="prompt cannot be empty"):
            register_persona(name="test", prompt="")

    def test_register_persona_rejects_builtin_override(self):
        """Cannot override built-in personas by default."""
        with pytest.raises(PersonaError, match="Cannot override built-in persona"):
            register_persona(name="general", prompt="Override attempt")

    def test_register_persona_allows_override_with_flag(self):
        """Can override built-in personas with allow_override=True."""
        original = SYSTEM_PERSONAS["general"]
        register_persona(
            name="general",
            prompt="Custom general",
            allow_override=True,
        )
        assert SYSTEM_PERSONAS["general"] == "Custom general"
        # Restore original
        SYSTEM_PERSONAS["general"] = original


class TestGetPersona:
    """Tests for retrieving personas."""

    def test_get_builtin_persona(self):
        """Can retrieve built-in personas."""
        persona = get_persona("coder")
        assert "Python programmer" in persona

    def test_get_custom_persona(self):
        """Can retrieve custom personas."""
        register_persona(name="test_persona", prompt="Test prompt")
        assert get_persona("test_persona") == "Test prompt"
        unregister_persona("test_persona")

    def test_get_unknown_persona_raises(self):
        """Raises PersonaError for unknown persona."""
        with pytest.raises(PersonaError, match="Unknown persona"):
            get_persona("nonexistent")


class TestUnregisterPersona:
    """Tests for unregistering personas."""

    def test_unregister_custom_persona(self):
        """Can unregister custom personas."""
        register_persona(name="temp", prompt="Temporary")
        unregister_persona("temp")
        assert "temp" not in SYSTEM_PERSONAS

    def test_unregister_builtin_raises(self):
        """Cannot unregister built-in personas."""
        with pytest.raises(PersonaError, match="Cannot unregister built-in"):
            unregister_persona("general")

    def test_unregister_unknown_raises(self):
        """Raises PersonaError for unknown persona."""
        with pytest.raises(PersonaError, match="Unknown persona"):
            unregister_persona("nonexistent")
```

### Step 1.2: Run test to verify it fails

```bash
pytest tests/test_personas.py -v
```

Expected: FAIL with `ImportError: cannot import name 'register_persona'`

### Step 1.3: Implement persona management in prompts.py

Modify `src/microthink/core/prompts.py` - add after the `SYSTEM_PERSONAS` dict:

```python
# Track which personas are built-in (cannot be removed)
_BUILTIN_PERSONAS = frozenset(SYSTEM_PERSONAS.keys())


class PersonaError(Exception):
    """Raised when persona operations fail."""
    pass


def register_persona(
    name: str,
    prompt: str,
    allow_override: bool = False,
) -> None:
    """
    Register a custom persona.

    Args:
        name: Unique name for the persona.
        prompt: The system prompt for this persona.
        allow_override: If True, allows overriding built-in personas.

    Raises:
        PersonaError: If name/prompt is empty or trying to override built-in.

    Example:
        >>> register_persona(
        ...     name="sql_expert",
        ...     prompt="You are an expert SQL developer.",
        ... )
    """
    if not name or not name.strip():
        raise PersonaError("Persona name cannot be empty")
    if not prompt or not prompt.strip():
        raise PersonaError("Persona prompt cannot be empty")
    if name in _BUILTIN_PERSONAS and not allow_override:
        raise PersonaError(
            f"Cannot override built-in persona '{name}'. "
            f"Use allow_override=True to override."
        )

    SYSTEM_PERSONAS[name] = prompt.strip()


def get_persona(name: str) -> str:
    """
    Get a persona prompt by name.

    Args:
        name: The persona name.

    Returns:
        The persona's system prompt.

    Raises:
        PersonaError: If persona doesn't exist.
    """
    if name not in SYSTEM_PERSONAS:
        available = ", ".join(sorted(SYSTEM_PERSONAS.keys()))
        raise PersonaError(f"Unknown persona '{name}'. Available: {available}")
    return SYSTEM_PERSONAS[name]


def unregister_persona(name: str) -> None:
    """
    Remove a custom persona.

    Args:
        name: The persona name to remove.

    Raises:
        PersonaError: If persona is built-in or doesn't exist.
    """
    if name in _BUILTIN_PERSONAS:
        raise PersonaError(f"Cannot unregister built-in persona '{name}'")
    if name not in SYSTEM_PERSONAS:
        raise PersonaError(f"Unknown persona '{name}'")

    del SYSTEM_PERSONAS[name]
```

### Step 1.4: Update prompts.py imports and exports

At the top of `src/microthink/core/prompts.py`, the file should export these. No changes needed to imports, but ensure the module structure allows importing `PersonaError`.

### Step 1.5: Run tests to verify they pass

```bash
pytest tests/test_personas.py -v
```

Expected: All tests PASS

### Step 1.6: Add client integration test

Add to `tests/test_personas.py`:

```python
class TestClientPersonaIntegration:
    """Tests for using custom personas with MicroThinkClient."""

    def test_client_register_persona_method(self):
        """Client exposes register_persona method."""
        from microthink import MicroThinkClient

        client = MicroThinkClient()
        client.register_persona(
            name="friendly_bot",
            prompt="You are a friendly and helpful assistant.",
        )
        assert "friendly_bot" in client.available_behaviors
        # Cleanup
        unregister_persona("friendly_bot")

    def test_client_available_behaviors_includes_custom(self):
        """available_behaviors property includes custom personas."""
        from microthink import MicroThinkClient

        register_persona(name="custom1", prompt="Custom persona 1")
        client = MicroThinkClient()
        assert "custom1" in client.available_behaviors
        unregister_persona("custom1")
```

### Step 1.7: Add register_persona method to client

Modify `src/microthink/client.py`:

Add import at top:
```python
from microthink.core.prompts import (
    SYSTEM_PERSONAS,
    build_system_prompt,
    register_persona as _register_persona,
    PersonaError,
)
```

Add method to `MicroThinkClient` class:
```python
def register_persona(
    self,
    name: str,
    prompt: str,
    allow_override: bool = False,
) -> None:
    """
    Register a custom persona for use with generate().

    Args:
        name: Unique name for the persona.
        prompt: The system prompt for this persona.
        allow_override: If True, allows overriding built-in personas.

    Example:
        >>> client = MicroThinkClient()
        >>> client.register_persona(
        ...     name="sql_expert",
        ...     prompt="You are an expert SQL developer.",
        ... )
        >>> client.generate("Write a query", behavior="sql_expert")
    """
    _register_persona(name, prompt, allow_override)
```

### Step 1.8: Export PersonaError from package

Modify `src/microthink/__init__.py`:

```python
from microthink.client import MicroThinkClient, MicroThinkError
from microthink.core.prompts import PersonaError, register_persona, unregister_persona

__version__ = "0.1.0"
__all__ = [
    "MicroThinkClient",
    "MicroThinkError",
    "PersonaError",
    "register_persona",
    "unregister_persona",
    "__version__",
]
```

### Step 1.9: Run all persona tests

```bash
pytest tests/test_personas.py -v
```

Expected: All tests PASS

### Step 1.10: Commit

```bash
git add src/microthink/core/prompts.py src/microthink/client.py src/microthink/__init__.py tests/test_personas.py
git commit -m "feat: add custom persona registration

- Add register_persona(), get_persona(), unregister_persona() to prompts.py
- Add PersonaError exception for persona-related errors
- Expose register_persona method on MicroThinkClient
- Protect built-in personas from accidental removal
- Export PersonaError from package"
```

---

## Task 2: Response Caching

Add LRU cache with TTL to avoid redundant model calls for identical prompts.

**Files:**
- Create: `src/microthink/core/cache.py`
- Modify: `src/microthink/client.py`
- Create: `tests/test_cache.py`

### Step 2.1: Write failing tests for cache module

Create `tests/test_cache.py`:

```python
"""Tests for response caching."""

import time
import pytest
from microthink.core.cache import ResponseCache


class TestResponseCache:
    """Tests for the ResponseCache class."""

    def test_cache_stores_and_retrieves(self):
        """Cache stores and retrieves values."""
        cache = ResponseCache(max_size=100, ttl=60)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_cache_returns_none_for_missing(self):
        """Cache returns None for missing keys."""
        cache = ResponseCache(max_size=100, ttl=60)
        assert cache.get("nonexistent") is None

    def test_cache_respects_ttl(self):
        """Cache entries expire after TTL."""
        cache = ResponseCache(max_size=100, ttl=0.1)  # 100ms TTL
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        time.sleep(0.15)  # Wait for expiry
        assert cache.get("key1") is None

    def test_cache_respects_max_size(self):
        """Cache evicts oldest entries when full."""
        cache = ResponseCache(max_size=2, ttl=60)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict key1
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_cache_clear(self):
        """Cache can be cleared."""
        cache = ResponseCache(max_size=100, ttl=60)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cache_stats(self):
        """Cache tracks hit/miss statistics."""
        cache = ResponseCache(max_size=100, ttl=60)
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("missing")  # Miss
        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(0.667, rel=0.01)

    def test_cache_disabled_when_ttl_zero(self):
        """Cache with ttl=0 acts as disabled."""
        cache = ResponseCache(max_size=100, ttl=0)
        cache.set("key1", "value1")
        assert cache.get("key1") is None  # Immediately expired


class TestCacheKeyGeneration:
    """Tests for cache key generation."""

    def test_make_cache_key(self):
        """Cache key is deterministic."""
        from microthink.core.cache import make_cache_key

        key1 = make_cache_key(
            model="llama3.2:3b",
            behavior="coder",
            prompt="Hello",
            expect_json=True,
            web_search=False,
        )
        key2 = make_cache_key(
            model="llama3.2:3b",
            behavior="coder",
            prompt="Hello",
            expect_json=True,
            web_search=False,
        )
        assert key1 == key2

    def test_different_prompts_different_keys(self):
        """Different prompts produce different keys."""
        from microthink.core.cache import make_cache_key

        key1 = make_cache_key("model", "general", "Hello", False, False)
        key2 = make_cache_key("model", "general", "World", False, False)
        assert key1 != key2

    def test_different_behaviors_different_keys(self):
        """Different behaviors produce different keys."""
        from microthink.core.cache import make_cache_key

        key1 = make_cache_key("model", "general", "Hello", False, False)
        key2 = make_cache_key("model", "coder", "Hello", False, False)
        assert key1 != key2
```

### Step 2.2: Run test to verify it fails

```bash
pytest tests/test_cache.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'microthink.core.cache'`

### Step 2.3: Implement cache module

Create `src/microthink/core/cache.py`:

```python
"""
Response caching for MicroThink.

Provides an LRU cache with TTL to avoid redundant model calls
for identical prompts.
"""

import hashlib
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Dict, Optional


def make_cache_key(
    model: str,
    behavior: str,
    prompt: str,
    expect_json: bool,
    web_search: bool,
) -> str:
    """
    Generate a deterministic cache key from request parameters.

    Args:
        model: The model name.
        behavior: The persona behavior.
        prompt: The user prompt.
        expect_json: Whether JSON output is expected.
        web_search: Whether web search is enabled.

    Returns:
        A hex digest string suitable as a cache key.
    """
    key_parts = f"{model}|{behavior}|{prompt}|{expect_json}|{web_search}"
    return hashlib.sha256(key_parts.encode()).hexdigest()


class ResponseCache:
    """
    Thread-safe LRU cache with TTL for caching model responses.

    Attributes:
        max_size: Maximum number of entries to store.
        ttl: Time-to-live in seconds (0 = disabled).

    Example:
        >>> cache = ResponseCache(max_size=1000, ttl=3600)
        >>> cache.set("key", "value")
        >>> cache.get("key")
        'value'
    """

    def __init__(self, max_size: int = 1000, ttl: float = 3600) -> None:
        """
        Initialize the cache.

        Args:
            max_size: Maximum entries to store (default: 1000).
            ttl: Time-to-live in seconds (default: 3600). Set to 0 to disable.
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from the cache.

        Args:
            key: The cache key.

        Returns:
            The cached value, or None if not found or expired.
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check TTL
            if self.ttl > 0 and time.time() - entry["timestamp"] > self.ttl:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry["value"]

    def set(self, key: str, value: Any) -> None:
        """
        Store a value in the cache.

        Args:
            key: The cache key.
            value: The value to store.
        """
        # If TTL is 0, caching is disabled
        if self.ttl <= 0:
            return

        with self._lock:
            # Remove if exists (to update timestamp)
            if key in self._cache:
                del self._cache[key]

            # Evict oldest if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)

            self._cache[key] = {
                "value": value,
                "timestamp": time.time(),
            }

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with hits, misses, size, and hit_rate.
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "hit_rate": hit_rate,
            }
```

### Step 2.4: Run tests to verify they pass

```bash
pytest tests/test_cache.py -v
```

Expected: All tests PASS

### Step 2.5: Write integration test for client caching

Add to `tests/test_cache.py`:

```python
class TestClientCacheIntegration:
    """Tests for caching in MicroThinkClient."""

    def test_client_accepts_cache_params(self):
        """Client accepts cache configuration."""
        from microthink import MicroThinkClient

        client = MicroThinkClient(
            cache=True,
            cache_ttl=3600,
            cache_max_size=500,
        )
        assert client._cache is not None
        assert client._cache.ttl == 3600
        assert client._cache.max_size == 500

    def test_client_cache_disabled_by_default(self):
        """Caching is disabled by default."""
        from microthink import MicroThinkClient

        client = MicroThinkClient()
        assert client._cache is None

    def test_client_cache_stats(self):
        """Client exposes cache statistics."""
        from microthink import MicroThinkClient

        client = MicroThinkClient(cache=True)
        stats = client.cache_stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats

    def test_client_cache_clear(self):
        """Client can clear the cache."""
        from microthink import MicroThinkClient

        client = MicroThinkClient(cache=True)
        client.clear_cache()  # Should not raise
```

### Step 2.6: Integrate caching into client

Modify `src/microthink/client.py`:

Add import:
```python
from microthink.core.cache import ResponseCache, make_cache_key
```

Update `__init__` method signature and body:
```python
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
    Initialize the MicroThink client.

    Args:
        model: The Ollama model to use (default: "llama3.2:3b").
        host: Optional Ollama host URL (default: uses Ollama default).
        timeout: Request timeout in seconds (default: 120.0).
        cache: Enable response caching (default: False).
        cache_ttl: Cache time-to-live in seconds (default: 3600).
        cache_max_size: Maximum cache entries (default: 1000).

    Raises:
        ValueError: If the model name is empty.
    """
    if not model:
        raise ValueError("Model name cannot be empty")

    self.model = model
    self.host = host
    self.timeout = timeout

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

Add cache methods to `MicroThinkClient`:
```python
def cache_stats(self) -> Dict[str, Any]:
    """
    Get cache statistics.

    Returns:
        Dict with hits, misses, size, and hit_rate.
        Returns empty stats if caching is disabled.
    """
    if self._cache is None:
        return {"hits": 0, "misses": 0, "size": 0, "hit_rate": 0.0, "enabled": False}
    stats = self._cache.stats()
    stats["enabled"] = True
    return stats

def clear_cache(self) -> None:
    """Clear the response cache."""
    if self._cache is not None:
        self._cache.clear()
```

Update `generate` method to use cache - add near the start after behavior validation:
```python
# Check cache first
cache_key = None
if self._cache is not None and not web_search:  # Don't cache web search results
    cache_key = make_cache_key(
        model=self.model,
        behavior=behavior,
        prompt=prompt,
        expect_json=expect_json,
        web_search=web_search,
    )
    cached = self._cache.get(cache_key)
    if cached is not None:
        if debug:
            log_info("Cache hit - returning cached response")
        return cached
```

And before returning, store in cache:
```python
# Store in cache before returning
if self._cache is not None and cache_key is not None:
    self._cache.set(cache_key, <result>)  # Replace <result> with actual return value
```

### Step 2.7: Run all cache tests

```bash
pytest tests/test_cache.py -v
```

Expected: All tests PASS

### Step 2.8: Commit

```bash
git add src/microthink/core/cache.py src/microthink/client.py tests/test_cache.py
git commit -m "feat: add response caching with TTL

- Add ResponseCache class with LRU eviction and TTL expiry
- Add make_cache_key() for deterministic cache keys
- Add cache, cache_ttl, cache_max_size params to MicroThinkClient
- Add cache_stats() and clear_cache() methods
- Skip caching for web_search=True (results change)"
```

---

## Task 3: Better Error Messages

Enhance `MicroThinkError` with diagnostic context and actionable suggestions.

**Files:**
- Modify: `src/microthink/client.py`
- Create: `tests/test_errors.py`

### Step 3.1: Write failing tests for enhanced errors

Create `tests/test_errors.py`:

```python
"""Tests for enhanced error messages."""

import pytest
from microthink.client import MicroThinkError


class TestMicroThinkErrorEnhancements:
    """Tests for enhanced MicroThinkError."""

    def test_error_includes_last_output(self):
        """Error includes the last model output."""
        error = MicroThinkError(
            message="JSON parsing failed",
            last_output='{"name": "Alice",}',
            attempts=3,
        )
        assert '{"name": "Alice",}' in str(error)

    def test_error_includes_attempt_count(self):
        """Error includes number of attempts."""
        error = MicroThinkError(
            message="JSON parsing failed",
            last_output="bad output",
            attempts=3,
        )
        assert "3" in str(error)

    def test_error_includes_json_error_position(self):
        """Error pinpoints JSON syntax error position."""
        error = MicroThinkError(
            message="JSON parsing failed",
            last_output='{"name": "Alice",}',
            attempts=3,
            json_error="Expecting property name: line 1 column 18",
        )
        error_str = str(error)
        assert "column 18" in error_str or "position" in error_str.lower()

    def test_error_includes_suggestion(self):
        """Error includes actionable suggestion."""
        error = MicroThinkError(
            message="JSON parsing failed",
            last_output='{"name": "Alice",}',
            attempts=3,
            json_error="Trailing comma",
        )
        # Should suggest using coder behavior or debug mode
        error_str = str(error)
        assert "suggestion" in error_str.lower() or "try" in error_str.lower()

    def test_error_suggestion_for_trailing_comma(self):
        """Trailing comma error gets specific suggestion."""
        error = MicroThinkError(
            message="JSON parsing failed",
            last_output='{"a": 1,}',
            attempts=3,
            json_error="Trailing comma",
        )
        assert "trailing comma" in str(error).lower()

    def test_error_preserves_exception_behavior(self):
        """Enhanced error still works as exception."""
        error = MicroThinkError("Test error")
        with pytest.raises(MicroThinkError):
            raise error

    def test_error_repr(self):
        """Error has useful repr."""
        error = MicroThinkError(
            message="Test",
            last_output="output",
            attempts=2,
        )
        repr_str = repr(error)
        assert "MicroThinkError" in repr_str
```

### Step 3.2: Run test to verify it fails

```bash
pytest tests/test_errors.py -v
```

Expected: FAIL (some tests will fail due to missing attributes/formatting)

### Step 3.3: Enhance MicroThinkError class

Modify `MicroThinkError` in `src/microthink/client.py`:

```python
class MicroThinkError(Exception):
    """
    Custom exception for MicroThink failures with diagnostic context.

    Provides rich error messages with:
    - The last model output before failure
    - Number of retry attempts made
    - JSON error details with position
    - Actionable suggestions for fixing the issue

    Attributes:
        message: Human-readable error description.
        last_output: The last output from the model before failure.
        attempts: Number of attempts made before failure.
        json_error: Specific JSON parsing error message.
        suggestion: Actionable suggestion for fixing the issue.
    """

    # Common error patterns and their suggestions
    _SUGGESTIONS = {
        "trailing comma": (
            "Model added a trailing comma. Try:\n"
            "  - Use behavior='coder' for cleaner JSON\n"
            "  - Add 'no trailing commas' to your prompt"
        ),
        "expecting property name": (
            "JSON structure is malformed. Try:\n"
            "  - Simplify the requested schema\n"
            "  - Use behavior='coder' for better formatting"
        ),
        "expecting value": (
            "JSON value is missing or malformed. Try:\n"
            "  - Ensure your prompt clearly specifies the expected structure\n"
            "  - Use generate_with_schema() for complex objects"
        ),
        "unterminated string": (
            "Model produced incomplete string. Try:\n"
            "  - Increase timeout if model is slow\n"
            "  - Use a smaller/faster model"
        ),
    }

    def __init__(
        self,
        message: str,
        last_output: Optional[str] = None,
        attempts: int = 0,
        json_error: Optional[str] = None,
    ) -> None:
        """
        Initialize MicroThinkError.

        Args:
            message: The error message.
            last_output: The last raw output from the model.
            attempts: How many attempts were made.
            json_error: Specific JSON parsing error.
        """
        super().__init__(message)
        self.message = message
        self.last_output = last_output
        self.attempts = attempts
        self.json_error = json_error
        self.suggestion = self._generate_suggestion()

    def _generate_suggestion(self) -> str:
        """Generate actionable suggestion based on error type."""
        if not self.json_error:
            return "Run with debug=True to see the model's thinking process."

        error_lower = self.json_error.lower()
        for pattern, suggestion in self._SUGGESTIONS.items():
            if pattern in error_lower:
                return suggestion

        return (
            "Run with debug=True to see the model's thinking process.\n"
            "Consider using behavior='coder' for more reliable JSON output."
        )

    def _format_output_with_pointer(self) -> str:
        """Format output with error position pointer if available."""
        if not self.last_output:
            return ""

        # Truncate long output
        max_len = 200
        output = self.last_output
        if len(output) > max_len:
            output = output[:max_len] + "..."

        # Try to extract position from json_error
        pointer_line = ""
        if self.json_error:
            import re
            # Match patterns like "column 18" or "position 17"
            col_match = re.search(r"column (\d+)", self.json_error, re.IGNORECASE)
            pos_match = re.search(r"position (\d+)", self.json_error, re.IGNORECASE)

            pos = None
            if col_match:
                pos = int(col_match.group(1)) - 1  # 0-indexed
            elif pos_match:
                pos = int(pos_match.group(1))

            if pos is not None and pos < len(output):
                pointer_line = "\n" + " " * pos + "^"

        return output + pointer_line

    def __str__(self) -> str:
        """Return a formatted, actionable error string."""
        parts = [f"MicroThinkError: {self.message}"]

        if self.attempts > 0:
            parts.append(f"  Attempts: {self.attempts}")

        if self.last_output:
            parts.append(f"\n  Last model output:\n    {self._format_output_with_pointer()}")

        if self.json_error:
            parts.append(f"\n  JSON Error: {self.json_error}")

        if self.suggestion:
            parts.append(f"\n  Suggestion: {self.suggestion}")

        return "\n".join(parts)

    def __repr__(self) -> str:
        """Return a detailed repr for debugging."""
        return (
            f"MicroThinkError(message={self.message!r}, "
            f"attempts={self.attempts}, "
            f"has_output={self.last_output is not None})"
        )
```

### Step 3.4: Update error raising in generate() method

In the `generate` method, update the error raising to include json_error:

```python
raise MicroThinkError(
    f"JSON parsing failed after {self.MAX_RETRIES} retries",
    last_output=answer_content,
    attempts=retries,
    json_error=last_error,  # Pass the JSON error message
)
```

### Step 3.5: Run tests to verify they pass

```bash
pytest tests/test_errors.py -v
```

Expected: All tests PASS

### Step 3.6: Commit

```bash
git add src/microthink/client.py tests/test_errors.py
git commit -m "feat: enhance MicroThinkError with diagnostic context

- Add json_error parameter for specific error details
- Add suggestion generation based on error patterns
- Add error position pointer for JSON syntax errors
- Add detailed __str__ and __repr__ methods
- Include actionable suggestions for common errors"
```

---

## Task 4: Final Integration and Cleanup

### Step 4.1: Run all tests

```bash
pytest tests/ -v
```

Expected: All tests PASS

### Step 4.2: Run type checking

```bash
mypy src/microthink/
```

Expected: No errors (or only minor ones to fix)

### Step 4.3: Update __init__.py exports if needed

Ensure all new public APIs are exported:

```python
from microthink.client import MicroThinkClient, MicroThinkError
from microthink.core.prompts import PersonaError, register_persona, unregister_persona

__version__ = "0.1.0"
__all__ = [
    "MicroThinkClient",
    "MicroThinkError",
    "PersonaError",
    "register_persona",
    "unregister_persona",
    "__version__",
]
```

### Step 4.4: Final commit

```bash
git add -A
git commit -m "chore: complete quick wins implementation

Features added:
- Custom persona registration (register_persona, unregister_persona)
- Response caching with LRU eviction and TTL
- Enhanced error messages with diagnostic context

All tests passing."
```

---

## Summary

| Task | Files Changed | Key Functions |
|------|---------------|---------------|
| Custom Personas | `prompts.py`, `client.py`, `__init__.py` | `register_persona()`, `unregister_persona()`, `get_persona()` |
| Response Caching | `cache.py` (new), `client.py` | `ResponseCache`, `cache_stats()`, `clear_cache()` |
| Better Errors | `client.py` | Enhanced `MicroThinkError.__str__()` |

Total estimated implementation time: ~2-3 hours with TDD approach.
