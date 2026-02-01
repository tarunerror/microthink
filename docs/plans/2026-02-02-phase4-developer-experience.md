# Phase 4: Developer Experience Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add callbacks/hooks, metrics/telemetry, and config file support to MicroThink.

**Architecture:** Callbacks are optional parameters on the client. Metrics use a dedicated class tracking request counts, latencies, and cache stats. Config files use YAML with a standard loading hierarchy.

**Tech Stack:** Python 3.9+, dataclasses, PyYAML (optional dependency), time module for metrics.

---

## Task 1: Callbacks/Hooks

Add lifecycle hooks for monitoring and customization.

**Files:**
- Create: `src/microthink/callbacks.py`
- Modify: `src/microthink/client.py`
- Modify: `src/microthink/__init__.py`
- Create: `tests/test_callbacks.py`

### Step 1.1: Write failing tests for callbacks

Create `tests/test_callbacks.py`:

```python
"""Tests for callbacks/hooks."""

import pytest
from unittest.mock import MagicMock, patch


class TestCallbackTypes:
    """Tests for callback type definitions."""

    def test_callbacks_importable(self):
        """Callbacks module is importable."""
        from microthink.callbacks import Callbacks
        assert Callbacks is not None

    def test_callbacks_has_fields(self):
        """Callbacks has all hook fields."""
        from microthink.callbacks import Callbacks

        cb = Callbacks()
        assert hasattr(cb, "on_request_start")
        assert hasattr(cb, "on_request_end")
        assert hasattr(cb, "on_retry")
        assert hasattr(cb, "on_cache_hit")


class TestClientCallbacks:
    """Tests for callbacks integration with client."""

    def test_client_accepts_callbacks(self):
        """Client accepts callbacks parameter."""
        from microthink import MicroThinkClient
        from microthink.callbacks import Callbacks

        callbacks = Callbacks()
        client = MicroThinkClient(callbacks=callbacks)
        assert client.callbacks is callbacks

    def test_on_request_start_called(self):
        """on_request_start is called before generation."""
        from microthink import MicroThinkClient
        from microthink.callbacks import Callbacks

        mock_callback = MagicMock()
        callbacks = Callbacks(on_request_start=mock_callback)

        with patch("microthink.client.ollama.Client") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.return_value = mock_client
            mock_client.chat.return_value = {
                "message": {"content": "<answer>Response</answer>"}
            }

            client = MicroThinkClient(callbacks=callbacks)
            client.generate("Test prompt")

            mock_callback.assert_called_once()

    def test_on_request_end_called(self):
        """on_request_end is called after generation."""
        from microthink import MicroThinkClient
        from microthink.callbacks import Callbacks

        mock_callback = MagicMock()
        callbacks = Callbacks(on_request_end=mock_callback)

        with patch("microthink.client.ollama.Client") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.return_value = mock_client
            mock_client.chat.return_value = {
                "message": {"content": "<answer>Response</answer>"}
            }

            client = MicroThinkClient(callbacks=callbacks)
            client.generate("Test prompt")

            mock_callback.assert_called_once()

    def test_on_retry_called_on_json_error(self):
        """on_retry is called when JSON parsing fails."""
        from microthink import MicroThinkClient
        from microthink.callbacks import Callbacks

        mock_callback = MagicMock()
        callbacks = Callbacks(on_retry=mock_callback)

        with patch("microthink.client.ollama.Client") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.return_value = mock_client
            # First response is invalid JSON, second is valid
            mock_client.chat.side_effect = [
                {"message": {"content": "<answer>{invalid json</answer>"}},
                {"message": {"content": "<answer>{\"valid\": true}</answer>"}},
            ]

            client = MicroThinkClient(callbacks=callbacks)
            try:
                client.generate("Test", expect_json=True)
            except:
                pass  # May or may not succeed

            # on_retry should have been called at least once
            assert mock_callback.call_count >= 1

    def test_on_cache_hit_called(self):
        """on_cache_hit is called when cache is used."""
        from microthink import MicroThinkClient
        from microthink.callbacks import Callbacks

        mock_callback = MagicMock()
        callbacks = Callbacks(on_cache_hit=mock_callback)

        with patch("microthink.client.ollama.Client") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.return_value = mock_client
            mock_client.chat.return_value = {
                "message": {"content": "<answer>Response</answer>"}
            }

            client = MicroThinkClient(cache=True, callbacks=callbacks)
            client.generate("Test prompt")
            client.generate("Test prompt")  # Should hit cache

            mock_callback.assert_called_once()
```

### Step 1.2: Run tests to verify they fail

```bash
pytest tests/test_callbacks.py -v
```

Expected: FAIL with import errors

### Step 1.3: Implement callbacks module

Create `src/microthink/callbacks.py`:

```python
"""
Callbacks for MicroThink lifecycle hooks.

Provides optional hooks that are called during request
processing for monitoring, logging, and customization.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class Callbacks:
    """
    Lifecycle callbacks for MicroThink operations.

    All callbacks are optional. If not provided, they are no-ops.

    Example:
        >>> def log_start(prompt, behavior):
        ...     print(f"Starting: {prompt[:50]}...")
        >>> callbacks = Callbacks(on_request_start=log_start)
        >>> client = MicroThinkClient(callbacks=callbacks)
    """

    on_request_start: Optional[Callable[[str, str], None]] = None
    """Called before a request. Args: (prompt, behavior)"""

    on_request_end: Optional[Callable[[str, float], None]] = None
    """Called after a request. Args: (result, duration_ms)"""

    on_retry: Optional[Callable[[int, str], None]] = None
    """Called on retry. Args: (attempt_number, error_message)"""

    on_cache_hit: Optional[Callable[[str], None]] = None
    """Called on cache hit. Args: (cache_key)"""

    on_thinking: Optional[Callable[[str], None]] = None
    """Called with thinking content. Args: (thinking_text)"""

    on_answer: Optional[Callable[[str], None]] = None
    """Called with answer content. Args: (answer_text)"""

    def invoke_request_start(self, prompt: str, behavior: str) -> None:
        """Safely invoke on_request_start if defined."""
        if self.on_request_start:
            self.on_request_start(prompt, behavior)

    def invoke_request_end(self, result: str, duration_ms: float) -> None:
        """Safely invoke on_request_end if defined."""
        if self.on_request_end:
            self.on_request_end(result, duration_ms)

    def invoke_retry(self, attempt: int, error: str) -> None:
        """Safely invoke on_retry if defined."""
        if self.on_retry:
            self.on_retry(attempt, error)

    def invoke_cache_hit(self, key: str) -> None:
        """Safely invoke on_cache_hit if defined."""
        if self.on_cache_hit:
            self.on_cache_hit(key)

    def invoke_thinking(self, thinking: str) -> None:
        """Safely invoke on_thinking if defined."""
        if self.on_thinking:
            self.on_thinking(thinking)

    def invoke_answer(self, answer: str) -> None:
        """Safely invoke on_answer if defined."""
        if self.on_answer:
            self.on_answer(answer)
```

### Step 1.4: Integrate callbacks into client

Modify `src/microthink/client.py`:

Add import:
```python
from microthink.callbacks import Callbacks
```

Modify `__init__` to accept callbacks:
```python
def __init__(
    self,
    model: str = "llama3.2:3b",
    host: str = "http://localhost:11434",
    cache: bool = False,
    cache_ttl: int = 3600,
    cache_max_size: int = 1000,
    logger: str = "rich",
    callbacks: Optional[Callbacks] = None,
) -> None:
    # ... existing init code ...
    self.callbacks = callbacks or Callbacks()
```

Add callback invocations in `generate()`:
```python
# At start of generate():
import time
start_time = time.time()
self.callbacks.invoke_request_start(prompt, behavior)

# On cache hit:
self.callbacks.invoke_cache_hit(cache_key)

# On retry:
self.callbacks.invoke_retry(retries, last_error)

# At end of generate():
duration_ms = (time.time() - start_time) * 1000
self.callbacks.invoke_request_end(str(result), duration_ms)
```

### Step 1.5: Export from package

Modify `src/microthink/__init__.py`:

Add import:
```python
from microthink.callbacks import Callbacks
```

Add to `__all__`:
```python
"Callbacks",
```

### Step 1.6: Run callback tests

```bash
pytest tests/test_callbacks.py -v
```

Expected: All tests PASS

### Step 1.7: Commit

```bash
git add src/microthink/callbacks.py src/microthink/client.py src/microthink/__init__.py tests/test_callbacks.py
git commit -m "feat: add callbacks/hooks for lifecycle events

- Add Callbacks dataclass with optional hooks
- Support on_request_start, on_request_end, on_retry, on_cache_hit
- Add on_thinking and on_answer hooks
- Integrate callbacks into MicroThinkClient"
```

---

## Task 2: Metrics/Telemetry

Add built-in observability with request metrics.

**Files:**
- Create: `src/microthink/metrics.py`
- Modify: `src/microthink/client.py`
- Modify: `src/microthink/__init__.py`
- Create: `tests/test_metrics.py`

### Step 2.1: Write failing tests for metrics

Create `tests/test_metrics.py`:

```python
"""Tests for metrics/telemetry."""

import pytest
from unittest.mock import MagicMock, patch


class TestMetrics:
    """Tests for Metrics class."""

    def test_metrics_importable(self):
        """Metrics class is importable."""
        from microthink.metrics import Metrics
        assert Metrics is not None

    def test_metrics_has_counters(self):
        """Metrics has request counters."""
        from microthink.metrics import Metrics

        m = Metrics()
        assert hasattr(m, "total_requests")
        assert hasattr(m, "successful_requests")
        assert hasattr(m, "failed_requests")
        assert hasattr(m, "total_retries")

    def test_metrics_has_cache_stats(self):
        """Metrics has cache statistics."""
        from microthink.metrics import Metrics

        m = Metrics()
        assert hasattr(m, "cache_hits")
        assert hasattr(m, "cache_misses")

    def test_metrics_has_latency(self):
        """Metrics tracks latency."""
        from microthink.metrics import Metrics

        m = Metrics()
        assert hasattr(m, "avg_latency_ms")
        assert hasattr(m, "total_latency_ms")

    def test_metrics_record_request(self):
        """record_request increments counters."""
        from microthink.metrics import Metrics

        m = Metrics()
        m.record_request(success=True, latency_ms=100)

        assert m.total_requests == 1
        assert m.successful_requests == 1
        assert m.total_latency_ms == 100

    def test_metrics_cache_hit_rate(self):
        """cache_hit_rate calculated correctly."""
        from microthink.metrics import Metrics

        m = Metrics()
        m.record_cache_hit()
        m.record_cache_miss()
        m.record_cache_hit()

        assert m.cache_hit_rate == pytest.approx(2 / 3)

    def test_metrics_reset(self):
        """reset() clears all metrics."""
        from microthink.metrics import Metrics

        m = Metrics()
        m.record_request(success=True, latency_ms=100)
        m.reset()

        assert m.total_requests == 0


class TestClientMetrics:
    """Tests for metrics integration with client."""

    def test_client_has_metrics(self):
        """Client has metrics attribute."""
        from microthink import MicroThinkClient

        client = MicroThinkClient()
        assert hasattr(client, "metrics")

    def test_metrics_updated_on_request(self):
        """Metrics are updated after requests."""
        from microthink import MicroThinkClient

        with patch("microthink.client.ollama.Client") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.return_value = mock_client
            mock_client.chat.return_value = {
                "message": {"content": "<answer>Response</answer>"}
            }

            client = MicroThinkClient()
            client.generate("Test prompt")

            assert client.metrics.total_requests == 1
            assert client.metrics.successful_requests == 1

    def test_metrics_track_cache(self):
        """Metrics track cache hits/misses."""
        from microthink import MicroThinkClient

        with patch("microthink.client.ollama.Client") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.return_value = mock_client
            mock_client.chat.return_value = {
                "message": {"content": "<answer>Response</answer>"}
            }

            client = MicroThinkClient(cache=True)
            client.generate("Test prompt")  # Miss
            client.generate("Test prompt")  # Hit

            assert client.metrics.cache_misses == 1
            assert client.metrics.cache_hits == 1
```

### Step 2.2: Run tests to verify they fail

```bash
pytest tests/test_metrics.py -v
```

Expected: FAIL with import errors

### Step 2.3: Implement metrics module

Create `src/microthink/metrics.py`:

```python
"""
Metrics and telemetry for MicroThink.

Provides built-in observability for request counts,
latencies, cache statistics, and error rates.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Metrics:
    """
    Telemetry metrics for MicroThink operations.

    Example:
        >>> client = MicroThinkClient()
        >>> client.generate("Hello")
        >>> print(client.metrics.total_requests)
        1
        >>> print(client.metrics.avg_latency_ms)
        150.5
    """

    # Request counters
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_retries: int = 0

    # Cache statistics
    cache_hits: int = 0
    cache_misses: int = 0

    # Latency tracking
    total_latency_ms: float = 0.0
    _latency_count: int = field(default=0, repr=False)

    @property
    def avg_latency_ms(self) -> float:
        """Average request latency in milliseconds."""
        if self._latency_count == 0:
            return 0.0
        return self.total_latency_ms / self._latency_count

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate as a ratio (0.0 to 1.0)."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    @property
    def success_rate(self) -> float:
        """Success rate as a ratio (0.0 to 1.0)."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    @property
    def retry_rate(self) -> float:
        """Average retries per request."""
        if self.total_requests == 0:
            return 0.0
        return self.total_retries / self.total_requests

    def record_request(self, success: bool, latency_ms: float) -> None:
        """
        Record a completed request.

        Args:
            success: Whether the request succeeded.
            latency_ms: Request duration in milliseconds.
        """
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        self.total_latency_ms += latency_ms
        self._latency_count += 1

    def record_retry(self) -> None:
        """Record a retry attempt."""
        self.total_retries += 1

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        self.cache_hits += 1

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        self.cache_misses += 1

    def reset(self) -> None:
        """Reset all metrics to zero."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_retries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_latency_ms = 0.0
        self._latency_count = 0

    def to_dict(self) -> Dict[str, Any]:
        """
        Export metrics as a dictionary.

        Returns:
            Dict with all metric values.
        """
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_retries": self.total_retries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "avg_latency_ms": self.avg_latency_ms,
            "cache_hit_rate": self.cache_hit_rate,
            "success_rate": self.success_rate,
            "retry_rate": self.retry_rate,
        }
```

### Step 2.4: Integrate metrics into client

Modify `src/microthink/client.py`:

Add import:
```python
from microthink.metrics import Metrics
```

Add metrics initialization in `__init__`:
```python
self.metrics = Metrics()
```

Add metric recording in `generate()`:
```python
# At start:
import time
start_time = time.time()

# On cache hit:
self.metrics.record_cache_hit()

# On cache miss:
self.metrics.record_cache_miss()

# On retry:
self.metrics.record_retry()

# At end (success):
duration_ms = (time.time() - start_time) * 1000
self.metrics.record_request(success=True, latency_ms=duration_ms)

# On error:
duration_ms = (time.time() - start_time) * 1000
self.metrics.record_request(success=False, latency_ms=duration_ms)
```

### Step 2.5: Export from package

Modify `src/microthink/__init__.py`:

Add import:
```python
from microthink.metrics import Metrics
```

Add to `__all__`:
```python
"Metrics",
```

### Step 2.6: Run metrics tests

```bash
pytest tests/test_metrics.py -v
```

Expected: All tests PASS

### Step 2.7: Commit

```bash
git add src/microthink/metrics.py src/microthink/client.py src/microthink/__init__.py tests/test_metrics.py
git commit -m "feat: add metrics/telemetry support

- Add Metrics class for request tracking
- Track total/successful/failed requests
- Track cache hits/misses and hit rate
- Track latency with average calculation
- Add reset() and to_dict() methods"
```

---

## Task 3: Config File Support

Add YAML configuration file support for defaults.

**Files:**
- Create: `src/microthink/config.py`
- Modify: `src/microthink/client.py`
- Modify: `src/microthink/__init__.py`
- Create: `tests/test_config.py`

### Step 3.1: Write failing tests for config

Create `tests/test_config.py`:

```python
"""Tests for config file support."""

import pytest
import tempfile
from pathlib import Path


class TestConfig:
    """Tests for Config class."""

    def test_config_importable(self):
        """Config class is importable."""
        from microthink.config import Config
        assert Config is not None

    def test_config_has_defaults(self):
        """Config has default values."""
        from microthink.config import Config

        config = Config()
        assert config.model == "llama3.2:3b"
        assert config.host == "http://localhost:11434"
        assert config.cache is False
        assert config.debug is False

    def test_config_from_dict(self):
        """Config can be created from dict."""
        from microthink.config import Config

        data = {"model": "phi3:mini", "cache": True}
        config = Config.from_dict(data)

        assert config.model == "phi3:mini"
        assert config.cache is True

    def test_config_from_yaml_file(self):
        """Config can be loaded from YAML file."""
        from microthink.config import Config

        yaml_content = """
model: llama3.1:8b
host: http://localhost:11434
cache: true
cache_ttl: 7200
"""
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", delete=False, mode="w"
        ) as f:
            f.write(yaml_content)
            f.flush()

            config = Config.from_file(f.name)

            assert config.model == "llama3.1:8b"
            assert config.cache is True
            assert config.cache_ttl == 7200

    def test_config_to_dict(self):
        """Config can be exported to dict."""
        from microthink.config import Config

        config = Config(model="test", cache=True)
        data = config.to_dict()

        assert data["model"] == "test"
        assert data["cache"] is True


class TestConfigLoading:
    """Tests for config file discovery."""

    def test_load_config_returns_defaults(self):
        """load_config returns defaults when no file found."""
        from microthink.config import load_config

        config = load_config(search_paths=[])
        assert config.model == "llama3.2:3b"

    def test_load_config_from_explicit_path(self):
        """load_config loads from explicit path."""
        from microthink.config import load_config

        yaml_content = "model: custom-model\n"
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", delete=False, mode="w"
        ) as f:
            f.write(yaml_content)
            f.flush()

            config = load_config(path=f.name)
            assert config.model == "custom-model"


class TestClientConfig:
    """Tests for config integration with client."""

    def test_client_accepts_config(self):
        """Client accepts config parameter."""
        from microthink import MicroThinkClient
        from microthink.config import Config

        config = Config(model="test-model")
        client = MicroThinkClient(config=config)

        assert client.model == "test-model"
```

### Step 3.2: Run tests to verify they fail

```bash
pytest tests/test_config.py -v
```

Expected: FAIL with import errors

### Step 3.3: Implement config module

Create `src/microthink/config.py`:

```python
"""
Configuration file support for MicroThink.

Provides YAML-based configuration with file discovery
and sensible defaults.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Try to import yaml, make it optional
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


DEFAULT_CONFIG_PATHS = [
    Path(".microthink.yaml"),
    Path(".microthink.yml"),
    Path.home() / ".microthink" / "config.yaml",
    Path.home() / ".microthink" / "config.yml",
]


@dataclass
class Config:
    """
    Configuration for MicroThink.

    Example:
        >>> config = Config.from_file("config.yaml")
        >>> client = MicroThinkClient(config=config)
    """

    # Model settings
    model: str = "llama3.2:3b"
    host: str = "http://localhost:11434"

    # Cache settings
    cache: bool = False
    cache_ttl: int = 3600
    cache_max_size: int = 1000

    # Behavior settings
    default_behavior: str = "general"
    debug: bool = False

    # Logging
    logger: str = "rich"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """
        Create a Config from a dictionary.

        Args:
            data: Dictionary with config values.

        Returns:
            A Config instance.
        """
        # Filter to only known fields
        known_fields = {
            "model", "host", "cache", "cache_ttl", "cache_max_size",
            "default_behavior", "debug", "logger"
        }
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        """
        Load a Config from a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            A Config instance.

        Raises:
            ImportError: If PyYAML is not installed.
            FileNotFoundError: If the file doesn't exist.
        """
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for config file support. "
                "Install with: pip install pyyaml"
            )

        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Export config as a dictionary.

        Returns:
            Dict with all config values.
        """
        return asdict(self)


def load_config(
    path: Optional[str | Path] = None,
    search_paths: Optional[List[Path]] = None,
) -> Config:
    """
    Load configuration with file discovery.

    Args:
        path: Explicit path to config file.
        search_paths: Paths to search for config files.

    Returns:
        A Config instance (defaults if no file found).
    """
    # If explicit path given, load it
    if path:
        return Config.from_file(path)

    # Search for config files
    if search_paths is None:
        search_paths = DEFAULT_CONFIG_PATHS

    for config_path in search_paths:
        if config_path.exists():
            try:
                return Config.from_file(config_path)
            except Exception:
                continue

    # Return defaults if no file found
    return Config()
```

### Step 3.4: Integrate config into client

Modify `src/microthink/client.py`:

Add import:
```python
from microthink.config import Config
```

Modify `__init__` to accept config:
```python
def __init__(
    self,
    model: Optional[str] = None,
    host: Optional[str] = None,
    cache: Optional[bool] = None,
    cache_ttl: Optional[int] = None,
    cache_max_size: Optional[int] = None,
    logger: Optional[str] = None,
    callbacks: Optional[Callbacks] = None,
    config: Optional[Config] = None,
) -> None:
    # Load config or use defaults
    if config is None:
        config = Config()

    # Apply config with parameter overrides
    self.model = model if model is not None else config.model
    self.host = host if host is not None else config.host
    # ... etc for other parameters
```

### Step 3.5: Export from package

Modify `src/microthink/__init__.py`:

Add import:
```python
from microthink.config import Config, load_config
```

Add to `__all__`:
```python
"Config",
"load_config",
```

### Step 3.6: Run config tests

```bash
pytest tests/test_config.py -v
```

Expected: All tests PASS

### Step 3.7: Commit

```bash
git add src/microthink/config.py src/microthink/client.py src/microthink/__init__.py tests/test_config.py
git commit -m "feat: add config file support

- Add Config dataclass for configuration
- Support YAML config files (optional PyYAML)
- Add load_config() with file discovery
- Support ~/.microthink/config.yaml location
- Allow config parameter on MicroThinkClient"
```

---

## Task 4: Final Integration and Verification

### Step 4.1: Run all tests

```bash
pytest tests/ -v
```

### Step 4.2: Verify all exports

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
    Callbacks,
    Metrics,
    Config,
    load_config,
    MicroThinkError,
    PersonaError,
    SchemaValidationError,
)
```

### Step 4.3: Final commit

```bash
git add -A
git commit -m "chore: complete Phase 4 Developer Experience

Features added:
- Callbacks/hooks for lifecycle events
- Metrics/telemetry for observability
- Config file support with YAML

All tests passing."
```

---

## Summary

| Task | Files Created/Modified | Key Features |
|------|----------------------|--------------|
| Callbacks | `callbacks.py`, `client.py` | Lifecycle hooks for monitoring |
| Metrics | `metrics.py`, `client.py` | Request/latency/cache tracking |
| Config | `config.py`, `client.py` | YAML config file support |

**Total new tests:** ~25+
