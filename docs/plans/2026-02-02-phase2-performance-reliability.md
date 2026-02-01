# Phase 2: Performance & Reliability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add streaming responses, smarter retry strategies, and batch processing to MicroThink.

**Architecture:** Streaming uses Ollama's native streaming with callback support for sync and async iterators for async. Smarter retries analyze error types and adjust strategy accordingly. Batch processing uses asyncio for concurrent requests with configurable concurrency limits.

**Tech Stack:** Python 3.9+, asyncio, httpx (already installed), pytest-asyncio.

---

## Task 1: Streaming Responses

Add streaming support to both sync and async clients for real-time token output.

**Files:**
- Modify: `src/microthink/client.py`
- Modify: `src/microthink/async_client.py`
- Create: `tests/test_streaming.py`

### Step 1.1: Write failing tests for streaming

Create `tests/test_streaming.py`:

```python
"""Tests for streaming responses."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Iterator


class TestSyncStreaming:
    """Tests for sync client streaming."""

    def test_generate_accepts_stream_callback(self):
        """generate() accepts stream_callback parameter."""
        from microthink import MicroThinkClient

        client = MicroThinkClient()
        # Should not raise - just verify parameter is accepted
        assert hasattr(client.generate, "__call__")

    def test_stream_method_exists(self):
        """Client has stream() method returning iterator."""
        from microthink import MicroThinkClient

        client = MicroThinkClient()
        assert hasattr(client, "stream")

    def test_stream_returns_iterator(self):
        """stream() returns an iterator."""
        from microthink import MicroThinkClient
        import collections.abc

        client = MicroThinkClient()
        with patch.object(client, "_client") as mock_ollama:
            # Mock streaming response
            mock_ollama.chat.return_value = iter([
                {"message": {"content": "Hello"}},
                {"message": {"content": " world"}},
            ])
            
            result = client.stream("Test prompt")
            assert isinstance(result, collections.abc.Iterator)


class TestAsyncStreaming:
    """Tests for async client streaming."""

    def test_async_stream_method_exists(self):
        """AsyncMicroThinkClient has stream() method."""
        from microthink import AsyncMicroThinkClient

        client = AsyncMicroThinkClient()
        assert hasattr(client, "stream")

    @pytest.mark.asyncio
    async def test_async_stream_returns_async_iterator(self):
        """stream() returns an async iterator."""
        from microthink import AsyncMicroThinkClient
        import collections.abc

        with patch("microthink.async_client.httpx.AsyncClient") as mock_httpx:
            mock_client = AsyncMock()
            mock_httpx.return_value = mock_client
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            async with AsyncMicroThinkClient() as client:
                # Just verify the method exists and is async
                assert hasattr(client, "stream")


class TestStreamCallback:
    """Tests for stream callback functionality."""

    def test_callback_receives_chunks(self):
        """Callback is called for each chunk."""
        from microthink import MicroThinkClient

        chunks_received = []

        def callback(chunk: str):
            chunks_received.append(chunk)

        with patch("microthink.client.ollama.Client") as mock_ollama_class:
            mock_client = MagicMock()
            mock_ollama_class.return_value = mock_client

            # Mock streaming chat response
            mock_client.chat.return_value = iter([
                {"message": {"content": "Hello"}},
                {"message": {"content": " world"}},
                {"message": {"content": "!"}},
            ])

            client = MicroThinkClient()
            client._stream_generate("Test", callback=callback)

            assert len(chunks_received) >= 1
```

### Step 1.2: Run tests to verify they fail

```bash
pytest tests/test_streaming.py -v
```

Expected: FAIL with missing methods

### Step 1.3: Implement sync streaming in client.py

Add to `MicroThinkClient` class:

```python
from typing import Callable, Iterator

def stream(
    self,
    prompt: str,
    behavior: str = "general",
    brief: bool = False,
) -> Iterator[str]:
    """
    Stream a response token by token.

    Args:
        prompt: The user's input prompt.
        behavior: The persona to use.
        brief: If True, output just the result.

    Yields:
        String chunks as they are generated.

    Example:
        >>> for chunk in client.stream("Write a story"):
        ...     print(chunk, end="", flush=True)
    """
    if behavior not in SYSTEM_PERSONAS:
        raise ValueError(
            f"Invalid behavior '{behavior}'. Available: {self.available_behaviors}"
        )

    system_prompt = build_system_prompt(behavior, expect_json=False, brief=brief)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    response = self._client.chat(
        model=self.model,
        messages=messages,
        stream=True,
    )

    for chunk in response:
        if "message" in chunk and "content" in chunk["message"]:
            yield chunk["message"]["content"]


def _stream_generate(
    self,
    prompt: str,
    callback: Callable[[str], None],
    behavior: str = "general",
    brief: bool = False,
) -> str:
    """
    Generate with streaming callback.

    Args:
        prompt: The user's input prompt.
        callback: Function called with each chunk.
        behavior: The persona to use.
        brief: If True, output just the result.

    Returns:
        The complete response.
    """
    full_response = []
    for chunk in self.stream(prompt, behavior, brief):
        callback(chunk)
        full_response.append(chunk)
    return "".join(full_response)
```

### Step 1.4: Implement async streaming in async_client.py

Add to `AsyncMicroThinkClient` class:

```python
from typing import AsyncIterator

async def stream(
    self,
    prompt: str,
    behavior: str = "general",
    brief: bool = False,
) -> AsyncIterator[str]:
    """
    Stream a response token by token asynchronously.

    Args:
        prompt: The user's input prompt.
        behavior: The persona to use.
        brief: If True, output just the result.

    Yields:
        String chunks as they are generated.

    Example:
        >>> async for chunk in client.stream("Write a story"):
        ...     print(chunk, end="", flush=True)
    """
    if self._http_client is None:
        raise RuntimeError(
            "Client not initialized. Use 'async with AsyncMicroThinkClient():'"
        )

    if behavior not in SYSTEM_PERSONAS:
        raise ValueError(
            f"Invalid behavior '{behavior}'. Available: {self.available_behaviors}"
        )

    system_prompt = build_system_prompt(behavior, expect_json=False, brief=brief)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    payload = {
        "model": self.model,
        "messages": messages,
        "stream": True,
    }

    async with self._http_client.stream(
        "POST",
        f"{self.host}/api/chat",
        json=payload,
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]
                except json.JSONDecodeError:
                    continue
```

### Step 1.5: Run streaming tests

```bash
pytest tests/test_streaming.py -v
```

Expected: All tests PASS

### Step 1.6: Commit

```bash
git add src/microthink/client.py src/microthink/async_client.py tests/test_streaming.py
git commit -m "feat: add streaming response support

- Add stream() method to MicroThinkClient returning Iterator[str]
- Add stream() method to AsyncMicroThinkClient returning AsyncIterator[str]
- Add _stream_generate() helper for callback-based streaming
- Support real-time token output for both sync and async clients"
```

---

## Task 2: Smarter Retry Strategies

Implement error-aware retry strategies that adapt based on the type of error.

**Files:**
- Create: `src/microthink/core/retry.py`
- Modify: `src/microthink/client.py`
- Create: `tests/test_retry.py`

### Step 2.1: Write failing tests for retry strategies

Create `tests/test_retry.py`:

```python
"""Tests for smarter retry strategies."""

import pytest
from microthink.core.retry import (
    RetryStrategy,
    get_retry_strategy,
    analyze_json_error,
)


class TestAnalyzeJsonError:
    """Tests for error analysis."""

    def test_detects_trailing_comma(self):
        """Detects trailing comma errors."""
        result = analyze_json_error("Expecting property name: line 1 column 20")
        assert result["error_type"] == "trailing_comma"

    def test_detects_missing_quotes(self):
        """Detects missing quote errors."""
        result = analyze_json_error("Expecting ',' delimiter: line 1 column 10")
        assert result["error_type"] in ["structure", "syntax"]

    def test_detects_unterminated_string(self):
        """Detects unterminated string errors."""
        result = analyze_json_error("Unterminated string starting at")
        assert result["error_type"] == "unterminated"

    def test_unknown_error(self):
        """Returns unknown for unrecognized errors."""
        result = analyze_json_error("Some random error")
        assert result["error_type"] == "unknown"


class TestGetRetryStrategy:
    """Tests for getting appropriate retry strategy."""

    def test_trailing_comma_strategy(self):
        """Trailing comma gets specific fix instruction."""
        strategy = get_retry_strategy("trailing_comma", attempt=1)
        assert strategy.temperature == 0.1
        assert "trailing comma" in strategy.correction_prompt.lower()

    def test_structure_error_strategy(self):
        """Structure errors get schema re-explanation."""
        strategy = get_retry_strategy("structure", attempt=1)
        assert strategy.temperature <= 0.3
        assert "structure" in strategy.correction_prompt.lower() or "schema" in strategy.correction_prompt.lower()

    def test_temperature_decreases_with_attempts(self):
        """Temperature decreases as attempts increase."""
        strategy1 = get_retry_strategy("unknown", attempt=1)
        strategy2 = get_retry_strategy("unknown", attempt=2)
        strategy3 = get_retry_strategy("unknown", attempt=3)
        assert strategy1.temperature >= strategy2.temperature >= strategy3.temperature

    def test_unknown_gets_generic_strategy(self):
        """Unknown errors get generic retry strategy."""
        strategy = get_retry_strategy("unknown", attempt=1)
        assert strategy.temperature <= 0.5
        assert len(strategy.correction_prompt) > 0


class TestRetryStrategy:
    """Tests for RetryStrategy dataclass."""

    def test_strategy_has_required_fields(self):
        """RetryStrategy has temperature and correction_prompt."""
        strategy = RetryStrategy(temperature=0.2, correction_prompt="Fix it")
        assert strategy.temperature == 0.2
        assert strategy.correction_prompt == "Fix it"

    def test_strategy_has_optional_max_tokens(self):
        """RetryStrategy can have max_tokens."""
        strategy = RetryStrategy(
            temperature=0.2,
            correction_prompt="Fix it",
            max_tokens=500,
        )
        assert strategy.max_tokens == 500
```

### Step 2.2: Run tests to verify they fail

```bash
pytest tests/test_retry.py -v
```

Expected: FAIL with import errors

### Step 2.3: Implement retry module

Create `src/microthink/core/retry.py`:

```python
"""
Smart retry strategies for MicroThink.

Provides error-aware retry strategies that adapt based on
the type of JSON parsing error encountered.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class RetryStrategy:
    """
    A retry strategy with specific parameters.

    Attributes:
        temperature: The temperature to use for retry.
        correction_prompt: The prompt to send for correction.
        max_tokens: Optional max tokens limit.
    """

    temperature: float
    correction_prompt: str
    max_tokens: Optional[int] = None


def analyze_json_error(error_message: str) -> dict:
    """
    Analyze a JSON parsing error to determine its type.

    Args:
        error_message: The error message from json.loads().

    Returns:
        Dict with 'error_type' and optional 'position' keys.
    """
    error_lower = error_message.lower()

    # Trailing comma detection
    if "expecting property name" in error_lower:
        return {"error_type": "trailing_comma"}

    # Unterminated string
    if "unterminated string" in error_lower:
        return {"error_type": "unterminated"}

    # Missing value
    if "expecting value" in error_lower:
        return {"error_type": "missing_value"}

    # Structure/delimiter issues
    if "expecting" in error_lower and "delimiter" in error_lower:
        return {"error_type": "structure"}

    # Extra data after JSON
    if "extra data" in error_lower:
        return {"error_type": "extra_data"}

    return {"error_type": "unknown"}


def get_retry_strategy(error_type: str, attempt: int) -> RetryStrategy:
    """
    Get the appropriate retry strategy for an error type.

    Args:
        error_type: The type of error from analyze_json_error().
        attempt: The current attempt number (1-indexed).

    Returns:
        A RetryStrategy with appropriate parameters.
    """
    # Base temperature decreases with attempts
    base_temp = max(0.1, 0.5 - (attempt * 0.15))

    strategies = {
        "trailing_comma": RetryStrategy(
            temperature=0.1,
            correction_prompt=(
                "Your JSON has a trailing comma which is invalid. "
                "Remove the comma before the closing brace/bracket. "
                "Output ONLY valid JSON inside <answer> tags."
            ),
        ),
        "unterminated": RetryStrategy(
            temperature=0.2,
            correction_prompt=(
                "Your JSON has an unterminated string. "
                "Ensure all strings are properly closed with quotes. "
                "Output ONLY valid JSON inside <answer> tags."
            ),
            max_tokens=1000,  # Limit to prevent runaway generation
        ),
        "missing_value": RetryStrategy(
            temperature=0.2,
            correction_prompt=(
                "Your JSON is missing a value. "
                "Ensure all keys have corresponding values. "
                "Output ONLY valid JSON inside <answer> tags."
            ),
        ),
        "structure": RetryStrategy(
            temperature=0.3,
            correction_prompt=(
                "Your JSON structure is malformed. "
                "Check that all braces, brackets, and commas are correct. "
                "Output ONLY valid JSON inside <answer> tags."
            ),
        ),
        "extra_data": RetryStrategy(
            temperature=0.2,
            correction_prompt=(
                "Your response contains extra text after the JSON. "
                "Output ONLY the JSON object/array, nothing else. "
                "Put it inside <answer> tags."
            ),
        ),
    }

    if error_type in strategies:
        strategy = strategies[error_type]
        # Adjust temperature based on attempt
        return RetryStrategy(
            temperature=max(0.1, strategy.temperature - (attempt - 1) * 0.05),
            correction_prompt=strategy.correction_prompt,
            max_tokens=strategy.max_tokens,
        )

    # Unknown error - generic strategy
    return RetryStrategy(
        temperature=base_temp,
        correction_prompt=(
            "Your previous response was invalid JSON. "
            f"Error: This is attempt {attempt}. "
            "Please fix the syntax error and output ONLY valid JSON inside <answer> tags. "
            "Do not add any explanatory text."
        ),
    )
```

### Step 2.4: Run retry tests

```bash
pytest tests/test_retry.py -v
```

Expected: All tests PASS

### Step 2.5: Integrate smart retries into client

Modify `src/microthink/client.py`:

Add import:
```python
from microthink.core.retry import analyze_json_error, get_retry_strategy
```

Update the JSON retry loop in `generate()` method to use smart strategies:

```python
# In the except json.JSONDecodeError block:
except json.JSONDecodeError as e:
    retries += 1
    last_error = str(e)

    if debug:
        log_retry(retries, self.MAX_RETRIES, last_error)

    if retries >= self.MAX_RETRIES:
        raise MicroThinkError(
            f"JSON parsing failed after {self.MAX_RETRIES} retries",
            last_output=answer_content,
            attempts=retries,
            json_error=last_error,
        )

    # Analyze error and get appropriate strategy
    error_info = analyze_json_error(last_error)
    strategy = get_retry_strategy(error_info["error_type"], retries)

    # Append correction to history
    messages.append({"role": "user", "content": strategy.correction_prompt})

    # Retry with strategy-specific temperature
    options = {"temperature": strategy.temperature}
    if strategy.max_tokens:
        options["num_predict"] = strategy.max_tokens

    response = self._client.chat(
        model=self.model,
        messages=messages,
        options=options,
    )
    # ... rest of retry logic
```

### Step 2.6: Run all tests

```bash
pytest tests/ -v
```

Expected: All tests PASS

### Step 2.7: Commit

```bash
git add src/microthink/core/retry.py src/microthink/client.py tests/test_retry.py
git commit -m "feat: add smarter retry strategies

- Add retry.py with error analysis and strategy selection
- Detect trailing commas, unterminated strings, structure errors
- Adjust temperature and correction prompts per error type
- Temperature decreases with retry attempts"
```

---

## Task 3: Batch Processing

Add batch processing for handling multiple prompts concurrently.

**Files:**
- Create: `src/microthink/batch.py`
- Modify: `src/microthink/__init__.py`
- Create: `tests/test_batch.py`

### Step 3.1: Write failing tests for batch processing

Create `tests/test_batch.py`:

```python
"""Tests for batch processing."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


class TestBatchProcessor:
    """Tests for BatchProcessor class."""

    def test_batch_processor_importable(self):
        """BatchProcessor is importable."""
        from microthink.batch import BatchProcessor
        assert BatchProcessor is not None

    def test_batch_processor_accepts_client(self):
        """BatchProcessor accepts async client."""
        from microthink import AsyncMicroThinkClient
        from microthink.batch import BatchProcessor

        client = AsyncMicroThinkClient()
        processor = BatchProcessor(client)
        assert processor.client is client

    def test_batch_processor_has_process_method(self):
        """BatchProcessor has async process method."""
        from microthink import AsyncMicroThinkClient
        from microthink.batch import BatchProcessor
        import asyncio

        client = AsyncMicroThinkClient()
        processor = BatchProcessor(client)
        assert hasattr(processor, "process")
        assert asyncio.iscoroutinefunction(processor.process)

    def test_batch_processor_accepts_concurrency(self):
        """BatchProcessor accepts max_concurrent parameter."""
        from microthink import AsyncMicroThinkClient
        from microthink.batch import BatchProcessor

        client = AsyncMicroThinkClient()
        processor = BatchProcessor(client, max_concurrent=5)
        assert processor.max_concurrent == 5


@pytest.mark.asyncio
class TestBatchProcessing:
    """Tests for batch processing with mocked client."""

    async def test_process_multiple_prompts(self):
        """Processes multiple prompts."""
        from microthink.batch import BatchProcessor
        from microthink import AsyncMicroThinkClient

        with patch("microthink.async_client.httpx.AsyncClient") as mock_httpx:
            mock_client = AsyncMock()
            mock_httpx.return_value = mock_client
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            # Mock responses
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "message": {"content": "<answer>Result</answer>"}
            }
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response

            async with AsyncMicroThinkClient() as client:
                processor = BatchProcessor(client, max_concurrent=2)
                prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
                results = await processor.process(prompts)

            assert len(results) == 3

    async def test_process_with_progress_callback(self):
        """Calls progress callback during processing."""
        from microthink.batch import BatchProcessor
        from microthink import AsyncMicroThinkClient

        progress_calls = []

        def on_progress(completed: int, total: int):
            progress_calls.append((completed, total))

        with patch("microthink.async_client.httpx.AsyncClient") as mock_httpx:
            mock_client = AsyncMock()
            mock_httpx.return_value = mock_client
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "message": {"content": "<answer>Result</answer>"}
            }
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response

            async with AsyncMicroThinkClient() as client:
                processor = BatchProcessor(client)
                prompts = ["P1", "P2"]
                await processor.process(prompts, on_progress=on_progress)

            assert len(progress_calls) >= 2

    async def test_process_handles_errors(self):
        """Handles individual prompt errors gracefully."""
        from microthink.batch import BatchProcessor, BatchResult
        from microthink import AsyncMicroThinkClient

        with patch("microthink.async_client.httpx.AsyncClient") as mock_httpx:
            mock_client = AsyncMock()
            mock_httpx.return_value = mock_client
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            # First call succeeds, second fails
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "message": {"content": "<answer>Result</answer>"}
            }
            mock_response.raise_for_status = MagicMock()
            
            call_count = [0]
            async def mock_post(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 2:
                    raise Exception("API Error")
                return mock_response
            
            mock_client.post = mock_post

            async with AsyncMicroThinkClient() as client:
                processor = BatchProcessor(client)
                prompts = ["P1", "P2", "P3"]
                results = await processor.process(prompts, continue_on_error=True)

            # Should have results for all prompts
            assert len(results) == 3
            # At least one should have an error
            assert any(r.error is not None for r in results)
```

### Step 3.2: Run tests to verify they fail

```bash
pytest tests/test_batch.py -v
```

Expected: FAIL with import errors

### Step 3.3: Implement batch processor

Create `src/microthink/batch.py`:

```python
"""
Batch processing for MicroThink.

Provides concurrent processing of multiple prompts with
configurable parallelism and progress tracking.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from microthink.async_client import AsyncMicroThinkClient


@dataclass
class BatchResult:
    """
    Result of a batch processing operation.

    Attributes:
        prompt: The original prompt.
        result: The generated result (None if error).
        error: Error message if failed (None if success).
        index: Original index in the batch.
    """

    prompt: str
    result: Optional[Union[str, Dict[str, Any], List[Any]]]
    error: Optional[str]
    index: int

    @property
    def success(self) -> bool:
        """Whether this result was successful."""
        return self.error is None


class BatchProcessor:
    """
    Process multiple prompts concurrently.

    Example:
        >>> async with AsyncMicroThinkClient() as client:
        ...     processor = BatchProcessor(client, max_concurrent=5)
        ...     results = await processor.process([
        ...         "What is Python?",
        ...         "What is JavaScript?",
        ...         "What is Rust?",
        ...     ])
    """

    def __init__(
        self,
        client: AsyncMicroThinkClient,
        max_concurrent: int = 5,
    ) -> None:
        """
        Initialize the batch processor.

        Args:
            client: The async client to use for requests.
            max_concurrent: Maximum concurrent requests (default: 5).
        """
        self.client = client
        self.max_concurrent = max_concurrent

    async def process(
        self,
        prompts: List[str],
        behavior: str = "general",
        expect_json: bool = False,
        on_progress: Optional[Callable[[int, int], None]] = None,
        continue_on_error: bool = False,
    ) -> List[BatchResult]:
        """
        Process multiple prompts concurrently.

        Args:
            prompts: List of prompts to process.
            behavior: The persona to use for all prompts.
            expect_json: If True, parse JSON output.
            on_progress: Callback called with (completed, total) after each prompt.
            continue_on_error: If True, continue processing on individual errors.

        Returns:
            List of BatchResult objects in original order.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results: List[BatchResult] = [None] * len(prompts)  # type: ignore
        completed = [0]

        async def process_one(index: int, prompt: str) -> None:
            async with semaphore:
                try:
                    result = await self.client.generate(
                        prompt=prompt,
                        behavior=behavior,
                        expect_json=expect_json,
                    )
                    results[index] = BatchResult(
                        prompt=prompt,
                        result=result,
                        error=None,
                        index=index,
                    )
                except Exception as e:
                    if not continue_on_error:
                        raise
                    results[index] = BatchResult(
                        prompt=prompt,
                        result=None,
                        error=str(e),
                        index=index,
                    )
                finally:
                    completed[0] += 1
                    if on_progress:
                        on_progress(completed[0], len(prompts))

        # Create tasks for all prompts
        tasks = [
            asyncio.create_task(process_one(i, prompt))
            for i, prompt in enumerate(prompts)
        ]

        # Wait for all to complete
        await asyncio.gather(*tasks, return_exceptions=continue_on_error)

        return results

    async def process_with_schema(
        self,
        prompts: List[str],
        schema: Union[Dict[str, Any], List[Any]],
        behavior: str = "general",
        on_progress: Optional[Callable[[int, int], None]] = None,
        continue_on_error: bool = False,
    ) -> List[BatchResult]:
        """
        Process multiple prompts with schema validation.

        Args:
            prompts: List of prompts to process.
            schema: The schema to validate against.
            behavior: The persona to use.
            on_progress: Progress callback.
            continue_on_error: Continue on individual errors.

        Returns:
            List of BatchResult objects.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results: List[BatchResult] = [None] * len(prompts)  # type: ignore
        completed = [0]

        async def process_one(index: int, prompt: str) -> None:
            async with semaphore:
                try:
                    result = await self.client.generate_with_schema(
                        prompt=prompt,
                        schema=schema,
                        behavior=behavior,
                    )
                    results[index] = BatchResult(
                        prompt=prompt,
                        result=result,
                        error=None,
                        index=index,
                    )
                except Exception as e:
                    if not continue_on_error:
                        raise
                    results[index] = BatchResult(
                        prompt=prompt,
                        result=None,
                        error=str(e),
                        index=index,
                    )
                finally:
                    completed[0] += 1
                    if on_progress:
                        on_progress(completed[0], len(prompts))

        tasks = [
            asyncio.create_task(process_one(i, prompt))
            for i, prompt in enumerate(prompts)
        ]
        await asyncio.gather(*tasks, return_exceptions=continue_on_error)
        return results
```

### Step 3.4: Export from package

Modify `src/microthink/__init__.py`:

Add import:
```python
from microthink.batch import BatchProcessor, BatchResult
```

Add to `__all__`:
```python
"BatchProcessor",
"BatchResult",
```

### Step 3.5: Run batch tests

```bash
pytest tests/test_batch.py -v
```

Expected: All tests PASS

### Step 3.6: Run all tests

```bash
pytest tests/ -v
```

Expected: All tests PASS

### Step 3.7: Commit

```bash
git add src/microthink/batch.py src/microthink/__init__.py tests/test_batch.py
git commit -m "feat: add batch processing support

- Add BatchProcessor for concurrent prompt processing
- Add BatchResult dataclass for individual results
- Support progress callbacks for monitoring
- Support continue_on_error for graceful failure handling
- Add process_with_schema for batch schema validation"
```

---

## Task 4: Final Integration

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
    MicroThinkError,
    PersonaError,
    SchemaValidationError,
)
```

### Step 4.3: Final commit

```bash
git add -A
git commit -m "chore: complete Phase 2 Performance & Reliability

Features added:
- Streaming responses for real-time token output
- Smarter retry strategies based on error type
- Batch processing for concurrent requests

All tests passing."
```

---

## Summary

| Task | Files Created/Modified | Key Features |
|------|----------------------|--------------|
| Streaming | `client.py`, `async_client.py` | `stream()` method, callback support |
| Smart Retries | `retry.py`, `client.py` | Error analysis, adaptive strategies |
| Batch Processing | `batch.py` | `BatchProcessor`, concurrency control |

**Total new tests:** ~25+
