"""
Callbacks for MicroThink lifecycle hooks.

Provides optional hooks that are called during request
processing for monitoring, logging, and customization.
"""

from dataclasses import dataclass
from typing import Callable, Optional


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
