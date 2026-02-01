"""
MicroThink - Smart wrapper for small LLMs.

A library that makes small, local language models (like Llama 3.2 3B, Phi-3,
or Gemma) perform at a higher level through automatic Chain-of-Thought
injection, JSON guardrails with self-correction, and persona optimization.

Example:
    >>> from microthink import MicroThinkClient
    >>>
    >>> client = MicroThinkClient(model="llama3.2:3b")
    >>>
    >>> # Simple reasoning task
    >>> answer = client.generate("How many r's in strawberry?", debug=True)
    >>>
    >>> # JSON output with validation
    >>> data = client.generate(
    ...     "Return a JSON list of 3 Python keywords",
    ...     behavior="coder",
    ...     expect_json=True
    ... )
"""

from microthink.async_client import AsyncMicroThinkClient
from microthink.batch import BatchProcessor, BatchResult
from microthink.client import MicroThinkClient, MicroThinkError
from microthink.core.prompts import (
    PersonaError,
    get_persona,
    register_persona,
    unregister_persona,
)
from microthink.core.schema import SchemaValidationError
from microthink.session import Session
from microthink.tools.registry import ToolInfo, ToolRegistry, tool

__version__ = "0.1.0"
__all__ = [
    "AsyncMicroThinkClient",
    "BatchProcessor",
    "BatchResult",
    "MicroThinkClient",
    "MicroThinkError",
    "PersonaError",
    "SchemaValidationError",
    "Session",
    "ToolInfo",
    "ToolRegistry",
    "get_persona",
    "register_persona",
    "tool",
    "unregister_persona",
    "__version__",
]
