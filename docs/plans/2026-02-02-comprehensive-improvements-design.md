# MicroThink Comprehensive Improvements Design

**Date:** 2026-02-02  
**Status:** Draft  
**Goal:** Improve MicroThink across reliability, capabilities, performance, and developer experience

---

## Overview

MicroThink is a Python library that enhances small local LLMs (via Ollama) through chain-of-thought injection, JSON self-correction, web search, and resilient parsing. This design document outlines improvements across all dimensions to make it more robust, capable, and pleasant to use.

---

## Current Architecture

```
src/microthink/
├── client.py          # Main MicroThinkClient class
├── core/
│   ├── parser.py      # 4-layer resilient response parser
│   └── prompts.py     # System personas and CoT instructions
├── tools/
│   └── search.py      # DuckDuckGo web search integration
└── utils/
    └── logger.py      # Rich console logging
```

**Current Features:**
- Chain-of-Thought injection with `<thinking>`/`<answer>` tags
- JSON output validation with self-correction retry loop (3 attempts)
- 4 personas: general, coder, analyst, reasoner
- Web search with fact extraction
- Debug mode with Rich console output

---

## Section 1: Reliability Improvements

### 1.1 Schema Validation

**Problem:** `generate_with_schema()` only validates JSON syntax, not schema conformance.

**Solution:** Add actual schema validation using jsonschema or custom validation.

```python
# Current behavior
client.generate_with_schema("Create user", schema={"name": "string", "age": "number"})
# Returns: {"foo": "bar"}  # Wrong schema, but valid JSON - no error!

# Proposed behavior
# Raises: MicroThinkError("Missing required field 'name', got fields: ['foo']")
```

**Implementation:**
- Add `validate_against_schema(result, schema)` function
- Support both example schemas (`{"name": "string"}`) and JSON Schema format
- Include schema violations in retry prompt for self-correction

### 1.2 Smarter Retry Strategies

**Problem:** Fixed 3 retries with same strategy regardless of error type.

**Solution:** Different strategies per error type + exponential backoff.

| Error Type | Strategy |
|------------|----------|
| Trailing comma | Direct fix instruction |
| Missing quotes | Lower temperature, stricter prompt |
| Wrong structure | Re-explain schema with examples |
| Timeout | Exponential backoff |

### 1.3 Response Caching

**Problem:** Identical prompts hit the model every time.

**Solution:** LRU cache with configurable TTL.

```python
client = MicroThinkClient(
    cache=True,
    cache_ttl=3600,  # seconds
    cache_max_size=1000,  # entries
)
```

**Cache key:** Hash of (model, behavior, prompt, expect_json, web_search)

### 1.4 Fallback Models

**Problem:** If primary model fails, entire request fails.

**Solution:** Configure fallback chain.

```python
client = MicroThinkClient(
    model="llama3.2:3b",
    fallback_models=["llama3.2:1b", "phi3:mini"],
)
```

### 1.5 Timeout Handling

**Problem:** Only global 120s timeout, no graceful handling.

**Solution:** Per-request timeouts with partial response recovery.

```python
result = client.generate(
    "Complex question",
    timeout=30,  # seconds
    on_timeout="return_partial",  # or "raise", "retry"
)
```

---

## Section 2: New Capabilities

### 2.1 Tool Calling

**Description:** Define Python functions the model can invoke during generation.

```python
@client.tool
def calculate(expression: str) -> float:
    """Evaluate a math expression safely."""
    import ast
    return eval(compile(ast.parse(expression, mode='eval'), '', 'eval'))

@client.tool
def get_weather(city: str) -> dict:
    """Get current weather for a city."""
    return weather_api.get(city)

result = client.generate(
    "What's 15% of Tokyo's current temperature?",
    tools=[calculate, get_weather]
)
# Model calls: get_weather("Tokyo") -> {"temp": 22}
# Model calls: calculate("22 * 0.15") -> 3.3
# Returns: "15% of Tokyo's temperature (22C) is 3.3C"
```

**Implementation:**
- Tool registry with function signatures
- Tool call detection in model output (`<tool_call>...</tool_call>`)
- Execution loop with result injection
- Max tool calls limit to prevent infinite loops

### 2.2 Conversation Memory/Sessions

**Description:** Persist context across multiple generate() calls.

```python
session = client.session(max_history=10)

session.generate("My name is Alice")
session.generate("I'm working on a Python project")
session.generate("What's my name and what am I working on?")
# "Your name is Alice and you're working on a Python project"

# Save/restore sessions
session.save("session.json")
restored = client.session.load("session.json")
```

**Implementation:**
- Session class wrapping MicroThinkClient
- Message history management with token limit awareness
- Automatic summarization when history too long
- Serialization for persistence

### 2.3 Streaming Responses

**Description:** Get tokens as they generate instead of waiting.

```python
# Async streaming
async for chunk in client.stream("Write a story"):
    print(chunk, end="", flush=True)

# Sync streaming with callback
def on_token(token: str):
    print(token, end="", flush=True)

client.generate("Write a story", stream_callback=on_token)
```

### 2.4 Multi-Model Routing

**Description:** Automatically route queries to appropriate models.

```python
client = MicroThinkClient(
    models={
        "fast": "llama3.2:1b",
        "balanced": "llama3.2:3b",
        "strong": "llama3.1:8b",
    },
    routing="auto",  # or "fast", "balanced", "strong", "cost_optimized"
)
```

**Routing logic:**
- Simple factual queries -> fast
- Code generation, analysis -> balanced
- Complex reasoning, multi-step -> strong
- User can override per-request

### 2.5 Image Input (Vision)

**Description:** Support vision models for image analysis.

```python
result = client.generate(
    "What's in this image?",
    images=["photo.jpg"],  # or PIL Image, base64, URL
    model="llava:13b",
)
```

### 2.6 Additional Output Formats

**Description:** Beyond JSON - support YAML, XML, Markdown tables, CSV.

```python
data = client.generate(
    "List Python web frameworks",
    output_format="yaml",  # or "xml", "markdown_table", "csv"
)
```

---

## Section 3: Performance Improvements

### 3.1 Async Support

**Description:** Full async/await API for concurrent requests.

```python
from microthink import AsyncMicroThinkClient

async with AsyncMicroThinkClient() as client:
    results = await asyncio.gather(
        client.generate("Question 1"),
        client.generate("Question 2"),
        client.generate("Question 3"),
    )
```

**Implementation:**
- AsyncMicroThinkClient mirroring sync API
- Use httpx or aiohttp for async HTTP
- Async versions of all public methods

### 3.2 Batch Processing

**Description:** Process multiple prompts efficiently.

```python
prompts = ["Summarize: " + doc for doc in documents]
results = client.batch_generate(
    prompts,
    max_concurrent=5,
    progress_callback=lambda done, total: print(f"{done}/{total}"),
)
```

### 3.3 Connection Pooling

**Description:** Reuse HTTP connections to Ollama.

**Implementation:**
- Use requests.Session or httpx.Client internally
- Configurable pool size
- Automatic reconnection on failure

### 3.4 Lazy Web Search

**Description:** Optimize web search based on query complexity.

```python
# Simple query - fewer results
"What is the capital of France?" -> 1 result

# Complex query - more results
"Compare React vs Vue in 2024" -> 5 results
```

---

## Section 4: Developer Experience

### 4.1 Typed Returns (Pydantic Integration)

**Description:** Type-safe JSON generation with IDE support.

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    email: str

user: User = client.generate(
    "Create a user named Bob",
    response_model=User,
)
print(user.name)  # IDE autocomplete works
```

### 4.2 Callbacks/Hooks

**Description:** Lifecycle hooks for monitoring and customization.

```python
client = MicroThinkClient(
    on_thinking=lambda t: log.debug(f"Thinking: {t}"),
    on_answer=lambda a: log.info(f"Answer: {a}"),
    on_retry=lambda n, e: metrics.increment("retries"),
    on_tool_call=lambda name, args: log.info(f"Tool: {name}"),
    on_cache_hit=lambda key: metrics.increment("cache_hits"),
)
```

### 4.3 Metrics/Telemetry

**Description:** Built-in observability.

```python
# Access metrics
print(client.metrics.total_requests)
print(client.metrics.total_tokens)
print(client.metrics.avg_latency_ms)
print(client.metrics.cache_hit_rate)
print(client.metrics.retry_rate)

# Export to Prometheus, StatsD, etc.
client.metrics.export_prometheus()
```

### 4.4 Better Error Messages

**Description:** Rich, actionable error messages.

```python
# Current:
# MicroThinkError: JSON parsing failed after 3 retries

# Proposed:
# MicroThinkError: JSON parsing failed after 3 retries
#
#   Last model output:
#     {"name": "Alice", "age": 30,}
#                                ^ Trailing comma at position 28
#
#   Suggestion: Model added trailing comma. Try:
#     - Use behavior="coder" for cleaner JSON
#     - Add "no trailing commas" to your prompt
#
#   Debug: Run with debug=True to see thinking process
```

### 4.5 Config File Support

**Description:** YAML/TOML configuration for defaults.

```yaml
# ~/.microthink/config.yaml
default_model: llama3.2:3b
timeout: 60
debug: false

cache:
  enabled: true
  ttl: 3600
  max_size: 1000

personas:
  customer_service:
    prompt: "You are a friendly customer service agent..."

models:
  fast: llama3.2:1b
  default: llama3.2:3b
  strong: llama3.1:8b

tools:
  - microthink.tools.calculator
  - myapp.tools.database
```

### 4.6 Custom Personas

**Description:** User-defined personas beyond the 4 built-in ones.

```python
client.register_persona(
    name="sql_expert",
    prompt="You are an expert SQL developer. You write efficient, "
           "well-indexed queries and always consider performance.",
)

result = client.generate("Query to find top users", behavior="sql_expert")
```

### 4.7 CLI Tool

**Description:** Command-line interface for quick queries.

```bash
# Basic usage
$ microthink "What is 25% of 840"
210

# JSON output
$ microthink --json "List 3 colors"
["red", "blue", "green"]

# Web search
$ microthink --web "Current Bitcoin price"
Bitcoin is currently trading at $67,432

# Pipe input
$ cat code.py | microthink --stdin "Explain this code"

# Specify model/behavior
$ microthink --model llama3.1:8b --behavior coder "Write fizzbuzz"

# Debug mode
$ microthink --debug "How many r's in strawberry"
```

### 4.8 Standard Logging

**Description:** Integrate with Python's standard logging, Rich optional.

```python
import logging
logging.basicConfig(level=logging.DEBUG)

client = MicroThinkClient(
    logger="microthink",  # Use standard logging
    # or
    logger="rich",  # Use Rich console (current behavior)
)
```

---

## Implementation Phases

### Phase 1: Core Foundations
1. Async Support
2. Schema Validation  
3. Custom Personas
4. Better Error Messages
5. Standard Logging

### Phase 2: Performance & Reliability
1. Response Caching
2. Streaming Responses
3. Smarter Retry Strategies
4. Connection Pooling
5. Batch Processing

### Phase 3: New Capabilities
1. Conversation Memory/Sessions
2. Tool Calling
3. Multi-Model Routing
4. Image Input (Vision)
5. Additional Output Formats

### Phase 4: Developer Experience Polish
1. Typed Returns (Pydantic)
2. Callbacks/Hooks
3. Metrics/Telemetry
4. Config File Support
5. CLI Tool

### Phase 5: Advanced (Future)
1. RAG Integration
2. Middleware Pipeline
3. Fallback Models
4. Prompt Caching

---

## Quick Wins (Independent, High Value)

These can be implemented anytime with minimal dependencies:

| Feature | Effort | Impact |
|---------|--------|--------|
| Custom Personas | 1 hour | High - user extensibility |
| Response Caching | 2 hours | High - instant repeated queries |
| Config File Support | 2 hours | Medium - convenience |
| CLI Tool | 3 hours | Medium - quick testing |
| Better Error Messages | 2 hours | High - debugging |

---

## File Structure (Proposed)

```
src/microthink/
├── __init__.py
├── client.py              # Sync client
├── async_client.py        # Async client (new)
├── session.py             # Conversation memory (new)
├── core/
│   ├── parser.py
│   ├── prompts.py
│   ├── schema.py          # Schema validation (new)
│   └── cache.py           # Response caching (new)
├── tools/
│   ├── search.py
│   ├── registry.py        # Tool registry (new)
│   └── builtin.py         # Built-in tools (new)
├── routing/
│   └── router.py          # Multi-model routing (new)
├── utils/
│   ├── logger.py
│   ├── config.py          # Config file support (new)
│   └── metrics.py         # Telemetry (new)
└── cli/
    └── main.py            # CLI tool (new)
```

---

## Next Steps

1. Review and approve this design
2. Choose which phase to start with
3. Create detailed implementation plan for chosen phase
4. Set up git worktree for isolated development
5. Begin implementation with TDD approach
