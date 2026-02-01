# Fix Code Review Issues Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all 12 issues identified in the code review (2 Critical, 6 Important, 4 Minor)

**Architecture:** Apply targeted fixes to existing modules without changing the overall architecture. Each fix is isolated and testable independently.

**Tech Stack:** Python 3.9+, pytest for testing

---

## Task 1: Fix JSON Parser Escape Sequence Bug (Critical)

**Files:**
- Modify: `src/microthink/core/parser.py:96-107`

**Step 1: Fix the escape handling logic**

The current code incorrectly handles consecutive backslashes. Fix by properly tracking escape state:

```python
# In clean_json_text(), replace lines 96-107:
    for i in range(start_index, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == '\\' and not escape_next:
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == start_char:
            depth += 1
        elif char == end_char:
            depth -= 1
            if depth == 0:
                end_index = i + 1
                break
```

**Step 2: Verify the fix works**

Run manual test:
```python
from microthink.core.parser import clean_json_text
# Test with escaped backslashes
result = clean_json_text('{"path": "C:\\\\Users\\\\test"}')
print(result)  # Should output: {"path": "C:\\Users\\test"}
```

**Step 3: Commit**

```bash
git add src/microthink/core/parser.py
git commit -m "fix: correct escape sequence handling in JSON parser"
```

---

## Task 2: Fix Silent Exception Swallowing in Web Search (Critical)

**Files:**
- Modify: `src/microthink/tools/search.py:40-46`

**Step 1: Add proper exception handling**

Replace the bare `except Exception` to not swallow KeyboardInterrupt and log errors:

```python
# In search_web(), replace lines 40-46:
    except KeyboardInterrupt:
        raise  # Don't swallow Ctrl+C
    except Exception as e:
        # Log the error for debugging while gracefully degrading
        import logging
        logging.warning(f"Web search failed: {e}")
        return []
```

**Step 2: Verify KeyboardInterrupt propagates**

The fix ensures Ctrl+C works properly during web searches.

**Step 3: Commit**

```bash
git add src/microthink/tools/search.py
git commit -m "fix: don't swallow KeyboardInterrupt in web search"
```

---

## Task 3: Remove Unreachable Code in Client (Important)

**Files:**
- Modify: `src/microthink/client.py:231-236`

**Step 1: Remove the unreachable code block**

Delete lines 231-236 (the unreachable `raise MicroThinkError` after the while loop):

```python
# DELETE these lines at the end of generate():
        # This should never be reached due to the raise in the loop
        raise MicroThinkError(
            "Unexpected error in JSON parsing loop",
            last_output=answer_content,
            attempts=retries,
        )
```

The function now ends after the while loop since all paths either return or raise within the loop.

**Step 2: Verify no syntax errors**

```bash
python -c "from microthink import MicroThinkClient; print('OK')"
```

**Step 3: Commit**

```bash
git add src/microthink/client.py
git commit -m "refactor: remove unreachable code in generate()"
```

---

## Task 4: Add Clear Import Error Handling (Important)

**Files:**
- Modify: `src/microthink/tools/search.py:14-17`

**Step 1: Add clear error message when both packages missing**

Replace the import block:

```python
# Replace lines 14-17:
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        raise ImportError(
            "Web search requires either 'ddgs' or 'duckduckgo-search' package. "
            "Install with: pip install ddgs"
        )
```

**Step 2: Verify import error is clear**

```bash
# In a venv without ddgs:
python -c "from microthink.tools.search import search_web" 2>&1 | grep -i "pip install"
```

**Step 3: Commit**

```bash
git add src/microthink/tools/search.py
git commit -m "fix: provide clear error when search packages missing"
```

---

## Task 5: Add Timeout Support for Ollama API Calls (Important)

**Files:**
- Modify: `src/microthink/client.py:89-103` (init) and lines 178, 218 (chat calls)

**Step 1: Add timeout parameter to __init__**

```python
# Modify __init__ signature and body (around line 89):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        """
        Initialize the MicroThink client.

        Args:
            model: The Ollama model to use (default: "llama3.2:3b").
            host: Optional Ollama host URL (default: uses Ollama default).
            timeout: Request timeout in seconds (default: 120.0).

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
```

**Step 2: Verify client initializes with timeout**

```bash
python -c "from microthink import MicroThinkClient; c = MicroThinkClient(timeout=60); print(c.timeout)"
```

**Step 3: Commit**

```bash
git add src/microthink/client.py
git commit -m "feat: add configurable timeout for Ollama API calls"
```

---

## Task 6: Add Missing Parameters to generate_with_schema (Important)

**Files:**
- Modify: `src/microthink/client.py:239-270`

**Step 1: Add web_search and brief parameters**

```python
# Modify generate_with_schema signature (around line 239):
    def generate_with_schema(
        self,
        prompt: str,
        schema: Dict[str, Any],
        behavior: str = "general",
        debug: bool = False,
        brief: bool = False,
        web_search: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate JSON output that conforms to a specified schema.

        This is a convenience method that includes the schema in the prompt
        to guide the model's output structure.

        Args:
            prompt: The user's input prompt.
            schema: A JSON schema or example structure to follow.
            behavior: The persona to use.
            debug: If True, log the reasoning process.
            brief: If True, output just the result without explanation.
            web_search: If True, search the web for current information.

        Returns:
            The parsed JSON response as a dictionary.

        Raises:
            MicroThinkError: If JSON parsing fails after all retries.
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
```

**Step 2: Verify new parameters work**

```bash
python -c "from microthink import MicroThinkClient; help(MicroThinkClient.generate_with_schema)" | grep -E "brief|web_search"
```

**Step 3: Commit**

```bash
git add src/microthink/client.py
git commit -m "feat: add brief and web_search params to generate_with_schema"
```

---

## Task 7: Fix Temperature Unit Handling (Important)

**Files:**
- Modify: `src/microthink/tools/search.py:78-92`

**Step 1: Preserve temperature units in extraction**

```python
# Replace the temperature extraction block (around lines 78-92):
        # Extract temperature patterns (e.g., "18°C", "65°F", "high of 68")
        # Pattern 1: Explicit unit (18°C, 65°F)
        temp_with_unit = re.findall(
            r"(\d+(?:\.\d+)?)\s*°\s*([CFcf])",
            snippet,
        )
        for temp, unit in temp_with_unit:
            try:
                t = float(temp)
                unit_upper = unit.upper()
                # Validate based on unit
                if unit_upper == 'C' and -50 <= t <= 60:
                    facts.append(f"Temperature: {int(t)}°C")
                elif unit_upper == 'F' and -58 <= t <= 140:
                    facts.append(f"Temperature: {int(t)}°F")
            except ValueError:
                pass

        # Pattern 2: No explicit unit (high of 68) - assume Fahrenheit for weather
        temp_no_unit = re.findall(
            r"(?:high|low|temperature|temp)\s*(?:of|:)?\s*(\d+)",
            snippet,
            re.IGNORECASE,
        )
        for temp in temp_no_unit:
            try:
                t = int(temp)
                if -58 <= t <= 140:  # Fahrenheit range
                    facts.append(f"Temperature: {t}°F")
            except ValueError:
                pass
```

**Step 2: Verify temperature extraction preserves units**

```bash
python -c "
from microthink.tools.search import extract_facts_from_results
results = [{'snippet': 'Today high of 75°F and low of 55°F', 'title': 'Weather'}]
print(extract_facts_from_results(results, 'weather'))
"
```

**Step 3: Commit**

```bash
git add src/microthink/tools/search.py
git commit -m "fix: preserve temperature units in web search extraction"
```

---

## Task 8: Add Stack Trace in Debug Mode for Examples (Minor)

**Files:**
- Modify: `examples/chat.py:191-192`
- Modify: `examples/demo.py:122`

**Step 1: Add traceback in debug mode for chat.py**

```python
# In chat.py, replace lines 191-192:
        except Exception as e:
            print(f"\nError: {e}")
            if debug:
                import traceback
                traceback.print_exc()
```

**Step 2: Add traceback option for demo.py**

```python
# In demo.py, replace line 122 (inside the try/except in main):
        except Exception as e:
            print(f"Demo '{name}' failed: {e}")
            print("(Make sure Ollama is running with the specified model)")
            import traceback
            traceback.print_exc()
            print()
```

**Step 3: Commit**

```bash
git add examples/chat.py examples/demo.py
git commit -m "fix: add traceback output for debugging in examples"
```

---

## Task 9: Use ASCII Fallbacks for Unicode Symbols (Minor)

**Files:**
- Modify: `src/microthink/utils/logger.py:70,77,84`

**Step 1: Replace Unicode symbols with Rich-compatible alternatives**

```python
# Replace the three log functions (lines 65-85):
def log_success(message: str) -> None:
    """
    Log a success message.

    Args:
        message: The success message to display.
    """
    console.print(f"[bold green][OK][/bold green] {message}")


def log_error(message: str) -> None:
    """
    Log an error message.

    Args:
        message: The error message to display.
    """
    console.print(f"[bold red][ERROR][/bold red] {message}")


def log_info(message: str) -> None:
    """
    Log an informational message.

    Args:
        message: The info message to display.
    """
    console.print(f"[bold blue][INFO][/bold blue] {message}")
```

**Step 2: Verify output works on Windows CMD**

```bash
python -c "from microthink.utils.logger import log_info, log_success, log_error; log_info('test'); log_success('ok'); log_error('fail')"
```

**Step 3: Commit**

```bash
git add src/microthink/utils/logger.py
git commit -m "fix: use ASCII fallbacks for cross-platform terminal support"
```

---

## Task 10: Add List to Type Imports (Minor)

**Files:**
- Modify: `src/microthink/core/parser.py:4`

**Step 1: Add List to imports**

```python
# Replace line 4:
from typing import Optional, Dict, List
```

**Step 2: Verify import**

```bash
python -c "from microthink.core.parser import parse_response; print('OK')"
```

**Step 3: Commit**

```bash
git add src/microthink/core/parser.py
git commit -m "fix: add List to type imports in parser"
```

---

## Task 11: Remove Dead Fallback Code in Prompts (Minor)

**Files:**
- Modify: `src/microthink/core/prompts.py:81`

**Step 1: Remove unnecessary fallback**

Since client.py validates behavior before calling build_system_prompt, the fallback is never triggered. Simplify:

```python
# Replace line 81:
    system_text = SYSTEM_PERSONAS[behavior]
```

**Step 2: Verify prompts still work**

```bash
python -c "from microthink.core.prompts import build_system_prompt; print(build_system_prompt('coder')[:50])"
```

**Step 3: Commit**

```bash
git add src/microthink/core/prompts.py
git commit -m "refactor: remove dead fallback code in build_system_prompt"
```

---

## Task 12: Final Verification

**Step 1: Run import test for all modules**

```bash
python -c "
from microthink import MicroThinkClient, MicroThinkError
from microthink.core.parser import parse_response, clean_json_text
from microthink.core.prompts import build_system_prompt
from microthink.tools.search import search_web
from microthink.utils.logger import log_info
print('All imports successful!')
"
```

**Step 2: Commit all changes if not already done**

```bash
git status
```

**Step 3: Create summary commit**

```bash
git log --oneline -12
```

---

## Summary

| Task | Severity | Issue | Status |
|------|----------|-------|--------|
| 1 | Critical | JSON parser escape bug | Pending |
| 2 | Critical | Silent exception swallowing | Pending |
| 3 | Important | Unreachable code | Pending |
| 4 | Important | Missing import error handling | Pending |
| 5 | Important | No Ollama timeout | Pending |
| 6 | Important | Missing params in generate_with_schema | Pending |
| 7 | Important | Temperature unit ambiguity | Pending |
| 8 | Minor | Lost stack traces | Pending |
| 9 | Minor | Unicode rendering issues | Pending |
| 10 | Minor | Missing List import | Pending |
| 11 | Minor | Dead fallback code | Pending |
| 12 | - | Final verification | Pending |
