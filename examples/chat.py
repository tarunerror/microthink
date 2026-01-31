#!/usr/bin/env python3
"""
MicroThink Interactive Chat - Continuous conversation mode with auto-detection.
"""

import re

from microthink import MicroThinkClient, MicroThinkError

# Keyword patterns for auto-detection
PATTERNS = {
    "json": [
        r"\bjson\b",
        r"\blist\s+(?:of|as)\b",
        r"\barray\s+of\b",
        r"\breturn\s+(?:a\s+)?(?:json|dict|list)\b",
        r"\bas\s+json\b",
        r"\bjson\s+(?:format|output|response)\b",
    ],
    "coder": [
        r"\b(?:write|create|make|build)\s+(?:a\s+)?(?:function|code|script|program|class)\b",
        r"\b(?:code|implement|program)\b",
        r"\b(?:python|javascript|java|rust|go|c\+\+)\s+(?:code|function|script)\b",
        r"\bfunction\s+(?:to|that|for)\b",
        r"\brefactor\b",
        r"\bdebug\s+this\b",
    ],
    "analyst": [
        r"\bcalculate\b",
        r"\bhow\s+much\b",
        r"\bpercentage\b",
        r"\baverage\b",
        r"\bsum\s+of\b",
        r"\bstatistics?\b",
        r"\banalyze\b",
        r"\bdata\b",
        r"\b\d+\s*[\+\-\*\/\%]\s*\d+\b",  # Math expressions like "15 + 20"
        r"\b(?:if\s+i\s+have|i\s+have)\s+\d+\b",  # "if I have 15 apples"
    ],
    "reasoner": [
        # Reasoning tasks that require step-by-step thinking
        r"\bhow\s+many\s+(?:times|letters?|words?|r'?s|occurrences?)\b",  # counting in strings
        r"\bcount\s+(?:the|how)\b",
        r"\bwhy\s+(?:does|do|is|are|did|would|should)\b",  # why questions needing explanation
        r"\bexplain\s+(?:how|why|the)\b",
        r"\breason(?:ing)?\b",
        r"\blogic(?:al|ally)?\b",
        r"\bprove\b",
        r"\bstep\s*by\s*step\b",
        r"\b(?:true|false)\s+(?:that|if)\b",
    ],
    # Simple factual questions - should NOT use complex reasoning
    "simple_factual": [
        r"\bhow\s+many\s+(?:legs?|eyes?|ears?|arms?|fingers?|toes?|wheels?|sides?)\b",
        r"\bwhat\s+(?:color|colour)\s+is\b",
        r"\bwhat\s+is\s+(?:the\s+)?(?:capital|largest|smallest|tallest)\b",
        r"\bwho\s+(?:is|was|are|were)\b",
        r"\bwhen\s+(?:is|was|did)\b",
        r"\bwhere\s+(?:is|was|are|were)\b",
    ],
    # Web search triggers - current events, real-time data
    "web_search": [
        r"\bcurrent(?:ly)?\b",
        r"\blatest\b",
        r"\btoday'?s?\b",
        r"\bright\s+now\b",
        r"\brecent(?:ly)?\b",
        r"\bnews\b",
        r"\bprice\s+of\b",
        r"\bweather\b",
        r"\bsearch\s+(?:for|the)?\b",
        r"\blook\s+up\b",
        r"\bfind\s+(?:out|me)\b",
        r"\b(?:2024|2025|2026)\b",  # Current/recent years
        r"\byesterday\b",
        r"\blast\s+(?:week|month|year)\b",
        r"\bwho\s+won\b",
        r"\bscore\s+of\b",
    ],
}


def detect_mode(prompt: str) -> tuple[str, bool, bool]:
    """
    Auto-detect the best behavior, whether JSON is expected, and if web search is needed.

    Args:
        prompt: The user's input prompt.

    Returns:
        Tuple of (behavior, expect_json, web_search).
    """
    prompt_lower = prompt.lower()

    # Check for JSON patterns first
    expect_json = any(re.search(pattern, prompt_lower) for pattern in PATTERNS["json"])

    # Check for web search patterns
    web_search = any(
        re.search(pattern, prompt_lower) for pattern in PATTERNS["web_search"]
    )

    # If JSON detected, default to coder persona
    if expect_json:
        return "coder", True, web_search

    # Check for simple factual questions FIRST - these should use general, not reasoner
    if any(re.search(pattern, prompt_lower) for pattern in PATTERNS["simple_factual"]):
        return "general", False, web_search

    # Check other patterns in priority order
    for behavior in ["coder", "analyst", "reasoner"]:
        if any(re.search(pattern, prompt_lower) for pattern in PATTERNS[behavior]):
            return behavior, False, web_search

    # Default
    return "general", False, web_search


def main():
    print("=" * 50)
    print("MicroThink Chat (Auto-Detect Mode)")
    print("=" * 50)
    print()
    print("Commands:")
    print("  /debug    - Toggle debug mode (show thinking)")
    print("  /brief    - Toggle brief mode (just answer, no explanation)")
    print("  /web      - Toggle web search mode")
    print("  /auto     - Toggle auto-detect mode")
    print("  /json     - Toggle JSON mode (manual)")
    print("  /coder    - Force coder persona")
    print("  /analyst  - Force analyst persona")
    print("  /reasoner - Force reasoner persona")
    print("  /general  - Force general persona")
    print("  /quit     - Exit")
    print()
    print("Auto-detect is ON. Persona and web search chosen based on your query.")
    print("Debug mode is ON. Answers include explanations by default.")
    print()

    # Initialize client
    client = MicroThinkClient(model="llama3.2:3b")

    # Settings
    debug = True
    brief = False  # Detailed answers by default
    auto_detect = True
    manual_json = False
    manual_web = False  # Manual web search toggle
    manual_behavior = "general"

    while True:
        try:
            # Show current mode
            if auto_detect:
                mode_display = "auto"
            else:
                mode_display = manual_behavior

            if manual_web:
                mode_display += "+web"

            prompt = input(f"\n[{mode_display}] You: ").strip()

            if not prompt:
                continue

            # Handle commands
            command = None
            if prompt.startswith("/"):
                parts = prompt.split(maxsplit=1)
                command = parts[0].lower()
                remaining = parts[1] if len(parts) > 1 else None

            if command == "/quit":
                print("Goodbye!")
                break
            elif command == "/debug":
                debug = not debug
                print(f"Debug mode: {'ON' if debug else 'OFF'}")
                if remaining:
                    prompt = remaining
                else:
                    continue
            elif command == "/brief":
                brief = not brief
                print(
                    f"Brief mode: {'ON' if brief else 'OFF'} (answers {'without' if brief else 'with'} explanation)"
                )
                if remaining:
                    prompt = remaining
                else:
                    continue
            elif command == "/auto":
                auto_detect = not auto_detect
                print(f"Auto-detect: {'ON' if auto_detect else 'OFF'}")
                if remaining:
                    prompt = remaining
                else:
                    continue
            elif command == "/web":
                manual_web = not manual_web
                print(f"Web search: {'ON' if manual_web else 'OFF'}")
                if remaining:
                    prompt = remaining
                else:
                    continue
            elif command == "/json":
                manual_json = not manual_json
                print(f"JSON mode: {'ON' if manual_json else 'OFF'}")
                if remaining:
                    prompt = remaining
                else:
                    continue
            elif command in ["/coder", "/analyst", "/reasoner", "/general"]:
                manual_behavior = command[1:]
                auto_detect = False  # Disable auto when manually selecting
                print(f"Switched to: {manual_behavior} (auto-detect OFF)")
                if remaining:
                    prompt = remaining
                else:
                    continue
            elif command and command.startswith("/"):
                print(f"Unknown command: {command}")
                continue

            # Detect or use manual settings
            if auto_detect:
                behavior, expect_json, web_search = detect_mode(prompt)
                # Manual overrides
                if manual_json:
                    expect_json = True
                if manual_web:
                    web_search = True
                if debug:
                    flags = []
                    if expect_json:
                        flags.append("json")
                    if web_search:
                        flags.append("web")
                    flag_str = ", " + ", ".join(flags) if flags else ""
                    print(f"  [auto: {behavior}{flag_str}]")
            else:
                behavior = manual_behavior
                expect_json = manual_json
                web_search = manual_web

            # Generate response
            response = client.generate(
                prompt=prompt,
                behavior=behavior,
                expect_json=expect_json,
                debug=debug,
                brief=brief,
                web_search=web_search,
            )

            print(f"\nAssistant: {response}")

        except MicroThinkError as e:
            print(f"\nError: {e.message}")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
