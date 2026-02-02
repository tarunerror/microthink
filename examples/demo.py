#!/usr/bin/env python3
"""
MicroThink Demo - Showcasing enhanced LLM capabilities.

This demo illustrates how MicroThink improves small model performance
through Chain-of-Thought reasoning and JSON self-correction.

Requirements:
    - Ollama running locally with llama3.2:3b (or another model)
    - pip install microthink

Usage:
    python demo.py
"""

from microthink import MicroThinkClient, MicroThinkError


def demo_reasoning_task() -> None:
    """
    Demo 1: Logic Task - The "Strawberry" Problem.

    Small models often fail at counting letters because they don't
    think step-by-step. MicroThink's silent CoT injection fixes this.
    """
    print("=" * 60)
    print("DEMO 1: Reasoning Task (Letter Counting)")
    print("=" * 60)
    print()
    print("Question: How many 'r's are in 'strawberry'?")
    print()
    print("Without CoT, small models often answer '2' (wrong).")
    print("With MicroThink's silent CoT, they reason through it...")
    print()

    client = MicroThinkClient(model="llama3.2:3b")

    # debug=True shows the thinking process
    answer = client.generate(
        prompt="How many letter 'r's are in the word 'strawberry'? Count carefully.",
        behavior="reasoner",
        debug=True,
    )

    print()
    print(f"Final Answer: {answer}")
    print()


def demo_json_extraction() -> None:
    """
    Demo 2: Coding Task - JSON Output with Self-Correction.

    When expect_json=True, MicroThink validates the output and
    automatically retries if the model produces invalid JSON.
    """
    print("=" * 60)
    print("DEMO 2: JSON Extraction with Self-Correction")
    print("=" * 60)
    print()
    print("Request: Return a JSON list of 3 Python keywords.")
    print()
    print("MicroThink will:")
    print("  1. Force structured output with <answer> tags")
    print("  2. Validate JSON syntax")
    print("  3. Auto-retry if parsing fails (up to 3 times)")
    print()

    client = MicroThinkClient(model="llama3.2:3b")

    try:
        data = client.generate(
            prompt="Return a JSON list of exactly 3 Python reserved keywords. Just the keywords as strings in an array.",
            behavior="coder",
            expect_json=True,
            debug=True,
        )

        print()
        print(f"Parsed JSON: {data}")
        print(f"Type: {type(data).__name__}")
        print()

    except MicroThinkError as e:
        print(f"Failed: {e}")
        print(f"Last output: {e.last_output}")


def demo_structured_data() -> None:
    """
    Demo 3: Complex Structured Data with Schema.

    Using generate_with_schema() to get JSON matching a specific structure.
    """
    print("=" * 60)
    print("DEMO 3: Structured Data with Schema")
    print("=" * 60)
    print()

    client = MicroThinkClient(model="llama3.2:3b")

    schema = {
        "name": "string",
        "language": "string",
        "year_created": "number",
        "paradigms": ["string"],
    }

    print(f"Schema: {schema}")
    print()

    try:
        data = client.generate_with_schema(
            prompt="Describe the Python programming language",
            schema=schema,
            behavior="analyst",
            debug=True,
        )

        print()
        print(f"Result: {data}")
        print()

    except MicroThinkError as e:
        print(f"Failed: {e}")


def demo_analyst_persona() -> None:
    """
    Demo 4: Analyst Persona - Data Analysis.

    The 'analyst' persona is optimized for precise calculations.
    """
    print("=" * 60)
    print("DEMO 4: Analyst Persona")
    print("=" * 60)
    print()

    client = MicroThinkClient(model="llama3.2:3b")

    answer = client.generate(
        prompt="If I have 15 apples and give away 40% of them, then buy 3 more, how many do I have?",
        behavior="analyst",
        debug=True,
    )

    print()
    print(f"Answer: {answer}")
    print()


def main() -> None:
    """Run all demos."""
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║           MicroThink v0.1.0 - Demo Suite                 ║")
    print("║   Making Small LLMs Think Bigger                         ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    demos = [
        ("Reasoning", demo_reasoning_task),
        ("JSON Extraction", demo_json_extraction),
        ("Structured Data", demo_structured_data),
        ("Analyst Persona", demo_analyst_persona),
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            demo_func()
        except Exception as e:
            print(f"Demo '{name}' failed: {e}")
            print("(Make sure Ollama is running with the specified model)")
            import traceback

            traceback.print_exc()
            print()

        if i < len(demos):
            print()
            try:
                input("Press Enter to continue to next demo...")
            except (EOFError, OSError):
                print("Non-interactive mode detected. Continuing...")
            print()

    print("=" * 60)
    print("All demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
