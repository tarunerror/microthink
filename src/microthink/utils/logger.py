"""
Rich logging utilities for MicroThink debug mode.

Provides beautiful console output for debugging the Chain-of-Thought
reasoning process and JSON correction attempts.
"""

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Shared console instance
console = Console()


def log_thinking(content: str) -> None:
    """
    Log the model's thinking/reasoning process.

    Args:
        content: The content from <thinking> tags.

    Example:
        >>> log_thinking("Step 1: Count the letters...")
        # Displays a yellow panel with the thinking content
    """
    panel = Panel(
        content,
        title="[bold yellow]Thought Process[/bold yellow]",
        border_style="yellow",
        padding=(1, 2),
    )
    console.print(panel)


def log_answer(content: str, is_json: bool = False) -> None:
    """
    Log the model's final answer.

    Args:
        content: The content from <answer> tags.
        is_json: Whether this is a JSON response.

    Example:
        >>> log_answer('{"result": 42}', is_json=True)
        # Displays a green panel with the JSON content
    """
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
    """
    Log a JSON correction retry attempt.

    Args:
        attempt: Current attempt number (1-indexed).
        max_attempts: Maximum number of attempts allowed.
        error: The error message from the failed JSON parse.

    Example:
        >>> log_retry(1, 3, "Expecting ',' delimiter")
        # Displays a red warning about the retry
    """
    text = Text()
    text.append(f"[Reflexion] ", style="bold red")
    text.append(f"Retry {attempt}/{max_attempts} - ", style="yellow")
    text.append(f"Error: {error}", style="dim")
    console.print(text)


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
