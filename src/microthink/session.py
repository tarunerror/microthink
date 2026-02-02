"""
Conversation session management for MicroThink.

Provides stateful conversation support with history tracking,
persistence, and automatic history trimming.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from microthink.client import MicroThinkClient


class Session:
    """
    A conversation session with history tracking.

    Example:
        >>> client = MicroThinkClient()
        >>> session = Session(client)
        >>> session.generate("My name is Alice")
        >>> session.generate("What's my name?")
        'Your name is Alice'

        >>> # Save and restore
        >>> session.save("chat.json")
        >>> restored = Session.load(client, "chat.json")
    """

    def __init__(
        self,
        client: MicroThinkClient,
        max_history: int = 20,
        behavior: str = "general",
    ) -> None:
        """
        Initialize a conversation session.

        Args:
            client: The MicroThinkClient to use.
            max_history: Maximum messages to keep in history.
            behavior: The persona to use for this session.
        """
        self.client = client
        self.max_history = max_history
        self.behavior = behavior
        self._history: List[Dict[str, str]] = []

    @property
    def history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self._history.copy()

    def generate(
        self,
        prompt: str,
        expect_json: bool = False,
        web_search: bool = False,
        debug: bool = False,
    ) -> Union[str, Dict[str, Any], List[Any]]:
        """
        Generate a response within the session context.

        Args:
            prompt: The user's input prompt.
            expect_json: If True, parse JSON output.
            web_search: If True, include web search context.
            debug: If True, output debug information.

        Returns:
            The model's response.
        """
        # Build conversation context from history
        context_parts = []
        for msg in self._history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                context_parts.append(f"User: {content}")
            elif role == "assistant":
                context_parts.append(f"Assistant: {content}")

        # Create enhanced prompt with history context
        if context_parts:
            context = "\n".join(context_parts)
            enhanced_prompt = f"Previous conversation:\n{context}\n\nUser: {prompt}"
        else:
            enhanced_prompt = prompt

        # Delegate to client's generate method (handles web_search, debug, JSON retries)
        result = self.client.generate(
            prompt=enhanced_prompt,
            behavior=self.behavior,
            expect_json=expect_json,
            web_search=web_search,
            debug=debug,
        )

        # Add user message to history
        self._history.append({"role": "user", "content": prompt})

        # Add assistant response to history
        if isinstance(result, str):
            self._history.append({"role": "assistant", "content": result})
        else:
            # For JSON results, store as string representation
            self._history.append({"role": "assistant", "content": json.dumps(result)})

        # Trim history if needed
        self._trim_history()

        return result

    def _trim_history(self) -> None:
        """Trim history to max_history size."""
        if len(self._history) > self.max_history:
            # Keep the most recent messages
            self._history = self._history[-self.max_history :]

    def clear(self) -> None:
        """Clear the conversation history."""
        self._history = []

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the session to a JSON file.

        Args:
            path: Path to save the session.
        """
        data = {
            "history": self._history,
            "behavior": self.behavior,
            "max_history": self.max_history,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, client: MicroThinkClient, path: Union[str, Path]) -> "Session":
        """
        Load a session from a JSON file.

        Args:
            client: The MicroThinkClient to use.
            path: Path to the saved session.

        Returns:
            A restored Session instance.
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        session = cls(
            client=client,
            max_history=data.get("max_history", 20),
            behavior=data.get("behavior", "general"),
        )
        session._history = data.get("history", [])

        return session
