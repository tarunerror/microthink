"""Tests for conversation sessions."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestSession:
    """Tests for Session class."""

    def test_session_importable(self):
        """Session is importable from package."""
        from microthink import Session

        assert Session is not None

    def test_session_created_from_client(self):
        """Session can be created from client."""
        from microthink import MicroThinkClient, Session

        client = MicroThinkClient()
        session = Session(client)
        assert session.client is client

    def test_session_has_generate_method(self):
        """Session has generate method."""
        from microthink import MicroThinkClient, Session

        client = MicroThinkClient()
        session = Session(client)
        assert hasattr(session, "generate")
        assert callable(session.generate)

    def test_session_tracks_history(self):
        """Session tracks message history."""
        from microthink import MicroThinkClient, Session

        with patch("microthink.client.ollama.Client") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.return_value = mock_client
            mock_client.chat.return_value = {
                "message": {"content": "<answer>Hello!</answer>"}
            }

            client = MicroThinkClient()
            session = Session(client)
            session.generate("Hi there")

            assert len(session.history) >= 2  # user + assistant

    def test_session_includes_history_in_next_call(self):
        """Session includes history in subsequent calls."""
        from microthink import MicroThinkClient, Session

        with patch("microthink.client.ollama.Client") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.return_value = mock_client
            mock_client.chat.return_value = {
                "message": {"content": "<answer>Response</answer>"}
            }

            client = MicroThinkClient()
            session = Session(client)
            session.generate("First message")
            session.generate("Second message")

            # Check last call included history
            last_call = mock_client.chat.call_args
            messages = last_call[1]["messages"]
            # Should have system + first user + first assistant + second user
            assert len(messages) >= 4

    def test_session_max_history(self):
        """Session respects max_history limit."""
        from microthink import MicroThinkClient, Session

        with patch("microthink.client.ollama.Client") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.return_value = mock_client
            mock_client.chat.return_value = {
                "message": {"content": "<answer>Response</answer>"}
            }

            client = MicroThinkClient()
            session = Session(client, max_history=4)

            # Generate more than max_history
            for i in range(5):
                session.generate(f"Message {i}")

            # History should be trimmed
            assert len(session.history) <= 4

    def test_session_clear(self):
        """Session can clear history."""
        from microthink import MicroThinkClient, Session

        with patch("microthink.client.ollama.Client") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.return_value = mock_client
            mock_client.chat.return_value = {
                "message": {"content": "<answer>Response</answer>"}
            }

            client = MicroThinkClient()
            session = Session(client)
            session.generate("Hello")
            session.clear()

            assert len(session.history) == 0


class TestSessionPersistence:
    """Tests for session save/load."""

    def test_session_save(self):
        """Session can be saved to file."""
        from microthink import MicroThinkClient, Session

        with patch("microthink.client.ollama.Client") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.return_value = mock_client
            mock_client.chat.return_value = {
                "message": {"content": "<answer>Response</answer>"}
            }

            client = MicroThinkClient()
            session = Session(client)
            session.generate("Hello")

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                session.save(f.name)
                assert Path(f.name).exists()

                # Verify content
                with open(f.name) as saved:
                    data = json.load(saved)
                    assert "history" in data

    def test_session_load(self):
        """Session can be loaded from file."""
        from microthink import MicroThinkClient, Session

        client = MicroThinkClient()

        # Create a session file
        session_data = {
            "history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            "behavior": "general",
        }

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(session_data, f)
            f.flush()

            loaded = Session.load(client, f.name)
            assert len(loaded.history) == 2
