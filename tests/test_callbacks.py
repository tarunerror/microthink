"""Tests for callbacks/hooks."""

from unittest.mock import MagicMock, patch

import pytest


class TestCallbackTypes:
    """Tests for callback type definitions."""

    def test_callbacks_importable(self):
        """Callbacks module is importable."""
        from microthink.callbacks import Callbacks

        assert Callbacks is not None

    def test_callbacks_has_fields(self):
        """Callbacks has all hook fields."""
        from microthink.callbacks import Callbacks

        cb = Callbacks()
        assert hasattr(cb, "on_request_start")
        assert hasattr(cb, "on_request_end")
        assert hasattr(cb, "on_retry")
        assert hasattr(cb, "on_cache_hit")


class TestClientCallbacks:
    """Tests for callbacks integration with client."""

    def test_client_accepts_callbacks(self):
        """Client accepts callbacks parameter."""
        from microthink import MicroThinkClient
        from microthink.callbacks import Callbacks

        callbacks = Callbacks()
        client = MicroThinkClient(callbacks=callbacks)
        assert client.callbacks is callbacks

    def test_on_request_start_called(self):
        """on_request_start is called before generation."""
        from microthink import MicroThinkClient
        from microthink.callbacks import Callbacks

        mock_callback = MagicMock()
        callbacks = Callbacks(on_request_start=mock_callback)

        with patch("microthink.client.ollama.Client") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.return_value = mock_client
            mock_client.chat.return_value = {
                "message": {"content": "<answer>Response</answer>"}
            }

            client = MicroThinkClient(callbacks=callbacks)
            client.generate("Test prompt")

            mock_callback.assert_called_once()

    def test_on_thinking_called(self):
        """on_thinking is called when thinking content is present."""
        from microthink import MicroThinkClient
        from microthink.callbacks import Callbacks

        mock_callback = MagicMock()
        callbacks = Callbacks(on_thinking=mock_callback)

        with patch("microthink.client.ollama.Client") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.return_value = mock_client
            mock_client.chat.return_value = {
                "message": {
                    "content": "<thinking>Some thoughts</thinking><answer>Response</answer>"
                }
            }

            client = MicroThinkClient(callbacks=callbacks)
            client.generate("Test prompt")

            mock_callback.assert_called_once_with("Some thoughts")

    def test_on_answer_called(self):
        """on_answer is called with the answer content."""
        from microthink import MicroThinkClient
        from microthink.callbacks import Callbacks

        mock_callback = MagicMock()
        callbacks = Callbacks(on_answer=mock_callback)

        with patch("microthink.client.ollama.Client") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.return_value = mock_client
            mock_client.chat.return_value = {
                "message": {"content": "<answer>Response</answer>"}
            }

            client = MicroThinkClient(callbacks=callbacks)
            client.generate("Test prompt")

            mock_callback.assert_called_once_with("Response")

    def test_on_request_end_called(self):
        """on_request_end is called after generation."""
        from microthink import MicroThinkClient
        from microthink.callbacks import Callbacks

        mock_callback = MagicMock()
        callbacks = Callbacks(on_request_end=mock_callback)

        with patch("microthink.client.ollama.Client") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.return_value = mock_client
            mock_client.chat.return_value = {
                "message": {"content": "<answer>Response</answer>"}
            }

            client = MicroThinkClient(callbacks=callbacks)
            client.generate("Test prompt")

            mock_callback.assert_called_once()

    def test_on_retry_called_on_json_error(self):
        """on_retry is called when JSON parsing fails."""
        from microthink import MicroThinkClient
        from microthink.callbacks import Callbacks

        mock_callback = MagicMock()
        callbacks = Callbacks(on_retry=mock_callback)

        with patch("microthink.client.ollama.Client") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.return_value = mock_client
            mock_client.chat.side_effect = [
                {"message": {"content": "<answer>{invalid json</answer>"}},
                {"message": {"content": '<answer>{"valid": true}</answer>'}},
            ]

            client = MicroThinkClient(callbacks=callbacks)
            try:
                client.generate("Test", expect_json=True)
            except Exception:
                pass

            assert mock_callback.call_count >= 1

    def test_on_cache_hit_called(self):
        """on_cache_hit is called when cache is used."""
        from microthink import MicroThinkClient
        from microthink.callbacks import Callbacks

        mock_callback = MagicMock()
        callbacks = Callbacks(on_cache_hit=mock_callback)

        with patch("microthink.client.ollama.Client") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.return_value = mock_client
            mock_client.chat.return_value = {
                "message": {"content": "<answer>Response</answer>"}
            }

            client = MicroThinkClient(cache=True, callbacks=callbacks)
            client.generate("Test prompt")
            client.generate("Test prompt")

            mock_callback.assert_called_once()
