"""Tests for standard logging integration."""

import logging
from io import StringIO

import pytest


class TestLoggingConfiguration:
    """Tests for logging configuration."""

    def test_get_logger_returns_logger(self):
        """get_logger returns a Python logger."""
        from microthink.utils.logging_config import get_logger

        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)

    def test_logger_name_prefix(self):
        """Logger names are prefixed with 'microthink'."""
        from microthink.utils.logging_config import get_logger

        logger = get_logger("client")
        assert logger.name == "microthink.client"

    def test_get_root_logger(self):
        """Empty name returns root microthink logger."""
        from microthink.utils.logging_config import get_logger

        logger = get_logger()
        assert logger.name == "microthink"

    def test_configure_logging_sets_level(self):
        """configure_logging sets the log level."""
        from microthink.utils.logging_config import configure_logging, get_logger

        configure_logging(level=logging.DEBUG)
        logger = get_logger()
        assert logger.isEnabledFor(logging.DEBUG)

    def test_configure_logging_accepts_string_level(self):
        """configure_logging accepts string level names."""
        from microthink.utils.logging_config import configure_logging

        configure_logging(level="INFO")  # Should not raise


class TestClientLoggingIntegration:
    """Tests for logging in MicroThinkClient."""

    def test_client_accepts_logger_param(self):
        """Client accepts logger configuration."""
        from microthink import MicroThinkClient

        client = MicroThinkClient(logger="standard")
        assert hasattr(client, "_use_rich_logging")
        assert client._use_rich_logging is False

    def test_client_default_uses_rich(self):
        """Client uses Rich logging by default."""
        from microthink import MicroThinkClient

        client = MicroThinkClient()
        assert client._use_rich_logging is True

    def test_client_rich_logger_option(self):
        """Setting logger='rich' enables Rich."""
        from microthink import MicroThinkClient

        client = MicroThinkClient(logger="rich")
        assert client._use_rich_logging is True

    def test_client_standard_logging_logs_to_python_logger(self):
        """Setting logger='standard' uses Python logging."""
        import logging
        from io import StringIO

        from microthink import MicroThinkClient
        from microthink.utils.logging_config import configure_logging

        # Set up a handler to capture logs
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        configure_logging(level="DEBUG", handler=handler)

        client = MicroThinkClient(logger="standard")
        # The client should use standard logging, not Rich
        assert client._use_rich_logging is False
