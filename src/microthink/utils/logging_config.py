"""
Standard logging configuration for MicroThink.

Provides Python standard library logging integration,
with Rich as an optional enhancer for pretty console output.
"""

import logging
from typing import Optional, Union

LOGGER_NAME = "microthink"


def get_logger(name: str = "") -> logging.Logger:
    """
    Get a logger for a MicroThink module.

    Args:
        name: The module name (prefixed with 'microthink.').
              Empty string returns the root microthink logger.

    Returns:
        A configured Logger instance.

    Example:
        >>> logger = get_logger("client")
        >>> logger.name
        'microthink.client'
    """
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)


def configure_logging(
    level: Union[int, str] = logging.WARNING,
    format: Optional[str] = None,
    handler: Optional[logging.Handler] = None,
) -> None:
    """
    Configure MicroThink logging.

    Args:
        level: Log level (int or string like "DEBUG", "INFO").
        format: Log message format. Defaults to standard format.
        handler: Custom handler. Defaults to StreamHandler.

    Example:
        >>> configure_logging(level="DEBUG")
    """
    logger = get_logger()

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.WARNING)

    logger.setLevel(level)

    if handler:
        logger.addHandler(handler)
    elif not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        if format is None:
            format = "[%(levelname)s] %(name)s: %(message)s"

        formatter = logging.Formatter(format)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
