"""
Configuration file support for MicroThink.

Provides YAML-based configuration with file discovery
and sensible defaults.
"""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Try to import yaml, make it optional
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


DEFAULT_CONFIG_PATHS = [
    Path(".microthink.yaml"),
    Path(".microthink.yml"),
    Path.home() / ".microthink" / "config.yaml",
    Path.home() / ".microthink" / "config.yml",
]


@dataclass
class Config:
    """
    Configuration for MicroThink.

    Example:
        >>> config = Config.from_file("config.yaml")
        >>> client = MicroThinkClient(config=config)
    """

    # Model settings
    model: str = "llama3.2:3b"
    host: str = "http://localhost:11434"

    # Cache settings
    cache: bool = False
    cache_ttl: float = 3600.0
    cache_max_size: int = 1000

    # Behavior settings
    default_behavior: str = "general"
    debug: bool = False

    # Logging
    logger: str = "rich"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create a Config from a dictionary."""
        known_fields = {
            "model",
            "host",
            "cache",
            "cache_ttl",
            "cache_max_size",
            "default_behavior",
            "debug",
            "logger",
        }
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Config":
        """Load a Config from a YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for config file support. "
                "Install with: pip install pyyaml"
            )

        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Export config as a dictionary."""
        return asdict(self)


def load_config(
    path: Optional[Union[str, Path]] = None,
    search_paths: Optional[List[Path]] = None,
) -> Config:
    """Load configuration with file discovery."""
    if path:
        return Config.from_file(path)

    if search_paths is None:
        search_paths = DEFAULT_CONFIG_PATHS

    for config_path in search_paths:
        if config_path.exists():
            try:
                return Config.from_file(config_path)
            except ImportError:
                # PyYAML not installed - skip config files
                logging.debug(f"Skipping {config_path}: PyYAML not installed")
                continue
            except Exception as e:
                # Log the error instead of silently swallowing
                logging.warning(f"Failed to load config from {config_path}: {e}")
                continue

    return Config()
