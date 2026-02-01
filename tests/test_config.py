"""Tests for config file support."""

import tempfile
from pathlib import Path

import pytest


class TestConfig:
    """Tests for Config class."""

    def test_config_importable(self):
        """Config class is importable."""
        from microthink.config import Config

        assert Config is not None

    def test_config_has_defaults(self):
        """Config has default values."""
        from microthink.config import Config

        config = Config()
        assert config.model == "llama3.2:3b"
        assert config.host == "http://localhost:11434"
        assert config.cache is False
        assert config.debug is False

    def test_config_from_dict(self):
        """Config can be created from dict."""
        from microthink.config import Config

        data = {"model": "phi3:mini", "cache": True}
        config = Config.from_dict(data)

        assert config.model == "phi3:mini"
        assert config.cache is True

    def test_config_from_yaml_file(self):
        """Config can be loaded from YAML file."""
        from microthink.config import Config

        yaml_content = """
model: llama3.1:8b
host: http://localhost:11434
cache: true
cache_ttl: 7200
"""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write(yaml_content)
            f.flush()

            config = Config.from_file(f.name)

            assert config.model == "llama3.1:8b"
            assert config.cache is True
            assert config.cache_ttl == 7200

    def test_config_to_dict(self):
        """Config can be exported to dict."""
        from microthink.config import Config

        config = Config(model="test", cache=True)
        data = config.to_dict()

        assert data["model"] == "test"
        assert data["cache"] is True


class TestConfigLoading:
    """Tests for config file discovery."""

    def test_load_config_returns_defaults(self):
        """load_config returns defaults when no file found."""
        from microthink.config import load_config

        config = load_config(search_paths=[])
        assert config.model == "llama3.2:3b"

    def test_load_config_from_explicit_path(self):
        """load_config loads from explicit path."""
        from microthink.config import load_config

        yaml_content = "model: custom-model\n"
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write(yaml_content)
            f.flush()

            config = load_config(path=f.name)
            assert config.model == "custom-model"


class TestClientConfig:
    """Tests for config integration with client."""

    def test_client_accepts_config(self):
        """Client accepts config parameter."""
        from microthink.config import Config

        from microthink import MicroThinkClient

        config = Config(model="test-model")
        client = MicroThinkClient(config=config)

        assert client.model == "test-model"
