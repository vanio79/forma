"""Tests for configuration."""

import pytest

from forma.config import Settings


def test_default_settings():
    """Test default settings values."""
    settings = Settings()
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.debug is False
    assert settings.upstream_base_url == "https://api.openai.com/v1"
    assert settings.upstream_timeout == 300.0


def test_model_mapping_empty():
    """Test empty model mapping."""
    settings = Settings(model_mapping="")
    assert settings.get_model_mapping() == {}


def test_model_mapping_single():
    """Test single model mapping."""
    settings = Settings(model_mapping="gpt-4:llama-3-70b")
    assert settings.get_model_mapping() == {"gpt-4": "llama-3-70b"}


def test_model_mapping_multiple():
    """Test multiple model mappings."""
    settings = Settings(model_mapping="gpt-4:llama-3-70b, gpt-3.5-turbo:llama-3-8b")
    mapping = settings.get_model_mapping()
    assert mapping == {
        "gpt-4": "llama-3-70b",
        "gpt-3.5-turbo": "llama-3-8b",
    }
