"""Test configuration for pytest."""

import pytest

pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def anyio_backend():
    """Set anyio backend for async tests."""
    return "asyncio"
