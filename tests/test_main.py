"""Tests for Forma proxy."""

import pytest
from fastapi.testclient import TestClient

from forma.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_health(client):
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_openapi_available(client):
    """Test that OpenAPI schema is available."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "/v1/chat/completions" in schema["paths"]


def test_docs_available(client):
    """Test that docs endpoint is available."""
    response = client.get("/docs")
    assert response.status_code == 200
