"""Tests for the health check endpoint."""

import pytest
from httpx import ASGITransport, AsyncClient

from backend.main import app


@pytest.mark.asyncio
async def test_health_returns_ok():
    """GET /health should return {"status": "ok"} with a 200 status code."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
