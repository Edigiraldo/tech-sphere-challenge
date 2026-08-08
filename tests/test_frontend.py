"""Tests for the frontend static-file serving and page routes.

These tests verify that the FastAPI app correctly serves the static frontend
HTML, CSS, and JS assets, and that existing API routes are not disrupted by
the static mount.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from backend.main import app


# ---------------------------------------------------------------------------
# HTML page routes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_index_html_returns_200():
    """GET / should return the index.html page."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")

    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "text/html" in content_type
    body = response.text
    # Key content checks
    assert "Tech Sphere Challenge" in body
    assert "Seleccionar Paciente" in body
    assert "patient-select" in body
    assert "Iniciar Llamada" in body


@pytest.mark.asyncio
async def test_call_html_returns_200():
    """GET /call should return the call.html page."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/call")

    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "text/html" in content_type
    body = response.text
    # Key content checks
    assert "Llamada en Curso" in body
    assert "call-state-badge" in body
    assert "Transcripción" in body
    assert "Historial de Conversación" in body
    assert "Respuesta de Audio" in body


# ---------------------------------------------------------------------------
# Static assets
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_styles_css_served():
    """GET /static/styles.css should return CSS with correct content-type."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/static/styles.css")

    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "text/css" in content_type
    # Verify it's not an error page by checking for expected CSS content
    body = response.text
    assert "Tech Sphere Challenge" in body  # CSS comment header


@pytest.mark.asyncio
async def test_app_js_served():
    """GET /static/app.js should return JavaScript with correct content-type."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/static/app.js")

    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "javascript" in content_type or "text/javascript" in content_type
    body = response.text
    assert "PATIENTS" in body
    assert "patient-select" in body


@pytest.mark.asyncio
async def test_call_js_served():
    """GET /static/call.js should return JavaScript."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/static/call.js")

    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "javascript" in content_type or "text/javascript" in content_type
    body = response.text
    assert "currentState" in body
    assert "toggleRecording" in body


@pytest.mark.asyncio
async def test_data_js_served():
    """GET /static/data.js should return JavaScript with shared PATIENTS data."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/static/data.js")

    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "javascript" in content_type or "text/javascript" in content_type
    body = response.text
    assert "PATIENTS" in body
    assert "Paciente 001" in body
    assert "Apendicectomía laparoscópica" in body


# ---------------------------------------------------------------------------
# Regression: API routes must still work
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_still_works():
    """GET /health must return {"status": "ok"} even with frontend mounted."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# 404 for non-existent paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_nonexistent_static_returns_404():
    """GET /static/nonexistent.file should return 404."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/static/nonexistent.file")

    assert response.status_code == 404
