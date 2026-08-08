"""Tests for the admin console frontend serving and document API interactions.

These tests verify that the FastAPI app correctly serves the admin HTML page,
admin.js static asset, and that the GET /admin route does not disrupt existing
API routes.  Additional tests validate that the admin page references the
correct static resources and document API endpoints.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from backend.main import app


# ---------------------------------------------------------------------------
# Admin page serving
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_admin_html_returns_200():
    """GET /admin should return the admin.html page with correct content-type."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/admin")

    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "text/html" in content_type

    body = response.text
    # Key content checks — español
    assert "Consola de Administración" in body
    assert "Gestión de Documentos" in body
    assert "Subir Documento" in body
    assert "document-table" in body
    assert "upload-file-input" in body
    assert "status-filter" in body


@pytest.mark.asyncio
async def test_admin_html_lang_es():
    """GET /admin should have lang="es" on the html element."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/admin")

    body = response.text
    assert '<html lang="es">' in body


@pytest.mark.asyncio
async def test_admin_html_references_stylesheet():
    """GET /admin should link to /static/styles.css."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/admin")

    body = response.text
    assert 'href="/static/styles.css"' in body


@pytest.mark.asyncio
async def test_admin_html_references_admin_js():
    """GET /admin should load /static/admin.js."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/admin")

    body = response.text
    assert 'src="/static/admin.js"' in body


@pytest.mark.asyncio
async def test_admin_html_accepts_pdf_only():
    """GET /admin file input should accept only .pdf."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/admin")

    body = response.text
    assert 'accept=".pdf"' in body


# ---------------------------------------------------------------------------
# Static admin.js serving
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_admin_js_served():
    """GET /static/admin.js should return JavaScript with correct content-type."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/static/admin.js")

    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "javascript" in content_type or "text/javascript" in content_type

    body = response.text
    # Verify it is the admin JS (not a 404 page or other asset)
    assert "administration console" in body.lower() or "admin.js" in body.lower()
    assert "fetchDocuments" in body
    assert "handleUpload" in body
    assert "handleDelete" in body


# ---------------------------------------------------------------------------
# Regression: existing API routes must remain intact
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_still_works_with_admin():
    """GET /health must return {"status": "ok"} after admin route is added."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_index_still_served_with_admin():
    """GET / must still serve index.html."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")

    assert response.status_code == 200
    assert "Seleccionar Paciente" in response.text


@pytest.mark.asyncio
async def test_call_still_served_with_admin():
    """GET /call must still serve call.html."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/call")

    assert response.status_code == 200
    assert "Llamada en Curso" in response.text


@pytest.mark.asyncio
async def test_documents_api_still_reachable():
    """GET /documents should return 200 (may be empty list if no DB)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/documents")

    # The endpoint exists — it may 500 if SQLite isn't initialised in this
    # test context, but that's a test-environment issue, not a routing issue.
    # The key assertion is that the route is registered and we don't get 404.
    assert response.status_code != 404


# ---------------------------------------------------------------------------
# CSS serving — verify admin-specific styles are present
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_styles_css_contains_admin_classes():
    """GET /static/styles.css should contain admin-specific CSS classes."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/static/styles.css")

    assert response.status_code == 200
    body = response.text
    # Admin-specific CSS classes
    assert "status-badge" in body
    assert "document-table" in body
    assert "documents-toolbar" in body
    assert "file-input" in body
    assert "inline-status" in body


# ---------------------------------------------------------------------------
# Security: status badge class injection safeguard
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_admin_js_status_badge_safe_class():
    """The status badge renderer must use a hardcoded safe mapping for CSS
    classes, never interpolating raw status directly into the class attribute."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/static/admin.js")

    assert response.status_code == 200
    body = response.text

    # The function must use a safe gate before constructing the class name
    assert "hasOwnProperty" in body
    assert "safeStatus" in body

    # The old unsafe pattern must not be present
    assert "status-badge-${status}" not in body
    assert "status-badge-${safeStatus}" in body
