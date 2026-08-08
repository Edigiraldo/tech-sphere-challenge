"""Application entry point.

FastAPI application with CORS, health endpoint, API routers, frontend asset
serving, and provider wiring through the startup lifespan.

The application automatically loads environment variables from a ``.env`` file
(via ``python-dotenv``) before any configuration is read.  See the project
README for ``.env`` setup instructions.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env *before* any other import that may read os.environ / os.getenv.
# load_dotenv() is a no-op when the file does not exist, so missing .env is
# safe in production (env vars are expected to be set by the platform).
# ---------------------------------------------------------------------------
_dotenv_loaded: bool = load_dotenv()

import logging  # noqa: E402
import sys     # noqa: E402

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.calls import calls_router
from backend.api.documents import documents_router
from backend.api.metrics import metrics_router
from backend.api.rag import rag_router

logger = logging.getLogger(__name__)

if _dotenv_loaded:
    logger.info(".env file loaded — environment variables set from local file.")
else:
    logger.info(".env file not found — using existing environment variables.")


# ---------------------------------------------------------------------------
# Startup lifespan — wire voice providers before serving requests
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):  # type: ignore[arg-type]
    """Wire STT and TTS providers on application startup.

    Construction errors are caught and logged — a broken provider leaves its
    injection slot at ``None`` so that non-voice endpoints remain reachable.
    """
    from backend.voice.initialization import configure_providers

    logger.info("Application startup — wiring voice providers …")
    configure_providers()
    logger.info("Application startup complete.")
    yield
    # No teardown work needed for in-process providers.


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    app = FastAPI(
        title="Tech Sphere Challenge",
        description="Spanish voice agent for postoperative follow-up",
        version="0.1.0",
        lifespan=_lifespan,
    )

    # CORS: allow browser-based call interface and administration console
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------------
    # Health check (required for the 15-minute setup gate)
    # -----------------------------------------------------------------------
    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # -----------------------------------------------------------------------
    # API routers
    # -----------------------------------------------------------------------
    app.include_router(calls_router)
    app.include_router(metrics_router)
    app.include_router(rag_router)
    app.include_router(documents_router)

    # -----------------------------------------------------------------------
    # Frontend static assets (served from the sibling frontend/ directory)
    # -----------------------------------------------------------------------
    _frontend_dir = Path(__file__).resolve().parent.parent / "frontend"

    if _frontend_dir.is_dir():

        @app.get("/")
        async def _serve_index() -> FileResponse:
            return FileResponse(_frontend_dir / "index.html")

        @app.get("/call")
        async def _serve_call() -> FileResponse:
            return FileResponse(_frontend_dir / "call.html")

        @app.get("/admin")
        async def _serve_admin() -> FileResponse:
            return FileResponse(_frontend_dir / "admin.html")

        @app.get("/metrics")
        async def _serve_metrics() -> FileResponse:
            return FileResponse(_frontend_dir / "metrics.html")

        app.mount(
            "/static",
            StaticFiles(directory=str(_frontend_dir)),
            name="static",
        )

    return app


# Top-level app instance (used by uvicorn via CLI or programmatic run)
app = create_app()


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the application with uvicorn (entry point for `tech-sphere` script)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        stream=sys.stdout,
    )

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
