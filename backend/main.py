"""Application entry point.

Minimal FastAPI application for the Phase 1 project skeleton. Subsequent phases
will register additional routers, middleware, and lifecycle hooks.

The application automatically loads environment variables from a ``.env`` file
(via ``python-dotenv``) before any configuration is read.  See the project
README for ``.env`` setup instructions.
"""

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

from backend.api.rag import rag_router
from backend.api.documents import documents_router

logger = logging.getLogger(__name__)

if _dotenv_loaded:
    logger.info(".env file loaded — environment variables set from local file.")
else:
    logger.info(".env file not found — using existing environment variables.")

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    app = FastAPI(
        title="Tech Sphere Challenge",
        description="Spanish voice agent for postoperative follow-up",
        version="0.1.0",
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
    app.include_router(rag_router)
    app.include_router(documents_router)

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
