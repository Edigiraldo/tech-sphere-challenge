"""Application entry point.

Minimal FastAPI application for the Phase 1 project skeleton. Subsequent phases
will register additional routers, middleware, and lifecycle hooks.
"""

import logging
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.rag import rag_router

logger = logging.getLogger(__name__)

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
