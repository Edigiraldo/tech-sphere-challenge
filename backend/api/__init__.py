"""REST API endpoints for the Tech Sphere Challenge backend.

Each sub-module registers one or more FastAPI routers that are mounted
in ``backend.main.create_app()``.
"""

from backend.api.rag import rag_router
from backend.api.documents import documents_router

__all__ = ["rag_router", "documents_router"]
