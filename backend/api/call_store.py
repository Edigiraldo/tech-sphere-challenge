"""Thread-safe in-memory store for active voice calls.

Maps ``call_id`` → ``ConversationOrchestrator`` instances.  A single
module-level ``call_store`` singleton is used by the API endpoints.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from backend.conversation.orchestrator import ConversationOrchestrator


class CallStore:
    """Thread-safe in-memory store for active voice calls.

    Wraps a plain ``dict`` behind an ``asyncio.Lock`` so the store is safe
    for use with ASGI servers that run the event loop across multiple
    concurrent requests.
    """

    def __init__(self) -> None:
        self._calls: dict[str, ConversationOrchestrator] = {}
        self._lock = asyncio.Lock()

    async def put(
        self, call_id: str, orchestrator: ConversationOrchestrator
    ) -> None:
        """Register an orchestrator for *call_id*."""
        async with self._lock:
            self._calls[call_id] = orchestrator

    async def get(
        self, call_id: str
    ) -> Optional[ConversationOrchestrator]:
        """Return the orchestrator for *call_id*, or ``None``."""
        async with self._lock:
            return self._calls.get(call_id)

    async def remove(self, call_id: str) -> None:
        """Remove the orchestrator for *call_id* (no-op if absent)."""
        async with self._lock:
            self._calls.pop(call_id, None)

    async def exists(self, call_id: str) -> bool:
        """Return ``True`` if *call_id* is registered."""
        async with self._lock:
            return call_id in self._calls


# Module-level singleton shared by all API endpoints.
call_store = CallStore()
