"""LLM adapter module — permitted model interface.

Provides a unified adapter for the permitted language model (default: Gemini
1.5 Flash).  The adapter accepts a prompt with RAG context and returns a
validated, structured ``RagAnswer`` with Spanish clinical text and traceable
citations.
"""

from backend.llm.adapter import RagAnswer, RagCitation, generate_rag_answer
from backend.llm.config import LlmConfig

__all__ = [
    "LlmConfig",
    "generate_rag_answer",
    "RagAnswer",
    "RagCitation",
]
