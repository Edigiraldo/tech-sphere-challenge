"""LLM adapter module — permitted model interface.

Provides a unified adapter for Groq Llama 3.3 70B Versatile. The adapter accepts
a prompt with RAG context and returns a validated, structured ``RagAnswer``
with Spanish clinical text and traceable citations.

Also provides ``llm_second_approval`` for conservative safety review of
deterministic escalation classifications during follow-up questions, and
centralized prompt-injection detection via ``backend.llm.injection``.
"""

from backend.llm.adapter import RagAnswer, RagCitation, generate_rag_answer
from backend.llm.approval import llm_second_approval, LlmApprovalResult
from backend.llm.config import LlmConfig

__all__ = [
    "generate_rag_answer",
    "llm_second_approval",
    "LlmApprovalResult",
    "LlmConfig",
    "RagAnswer",
    "RagCitation",
]
