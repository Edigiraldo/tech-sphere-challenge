# Project Guide

## Purpose

Tech Sphere Challenge implementation: a Spanish voice agent for postoperative
follow-up using synthetic Colombian patient data.

The product must support:

- Browser-based voice conversations.
- Clinical retrieval-augmented generation (RAG).
- Live document upload, listing, processing, and deletion.
- Traceable sources for clinical responses.
- Conservative escalation decisions.
- Structured call summaries and observable metrics.

## Repository Map

```text
backend/       Application API and domain modules (planned)
frontend/      Browser call and administration interfaces (planned)
dataset/       Synthetic challenge data and reference PDFs
docs/          Maintained project documentation
.challenge-docs/ Challenge requirements and evaluation rules
```

## Where To Look

| Task | First files to inspect |
| --- | --- |
| Architecture | `docs/ARCHITECTURE.md` |
| Current work | `docs/STATUS.md` |
| RAG or documents | `backend/rag/`, `backend/documents/` |
| Conversation | `backend/conversation/` |
| Escalation | `backend/decision/` |
| API contract | `backend/api/` and relevant models |
| Browser UI | `frontend/` |
| Challenge constraints | `.challenge-docs/README.md` and `.challenge-docs/rubrica-evaluacion.md` |

## Working Rules

- Read this file and `docs/STATUS.md` before exploring the repository.
- Read `docs/ARCHITECTURE.md` only when the task affects architecture or interfaces.
- Inspect only the files listed by the planner and their direct dependencies.
- Keep implementation details in code, docstrings, and tests.
- Keep stable architectural facts in `docs/ARCHITECTURE.md`.
- Keep current progress and blockers in `docs/STATUS.md`.
- Remove stale documentation when the behavior it describes is removed.
