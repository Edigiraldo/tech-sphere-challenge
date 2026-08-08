# Project Guide

## Purpose

Tech Sphere Challenge implementation: a Spanish voice agent for postoperative
follow-up using synthetic Colombian patient data.

The product must support:

- Spanish voice conversations via HTTP REST endpoints (``POST /calls``,
  ``POST /calls/{call_id}/turn``) with base64-encoded WAV audio and browser-native
  microphone capture (MediaRecorder). WebSocket/streaming transport remains future
  work.
- Clinical retrieval-augmented generation (RAG) with traceable source citations.
- Live document upload, listing, processing-status tracking, and deletion (including
  purging indexed chunks) via a REST API (``POST/GET/DELETE /documents``) and a
  graphical administration console at ``/admin`` with upload, status polling, refresh,
  and deletion.
- Conservative escalation decisions with a safety-first classification policy.
- Structured call summaries with patient, procedure, symptoms, decision, sources, and
  next steps.
- Observable metrics: latency, token consumption, model invocations, RAG queries, and
  estimated cost per call. The metrics collector module feeds a read-only typed
  metrics API endpoint (``GET /metrics``) and a metrics frontend view.

The language model must be one of the four permitted by the challenge
(`.challenge-docs/stack-tecnico.md`). The rest of the stack — orchestration, voice,
RAG, embeddings — is open choice.

## Architecture

The target is a modular monolith: one Python backend with internal modules and a
browser frontend. The frontend provides a call interface with real voice
integration (MediaRecorder + WAV playback), an administration console at ``/admin``
for document lifecycle management, and a metrics view. The document lifecycle
backend (``POST/GET/DELETE /documents``) is a separate module from the
administration console UI.
See `docs/ARCHITECTURE.md` for the full module catalog, data flows, persistence
boundaries, permitted adapters, phased implementation plan, and open decisions.

## Repository Map

```text
backend/               Application backend (Python modular monolith)
  data/                Normalized read-only dataset access (patients, trajectories,
                         conversations, PDF resolver)
  api/                 REST endpoints (calls, documents, RAG, metrics); WebSocket not yet implemented
  voice/               STT and TTS adapters
  conversation/        Dialogue orchestration and state machine
  llm/                 Permitted language model adapter
  rag/                 Document ingestion, embedding, and retrieval
  documents/           Document lifecycle (upload, list, status, delete)
  decision/            Escalation classification
  summaries/           Structured call summary generation
  metrics/             Latency, token, and cost instrumentation
  persistence/         SQLite and ChromaDB access layer
frontend/              Browser UI (vanilla HTML/CSS/JS): call interface with MediaRecorder + API, admin console, metrics
dataset/               Synthetic challenge data and reference PDFs
docs/                  Maintained project documentation
.challenge-docs/       Challenge requirements and evaluation rules
```

## Where To Look

| Task | First files to inspect |
| --- | --- |
| Architecture | `docs/ARCHITECTURE.md` |
| Current work and open decisions | `docs/STATUS.md` |
| Document lifecycle | `backend/documents/`, `backend/rag/` |
| RAG retrieval | `backend/rag/` |
| Conversation flow | `backend/conversation/` |
| Escalation logic | `backend/decision/` |
| Voice I/O | `backend/voice/` |
| LLM adapter | `backend/llm/` |
| API contract | `backend/api/` |
| Persistence schema | `backend/persistence/` |
| Browser UI | `frontend/` |
| Challenge constraints | `.challenge-docs/README.md` and `.challenge-docs/rubrica-evaluacion.md` |
| Permitted models | `.challenge-docs/stack-tecnico.md` |

## Working Rules

- Read this file and `docs/STATUS.md` before exploring the repository.
- Read `docs/ARCHITECTURE.md` only when the task affects architecture or interfaces.
- Inspect only the files listed by the planner and their direct dependencies.
- Keep implementation details in code, docstrings, and tests.
- Keep stable architectural facts in `docs/ARCHITECTURE.md`.
- Keep current progress, milestones, and blockers in `docs/STATUS.md`.
- Track unresolved architectural decisions in `docs/ARCHITECTURE.md` § Open Decisions.
- Remove stale documentation when the behavior it describes is removed.
