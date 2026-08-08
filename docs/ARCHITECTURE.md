# Architecture

## Overview

A single deployable Python backend (modular monolith) serving a browser frontend. The
application implements a Spanish voice agent for postoperative follow-up using synthetic
Colombian patient data.

The architecture is designed for the challenge's constraints: reproducible setup in
15 minutes or less, permitted language models only, local-first RAG with traceable
sources, conservative escalation, and two browser surfaces (administration console and
voice-call interface).

## Architecture Diagram

```text
┌──────────────────────────────────────────────────────────────────────┐
│                           BROWSER                                    │
│                                                                      │
│  ┌──────────────────────────┐     ┌──────────────────────────────┐  │
│  │  Call Interface          │     │  Administration Console       │  │
│  │  - Mic capture (WebRTC)  │     │  - Upload document            │  │
│  │  - Audio playback (TTS)  │     │  - List documents + status    │  │
│  │  - Spanish conversation  │     │  - Delete document + chunks   │  │
│  └────────────┬─────────────┘     └──────────────┬───────────────┘  │
└───────────────┼──────────────────────────────────┼───────────────────┘
                │                                  │
        WebSocket/HTTP                     HTTP REST
                │                                  │
┌───────────────┴──────────────────────────────────┴───────────────────┐
│                     APPLICATION BACKEND (Python)                     │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  api/                      REST + WebSocket endpoints         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│        │              │               │               │             │
│  ┌─────┴────┐  ┌──────┴──────┐  ┌─────┴─────┐  ┌──────┴──────┐     │
│  │ voice/   │  │conversation/│  │  rag/     │  │ documents/  │     │
│  │ STT, TTS │  │ state, flow │  │ ingest,   │  │ upload,     │     │
│  │ adapters │  │ orchestrate │  │ retrieve  │  │ list, delete│     │
│  └─────┬────┘  └──────┬──────┘  └─────┬─────┘  └──────┬──────┘     │
│        │              │               │               │             │
│  ┌─────┴──────────────┴───────────────┴───────────────┴──────┐     │
│  │  llm/                     Llama 3.1 70B (Groq) adapter     │     │
│  └────────────────────────────────────────────────────────────┘     │
│        │              │                                             │
│  ┌─────┴────┐  ┌──────┴──────┐                                      │
│  │decision/ │  │ summaries/  │                                      │
│  │escalate  │  │ structured  │                                      │
│  │classify  │  │ call record │                                      │
│  └─────┬────┘  └──────┬──────┘                                      │
│        │              │                                             │
│  ┌─────┴──────────────┴──────┐                                      │
│  │  metrics/                 │                                      │
│  │  latency, tokens, cost    │                                      │
│  └───────────────────────────┘                                      │
│        │                                                            │
│  ┌─────┴───────────────────────────────────────────────┐            │
│  │  persistence/                                       │            │
│  │  ┌─────────────────┐  ┌──────────────────────────┐  │            │
│  │  │ SQLite           │  │ ChromaDB                 │  │            │
│  │  │ - calls          │  │ - document chunks        │  │            │
│  │  │ - summaries      │  │ - embeddings (BGE-M3)    │  │            │
│  │  │ - documents meta │  │ - source metadata        │  │            │
│  │  │ - metrics        │  │                          │  │            │
│  │  └─────────────────┘  └──────────────────────────┘  │            │
│  └─────────────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────────────┘
```

## Module Boundaries

| Module | Responsibility | Key boundary |
|--------|---------------|-------------|
| `api/` | HTTP REST + WebSocket surface. Validates inputs, delegates to domain modules. | Only layer the browser touches. No business logic. |
| `voice/` | STT and TTS adapters behind a common interface. | Pure I/O adapter. Owns no state, patient data, or clinical knowledge. |
| `conversation/` | Call state machine: greeting → consent → structured questions → close. Composes prompts from patient profile + RAG chunks, calls `llm/` for reasoning, `decision/` for classification. | Owns turn state and prompt assembly. Never calls `documents/` or touches persistence/embeddings directly. |
| `llm/` | Adapter for Llama 3.1 70B Versatile via Groq Cloud (the only model currently integrated). Accepts prompt → returns structured response. | Knows nothing about voice, documents, RAG, or escalation. Pure text-in/text-out. |
| `rag/` | Ingestion (extract → chunk → embed BGE-M3 → store in ChromaDB) and retrieval (embed query → similarity search → return chunks + metadata). | Owns the embedding model, ChromaDB collection, chunking and retrieval. Does not own document lifecycle or know about patients/conversations. |
| `documents/` | Document lifecycle: upload, list, status, delete. Orchestrates metadata in SQLite and triggers RAG ingestion/deletion. | Calls `rag/` for ingestion and purge. Does not call `rag/` for retrieval. |
| `decision/` | Escalation classifier (Green / Yellow / Red). Runs after every LLM response using explicit symptom rules cross-checked against the LLM's classification. | Isolated from RAG, voice, and documents. Produces a verdict; does not modify conversation flow. Conservative: false negatives are catastrophic. |
| `summaries/` | At call end, produces a structured summary (patient, procedure, symptoms, decision, cited sources, next steps). SQLite persistence of the summary record is handled by the `persistence/` module. | Write-only, read-only on conversation history. |
| `metrics/` | Observes latency (P50/P95), token consumption, model invocations, RAG queries, estimated cost. Logs structured output; exposed via API. | Non-blocking observer. Never modifies application behavior. |
| `persistence/` | SQLite (calls, turns, summaries, document metadata, alerts) and ChromaDB (chunks, embeddings, source metadata). | Only owning modules write. `rag/` owns ChromaDB; `documents/`, `conversation/`, `summaries/`, `decision/` own their SQLite tables. |

## Data Flows

### Voice conversation (per turn)

```
Browser (mic) → voice/STT → conversation/ orchestrator:
  1. Load patient profile + turn history
  2. Call rag/retrieve for relevant clinical chunks
  3. Call llm/generate with prompt [system + patient + history + chunks + output schema]
  4. Validate structured output (JSON, citations, no hallucinations)
  5. Call decision/classify → Green/Yellow/Red
  → If Red: end call, create alert, produce summary
  → If Yellow (x2): escalate
  → If Green/Yellow (first): prepare next turn
→ voice/TTS → Browser (speaker)
```

### Document lifecycle

```
Upload:   Browser → api/ → documents/upload → persistence/SQLite (metadata)
                                              → rag/ingest → ChromaDB (chunks + embeddings)
Delete:   Browser → api/ → documents/delete → rag/delete_chunks (ChromaDB purge by document_id)
                                              → persistence/SQLite (soft-delete: status='deleted')
```

### RAG retrieval during conversation

```
conversation/ → rag/retrieve(query) → BGE-M3 embed → ChromaDB top-k → chunks + metadata
conversation/ assembles prompt with retrieved chunks and source citations
```

## Persistence Boundaries

### SQLite schema (conceptual)

```
calls                — call_id, paciente_id, nombre_completo, procedimiento,
                       dia_postop, eps, state, started_at, ended_at,
                       total_turns, escalated
conversation_turns   — turn_id, call_id, turn_index, role, text, timestamp,
                       severity, domain
summaries            — summary_id, call_id, created_at, patient_summary,
                       procedure_summary, symptoms_summary, decision_summary,
                       sources_json, next_steps
documents            — document_id, filename, status, uploaded_at, size_bytes,
                       error_message
escalation_alerts    — alert_id, call_id, created_at, severity, reason, domain
```

Document deletion uses **soft deletion**: deleting a document changes its ``status``
to ``deleted`` and purges ChromaDB chunks, but the SQLite row is retained for
auditability. The ``error_message`` column stores a human-readable description
when ``status`` is ``failed``.

### ChromaDB schema

```
collection: clinical_knowledge
  distance: cosine  (configured via hnsw:space metadata)
  id: uuid
  embedding: BGE-M3 float vector (1024 dimensions, L2-normalised)
  document: chunk text
  metadata: document_id, source_filename, chunk_index, page_number, ingested_at (UTC ISO-8601)
```

### Key rules

- Deleting a document must remove all ChromaDB chunks with matching `document_id` and soft-delete the SQLite row (``status = 'deleted'``). The metadata row is retained for auditability. No orphaned ChromaDB chunks.
- Call data and summaries are never deleted through the admin console.
- The vector store is rebuilt only on explicit re-index, never on restart.
- Synthetic patient data is loaded from `dataset/` XLSX at startup and is read-only.

## Permitted Models and Voice Adapters

The language model is **Llama 3.1 70B Versatile** (Groq Cloud) — the only model
currently integrated. It was chosen for its fast inference via Groq, native
structured JSON output support, strong Spanish-language performance, and
inclusion on the challenge's permitted model list
(``.challenge-docs/stack-tecnico.md``). Other permitted models (Llama 3.2,
Phi-3.5) are not integrated. The model is fixed in ``backend/llm/config.py``
and cannot be changed through environment variables or constructor arguments
— ``LlmConfig.__post_init__`` rejects any ``model_name`` other than exactly
``"llama-3.1-70b-versatile"``.

Voice adapters (STT and TTS) are free choice. Recommended STT: Groq Whisper Large V3,
browser Web Speech API, or local Whisper via Ollama. TTS provider: Kokoro-82M (resolved D3). Adapters are selected at startup via
configuration and wrapped behind a common interface so the conversation module never
depends on a specific provider.

## Safety and Validation Boundaries

### Structured output validation

Before output reaches the patient, application code validates the LLM's structured
response:

1. JSON parses and all required fields are present.
2. Cited source `document_id` values exist in the document registry.
3. The escalation signal is consistent with the symptom list and `decision/` classification.
4. The patient-facing message contains no medication dose, invented procedure, or
   clinical claim not traceable to a cited source.
5. The message is in Spanish.

If validation fails, the response is discarded and either retried or escalated to a
safe fallback.

### Escalation policy (safety-first)

- **Red always escalates.** No reassurance overrides it. Agent communicates next steps
  and ends the call.
- **Yellow escalates on accumulation.** Two consecutive yellow turns trigger escalation.
- **Unknown is yellow.** Unclassifiable or validation-failed turns default to yellow.
- **Ambiguity triggers inquiry.** One clarifying question before classifying.

### Prompt injection defense

- System instructions are in a separate message role from user input.
- Patient speech is never concatenated into instructions.
- The structured output schema constrains the LLM to a fixed JSON shape.
- Role-switching attempts in LLM output are rejected during validation.

### Clinical hallucination prevention

- The LLM is instructed to only cite sources from the provided RAG context.
- If no RAG chunks are retrieved (below similarity threshold), the agent states it
  lacks information rather than fabricating.
- The `decision/` module cross-checks the LLM's clinical reasoning against explicit
  red-flag rules independent of the LLM.

## Phased Implementation Plan

Implementation is ordered so each phase produces a testable artifact and the 15-minute
setup gate is verifiable from the earliest phase.

| Phase | Focus | Deliverable |
|-------|-------|-------------|
| 1 | Project skeleton and persistence | App starts, SQLite + ChromaDB init, schema tests pass |
| 2 | Document ingestion and deletion (RAG) | Upload → index → retrieve → delete → chunks gone; tests pass |
| 3 | LLM adapter and structured output | Text-in/text-out with validated JSON from permitted model |
| 4 | Voice adapters | STT + TTS round-trip in Spanish |
| 5 | Conversation orchestration and decision | Text-based conversation with RAG, escalation, summary |
| 6 | Browser call interface | Full voice conversation in Spanish from browser (gate G4) |
| 7 | Administration console | Upload, list, delete documents with live knowledge (gate G5) |
| 8 | Summaries, metrics, and polish | README metrics, edge cases, all gates passable |

## Open Decisions

These decisions affect multiple modules or have meaningful alternatives. Each has a
"resolve by" deadline tied to the phase that needs it.

| # | Decision | Alternatives | Depends on | Resolve by |
|---|----------|-------------|------------|------------|
| D5 | Audio transport format | Raw PCM16 / Opus-encoded / MediaRecorder chunks | D4 | Start of Phase 6 |
| D8 | Patient data loading | Load all 40 profiles at startup vs. lazy-load per call | D1, D4 | Start of Phase 5 |

## Resolved Decisions

| # | Decision | Chosen option | Rationale | Resolved |
|---|----------|--------------|-----------|----------|
| D1 | Language model | **Llama 3.1 70B Versatile (Groq Cloud)** | Fast inference, native structured JSON output, strong Spanish performance, and inclusion on the challenge's permitted model list. Hosted on Groq Cloud with ``groq`` Python SDK. Implemented in Phase 3 (``backend/llm/``). | Phase 3 |
| D2 | STT provider | **Groq Whisper Large V3** | Recommended in the challenge's ``stack-tecnico.md`` for ultra-low-latency Spanish transcription. Free tier via Groq Cloud. Implemented in Phase 4 (``backend/voice/``). | Phase 4 |
| D4 | Backend framework | **FastAPI** (async, WebSocket-native) | Required for WebSocket call interface (Phase 6), async-native, strong OpenAPI support for the administration console (Phase 7). Already implemented in Phase 1. | Phase 1 |
| D6 | Chunking strategy | **Fixed-size with overlap** (800 chars, 150 overlap) | Simple, predictable, well-tested. The 800-character default balances context completeness against retrieval precision for the challenge's clinical PDFs (typically 1–3 paragraphs per page). The 150-character overlap prevents splitting mid-sentence while keeping the duplication ratio below 19 %. Both values are tunable via env vars (``RAG_CHUNK_SIZE``, ``RAG_CHUNK_OVERLAP``). Implemented in Phase 2. | Phase 2 |
| D7 | LLM provider failover | **Single provider** (no failover chain) | The challenge scope allows a single permitted model; a failover chain adds complexity without a corresponding evaluation requirement. The API endpoint returns a safe fallback response when the LLM is unreachable. | Phase 3 |
| D3 | TTS provider | **Kokoro-82M** | Minimal model size (82 M parameters, ~0.6 GB RAM) that runs on CPU without GPU; native Spanish voice (``ef_dora``), natural prosody for clinical tone; outputs 24 kHz float32 audio that the adapter normalises to 16-bit PCM WAV for browser playback.  The ``KokoroAdapter`` implements the ``TTSProvider`` Protocol with lazy dependency loading — the optional ``kokoro`` package is only imported on the first ``synthesize()`` call.  Piper was considered but Kokoro's Spanish voice quality is significantly better for the challenge's clinical conversation UX.  Implemented in Phase 4 (``backend/voice/tts/kokoro.py``). | Phase 4 |
| D9 | PDF text extraction library | **pdfplumber** | Reliable character-level extraction, explicit page numbers, and consistent Unicode handling for Spanish clinical text. pdfplumber is actively maintained, has no external system dependencies, and produces structured page-by-page output — all critical for traceable source citations. pyMuPDF was considered but its AGPL license is incompatible with the challenge's proprietary license. | Phase 2 |

When an open decision is resolved, move it to this resolved section with the chosen
option and rationale. Add decisions here only when they affect multiple modules, are
difficult to reverse, or have meaningful alternatives — avoid routine implementation
steps.
