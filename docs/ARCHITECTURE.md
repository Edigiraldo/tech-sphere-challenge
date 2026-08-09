# Architecture

## Overview

A single deployable Python backend (modular monolith) serving a browser frontend. The
application implements a Spanish voice agent for postoperative follow-up using synthetic
Colombian patient data.

The architecture is designed for the challenge's constraints: reproducible setup in
15 minutes or less, permitted language models only, local-first RAG with traceable
sources, conservative escalation, a REST API for document lifecycle management,
a browser voice-call interface with real microphone capture, an administration
console at ``/admin``, and a read-only metrics API. WebSocket/streaming transport
remains future work.

## Architecture Diagram

```text
┌──────────────────────────────────────────────────────────────────────┐
│                           BROWSER                                    │
│                                                                      │
│  ┌──────────────────────────┐     ┌──────────────────────────────┐  │
│  │  Call Interface          │     │  Administration Console       │  │
│  │  - Patient selection     │     │  - Upload document            │  │
│  │  - MediaRecorder capture │     │  - List documents + status    │  │
│  │  - WAV playback          │     │  - Status polling + refresh   │  │
│  │  - Transcript + history  │     │  - Delete document + chunks   │  │
│  │  - Citations + escalation│     │                               │  │
│  └────────────┬─────────────┘     └──────────────┬───────────────┘  │
└───────────────┼──────────────────────────────────┼───────────────────┘
                │                                  │
           HTTP REST                         HTTP REST
                │                                  │
┌───────────────┴──────────────────────────────────┴───────────────────┐
│                     APPLICATION BACKEND (Python)                     │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  api/                      REST endpoints                     │   │
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
│  │  └─────────────────┘  └──────────────────────────┘  │            │
│  └─────────────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────────────┘
```

## Module Boundaries

| Module | Responsibility | Key boundary |
|--------|---------------|-------------|
| `api/` | HTTP REST surface. Validates inputs, delegates to domain modules. Includes calls, documents, RAG, and metrics routers. WebSocket endpoints are not yet implemented. | Only layer the browser touches. No business logic. |
| `voice/` | STT and TTS adapters behind a common interface. | Pure I/O adapter. Owns no state, patient data, or clinical knowledge. |
| `conversation/` | Call state machine: greeting → consent → structured questions → close. Composes prompts from patient profile + RAG chunks, calls `llm/` for reasoning, `decision/` for classification. | Owns turn state and prompt assembly. Never calls `documents/` or touches persistence/embeddings directly. |
| `llm/` | Adapter for Llama 3.1 70B Versatile via Groq Cloud (the only model currently integrated). Accepts prompt → returns structured response. | Knows nothing about voice, documents, RAG, or escalation. Pure text-in/text-out. |
| `rag/` | Ingestion (extract → chunk → embed BGE-M3 → store in ChromaDB) and retrieval (embed query → similarity search → return chunks + metadata). | Owns the embedding model, ChromaDB collection, chunking and retrieval. Does not own document lifecycle or know about patients/conversations. |
| `documents/` | Document lifecycle: upload, list, status, delete. Orchestrates metadata in SQLite and triggers RAG ingestion/deletion. | Calls `rag/` for ingestion and purge. Does not call `rag/` for retrieval. |
| `decision/` | Escalation classifier (Green / Yellow / Red). Runs after every LLM response using explicit symptom rules cross-checked against the LLM's classification. | Isolated from RAG, voice, and documents. Produces a verdict; does not modify conversation flow. Conservative: false negatives are catastrophic. |
| `summaries/` | At call end, produces a structured summary (patient, procedure, symptoms, decision, cited sources, next steps). SQLite persistence of the summary record is handled by the `persistence/` module. | Write-only, read-only on conversation history. |
| `metrics/` | Observes latency (P50/P95), token consumption, model invocations, RAG queries, estimated cost. The collector module feeds read-only typed ``GET /metrics/summary``, ``GET /metrics/calls``, and ``GET /metrics/calls/{call_id}`` endpoints plus a metrics frontend view. The collector and reporting API are distinct concerns. | Non-blocking observer. Never modifies application behavior. |
| `persistence/` | SQLite (calls, turns, summaries, document metadata, alerts) and ChromaDB (chunks, embeddings, source metadata). | Only owning modules write. `rag/` owns ChromaDB; `documents/`, `conversation/`, `summaries/`, `decision/` own their SQLite tables. |

## Data Flows

### Voice conversation (per turn)

The current transport is **HTTP REST** with base64-encoded WAV audio
(``POST /calls/{call_id}/turn``). The browser captures audio via MediaRecorder
and sends base64-encoded WAV chunks; responses are decoded and played back as
WAV audio. WebSocket/streaming transport remains future work.

The orchestrator implements a **safety-first flow**: each patient answer is
classified by the ``decision/`` module before any RAG/LLM call. GREEN and
first-YELLOW classifications receive deterministic Spanish acknowledgments
without RAG/LLM. RED answers short-circuit immediately to ENDED with an urgent
safety message and ``call_ended=True`` (no further turns possible).
Two consecutive YELLOW results
trigger escalation. Clinical questions during CLOSING are answered via
RAG+LLM with citations; non-questions end the call.

```
Browser (base64 WAV via HTTP POST) → voice/STT → conversation/ orchestrator:
  1. Load patient profile + turn history
  2. **Classify** patient answer against symptom domain (decision/classify)
  3a. If RED → short-circuit: no RAG/LLM, urgent safety message,
       transition directly to ENDED, ``call_ended=True``
  3b. If GREEN / first YELLOW → deterministic acknowledgment, ask next
      question (no RAG/LLM)
  3c. If second consecutive YELLOW → escalate to CLOSING
  4. (CLOSING only) If clinical question → call rag/retrieve + llm/generate
     with citations, stay in CLOSING
  5. If CLOSING non-question → end call
  → voice/TTS → Browser (base64 WAV in HTTP response)
```

### Document lifecycle

```
Upload:   Browser → api/ → documents/upload
           → SHA-256 content hash computed
           → If active record with same hash exists → return existing (idempotent)
           → persistence/SQLite (metadata + content_hash)
           → rag/ingest → ChromaDB (chunks + embeddings, keyed by document_id)
Delete:   Browser → api/ → documents/delete → rag/delete_chunks (ChromaDB purge by document_id)
           → persistence/SQLite (soft-delete: status='deleted', row preserved for audit)

Reconcile:  POST /documents/reconcile
           → Compare ChromaDB document_ids vs SQLite registry
           → Report orphaned ChromaDB IDs (missing from registry or deleted with lingering chunks)
           → Report missing ChromaDB entries (SQLite ready/processing but no indexed chunks)
           → ?clean=true deletes orphaned ChromaDB chunks
```

### RAG retrieval during conversation

```
conversation/ → rag/retrieve(query, valid_document_ids) → BGE-M3 embed → ChromaDB top-k
→ Filter out chunks whose document_id is deleted or unregistered
→ chunks + metadata with traceable citations
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
                        content_hash, error_message
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
- Duplicate uploads are detected via SHA-256 content hash: if an active (non-deleted) record exists with the same hash, the service returns the existing record without creating a new one. If the original was deleted, a new record is created.
- Retrieval automatically excludes chunks whose ``document_id`` is not in the SQLite registry or whose registry status is ``DELETED``, ensuring only active, registered documents contribute to search results.
- Reconciliation (``POST /documents/reconcile``) compares ChromaDB document IDs against the SQLite registry and can clean orphaned chunks on demand.
- Corpus ingestion is explicit (``scripts/ingest_corpus.py``) and never runs at startup. It is idempotent: re-running is safe as duplicates are detected by content hash.
- Call data and summaries are never deleted through the document lifecycle API.
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

Voice adapters (STT and TTS) are free choice. Selected STT: **Groq Whisper Large V3**
(resolved D2) — ultra-low-latency Spanish transcription via Groq Cloud, implemented
in ``backend/voice/groq.py`` behind the ``SttProvider`` Protocol. Selected TTS:
**Kokoro-82M** (resolved D3) — CPU-only Spanish voice (``ef_dora``), implemented in
``backend/voice/tts/kokoro.py`` behind the ``TTSProvider`` Protocol. Adapters are
selected at startup via configuration and wrapped behind a common interface so the
conversation module never depends on a specific provider.

## Safety and Validation Boundaries

### Retrieval sufficiency gates

Before the LLM is invoked, the RAG retrieval pipeline applies quantitative
quality gates to prevent weak or empty retrieval from reaching the model:

1. **Similarity threshold** (default 0.25, env ``RAG_SIMILARITY_THRESHOLD``):
   chunks below this cosine-similarity floor are discarded.
2. **Minimum chunk count** (default 2, env ``RAG_MIN_CHUNKS``): at least this
   many chunks must pass the similarity threshold.
3. **Minimum average similarity** (default 0.30, env ``RAG_MIN_AVG_SIMILARITY``):
   the mean similarity of all retrieved chunks must meet this bar.

When any gate fails, the ``RetrievalResult.sufficient`` flag is ``False``.
Callers (API endpoint, orchestrator) fall back to ``insufficient_knowledge``
without invoking the LLM — no weak context ever reaches the model.

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

### Post-hoc grounding validation

After the LLM produces a response, a secondary grounding validator
(``_validate_grounding``) checks that:

- All cited chunk IDs exist in the context and carry non-empty text.
- When the answer mentions a medication dose, at least one cited excerpt
  shares a significant token (>= 5 characters) with the answer.

Grounding warnings are logged server-side and exposed in ``validation_warnings``
only when ``debug=True``.  They never reach the patient-facing output.

### Escalation policy (safety-first)

- **Classification happens before RAG/LLM.** During the QUESTIONS phase, the
  ``decision/classify`` call gates all downstream processing.
- **Red always escalates immediately.** The orchestrator short-circuits: no RAG/LLM
  call, a clear Spanish urgent safety message is returned, the state
  transitions directly to ENDED with ``call_ended=True``, and the frontend
  disables further recording.  No further turns are possible.
- **Yellow escalates on accumulation.** Two consecutive YELLOW turns trigger
  escalation (transition to CLOSING) with ``should_escalate=True``.
  First YELLOW receives a deterministic acknowledgment without RAG/LLM.
- **Green receives deterministic acknowledgment.** GREEN answers get a
  domain-specific positive message and the next structured question, without
  RAG/LLM.
- **Unknown is yellow.** Unclassifiable or validation-failed turns default to
  yellow.
- **Ambiguity triggers inquiry.** One clarifying question before classifying.

### Prompt injection defense

- System instructions are in a separate message role from user input (Groq
  API role separation).
- Patient speech is never concatenated into instructions.
- The structured output schema constrains the LLM to a fixed JSON shape.
- Role-switching attempts in LLM output are rejected during validation.
- **Input-level injection detection** (``_detect_injection``): the query is
  scanned for known jailbreak patterns (role-switching, system prompt
  extraction, delimiter injection, ``[INST]`` tags, etc.) before any LLM
  call.  When a pattern matches, the call returns a safe Spanish fallback
  (``insufficient_knowledge=True``) without invoking the model.
- **Length check**: queries longer than 2000 characters are rejected at
  the injection-detection layer.
- **Output-level checks**: the grounding validator catches hallucinated
  citations and ungrounded medication claims in the LLM output.

### Clinical hallucination prevention

- The LLM is instructed to only cite sources from the provided RAG context.
- If no RAG chunks meet the sufficiency gates, the agent states it lacks
  information rather than fabricating.
- The `decision/` module cross-checks the LLM's clinical reasoning against explicit
  red-flag rules independent of the LLM.
- Post-hoc grounding validation verifies that medication-dose claims in the
  answer are supported by the cited excerpts.

## Phased Implementation Plan

Implementation is ordered so each phase produces a testable artifact and the 15-minute
setup gate is verifiable from the earliest phase.

| Phase | Focus | Deliverable | Status |
|-------|-------|-------------|--------|
| 1 | Project skeleton and persistence | App starts, SQLite + ChromaDB init, schema tests pass | ✅ Complete |
| 2 | Document ingestion and deletion (RAG) | Upload → index → retrieve → delete → chunks gone; tests pass | ✅ Complete |
| 3 | LLM adapter and structured output | Text-in/text-out with validated JSON from permitted model | ✅ Complete |
| 4 | Voice adapters | STT + TTS round-trip in Spanish | ✅ Complete |
| 5 | Conversation orchestration and decision | Text-based conversation with RAG, escalation, summaries, metrics collector | ✅ Complete |
| 6 | Browser call interface | Real browser voice capture and audio playback in Spanish (gate G4) | ✅ Complete — MediaRecorder capture, fetch-based POST to /calls and /calls/{call_id}/turn, WAV playback, transcript/history/citations/escalation display; WebSocket streaming not yet implemented |
| 7 | Administration console | Graphical console for upload, list, delete documents with live knowledge (gate G5) | ✅ Complete — ``/admin`` page with upload, listing, status polling, refresh, and deletion; document lifecycle REST API remains distinct from admin console |
| 8 | Metrics API and polish | Metrics endpoint, README metrics, edge cases, all gates passable | ✅ Complete — read-only typed metrics endpoints (summary, calls, and per-call detail) and metrics frontend view; metrics collector module is distinct from the reporting API; summaries generator exists |

## Open Decisions

These decisions affect multiple modules or have meaningful alternatives. Each has a
"resolve by" deadline tied to the phase that needs it.

| # | Decision | Alternatives | Depends on | Resolve by |
|---|----------|-------------|------------|------------|
| D5 | Audio transport format | Raw PCM16 / Opus-encoded / MediaRecorder chunks / WebSocket streaming | D4 | Phase 6 — de facto: HTTP REST with base64-encoded WAV for voice turn endpoints; streaming/WebSocket transport remains a future option |

## Resolved Decisions

| # | Decision | Chosen option | Rationale | Resolved |
|---|----------|--------------|-----------|----------|
| D1 | Language model | **Llama 3.1 70B Versatile (Groq Cloud)** | Fast inference, native structured JSON output, strong Spanish performance, and inclusion on the challenge's permitted model list. Hosted on Groq Cloud with ``groq`` Python SDK. Implemented in Phase 3 (``backend/llm/``). | Phase 3 |
| D2 | STT provider | **Groq Whisper Large V3** | Recommended in the challenge's ``stack-tecnico.md`` for ultra-low-latency Spanish transcription. Free tier via Groq Cloud. Implemented in Phase 4 (``backend/voice/``). | Phase 4 |
| D4 | Backend framework | **FastAPI** (async, WebSocket-native) | Required for WebSocket call interface (Phase 6), async-native, strong OpenAPI support for the administration console (Phase 7). Already implemented in Phase 1. | Phase 1 |
| D6 | Chunking strategy | **Fixed-size with overlap** (800 chars, 150 overlap) | Simple, predictable, well-tested. The 800-character default balances context completeness against retrieval precision for the challenge's clinical PDFs (typically 1–3 paragraphs per page). The 150-character overlap prevents splitting mid-sentence while keeping the duplication ratio below 19 %. Both values are tunable via env vars (``RAG_CHUNK_SIZE``, ``RAG_CHUNK_OVERLAP``). Implemented in Phase 2. | Phase 2 |
| D7 | LLM provider failover | **Single provider** (no failover chain) | The challenge scope allows a single permitted model; a failover chain adds complexity without a corresponding evaluation requirement. The API endpoint returns a safe fallback response when the LLM is unreachable. | Phase 3 |
| D8 | Patient data loading | **Load all 40 profiles at startup** | The dataset is small (40 patients, ~160 trajectory days, ~4 000 conversation turns), fitting comfortably in memory. Startup loading from ``dataset/`` XLSX via ``backend/data/loader.py`` is simpler than lazy-loading, avoids race conditions during call creation, and ensures immediate availability of patient demographics, trajectories, and reference conversations for the conversation orchestrator. Implemented during Phases 1–5 (``backend/data/`` read-only dataclass access layer). | Phase 5 |
| D3 | TTS provider | **Kokoro-82M** | Minimal model size (82 M parameters, ~0.6 GB RAM) that runs on CPU without GPU; native Spanish voice (``ef_dora``), natural prosody for clinical tone; outputs 24 kHz float32 audio that the adapter normalises to 16-bit PCM WAV for browser playback.  The ``KokoroAdapter`` implements the ``TTSProvider`` Protocol with lazy dependency loading — the optional ``kokoro`` package is only imported on the first ``synthesize()`` call.  Piper was considered but Kokoro's Spanish voice quality is significantly better for the challenge's clinical conversation UX.  Implemented in Phase 4 (``backend/voice/tts/kokoro.py``). | Phase 4 |
| D9 | PDF text extraction library | **pdfplumber** | Reliable character-level extraction, explicit page numbers, and consistent Unicode handling for Spanish clinical text. pdfplumber is actively maintained, has no external system dependencies, and produces structured page-by-page output — all critical for traceable source citations. pyMuPDF was considered but its AGPL license is incompatible with the challenge's proprietary license. | Phase 2 |

When an open decision is resolved, move it to this resolved section with the chosen
option and rationale. Add decisions here only when they affect multiple modules, are
difficult to reverse, or have meaningful alternatives — avoid routine implementation
steps.
