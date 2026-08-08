# Project Status

## Current Phase

All eight phases of the implementation plan (`docs/ARCHITECTURE.md` § Phased
Implementation Plan) are complete. The application implements:

**Fully integrated:**
- Persistence (SQLite + ChromaDB, typed schemas for calls, turns, summaries, alerts,
  documents).
- RAG pipeline (extract → chunk → embed BGE-M3 → store → retrieve with traceable
  citations).
- LLM adapter (Llama 3.1 70B Versatile via Groq Cloud; fixed model, validated
  structured output, Spanish prompts).
- Document lifecycle REST API (``POST/GET/DELETE /documents`` with soft-delete +
  ChromaDB chunk purge). A graphical administration console at ``/admin`` provides
  upload, listing with status polling, refresh, and deletion. The document lifecycle
  backend module is distinct from the administration console UI.
- Voice adapters (STT: Groq Whisper Large V3; TTS: Kokoro-82M ``ef_dora``).
- HTTP voice turn REST endpoints (``POST /calls`` creates a call and returns a
  base64-encoded WAV greeting; ``POST /calls/{call_id}/turn`` accepts base64 WAV,
  STT-transcribes, runs through the orchestrator, classifies escalation, synthesises
  a TTS response, and returns base64 WAV + transcription + citations + escalation
  info).
- Conversation orchestrator (finite state machine: IDLE → GREETING → CONSENT →
  QUESTIONS → CLOSING → ENDED; 6 structured Spanish follow-up questions;
  RAG+LLM integration; safe fallbacks).
- Escalation engine (GREEN/YELLOW/RED classifier with Spanish red-flag lexicons,
  numeric thresholds, negation handling; wired into voice turn endpoints via
  ``EscalationInfo`` in ``TurnResponse``).
- Structured summary generator (deterministic Spanish summaries: patient demographics,
  procedure, six symptom domains, decision, next steps).
- Metrics collector module (``InMemoryMetricsCollector`` with typed models, cost
  estimation, P50/P95 percentiles; stdlib-only, thread-safe). A read-only typed
  ``GET /metrics`` endpoint exposes collector data; a metrics frontend view
  renders the data. The collector and the reporting API are distinct concerns.
- Frontend call interface (vanilla HTML/CSS/JS at ``/`` and ``/call``: patient
  selection, MediaRecorder microphone capture, fetch-based calls to ``POST /calls``
  and ``POST /calls/{call_id}/turn``, state badge, conversation history, transcript
  area, citations and escalation display, and WAV audio playback for TTS responses).

**Pending (future work):**
- WebSocket/streaming transport for real-time voice conversations. The current
  implementation uses HTTP REST with base64-encoded WAV audio.

One open decision remains: D5 (audio transport format). The current implementation
uses HTTP REST with base64-encoded WAV; streaming/WebSocket alternatives remain as
future work.

The modular monolith architecture, module boundaries, data flows, persistence design,
adapter contracts, and phased implementation plan are documented in
`docs/ARCHITECTURE.md`.

## Completed

- Challenge documentation reviewed: stack-tecnico, requerimientos, rubrica-evaluacion,
  flujo-de-conocimiento, habeas-data, terminos-y-condiciones.
- Synthetic dataset inventoried: 4 XLSX files (40 patients, 160 trajectory days,
  3 991 conversation turns), 102 clinical PDFs across 5 procedures.
- OpenCode planner, coder, and auditor configuration prepared.
- Initial repository README created.
- Architecture defined: module catalog, data flows, persistence boundaries, permitted
  adapters, phased implementation plan, open decisions tracked.
- ``backend/data/`` package implemented: typed read-only data models, XLSX loaders,
  PDF path resolver. ``label_ground_truth`` isolated from runtime code. 58 dataset
  tests pass.
- FastAPI project skeleton: `pyproject.toml`, `backend/main.py` with CORS middleware,
  `GET /health`, entry-point script.
- Persistence layer: ChromaDB ``ChromaStore`` wrapper (collection init, chunk
  insertion, document-chunk deletion, singleton) and SQLite layer with typed
  dataclasses and CRUD for ``calls``, ``conversation_turns``, ``summaries``,
  ``escalation_alerts``, and ``documents`` tables. 41 persistence tests pass.
- RAG pipeline: ``backend/rag/`` — extract (pdfplumber), chunk (fixed-size with
  overlap, 800/150 chars), embed (BGE-M3 via sentence-transformers), store
  (ChromaDB cosine), ingestion, retrieval (``RetrievalResult`` with citations).
  14 fast + 10 slow tests pass. D6 (chunking strategy) resolved.
- LLM adapter: ``backend/llm/`` — fixed to Llama 3.1 70B Versatile (Groq Cloud),
  structured JSON output, Spanish prompts, citation mapping, multi-layer safety
  validation. 41 tests pass. D1 (language model) and D7 (single provider) resolved.
- RAG endpoint: ``backend/api/rag.py`` — ``POST /rag/query`` with fallback to
  ``insufficient_knowledge``. 13 tests pass.
- Conversation domain foundation: ``backend/conversation/`` — finite state machine
  (7 valid transitions), ``Message`` / ``History`` / ``PatientContext`` /
  ``CallContext``. 98 tests pass.
- Document lifecycle: ``backend/documents/`` — ``Document`` / ``DocumentStatus``,
  ``DocumentService`` (upload/list/delete). ``backend/api/documents.py`` —
  POST/GET/DELETE /documents. 9 fast + 6 slow tests pass.
- Conversation orchestrator: ``backend/conversation/orchestrator.py`` — text-only
  deterministic flow through IDLE → GREETING → CONSENT → QUESTIONS (6 structured
  Spanish follow-up questions: pain, fever, wound, appetite, sleep, mobility) →
  CLOSING → ENDED. Integrates RAG retrieval + LLM with safe fallbacks. 153 total
  conversation tests pass.
- Escalation decision engine: ``backend/decision/`` — ``classify()`` returns typed
  ``EscalationResult`` (GREEN/YELLOW/RED) with deterministic Spanish red-flag
  lexicons, numeric thresholds, negation handling, ambiguity detection. 125 tests
  pass. Stdlib-only, text-only.
- STT adapter: ``backend/voice/`` — ``SttProvider`` Protocol, ``GroqWhisperProvider``
  (model fixed to ``whisper-large-v3``, language ``"es"``), async via
  ``groq.AsyncGroq``, robust error mapping. 56 tests pass. D2 resolved.
- TTS adapter: ``backend/voice/tts/`` — ``TTSProvider`` Protocol, ``KokoroAdapter``
  with lazy ``kokoro`` loading, Spanish ``ef_dora`` voice, 16-bit PCM mono WAV
  output, valid silent WAV for empty text. 51 tests pass. D3 resolved.
- Frontend: ``frontend/`` — vanilla HTML/CSS/JS. ``index.html`` (patient selection),
  ``call.html`` / ``call.js`` (call interface with MediaRecorder microphone capture,
  state badge, conversation history, transcript area, citations and escalation display,
  WAV audio playback), ``admin.html`` / ``admin.js`` (administration console with
  upload, listing, status polling, refresh, deletion), metrics frontend view. 8 tests
  pass. ``backend/main.py`` serves assets via ``FileResponse`` and ``StaticFiles``.
- Metrics instrumentation: ``backend/metrics/`` — ``InMemoryMetricsCollector``
  (thread-safe, ``asyncio.Lock``), ``TurnMetrics`` / ``CallMetrics`` /
  ``MetricsSummary``, cost estimation, P50/P95 percentiles. 82 tests pass.
  Stdlib-only.
- Summaries module: ``backend/summaries/`` — deterministic Spanish summary generator
  (patient demographics, procedure, six symptom domains, escalation decision, next
  steps). 44 tests pass. Stdlib-only.
- Voice turn endpoints: ``backend/api/calls.py`` — ``POST /calls`` (create call,
  return agent greeting as base64 WAV), ``POST /calls/{call_id}/turn`` (STT
  transcribe → orchestrator → escalation classify → TTS synthesise → base64 WAV).
  ``backend/api/call_store.py`` — thread-safe in-memory ``CallStore``. 41 tests pass.
- Escalation classification wired in voice turn endpoints: domain inferred from
  question index during QUESTIONS phase; ``EscalationInfo`` returned in
  ``TurnResponse``.
- ``backend/main.py`` registers all routers (``calls_router``, ``rag_router``,
  ``documents_router``, ``metrics_router``) and serves frontend assets including
  ``/``, ``/call``, ``/admin``, and metrics views.
- ``.env`` auto-loading via ``python-dotenv`` at module level in ``backend/main.py``
  before any configuration imports.
- ``.gitignore`` updated with standard Python cache/build ignores, model cache, and
  ChromaDB runtime data.
- Decisions resolved: D1 (language model: Llama 3.1 70B Versatile, Groq), D2 (STT:
  Groq Whisper Large V3), D3 (TTS: Kokoro-82M, ef_dora), D4 (framework: FastAPI),
  D6 (chunking: fixed-size 800/150), D7 (LLM failover: single provider), D8
  (patient data loading: load all 40 at startup), D9 (PDF extraction: pdfplumber).
- Browser voice integration: ``frontend/call.js`` — MediaRecorder microphone
  capture, fetch-based calls to ``POST /calls`` and
  ``POST /calls/{call_id}/turn``, WAV audio playback, transcript rendering,
  conversation history with citation and escalation display.
- Administration console: ``/admin`` page with document upload, listing with
  status polling, refresh, and deletion; backed by the document lifecycle REST API
  but implemented as a distinct UI module.
- Metrics API and frontend: read-only typed ``GET /metrics`` endpoint and metrics
  frontend view; metrics collector module is distinct from the reporting API.

Test totals: 761 fast tests (pytest), 16 slow tests (`pytest -m slow`).

## In Progress

- Audio transport format — decision D5 is de facto HTTP REST with base64 WAV for the
  voice turn endpoints; WebSocket streaming remains an open option for future phases.

## Next Milestones

Implementation follows the eight-phase plan in `docs/ARCHITECTURE.md` § Phased
Implementation Plan (sole source of truth for milestones and deliverables).

All eight phases are complete. The immediate next steps are:

1. **WebSocket/streaming transport:** Evaluate adding real-time WebSocket/streaming
   for voice conversations (future work beyond current phases).
2. **Edge cases and polish:** Hardening, error handling, and remaining edge cases
   across all modules.

D8 (patient data loading) was resolved during Phase 5 (load all 40 profiles at startup).

## Open Architectural Decisions

These are tracked in `docs/ARCHITECTURE.md` § Open Decisions. One decision remains
open:

- **D5** — Audio transport format. De facto: HTTP REST with base64-encoded WAV for
  the voice turn endpoints (``POST /calls``, ``POST /calls/{call_id}/turn``).
  Streaming/WebSocket transport remains a future option for real-time browser
  integration.

Eight decisions have been resolved: D1 (LLM: Llama 3.1 70B, Groq), D2 (STT: Groq
Whisper Large V3), D3 (TTS: Kokoro-82M), D4 (framework: FastAPI), D6 (chunking:
800/150), D7 (LLM failover: single provider), D8 (patient data loading: load all
40 at startup), D9 (PDF extraction: pdfplumber).

## Known Constraints

- The language model must be one of the four permitted by `.challenge-docs/stack-tecnico.md`.
- The supplied clinical and patient data is synthetic and not clinically validated.
- No real patient data, secrets, recordings, or credentials may be committed.
- The final setup must be reproducible in fifteen minutes or less.
- The agent converses in Spanish with Colombian regionalisms.
- False negatives (failing to escalate when needed) are catastrophic; the architecture
  enforces conservative escalation.

## Update Rules

Update this file when a phase is completed, an open decision is resolved, a blocker
appears, or the next milestone changes. Keep completed items concise and
deduplicated — detailed changelog entries belong in commit history, not here.
