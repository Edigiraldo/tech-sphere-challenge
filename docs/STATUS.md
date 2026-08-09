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
   a TTS response, and returns base64 WAV + transcription + patient transcription +
   citations + escalation info). The orchestrator is wired with live ``RagConfig``
   and ``LlmConfig`` (from environment variables); built-in safe fallbacks handle
   cases where RAG or LLM providers are unavailable.
- Conversation orchestrator (finite state machine: IDLE → GREETING → CONSENT →
  QUESTIONS → CLOSING → ENDED; 6 structured Spanish follow-up questions;
   **safety-first flow**: classifies patient answers before any RAG/LLM call —
   RED short-circuits directly to ENDED with an urgent safety message,
   ``call_ended=True``, and no further processing; GREEN and first YELLOW
   use deterministic acknowledgments without RAG/LLM; two consecutive YELLOW
   results trigger escalation with ``should_escalate=True``; clinical questions
  during CLOSING are answered with RAG+LLM (with citations) while non-questions
  end the call; safe fallbacks).
- Escalation engine (GREEN/YELLOW/RED classifier with Spanish red-flag lexicons,
  numeric thresholds, negation handling; wired into voice turn endpoints via
  ``EscalationInfo`` in ``TurnResponse``).
- Structured summary generator (deterministic Spanish summaries: patient demographics,
  procedure, six symptom domains, decision, next steps).
- Metrics collector module (``InMemoryMetricsCollector`` with typed models, cost
  estimation, P50/P95 percentiles; stdlib-only, thread-safe). A read-only typed
  ``GET /metrics/summary``, ``GET /metrics/calls``, and
  ``GET /metrics/calls/{call_id}`` expose collector data; a metrics frontend view
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
  structured JSON output, Spanish prompts, multi-layer safety validation
  including input-level prompt-injection detection (pattern-based jailbreak
  scanning, length check, Spanish safe fallback) and post-hoc grounding
   validation (citation-integrity and medication-dose grounding checks).
   63 tests pass. D1 (language model) and D7 (single provider) resolved.
- RAG endpoint: ``backend/api/rag.py`` — ``POST /rag/query`` with retrieval
   sufficiency gates (minimum chunk count, average similarity threshold) and
   fallback to ``insufficient_knowledge``. 14 tests pass.
- Conversation domain foundation: ``backend/conversation/`` — finite state machine
  (7 valid transitions), ``Message`` / ``History`` / ``PatientContext`` /
  ``CallContext``. 98 tests pass.
- Document lifecycle: ``backend/documents/`` — ``Document`` / ``DocumentStatus``,
  ``DocumentService`` (upload/list/delete). ``backend/api/documents.py`` —
  POST/GET/DELETE /documents, ``POST /documents/reconcile``.
  32 fast + 10 slow tests pass, including duplicate-document isolation
  (deleting one document does not affect a different document's chunks).
  Content-hash (SHA-256) duplicate detection makes upload idempotent:
  identical content returns the existing active record. Reconciliation
  detects and can clean orphaned ChromaDB chunks. Registry-filtered
  retrieval excludes deleted/unregistered document IDs from search results.
- Conversation orchestrator: ``backend/conversation/orchestrator.py`` — text-only
  deterministic flow through IDLE → GREETING → CONSENT → QUESTIONS (6 structured
  Spanish follow-up questions: pain, fever, wound, appetite, sleep, mobility) →
  CLOSING → ENDED. Integrates RAG retrieval + LLM with retrieval sufficiency
  gates and safe fallbacks. 193 tests pass.
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
- Frontend-backend contract integration tests: ``tests/test_frontend_integration.py`` —
  26 fast tests covering the HTTP contract consumed by ``call.js`` and ``app.js``:
  ``POST /calls`` and ``POST /calls/{call_id}/turn`` response shapes, base64 audio
  round-trip, full call flow from GREETING to ENDED, escalation info shape and timing,
  citation structure, error-handling contract, patient_transcription rendering contract
  (content and type semantics), agent transcription preservation across every turn,
  and call-state progression with monotonic ordering.
- Metrics instrumentation: ``backend/metrics/`` — ``InMemoryMetricsCollector``
  (thread-safe, ``asyncio.Lock``), ``TurnMetrics`` / ``CallMetrics`` /
  ``MetricsSummary``, cost estimation, P50/P95 percentiles. 82 tests pass.
  Stdlib-only.
- Summaries module: ``backend/summaries/`` — deterministic Spanish summary generator
  (patient demographics, procedure, six symptom domains, escalation decision, next
  steps). 44 tests pass. Stdlib-only.
 - Voice turn endpoints with persistence: ``backend/api/calls.py`` — ``POST /calls``
   creates a call and returns the agent greeting as base64 WAV;
   ``POST /calls/{call_id}/turn`` transcribes patient audio (STT), runs the
   orchestrator (with live ``RagConfig`` and ``LlmConfig``), classifies escalation,
   synthesises a TTS response, and returns base64 WAV + transcription + patient
   transcription + citations + escalation info. ``TurnResponse`` includes
   ``patient_transcription`` (the STT output for the patient's speech) for frontend
   display. Real patient profiles are loaded from the dataset when available, with a
   request-body fallback for patients not found in the dataset.
   ``backend/api/call_store.py`` — thread-safe in-memory ``CallStore`` for runtime
   orchestrator instances. Voice persistence is fully integrated with the SQLite
   layer: call creation inserts a ``CallRecord``; each turn persists
   ``ConversationTurnRecord`` entries; YELLOW/RED escalation classifications persist
   ``EscalationAlertRecord``; call completion generates a structured summary via
   ``backend/summaries/generator.py`` and persists a ``SummaryRecord``. Incomplete
   calls are tracked with ``ended_at=None``. SQLite is initialised at application
   startup in ``backend/main.py``. Calls, turns, summaries, and alerts are
   restart-safe: they survive process restarts because the data is in SQLite, not
    only in memory. 63 tests pass (9 persistence-focused).
- Escalation classification wired in voice turn endpoints: prefer orchestrator's
  classification when available (``turn.escalation``), falling back to the
  endpoint-level classifier with API-boundary consecutive-YELLOW accumulation
  (``_classify_response`` tracks per-call consecutive YELLOWs via
  ``_call_consecutive_yellows`` and sets ``should_escalate=True`` on the
  second consecutive YELLOW).  The orchestrator's ``_consecutive_yellows``
  counter controls state transitions; the API-boundary counter is authoritative
  for the HTTP response escalation verdict in the fallback path.  Counter
  resets on GREEN, RED, consent refusal, and call completion.  ``EscalationInfo``
  returned in ``TurnResponse``.  Regression-tested at the API boundary for two
  consecutive YELLOW escalations producing ``should_escalate=True``, including
  persisted ``EscalationAlertRecord`` with ``severity=YELLOW`` and call-level
  ``escalated=True`` flag.
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
  patient transcription display (from ``patient_transcription`` field),
  conversation history with citation and escalation display.
- Administration console: ``/admin`` page with document upload, listing with
  status polling, refresh, and deletion; backed by the document lifecycle REST API
  but implemented as a distinct UI module.
- Metrics API and frontend: read-only typed ``GET /metrics/summary``,
  ``GET /metrics/calls``, and ``GET /metrics/calls/{call_id}`` endpoints and metrics
  frontend view; metrics collector module is distinct from the reporting API.
- RAG/LLM safety hardening: retrieval sufficiency gates (minimum chunk count,
  average similarity threshold, configurable via env vars), input-level prompt
  injection detection (pattern-based jailbreak scanning with Spanish safe
  fallback), post-hoc grounding validation (citation-integrity checks), and
  medication-dose grounding enforcement (ungrounded medication/dose claims
  force ``insufficient_knowledge=True`` with safe fallback, preserving valid
  citations), deletion isolation (deleting one document preserves
  other documents' chunks), registry-filtered retrieval (deleted and
  unregistered document IDs excluded automatically), and real Appendicitis
  PDF filename integration tests.  63 LLM tests, 14 RAG API tests, and all
  existing conversation and document tests pass with the new safety layers.

- Dependency audit: ``openpyxl>=3.0.0``, ``numpy>=1.24.0``, and ``pydantic>=2.0.0``
  declared as explicit base dependencies in ``pyproject.toml``; ``numpy`` removed from
  ``voice`` extra; ``kokoro>=0.7.0`` copied to ``dev`` extra.

Test totals: 947 fast tests (pytest), 27 slow tests (`pytest -m slow`), 974 tests total.

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
