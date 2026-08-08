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

<<<<<<< HEAD
- Audio transport format — decision D5 pending (Phase 6).

## Completed (frontend)

- **2026-08-08:** Frontend voice integration implemented:
  - ``frontend/app.js`` — patient selection page now creates calls via
    ``POST /calls``, stores the response (``call_id``, ``greeting_audio_b64``,
    ``total_questions``) in ``sessionStorage``, and navigates to ``/call``.
  - ``frontend/call.js`` — full real-time voice call interface with
    ``MediaRecorder`` microphone capture (audio/webm), record/stop toggle
    controls, base64 audio upload to ``POST /calls/{call_id}/turn``,
    base64 WAV playback via ``<audio>``, live transcript and conversation
    history, escalation severity banner (GREEN/YELLOW/RED), source citation
    badges, loading overlay with spinner, and call-completed modal.
    Error states (404, 400, 422, 5xx) are surfaced to the user.
  - ``frontend/call.html`` — added escalation banner, loading overlay,
    and completed-banner DOM elements.
  - ``frontend/styles.css`` — new styles for escalation banner (animated
    severity colours), citation badges, loading overlay with spinner
    animation, completed banner with scale-in, and call-status display.
   - ``tests/test_frontend_integration.py`` — 15 backend contract tests
    verifying the exact HTTP API surface consumed by the vanilla frontend:
    ``CreateCallResponse`` shape, ``TurnResponse`` shape, base64 audio
    round-trip, MediaRecorder-compatible formats, full call flow
    (GREETING→ENDED), escalation info shape (GREEN/YELLOW/RED fields),
    citation structure, error handling (404/400/422), and sessionStorage
    field completeness.  All tests use mocked STT/TTS.
  - All 8 existing ``test_frontend.py`` static-serving tests continue to
    pass.

## Completed (setup)

- **2026-08-07:** Automatic ``.env`` file loading via ``python-dotenv``.
  ``backend/main.py`` calls ``load_dotenv()`` at module level before any
  configuration-importing code runs.  ``python-dotenv>=1.0.0`` added to
  ``pyproject.toml`` dependencies.  ``.env`` is already excluded by
  ``.gitignore``.  README updated with ``.env`` creation instructions.

## Recent Changes

- **2026-08-08:** Frontend shell implemented.  ``frontend/`` directory with vanilla
  HTML/CSS/JS (index.html, call.html, styles.css, app.js, call.js).  ``backend/main.py``
  serves static assets via ``FileResponse`` routes and ``StaticFiles`` mount.
  8 focused tests pass.  No backend API connections, no microphone/audio APIs.
- **2026-08-08:** TTS foundation implemented.  ``backend/voice/tts/`` with typed
  ``TTSProvider`` Protocol, ``TTSConfig`` (frozen dataclass, Spanish defaults),
  ``TTSResult`` (normalised WAV bytes), ``TTSSynthesisError``, and ``KokoroAdapter``
  with lazy ``kokoro`` dependency loading.  Empty/whitespace text produces valid
  silent WAV (not an invalid payload).  WAV serialisation produces 16-bit PCM mono
  RIFF containers suitable for browser playback.  Decision D3 resolved: **Kokoro-82M**
  selected for its minimal footprint (~0.6 GB RAM), CPU-only inference, and natural
  Spanish voice quality (``ef_dora``).  ``kokoro>=0.7.0`` added to optional
  ``voice`` extras in ``pyproject.toml``.  51 fast tests pass.
- **2026-08-08:** Metrics instrumentation implemented (``backend/metrics/``).
  Typed frozen dataclasses (``TurnMetrics``, ``CallMetrics``, ``MetricsSummary``)
  with full field-level validation.  ``MetricsCollector`` Protocol defining the
  public contract (``start_call``, ``record_turn``, ``end_call``,
  ``get_call_metrics``, ``get_summary``).  Thread-safe ``InMemoryMetricsCollector``
  implementation with ``threading.Lock``, defensive lifecycle management, and
  immutable-snapshot queries.  ``CallMetrics.from_turns()`` aggregates token
  counts with "all-None → None" semantics (absent values are zero when any turn
  has data).  ``CostConfig`` frozen dataclass with non-negative per-million-token
  rates and ``estimate_cost()`` function.  Percentile computation via linear
  interpolation (returns ``None`` for empty data, validates ``p`` ∈ [0, 100],
  P50/P95 tested).  ``MetricsSummary`` includes per-turn latency and optional
  component-duration (TTS/STT/LLM) P50/P95 percentiles.  82 focused tests pass
  covering recording, aggregation, missing optional values, thread safety,
  defensive edge cases, cost estimation, and percentile correctness.  No new
  dependencies.  Stdlib-only.
- **2026-08-08:** STT adapter made truly async.  ``GroqWhisperProvider._call_groq``
  now uses ``groq.AsyncGroq`` (instead of synchronous ``groq.Groq``) and awaits the
  transcription API call.  All 56 mocked unit tests updated; a focused test
  (``test_async_groq_client_instantiated_and_awaited``) proves ``AsyncGroq`` is
  instantiated and awaited.  Error mapping and behaviour preserved.
- **2026-08-08:** Provider startup wiring implemented.
  ``backend/voice/initialization.py`` ``configure_providers()`` constructs
  ``GroqWhisperConfig``/``GroqWhisperProvider`` and
  ``TTSConfig``/``KokoroAdapter`` and wires them into the API layer via
  ``set_stt()``/``set_tts()``.  Each provider's construction is wrapped in its
  own try/except so a failure in one does not prevent the other from wiring,
  and neither failure crashes application startup.  ``backend/main.py`` calls
  ``configure_providers()`` through the FastAPI lifespan before serving
  requests.  13 focused tests pass covering successful wiring, missing
  ``GROQ_API_KEY``, construction errors, injection overrides, and all
  four provider-readiness log combinations.
- **2026-08-08:** Metrics reporting endpoints and frontend dashboard implemented.
  ``backend/api/metrics.py`` provides read-only ``GET /metrics/summary``,
  ``GET /metrics/calls``, and ``GET /metrics/calls/{call_id}`` with typed
  Pydantic response models.  Module-level ``InMemoryMetricsCollector`` singleton
  shared with calls API for instrumentation.  ``InMemoryMetricsCollector``
  extended with ``get_all_call_metrics()`` (sorted by ``call_id``) and
  ``get_call_turns()`` (raw per-turn observations for ended calls), plus
  ``reset()`` for test isolation.  ``backend/api/calls.py`` instruments
  ``create_call`` and ``process_turn`` with ``start_call``, ``record_turn``
  (latency-timed), and ``end_call`` calls without changing provider behaviour.
  ``/metrics`` frontend page (``metrics.html`` + ``metrics.js``) displays a
  summary dashboard, calls table, and per-call turn detail using the existing
  visual language.  New focused tests cover the metrics API (17), collector behaviour (12), and frontend dashboard (2).
  775 total fast tests pass.  Metrics are **in-memory only** — data is lost
  on server restart.
  ``backend/api/calls.py`` — ``POST /calls`` (create call, return agent greeting
  as base64-encoded WAV), ``POST /calls/{call_id}/turn`` (transcribe patient audio
  via STT, delegate to ``ConversationOrchestrator``, classify escalation via
  ``backend/decision/classify``, synthesise agent response via TTS, return
  base64-encoded WAV).  ``backend/api/call_store.py`` — thread-safe in-memory
  ``CallStore`` (``asyncio.Lock``-protected ``dict``).  Orchestrator runs with
  ``rag_config=None``, ``llm_config=None`` (deterministic fallback messages).
  Escalation classification is wired during the QUESTIONS phase: domain is
  inferred from question index.  STT/TTS are injectable via module-level
  ``set_stt()`` / ``set_tts()``.  41 focused tests pass (all STT/TTS calls
  mocked).  ``backend/main.py`` registers the ``calls_router``.
  All 278 existing conversation + decision tests continue to pass.
- **2026-08-08:** STT adapter foundation (Phase 4 partial).  ``backend/voice/`` with
  typed ``SttProvider`` Protocol, ``TranscriptionResult``, ``SttError`` exception
  hierarchy, frozen ``GroqWhisperConfig``, ``GroqWhisperProvider`` adapter, and
   ``transcribe_audio()`` injection API.  56 mocked unit tests pass.  Resolved D2
  (STT provider = Groq Whisper Large V3).  ``groq>=0.9.0`` added to dependencies.
- **2026-08-08:** Escalation decision engine (``backend/decision/``) implemented.
  ``classify()`` returns typed ``EscalationResult`` with ``Severity`` (GREEN/YELLOW/RED),
  deterministic Spanish keyword + numeric classification across 6 symptom domains,
  negation handling, ambiguity detection, and cross-cutting critical flags.
  125 tests pass.  Stdlib-only, text-only, no dependencies on RAG/LLM/voice/persistence.
- **2026-08-07 (pm):** Conversation orchestrator implemented.  ``ConversationOrchestrator``
  drives the state machine through all six phases (IDLE → GREETING → CONSENT →
  QUESTIONS → CLOSING → ENDED), asks 6 structured Spanish follow-up questions,
  integrates RAG retrieval and LLM answer generation, and falls back safely when
  RAG/LLM is unavailable.  Escalation is explicitly out of scope for this phase.
  55 tests pass, 153 total in ``tests/conversation/``.
- **2026-08-07:** Document lifecycle foundation implemented. ``backend/documents/``
  (``Document`` model, ``DocumentStatus`` enum, ``DocumentService``),
  ``backend/persistence/sqlite.py`` (SQLite documents table), ``backend/api/documents.py``
  (``POST``/``GET``/``DELETE /documents``), 15 tests (9 fast, 6 slow). The same
  ``document_id`` is attached to all ChromaDB chunks and used for targeted deletion
  via ``ChromaStore.delete_document_chunks()``. All 250 existing tests pass with
  no regressions. ``python-multipart`` added to dependencies.
- **2026-08-07:** Conversation domain foundation implemented (`backend/conversation/`).
  Finite state machine (``State`` / ``Event`` enums) with 7 valid transitions and
  ``InvalidTransitionError`` for unsupported pairs.  ``Message`` (frozen slots
  dataclass with 0-based ``turn_index``, ``MessageRole``, validators) and append-only
  ``History`` (len, iter, index, tuple snapshot).  ``PatientContext`` wraps
  ``backend.data.models.Patient`` with ``dia_postop >= 0`` and non-empty
  ``procedimiento`` guards.  ``CallContext`` aggregates patient, ``State``, ``History``,
  and UTC ``created_at``.  Stdlib-only, text-only — no voice/frontend/LLM/RAG
  dependencies.  98 tests pass.
- **2026-08-07 (pm):** First RAG-backed clinical answer endpoint implemented.
  ``backend/llm/`` (Llama 3.1 70B Versatile adapter with validation),
  ``backend/api/rag.py`` (``POST /rag/query``), 54 tests pass. Resolved D1
  (model = Llama 3.1 70B Versatile via Groq) and D7 (single provider, no
  failover).
- **2026-08-07 (pm):** RAG slice audit remediation. Configured ChromaDB collection
  with ``hnsw:space: cosine`` for correct cosine similarity semantics with BGE-M3
  embeddings. Added UTC ``ingested_at`` metadata to every stored chunk. Added
  autouse test fixture resetting the ChromaStore singleton between tests. Added
  security comment for ``trust_remote_code=True``. Documented pdfplumber choice
  and 800/150 defaults rationale in ``docs/ARCHITECTURE.md``.
- **2026-08-07:** First minimal RAG slice implemented. Added ``backend/persistence/``
  (ChromaDB access), ``backend/rag/`` (extract, chunk, embed, store, retrieve),
  ``tests/rag/`` (fixtures and 25 tests, 10 slow). Updated ``pyproject.toml`` with
  chromadb, sentence-transformers, pdfplumber dependencies. Added model cache and
  ChromaDB runtime data to ``.gitignore``. Resolved open decision D6 (chunking strategy
  = fixed-size with overlap).
=======
- Audio transport format — decision D5 is de facto HTTP REST with base64 WAV for the
  voice turn endpoints; WebSocket streaming remains an open option for future phases.
>>>>>>> 2d429fb (docs: synchronize current implementation documentation)

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
