# Project Status

## Current Phase

Persistence (ChromaDB + SQLite), RAG ingestion/retrieval (extract → chunk → embed →
store → retrieve), the LLM adapter (Llama 3.1 70B Versatile via Groq), document lifecycle endpoints
(POST/GET/DELETE /documents), and the conversation domain foundation (state machine,
message history, patient/call context) are implemented and tested. Voice adapters
(STT) and conversation orchestration with RAG-backed dialogue and escalation remain.

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
- ``backend/data/`` package implemented: typed read-only data models (``Patient``,
  ``Trajectory``, ``Conversation``, ``ConversationTurn``, ``PDFReference``), XLSX
  loaders, and PDF path resolver.  ``label_ground_truth`` is isolated from runtime
  code via module boundaries and static tests.  58 dataset tests pass.

- **2026-08-07:** ``backend/data/`` dataset access package implemented.  Normalized
  read-only models for patients (merged clinical + demographic), trajectories,
  conversations (runtime-safe and evaluation-only loaders), and PDF corpus resolver.
  58 tests cover all loaders, models, PDF resolver, and static isolation of
  ``label_ground_truth`` from runtime code.  Dataset PDF count verified at 102
  (accessible via the PDF resolver).
- **2026-08-07:** Auditor findings F1–F7 remediated in ``backend/data/``:
  - **F1:** PDF discovery now logs OSError/UnicodeError instead of silently skipping
    inaccessible paths.  Tests use on-disk cross-checks instead of hardcoded counts.
  - **F2:** Standard Python cache/build ignores added to ``.gitignore``.
  - **F3:** Unparseable ``fecha_cirugia`` now raises ``ValueError`` (no silent
    fallback to ``today()``).
  - **F4:** ``_parse_comorbilidades`` and ``_parse_adaptation_fields`` deduplicated
    into shared ``_parse_json_list`` helper.
  - **F5:** Conversation loading validates ``dia_postop`` and ``paciente_id``
    homogeneity across turns; raises ``ValueError`` on mismatch.
  - **F6/F7:** ``_coerce_int`` / ``_coerce_float`` log warnings on non-numeric
    input; empty-conversation guard added.
  All 58 tests pass.
- **2026-08-07:** Architecture documentation remediated: ARCHITECTURE.md reduced for
  conciseness and sustainability; duplicated milestones table removed from STATUS.md
  in favor of a pointer to the architecture roadmap; README.md updated with link to
  full technical architecture.
- FastAPI project skeleton: `pyproject.toml` with dependencies, local venv setup
  instructions, `backend/main.py` with CORS middleware and `GET /health` returning
  `{"status": "ok"}`, and entry-point script (`tech-sphere`).
- `tests/test_health.py` validates the `/health` endpoint.
- **Persistence layer (partial):** `backend/persistence/chroma.py` — ChromaDB
  `ChromaStore` wrapper with collection init, chunk insertion, document-chunk deletion,
  and module-level singleton.
- **RAG pipeline:** `backend/rag/` — `config.py` (dataclass with env-var overrides),
  `extract.py` (pdfplumber-based PDF text extraction), `chunking.py` (fixed-size
  overlapping chunks with full metadata), `embeddings.py` (BGE-M3 via
  sentence-transformers as ChromaDB embedding function), `store.py` (ChromaDB add/query
  ops), `ingestion.py` (extract → chunk → store pipeline), `retrieval.py`
  (embed query → similarity search → `RetrievalResult` with citations).
- Open decision D6 (chunking strategy) resolved: fixed-size with overlap.
- Fast tests (15) pass: chunking unit tests, extraction error paths, health endpoint.
  Slow tests (10) for full ingestion/retrieval are gated behind ``pytest.mark.slow``.
- **Phase 3 (LLM adapter):** ``backend/llm/`` — ``config.py`` (model selection,
  API key, temperature), ``adapter.py`` (Llama 3.1 70B Versatile via Groq with
  structured JSON output, Spanish prompt assembly, citation mapping, and multi-layer
  safety validation). Open decision D1 resolved: **Llama 3.1 70B Versatile** via
  **Groq Cloud** selected for its fast inference, native JSON output, and strong
  Spanish performance. Open decision D7 resolved: single provider (no failover
  chain for Phase 3).
- **First RAG endpoint:** ``backend/api/rag.py`` — ``POST /rag/query`` accepts a
  Spanish clinical question, retrieves relevant chunks from ChromaDB, generates a
  validated answer via Llama 3.1 70B Versatile, and returns traceable source
  citations.  Falls back to ``insufficient_knowledge: true`` without calling the
  LLM when no chunk exceeds the similarity threshold.
- 54 new fast tests (41 LLM adapter, 13 API endpoint) pass; all 58 existing
  dataset tests and 24 RAG tests (14 fast, 10 slow) continue to pass.
- ``pyproject.toml`` updated with ``groq>=0.9.0`` dependency for the LLM adapter.
- ``backend/main.py`` registers the ``rag_router``.
- **2026-08-07:** Conversation domain foundation implemented (`backend/conversation/`).
  Finite state machine (``State`` / ``Event`` enums) with 7 valid transitions and
  ``InvalidTransitionError`` for unsupported pairs.  ``Message`` (frozen slots
  dataclass with 0-based ``turn_index``, ``MessageRole``, validators) and append-only
  ``History`` (len, iter, index, tuple snapshot).  ``PatientContext`` wraps
  ``backend.data.models.Patient`` with ``dia_postop >= 0`` and non-empty
  ``procedimiento`` guards.  ``CallContext`` aggregates patient, ``State``, ``History``,
  and UTC ``created_at``.  Stdlib-only, text-only — no voice/frontend/LLM/RAG
  dependencies.  98 tests pass.
- **2026-08-07:** Document lifecycle foundation implemented:
  - ``backend/documents/`` — ``models.py`` (``Document`` dataclass, ``DocumentStatus``
    enum with ``pending``, ``processing``, ``ready``, ``failed``, ``deleted``),
    ``service.py`` (``DocumentService`` — upload, list, delete).
  - ``backend/persistence/sqlite.py`` — SQLite ``documents`` table with CRUD
    operations, WAL journal mode, and ``init_sqlite`` / ``_reset_sqlite`` for test
    lifecycle management.
  - ``backend/api/documents.py`` — ``POST /documents`` (upload PDF, ingest into RAG,
    return status), ``GET /documents`` (list with optional status filter),
    ``DELETE /documents/{document_id}`` (purge ChromaDB chunks, mark deleted).
  - Critical invariant: the same ``document_id`` is attached to every ChromaDB chunk
    during ingestion and used for targeted deletion via
    ``ChromaStore.delete_document_chunks()``.
  - ``python-multipart`` added to project dependencies for file upload support.
  - 9 fast API-level tests pass (upload validation, processing failure, listing,
    deletion of non-existent).
  - 6 slow end-to-end tests (upload real PDF → ready, list after upload, filter by
    status, retrieval availability, complete indexed-chunk deletion, post-deletion
    retrieval returns nothing) gated behind ``pytest.mark.slow``.
  - All 250 existing tests continue to pass with no regressions.
  - ``docs/STATUS.md`` updated; document lifecycle moved from In Progress to Completed.

- **2026-08-07:** Conversation orchestrator implemented
  (``backend/conversation/orchestrator.py``).  ``ConversationOrchestrator`` connects
  the existing domain primitives (``PatientContext``, ``CallContext``, state machine,
  ``History``, ``Message``) with RAG retrieval (``backend/rag/retrieval.retrieve``) and
  LLM answer generation (``backend/llm/adapter.generate_rag_answer``) into a
  deterministic Spanish text-only call flow.  6 structured follow-up questions cover
  pain, fever, wound, appetite, sleep, and mobility.  Fallback messages
  when RAG/LLM is unavailable or returns insufficient knowledge.  55 new tests pass
  (153 total in the conversation module).
- **2026-08-08:** Escalation decision engine implemented (``backend/decision/``).
  ``classify(patient_text, domain)`` returns a typed ``EscalationResult`` with
  ``Severity`` (GREEN / YELLOW / RED), ``should_escalate``, ``reason``, and
  ``next_action``.  The engine is stdlib-only, text-only, deterministic, and
  conservative: false negatives are catastrophic so the engine biases toward
  YELLOW and RED.  Six symptom domains mirror the follow-up questions (pain,
  fever, wound, appetite, sleep, mobility).  Classification uses explicit
  Spanish red-flag lexicons, numeric thresholds (pain >= 8 → RED, temp >= 38.5 → RED),
  negation handling, ambiguity detection, and cross-cutting critical flags.
  The standalone engine is complete: 125 tests pass across
  ``tests/decision/`` (30 lexicon, 13 model, 82 rule engine).  Orchestrator
  integration (wiring RED/``ESCALATION_TRIGGER`` and two-consecutive-YELLOW
  escalation into the conversation flow) remains pending.

- **2026-08-08:** STT adapter foundation implemented (``backend/voice/``).

- **2026-08-08:** Phase 1 SQLite tables completed and summaries module implemented.
  ``backend/persistence/sqlite.py`` extended with frozen typed ``CallRecord``,
  ``ConversationTurnRecord``, ``SummaryRecord``, ``EscalationAlertRecord`` dataclasses
  and SQLite schema + CRUD for ``calls``, ``conversation_turns``, ``summaries``, and
  ``escalation_alerts`` tables.  ``backend/summaries/`` created with pure typed summary
  generator (``generate_summary``, ``SummaryResult``, ``SummarySection``,
  ``SourceReference``) — stdlib-only, deterministic, no LLM/RAG/network calls.
  Generator produces Spanish-language sections for patient demographics, procedure,
  six symptom domains, escalation decision, and next steps.  41 persistence tests
  (model validation + CRUD) and 44 summary tests (generator logic, model validation,
  domain mapping, escalation decision, next steps, determinism) pass.  All 622
  existing fast tests pass with no regressions.

- **2026-08-08:** Frontend shell implemented.  ``frontend/`` directory with vanilla
  HTML/CSS/JS (no frameworks, no backend API connections, no microphone/audio APIs).
  ``index.html`` provides synthetic patient selection (5 hardcoded patients) and a
  start-call button that navigates to ``/call``.  ``call.html`` renders a call
  interface with visible call-state badge, record/stop placeholder toggle,
  conversation history with role-labelled messages, transcript area, audio-response
  player placeholder, and call timer.  ``backend/main.py`` serves frontend assets
  via ``FileResponse`` routes (GET ``/``, GET ``/call``) and a ``StaticFiles`` mount
  at ``/static``.  8 focused tests verify HTML, CSS, JS serving and confirm the
  ``/health`` endpoint is not disrupted.  All existing tests continue to pass.
  Typed ``SttProvider`` Protocol, normalised ``TranscriptionResult`` dataclass,
  ``SttError`` / ``SttConfigError`` / ``SttProviderError`` / ``SttAudioError``
  exception hierarchy, frozen Spanish-first ``GroqWhisperConfig`` (model fixed to
  ``whisper-large-v3``, language fixed to ``"es"``), ``GroqWhisperProvider`` adapter
  with bytes/file handling and robust error mapping for empty/invalid audio, missing
  API key, rate-limit, auth, network, and provider errors, and public
   ``transcribe_audio()`` dependency-injection entry point.  56 unit tests pass
    (all Groq API calls mocked).  Open decision D2 resolved: STT provider = Groq
    Whisper Large V3.  ``groq>=0.9.0`` added to project dependencies.

## In Progress

- TTS adapter (Phase 4, depends on D5).
- Audio transport format — decision D5 pending (Phase 6).

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
- **2026-08-08:** Voice turn endpoints implemented.
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

## Next Milestones

Implementation follows the eight-phase plan in `docs/ARCHITECTURE.md` § Phased
Implementation Plan (sole source of truth for milestones and deliverables).

Phase 1 (persistence) is complete: ChromaDB, SQLite documents table, and the new
``calls``, ``conversation_turns``, ``summaries``, and ``escalation_alerts`` tables
are all implemented with typed models and CRUD operations.  Phase 2 (document
lifecycle + RAG) is substantially complete.  The summaries module
(``backend/summaries/``) provides a pure deterministic summary generator.
Phase 3 (LLM adapter), Phase 4 (STT + TTS adapters), and Phase 5 (conversation
orchestration with escalation) are substantially complete.  The immediate next
step is resolving the audio transport format decision (D5).

## Open Architectural Decisions

These are tracked in `docs/ARCHITECTURE.md` § Open Decisions. The two open decisions
(D5, D8) cover audio transport format and patient data loading strategy. Seven 
decisions (D1, D2, D3, D4, D6, D7, D9) have been resolved: language model (Llama
3.1 70B Versatile via Groq), STT provider (Groq Whisper Large V3), TTS provider (Kokoro-82M), backend
framework (FastAPI), chunking strategy (fixed-size with overlap), LLM failover (single
provider), and PDF extraction library (pdfplumber).  Each open decision has a "resolve by"
deadline tied to the phase that needs it.

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
appears, or the next milestone changes. Do not turn it into a detailed changelog.
