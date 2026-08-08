# Project Status

## Current Phase

Phase 1 (persistence) and Phase 2 (RAG ingestion/retrieval) core are implemented as
the first minimal RAG slice. The ChromaDB access layer and full RAG pipeline (extract →
chunk → embed → store → retrieve) are available and tested. No HTTP endpoints, document
lifecycle, voice, or LLM modules have been added yet.

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
  and module-level singleton. SQLite tables remain unimplemented (deferred to later
  Phase 1 work).
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
  API key, temperature), ``adapter.py`` (Gemini 1.5 Flash integration with structured
  JSON output, Spanish prompt assembly, citation mapping, and multi-layer safety
  validation). Open decision D1 resolved: **Gemini 1.5 Flash** selected for its
  1M-token context window and free tier. Open decision D7 resolved: single provider
  (no failover chain for Phase 3).
- **First RAG endpoint:** ``backend/api/rag.py`` — ``POST /rag/query`` accepts a
  Spanish clinical question, retrieves relevant chunks from ChromaDB, generates a
  validated answer via Gemini 1.5 Flash, and returns traceable source citations.
  Falls back to ``insufficient_knowledge: true`` without calling the LLM when no
  chunk exceeds the similarity threshold.
- 54 new fast tests (41 LLM adapter, 13 API endpoint) pass; all 58 existing
  dataset tests and 24 RAG tests (14 fast, 10 slow) continue to pass.
- ``pyproject.toml`` updated with ``google-generativeai>=0.8.0`` dependency.
- ``backend/main.py`` registers the ``rag_router``.

## Completed (continued)

- **2026-08-07:** Conversation domain foundation implemented (`backend/conversation/`).
  Finite state machine (``State`` / ``Event`` enums) with 7 valid transitions and
  ``InvalidTransitionError`` for unsupported pairs.  ``Message`` (frozen slots
  dataclass with 0-based ``turn_index``, ``MessageRole``, validators) and append-only
  ``History`` (len, iter, index, tuple snapshot).  ``PatientContext`` wraps
  ``backend.data.models.Patient`` with ``dia_postop >= 0`` and non-empty
  ``procedimiento`` guards.  ``CallContext`` aggregates patient, ``State``, ``History``,
  and UTC ``created_at``.  Stdlib-only, text-only — no voice/frontend/LLM/RAG
  dependencies.  98 tests pass.

- **2026-08-07:** Conversation orchestrator implemented
  (``backend/conversation/orchestrator.py``).  ``ConversationOrchestrator`` connects
  the existing domain primitives (``PatientContext``, ``CallContext``, state machine,
  ``History``, ``Message``) with RAG retrieval (``backend/rag/retrieval.retrieve``) and
  LLM answer generation (``backend/llm/adapter.generate_rag_answer``) into a
  deterministic Spanish text-only call flow.  6 structured follow-up questions cover
  pain, fever, wound, appetite, sleep, and mobility.  Fallback messages
  when RAG/LLM is unavailable or returns insufficient knowledge.  55 new tests pass
  (153 total in the conversation module).

## In Progress

- Full Phase 1 (SQLite persistence tables for calls, summaries, documents).
- Document lifecycle module (``backend/documents/``).

## Completed (setup)

- **2026-08-07:** Automatic ``.env`` file loading via ``python-dotenv``.
  ``backend/main.py`` calls ``load_dotenv()`` at module level before any
  configuration-importing code runs.  ``python-dotenv>=1.0.0`` added to
  ``pyproject.toml`` dependencies.  ``.env`` is already excluded by
  ``.gitignore``.  README updated with ``.env`` creation instructions.

## Recent Changes

- **2026-08-07 (pm):** Conversation orchestrator implemented.  ``ConversationOrchestrator``
  drives the state machine through all six phases (IDLE → GREETING → CONSENT →
  QUESTIONS → CLOSING → ENDED), asks 6 structured Spanish follow-up questions,
  integrates RAG retrieval and LLM answer generation, and falls back safely when
  RAG/LLM is unavailable.  Escalation is explicitly out of scope for this phase.
  55 tests pass, 153 total in ``tests/conversation/``.
- **2026-08-07 (pm):** First RAG-backed clinical answer endpoint implemented.
  ``backend/llm/`` (Gemini 1.5 Flash adapter with validation), ``backend/api/rag.py``
  (``POST /rag/query``), 54 tests pass. Resolved D1 (model = Gemini 1.5 Flash) and
  D7 (single provider, no failover).
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

The immediate next milestone is completing Phase 1 (SQLite schema) and Phase 2
(document lifecycle + HTTP endpoints for document operations).

## Open Architectural Decisions

These are tracked in `docs/ARCHITECTURE.md` § Open Decisions. The four open decisions
(D2, D3, D5, D8) cover STT provider, TTS provider, audio transport format, and patient
data loading strategy. Five decisions (D1, D4, D6, D7, D9) have been resolved: language
model (Gemini 1.5 Flash), backend framework (FastAPI), chunking strategy (fixed-size
with overlap), LLM failover (single provider), and PDF extraction library (pdfplumber).
Each open decision has a "resolve by" deadline tied to the phase that needs it.

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
