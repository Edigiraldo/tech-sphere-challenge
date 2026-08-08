# Project Status

## Current Phase

Phase 1 (project skeleton and persistence) in progress.  Architecture definition is
complete; the first implementation artifact — the normalized read-only dataset access
package — is built and tested.

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

## In Progress

- Phase 1 continuation: project skeleton (backend framework, config) and persistence
  layer (SQLite + ChromaDB initialisation).  Framework decision D4 is the next open
  decision to resolve.

## Recent Changes

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

## Next Milestones

Implementation follows the eight-phase plan in `docs/ARCHITECTURE.md` § Phased
Implementation Plan (sole source of truth for milestones and deliverables).

The immediate next milestone is Phase 1: project skeleton and persistence.

## Open Architectural Decisions

These are tracked in `docs/ARCHITECTURE.md` § Open Decisions. The eight open decisions
(D1–D8) cover language model selection, STT provider selection, TTS provider selection,
backend framework, audio transport format, chunking strategy, LLM failover, and patient
data loading strategy. Each has a "resolve by" deadline tied to the phase that needs it.

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
