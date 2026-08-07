# Project Status

## Current Phase

Architecture definition. The modular monolith architecture, module boundaries, data
flows, persistence design, adapter contracts, and phased implementation plan are
documented in `docs/ARCHITECTURE.md`. Application code has not been written yet.

## Completed

- Challenge documentation reviewed: stack-tecnico, requerimientos, rubrica-evaluacion,
  flujo-de-conocimiento, habeas-data, terminos-y-condiciones.
- Synthetic dataset and reference documents (107 PDFs, 4 XLSX files) available locally.
- OpenCode planner, coder, and auditor configuration prepared.
- Initial repository README created.
- Architecture defined: module catalog, data flows, persistence boundaries, permitted
  adapters, phased implementation plan, open decisions tracked.

## In Progress

- Architecture definition complete. See `docs/ARCHITECTURE.md` for the module
  boundaries, data flows, persistence, safety boundary, phased roadmap, and open
  decisions.

## Recent Changes

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
