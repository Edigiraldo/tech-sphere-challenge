# Project Status

## Current Phase

Initial repository setup and architecture preparation.

## Completed

- Challenge documentation reviewed.
- Synthetic dataset and reference documents available locally.
- OpenCode planner, coder, and auditor configuration prepared.
- Initial repository README created.

## In Progress

- Application implementation has not started.

## Next Milestone

Build a minimal text-based RAG slice that can:

1. Ingest a small sample of clinical documents.
2. Retrieve relevant chunks with source metadata.
3. Delete a document and all of its indexed chunks.
4. Verify retrieval and deletion with automated tests.

## Known Constraints

- The language model must be one of the models permitted by `.challenge-docs/stack-tecnico.md`.
- The supplied clinical and patient data is synthetic and not clinically validated.
- No real patient data, secrets, recordings, or credentials may be committed.
- The final setup must be reproducible in fifteen minutes or less.

## Update Rules

Update this file when a milestone is completed, a blocker appears, or the next milestone
changes. Do not turn it into a detailed changelog.
