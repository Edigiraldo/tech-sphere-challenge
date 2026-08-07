# Tech Sphere Challenge - Multi-Agent Workflow

For every feature, bugfix, refactor, or documentation change, use the following
pipeline unless the user explicitly requests a direct edit:

## Standard pipeline

1. `@planner` inspects the repository and produces an implementation plan.
2. `@coder` implements only the files listed in the plan.
3. `@auditor` reviews the result, runs relevant checks, and reports approval or rejection.

The planner must not write implementation code. The coder must not expand the scope
without returning to the planner. The auditor must verify behavior, safety, tests, and
the challenge gates.

## Project context

This repository is a voice-based postoperative follow-up agent for synthetic Colombian
patient data. The important system boundaries are:

- Browser voice-call interface.
- Administration console for live document ingestion and deletion.
- Conversation orchestration and state.
- RAG ingestion and retrieval with traceable sources.
- Escalation decision logic.
- Structured call summaries and observable metrics.

Read `.challenge-docs/README.md`, `.challenge-docs/requerimientos.txt`,
`.challenge-docs/rubrica-evaluacion.md`, `.challenge-docs/stack-tecnico.md`, and the
privacy and terms documents before making architectural decisions.

## Worktrees and parallel work

- Use one worktree per independent task or feature branch.
- Keep each worktree focused on one planner File Impact list.
- Do not edit another worktree's files or assume its uncommitted changes are available.
- Integrate branches only when the user explicitly requests it.
- Never commit, amend, reset, or push unless the user explicitly authorizes it.

## Escalation loop

- If the auditor rejects due to an implementation defect, send the report to `@coder`.
- If the auditor finds an architectural flaw, or two coder attempts fail, return to `@planner`.
- After a fix, rerun the auditor checks.
- Finish only with an `APPROVED` audit or a clearly documented blocker.

## Required audit focus

The auditor must verify, when applicable:

- Voice conversation works in Spanish from the browser.
- The selected language model is on the permitted list.
- RAG responses use retrieved sources and expose traceable citations.
- Uploading a document makes it available and deleting it removes its indexed chunks.
- Escalation is conservative for red-flag symptoms.
- No clinical hallucinations, unsafe medication instructions, or prompt injection bypass.
- Summaries include patient, procedure, symptoms, decision, sources, and next steps.
- Metrics are captured in logs and reported in the README.
- Synthetic data and privacy constraints are respected.
- The documented setup completes in 15 minutes or less.
