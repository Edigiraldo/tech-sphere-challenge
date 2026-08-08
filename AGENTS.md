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

## Orchestrator Responsibilities

The orchestrator coordinates the three specialized agents; it does not duplicate their
work.

- Do not independently implement code that belongs to `@coder`.
- Do not independently rerun tests, linting, builds, or audits already reported by
  `@auditor` unless a report is missing, contradictory, or the user explicitly asks for
  an independent verification.
- Do not repeat repository-wide exploration already performed by `@planner`.
- Trust the specialized agent's report as the source of truth for its assigned phase.
- Forward the planner proposal to the user for approval before invoking `@coder`.
- Forward auditor findings to `@coder` or `@planner` according to the escalation rules.
- Summarize agent reports and workflow state without replacing their technical checks.

### Planner approval gate

- `@planner` must only inspect and produce a proposal.
- `@planner` must stop after returning the proposal.
- `@coder` must not be invoked until the user explicitly approves the plan.
- If the user requests changes, return to `@planner` before implementation.

## Project context

This repository is a voice-based postoperative follow-up agent for synthetic Colombian
patient data. The important system boundaries are:

- Browser voice-call interface.
- Administration console for live document ingestion and deletion.
- Conversation orchestration and state.
- RAG ingestion and retrieval with traceable sources.
- Escalation decision logic.
- Structured call summaries and observable metrics.

The agent-facing project documentation is English and must remain English. Read
`docs/PROJECT.md` and `docs/STATUS.md` first. Read `docs/ARCHITECTURE.md` only when the
task affects architecture or interfaces. Read the relevant `.challenge-docs` files only
when the task requires their constraints; do not load the entire repository or dataset
by default.

## Worktrees and parallel work

### Mandatory session protocol

At the start of every coding session:

1. Run `git rev-parse --show-toplevel` and `git branch --show-current`.
2. Determine whether the current directory is the primary worktree or a task worktree.
3. If the current directory is the primary worktree and the user requests code or
   documentation changes:
   - Derive a short kebab-case task slug from the request.
   - Create `.worktrees/<task-slug>` with branch `task/<task-slug>`.
   - Continue all work from that task worktree.
4. If already inside a task worktree, do not create another nested worktree.
5. Report the active worktree and branch before implementation begins.

Before creating a task worktree, verify that the primary worktree does not contain
uncommitted or staged baseline changes required by the task. If it does, stop and ask the
user whether those changes should be committed or otherwise handled first.

### Parallel work rules

- Use one worktree per independent task or feature branch.
- Keep each worktree focused on one planner File Impact list.
- Do not edit another worktree's files or assume its uncommitted changes are available.
- Integrate branches only when the user explicitly requests it.
- Never commit, amend, reset, or push unless the user explicitly authorizes it.

## Escalation loop

- If the auditor rejects due to an implementation defect, send the report to `@coder`.
- If the auditor finds an architectural flaw, or two coder attempts fail, return to `@planner`.
- After a fix, rerun the auditor checks.
- Every auditor finding, doubt, or actionable suggestion is mandatory. The coder must
  resolve it or the planner must explicitly revise the architecture and scope.
- The work is not complete while the auditor has unresolved findings, doubts, or
  suggestions.
- After every remediation, rerun the auditor checks.
- Finish only with an `APPROVED` audit and no unresolved findings, or a clearly
  documented blocker explicitly accepted by the user.

### Definition of done

A task is complete only when:

- The approved scope is implemented.
- Relevant checks have been run by `@auditor`.
- Documentation impact is resolved.
- `@auditor` returns `APPROVED`.
- No findings, doubts, or suggestions remain unresolved.
- The worktree is clean, unless the user explicitly requests a staged or uncommitted result.

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

## Documentation Maintenance

- The planner must include a `Documentation Impact` section in every plan.
- The coder must update only the relevant documentation when behavior, architecture,
  interfaces, dependencies, milestones, or blockers change.
- The coder must remove or correct stale documentation found in the affected documents.
- The auditor must verify that changed documentation matches the code and that no stale
  statements remain in the affected scope.
- Do not create a new document for a routine implementation step.
