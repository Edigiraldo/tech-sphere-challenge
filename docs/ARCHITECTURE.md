# Architecture

## Current State

The repository is in the initial setup stage. The application runtime has not been
implemented yet.

## Target Shape

Use one deployable application with clear internal modules before considering service
separation:

```text
Browser
  |
  v
Application Backend
  |- Conversation orchestration and state
  |- Voice adapters
  |- RAG ingestion and retrieval
  |- Document management
  |- Escalation decision logic
  |- Call summaries
  `- Metrics and persistence
       |
       |- Relational persistence
       |- Vector store
       `- Permitted language model provider
```

## Module Responsibilities

- **Conversation:** controls turn-taking, context, follow-up questions, and response flow.
- **Voice:** adapts browser audio to transcription and text-to-speech providers.
- **RAG:** extracts documents, creates embeddings, retrieves relevant chunks, and preserves source metadata.
- **Documents:** handles upload, processing status, listing, and deletion of indexed chunks.
- **Decision:** classifies criticity and escalation independently from patient-facing wording.
- **Summaries:** stores patient, procedure, symptoms, decision, sources, and next steps.
- **Metrics:** records latency, token usage, model calls, RAG calls, and estimated cost.

## Data Flow

```text
Patient audio
  -> transcription
  -> conversation state
  -> RAG retrieval
  -> permitted language model
  -> structured response validation
  -> escalation decision and source citations
  -> text-to-speech
  -> call summary and metrics
```

## Relevant Decisions

### Modular monolith first

Keep the first implementation in one backend repository with internal modules. This
reduces operational complexity and supports the challenge's fifteen-minute setup gate.

### RAG is a subsystem, not necessarily a separate backend

The RAG may call a vector store through a library or adapter. It does not need to be a
separate service unless deployment needs later justify that split.

### Safety boundary

The language model may propose a structured interpretation, but application code must
validate the output and apply conservative escalation behavior.

## Documentation Policy

Add a decision here only when it affects multiple modules, is difficult to reverse, or
has meaningful alternatives and risks. Do not record routine implementation steps.
