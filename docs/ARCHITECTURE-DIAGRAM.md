# Architecture Diagram

## Runtime Architecture

```mermaid
flowchart TB
    Browser[Browser]

    subgraph Frontend[Browser Frontend]
        CallUI[Call interface<br/>MediaRecorder + WAV playback]
        AdminUI[Administration console<br/>upload, list, delete]
        MetricsUI[Metrics view]
    end

    subgraph App[FastAPI Modular Monolith]
        API[API routers]
        Calls[Voice call endpoints<br/>POST /calls<br/>POST /calls/{id}/turn]
        Documents[Document lifecycle<br/>POST/GET/DELETE /documents]
        RAGAPI[RAG query endpoint<br/>POST /rag/query]
        MetricsAPI[Metrics reporting API]

        Conversation[Conversation orchestrator<br/>state, history, patient context]
        Decision[Deterministic escalation engine<br/>GREEN / YELLOW / RED]
        RAG[RAG engine<br/>retrieve chunks + citations]
        LLM[Groq Llama 3.3 70B adapter<br/>structured JSON + safety validation]
        Voice[Voice adapters]
        STT[Groq Whisper Large V3]
        TTS[Kokoro ef_dora]
        Summary[Structured summary generator]
        Metrics[Metrics collector<br/>latency, tokens, cost]
        Persistence[Persistence layer]
    end

    subgraph Storage[Runtime Storage]
        SQLite[(SQLite<br/>calls, turns, summaries,<br/>alerts, documents)]
        Chroma[(ChromaDB<br/>chunks, embeddings,<br/>source metadata)]
        Uploads[(Uploaded PDFs)]
    end

    Groq[Groq Cloud]

    Browser --> CallUI
    Browser --> AdminUI
    Browser --> MetricsUI

    CallUI --> Calls
    AdminUI --> Documents
    MetricsUI --> MetricsAPI

    Calls --> Voice
    Voice --> STT
    Voice --> TTS
    STT --> Groq
    LLM --> Groq

    Calls --> Conversation
    Conversation -->|QUESTIONS: classify first| Decision
    Conversation -->|CLOSING clinical question only| RAG
    RAG -->|sufficient context only| LLM
    Calls --> Summary
    Calls --> Metrics
    Calls --> Persistence

    RAGAPI --> RAG
    RAGAPI -->|sufficient context only| LLM
    Documents --> Uploads
    Documents --> RAG
    Documents --> Persistence
    MetricsAPI --> Metrics

    RAG --> Chroma
    Persistence --> SQLite
    Persistence --> Chroma
    Summary --> SQLite
    Metrics --> SQLite
```

## Voice Turn Flow

```mermaid
sequenceDiagram
    participant P as Patient browser
    participant API as FastAPI
    participant STT as Groq Whisper
    participant C as Conversation orchestrator
    participant R as ChromaDB RAG
    participant L as Groq Llama 3.3
    participant D as Decision engine
    participant T as Kokoro TTS
    participant DB as SQLite

    P->>API: POST /calls/{call_id}/turn (base64 audio)
    API->>STT: Transcribe audio in Spanish
    STT-->>API: Patient transcription
    API->>C: Process patient message and current state
    C->>D: Classify symptom before clinical generation

    alt RED signal
        D-->>C: RED, should_escalate=true
        C-->>API: Urgent response, ENDED, no RAG/LLM
    else GREEN or first YELLOW
        D-->>C: Classification
        C-->>API: Deterministic acknowledgement + next question
    else Second YELLOW
        D-->>C: Escalation, transition to CLOSING
        C-->>API: Deterministic escalation response
    else CLOSING clinical question
        C->>R: Retrieve relevant chunks
        R-->>C: Chunks and traceable citations
        C->>L: Question + patient context + retrieved sources
        L-->>C: Validated Spanish JSON response
        C-->>API: Answer and citations
    else CLOSING non-question
        C-->>API: Deterministic farewell, ENDED, no RAG/LLM
    end

    API->>T: Synthesize response text
    T-->>API: WAV audio
    API->>DB: Persist turn, alert, and metrics
    opt Call ended
        API->>DB: Generate and persist structured summary
    end
    API-->>P: Audio, transcription, state, citations, escalation
```

## Document Lifecycle

```mermaid
sequenceDiagram
    participant A as Admin browser
    participant API as FastAPI
    participant S as Document service
    participant DB as SQLite registry
    participant R as RAG ingestion
    participant C as ChromaDB

    A->>API: POST /documents (PDF)
    API->>S: Validate and hash content
    S->>DB: Register document and status
    S->>R: Extract, chunk, embed
    R->>C: Store chunks with document_id
    S->>DB: Mark document ready
    API-->>A: ready + document_id

    A->>API: DELETE /documents/{document_id}
    API->>C: Delete chunks by document_id
    API->>DB: Mark document deleted
    API-->>A: deleted
```

## Deployment Shape

The current application is a single FastAPI modular monolith. Docker distribution is
intended to provide a prebuilt image containing the application, dependencies, BGE-M3
cache, and pre-indexed corpus. Runtime secrets such as `GROQ_API_KEY` are injected by
the environment and are never included in the image.

```mermaid
flowchart LR
    Judge[Judge machine]
    Image[Prebuilt application image]
    Volume[(Persistent runtime volume)]
    App[FastAPI + frontend]
    Groq[Groq Cloud]

    Judge -->|docker compose pull/up| Image
    Image --> App
    App <--> Volume
    App -->|GROQ_API_KEY| Groq
```
