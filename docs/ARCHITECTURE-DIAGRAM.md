# Diagrama de arquitectura

## Arquitectura en tiempo de ejecución

```mermaid
flowchart TB
    Browser[Navegador]

    subgraph Frontend[Frontal del navegador]
        CallUI[Interfaz de llamada<br/>MediaRecorder + reproducción WAV]
        AdminUI[Consola de administración<br/>cargar, listar, eliminar]
        MetricsUI[Vista de métricas]
    end

    subgraph App[Monolito modular FastAPI]
        API[Routers de API]
        Calls[Endpoints de llamadas de voz<br/>POST /calls<br/>POST /calls/{id}/turn<br/>POST /calls/{id}/end]
        Documents[Ciclo de vida de documentos<br/>POST/GET/DELETE /documents<br/>POST /documents/reconcile]
        RAGAPI[Endpoint de consulta RAG<br/>POST /rag/query]
        MetricsAPI[API de reporte de métricas]

        Conversation[Orquestador de conversación<br/>estado, historial, contexto del paciente]
        Decision[Motor de escalamiento determinista<br/>GREEN / YELLOW / RED]
        RAG[Motor RAG<br/>recuperar chunks + citas]
        LLM[Adaptador Groq Llama 3.3 70B<br/>JSON estructurado + validación de seguridad]
        Voice[Adaptadores de voz]
        STT[Groq Whisper Large V3]
        TTS[Kokoro ef_dora]
        Summary[Generador de resúmenes estructurados]
        Metrics[Colector de métricas<br/>latencia, tokens, costo]
        Persistence[Capa de persistencia]
    end

    subgraph Storage[Almacenamiento en ejecución]
        SQLite[(SQLite<br/>llamadas, turnos, resúmenes,<br/>alertas, documentos)]
        Chroma[(ChromaDB<br/>chunks, embeddings,<br/>metadatos de fuente)]
        Uploads[(PDFs cargados)]
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
    Conversation -->|QUESTIONS: clasificar → RED cortocircuito| Decision
    Conversation -->|no-RED: gate de duda| LLM
    Conversation -->|duda confirmada → RAG inline| RAG
    Conversation -->|no-RED no-duda: aprobación 2.ª| LLM
    Conversation -->|CLOSING: pregunta clínica| RAG
    RAG -->|solo contexto suficiente| LLM
    Calls --> Summary
    Calls --> Metrics
    Calls --> Persistence

    RAGAPI --> RAG
    RAGAPI -->|solo contexto suficiente| LLM
    Documents --> Uploads
    Documents --> RAG
    Documents --> Persistence
    MetricsAPI --> Metrics

    RAG --> Chroma
    Persistence --> SQLite
    Persistence --> Chroma
    Summary --> SQLite
```

## Flujo de turno de voz

> **Nota:** La clasificación determinista de síntomas (GREEN / YELLOW / RED) ocurre
> antes de cualquier gate de duda o invocación LLM. Una clasificación RED cortocircuita
> directamente a escalamiento sin pasar por el gate de duda ni consultar modelos.

```mermaid
sequenceDiagram
    participant P as Navegador del paciente
    participant API as FastAPI
    participant STT as Groq Whisper
    participant C as Orquestador de conversación
    participant R as ChromaDB RAG
    participant L as Groq Llama 3.3
    participant D as Motor de decisión
    participant T as Kokoro TTS
    participant DB as SQLite

    P->>API: POST /calls/{call_id}/turn (audio base64)
    API->>STT: Transcribir audio en español
    STT-->>API: Transcripción del paciente
    API->>C: Procesar mensaje del paciente y estado actual
    C->>D: Clasificar síntoma antes de cualquier RAG/LLM

    alt Señal RED
        D-->>C: RED, should_escalate=true
        C-->>API: Respuesta urgente, ENDED, sin RAG/LLM, sin gate de duda
        Note over C: Alerta RED persistida (should_escalate=True)
    else No-RED → puerta de intención de duda (_check_doubt_intent)
        D-->>C: No-RED
        C->>L: ¿Es una duda clínica? (llm_confirm_doubt)
        alt Duda confirmada
            L-->>C: is_doubt=true, rag_query
            C->>R: RAG inline con rag_query
            R-->>C: Chunks y citas
            C->>L: Generar respuesta con fuentes
            L-->>C: Respuesta JSON validada
            C-->>API: Respuesta RAG + repetir la pregunta pendiente (sin avanzar índice)
            Note over C: Sin acumular YELLOW, sin alerta
        else No es duda → aprobación 2.ª
            L-->>C: is_doubt=false
            C->>L: llm_second_approval (confirmar/escalar/aclarar/rag)
            L-->>C: Acción de aprobación
            alt Confirmar o escalar (sin RED)
                C-->>API: Acuse determinista + siguiente pregunta
            else Solicitar aclaración
                C-->>API: Pregunta de aclaración (máx. 1), sin avanzar índice
            else Solicitar RAG
                C->>R: RAG en QUESTIONS
                R-->>C: Chunks y citas
                C->>L: Generar respuesta
                L-->>C: Respuesta JSON
                C-->>API: Respuesta + siguiente pregunta
            end
        end
    else Segundo YELLOW consecutivo
        D-->>C: should_escalate=true
        C-->>API: Respuesta de escalamiento, transición a CLOSING
        Note over C: Alerta YELLOW→RED o 2.º YELLOW persistida
    else CLOSING pregunta clínica
        C->>R: Recuperar chunks relevantes
        R-->>C: Chunks y citas trazables
        C->>L: Pregunta + contexto del paciente + fuentes recuperadas
        L-->>C: Respuesta JSON validada en español
        C-->>API: Respuesta y citas
    else CLOSING no-pregunta
        C-->>API: Despedida determinista, ENDED, sin RAG/LLM
    end

    API->>T: Sintetizar texto de respuesta
    T-->>API: Audio WAV
    API->>DB: Persistir turno
    opt Escalamiento conclusivo (should_escalate=True)
        API->>DB: Persistir alerta (INSERT OR IGNORE, idempotente)
    end
    opt Llamada finalizada
        API->>DB: Generar y persistir resumen estructurado
    end
    API-->>P: Audio, transcripción, estado, citas, escalamiento
```

## Ciclo de vida de documentos

```mermaid
sequenceDiagram
    participant A as Navegador de administración
    participant API as FastAPI
    participant S as Servicio de documentos
    participant DB as Registro SQLite
    participant R as Ingestión RAG
    participant C as ChromaDB

    A->>API: POST /documents (PDF)
    API->>S: Validar y calcular hash del contenido
    S->>DB: Registrar documento y estado
    S->>R: Extraer, chunking, embedding
    R->>C: Almacenar chunks con document_id
    S->>DB: Marcar documento como listo
    API-->>A: listo + document_id

    A->>API: DELETE /documents/{document_id}
    API->>C: Eliminar chunks por document_id
    API->>DB: Marcar documento como eliminado
    API-->>A: eliminado
```

## Finalización manual de llamada

```mermaid
sequenceDiagram
    participant P as Navegador del paciente
    participant API as FastAPI
    participant CS as CallStore (memoria)
    participant C as Orquestador (si disponible)
    participant DB as SQLite

    P->>API: POST /calls/{call_id}/end
    API->>DB: Verificar existencia de la llamada
    DB-->>API: CallRecord

    alt Llamada ya finalizada
        API->>DB: Obtener resumen existente
        DB-->>API: SummaryRecord
        API-->>P: 200 + resumen existente (idempotente)
    else Llamada activa
        API->>CS: Buscar orquestador en memoria
        alt Orquestador disponible
            CS-->>API: ConversationOrchestrator
            API->>C: Generar resumen desde historial
            C-->>API: SummaryRecord
        else Orquestador ausente (reinicio)
            CS-->>API: None
            API->>DB: Cargar turnos persistidos
            DB-->>API: ConversationTurnRecords
            API->>API: Generar resumen desde turnos SQLite
        end
        API->>DB: Persistir resumen + marcar ENDED + finalizar métricas
        API->>CS: Eliminar orquestador
        API->>DB: update_call_metrics_ended
        API-->>P: 200 + resumen completo
    end
```

## Forma de despliegue

La aplicación se distribuye como un monolito modular FastAPI en una imagen Docker. Docker
Desktop o Docker Engine con Compose ejecuta Uvicorn en `0.0.0.0:8000`, sin reload, y
permite configurar el puerto del host mediante `APP_PORT`. Los secretos de ejecución como
`GROQ_API_KEY` se inyectan desde `.env` y nunca se incluyen en la imagen. BGE-M3 se
descarga y carga bajo demanda en la primera operación RAG; su caché puede perderse al
eliminar el contenedor.

```mermaid
flowchart LR
    Judge[Máquina del jurado]
    Image[Imagen de aplicación]
    Compose[Docker Compose]
    SQLite[(Volumen sqlite_data)]
    Chroma[(Volumen chroma_data)]
    Uploads[(Volumen uploads_data)]
    App[FastAPI + frontal]
    Groq[Groq Cloud]

    Judge -->|docker compose up --build| Compose
    Compose --> Image
    Image --> App
    App <--> SQLite
    App <--> Chroma
    App <--> Uploads
    App -->|GROQ_API_KEY| Groq
```
