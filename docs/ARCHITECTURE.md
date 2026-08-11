# Arquitectura

## Visión general

Un backend Python desplegable único (monolito modular) que sirve un frontal de
navegador. La aplicación implementa un agente de voz en español para seguimiento
postoperatorio usando datos sintéticos de pacientes colombianos.

La arquitectura está diseñada para las restricciones del reto: configuración
reproducible en 15 minutos o menos, solo modelos de lenguaje permitidos, RAG
local-first con fuentes trazables, escalamiento conservador, una API REST para la
gestión del ciclo de vida de documentos, una interfaz de llamada de voz en el
navegador con captura real de micrófono, una consola de administración en ``/admin``
y una API de métricas de solo lectura. El transporte WebSocket/streaming queda como
trabajo futuro.

## Diagrama de arquitectura

```text
┌──────────────────────────────────────────────────────────────────────┐
│                           NAVEGADOR                                   │
│                                                                      │
│  ┌──────────────────────────┐     ┌──────────────────────────────┐  │
│  │  Interfaz de llamada     │     │  Consola de administración    │  │
│  │  - Selección de paciente │     │  - Cargar documento           │  │
│  │  - Captura MediaRecorder │     │  - Listar documentos + estado │  │
│  │  - Reproducción WAV      │     │  - Sondeo de estado + refresco│  │
│  │  - Transcripción + hist. │     │  - Eliminar documento + chunks│  │
│  │  - Citas + escalamiento  │     │                               │  │
│  └────────────┬─────────────┘     └──────────────┬───────────────┘  │
└───────────────┼──────────────────────────────────┼───────────────────┘
                │                                  │
            HTTP REST                         HTTP REST
                │                                  │
┌───────────────┴──────────────────────────────────┴───────────────────┐
│                     BACKEND DE APLICACIÓN (Python)                    │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  api/                      Endpoints REST                     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│        │              │               │               │             │
│  ┌─────┴────┐  ┌──────┴──────┐  ┌─────┴─────┐  ┌──────┴──────┐     │
│  │ voice/   │  │conversation/│  │  rag/     │  │ documents/  │     │
│  │ STT, TTS │  │ estado,     │  │ ingerir,  │  │ cargar,     │     │
│  │ adaptad. │  │ flujo orq.  │  │ recuperar │  │ listar, elim│     │
│  └─────┬────┘  └──────┬──────┘  └─────┬─────┘  └──────┬──────┘     │
│        │              │               │               │             │
│  ┌─────┴──────┐  ┌────┴────────┐  ┌───┴──────────────┐              │
│  │ decision/  │  │ summaries/ │  │ llm/ (condicional) │              │
│  │ GREEN/     │  │ registro   │  │ Solo preguntas     │              │
│  │ YELLOW/RED │  │ estructur. │  │ clínicas en CIERRE │              │
│  └─────┬──────┘  └────┬────────┘  └───┬──────────────┘              │
│        │              │               │                              │
│        │   GREEN/YELLOW determinista; RED termina inmediatamente     │
│        │              │               │                              │
│        └──────────────┴──────────────┘                              │
│        │              │                                             │
│  ┌─────┴──────────────┴──────┐                                      │
│  │  metrics/                 │                                      │
│  │  latencia, tokens, costo  │                                      │
│  └───────────────────────────┘                                      │
│        │                                                            │
│  ┌─────┴───────────────────────────────────────────────┐            │
│  │  persistence/                                       │            │
│  │  ┌─────────────────┐  ┌──────────────────────────┐  │            │
│  │  │ SQLite           │  │ ChromaDB                 │  │            │
│  │  │ - llamadas       │  │ - chunks de documentos   │  │            │
│  │  │ - resúmenes      │  │ - embeddings (BGE-M3)    │  │            │
│  │  │ - metadatos docs │  │ - metadatos de fuente    │  │            │
│  │  └─────────────────┘  └──────────────────────────┘  │            │
│  └─────────────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────────────┘
```

## Límites de módulos

| Módulo | Responsabilidad | Límite clave |
|--------|----------------|-------------|
| `api/` | Superficie HTTP REST. Valida entradas, delega a módulos de dominio. Incluye routers de llamadas, documentos, RAG, métricas y resúmenes. Los endpoints WebSocket aún no están implementados. | Única capa que el navegador toca. Sin lógica de negocio. |
| `voice/` | Adaptadores STT y TTS tras una interfaz común. | Adaptador de E/S puro. No posee estado, datos de pacientes ni conocimiento clínico. |
| `conversation/` | Máquina de estados de llamada: saludo → consentimiento → preguntas estructuradas → cierre. Clasifica cada respuesta de seguimiento antes del procesamiento posterior. Invoca LLM second-approval para cada respuesta no-RED. Usa respuestas deterministas para GREEN/primer-YELLOW, termina RED inmediatamente y llama a RAG + `llm/` solo para preguntas clínicas durante CLOSING y dudas RAG solicitadas por la aprobación LLM. | Posee el estado de turno y el ensamblaje de prompts. Nunca llama a `documents/` ni toca directamente persistencia/embeddings. |
| `llm/` | Adaptador para **Llama 3.3 70B Versatile** vía Groq Cloud con JSON estructurado validado, controles de fundamentación, detección de inyección de prompts, fallbacks seguros, y **aprobación secundaria de seguridad** de clasificaciones deterministas (``backend/llm/approval.py``). | No sabe nada de voz, documentos, RAG o escalamiento. Texto puro de entrada/salida. |
| `rag/` | Ingestión (extraer → chunking → embedding BGE-M3 → almacenar en ChromaDB) y recuperación (embedding de consulta → búsqueda por similitud → devolver chunks + metadatos). | Posee el modelo de embedding, la colección ChromaDB, el chunking y la recuperación. No posee el ciclo de vida de documentos ni conoce pacientes/conversaciones. |
| `documents/` | Ciclo de vida de documentos: cargar, listar, estado, eliminar. Orquesta metadatos en SQLite y dispara la ingestión/eliminación de RAG. | Llama a `rag/` para ingestión y purgado. No llama a `rag/` para recuperación. |
| `decision/` | Clasificador de escalamiento (Green / Yellow / Red). Se ejecuta sobre las respuestas del paciente antes de cualquier llamada RAG/LLM durante QUESTIONS, usando reglas explícitas de síntomas, umbrales, manejo de negaciones y detección de ambigüedad. | Aislado de RAG, voz, documentos y salida del LLM. Produce un veredicto consumido por `conversation/`. Conservador: los falsos negativos son catastróficos. |
| `summaries/` | Al finalizar la llamada, produce un resumen estructurado (paciente, procedimiento, síntomas, decisión, fuentes citadas, próximos pasos). La persistencia SQLite del registro de resumen la maneja el módulo `persistence/`. | Solo escritura, solo lectura sobre el historial de conversación. |
| `metrics/` | Observa latencia (P50/P95), consumo de tokens, invocaciones del modelo, consultas RAG, costo estimado. El módulo colector alimenta endpoints tipados de solo lectura ``GET /metrics/summary``, ``GET /metrics/calls`` y ``GET /metrics/calls/{call_id}`` más una vista frontal de métricas. El colector y la API de reporte son responsabilidades distintas. | Observador no bloqueante. Nunca modifica el comportamiento de la aplicación. |
| `persistence/` | SQLite (llamadas, turnos, resúmenes, metadatos de documentos, alertas) y ChromaDB (chunks, embeddings, metadatos de fuente). | Solo los módulos propietarios escriben. `rag/` posee ChromaDB; `documents/`, `conversation/`, `summaries/`, `decision/` poseen sus tablas SQLite. |

## Flujos de datos

### Conversación de voz (por turno)

El transporte actual es **HTTP REST** con audio WAV codificado en base64
(``POST /calls/{call_id}/turn``). El navegador captura audio mediante MediaRecorder
y envía fragmentos WAV codificados en base64; las respuestas se decodifican y
reproducen como audio WAV. El transporte WebSocket/streaming queda como trabajo
futuro.

El orquestador implementa un **flujo que prioriza la seguridad**: cada respuesta del
paciente es clasificada por el módulo ``decision/`` antes de cualquier llamada
RAG/LLM. Las respuestas RED derivan inmediatamente a ENDED con un mensaje urgente de
seguridad y ``call_ended=True`` (no se permiten más turnos) sin pasar por aprobación
LLM. Cada respuesta no-RED durante QUESTIONS pasa por una **aprobación secundaria por
LLM** (``backend/llm/approval.py``) como revisor conservador de seguridad: el LLM puede
confirmar la clasificación determinista, subir la severidad (nunca degradarla),
solicitar una aclaración (máximo una por pregunta) o solicitar RAG por duda clínica.
Fallos, timeouts, salida inválida o intentos de degradación caen automáticamente a la
clasificación determinista. GREEN y primer YELLOW confirmados reciben acuses
deterministas en español sin generación RAG+LLM. Dos resultados YELLOW consecutivos
disparan el escalamiento. Las preguntas clínicas durante CLOSING se responden mediante
RAG+LLM con citas; las no-preguntas finalizan la llamada.

**LLM second-approval (aprobación secundaria por LLM):** Después de la clasificación
determinista en cada respuesta no-RED durante QUESTIONS, se invoca al LLM como revisor
conservador de seguridad (``backend/llm/approval.py``). El LLM puede confirmar la
clasificación, subir la severidad (nunca bajarla), solicitar una aclaración al paciente
(máximo una por pregunta), o solicitar RAG por duda clínica. RED nunca pasa por
aprobación LLM — deriva directamente a ENDED. Fallos, timeouts, salida inválida o
intentos de degradación de severidad caen automáticamente a la clasificación
determinista.

```
Navegador (WAV base64 vía HTTP POST) → voice/STT → conversation/ orquestador:
  1. Cargar perfil del paciente + historial de turnos
  2. **Clasificar** respuesta del paciente contra el dominio de síntomas (decision/classify)
  3a. Si RED → derivación inmediata: sin RAG/LLM, mensaje urgente de seguridad,
       transición directa a ENDED, ``call_ended=True``
  3b. Si GREEN / YELLOW (no-RED) → **LLM second-approval** (``backend/llm/approval.py``)
       - Confirmar clasificación determinista
       - Subir severidad (nunca bajarla; YELLOW no puede bajar a GREEN)
       - Solicitar aclaración (máximo 1 por pregunta; no avanza índice)
       - Solicitar RAG por duda (ejecutar RAG en QUESTIONS, continuar después)
       - Fallos/timeout salida inválida → caer a clasificación determinista
  3c. Si segundo YELLOW consecutivo → escalar a CLOSING
  4. (Solo CLOSING) Si pregunta clínica → llamar rag/retrieve + llm/generate
      con citas, permanecer en CLOSING
  5. Si CLOSING no-pregunta → finalizar llamada
  → voice/TTS → Navegador (WAV base64 en respuesta HTTP)
```

### Ciclo de vida de documentos

```
Carga:    Navegador → api/ → documents/upload
           → Calcular hash SHA-256 del contenido
           → Si existe registro activo con el mismo hash → devolver existente (idempotente)
           → persistence/SQLite (metadatos + content_hash)
           → rag/ingest → ChromaDB (chunks + embeddings, indexados por document_id)
Eliminar: Navegador → api/ → documents/delete → rag/delete_chunks (purgado ChromaDB por document_id)
           → persistence/SQLite (borrado suave: status='deleted', fila preservada para auditoría)

Reconciliar:  POST /documents/reconcile
           → Comparar document_ids de ChromaDB vs registro SQLite
           → Reportar IDs huérfanos en ChromaDB (faltantes en el registro o eliminados con chunks residuales)
           → Reportar entradas faltantes en ChromaDB (SQLite ready/processing pero sin chunks indexados)
           → ?clean=true elimina chunks huérfanos de ChromaDB
```

### Recuperación RAG durante la conversación

```
conversation/ → rag/retrieve(query, valid_document_ids) → BGE-M3 embed → ChromaDB top-k
→ Filtrar chunks cuyo document_id esté eliminado o no registrado
→ chunks + metadatos con citas trazables
conversation/ ensambla el prompt con los chunks recuperados y las citas de fuente
```

## Límites de persistencia

### Esquema SQLite (conceptual)

```
calls                — call_id, paciente_id, nombre_completo, procedimiento,
                       dia_postop, eps, state, started_at, ended_at,
                       total_turns, escalated
conversation_turns   — turn_id, call_id, turn_index, role, text, timestamp,
                       severity, domain
summaries            — summary_id, call_id, created_at, patient_summary,
                       procedure_summary, symptoms_summary, decision_summary,
                       sources_json, next_steps
documents            — document_id, filename, status, uploaded_at, size_bytes,
                        content_hash, error_message
escalation_alerts    — alert_id, call_id, created_at, severity, reason, domain
```

La eliminación de documentos usa **borrado suave**: eliminar un documento cambia su
``status`` a ``deleted`` y purga los chunks de ChromaDB, pero la fila SQLite se
conserva para auditabilidad. La columna ``error_message`` almacena una descripción
legible por humanos cuando ``status`` es ``failed``.

### Esquema ChromaDB

```
collection: clinical_knowledge
  distance: cosine  (configurado mediante metadatos hnsw:space)
  id: uuid
  embedding: vector float BGE-M3 (1024 dimensiones, normalizado L2)
  document: texto del chunk
  metadata: document_id, source_filename, chunk_index, page_number, ingested_at (UTC ISO-8601)
```

### Reglas clave

- Eliminar un documento debe eliminar todos los chunks de ChromaDB con el `document_id`
  correspondiente y hacer borrado suave de la fila SQLite (``status = 'deleted'``). La
  fila de metadatos se conserva para auditabilidad. Sin chunks huérfanos en ChromaDB.
- Las cargas duplicadas se detectan mediante hash SHA-256 del contenido: si existe un
  registro activo (estados READY o PROCESSING) con el mismo hash, el servicio devuelve
  el registro existente sin crear uno nuevo. Los registros FAILED también permiten la
  creación de un nuevo registro de ingesta, sin bloquear la recarga. Si el original fue
  eliminado, se crea un nuevo registro.
- La recuperación excluye automáticamente los chunks cuyo ``document_id`` no esté en
  el registro SQLite o cuyo estado en el registro sea ``DELETED``, asegurando que solo
  documentos activos y registrados contribuyan a los resultados de búsqueda.
- La reconciliación (``POST /documents/reconcile``) compara los IDs de documento de
  ChromaDB con el registro SQLite y puede limpiar chunks huérfanos bajo demanda.
- La ingestión del corpus es explícita (``scripts/ingest_corpus.py``) y nunca se
  ejecuta al inicio. Es idempotente: re-ejecutar es seguro ya que los duplicados se
  detectan por hash de contenido.
- Los datos de llamadas y resúmenes nunca se eliminan a través de la API de ciclo de
  vida de documentos.
- El almacén vectorial se reconstruye solo en re-indexación explícita, nunca al
  reiniciar.
- Los datos sintéticos de pacientes se cargan desde `dataset/` XLSX al inicio y son de
  solo lectura.

## Modelos permitidos y adaptadores de voz

El modelo de lenguaje es **Llama 3.3 70B Versatile** vía Groq Cloud, el sucesor actual
autorizado por los organizadores del reto. Proporciona inferencia rápida, salida JSON
estructurada y buen rendimiento en español. El modelo se selecciona mediante
``LLM_MODEL`` y se valida contra la lista blanca de modelos Groq permitidos.

Los adaptadores de voz (STT y TTS) son de libre elección. STT seleccionado:
**Groq Whisper Large V3** (D2 resuelta) — transcripción en español de latencia
ultra-baja vía Groq Cloud, implementado en ``backend/voice/groq.py`` tras el Protocolo
``SttProvider``. TTS seleccionado: **Kokoro-82M** (D3 resuelta) — voz española solo
CPU (``ef_dora``), implementado en ``backend/voice/tts/kokoro.py`` tras el Protocolo
``TTSProvider``. Los adaptadores se seleccionan al inicio mediante configuración y se
envuelven tras una interfaz común para que el módulo de conversación nunca dependa de
un proveedor específico.

## Límites de seguridad y validación

### Controles de suficiencia de recuperación

Antes de invocar el LLM, el pipeline de recuperación RAG aplica controles de calidad
cuantitativos para evitar que una recuperación débil o vacía llegue al modelo:

1. **Umbral de similitud** (defecto 0.25, env ``RAG_SIMILARITY_THRESHOLD``):
   los chunks por debajo de este piso de similitud coseno se descartan.
2. **Cantidad mínima de chunks** (defecto 2, env ``RAG_MIN_CHUNKS``): al menos esta
   cantidad de chunks debe superar el umbral de similitud.
3. **Similitud promedio mínima** (defecto 0.30, env ``RAG_MIN_AVG_SIMILARITY``):
   la similitud media de todos los chunks recuperados debe alcanzar este nivel.

Cuando algún control falla, la bandera ``RetrievalResult.sufficient`` es ``False``.
Los llamadores (endpoint API, orquestador) recurren a ``insufficient_knowledge`` sin
invocar el LLM — ningún contexto débil llega al modelo.

### Validación de salida estructurada

Antes de que la salida llegue al paciente, el código de aplicación valida la respuesta
estructurada del LLM:

1. El JSON se parsea y todos los campos requeridos están presentes.
2. Los valores `document_id` de las fuentes citadas existen en el registro de
   documentos.
3. La señal de escalamiento es consistente con la lista de síntomas y la clasificación
   de `decision/`.
4. El mensaje dirigido al paciente no contiene dosis de medicamento, procedimiento
   inventado ni afirmación clínica no trazable a una fuente citada.
5. El mensaje está en español.

Si la validación falla, la respuesta se descarta y se reintenta o se escala a un
fallback seguro.

### Validación de fundamentación post-hoc

Después de que el LLM produce una respuesta, un validador secundario de fundamentación
(``_validate_grounding``) verifica que:

- Todos los IDs de chunk citados existan en el contexto y tengan texto no vacío.
- Cuando la respuesta menciona una dosis de medicamento, al menos un extracto citado
  comparta un token significativo (>= 5 caracteres) con la respuesta.

Las advertencias de fundamentación se registran en el servidor y se exponen en
``validation_warnings`` solo cuando ``debug=True``. Nunca llegan a la salida visible
para el paciente.

### Política de escalamiento (la seguridad primero)

- **La clasificación ocurre antes de RAG/LLM.** Durante la fase QUESTIONS, la llamada a
  ``decision/classify`` controla todo el procesamiento posterior.
- **Red siempre escala inmediatamente.** El orquestador deriva: sin llamada RAG/LLM, se
  devuelve un mensaje claro de seguridad urgente en español, el estado transita
  directamente a ENDED con ``call_ended=True``, y el frontal deshabilita la grabación
  adicional. No se permiten más turnos.
- **Yellow escala por acumulación.** Dos turnos YELLOW consecutivos disparan el
  escalamiento (transición a CLOSING) con ``should_escalate=True``. El primer YELLOW
  recibe un acuse determinista sin RAG/LLM.
- **Green recibe acuse determinista.** Las respuestas GREEN obtienen un mensaje
  positivo específico del dominio y la siguiente pregunta estructurada, sin RAG/LLM.
- **Desconocido es yellow.** Los turnos no clasificables o con validación fallida se
  tratan como yellow por defecto.
- **La ambigüedad dispara indagación.** Una pregunta aclaratoria antes de clasificar.

### Defensa contra inyección de prompts

- Las instrucciones del sistema están en un rol de mensaje separado de la entrada del
  usuario (separación de roles de la API de Groq).
- El habla del paciente nunca se concatena en las instrucciones.
- El esquema de salida estructurada restringe al LLM a una forma JSON fija.
- Los intentos de cambio de rol en la salida del LLM se rechazan durante la validación.
- **Detección de inyección a nivel de entrada** (``_detect_injection``): la consulta se
  escanea en busca de patrones conocidos de jailbreak (cambio de rol, extracción de
  prompt del sistema, inyección de delimitadores, etiquetas ``[INST]``, etc.) antes de
  cualquier llamada al LLM. Cuando un patrón coincide, la llamada devuelve un fallback
  seguro en español (``insufficient_knowledge=True``) sin invocar el modelo.
- **Control de longitud**: las consultas de más de 2000 caracteres se rechazan en la
  capa de detección de inyección.
- **Controles a nivel de salida**: el validador de fundamentación detecta citas
  alucinadas y afirmaciones de medicamentos sin fundamento en la salida del LLM.

### Prevención de alucinaciones clínicas

- Se instruye al LLM para que solo cite fuentes del contexto RAG proporcionado.
- Si ningún chunk RAG cumple los controles de suficiencia, el agente indica que carece
  de información en lugar de inventar.
- El módulo `decision/` verifica de forma cruzada el razonamiento clínico del LLM
  contra reglas explícitas de señales de alarma independientes del LLM.
- La validación de fundamentación post-hoc verifica que las afirmaciones de dosis de
  medicamentos en la respuesta estén respaldadas por los extractos citados.
- Cuando el proveedor Groq falla (error de red, timeout o salida no parseable) o
  devuelve conocimiento insuficiente a pesar de tener chunks recuperados con suficiente
  similitud, un **fallback extractivo seguro de RAG** usa solo el chunk de mayor
  similitud con metadatos de cita preservados y marca ``insufficient_knowledge`` como
  ``True`` cuando la similitud del mejor chunk está por debajo de 0.30.
- Los fallos de Groq usan un fallback seguro genérico sin contenido extractivo de
  chunks, manteniendo el comportamiento original del proveedor aislado.

## Plan de implementación por fases

La implementación está ordenada para que cada fase produzca un artefacto comprobable y
la puerta de configuración en 15 minutos sea verificable desde la fase más temprana.

| Fase | Enfoque | Entregable | Estado |
|-------|-------|-------------|--------|
| 1 | Esqueleto del proyecto y persistencia | La app inicia, SQLite + ChromaDB init, pruebas de esquema pasan | ✅ Completo |
| 2 | Ingestión y eliminación de documentos (RAG) | Cargar → indexar → recuperar → eliminar → chunks eliminados; pruebas pasan | ✅ Completo |
| 3 | Adaptador LLM y salida estructurada | Texto de entrada/salida con JSON validado del modelo permitido | ✅ Completo |
| 4 | Adaptadores de voz | Ida y vuelta STT + TTS en español | ✅ Completo |
| 5 | Orquestación de conversación y decisión | Conversación basada en texto con RAG, escalamiento, resúmenes, colector de métricas | ✅ Completo |
| 6 | Interfaz de llamada en navegador | Captura real de voz en navegador y reproducción de audio en español (puerta G4) | ✅ Completo — captura MediaRecorder, POST vía fetch a /calls y /calls/{call_id}/turn, reproducción WAV, visualización de transcripción/historial/citas/escalamiento; streaming WebSocket aún no implementado |
| 7 | Consola de administración | Consola gráfica para cargar, listar, eliminar documentos con conocimiento en vivo (puerta G5) | ✅ Completo — página ``/admin`` con carga, listado, sondeo de estado, refresco y eliminación; la API REST de ciclo de vida de documentos permanece separada de la consola de administración |
| 8 | API de métricas y pulido | Endpoint de métricas, métricas en README, casos límite, todas las puertas superables | ✅ Completo — endpoints de métricas tipados de solo lectura (resumen, llamadas y detalle por llamada) y vista frontal de métricas; el módulo colector de métricas es distinto de la API de reporte; el generador de resúmenes existe |

## Decisiones abiertas

Estas decisiones afectan a múltiples módulos o tienen alternativas significativas. Cada
una tiene una fecha límite de resolución vinculada a la fase que la necesita.

| # | Decisión | Alternativas | Depende de | Resolver para |
|---|----------|-------------|------------|------------|
| D5 | Formato de transporte de audio | PCM16 crudo / codificado Opus / fragmentos MediaRecorder / streaming WebSocket | D4 | Fase 6 — de facto: HTTP REST con WAV codificado en base64 para los endpoints de turnos de voz; el transporte streaming/WebSocket sigue como opción futura |

## Decisiones resueltas

| # | Decisión | Opción elegida | Justificación | Resuelta |
|---|----------|--------------|-----------|----------|
| D1 | Modelo de lenguaje | **Llama 3.3 70B Versatile (Groq Cloud)** | Sucesor actual autorizado por los organizadores del reto tras la retirada del modelo Groq originalmente sugerido. Proporciona inferencia rápida y respuestas estructuradas en español. | Fase 3 |
| D2 | Proveedor STT | **Groq Whisper Large V3** | Recomendado en el ``stack-tecnico.md`` del reto para transcripción en español de latencia ultra-baja. Capa gratuita vía Groq Cloud. Implementado en la Fase 4 (``backend/voice/``). | Fase 4 |
| D4 | Framework backend | **FastAPI** (asíncrono, nativo WebSocket) | Requerido para la interfaz de llamada WebSocket (Fase 6), nativo asíncrono, soporte OpenAPI sólido para la consola de administración (Fase 7). Ya implementado en la Fase 1. | Fase 1 |
| D6 | Estrategia de chunking | **Tamaño fijo con solapamiento** (800 caracteres, 150 de solapamiento) | Simple, predecible, bien probado. El valor por defecto de 800 caracteres equilibra la completitud del contexto con la precisión de recuperación para los PDFs clínicos del reto (típicamente 1–3 párrafos por página). El solapamiento de 150 caracteres evita dividir a mitad de frase manteniendo la tasa de duplicación por debajo del 19 %. Ambos valores son ajustables mediante variables de entorno (``RAG_CHUNK_SIZE``, ``RAG_CHUNK_OVERLAP``). Implementado en la Fase 2. | Fase 2 |
| D7 | Failover del proveedor LLM | **Proveedor único Groq** | Un solo proveedor actual mantiene simple la configuración de ejecución y Docker; el fallback extractivo seguro de RAG maneja los fallos del proveedor. | Fase 3 |
| D8 | Carga de datos de pacientes | **Cargar los 40 perfiles al inicio** | El dataset es pequeño (40 pacientes, ~160 días de trayectoria, ~4 000 turnos de conversación), cabe cómodamente en memoria. La carga al inicio desde ``dataset/`` XLSX mediante ``backend/data/loader.py`` es más simple que la carga perezosa, evita condiciones de carrera durante la creación de llamadas y asegura disponibilidad inmediata de datos demográficos, trayectorias y conversaciones de referencia para el orquestador de conversación. Implementado durante las Fases 1–5 (capa de acceso con dataclasses de solo lectura en ``backend/data/``). | Fase 5 |
| D3 | Proveedor TTS | **Kokoro-82M** | Tamaño de modelo mínimo (82 M de parámetros, ~0.6 GB de RAM) que se ejecuta en CPU sin GPU; voz nativa en español (``ef_dora``), prosodia natural para tono clínico; genera audio float32 a 24 kHz que el adaptador normaliza a WAV PCM 16-bit para reproducción en navegador. El ``KokoroAdapter`` implementa el Protocolo ``TTSProvider`` con carga perezosa de dependencias — el paquete opcional ``kokoro`` solo se importa en la primera llamada a ``synthesize()``. Se consideró Piper pero la calidad de la voz española de Kokoro es significativamente mejor para la UX de conversación clínica del reto. Implementado en la Fase 4 (``backend/voice/tts/kokoro.py``). | Fase 4 |
| D9 | Biblioteca de extracción de texto PDF | **pdfplumber** | Extracción fiable a nivel de caracteres, números de página explícitos y manejo consistente de Unicode para texto clínico en español. pdfplumber tiene mantenimiento activo, no tiene dependencias externas del sistema y produce salida estructurada página por página — todo crítico para citas trazables de fuentes. Se consideró pyMuPDF pero su licencia AGPL es incompatible con la licencia propietaria del reto. | Fase 2 |

Cuando una decisión abierta se resuelva, muévela a esta sección de resueltas con la
opción elegida y su justificación. Añade decisiones aquí solo cuando afecten a
múltiples módulos, sean difíciles de revertir o tengan alternativas significativas —
evita pasos rutinarios de implementación.
