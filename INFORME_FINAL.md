# Informe Final — Tech Sphere Challenge 2026

## 1. Resumen ejecutivo

Tech Sphere es un agente de voz en español para seguimiento postoperatorio de pacientes
sintéticos colombianos. Conversa con el paciente mediante el navegador, consulta
conocimiento clínico mediante RAG, detecta señales de alarma, escala casos de riesgo y
genera un resumen trazable de cada llamada.

La solución implementa:

- Interfaz de voz real en navegador con `MediaRecorder` y reproducción WAV.
- RAG clínico con BGE-M3 y ChromaDB con citas trazables.
- LLM Llama 3.3 70B Versatile vía Groq Cloud.
- STT Groq Whisper Large V3 y TTS Kokoro-82M.
- Escalamiento conservador `GREEN/YELLOW/RED` con reglas deterministas.
- Consola de administración para gestión de conocimiento en vivo.
- Métricas observables: latencia, tokens, costos.

## 2. Declaración del modelo utilizado

### Modelo seleccionado

**Llama 3.3 70B Versatile** vía Groq Cloud.

### Por qué se eligió

1. **Autorización del organizador.** El modelo originalmente sugerido (Llama 3.1 70B vía Groq) ya no está disponible. El organizador confirmó por correo que se debe usar el sucesor vigente del mismo proveedor. Llama 3.3 70B Versatile es el sucesor directo en Groq Cloud.

2. **Rendimiento en español.** Llama 3.3 70B ofrece excelente comprensión y generación en español, incluyendo regionalismos colombianos y terminología clínica.

3. **Latencia.** Groq Cloud entrega tokens a velocidad casi instantánea mediante sus LPU, lo que permite respuestas fluidas en una conversación de voz en tiempo real.

4. **Salida estructurada.** El modelo soporta JSON estructurado, lo que facilita la validación de citas, clasificación de escalamiento y extracción de información clínica.

5. **Disponibilidad gratuita.** Groq Cloud ofrece nivel gratuito suficiente para desarrollo y demostración.

### Alternativas evaluadas

| Modelo | Por qué se descartó |
|---|---|
| Google Gemini 1.5 Flash | Buena ventana de contexto pero latencia mayor que Groq para conversación de voz en tiempo real |
| Llama 3.2 1B/3B local | Capacidad clínica insuficiente para RAG con documentos médicos complejos |
| Phi-3.5 Mini local | Capacidad limitada para generación de respuestas clínicas fundamentadas |

## 3. Arquitectura de la solución

### Tipo de arquitectura

Monolito modular en Python con FastAPI.

### Componentes principales

```text
Browser
├── Interfaz de llamada (MediaRecorder + WAV playback)
├── Consola de administración (/admin)
└── Vista de métricas (/metrics)

Backend (FastAPI)
├── api/          Endpoints REST
├── voice/        STT (Groq Whisper) + TTS (Kokoro)
├── conversation/ Orquestador y máquina de estados
├── decision/     Clasificador GREEN/YELLOW/RED
├── rag/          Ingestión, chunking, embeddings, recuperación
├── llm/          Adaptador Groq Llama 3.3
├── documents/    Ciclo de vida de documentos
├── summaries/    Generador de resúmenes estructurados
├── metrics/      Colector de métricas
└── persistence/  SQLite + ChromaDB
```

### Flujo de voz por turno

```text
1. Paciente habla → MediaRecorder graba audio
2. POST /calls/{call_id}/turn → envía audio base64
3. STT (Groq Whisper) → transcribe a texto español
4. Orquestador → consulta máquina de estados
5. Decision → clasifica GREEN/YELLOW/RED
6. Según clasificación y aprobación secundaria:
   - RED: mensaje urgente → ENDED (sin aprobación LLM)
   - GREEN/YELLOW no-RED: aprobación LLM conservadora, sin degradar severidad
   - Confirmación: respuesta determinista → siguiente pregunta
   - Duda aprobada: RAG → LLM → respuesta con citas → siguiente pregunta
   - Respuesta poco clara: aclaración y permanencia en la misma pregunta
   - Dos resultados YELLOW consecutivos: escalamiento → CLOSING
   - Pregunta clínica en CLOSING: RAG → LLM → respuesta con citas
7. TTS (Kokoro) → genera audio WAV
8. Browser reproduce la respuesta
```

## 4. Decisiones técnicas clave

### D1: LLM — Llama 3.3 70B Versatile (Groq Cloud)

**Alternativas:** Gemini 1.5 Flash, Llama 3.2 local, Phi-3.5 Mini local.

**Por qué Groq:** latencia ultra-baja para conversación de voz, español de alta calidad, salida estructurada, nivel gratuito.

**Riesgo:** dependencia de proveedor en la nube. Mitigado con fallback seguro en español.

### D2: STT — Groq Whisper Large V3

**Alternativas:** Whisper local, otros servicios de STT.

**Por qué:** transcripción en español con latencia ultra-baja vía Groq Cloud. Implementado en `backend/voice/groq.py` detrás del Protocolo `SttProvider`.

### D3: TTS — Kokoro-82M (ef_dora)

**Alternativas:** Piper, otros TTS.

**Por qué:** modelo ligero (~0.6 GB RAM), voz nativa en español (`ef_dora`), calidad comparable a modelos comerciales, funciona en CPU sin GPU. Implementado en `backend/voice/tts/kokoro.py` detrás del Protocolo `TTSProvider`.

### D4: Framework — FastAPI

**Alternativas:** Flask, Django, Starlette.

**Por qué:** async nativo, soporte WebSocket futuro, OpenAPI automático, buenas prácticas ASGI.

### D5: Transporte de audio — HTTP REST + WAV base64

**Estado actual:** HTTP REST con audio codificado en base64.

**Futuro:** WebSocket/streaming para conversación en tiempo real.

### D6: Chunking — Fijo 800/150

**Por qué:** simple, predecible, bien probado. 800 caracteres equilibra contexto completo contra precisión de recuperación. 150 caracteres de solapamiento evita cortar oraciones.

### D7: Failover LLM — Proveedor único (Groq)

**Por qué:** simplicidad operativa. Fallback extractivo seguro maneja fallos del proveedor.

### D8: Carga de datos — 40 perfiles al inicio

**Por qué:** dataset pequeño, carga simple, sin carreras de concurrencia, disponibilidad inmediata.

### D9: Extracción PDF — pdfplumber

**Por qué:** extracción confiable, manejo de Unicode, números de página explícitos, sin dependencias de sistema.

## 5. Seguridad y validación

### Flujo de seguridad first

1. Cada respuesta del paciente se clasifica **antes** de cualquier llamada a RAG/LLM.
2. `RED` termina inmediatamente sin consultar fuentes.
3. `GREEN` y primer `YELLOW` usan respuestas deterministas.
4. Solo preguntas clínicas en `CLOSING` activan RAG + LLM.

### Defensa contra inyección de prompts

- Separación de roles system/user en la API de Groq.
- Detección de patrones de jailbreak antes de llamar al LLM.
- Validación de salida estructurada JSON.
- Validación post-hoc de grounding y citas.
- Rechazo de afirmaciones de medicamentos/dosis sin soporte.

### Controles de suficiencia RAG

- Similitud mínima: 0.25
- Mínimo de chunks: 2
- Similitud promedio mínima: 0.30
- Si no se cumplen: `insufficient_knowledge` sin llamar al LLM.

## 6. Gestión de conocimiento (RAG)

### Pipeline

```text
PDF → extracción (pdfplumber) → chunking (800/150)
→ embeddings (BGE-M3) → ChromaDB → recuperación por similitud
```

### Ciclo de vida de documentos

- **Subida:** SHA-256 hash → detección de duplicados → SQLite + ChromaDB.
- **Consulta:** embedding de pregunta → búsqueda por similitud → filtros de suficiencia → LLM con contexto.
- **Eliminación:** soft-delete en SQLite + purge de chunks en ChromaDB.
- **Reconciliación:** `POST /documents/reconcile` detecta chunks huérfanos.

### Ingestión del corpus

```powershell
python scripts/ingest_corpus.py
```

Ingestión idempotente: re-ejecutar es seguro, los duplicados se detectan por hash.

## 7. Prompts y configuraciones

### Prompt del sistema (español)

El prompt implementado en `backend/llm/adapter.py` comienza así:

```text
Eres un asistente clínico virtual que ayuda a pacientes postoperatorios
en Colombia. Responde ÚNICAMENTE basándote en las fuentes proporcionadas.
```

Sus reglas principales son:

- Rol del agente como asistente de seguimiento postoperatorio.
- Instrucción de responder solo con base en el contexto RAG proporcionado.
- Restricción de no inventar medicamentos, dosis o procedimientos.
- Citar siempre el `chunk_id` exacto de cada afirmación.
- Responder en español colombiano, con tono claro y empático.
- Marcar `insufficient_knowledge: true` cuando las fuentes no sean suficientes.
- Formato de salida JSON estructurado con campos obligatorios.

La plantilla de usuario implementada incluye la pregunta del paciente y las fuentes
con sus identificadores, archivos y páginas:

```text
PREGUNTA DEL PACIENTE:
{query}

FUENTES DISPONIBLES:
{context}

Responde exclusivamente en formato JSON.
```

### Configuración del LLM

| Parámetro | Valor |
|---|---|
| Modelo | Llama 3.3 70B Versatile |
| Temperatura | 0.2 |
| Max tokens | 1024 |
| Proveedor | Groq Cloud |

### Configuración del RAG

| Parámetro | Valor | Variable de entorno |
|---|---|---|
| Modelo de embeddings | BGE-M3 | `RAG_EMBEDDING_MODEL` |
| Tamaño de chunk | 800 chars | `RAG_CHUNK_SIZE` |
| Solapamiento | 150 chars | `RAG_CHUNK_OVERLAP` |
| Top-k | 5 | `RAG_TOP_K` |
| Umbral de similitud | 0.25 | `RAG_SIMILARITY_THRESHOLD` |
| Mínimo de chunks | 2 | `RAG_MIN_CHUNKS` |
| Similitud promedio mínima | 0.30 | `RAG_MIN_AVG_SIMILARITY` |

## 8. Pruebas

### Cobertura

La suite rápida (`pytest`) cubre dataset, salud, chunking, LLM mock, RAG API,
persistencia, voz, conversación, escalamiento, resúmenes, métricas y frontend.
La suite lenta (`pytest -m slow`) cubre el pipeline RAG completo con BGE-M3 y PDFs
reales. Los conteos deben obtenerse de la ejecución final que acompañe la entrega y
no se fijan aquí para evitar reportar cifras obsoletas.

### Estrategia

- Pruebas unitarias para cada módulo.
- Pruebas de integración para endpoints API.
- Pruebas de regresión para el flujo completo de llamada.
- Mock de Groq para pruebas rápidas.
- Pruebas reales con corpus para pruebas lentas.

## 9. Métricas observables

La aplicación expone métricas en:

```text
GET /metrics/summary
GET /metrics/calls
GET /metrics/calls/{call_id}
```

Y una vista frontend en `/metrics`.

### Métricas capturadas

- Latencia total por turno (P50/P95).
- Latencia STT, TTS y LLM por turno.
- Tokens de entrada y salida.
- Invocaciones al modelo.
- Consultas RAG.
- Costo estimado por llamada.

## 10. Proceso de desarrollo con IA

### Herramientas utilizadas

- **OpenCode** como orquestador multi-agente (planner, coder, auditor).
- **Llama 3.3 70B** vía Groq para asistencia de desarrollo.
- **BGE-M3** para embeddings del RAG.

### Flujo de trabajo

1. `@planner` inspecciona el repositorio y produce un plan de implementación.
2. `@coder` implementa solo los archivos del plan.
3. `@auditor` revisa, ejecuta pruebas y reporta hallazgos.
4. Cada hallazgo se corrige antes de continuar.

### Iteraciones principales

1. **Fase 1-2:** esqueleto + persistencia + RAG pipeline.
2. **Fase 3:** adaptador LLM con validación estructurada.
3. **Fase 4:** adaptadores de voz STT/TTS.
4. **Fase 5:** orquestación conversacional + escalamiento + resúmenes.
5. **Fase 6:** interfaz de navegador con MediaRecorder.
6. **Fase 7:** consola de administración.
7. **Fase 8:** métricas + pulido.
8. **Iteración de estabilización:** corrección del detector de preguntas habladas en
   transcripciones STT sin tildes ni puntuación.

### Evaluación y ajuste de prompts

- Los prompts se refinaron iterativamente para respuestas seguras en español.
- Se añadieron capas de validación post-hoc de grounding.
- Se implementó detección de inyección de prompts a nivel de entrada.
- Se ajustaron umbrales de suficiencia RAG para evitar respuestas débiles.

## 11. Evidencias de la entrega

La evidencia audiovisual principal está en el video de demo de la entrega. Debe mostrar
la llamada de voz, una respuesta normal, una alerta `RED`, el ciclo de conocimiento vivo
desde `/admin`, las citas RAG, el rechazo de prompt injection y la respuesta a las dos
preguntas de cierre del reto.

La evidencia técnica reproducible está en:

- Arquitectura: `docs/ARCHITECTURE.md` y `docs/ARCHITECTURE-DIAGRAM.md`.
- Pruebas del orquestador: `tests/conversation/test_orchestrator.py`.
- Pruebas RAG: `tests/rag/` y `tests/test_rag_api.py`.
- Pruebas de documentos: `tests/test_documents.py`.
- Pruebas de voz: `tests/test_voice.py` y `tests/voice/test_tts.py`.
- Pruebas de métricas: `tests/metrics/` y `tests/test_metrics_api.py`.
- Configuración real del prompt: `backend/llm/adapter.py`.
- Modelo y dependencias: `pyproject.toml`, `.env` local y `backend/llm/config.py`.

La autorización del organizador para utilizar el sucesor vigente de un modelo retirado
debe conservarse como evidencia externa del informe y presentarse junto con la entrega.

## 12. Con más tiempo cambiaría

1. **WebSocket/streaming:** transporte de audio en tiempo real para reducir latencia percibida.
2. **Preguntas adaptativas:** generar preguntas de seguimiento basadas en el contexto del paciente en vez de un cuestionario fijo.
3. **Evaluación automatizada:** suite de evaluación con escenarios clínicos predefinidos y métricas de calidad de respuesta.
4. **Múltiples documentos de referencia:** permitir que el paciente suba documentos personales (recetas, órdenes médicas).
5. **Dashboard de monitoreo:** Panel en tiempo real para supervisar llamadas activas y alertas.
6. **Internacionalización:** soporte para otros idiomas además del español.
7. **Tests de audio end-to-end:** pruebas automatizadas que validan el flujo completo de micrófono a altavoz.
8. **Optimización de costos:** cache de respuestas frecuentes y optimización de tokens.

## 13. Limitaciones conocidas

1. El cuestionario de seguimiento es fijo (6 preguntas hardcodeadas).
2. Las preguntas clínicas con RAG solo funcionan durante `CLOSING`.
3. El transporte es HTTP REST, no WebSocket.
4. Los datos son sintéticos y no clínicamente validados.
5. No hay autenticación ni control de acceso.
6. El agente no sustituye criterio médico profesional.
