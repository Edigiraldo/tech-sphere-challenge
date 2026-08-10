# Guía del proyecto

## Propósito

Implementación del Tech Sphere Challenge: un agente de voz en español para
seguimiento postoperatorio usando datos sintéticos de pacientes colombianos.

El producto debe soportar:

- Conversaciones de voz en español mediante endpoints HTTP REST (``POST /calls``,
  ``POST /calls/{call_id}/turn``) con audio WAV codificado en base64 y captura de
  micrófono nativa del navegador (MediaRecorder). El transporte WebSocket/streaming
  queda como trabajo futuro.
- Generación aumentada por recuperación (RAG) clínica con citas trazables de fuentes.
- Carga, listado, seguimiento de estado de procesamiento y eliminación de documentos
  en vivo (incluyendo purgado de chunks indexados) mediante una API REST
  (``POST/GET/DELETE /documents``) y una consola de administración gráfica en
  ``/admin`` con carga, sondeo de estado, refresco y eliminación.
- Decisiones de escalamiento conservadoras con una política de clasificación que
  prioriza la seguridad.
- Resúmenes estructurados de llamadas con paciente, procedimiento, síntomas, decisión,
  fuentes y próximos pasos.
- Métricas observables: latencia, consumo de tokens, invocaciones del modelo, consultas
  RAG y costo estimado por llamada. El módulo colector de métricas alimenta endpoints
  de API de métricas tipados de solo lectura (``GET /metrics/summary``,
  ``GET /metrics/calls`` y ``GET /metrics/calls/{call_id}``) y una vista frontal de
  métricas.

El modelo de lenguaje debe ser uno de los cuatro permitidos por el reto
(`.challenge-docs/stack-tecnico.md`). El resto del stack — orquestación, voz, RAG,
embeddings — es de libre elección.

## Arquitectura

El objetivo es un monolito modular: un backend Python con módulos internos y un
frontal de navegador. El frontal proporciona una interfaz de llamada con integración
de voz real (MediaRecorder + reproducción WAV), una consola de administración en
``/admin`` para la gestión del ciclo de vida de documentos y una vista de métricas.
El backend del ciclo de vida de documentos (``POST/GET/DELETE /documents``) es un
módulo separado de la UI de la consola de administración.
Consulta `docs/ARCHITECTURE.md` para el catálogo completo de módulos, flujos de datos,
límites de persistencia, adaptadores permitidos, plan de implementación por fases y
decisiones abiertas.

## Mapa del repositorio

```text
backend/               Backend de la aplicación (monolito modular Python)
  data/                Acceso tipado de solo lectura al dataset (pacientes, trayectorias,
                         conversaciones, resolvedor de PDFs)
  api/                 Endpoints REST (llamadas, documentos, RAG, métricas); WebSocket aún no implementado
  voice/               Adaptadores STT y TTS
  conversation/        Orquestación de diálogo y máquina de estados
  llm/                 Adaptador del modelo de lenguaje permitido
  rag/                 Ingestión, embedding y recuperación de documentos
  documents/           Ciclo de vida de documentos (cargar, listar, estado, eliminar)
  decision/            Clasificación de escalamiento
  summaries/           Generación de resúmenes estructurados de llamadas
  metrics/             Instrumentación de latencia, tokens y costo
  persistence/         Capa de acceso a SQLite y ChromaDB
frontend/              UI de navegador (HTML/CSS/JS vanilla): interfaz de llamada con MediaRecorder + API, consola de administración, métricas
dataset/               Datos sintéticos del reto y PDFs de referencia
docs/                  Documentación mantenida del proyecto
.challenge-docs/       Requisitos del reto y reglas de evaluación
```

## Dónde buscar

| Tarea | Primeros archivos a inspeccionar |
| --- | --- |
| Arquitectura | `docs/ARCHITECTURE.md` |
| Trabajo actual y decisiones abiertas | `docs/STATUS.md` |
| Ciclo de vida de documentos | `backend/documents/`, `backend/rag/` |
| Recuperación RAG | `backend/rag/` |
| Flujo de conversación | `backend/conversation/` |
| Lógica de escalamiento | `backend/decision/` |
| Entrada/salida de voz | `backend/voice/` |
| Adaptador LLM | `backend/llm/` |
| Contrato de API | `backend/api/` |
| Esquema de persistencia | `backend/persistence/` |
| UI del navegador | `frontend/` |
| Restricciones del reto | `.challenge-docs/README.md` y `.challenge-docs/rubrica-evaluacion.md` |
| Modelos permitidos | `.challenge-docs/stack-tecnico.md` |

## Reglas de trabajo

- Lee este archivo y `docs/STATUS.md` antes de explorar el repositorio.
- Lee `docs/ARCHITECTURE.md` solo cuando la tarea afecte la arquitectura o las interfaces.
- Inspecciona únicamente los archivos listados por el planificador y sus dependencias directas.
- Mantén los detalles de implementación en el código, docstrings y pruebas.
- Mantén los hechos arquitectónicos estables en `docs/ARCHITECTURE.md`.
- Usa `docs/ARCHITECTURE-DIAGRAM.md` para los flujos visuales actuales de ejecución y despliegue.
- Mantén el progreso actual, hitos y bloqueos en `docs/STATUS.md`.
- Registra las decisiones arquitectónicas no resueltas en `docs/ARCHITECTURE.md` § Decisiones abiertas.
- Elimina documentación obsoleta cuando el comportamiento que describe sea eliminado.
