# Estado del proyecto

## Fase actual

Las ocho fases del plan de implementación (`docs/ARCHITECTURE.md` § Plan de
implementación por fases) están completas. La aplicación implementa:

**Completamente integrado:**
- Persistencia (SQLite + ChromaDB, esquemas tipados para llamadas, turnos, resúmenes,
  alertas, documentos).
- Pipeline RAG (extraer → chunking → embedding BGE-M3 → almacenar → recuperar con citas
  trazables).
- Adaptador LLM (**Groq Llama 3.3 70B Versatile**, sucesor autorizado por los
  organizadores) con salida estructurada validada, prompts en español, controles de
  fundamentación y fallback extractivo seguro.
- API REST de ciclo de vida de documentos (``POST/GET/DELETE /documents`` con borrado
  suave + purgado de chunks en ChromaDB). Una consola de administración gráfica en
  ``/admin`` proporciona carga, listado con sondeo de estado, refresco y eliminación.
  El módulo backend del ciclo de vida de documentos es distinto de la UI de la consola
  de administración.
- Adaptadores de voz (STT: Groq Whisper Large V3; TTS: Kokoro-82M ``ef_dora``).
- Endpoints REST de turnos de voz HTTP (``POST /calls`` crea una llamada y devuelve un
  saludo WAV codificado en base64; ``POST /calls/{call_id}/turn`` acepta WAV base64,
  transcribe con STT, ejecuta el orquestador, clasifica el escalamiento, sintetiza una
  respuesta TTS y devuelve WAV base64 + transcripción + transcripción del paciente +
  citas + información de escalamiento). El orquestador está conectado con ``RagConfig``
  y ``LlmConfig`` en vivo (desde variables de entorno); los fallbacks seguros
  incorporados manejan los casos en que los proveedores RAG o LLM no estén disponibles.
- Orquestador de conversación (máquina de estados finita: IDLE → GREETING → CONSENT →
  QUESTIONS → CLOSING → ENDED; 6 preguntas estructuradas de seguimiento en español;
  **flujo que prioriza la seguridad**: clasifica las respuestas del paciente antes de
  cualquier llamada a RAG/LLM — RED deriva directamente a ENDED con un mensaje urgente
  de seguridad, ``call_ended=True`` y sin procesamiento adicional; cada respuesta
  no-RED durante QUESTIONS pasa por una **aprobación secundaria por LLM**
  (``backend/llm/approval.py``) que actúa como revisor conservador — puede confirmar la
  clasificación, subir la severidad (nunca degradarla), solicitar una aclaración (máximo
  una por pregunta) o solicitar RAG por duda clínica; fallos, timeouts o salida inválida
  del LLM caen automáticamente a la clasificación determinista; GREEN y primer YELLOW
  confirmados reciben acuses deterministas sin generación RAG+LLM; dos resultados
  YELLOW consecutivos disparan escalamiento con ``should_escalate=True``; las preguntas
  clínicas durante CLOSING se responden con RAG+LLM (con citas) mientras que las
  no-preguntas finalizan la llamada; fallbacks seguros). **Puerta de intención de
  duda determinista** (``_check_doubt_intent``): después de la clasificación
  (solo para respuestas no-RED), el orquestador verifica si la entrada del paciente
  es una pregunta clínica (no un reporte de síntoma) usando marcadores explícitos
  (signos de interrogación, frases compuestas con "como", consultas explícitas);
  la aprobación LLM (``llm_confirm_doubt``) confirma con mayor precisión; fallos
  del LLM preservan las dudas explícitas mediante el fallback determinista; las
  dudas confirmadas ejecutan RAG inline con citas, repiten la misma pregunta sin
  avanzar el índice y no acumulan YELLOW ni generan alertas; RED se detecta
  primero para que texto con forma de pregunta que contiene señales RED
  (ej. ``"es normal que me duela un 9?"``) derive directamente a ENDED sin
  pasar por la puerta de duda; la sexta pregunta (movilidad) permanece en QUESTIONS
  hasta recibir una respuesta válida.
- Motor de escalamiento (clasificador GREEN/YELLOW/RED con lexicones de señales de
  alarma en español, umbrales numéricos, manejo de negaciones; conectado a los endpoints
  de turnos de voz mediante ``EscalationInfo`` en ``TurnResponse``).
- Generador de resúmenes estructurados (resúmenes deterministas en español: datos
  demográficos del paciente, procedimiento, seis dominios de síntomas, decisión,
  próximos pasos).
- Módulo colector de métricas (``InMemoryMetricsCollector`` con modelos tipados,
  estimación de costos, percentiles P50/P95; solo stdlib, thread-safe). Endpoints
  tipados de solo lectura ``GET /metrics/summary``, ``GET /metrics/calls`` y
  ``GET /metrics/calls/{call_id}`` exponen los datos del colector; una vista frontal
  de métricas renderiza los datos. El colector y la API de reporte son
  responsabilidades distintas.
- Interfaz de llamada frontal (HTML/CSS/JS vanilla en ``/`` y ``/call``: selección de
  paciente, captura de micrófono MediaRecorder, llamadas fetch a ``POST /calls`` y
  ``POST /calls/{call_id}/turn``, insignia de estado, historial de conversación, área
  de transcripción, visualización de citas, y reproducción de audio WAV
  para respuestas TTS). El banner de escalamiento solo se muestra para
  escalamientos conclusivos (``should_escalate=True``); las clasificaciones
  por turno no conclusivas (GREEN, primer YELLOW) se registran en el historial
  pero no activan la alerta visual.

**Pendiente (trabajo futuro):**
- Transporte WebSocket/streaming para conversaciones de voz en tiempo real. La
  implementación actual usa HTTP REST con audio WAV codificado en base64.

Queda una decisión abierta: D5 (formato de transporte de audio). La implementación
actual usa HTTP REST con WAV codificado en base64; las alternativas de
streaming/WebSocket quedan como trabajo futuro.

La arquitectura de monolito modular, los límites de módulos, los flujos de datos, el
diseño de persistencia, los contratos de adaptadores y el plan de implementación por
fases están documentados en `docs/ARCHITECTURE.md`.

## Completado

- Documentación del reto revisada: stack-tecnico, requerimientos, rubrica-evaluacion,
  flujo-de-conocimiento, habeas-data, terminos-y-condiciones.
- Dataset sintético inventariado: 4 archivos XLSX (40 pacientes, 160 días de
  trayectoria, 3 991 turnos de conversación), 107 PDFs clínicos en 5 procedimientos.
- Configuración de OpenCode planner, coder y auditor preparada.
- README inicial del repositorio creado.
- Arquitectura definida: catálogo de módulos, flujos de datos, límites de persistencia,
  adaptadores permitidos, plan de implementación por fases, decisiones abiertas
  registradas.
- Paquete ``backend/data/`` implementado: modelos de datos tipados de solo lectura,
  cargadores XLSX, resolvedor de rutas PDF. ``label_ground_truth`` aislado del código
  de ejecución. 58 pruebas de dataset pasan.
- Esqueleto del proyecto FastAPI: `pyproject.toml`, `backend/main.py` con middleware
  CORS, `GET /health`, script de punto de entrada.
- Capa de persistencia: envoltorio ChromaDB ``ChromaStore`` (inicialización de
  colección, inserción de chunks, eliminación de chunks por documento, singleton) y capa
  SQLite con dataclasses tipados y CRUD para las tablas ``calls``,
  ``conversation_turns``, ``summaries``, ``escalation_alerts`` y ``documents``.
  41 pruebas de persistencia pasan.
- Pipeline RAG: ``backend/rag/`` — extraer (pdfplumber), chunking (tamaño fijo con
  solapamiento, 800/150 caracteres), embedding (BGE-M3 vía sentence-transformers),
  almacenar (ChromaDB coseno), ingestión, recuperación (``RetrievalResult`` con citas).
  35 pruebas pasan. D6 (estrategia de chunking) resuelta.
- Adaptador LLM: ``backend/llm/`` — **Groq Llama 3.3 70B Versatile** con salida JSON
  estructurada, prompts en español, validación de seguridad en múltiples capas
  incluyendo detección de inyección de prompts a nivel de entrada (escaneo de patrones
  de jailbreak, límite de longitud, fallback seguro en español) y validación de
  fundamentación post-hoc (comprobaciones de integridad de citas y fundamentación de
  dosis de medicamentos). D1 (modelo de lenguaje: Groq Llama 3.3 sucesor) y D7
  (proveedor único) resueltas.
- Endpoint RAG: ``backend/api/rag.py`` — ``POST /rag/query`` con controles de
  suficiencia de recuperación (cantidad mínima de chunks, umbral de similitud promedio)
  y fallback a ``insufficient_knowledge``. 14 pruebas pasan.
- Fundación del dominio de conversación: ``backend/conversation/`` — máquina de estados
  finita (7 transiciones válidas), ``Message`` / ``History`` / ``PatientContext`` /
  ``CallContext``. 98 pruebas pasan.
- Ciclo de vida de documentos: ``backend/documents/`` — ``Document`` /
  ``DocumentStatus``, ``DocumentService`` (cargar/listar/eliminar).
  ``backend/api/documents.py`` — POST/GET/DELETE /documents,
  ``POST /documents/reconcile``. 34 pruebas rápidas + 10 lentas pasan, incluyendo
  aislamiento de documentos duplicados (eliminar un documento no afecta los chunks de
  otro documento). La detección de duplicados por hash de contenido (SHA-256) hace la
  carga idempotente: contenido idéntico devuelve el registro activo existente (estados
  ``READY`` o ``PROCESSING``). Documentos ``FAILED`` no bloquean la recarga — la
  búsqueda por hash de contenido excluye registros fallidos para permitir
  reintentos de ingesta. La reconciliación detecta y puede limpiar chunks huérfanos
  en ChromaDB. La recuperación filtrada por registro excluye IDs de documentos
  eliminados o no registrados de los resultados de búsqueda.
- Orquestador de conversación: ``backend/conversation/orchestrator.py`` — flujo
  determinista solo texto a través de IDLE → GREETING → CONSENT → QUESTIONS (6
  preguntas estructuradas de seguimiento en español: dolor, fiebre, herida, apetito,
  sueño, movilidad) → CLOSING → ENDED. Integra recuperación RAG + LLM con controles
  de suficiencia de recuperación y fallbacks seguros. **LLM second-approval** integrado
  para cada respuesta no-RED durante QUESTIONS (``backend/llm/approval.py``). 229
  pruebas pasan.
- Motor de decisión de escalamiento: ``backend/decision/`` — ``classify()`` devuelve
  ``EscalationResult`` tipado (GREEN/YELLOW/RED) con lexicones deterministas de señales
  de alarma en español, umbrales numéricos, manejo de negaciones, detección de
  ambigüedad. 178 pruebas pasan. Solo stdlib, solo texto. Las pruebas de aprobación
  secundaria por LLM (78, en ``backend/llm/approval.py``) y las pruebas de integración
  del orquestador con la aprobación (16) se contabilizan por separado
  (ver sección LLM second-approval más abajo).
- Adaptador STT: ``backend/voice/`` — Protocolo ``SttProvider``,
  ``GroqWhisperProvider`` (modelo fijado a ``whisper-large-v3``, idioma ``"es"``),
  asíncrono vía ``groq.AsyncGroq``, mapeo robusto de errores. 56 pruebas pasan.
  D2 resuelta.
- Adaptador TTS: ``backend/voice/tts/`` — Protocolo ``TTSProvider``,
  ``KokoroAdapter`` con carga perezosa de ``kokoro``, voz española ``ef_dora``,
  salida WAV mono PCM 16-bit, WAV silencioso válido para texto vacío. 51 pruebas
  pasan. D3 resuelta.
- Frontal: ``frontend/`` — HTML/CSS/JS vanilla. ``index.html`` (selección de paciente),
  ``call.html`` / ``call.js`` (interfaz de llamada con captura de micrófono
  MediaRecorder, insignia de estado, historial de conversación, área de transcripción,
  visualización de citas y escalamiento, reproducción de audio WAV, resumen en línea
  al finalizar la llamada), ``summary.html`` / ``summary.js`` (página independiente de
  resumen estructurado con citas trazables y severidad codificada por colores),
  ``admin.html`` / ``admin.js`` (consola de administración con carga, listado, sondeo
  de estado, refresco, eliminación), vista frontal de métricas. 10 pruebas pasan.
  ``backend/main.py`` sirve activos mediante ``FileResponse`` y ``StaticFiles``.
- Pruebas de integración del contrato frontal-backend:
  ``tests/test_frontend_integration.py`` — 30 pruebas rápidas que cubren el contrato
  HTTP consumido por ``call.js`` y ``app.js``: formas de respuesta de ``POST /calls``
  y ``POST /calls/{call_id}/turn``, ida y vuelta de audio base64, flujo completo de
  llamada desde GREETING hasta ENDED, forma y momento de la información de
  escalamiento, estructura de citas, contrato de manejo de errores, contrato de
  renderizado de ``patient_transcription`` (semántica de contenido y tipo), preservación
  de la transcripción del agente en cada turno y progresión del estado de llamada con
  orden monótono.
- Instrumentación de métricas: ``backend/metrics/`` — ``InMemoryMetricsCollector``
  (thread-safe, ``threading.Lock``), ``TurnMetrics`` / ``CallMetrics`` /
  ``MetricsSummary``, estimación de costos, percentiles P50/P95. Persistencia de
  métricas en SQLite (tablas ``calls_metrics`` y ``turn_metrics``) con reconstrucción
  automática al inicio para que los endpoints ``GET /metrics/summary``,
  ``GET /metrics/calls`` y ``GET /metrics/calls/{call_id}`` muestren llamadas
   completadas después de reiniciar el servidor. La validación live de reinicio se
   ejecutó en el puerto alternativo ``18001`` y no modificó el frontend. 105 pruebas
   pasan.
  Solo stdlib.
- Módulo de resúmenes: ``backend/summaries/`` — generador determinista de resúmenes en
  español (datos demográficos del paciente, procedimiento, seis dominios de síntomas,
  decisión de escalamiento, próximos pasos). La decisión distingue entre escalamiento
  conclusivo (``ESCALAMIENTO INMEDIATO (ROJO)``, ``ESCALAMIENTO POR ACUMULACION (AMARILLO)``)
  e indicadores YELLOW no conclusivos observados (``INDICADOR DETECTADO (AMARILLO)``).
  44 pruebas pasan. Solo stdlib.
- Endpoints de turnos de voz con persistencia: ``backend/api/calls.py`` —
  ``POST /calls`` crea una llamada y devuelve el saludo del agente como WAV base64;
  ``POST /calls/{call_id}/turn`` transcribe el audio del paciente (STT), ejecuta el
  orquestador (con ``RagConfig`` y ``LlmConfig`` en vivo), clasifica el escalamiento,
  sintetiza una respuesta TTS y devuelve WAV base64 + transcripción + transcripción del
  paciente + citas + información de escalamiento. ``TurnResponse`` incluye
  ``patient_transcription`` (la salida STT del habla del paciente) para visualización
  en el frontal. Los perfiles reales de pacientes se cargan desde el dataset cuando
  están disponibles, con un fallback en el cuerpo de la solicitud para pacientes no
  encontrados en el dataset. ``backend/api/call_store.py`` — ``CallStore`` en memoria
  thread-safe para instancias del orquestador en ejecución. La persistencia de voz está
  completamente integrada con la capa SQLite: la creación de llamada inserta un
  ``CallRecord``; cada turno persiste entradas ``ConversationTurnRecord``; **solo las
  clasificaciones de escalamiento conclusivas** (``should_escalate=True``: RED,
  segundo YELLOW consecutivo, YELLOW ascendido por LLM) persisten
  ``EscalationAlertRecord`` — la primera observación YELLOW, las aclaraciones, las
  dudas RAG y los turnos de duda se registran por turno pero no generan alertas
  persistentes. La persistencia de alertas es **idempotente** mediante IDs
  determinísticos (SHA-256 sobre ``call_id``, severidad y dominio) y
  ``INSERT OR IGNORE``; los reinicios y reintentos no duplican alertas. Las alertas
  conclusivas (RED, segundo YELLOW consecutivo, YELLOW ascendido por LLM) son las
  únicas que persisten. 6 pruebas nuevas de idempotencia de alertas.
  ``backend/summaries/generator.py`` y persiste un ``SummaryRecord``. Las llamadas
  incompletas se rastrean con ``ended_at=None``. SQLite se inicializa al arrancar la
  aplicación en ``backend/main.py``. Las llamadas, turnos, resúmenes y alertas son
  seguros ante reinicios: sobreviven a reinicios del proceso porque los datos están en
  SQLite, no solo en memoria. **No se implementa rehidratación de llamadas activas**
  (el ``CallStore`` en memoria se pierde en cada reinicio; las llamadas en curso deben
  reiniciarse). Los datos históricos (turnos, resúmenes, alertas) persisten
  correctamente. 71 pruebas pasan (9 enfocadas en persistencia).
- Clasificación de escalamiento conectada en los endpoints de turnos de voz: prefiere
  la clasificación del orquestador cuando está disponible (``turn.escalation``), con
  fallback al clasificador a nivel de endpoint con acumulación de YELLOW consecutivos
  en el límite de la API (``_classify_response`` rastrea YELLOW consecutivos por
  llamada mediante ``_call_consecutive_yellows`` y establece
  ``should_escalate=True`` en el segundo YELLOW consecutivo). El contador
  ``_consecutive_yellows`` del orquestador controla las transiciones de estado; el
  contador del límite de la API es la fuente autoritativa para el veredicto de
  escalamiento en la respuesta HTTP en la ruta de fallback. El contador se reinicia
  en GREEN, RED, rechazo de consentimiento y finalización de llamada.
  ``EscalationInfo`` se devuelve en ``TurnResponse``. Probado en regresión en el límite
  de la API para dos escalamientos YELLOW consecutivos que producen
  ``should_escalate=True``, incluyendo ``EscalationAlertRecord`` persistido con
  ``severity=YELLOW`` y la bandera ``escalated=True`` a nivel de llamada.
- ``backend/main.py`` registra todos los routers (``calls_router``, ``rag_router``,
  ``documents_router``, ``metrics_router``, ``summaries_router``) y sirve activos del
  frontal incluyendo ``/``, ``/call``, ``/summary``, ``/admin`` y vistas de métricas.
- Carga automática de ``.env`` mediante ``python-dotenv`` a nivel de módulo en
  ``backend/main.py`` antes de cualquier importación de configuración.
- ``.gitignore`` actualizado con exclusiones estándar de caché/build de Python, caché
  de modelos y datos de ejecución de ChromaDB.
- Decisiones resueltas: D1 (modelo de lenguaje: Groq Llama 3.3 sucesor), D2 (STT:
  Groq Whisper Large V3), D3 (TTS: Kokoro-82M, ef_dora), D4 (framework: FastAPI),
  D6 (chunking: tamaño fijo 800/150), D7 (failover LLM: proveedor único), D8 (carga
  de datos de pacientes: cargar los 40 al inicio), D9 (extracción PDF: pdfplumber).
- Integración de voz en el navegador: ``frontend/call.js`` — captura de micrófono
  MediaRecorder, llamadas fetch a ``POST /calls`` y ``POST /calls/{call_id}/turn``,
  reproducción de audio WAV, renderizado de transcripciones, visualización de
  transcripción del paciente (desde el campo ``patient_transcription``), historial de
  conversación con visualización de citas y escalamiento, y resumen en línea al
  finalizar la llamada con enlace a la página independiente ``/summary``.
- Consola de administración: página ``/admin`` con carga de documentos, listado con
  sondeo de estado, refresco y eliminación; respaldada por la API REST de ciclo de
  vida de documentos pero implementada como un módulo UI distinto.
- Endpoint de resumen de solo lectura: ``backend/api/summaries.py`` —
  ``GET /calls/{call_id}/summary`` devuelve el resumen estructurado persistido
  (datos del paciente, procedimiento, síntomas, decisión, fuentes, próximos pasos)
  desde SQLite. Es estrictamente de solo lectura; no genera resúmenes — esa
  responsabilidad es de ``_persist_call_summary``. Páginas frontales
  ``/summary?call_id=...`` (independiente) y sección en línea en ``/call``
  (después de finalizar la llamada) renderizan el resumen con escape XSS y
  citas trazables. 16 pruebas nuevas pasan (12 API + 4 contrato de integración).
- API de métricas y frontal: endpoints tipados de solo lectura ``GET /metrics/summary``,
  ``GET /metrics/calls`` y ``GET /metrics/calls/{call_id}`` y vista frontal de
  métricas; el módulo colector de métricas es distinto de la API de reporte. Las
  métricas de llamadas completadas sobreviven reinicios del servidor mediante
  persistencia en SQLite y reconstrucción automática en el arranque.
- Refuerzo de seguridad RAG/LLM: controles de suficiencia de recuperación (cantidad
  mínima de chunks, umbral de similitud promedio, configurables mediante variables de
  entorno), detección de inyección de prompts a nivel de entrada (escaneo de patrones
  de jailbreak con fallback seguro en español), validación de fundamentación post-hoc
  (comprobaciones de integridad de citas) y aplicación de fundamentación de dosis de
  medicamentos (menciones de dosis/medicamentos sin fundamento fuerzan
  ``insufficient_knowledge=True`` con fallback seguro, preservando citas válidas),
  aislamiento de eliminación (eliminar un documento preserva los chunks de otros
  documentos), recuperación filtrada por registro (IDs de documentos eliminados y no
  registrados excluidos automáticamente), fallback extractivo seguro de RAG para fallos
  del proveedor (usa el chunk de mayor similitud con metadatos de cita preservados
  cuando el LLM falla o devuelve conocimiento insuficiente a pesar de tener chunks
  recuperados con suficiente similitud). La validación de salida estructurada de Groq
  y los controles de fundamentación siguen vigentes. 63 pruebas LLM y 14 pruebas de
  API RAG, y todas las pruebas existentes de conversación y documentos pasan con las
  nuevas capas de seguridad.

- Auditoría de dependencias: ``openpyxl>=3.0.0``, ``numpy>=1.24.0`` y ``pydantic>=2.0.0``
  declaradas como dependencias base explícitas en ``pyproject.toml``; ``numpy`` eliminado
  del extra ``voice``; ``kokoro>=0.7.0`` copiado al extra ``dev``.

- **LLM second-approval (aprobación secundaria por LLM):** ``backend/llm/approval.py`` —
  ``llm_second_approval()`` actúa como revisor conservador de seguridad después de la
  clasificación determinista para cada respuesta no-RED en QUESTIONS. El LLM puede
  confirmar la clasificación, subir la severidad (GREEN→YELLOW, GREEN→RED, YELLOW→RED;
  nunca degradar), solicitar una aclaración (máximo una por pregunta, se queda en la misma
  pregunta), o solicitar RAG por duda clínica (ejecuta RAG en QUESTIONS y continúa).
  RED nunca pasa por aprobación LLM — el orquestador deriva directamente a ENDED. Fallos,
  timeouts, salida inválida, intentos de degradación e inyección de prompts caen
  automáticamente a la clasificación determinista. La pregunta final (movilidad) procede
  a CLOSING después de respuesta/RAG; la aclaración se queda en la pregunta 6. Los
  controles de inyección de prompts y la política conservadora de escalamiento se
  preservan en todos los caminos. Adicionalmente, ``llm_confirm_doubt()`` proporciona
  confirmación LLM para la puerta de intención de duda, con fallback determinista que
  preserva dudas explícitas. 78 pruebas de aprobación + 16 de integración con el
  orquestador (94 total en approval + orchestrator integration).
- **Puerta de intención de duda determinista:** ``_check_doubt_intent()`` en el
  orquestador verifica si la entrada del paciente es una pregunta clínica (no un
  reporte de síntoma) **después** de la clasificación determinista — solo para
  respuestas no-RED. Combina detección determinista por marcadores explícitos
  (signos de interrogación, frases compuestas con "como", consultas explícitas)
  con confirmación LLM (``llm_confirm_doubt``). Las dudas confirmadas ejecutan RAG
  inline con citas trazables, repiten la misma pregunta sin avanzar el índice y no
  acumulan YELLOW ni generan alertas. RED siempre se detecta primero — texto con
  forma de pregunta que contiene señales RED (ej. ``"es normal que me duela un 9?"``)
  deriva directamente a ENDED sin pasar por la puerta de duda. La sexta pregunta
  (movilidad) permanece en QUESTIONS hasta recibir una respuesta válida. 11 pruebas
  enfocadas en el orquestador cubren apendicectomía, movilidad, RED dentro de duda,
  RED con forma de pregunta, fallo de LLM e intención ambigua.

Totales de pruebas: 1 228 pruebas rápidas (pytest), 26 pruebas lentas (`pytest -m slow`),
1 254 pruebas en total. 12 escenarios de validación live secuencial (puerto alterno 18001)
en ``tests/live_ten_call_validation.py``.

Los conteos por módulo repartidos en esta sección «Completado» son
instantáneas históricas del momento en que se completó cada módulo y pueden
no reflejar adiciones, refactorizaciones o reorganizaciones posteriores.
Los totales agregados arriba (1 228 rápidas + 26 lentas = 1 254 total)
son el conteo autoritativo actual.

- **Finalización manual de llamadas:** endpoint ``POST /calls/{call_id}/end`` que
  finaliza una llamada activa manualmente: obtiene el orquestador (o los turnos
  persistidos si el orquestador ya no está en memoria), genera y persiste el
  resumen estructurado, persiste las alertas de escalamiento conclusivas pendientes
  (con ``INSERT OR IGNORE`` para evitar duplicados), marca la llamada como ENDED
  en SQLite, cierra métricas, elimina la entrada del store y el estado transitorio
  por llamada, y devuelve una respuesta que permite al frontend renderizar el
  resumen directamente. Es idempotente (llamadas repetidas devuelven 200 con el
  resumen existente) y retorna 404 para llamadas inexistentes. El campo
  ``summary_generated`` indica si el resumen fue generado exitosamente; cuando es
  ``False`` los campos del resumen contienen texto descriptivo de la situación.
  El frontend ``call.js`` llama a este endpoint y solo muestra completado/carga
  el resumen tras una respuesta exitosa, manejando errores de red/API sin afirmar
  falsamente la finalización.

  **Dos caminos hacia ENDED:** La finalización automática (cuando el orquestador
  establece ``call_ended=True`` durante el flujo normal de la conversación) también
  genera resumen, persiste alertas, cierra métricas y limpia el estado transitorio —
  exactamente las mismas operaciones que el endpoint manual. El endpoint manual es
  una alternativa para forzar el cierre cuando el llamante no completa el flujo
  normal (por ejemplo, si el usuario cuelga antes de que el orquestador alcance
  ENDED). Ambos caminos comparten la misma lógica de persistencia de resúmenes
  y métricas, y ninguno duplica datos gracias al diseño idempotente de las
  operaciones de escritura. 18 pruebas pasan (13 API en ``test_calls_api.py`` + 5
  contrato de integración frontend en ``test_frontend_integration.py``).

## En progreso

- Formato de transporte de audio — la decisión D5 es de facto HTTP REST con WAV base64
  para los endpoints de turnos de voz; el streaming WebSocket sigue como opción abierta
  para fases futuras.

## Próximos hitos

La implementación sigue el plan de ocho fases en `docs/ARCHITECTURE.md` § Plan de
implementación por fases (fuente única de verdad para hitos y entregables).

Las ocho fases están completas. Los próximos pasos inmediatos son:

1. **Transporte WebSocket/streaming:** Evaluar la adición de streaming/WebSocket en
   tiempo real para conversaciones de voz (trabajo futuro más allá de las fases
   actuales).
2. **Casos límite y pulido:** Refuerzo, manejo de errores y casos límite restantes en
   todos los módulos.

D8 (carga de datos de pacientes) se resolvió durante la Fase 5 (cargar los 40 perfiles
al inicio).

## Decisiones arquitectónicas abiertas

Están registradas en `docs/ARCHITECTURE.md` § Decisiones abiertas. Queda una decisión
abierta:

- **D5** — Formato de transporte de audio. De facto: HTTP REST con WAV codificado en
  base64 para los endpoints de turnos de voz (``POST /calls``,
  ``POST /calls/{call_id}/turn``). El transporte streaming/WebSocket sigue como opción
  futura para la integración con el navegador en tiempo real.

Ocho decisiones han sido resueltas: D1 (LLM: Groq Llama 3.3 sucesor), D2 (STT: Groq
Whisper Large V3), D3 (TTS: Kokoro-82M), D4 (framework: FastAPI), D6 (chunking:
800/150), D7 (failover LLM: proveedor único), D8 (carga de datos de pacientes: cargar
los 40 al inicio), D9 (extracción PDF: pdfplumber).

## Restricciones conocidas

- El modelo de lenguaje debe ser uno de los cuatro permitidos por
  `.challenge-docs/stack-tecnico.md`.
- Los datos clínicos y de pacientes proporcionados son sintéticos y no están validados
  clínicamente.
- No se deben versionar datos reales de pacientes, secretos, grabaciones ni credenciales.
- La configuración final debe ser reproducible en quince minutos o menos.
- El agente conversa en español con regionalismos colombianos.
- Los falsos negativos (no escalar cuando es necesario) son catastróficos; la
  arquitectura impone un escalamiento conservador.

## Reglas de actualización

Actualiza este archivo cuando se complete una fase, se resuelva una decisión abierta,
aparezca un bloqueo o cambie el próximo hito. Mantén los elementos completados concisos
y sin duplicaciones — las entradas detalladas de changelog pertenecen al historial de
commits, no aquí.
