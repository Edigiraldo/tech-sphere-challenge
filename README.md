# Tech Sphere Challenge

Agente de voz con IA para seguimiento postoperatorio de pacientes sintéticos
colombianos.

## Objetivo

La solución busca conversar con el paciente en español, consultar conocimiento clínico,
identificar señales de alarma y decidir cuándo escalar el caso a personal capacitado.
También debe permitir actualizar el conocimiento y generar un resumen trazable de cada
llamada.

## Estado

La aplicación implementa el pipeline completo de RAG (extracción → chunking →
embedding BGE-M3 → almacenamiento ChromaDB → recuperación con citas trazables y
controles de suficiencia: umbral de similitud, mínimo de chunks y similitud
promedio), el adaptador LLM (Llama 3.1 70B Versatile vía Groq con validación
estructurada, detección de inyección de prompts, validación de fundamentación
post-hoc y fallback seguro en español), ciclo de vida de documentos
(POST/GET/DELETE /documents con eliminación de chunks indexados y aislamiento
de copias duplicadas), adaptadores de voz (STT Groq Whisper Large V3, TTS
Kokoro-82M), endpoints de turnos de voz HTTP (POST /calls,
POST /calls/{call_id}/turn con audio WAV base64), orquestación de conversación
(máquina de estados, preguntas de seguimiento, RAG+LLM integrados con controles
de suficiencia), motor de escalamiento (clasificación GREEN/YELLOW/RED con
lexicones en español), módulo de resúmenes estructurados, colector de métricas,
y capa de persistencia (SQLite + ChromaDB).

El frontal (vanilla HTML/CSS/JS) está disponible en `/` y `/call` con
selector de pacientes, interfaz de llamada con integración real de micrófono
(MediaRecorder), llamadas `fetch()` a los endpoints de voz, historial de
conversación, área de transcripción, visualización de citas trazables e
información de escalamiento, y reproducción de audio WAV del navegador.

El endpoint ``POST /calls/{call_id}/turn`` implementa el pipeline de voz completo:
transcripción STT → orquestador → clasificación de escalamiento → síntesis TTS,
devolviendo audio WAV base64, transcripción del agente, citas trazables e
información de escalamiento en cada turno. La API REST de documentos
(``POST/GET/DELETE /documents``) gestiona el ciclo de vida completo incluyendo
purgado de chunks en ChromaDB. El módulo colector de métricas
(``InMemoryMetricsCollector``) está expuesto mediante un endpoint API de solo
lectura tipado (``GET /metrics/summary``, ``GET /metrics/calls`` y
``GET /metrics/calls/{call_id}``) y una vista frontal de métricas. El colector
y la API de reporte son módulos independientes.

**Pendiente (futuro):**
- Transporte WebSocket/streaming para conversación de voz en tiempo real.

## Requisitos

- Python 3.11 o superior
- pip (incluido con Python)

## Instalación y ejecución

```bash
# 1. Crear entorno virtual
python -m venv .venv

# 2. Activar entorno (Windows PowerShell)
.venv\Scripts\Activate.ps1
# -- o en Linux/macOS --
# source .venv/bin/activate

# 3. Instalar el proyecto en modo editable
pip install -e ".[dev]"

# 4. Ejecutar la aplicación
python -m backend.main
# -- o con el script registrado --
# tech-sphere
```

La aplicación queda disponible en `http://127.0.0.1:8000`.

Verificar que está corriendo:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```

Documentación interactiva de la API (Swagger UI) en `http://127.0.0.1:8000/docs`.

## Variables de entorno

La aplicación carga automáticamente las variables de entorno desde un archivo
`.env` ubicado en la raíz del proyecto (usando `python-dotenv`). La carga
ocurre antes de leer cualquier configuración, por lo que los valores definidos
en `.env` están disponibles desde el inicio.

Si el archivo `.env` no existe, la aplicación continúa normalmente usando las
variables ya presentes en el entorno del sistema.

### Crear el archivo `.env`

Copia este contenido en un archivo nuevo llamado `.env` en la raíz del proyecto:

```ini
# Tech Sphere Challenge — configuración de entorno local
GROQ_API_KEY=tu-clave-de-api-aqui

# Opcionales (con sus valores por defecto)
# LLM_TEMPERATURE=0.2
# LLM_MAX_TOKENS=1024
```

### Requerida

| Variable | Descripción |
| --- | --- |
| `GROQ_API_KEY` | Clave de API de Groq Cloud para el modelo Llama 3.1 70B Versatile |

### Opcionales

| Variable | Valor por defecto | Descripción |
| --- | --- | --- |
| `LLM_TEMPERATURE` | `0.2` | Temperatura de muestreo (0–2). Valores bajos favorecen respuestas determinísticas y basadas en fuentes |
| `LLM_MAX_TOKENS` | `1024` | Máximo de tokens en la respuesta generada |
| `RAG_SIMILARITY_THRESHOLD` | `0.25` | Similitud coseno mínima para incluir un chunk en los resultados |
| `RAG_MIN_CHUNKS` | `2` | Mínimo de chunks requeridos antes de invocar el LLM |
| `RAG_MIN_AVG_SIMILARITY` | `0.30` | Similitud promedio mínima entre todos los chunks recuperados |

El modelo de lenguaje está fijado a **Llama 3.1 70B Versatile** — el único modelo
integrado en esta fase. No se puede seleccionar otro modelo mediante variables de
entorno.

### ⚠️ Nunca incluir claves en el repositorio

El archivo `.env` está incluido en `.gitignore` y **no debe ser versionado**.
Nunca subas claves de API, secretos ni credenciales al repositorio. La sección
[Privacidad y seguridad](#privacidad-y-seguridad) detalla las políticas del
proyecto.

## Pruebas

### Pruebas rápidas (sin modelo ni PDF)

```bash
pytest
```

Estas pruebas (948) validan dataset, salud del servidor, chunking y extracción de
PDF (con error paths), el adaptador LLM (Llama 3.1 70B Versatile con prompts,
validación, respuestas estructuradas, detección de inyección de prompts y
validación de fundamentación), el endpoint RAG `/rag/query` con controles de
suficiencia, la capa de persistencia, los módulos de voz (STT/TTS), el motor de
conversación, el clasificador de escalamiento, los endpoints de turnos de voz,
el módulo de resúmenes, el colector de métricas y el frontal del navegador.
Todas las llamadas a las APIs de Groq están mockeadas. No descargan el modelo
de embeddings ni procesan PDFs reales.

### Pruebas lentas (requieren BGE-M3 y PDFs)

```bash
pytest -m slow
```

Estas pruebas (24) validan el pipeline completo de RAG: ingestión de PDFs reales
(Apendicectomía en inglés y español), embedding con BGE-M3, recuperación por similitud,
controles de suficiencia, eliminación de chunks, generación de citas trazables,
verificación de nombres de archivo reales en disco, y aislamiento de documentos
duplicados. El modelo de embeddings (~2 GB) se descarga automáticamente en el
primer uso.

## Contenido versionado

```text
.
├── backend/               Aplicación Python (FastAPI)
│   ├── __init__.py
│   ├── main.py            Punto de entrada, incluye servido de archivos estáticos
│   ├── api/               Endpoints REST
│   │   ├── rag.py         POST /rag/query (consulta clínica con RAG)
│   │   ├── documents.py   POST/GET/DELETE /documents
│   │   ├── calls.py       POST /calls, POST /calls/{id}/turn (voz)
│   │   └── call_store.py  Almacenamiento en memoria de llamadas
│   ├── llm/               Adaptador de modelo de lenguaje
│   │   ├── __init__.py
│   │   ├── config.py      Configuración fija (Llama 3.1 70B)
│   │   └── adapter.py     Generación validada con citas trazables
│   ├── data/              Acceso tipado de solo lectura a los datos sintéticos
│   ├── rag/               RAG pipeline (ingestión, recuperación)
│   │   ├── config.py
│   │   ├── chunking.py
│   │   ├── embeddings.py
│   │   ├── extract.py
│   │   ├── ingestion.py
│   │   ├── retrieval.py
│   │   └── store.py
│   ├── documents/         Ciclo de vida de documentos
│   ├── decision/          Clasificación de escalamiento
│   ├── conversation/      Orquestación de conversación
│   ├── voice/             Adaptadores STT y TTS
│   ├── summaries/         Generación de resúmenes estructurados
│   ├── metrics/           Instrumentación de latencia, tokens y costo
│   └── persistence/       Acceso a SQLite y ChromaDB
│       ├── chroma.py
│       └── sqlite.py
├── frontend/              Frontal de navegador (HTML/CSS/JS vanilla)
│   ├── index.html         Página de selección de paciente
│   ├── call.html          Interfaz de llamada con MediaRecorder e integración API de voz
│   ├── admin.html         Consola de administración de documentos
│   ├── metrics.html       Vista frontal de métricas
│   ├── styles.css         Estilos compartidos
│   ├── data.js            Catálogo compartido de pacientes sintéticos
│   ├── app.js             Lógica de selección de paciente
│   ├── call.js            Lógica de UI de llamada con captura de micrófono y API
│   ├── admin.js           Lógica de administración con sondeo de estado
│   └── metrics.js         Lógica de visualización de métricas
├── tests/                 Pruebas automatizadas
│   ├── __init__.py
│   ├── test_frontend.py   Pruebas de servido de archivos estáticos (8)
│   ├── test_health.py     Prueba del endpoint /health (1)
│   ├── test_llm.py        Pruebas del adaptador LLM (63)
│   ├── test_rag_api.py    Pruebas del endpoint /rag/query (14)
│   ├── test_documents.py  Pruebas del ciclo de vida de documentos (16)
│   ├── test_calls_api.py  Pruebas de endpoints de turnos de voz (41)
│   ├── test_persistence_extended.py  Pruebas de capa SQLite extendida (41)
│   ├── test_summaries.py  Pruebas del generador de resúmenes (44)
│   ├── test_voice.py      Pruebas del adaptador STT (56)
│   ├── test_env_loading.py  Pruebas de carga de .env (8)
│   ├── test_dataset/      Pruebas de acceso a datos sintéticos (58)
│   ├── rag/               Pruebas del pipeline RAG (24)
│   │   ├── conftest.py
│   │   ├── test_chunking.py
│   │   ├── test_extract.py
│   │   └── test_ingestion_retrieval.py
│   ├── decision/          Pruebas del motor de escalamiento (125)
│   ├── conversation/      Pruebas de orquestación (193)
│   ├── metrics/           Pruebas del colector de métricas (82)
│   └── voice/             Pruebas del adaptador TTS (51)
├── docs/                  Documentación del proyecto
│   ├── ARCHITECTURE.md
│   ├── PROJECT.md
│   └── STATUS.md
├── dataset/               Datos sintéticos y documentos clínicos de referencia
├── .challenge-docs/       Documentación disponible del reto
├── pyproject.toml         Declaración del proyecto y dependencias
├── LICENSE                Licencia del repositorio
└── README.md              Información del proyecto
```

Los archivos de configuración del flujo de trabajo del proyecto (`AGENTS.md`,
`opencode.json` y `.opencode/`) son parte del repositorio versionado y se mantienen
para coordinar el desarrollo con el pipeline planificador/codificador/auditor.

## Documentación

La documentación del proyecto incluye:

- [Arquitectura técnica completa](docs/ARCHITECTURE.md)
- [Estado del proyecto](docs/STATUS.md)

La documentación del reto se encuentra en [`.challenge-docs/`](.challenge-docs/):

- [README del reto](.challenge-docs/README.md)
- [Rúbrica de evaluación](.challenge-docs/rubrica-evaluacion.md)
- [Stack técnico y modelos permitidos](.challenge-docs/stack-tecnico.md)
- [Flujo de conocimiento](.challenge-docs/flujo-de-conocimiento.md)
- [Ficha técnica](.challenge-docs/ficha-tecnica.md)
- [Política de habeas data](.challenge-docs/politica-de-habeas-data.md)
- [Términos y condiciones](.challenge-docs/terminos-y-condiciones.md)

## Dataset

El directorio [`dataset/`](dataset/) contiene material sintético del reto:

- Conversaciones postoperatorias.
- Trayectorias clínicas por caso.
- Perfiles clínicos y demográficos sintéticos.
- Documentos PDF para el conocimiento del RAG.

Los datos no representan pacientes reales y no deben utilizarse con fines clínicos,
diagnósticos o asistenciales. Los documentos PDF conservan los derechos de sus autores
y editores y se incluyen como material de referencia del reto.

## Desarrollo

El desarrollo sigue el plan de fases documentado en
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#phased-implementation-plan). La
arquitectura es un monolito modular con backend Python (FastAPI), un frontal
de navegador y una API REST de administración.

La aplicación expone:

- Una interfaz de llamada desde el navegador en `/` y `/call` con selección
  de pacientes, integración de micrófono real (MediaRecorder), envío de audio
  a los endpoints ``POST /calls`` y ``POST /calls/{call_id}/turn``, historial
  de conversación, visualización de transcripciones, citas trazables e
  información de escalamiento, y reproducción de audio WAV.
- Una consola de administración gráfica en ``/admin`` para subir, listar con
  sondeo de estado, refrescar y eliminar documentos del conocimiento con
  purgado de chunks indexados. El ciclo de vida de documentos en el backend
  (``POST/GET/DELETE /documents``) es un módulo independiente de la consola
  de administración.
- Endpoints de métricas de solo lectura tipados (``GET /metrics/summary``,
  ``GET /metrics/calls`` y ``GET /metrics/calls/{call_id}``) y una
  vista frontal de métricas. El colector de métricas (``InMemoryMetricsCollector``)
  es un módulo independiente del endpoint de reporte.

El flujo esperado es:

```text
Voz del paciente
  -> transcripción
  -> recuperación de fuentes clínicas (RAG)
  -> respuesta segura y citada
  -> decisión de escalamiento
  -> voz y resumen de la llamada
```

## Privacidad y seguridad

Aunque los datos entregados son sintéticos, la aplicación debe tratar la información
clínica como sensible. No deben subirse datos reales, grabaciones reales, credenciales,
claves API ni archivos `.env` al repositorio.

### Defensa contra inyección de prompts

El sistema implementa múltiples capas de defensa:

- **Separación de roles**: las instrucciones del sistema y la entrada del paciente
  se envían en roles de mensaje separados (system/user) en la API de Groq.
- **Detección a nivel de entrada**: antes de cualquier llamada al LLM, la consulta
  se escanea en busca de patrones conocidos de jailbreak (cambio de rol, extracción
  de prompt del sistema, inyección de delimitadores, etiquetas `[INST]`, comandos
  de ejecución, etc.). Cuando se detecta un patrón, se devuelve un fallback seguro
  en español sin invocar el modelo.
- **Salida estructurada**: el esquema JSON obligatorio impide que el LLM devuelva
  texto libre fuera del formato esperado.
- **Validación de fundamentación post-hoc**: después de la generación, un
  validador secundario verifica que las citas referencien chunks existentes con
  texto no vacío, y que las menciones de dosis de medicamentos estén respaldadas
  por los extractos citados.

### Controles de suficiencia RAG

Antes de invocar el LLM, la recuperación RAG aplica controles de calidad
configurables:

- ``RAG_SIMILARITY_THRESHOLD`` (defecto: 0.25) — umbral mínimo de similitud coseno.
- ``RAG_MIN_CHUNKS`` (defecto: 2) — mínimo de chunks requeridos.
- ``RAG_MIN_AVG_SIMILARITY`` (defecto: 0.30) — similitud promedio mínima.

Si algún control falla, el sistema devuelve ``insufficient_knowledge`` sin llamar
al LLM.

## Licencia

Consulta [`LICENSE`](LICENSE).
