# Tech Sphere Challenge

Agente de voz con IA para seguimiento postoperatorio de pacientes sintéticos
colombianos.

## Objetivo

La solución busca conversar con el paciente en español, consultar conocimiento clínico,
identificar señales de alarma y decidir cuándo escalar el caso a personal capacitado.
También debe permitir actualizar el conocimiento y generar un resumen trazable de cada
llamada.

## Estado

Fase 1 (persistencia) y Fase 2 (RAG) parciales: la aplicación arranca, expone un
endpoint de salud, y el pipeline completo de RAG (extracción → chunking → embedding
BGE-M3 → almacenamiento ChromaDB → recuperación con citas trazables) está implementado
y probado. El shell frontal local (vanilla HTML/CSS/JS) está disponible en `/`
(selección de pacientes), `/call` (interfaz de llamada simulada), `/admin` (consola
de administración para documentos clínicos) y `/metrics` (panel de métricas del
sistema).

### API de Métricas

La aplicación expone endpoints de solo lectura para observar el rendimiento del
sistema.  Las métricas se almacenan **únicamente en memoria** — los datos se pierden
al reiniciar el servidor.

| Método | Ruta | Descripción |
| --- | --- | --- |
| `GET` | `/metrics/summary` | Agregado global de todas las llamadas finalizadas |
| `GET` | `/metrics/calls` | Lista de agregados por llamada (finalizadas) |
| `GET` | `/metrics/calls/{call_id}` | Detalle por llamada con desglose por turno |

**GET /metrics/summary** devuelve: `call_count`, `total_turns`,
`total_input_tokens`, `total_output_tokens`, `total_rag_queries`,
`total_model_calls`, `total_estimated_cost_usd`, `latency_p50_ms`,
`latency_p95_ms`, `tts_p50_ms`, `tts_p95_ms`, `stt_p50_ms`, `stt_p95_ms`,
`llm_p50_ms`, `llm_p95_ms`.  Los campos opcionales (tokens, costos, percentiles de
componentes) se serializan como `null` cuando no hay datos.

**GET /metrics/calls** devuelve `{"calls": [...]}` con `call_id`, `patient_id`,
`turn_count`, `total_latency_ms`, `total_input_tokens`, `total_output_tokens`,
`total_rag_queries`, `model_calls`, `estimated_cost_usd` por llamada.

**GET /metrics/calls/{call_id}** devuelve los mismos campos de agregado más un
array `turns` con `call_id`, `turn_index`, `total_latency_ms`, `model`,
`rag_queries`, `timestamp`, `tts_duration_ms`, `stt_duration_ms`,
`llm_duration_ms`, `input_tokens`, `output_tokens`, `estimated_cost_usd` por turno.

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

Estas pruebas (775) validan dataset, salud del servidor, chunking y extracción de
PDF (con error paths), el adaptador LLM (Llama 3.1 70B Versatile con prompts,
validación y respuestas estructuradas), el endpoint RAG `/rag/query`, la capa de persistencia,
los módulos de voz (STT/TTS), el motor de conversación, el clasificador de
escalamiento, el shell frontal, y los endpoints y colector de métricas.
Todas las llamadas a la API de Gemini y Groq están mockeadas. No descargan el modelo
de embeddings ni procesan PDFs reales.

### Pruebas lentas (requieren BGE-M3 y PDFs)

```bash
pytest -m slow
```

Estas pruebas (16) validan el pipeline completo de RAG: ingestión de PDFs reales
(Apendicectomía en inglés y español), embedding con BGE-M3, recuperación por similitud,
eliminación de chunks y generación de citas trazables. El modelo de embeddings
(~2 GB) se descarga automáticamente en el primer uso.

## Contenido versionado

```text
.
├── backend/               Aplicación Python (FastAPI)
│   ├── __init__.py
│   ├── main.py            Punto de entrada, incluye servido de archivos estáticos
│   ├── api/               Endpoints REST
│   │   ├── rag.py         POST /rag/query (consulta clínica con RAG)
│   │   ├── documents.py   POST/GET/DELETE /documents
│   │   ├── calls.py       POST /calls, POST /calls/{call_id}/turn
│   │   └── metrics.py     GET /metrics/summary, /metrics/calls, /metrics/calls/{call_id}
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
│   ├── metrics/            Instrumentación de métricas
│   │   ├── collector.py    Colector thread-safe en memoria
│   │   ├── models.py       Modelos de datos (TurnMetrics, CallMetrics, MetricsSummary)
│   │   ├── cost.py         Estimación de costos de tokens
│   │   └── percentiles.py  Cómputo de percentiles por interpolación lineal
│   └── persistence/       Acceso a SQLite y ChromaDB
│       ├── chroma.py
│       └── sqlite.py
├── frontend/              Shell frontal (HTML/CSS/JS vanilla)
│   ├── index.html         Página de selección de paciente
│   ├── call.html          Interfaz de llamada simulada
│   ├── metrics.html       Panel de métricas del sistema
│   ├── admin.html         Consola de administración de documentos
│   ├── styles.css         Estilos compartidos
│   ├── data.js            Catálogo compartido de pacientes sintéticos
│   ├── app.js             Lógica de selección de paciente
│   ├── call.js            Lógica de interfaz de llamada
│   ├── admin.js            Lógica de consola de administración
│   └── metrics.js          Lógica del panel de métricas
├── tests/                 Pruebas automatizadas
│   ├── __init__.py
│   ├── test_frontend.py   Pruebas de servido de archivos estáticos (8)
│   ├── test_health.py
│   ├── test_llm.py        Pruebas del adaptador LLM (41)
│   ├── test_rag_api.py          Pruebas del endpoint /rag/query (13)
│   ├── test_admin_console.py    Pruebas de la consola de administración (12)
│   ├── test_dataset/            Pruebas de acceso a datos sintéticos (58)
│   ├── rag/
│   │   ├── conftest.py
│   │   ├── test_chunking.py
│   │   ├── test_extract.py
│   │   └── test_ingestion_retrieval.py
│   ├── decision/          Pruebas del motor de escalamiento
│   ├── conversation/      Pruebas de orquestación
│   └── voice/             Pruebas de adaptadores de voz
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
arquitectura es un monolito modular con backend Python (FastAPI) y dos superficies
de navegador.

La aplicación incluirá dos superficies funcionales:

- Una interfaz de llamada desde el navegador (conversación por voz en español)
  accesible desde `/` y `/call`.
- Una consola de administración en `/admin` para subir, listar y eliminar
  documentos del conocimiento clínico.

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

## Licencia

Consulta [`LICENSE`](LICENSE).
