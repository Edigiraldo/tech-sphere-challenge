# Tech Sphere Challenge

Agente de voz con IA para seguimiento postoperatorio de pacientes sintéticos
colombianos.

## Objetivo

La solución busca conversar con el paciente en español, consultar conocimiento clínico,
identificar señales de alarma y decidir cuándo escalar el caso a personal capacitado.
También debe permitir actualizar el conocimiento y generar un resumen trazable de cada
llamada.

## Estado

Fase 1 en progreso. El paquete `backend/data/` proporciona acceso tipado de solo
lectura a los datos sintéticos y la aplicación arranca con un endpoint de salud.
La persistencia, RAG, voz, conversación y las superficies de navegador se incorporan
por fases.

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

## Pruebas

```bash
pytest
```

## Contenido versionado

```text
.
├── backend/           Backend de la aplicación (Python)
│   ├── main.py
│   └── data/          Acceso tipado de solo lectura a los datos sintéticos
├── tests/             Pruebas automatizadas
├── .challenge-docs/   Documentación disponible del reto
├── backend/           Aplicación Python (FastAPI)
│   └── rag/            Ingestión y recuperación de conocimiento clínico
├── docs/              Documentación del proyecto
│   ├── ARCHITECTURE.md
│   ├── PROJECT.md
│   └── STATUS.md
├── dataset/           Datos sintéticos y documentos clínicos de referencia
├── pyproject.toml     Declaración del proyecto y dependencias
├── LICENSE            Licencia del repositorio
└── README.md          Información del proyecto
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

- Una interfaz de llamada desde el navegador (conversación por voz en español).
- Una consola de administración para subir, listar y eliminar documentos del
  conocimiento.

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
