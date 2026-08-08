"""Normalized read-only data models for the synthetic Colombian patient dataset.

These models represent the four XLSX data sources as typed Python dataclasses.
No model here contains business logic or runtime behaviour — they are pure data
containers loaded once at startup.

``label_ground_truth`` is an *optional* field and is only populated when
conversations are loaded through the evaluation-only path.  Runtime code must
never depend on it.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import List as ListType
from typing import Optional


# ---------------------------------------------------------------------------
# Patient
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Patient:
    """Merged view of a single synthetic patient.

    Combines columns from ``perfiles_clinicos_pacientes_silver_contest.xlsx``
    and ``perfiles_pacientes_co.xlsx`` joined on ``paciente_id``.

    All fields are read-only after construction.
    """

    paciente_id: str
    # -- clinical profile --
    bundle_id: str
    synthea_runtime: str
    modulo_synthea: str
    procedimiento: str
    fecha_cirugia: datetime.date
    edad: int
    genero: str
    comorbilidades: ListType[str]
    complicacion_encounter: bool
    # -- demographic profile --
    nombre_completo: str
    direccion: str
    ciudad: str
    departamento: str
    documento_cc: str
    eps: str
    source_country: str
    adapted_country: str
    adaptation_fields: ListType[str]


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Trajectory:
    """A single post-operative day observation from the clinical trajectory.

    Source: ``trayectorias_postop_silver.xlsx``.
    """

    trayectoria_id: str
    paciente_id: str
    dia_postop: int
    arquetipo_trayectoria: str
    dolor_nrs: int
    fiebre_c: float
    movilidad: str
    herida: str
    apetito: str
    sueno: str
    seed: int


# ---------------------------------------------------------------------------
# Conversation / Turn
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """A single turn within a synthetic postoperative conversation.

    Source: ``dataset_final.xlsx`` (one row == one turn).

    ``label_ground_truth`` is **optional** — it is ``None`` when loaded via the
    default runtime path and only populated through
    ``load_conversations_for_evaluation()``.
    """

    dialogo_id: str
    caso_id: str
    paciente_id: str
    dia_postop: int
    turno_idx: int
    hablante: str               # "agente" or "paciente"
    texto: str
    label_ground_truth: Optional[str] = None
    estilo_paciente: str = ""
    modelo_paciente: str = ""
    modelo_agente: str = ""
    capa: str = ""


@dataclass(frozen=True, slots=True)
class Conversation:
    """A complete multi-turn conversation grouped by ``caso_id``.

    ``turns`` are ordered by ``turno_idx`` ascending.
    """

    caso_id: str
    paciente_id: str
    dia_postop: int
    turns: ListType[ConversationTurn] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PDF reference
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PDFReference:
    """Reference to a single clinical PDF in the dataset.

    Does **not** read the file — it only records the path and procedure
    association so that downstream RAG ingestion can locate the document.
    """

    procedure: str          # normalized procedure key (modulo_synthea value)
    filename: str           # file name on disk
    path: Path              # absolute path to the PDF file
