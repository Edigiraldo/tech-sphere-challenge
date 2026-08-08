"""Shared fixtures for RAG tests.

Provides a RagConfig instance, paths to the two Appendicitis test PDFs, and
an autouse fixture that resets the ChromaStore singleton between tests.
"""

import os
import tempfile
from pathlib import Path

import pytest

import backend.persistence.chroma as chroma_mod
from backend.rag.config import RagConfig


@pytest.fixture(autouse=True)
def _reset_chroma_singleton():
    """Reset the module-level ChromaStore singleton before and after each test.

    Without this, a ``ChromaStore`` initialised with one test's temp directory
    or collection name leaks into subsequent tests that rely on a different
    store instance.
    """
    original = chroma_mod._store
    chroma_mod._store = None
    yield
    chroma_mod._store = original


@pytest.fixture
def rag_config():
    """Return a RagConfig with ChromaDB pointed at a temp directory."""
    return RagConfig(
        chroma_persist_dir=Path(tempfile.mkdtemp(prefix="chroma_test_")),
        chunk_size=400,
        chunk_overlap=80,
        retrieval_top_k=3,
        collection_name="test_clinical_knowledge",
    )


@pytest.fixture
def test_pdf_dir():
    """Return the path to the Appendicitis test PDF directory."""
    return Path(__file__).parents[2] / "dataset" / "textos" / "Appendicitis"


@pytest.fixture
def post_op_en_pdf(test_pdf_dir: Path) -> Path:
    """Path to the English post-operative instructions PDF."""
    return test_pdf_dir / "POST OPERATIVE INSTRUCTIONS FOR APPENDECTOMY .pdf"


@pytest.fixture
def plan_cuidado_es_pdf(test_pdf_dir: Path) -> Path:
    """Path to the Spanish home care plan PDF."""
    return test_pdf_dir / "PLAN DE CUIDADO EN CASA DE PACIENTE EN POSTOPERATORIO DE APENDICECTOMÍA.pdf"


@pytest.fixture
def sample_spanish_text():
    """A short Spanish clinical text for chunking tests."""
    return (
        "Después de una apendicectomía, es importante mantener la herida "
        "quirúrgica limpia y seca. Se recomienda cambiar el apósito "
        "diariamente y vigilar signos de infección como enrojecimiento, "
        "hinchazón, calor local o secreción purulenta. "
        "El paciente debe guardar reposo relativo durante los primeros "
        "siete días y evitar levantar objetos pesados. "
        "La dieta debe ser blanda y rica en fibra para evitar el "
        "estreñimiento. Es fundamental mantener una adecuada hidratación. "
        "Ante la presencia de fiebre superior a 38°C, dolor abdominal "
        "intenso, vómitos persistentes o sangrado por la herida, debe "
        "acudir inmediatamente al servicio de urgencias."
    )
