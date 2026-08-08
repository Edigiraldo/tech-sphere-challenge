"""Tests for backend.data.pdfs — PDF reference resolver."""

from pathlib import Path

from backend.data.pdfs import (
    count_pdfs,
    get_pdfs_by_procedure,
    get_procedure_names,
    list_pdfs,
    resolve_pdf_path,
)
from backend.data.models import PDFReference


def _on_disk_pdf_count() -> int:
    """Count PDF files directly on disk (bypassing pdfs.py) as a cross-check."""
    import backend.data.pdfs as _pdfs
    root = _pdfs._textos_root()
    total = 0
    for child in root.rglob("*.pdf"):
        if child.is_file():
            total += 1
    return total


class TestCountPDFs:
    def test_count_positive(self):
        """PDF count must be at least 90 and match the on-disk glob."""
        count = count_pdfs()
        on_disk = _on_disk_pdf_count()
        assert count >= 90, f"Expected >=90 PDFs, got {count}"
        assert count == on_disk, (
            f"list_pdfs reports {count} but filesystem glob found {on_disk}"
        )

    def test_count_matches_list(self):
        assert count_pdfs() == len(list_pdfs())


class TestListPDFs:
    def test_returns_pdf_references(self):
        refs = list_pdfs()
        assert isinstance(refs, list)
        assert len(refs) > 0
        for ref in refs:
            assert isinstance(ref, PDFReference)
            assert ref.filename.endswith(".pdf")
            assert isinstance(ref.path, Path)
            assert ref.procedure in (
                "appendicitis",
                "breast_cancer",
                "cholecystitis",
                "colorectal_cancer",
                "total_joint_replacement",
            )

    def test_all_paths_exist(self):
        for ref in list_pdfs():
            assert ref.path.is_file(), f"Missing: {ref.path}"


class TestGetPDFsByProcedure:
    def test_five_procedures(self):
        grouped = get_pdfs_by_procedure()
        assert len(grouped) == 5

    def test_per_procedure_counts_reasonable(self):
        """Each procedure should have a non-trivial number of PDFs."""
        grouped = get_pdfs_by_procedure()
        for proc, refs in grouped.items():
            assert len(refs) >= 10, (
                f"{proc}: expected >=10 PDFs, got {len(refs)}"
            )
            assert len(refs) <= 40, (
                f"{proc}: expected <=40 PDFs, got {len(refs)}"
            )

    def test_total_matches_on_disk_count(self):
        grouped = get_pdfs_by_procedure()
        total = sum(len(v) for v in grouped.values())
        on_disk = _on_disk_pdf_count()
        assert total == on_disk, (
            f"Grouped total {total} != filesystem glob {on_disk}"
        )


class TestResolvePDFPath:
    def test_valid_path(self):
        path = resolve_pdf_path("appendicitis", "Apendicitis.pdf")
        assert path is not None
        assert path.is_file()
        assert path.name == "Apendicitis.pdf"

    def test_missing_file(self):
        path = resolve_pdf_path("appendicitis", "nonexistent.pdf")
        assert path is None

    def test_missing_procedure(self):
        path = resolve_pdf_path("unknown", "file.pdf")
        assert path is None

    def test_case_insensitive_procedure(self):
        """Should handle case variation in procedure name."""
        # The actual dir is 'Appendicitis', modulo key is 'appendicitis'
        path = resolve_pdf_path("APPENDICITIS", "Apendicitis.pdf")
        assert path is not None
        assert path.is_file()

    def test_space_vs_underscore(self):
        """colorectal_cancer dir has a space."""
        # The dir is 'colorectal cancer', modulo key is 'colorectal_cancer'
        grouped = get_pdfs_by_procedure()
        first = grouped["colorectal_cancer"][0]
        # Resolve by modulo key
        path = resolve_pdf_path("colorectal_cancer", first.filename)
        assert path is not None
        assert path.is_file()


class TestGetProcedureNames:
    def test_five_names(self):
        names = get_procedure_names()
        assert len(names) == 5
        expected = [
            "appendicitis",
            "breast_cancer",
            "cholecystitis",
            "colorectal_cancer",
            "total_joint_replacement",
        ]
        assert names == expected
