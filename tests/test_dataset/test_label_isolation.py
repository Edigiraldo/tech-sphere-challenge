"""Verify ``label_ground_truth`` isolation from runtime modules.

The evaluation-only function ``load_conversations_for_evaluation`` must never
be imported by production modules.  This test statically checks for such
forbidden imports.

We use module boundaries rather than runtime import guards to avoid breaking
normal pytest imports.  The test scans candidate runtime directories for any
reference to the evaluation-only function name.
"""

from pathlib import Path

import pytest

# Runtime module directories that must NEVER depend on ground-truth labels.
_RUNTIME_DIRS = [
    "conversation",
    "llm",
    "decision",
    "summaries",
    "rag",
    "documents",
    "voice",
    "metrics",
    "api",
    "persistence",
]

# The forbidden function name.
_FORBIDDEN = "load_conversations_for_evaluation"


class TestLabelGroundTruthIsolation:
    @pytest.mark.parametrize("module_dir", _RUNTIME_DIRS)
    def test_runtime_module_does_not_import_eval_loader(self, module_dir: str):
        """No runtime module may import or call the evaluation-only loader."""
        import ast

        backend_root = Path(__file__).resolve().parent.parent.parent / "backend"
        target_dir = backend_root / module_dir

        if not target_dir.is_dir():
            # Module directory does not exist yet — that is fine.
            return

        violations: list[tuple[str, int]] = []
        for py_file in target_dir.rglob("*.py"):
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                # Catch ``from backend.data.loader import load_conversations_for_evaluation``
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name == _FORBIDDEN:
                            violations.append(
                                (str(py_file.relative_to(backend_root)), node.lineno)
                            )
                # Catch ``backend.data.loader.load_conversations_for_evaluation``
                if isinstance(node, ast.Attribute) and node.attr == _FORBIDDEN:
                    violations.append(
                        (str(py_file.relative_to(backend_root)), node.lineno)
                    )

        assert not violations, (
            f"Runtime module(s) reference forbidden function "
            f"'{_FORBIDDEN}':\n"
            + "\n".join(f"  {f}:{n}" for f, n in violations[:20])
            + ("\n  ..." if len(violations) > 20 else "")
        )

    def test_public_api_excludes_eval_loader(self):
        """The public __init__.py must not re-export the eval-only loader
        in __all__ or in executable import statements.

        Docstrings and comments may reference the function name for
        documentation purposes — that is intentional and safe.
        """
        import ast

        init_path = (
            Path(__file__).resolve().parent.parent.parent
            / "backend" / "data" / "__init__.py"
        )
        tree = ast.parse(init_path.read_text(encoding="utf-8"))

        # 1. __all__ must not include the eval-only function
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id == "__all__"
                        and isinstance(node.value, ast.List)
                    ):
                        all_names = [
                            elt.value
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant)
                        ]
                        assert _FORBIDDEN not in all_names, (
                            f"__all__ includes forbidden function "
                            f"'{_FORBIDDEN}'"
                        )
            # 2. No ImportFrom should bring in the eval-only function
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    assert alias.name != _FORBIDDEN, (
                        f"__init__.py imports forbidden function "
                        f"'{_FORBIDDEN}'"
                    )

    def test_runtime_loader_leaves_labels_null(self):
        """Confirm that the default loader never populates labels."""
        from backend.data.loader import load_conversations

        convs = load_conversations()
        for c in convs:
            for t in c.turns:
                assert t.label_ground_truth is None, (
                    "label_ground_truth MUST be None in runtime-safe loader"
                )
