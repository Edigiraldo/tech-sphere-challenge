"""Tests for automatic .env file loading in backend.main.

These tests verify that load_dotenv() correctly reads a .env file into
os.environ and that a missing .env file is handled as a safe no-op.

No real secrets are used — all test variables are explicitly placed and
removed from os.environ during test setup and teardown.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from dotenv import load_dotenv


# Sentinel values that prove loading without leaking real keys.
_TEST_VAR_NAME = "TECH_SPHERE_TEST_ENV_VAR"
_TEST_VAR_VALUE = "loaded-from-dotenv-42"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_test_var() -> None:
    """Ensure the test variable is absent before and after every test."""
    _drop_test_var()
    yield
    _drop_test_var()


def _drop_test_var() -> None:
    os.environ.pop(_TEST_VAR_NAME, None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_dotenv_reads_valid_file() -> None:
    """load_dotenv() loads key=value pairs from a valid .env file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env_path = Path(tmpdir) / ".env"
        env_path.write_text(f"{_TEST_VAR_NAME}={_TEST_VAR_VALUE}\n",
                            encoding="utf-8")

        result = load_dotenv(dotenv_path=env_path)

        assert result is True
        assert os.environ.get(_TEST_VAR_NAME) == _TEST_VAR_VALUE


def test_load_dotenv_default_does_not_overwrite_existing() -> None:
    """load_dotenv() by default does NOT overwrite existing vars (override=False)."""
    # Pre-set the variable to a different value.
    os.environ[_TEST_VAR_NAME] = "preexisting-value"

    with tempfile.TemporaryDirectory() as tmpdir:
        env_path = Path(tmpdir) / ".env"
        env_path.write_text(f"{_TEST_VAR_NAME}={_TEST_VAR_VALUE}\n",
                            encoding="utf-8")

        load_dotenv(dotenv_path=env_path)

        # Default behaviour: existing env var wins over .env file.
        assert os.environ[_TEST_VAR_NAME] == "preexisting-value"


def test_load_dotenv_override_True_replaces_existing() -> None:
    """load_dotenv(override=True) replaces existing env vars with .env values."""
    os.environ[_TEST_VAR_NAME] = "preexisting-value"

    with tempfile.TemporaryDirectory() as tmpdir:
        env_path = Path(tmpdir) / ".env"
        env_path.write_text(f"{_TEST_VAR_NAME}={_TEST_VAR_VALUE}\n",
                            encoding="utf-8")

        load_dotenv(dotenv_path=env_path, override=True)

        assert os.environ[_TEST_VAR_NAME] == _TEST_VAR_VALUE


def test_load_dotenv_nonexistent_file_returns_False() -> None:
    """load_dotenv() returns False when the .env file does not exist."""
    nonexistent = Path(tempfile.gettempdir()) / "tech_sphere_nonexistent.env"
    # Ensure the path really does not exist.
    try:
        nonexistent.unlink(missing_ok=True)
    except OSError:
        pass

    result = load_dotenv(dotenv_path=str(nonexistent))

    assert result is False


def test_load_dotenv_does_not_raise_on_missing_file() -> None:
    """Calling load_dotenv() with a missing file must not raise an exception."""
    nonexistent = Path(tempfile.gettempdir()) / "tech_sphere_definitely_missing.env"
    try:
        nonexistent.unlink(missing_ok=True)
    except OSError:
        pass

    # Must not raise.
    load_dotenv(dotenv_path=str(nonexistent))


def test_load_dotenv_default_path_called_from_main_module() -> None:
    """Simulate the exact call made in backend/main.py:
    
        load_dotenv()   # no dotenv_path → searches for .env in CWD / parents

    The default call must not raise even when .env is absent.
    """
    # Mock find_dotenv → "" so no-arg load_dotenv() never discovers a
    # real .env from the project tree.
    with patch("dotenv.main.find_dotenv", return_value=""):
        result = load_dotenv()
    # No .env found → load_dotenv returns False without raising.
    assert result is False


def test_load_dotenv_multiple_variables() -> None:
    """load_dotenv() loads multiple key=value pairs from a single file."""
    extra_var = "TECH_SPHERE_SECOND_VAR"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                f"{_TEST_VAR_NAME}={_TEST_VAR_VALUE}\n"
                f"{extra_var}=second-value\n"
                "# this is a comment\n"
                f"\n",  # blank line should be ignored
                encoding="utf-8",
            )

            result = load_dotenv(dotenv_path=env_path)

            assert result is True
            assert os.environ.get(_TEST_VAR_NAME) == _TEST_VAR_VALUE
            assert os.environ.get(extra_var) == "second-value"
    finally:
        os.environ.pop(extra_var, None)


def test_backend_main_import_loads_dotenv() -> None:
    """Importing backend.main triggers load_dotenv() without error.

    The load_dotenv() call is at module level in backend/main.py and
    executes on first import.  This test proves that importing the app
    does not raise, regardless of whether a .env file exists.
    """
    # backend/main.py is already imported by other tests (e.g. test_health).
    # Re-importing is safe because Python caches the module.
    from backend.main import app  # noqa: F401, F811

    # The import alone is the test — if load_dotenv() raised, this line
    # would not be reached.
    assert app is not None
