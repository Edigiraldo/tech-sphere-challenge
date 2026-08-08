"""Unit tests for the LLM adapter — prompt building, validation, and
structured output parsing.

All Gemini API calls are mocked so the tests execute quickly without
network access or an API key.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from backend.llm.adapter import (
    RagAnswer,
    RagCitation,
    _build_prompt,
    _call_gemini,
    _validate_answer,
    generate_rag_answer,
)
from backend.llm.config import LlmConfig


# ---------------------------------------------------------------------------
# LlmConfig
# ---------------------------------------------------------------------------


class TestLlmConfig:
    def test_defaults(self, monkeypatch):
        # LlmConfig reads GOOGLE_API_KEY from os.environ via a
        # default_factory lambda.  Since backend.main now calls
        # load_dotenv() at import time, the env var may already be set.
        # Clear it for this test so we assert the default (empty string).
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        cfg = LlmConfig()
        assert cfg.model_name == "gemini-1.5-flash"
        assert cfg.temperature == 0.2
        assert cfg.max_output_tokens == 1024
        assert cfg.api_key == ""

    def test_temperature_out_of_range_raises(self):
        with pytest.raises(ValueError, match="LLM_TEMPERATURE"):
            LlmConfig(temperature=3.0)

    def test_max_output_tokens_negative_raises(self):
        with pytest.raises(ValueError, match="LLM_MAX_TOKENS"):
            LlmConfig(max_output_tokens=0)

    def test_model_name_is_fixed_to_gemini_flash(self):
        """The model_name field is always ``gemini-1.5-flash`` — it is not
        configurable via environment variable or constructor argument."""
        cfg = LlmConfig()
        assert cfg.model_name == "gemini-1.5-flash"

        # Explicitly passing the valid name is accepted.
        cfg2 = LlmConfig(model_name="gemini-1.5-flash")
        assert cfg2.model_name == "gemini-1.5-flash"

    def test_model_name_not_gemini_flash_raises(self):
        """Any model_name other than exactly 'gemini-1.5-flash' raises
        ValueError at construction time."""
        with pytest.raises(ValueError, match="model_name must be"):
            LlmConfig(model_name="gemini-2.0-flash")

    def test_model_name_empty_string_raises(self):
        """An empty model_name is also rejected."""
        with pytest.raises(ValueError, match="model_name must be"):
            LlmConfig(model_name="")


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_system_prompt_is_in_spanish(self):
        sys_prompt, _ = _build_prompt("¿Cómo estoy?", [])
        assert "Eres un asistente" in sys_prompt
        assert "REGLAS ESTRICTAS" in sys_prompt
        assert "español" in sys_prompt

    def test_user_prompt_contains_query_and_context(self):
        chunks = [
            {
                "chunk_id": "c1",
                "text": "Mantener la herida limpia y seca.",
                "source_filename": "guia.pdf",
                "page_number": 3,
            }
        ]
        _, user_prompt = _build_prompt("cuidado de herida", chunks)
        assert "cuidado de herida" in user_prompt
        assert "c1" in user_prompt
        assert "guia.pdf" in user_prompt
        assert "Mantener la herida" in user_prompt

    def test_user_prompt_includes_multiple_chunks(self):
        chunks = [
            {
                "chunk_id": "c1",
                "text": "Texto uno.",
                "source_filename": "a.pdf",
                "page_number": 1,
            },
            {
                "chunk_id": "c2",
                "text": "Texto dos.",
                "source_filename": "b.pdf",
                "page_number": 2,
            },
        ]
        _, user_prompt = _build_prompt("pregunta", chunks)
        assert "c1" in user_prompt
        assert "c2" in user_prompt
        assert "---" in user_prompt  # separator

    def test_user_prompt_requests_json_format(self):
        _, user_prompt = _build_prompt("q", [])
        assert "JSON" in user_prompt
        assert '"answer"' in user_prompt
        assert '"cited_chunk_ids"' in user_prompt
        assert '"insufficient_knowledge"' in user_prompt


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidateAnswer:
    def test_empty_answer_rejected(self):
        warnings = _validate_answer("", [], set(), False)
        assert len(warnings) >= 1
        assert any("vacía" in w for w in warnings)

    def test_good_spanish_answer_passes(self):
        warnings = _validate_answer(
            "Debe mantener la herida limpia y seca después de la operación.",
            ["c1"],
            {"c1"},
            False,
        )
        assert warnings == []

    def test_answer_without_spanish_markers_warns(self):
        warnings = _validate_answer(
            "Keep the wound clean and dry.",  # English
            ["c1"],
            {"c1"},
            False,
        )
        assert any("español" in w for w in warnings)

    def test_unknown_chunk_ids_warn(self):
        warnings = _validate_answer(
            "Recomendación clínica.",
            ["c99", "c100"],
            {"c1", "c2"},
            False,
        )
        assert any("c99" in w or "c100" in w for w in warnings)
        assert any("chunk_ids" in w for w in warnings)

    def test_no_citations_when_claiming_knowledge_warns(self):
        warnings = _validate_answer(
            "La herida debe mantenerse seca.",
            [],
            {"c1"},
            False,  # not insufficient_knowledge
        )
        assert any("fuente" in w or "cita" in w for w in warnings)

    def test_no_citations_with_insufficient_knowledge_passes(self):
        warnings = _validate_answer(
            "No tengo información suficiente.",
            [],
            {"c1"},
            True,  # insufficient_knowledge
        )
        assert warnings == []

    def test_medication_dose_without_citations_warns(self):
        warnings = _validate_answer(
            "Debe tomar 500 mg de paracetamol cada 8 horas.",
            [],
            {"c1"},
            False,
        )
        assert any("medicamento" in w.lower() or "dosis" in w for w in warnings)

    def test_medication_dose_with_citations_passes(self):
        warnings = _validate_answer(
            "Según la guía, tome 500 mg de paracetamol.",
            ["c1"],
            {"c1"},
            False,
        )
        # The dose-without-citation check fires when cited_chunk_ids is empty;
        # here it is non-empty so no medication warning should fire.
        assert not any("medicamento" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# generate_rag_answer — integration with mocked Gemini
# ---------------------------------------------------------------------------


class TestGenerateRagAnswer:
    @pytest.fixture
    def config(self) -> LlmConfig:
        return LlmConfig(api_key="test-key", temperature=0.0)

    @pytest.fixture
    def context_chunks(self) -> list[dict]:
        return [
            {
                "chunk_id": "c1",
                "document_id": "doc-1",
                "source_filename": "guia_postop.pdf",
                "page_number": 3,
                "text": "Mantener la herida quirúrgica limpia y seca. "
                "Cambiar el apósito diariamente.",
            },
            {
                "chunk_id": "c2",
                "document_id": "doc-1",
                "source_filename": "guia_postop.pdf",
                "page_number": 5,
                "text": "Vigilar signos de infección: enrojecimiento, "
                "hinchazón, secreción purulenta o fiebre.",
            },
        ]

    # -- Empty context -------------------------------------------------------

    def test_empty_context_returns_insufficient_knowledge(self, config):
        result = generate_rag_answer("¿Cómo cuido mi herida?", [], config)
        assert result.insufficient_knowledge is True
        assert "No tengo suficiente información" in result.answer
        assert result.citations == []

    # -- Successful generation -----------------------------------------------

    def test_successful_generation(self, config, context_chunks):
        mock_response_text = json.dumps(
            {
                "answer": (
                    "Debe mantener la herida limpia y seca, y cambiar "
                    "el apósito diariamente según las indicaciones médicas."
                ),
                "cited_chunk_ids": ["c1"],
                "insufficient_knowledge": False,
            }
        )

        mock_response = MagicMock()
        mock_response.text = mock_response_text

        with patch(
            "backend.llm.adapter._call_gemini", return_value=json.loads(mock_response_text)
        ):
            result = generate_rag_answer(
                "¿Cómo cuido mi herida?", context_chunks, config
            )

        assert result.insufficient_knowledge is False
        assert "herida limpia" in result.answer
        assert len(result.citations) == 1
        assert result.citations[0].chunk_id == "c1"
        assert result.citations[0].document_id == "doc-1"
        assert result.citations[0].source_filename == "guia_postop.pdf"
        assert result.model == "gemini-1.5-flash"

    def test_model_flags_insufficient_knowledge(self, config, context_chunks):
        response = json.dumps(
            {
                "answer": (
                    "No tengo suficiente información en las fuentes "
                    "proporcionadas para responder sobre ese tema."
                ),
                "cited_chunk_ids": [],
                "insufficient_knowledge": True,
            }
        )

        with patch(
            "backend.llm.adapter._call_gemini", return_value=json.loads(response)
        ):
            result = generate_rag_answer(
                "¿Qué medicamentos debo tomar?", context_chunks, config
            )

        assert result.insufficient_knowledge is True

    def test_citations_include_excerpts(self, config, context_chunks):
        response = json.dumps(
            {
                "answer": "Respuesta con cita.",
                "cited_chunk_ids": ["c1"],
                "insufficient_knowledge": False,
            }
        )

        with patch(
            "backend.llm.adapter._call_gemini", return_value=json.loads(response)
        ):
            result = generate_rag_answer("pregunta", context_chunks, config)

        assert len(result.citations) == 1
        assert len(result.citations[0].excerpt) > 0

    # -- Debug parameter behaviour -------------------------------------------

    def test_validation_warnings_suppressed_by_default(self, config, context_chunks):
        """When debug=False (default), validation_warnings is empty even
        when the LLM answer has validation concerns."""
        # English answer triggers the Spanish-language heuristic warning
        response = json.dumps(
            {
                "answer": "Keep the wound clean and dry.",
                "cited_chunk_ids": ["c1"],
                "insufficient_knowledge": False,
            }
        )
        with patch(
            "backend.llm.adapter._call_gemini",
            return_value=json.loads(response),
        ):
            result = generate_rag_answer("pregunta", context_chunks, config)

        assert result.insufficient_knowledge is False
        assert result.validation_warnings == []

    def test_validation_warnings_exposed_with_debug(self, config, context_chunks):
        """When debug=True, validation warnings are populated."""
        response = json.dumps(
            {
                "answer": "Keep the wound clean and dry.",
                "cited_chunk_ids": ["c1"],
                "insufficient_knowledge": False,
            }
        )
        with patch(
            "backend.llm.adapter._call_gemini",
            return_value=json.loads(response),
        ):
            result = generate_rag_answer(
                "pregunta", context_chunks, config, debug=True,
            )

        assert result.insufficient_knowledge is False
        # Spanish-language warning should be present
        assert len(result.validation_warnings) >= 1
        assert any("español" in w for w in result.validation_warnings)

    # -- LLM failure fallback ------------------------------------------------

    def test_llm_failure_returns_safe_fallback(self, config, context_chunks):
        with patch(
            "backend.llm.adapter._call_gemini",
            side_effect=RuntimeError("API error"),
        ):
            result = generate_rag_answer(
                "pregunta", context_chunks, config
            )

        assert result.insufficient_knowledge is True
        assert "No puedo procesar" in result.answer
        # Default (debug=False): validation_warnings is empty
        assert result.validation_warnings == []

    def test_llm_failure_debug_exposes_warnings(self, config, context_chunks):
        """When debug=True, validation_warnings includes the LLM error."""
        with patch(
            "backend.llm.adapter._call_gemini",
            side_effect=RuntimeError("API error"),
        ):
            result = generate_rag_answer(
                "pregunta", context_chunks, config, debug=True,
            )

        assert result.insufficient_knowledge is True
        assert len(result.validation_warnings) >= 1
        assert any("API error" in w for w in result.validation_warnings)

    def test_llm_invalid_json_returns_safe_fallback(self, config, context_chunks):
        with patch(
            "backend.llm.adapter._call_gemini",
            side_effect=ValueError("not valid JSON"),
        ):
            result = generate_rag_answer(
                "pregunta", context_chunks, config
            )

        assert result.insufficient_knowledge is True
        # Default (debug=False): validation_warnings is empty
        assert result.validation_warnings == []

    def test_llm_invalid_json_debug_exposes_warnings(self, config, context_chunks):
        """When debug=True, JSON parse errors appear in validation_warnings."""
        with patch(
            "backend.llm.adapter._call_gemini",
            side_effect=ValueError("not valid JSON"),
        ):
            result = generate_rag_answer(
                "pregunta", context_chunks, config, debug=True,
            )

        assert result.insufficient_knowledge is True
        assert len(result.validation_warnings) >= 1

    # -- Hallucinated citations ----------------------------------------------

    def test_model_cites_nonexistent_chunk_ids_suppressed_default(
        self, config, context_chunks,
    ):
        """When the model cites chunk_ids not in context, the fallback
        triggers but validation_warnings is empty when debug=False."""
        response = json.dumps(
            {
                "answer": "Respuesta con cita inventada.",
                "cited_chunk_ids": ["c999"],  # not in context
                "insufficient_knowledge": False,
            }
        )

        with patch(
            "backend.llm.adapter._call_gemini",
            return_value=json.loads(response),
        ):
            result = generate_rag_answer("pregunta", context_chunks, config)

        # No valid citations → forced fallback
        assert result.insufficient_knowledge is True
        # Default (debug=False): no internal warnings leaked
        assert result.validation_warnings == []

    def test_model_cites_nonexistent_chunk_ids_debug(
        self, config, context_chunks,
    ):
        """When debug=True, hallucinated citation warnings are exposed."""
        response = json.dumps(
            {
                "answer": "Respuesta con cita inventada.",
                "cited_chunk_ids": ["c999"],  # not in context
                "insufficient_knowledge": False,
            }
        )

        with patch(
            "backend.llm.adapter._call_gemini",
            return_value=json.loads(response),
        ):
            result = generate_rag_answer(
                "pregunta", context_chunks, config, debug=True,
            )

        # No valid citations → forced fallback
        assert result.insufficient_knowledge is True
        assert any(
            "c999" in w for w in result.validation_warnings
        )

    # -- Markdown code fences stripped ---------------------------------------

    def test_markdown_json_fences_stripped(self, config, context_chunks):
        """_call_gemini strips ```json ... ``` fences before JSON parsing.

        This test mocks the Google genai module so _call_gemini exercises
        the real fence-stripping logic (unlike the old test which mocked
        _call_gemini itself and bypassed the code under test).
        """
        raw_json = json.dumps(
            {
                "answer": "Debe mantener la herida limpia.",
                "cited_chunk_ids": ["c1"],
                "insufficient_knowledge": False,
            }
        )
        fenced = f"```json\n{raw_json}\n```"

        mock_model = MagicMock()
        mock_model.generate_content.return_value.text = fenced

        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            with patch("backend.llm.adapter.genai", mock_genai, create=True):
                result = generate_rag_answer(
                    "¿Cómo cuidar la herida?", context_chunks, config
                )

        assert result.insufficient_knowledge is False
        assert "herida limpia" in result.answer
        assert len(result.citations) == 1
        assert result.citations[0].chunk_id == "c1"

    def test_markdown_fences_no_language_tag(self, config, context_chunks):
        """_call_gemini handles ``` without a language tag."""
        raw_json = json.dumps(
            {
                "answer": "Recomendación postoperatoria.",
                "cited_chunk_ids": ["c1"],
                "insufficient_knowledge": False,
            }
        )
        fenced = f"```\n{raw_json}\n```"

        mock_model = MagicMock()
        mock_model.generate_content.return_value.text = fenced

        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            with patch("backend.llm.adapter.genai", mock_genai, create=True):
                result = generate_rag_answer(
                    "pregunta", context_chunks, config
                )

        assert result.insufficient_knowledge is False


# ---------------------------------------------------------------------------
# _call_gemini unit tests (lazy import, configure, parsing, error wrapping)
# ---------------------------------------------------------------------------


class TestCallGemini:
    """Unit tests for _call_gemini that mock google.generativeai entirely.

    No real API calls are made; the Google genai module is replaced with
    a MagicMock via module-level patching.
    """

    @pytest.fixture
    def config(self) -> LlmConfig:
        return LlmConfig(api_key="test-key", temperature=0.0)

    @pytest.fixture
    def mock_genai(self):
        """Return a fresh MagicMock for the google.generativeai module."""
        mock = MagicMock()
        mock_model = MagicMock()
        mock_model.generate_content.return_value.text = json.dumps(
            {
                "answer": "Una respuesta clínica.",
                "cited_chunk_ids": ["c1"],
                "insufficient_knowledge": False,
            }
        )
        mock.GenerativeModel.return_value = mock_model
        return mock

    # -- Successful call -----------------------------------------------------

    def test_successful_call_parses_json(self, config, mock_genai):
        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            with patch("backend.llm.adapter.genai", mock_genai, create=True):
                result = _call_gemini("sys", "user", config)

        assert result["answer"] == "Una respuesta clínica."
        assert result["cited_chunk_ids"] == ["c1"]
        assert result["insufficient_knowledge"] is False

    # -- genai.configure is called with the API key --------------------------

    def test_genai_configure_called_with_api_key(self, config, mock_genai):
        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            with patch("backend.llm.adapter.genai", mock_genai, create=True):
                _call_gemini("sys", "user", config)

        mock_genai.configure.assert_called_once_with(api_key="test-key")

    # -- GenerativeModel created with correct params -------------------------

    def test_generative_model_created_with_correct_params(self, config, mock_genai):
        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            with patch("backend.llm.adapter.genai", mock_genai, create=True):
                _call_gemini("sys", "user", config)

        mock_genai.GenerativeModel.assert_called_once()
        call_kwargs = mock_genai.GenerativeModel.call_args.kwargs
        assert call_kwargs["model_name"] == "gemini-1.5-flash"
        assert call_kwargs["system_instruction"] == "sys"
        assert call_kwargs["generation_config"]["temperature"] == 0.0
        assert call_kwargs["generation_config"]["max_output_tokens"] == 1024
        assert call_kwargs["generation_config"]["response_mime_type"] == "application/json"

    # -- Missing API key -----------------------------------------------------

    def test_missing_api_key_raises_runtime_error(self, mock_genai):
        config_no_key = LlmConfig(api_key="", temperature=0.0)
        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            with patch("backend.llm.adapter.genai", mock_genai, create=True):
                with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
                    _call_gemini("sys", "user", config_no_key)

    # -- Lazy import: ImportError wrapping -----------------------------------

    def test_import_error_raises_runtime_error(self, config):
        """When google-generativeai is not installed, _call_gemini wraps
        the ImportError in a RuntimeError with a clear message."""
        with patch.dict("sys.modules", {"google.generativeai": None}):
            # Remove the module so the lazy import fails
            with patch("builtins.__import__", side_effect=ImportError("no module")):
                with pytest.raises(RuntimeError, match="google-generativeai"):
                    _call_gemini("sys", "user", config)

    # -- genai.generate_content RuntimeError wrapping ------------------------

    def test_generate_content_error_wraps_as_runtime_error(self, config, mock_genai):
        mock_genai.GenerativeModel.return_value.generate_content.side_effect = (
            Exception("API quota exceeded")
        )
        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            with patch("backend.llm.adapter.genai", mock_genai, create=True):
                with pytest.raises(RuntimeError, match="API quota exceeded"):
                    _call_gemini("sys", "user", config)

    # -- Invalid JSON response -----------------------------------------------

    def test_invalid_json_response_raises_value_error(self, config, mock_genai):
        mock_genai.GenerativeModel.return_value.generate_content.return_value.text = (
            "not valid json at all"
        )
        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            with patch("backend.llm.adapter.genai", mock_genai, create=True):
                with pytest.raises(ValueError, match="no devolvió JSON"):
                    _call_gemini("sys", "user", config)

    # -- Non-dict JSON response ----------------------------------------------

    def test_non_dict_json_response_raises_value_error(self, config, mock_genai):
        mock_genai.GenerativeModel.return_value.generate_content.return_value.text = (
            json.dumps(["not", "a", "dict"])
        )
        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            with patch("backend.llm.adapter.genai", mock_genai, create=True):
                with pytest.raises(ValueError, match="no es un objeto"):
                    _call_gemini("sys", "user", config)

    # -- Response.text is None -----------------------------------------------

    def test_response_text_none_returns_empty_dict_after_parse_failure(self, config, mock_genai):
        """When response.text is None, _call_gemini attempts to parse '' and
        raises ValueError because '' is not valid JSON."""
        mock_genai.GenerativeModel.return_value.generate_content.return_value.text = None
        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            with patch("backend.llm.adapter.genai", mock_genai, create=True):
                with pytest.raises(ValueError, match="no devolvió JSON"):
                    _call_gemini("sys", "user", config)
