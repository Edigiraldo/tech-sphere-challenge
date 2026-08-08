"""Unit tests for the LLM adapter — prompt building, validation, and
structured output parsing.

All Groq API calls are mocked so the tests execute quickly without
network access or an API key.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from backend.llm.adapter import (
    RagAnswer,
    RagCitation,
    _build_prompt,
    _call_groq,
    _validate_answer,
    generate_rag_answer,
)
from backend.llm.config import LlmConfig


# ---------------------------------------------------------------------------
# LlmConfig
# ---------------------------------------------------------------------------


class TestLlmConfig:
    def test_defaults(self, monkeypatch):
        # LlmConfig reads GROQ_API_KEY from os.environ via a
        # default_factory lambda.  Since backend.main now calls
        # load_dotenv() at import time, the env var may already be set.
        # Clear it for this test so we assert the default (empty string).
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        cfg = LlmConfig()
        assert cfg.model_name == "llama-3.1-70b-versatile"
        assert cfg.temperature == 0.2
        assert cfg.max_output_tokens == 1024
        assert cfg.api_key == ""

    def test_temperature_out_of_range_raises(self):
        with pytest.raises(ValueError, match="LLM_TEMPERATURE"):
            LlmConfig(temperature=3.0)

    def test_max_output_tokens_negative_raises(self):
        with pytest.raises(ValueError, match="LLM_MAX_TOKENS"):
            LlmConfig(max_output_tokens=0)

    def test_model_name_is_fixed_to_llama_3_1_70b(self):
        """The model_name field is always ``llama-3.1-70b-versatile`` — it
        is not configurable via environment variable or constructor
        argument."""
        cfg = LlmConfig()
        assert cfg.model_name == "llama-3.1-70b-versatile"

        # Explicitly passing the valid name is accepted.
        cfg2 = LlmConfig(model_name="llama-3.1-70b-versatile")
        assert cfg2.model_name == "llama-3.1-70b-versatile"

    def test_model_name_not_llama_3_1_70b_raises(self):
        """Any model_name other than exactly 'llama-3.1-70b-versatile'
        raises ValueError at construction time."""
        with pytest.raises(ValueError, match="model_name must be"):
            LlmConfig(model_name="gemini-1.5-flash")

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
# generate_rag_answer — integration with mocked Groq
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

        with patch(
            "backend.llm.adapter._call_groq", return_value=json.loads(mock_response_text)
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
        assert result.model == "llama-3.1-70b-versatile"

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
            "backend.llm.adapter._call_groq", return_value=json.loads(response)
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
            "backend.llm.adapter._call_groq", return_value=json.loads(response)
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
            "backend.llm.adapter._call_groq",
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
            "backend.llm.adapter._call_groq",
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
            "backend.llm.adapter._call_groq",
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
            "backend.llm.adapter._call_groq",
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
            "backend.llm.adapter._call_groq",
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
            "backend.llm.adapter._call_groq",
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
            "backend.llm.adapter._call_groq",
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
            "backend.llm.adapter._call_groq",
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
        """_call_groq strips ```json ... ``` fences before JSON parsing.

        This test mocks the Groq client so _call_groq exercises the real
        fence-stripping logic (unlike tests that mock _call_groq itself
        and bypass the code under test).
        """
        raw_json = json.dumps(
            {
                "answer": "Debe mantener la herida limpia.",
                "cited_chunk_ids": ["c1"],
                "insufficient_knowledge": False,
            }
        )
        fenced = f"```json\n{raw_json}\n```"

        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = fenced
        mock_client.chat.completions.create.return_value.choices = [mock_choice]

        mock_groq_module = MagicMock()
        mock_groq_module.Groq.return_value = mock_client

        with patch.dict("sys.modules", {"groq": mock_groq_module}):
            result = generate_rag_answer(
                "¿Cómo cuidar la herida?", context_chunks, config
            )

        assert result.insufficient_knowledge is False
        assert "herida limpia" in result.answer
        assert len(result.citations) == 1
        assert result.citations[0].chunk_id == "c1"

    def test_markdown_fences_no_language_tag(self, config, context_chunks):
        """_call_groq handles ``` without a language tag."""
        raw_json = json.dumps(
            {
                "answer": "Recomendación postoperatoria.",
                "cited_chunk_ids": ["c1"],
                "insufficient_knowledge": False,
            }
        )
        fenced = f"```\n{raw_json}\n```"

        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = fenced
        mock_client.chat.completions.create.return_value.choices = [mock_choice]

        mock_groq_module = MagicMock()
        mock_groq_module.Groq.return_value = mock_client

        with patch.dict("sys.modules", {"groq": mock_groq_module}):
            result = generate_rag_answer(
                "pregunta", context_chunks, config
            )

        assert result.insufficient_knowledge is False


# ---------------------------------------------------------------------------
# _call_groq unit tests (client construction, parsing, error wrapping)
# ---------------------------------------------------------------------------


class TestCallGroq:
    """Unit tests for _call_groq that mock the groq module entirely.

    No real API calls are made; the groq client is replaced with a
    MagicMock via module-level patching.
    """

    @pytest.fixture
    def config(self) -> LlmConfig:
        return LlmConfig(api_key="test-key", temperature=0.0)

    @pytest.fixture
    def mock_groq_module(self):
        """Return a fresh MagicMock for the groq module with a working
        chat completions endpoint."""
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(
            {
                "answer": "Una respuesta clínica.",
                "cited_chunk_ids": ["c1"],
                "insufficient_knowledge": False,
            }
        )
        mock_client.chat.completions.create.return_value.choices = [mock_choice]

        mock_module = MagicMock()
        mock_module.Groq.return_value = mock_client
        return mock_module

    def _call_with_module(self, sys_prompt, user_prompt, config, mock_module):
        """Helper: call _call_groq with the groq module pre-patched in
        sys.modules."""
        with patch.dict("sys.modules", {"groq": mock_module}):
            return _call_groq(sys_prompt, user_prompt, config)

    # -- Successful call -----------------------------------------------------

    def test_successful_call_parses_json(self, config, mock_groq_module):
        result = self._call_with_module("sys", "user", config, mock_groq_module)

        assert result["answer"] == "Una respuesta clínica."
        assert result["cited_chunk_ids"] == ["c1"]
        assert result["insufficient_knowledge"] is False

    # -- Groq client is instantiated with the API key ------------------------

    def test_groq_client_instantiated_with_api_key(self, config, mock_groq_module):
        self._call_with_module("sys", "user", config, mock_groq_module)

        mock_groq_module.Groq.assert_called_once_with(api_key="test-key")

    # -- Chat completions called with correct params -------------------------

    def test_chat_completions_called_with_correct_params(
        self, config, mock_groq_module,
    ):
        self._call_with_module("sys", "user", config, mock_groq_module)

        mock_groq_module.Groq.return_value.chat.completions.create.assert_called_once()
        call_kwargs = (
            mock_groq_module.Groq.return_value.chat.completions.create.call_args.kwargs
        )
        assert call_kwargs["model"] == "llama-3.1-70b-versatile"
        assert call_kwargs["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user"},
        ]
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["max_tokens"] == 1024
        assert call_kwargs["response_format"] == {"type": "json_object"}

    # -- Missing API key -----------------------------------------------------

    def test_missing_api_key_raises_runtime_error(self, mock_groq_module):
        config_no_key = LlmConfig(api_key="", temperature=0.0)
        with patch.dict("sys.modules", {"groq": mock_groq_module}):
            with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
                _call_groq("sys", "user", config_no_key)

    # -- Lazy import: ImportError wrapping -----------------------------------

    def test_import_error_raises_runtime_error(self, config):
        """When groq is not installed, _call_groq wraps the ImportError
        in a RuntimeError with a clear message."""
        with patch.dict("sys.modules", {"groq": None}):
            with patch(
                "builtins.__import__", side_effect=ImportError("no module"),
            ):
                with pytest.raises(RuntimeError, match="groq"):
                    _call_groq("sys", "user", config)

    # -- Chat completions RuntimeError wrapping ------------------------------

    def test_chat_completions_error_wraps_as_runtime_error(
        self, config, mock_groq_module,
    ):
        mock_groq_module.Groq.return_value.chat.completions.create.side_effect = (
            Exception("API rate limit exceeded")
        )
        with patch.dict("sys.modules", {"groq": mock_groq_module}):
            with pytest.raises(RuntimeError, match="API rate limit exceeded"):
                _call_groq("sys", "user", config)

    # -- Invalid JSON response -----------------------------------------------

    def test_invalid_json_response_raises_value_error(self, config, mock_groq_module):
        mock_groq_module.Groq.return_value.chat.completions.create.return_value.choices[
            0
        ].message.content = "not valid json at all"
        with patch.dict("sys.modules", {"groq": mock_groq_module}):
            with pytest.raises(ValueError, match="no devolvió JSON"):
                _call_groq("sys", "user", config)

    # -- Non-dict JSON response ----------------------------------------------

    def test_non_dict_json_response_raises_value_error(
        self, config, mock_groq_module,
    ):
        mock_groq_module.Groq.return_value.chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps(["not", "a", "dict"])
        with patch.dict("sys.modules", {"groq": mock_groq_module}):
            with pytest.raises(ValueError, match="no es un objeto"):
                _call_groq("sys", "user", config)

    # -- Response content is None --------------------------------------------

    def test_response_content_none_raises_value_error(
        self, config, mock_groq_module,
    ):
        """When message.content is None, _call_groq attempts to parse ''
        and raises ValueError because '' is not valid JSON."""
        mock_groq_module.Groq.return_value.chat.completions.create.return_value.choices[
            0
        ].message.content = None
        with patch.dict("sys.modules", {"groq": mock_groq_module}):
            with pytest.raises(ValueError, match="no devolvió JSON"):
                _call_groq("sys", "user", config)
