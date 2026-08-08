"""Tests for ``backend.decision.rules`` — the deterministic classify() engine."""

import pytest

from backend.decision.models import EscalationResult, Severity
from backend.decision.rules import (
    _extract_number,
    _find_matches,
    _is_negated,
    classify,
)


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestExtractNumber:
    """Numeric extraction from Spanish patient text."""

    def test_integer(self):
        assert _extract_number("me duele 7") == 7.0

    def test_decimal_comma(self):
        assert _extract_number("temperatura 37,8 grados") == 37.8

    def test_decimal_dot(self):
        assert _extract_number("fiebre de 38.5") == 38.5

    def test_no_number(self):
        assert _extract_number("no tengo fiebre") is None

    def test_first_number_only(self):
        assert _extract_number("dolor 5 y luego 9") == 5.0

    def test_single_digit(self):
        assert _extract_number("0") == 0.0


class TestIsNegated:
    """Negation detection works for common Spanish patterns."""

    def test_direct_negation(self):
        assert _is_negated("no tengo fiebre", "fiebre", 9) is True

    def test_no_negation(self):
        assert _is_negated("tengo fiebre", "fiebre", 6) is False

    def test_sin_negation(self):
        assert _is_negated("sin fiebre", "fiebre", 4) is True

    def test_negation_too_far(self):
        # "no" is far away, won't be caught in the token window
        lowered = "no tuve nada la semana pasada pero ahora tengo fiebre"
        idx = lowered.find("fiebre")  # far from "no"
        assert _is_negated(lowered, "fiebre", idx) is False

    def test_negation_close(self):
        lowered = "no tengo fiebre"
        idx = lowered.find("fiebre")
        assert _is_negated(lowered, "fiebre", idx) is True


class TestFindMatches:
    """Keyword matching is case-insensitive and negation-aware."""

    def test_finds_keyword(self):
        matches = _find_matches("tengo dolor insoportable", ["insoportable", "leve"])
        assert matches == ["insoportable"]

    def test_excludes_negated(self):
        matches = _find_matches(
            "no tengo insoportable dolor", ["insoportable"], respect_negation=True
        )
        assert matches == []

    def test_no_negation_check(self):
        matches = _find_matches(
            "no tengo insoportable dolor", ["insoportable"], respect_negation=False
        )
        assert matches == ["insoportable"]

    def test_multiple_matches(self):
        matches = _find_matches("dolor intenso e insoportable", ["insoportable", "intenso"])
        assert set(matches) == {"insoportable", "intenso"}


# ---------------------------------------------------------------------------
# classify() — domain / argument validation
# ---------------------------------------------------------------------------


class TestClassifyArgumentValidation:
    """Classify validates its input arguments."""

    def test_invalid_domain_raises(self):
        with pytest.raises(ValueError, match="Unknown domain"):
            classify("algo", "respiración")

    def test_negative_dia_postop_raises(self):
        with pytest.raises(ValueError, match="dia_postop must be"):
            classify("dolor leve", "dolor", dia_postop=-1)

    def test_domain_case_insensitive(self):
        result = classify("bien, sin dolor", "DOLOR")
        assert result.severity is Severity.GREEN

    def test_domain_with_accent(self):
        result = classify("sin fiebre, normal", "fiebre")
        assert result.severity is Severity.GREEN

    def test_domain_sueno_alias(self):
        result = classify("duermo bien", "sueno")
        assert result.severity is Severity.GREEN


# ---------------------------------------------------------------------------
# classify() — empty / invalid input
# ---------------------------------------------------------------------------


class TestClassifyInvalidInput:
    """Empty or whitespace-only input returns YELLOW with source='invalid'."""

    def test_empty_string(self):
        result = classify("", "dolor")
        assert result.severity is Severity.YELLOW
        assert result.should_escalate is False
        assert result.source == "invalid"
        assert result.domain == "dolor"

    def test_whitespace_only(self):
        result = classify("   \t  \n  ", "fiebre")
        assert result.severity is Severity.YELLOW
        assert result.source == "invalid"

    def test_invalid_has_reason(self):
        result = classify("", "herida")
        assert len(result.reason) > 0
        assert len(result.next_action) > 0


# ---------------------------------------------------------------------------
# classify() — cross-cutting red flags (domain-agnostic RED)
# ---------------------------------------------------------------------------


class TestCrossCuttingRedFlags:
    """Cross-cutting red flags trigger RED regardless of domain."""

    def test_chest_pain_crosses_domains(self):
        for domain in ("dolor", "fiebre", "herida", "apetito", "sueno", "movilidad"):
            result = classify("siento un dolor en el pecho muy fuerte", domain)
            assert result.severity is Severity.RED, f"Failed for domain {domain}"
            assert result.should_escalate is True

    def test_hemorrhage_red(self):
        result = classify("tengo una hemorragia", "herida")
        assert result.severity is Severity.RED

    def test_convulsion_red(self):
        result = classify("tuve convulsiones anoche", "fiebre")
        assert result.severity is Severity.RED

    def test_no_puedo_respirar_red(self):
        result = classify("no puedo respirar bien", "movilidad")
        assert result.severity is Severity.RED

    def test_negated_cross_cut_not_red(self):
        # "no he tenido dolor en el pecho" → negation should prevent RED
        result = classify("no he tenido dolor en el pecho", "dolor")
        assert result.severity is not Severity.RED


# ---------------------------------------------------------------------------
# classify() — domain-specific red flags
# ---------------------------------------------------------------------------


class TestRedFlagDolor:
    """Pain red flags trigger RED."""

    def test_insoportable(self):
        r = classify("el dolor es insoportable", "dolor")
        assert r.severity is Severity.RED

    def test_no_aguanto(self):
        r = classify("no aguanto este dolor", "dolor")
        assert r.severity is Severity.RED

    def test_peor_dolor(self):
        r = classify("es el peor dolor de mi vida", "dolor")
        assert r.severity is Severity.RED

    def test_negated_insoportable_not_red(self):
        r = classify("no es insoportable el dolor", "dolor")
        # "no" + "insoportable" → negated
        assert r.severity is not Severity.RED


class TestRedFlagFiebre:
    """Fever red flags trigger RED."""

    def test_fiebre_alta(self):
        r = classify("tengo fiebre alta", "fiebre")
        assert r.severity is Severity.RED

    def test_confusion(self):
        r = classify("estoy confundido y con fiebre", "fiebre")
        assert r.severity is Severity.RED

    def test_convulsiones(self):
        r = classify("mi hijo tuvo convulsiones con la fiebre", "fiebre")
        assert r.severity is Severity.RED


class TestRedFlagHerida:
    """Wound red flags trigger RED."""

    def test_pus(self):
        r = classify("la herida tiene pus", "herida")
        assert r.severity is Severity.RED

    def test_se_abrio(self):
        r = classify("la herida se abrió ayer", "herida")
        assert r.severity is Severity.RED

    def test_mal_olor(self):
        r = classify("tiene mal olor la herida", "herida")
        assert r.severity is Severity.RED

    def test_negated_pus_not_red(self):
        r = classify("no tiene pus la herida", "herida")
        assert r.severity is not Severity.RED


class TestRedFlagApetito:
    """Appetite red flags trigger RED."""

    def test_no_tolero_nada(self):
        r = classify("no tolero nada, todo lo vomito", "apetito")
        assert r.severity is Severity.RED

    def test_no_he_comido(self):
        r = classify("no he comido en tres días", "apetito")
        assert r.severity is Severity.RED

    def test_vomito_con_sangre(self):
        r = classify("vomito con sangre", "apetito")
        assert r.severity is Severity.RED


class TestRedFlagSueno:
    """Sleep red flags trigger RED."""

    def test_no_duermo_nada(self):
        r = classify("no duermo nada desde la cirugía", "sueno")
        assert r.severity is Severity.RED

    def test_me_despierta_dolor(self):
        r = classify("me despierta el dolor cada hora", "sueno")
        assert r.severity is Severity.RED


class TestRedFlagMovilidad:
    """Mobility red flags trigger RED."""

    def test_no_me_puedo_levantar(self):
        r = classify("no me puedo levantar de la cama", "movilidad")
        assert r.severity is Severity.RED

    def test_me_cai(self):
        r = classify("me caí ayer en el baño", "movilidad")
        assert r.severity is Severity.RED

    def test_no_puedo_caminar(self):
        r = classify("no puedo caminar sin ayuda", "movilidad")
        assert r.severity is Severity.RED


# ---------------------------------------------------------------------------
# classify() — numeric thresholds
# ---------------------------------------------------------------------------


class TestNumericPain:
    """Pain NRS scores trigger the correct severity."""

    def test_pain_9_red(self):
        r = classify("mi dolor es 9 de 10", "dolor")
        assert r.severity is Severity.RED
        assert r.source == "numeric"

    def test_pain_8_red(self):
        r = classify("dolor 8", "dolor")
        assert r.severity is Severity.RED

    def test_pain_7_yellow(self):
        r = classify("dolor como 7", "dolor")
        assert r.severity is Severity.YELLOW
        assert r.source == "numeric"

    def test_pain_5_yellow(self):
        r = classify("me duele 5", "dolor")
        assert r.severity is Severity.YELLOW

    def test_pain_4_green_falls_through(self):
        # 4 is below YELLOW threshold → falls through to green indicators
        r = classify("dolor 4, estoy mejor", "dolor")
        assert r.severity is Severity.GREEN

    def test_pain_0_green(self):
        r = classify("cero dolor", "dolor")
        assert r.severity is Severity.GREEN


class TestNumericTemperature:
    """Temperature values trigger the correct severity."""

    def test_temp_39_red(self):
        r = classify("tuve 39 de fiebre", "fiebre")
        assert r.severity is Severity.RED
        assert r.source == "numeric"

    def test_temp_38_5_red(self):
        r = classify("temperatura de 38.5", "fiebre")
        assert r.severity is Severity.RED

    def test_temp_38_yellow(self):
        r = classify("fiebre de 38 grados", "fiebre")
        assert r.severity is Severity.YELLOW

    def test_temp_37_8_yellow(self):
        r = classify("temperatura 37,8", "fiebre")
        assert r.severity is Severity.YELLOW

    def test_temp_37_normal(self):
        r = classify("temperatura normal 37", "fiebre")
        assert r.severity is Severity.GREEN


# ---------------------------------------------------------------------------
# classify() — ambiguity / uncertainty
# ---------------------------------------------------------------------------


class TestAmbiguity:
    """Ambiguous answers return YELLOW with source='ambig'."""

    def test_no_se(self):
        r = classify("no sé, la verdad", "dolor")
        assert r.severity is Severity.YELLOW
        assert r.source == "ambig"

    def test_mas_o_menos(self):
        r = classify("más o menos, a veces duele", "dolor")
        assert r.severity is Severity.YELLOW
        assert r.source == "ambig"

    def test_regular(self):
        r = classify("regular", "herida")
        assert r.severity is Severity.YELLOW
        assert r.source == "ambig"

    def test_no_estoy_seguro(self):
        r = classify("no estoy seguro si ha mejorado", "movilidad")
        assert r.severity is Severity.YELLOW

    def test_ambig_before_yellow(self):
        # "fiebre" is a yellow trigger, but "no sé si tengo fiebre"
        # should be caught as ambiguous first
        r = classify("no sé si tengo fiebre o no", "fiebre")
        assert r.severity is Severity.YELLOW
        assert r.source == "ambig"


# ---------------------------------------------------------------------------
# classify() — yellow triggers
# ---------------------------------------------------------------------------


class TestYellowTriggers:
    """Concerning symptoms trigger YELLOW."""

    def test_dolor_moderado(self):
        r = classify("tengo un dolor moderado", "dolor")
        assert r.severity is Severity.YELLOW
        assert r.source == "rule"

    def test_fiebre_general(self):
        r = classify("tuve un poco de fiebre ayer", "fiebre")
        assert r.severity is Severity.YELLOW

    def test_herida_enrojecimiento(self):
        r = classify("la herida tiene enrojecimiento", "herida")
        assert r.severity is Severity.YELLOW

    def test_poco_apetito(self):
        r = classify("tengo poco apetito", "apetito")
        assert r.severity is Severity.YELLOW

    def test_duermo_mal(self):
        r = classify("duermo mal, me despierto seguido", "sueno")
        assert r.severity is Severity.YELLOW

    def test_debilidad(self):
        r = classify("me siento débil", "movilidad")
        assert r.severity is Severity.YELLOW

    def test_negated_yellow_not_yellow(self):
        r = classify("no tengo fiebre ni escalofríos", "fiebre")
        assert r.severity is not Severity.YELLOW


# ---------------------------------------------------------------------------
# classify() — green indicators
# ---------------------------------------------------------------------------


class TestGreenIndicators:
    """Reassuring symptoms trigger GREEN."""

    def test_sin_dolor(self):
        r = classify("no tengo dolor, estoy bien", "dolor")
        assert r.severity is Severity.GREEN

    def test_mejorando(self):
        r = classify("estoy mejorando cada día", "dolor")
        assert r.severity is Severity.GREEN

    def test_sin_fiebre(self):
        r = classify("no he tenido fiebre", "fiebre")
        assert r.severity is Severity.GREEN

    def test_herida_bien(self):
        r = classify("la herida está bien, cicatrizando", "herida")
        assert r.severity is Severity.GREEN

    def test_buen_apetito(self):
        r = classify("tengo buen apetito, como normal", "apetito")
        assert r.severity is Severity.GREEN

    def test_duermo_bien(self):
        r = classify("duermo bien toda la noche", "sueno")
        assert r.severity is Severity.GREEN

    def test_camino_bien(self):
        r = classify("camino bien, sin ayuda", "movilidad")
        assert r.severity is Severity.GREEN


# ---------------------------------------------------------------------------
# classify() — fallback (incomplete)
# ---------------------------------------------------------------------------


class TestFallbackIncomplete:
    """When nothing matches, return YELLOW with source='incomplete'."""

    def test_unrecognisable_text(self):
        r = classify("xyz abc 123", "dolor")
        assert r.severity is Severity.YELLOW
        assert r.source == "incomplete"
        assert r.should_escalate is False

    def test_greeting_not_symptom(self):
        r = classify("buenos días doctor", "dolor")
        assert r.severity is Severity.YELLOW
        assert r.source == "incomplete"


# ---------------------------------------------------------------------------
# classify() — return type integrity
# ---------------------------------------------------------------------------


class TestReturnTypeIntegrity:
    """Every classify() return value passes EscalationResult validation."""

    def test_all_domains_produce_valid_results(self):
        inputs = [
            ("tengo dolor 7", "dolor"),
            ("sin fiebre", "fiebre"),
            ("la herida está bien", "herida"),
            ("como normal", "apetito"),
            ("duermo bien", "sueno"),
            ("camino bien", "movilidad"),
            ("", "dolor"),
        ]
        for text, domain in inputs:
            r = classify(text, domain)
            assert isinstance(r, EscalationResult)
            assert isinstance(r.severity, Severity)
            assert isinstance(r.reason, str) and len(r.reason) > 0
            assert isinstance(r.next_action, str) and len(r.next_action) > 0
            assert isinstance(r.should_escalate, bool)

    def test_red_always_should_escalate(self):
        red_inputs = [
            ("dolor insoportable 9", "dolor"),
            ("fiebre alta con confusión", "fiebre"),
            ("la herida tiene pus y mal olor", "herida"),
            ("no tolero nada, vómito todo", "apetito"),
            ("me despierta el dolor cada hora", "sueno"),
            ("no me puedo levantar y me caí", "movilidad"),
            ("tengo dolor en el pecho", "dolor"),
        ]
        for text, domain in red_inputs:
            r = classify(text, domain)
            assert r.severity is Severity.RED, f"Not RED: {text!r} in {domain}"
            assert r.should_escalate is True, f"Not escalating: {text!r} in {domain}"

    def test_green_never_should_escalate(self):
        green_inputs = [
            ("sin dolor, estoy bien", "dolor"),
            ("no tengo fiebre", "fiebre"),
            ("herida limpia y seca", "herida"),
            ("como bien, buen apetito", "apetito"),
            ("duermo bien toda la noche", "sueno"),
            ("camino sin ayuda, mejorando", "movilidad"),
        ]
        for text, domain in green_inputs:
            r = classify(text, domain)
            if r.severity is Severity.GREEN:
                assert r.should_escalate is False
