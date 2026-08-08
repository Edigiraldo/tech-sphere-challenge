"""Tests for ``backend.decision.lexicon`` — keyword lists and threshold values."""

import string

from backend.decision.lexicon import (
    ALL_DOMAINS,
    AMBIGUITY_PHRASES,
    CROSS_CUTTING_RED_FLAGS,
    DOMAIN_DOLOR,
    DOMAIN_FIEBRE,
    DOMAIN_HERIDA,
    DOMAIN_APETITO,
    DOMAIN_SUENO,
    DOMAIN_MOVILIDAD,
    DOMAIN_DISPATCH,
    GREEN_INDICATORS,
    NEGATION_MARKERS,
    PAIN_RED_THRESHOLD,
    PAIN_YELLOW_THRESHOLD,
    RED_FLAGS,
    TEMP_RED_THRESHOLD,
    TEMP_YELLOW_THRESHOLD,
    YELLOW_TRIGGERS,
)


class TestDomainKeys:
    """Domain dispatch covers all six follow-up domains."""

    def test_all_domains_have_six_entries(self):
        assert len(ALL_DOMAINS) == 6

    def test_dispatch_covers_all_canonical(self):
        for d in ALL_DOMAINS:
            assert d in DOMAIN_DISPATCH
            assert DOMAIN_DISPATCH[d] == d

    def test_dispatch_sueno_alias(self):
        assert DOMAIN_DISPATCH["sueno"] == DOMAIN_SUENO
        assert DOMAIN_DISPATCH["sueño"] == DOMAIN_SUENO

    def test_dispatch_case_insensitive(self):
        for key in ("Dolor", "DOLOR", "dolor"):
            assert DOMAIN_DISPATCH.get(key.lower()) == DOMAIN_DOLOR


class TestRedFlags:
    """Every domain has at least one red-flag keyword."""

    def test_each_domain_has_red_flags(self):
        for d in ALL_DOMAINS:
            assert len(RED_FLAGS[d]) > 0, f"No red flags for {d}"

    def test_red_flags_lowercase(self):
        for d in ALL_DOMAINS:
            for kw in RED_FLAGS[d]:
                assert kw == kw.lower(), f"Red flag not lowercase: {kw!r}"

    def test_dolor_has_insoportable(self):
        assert "insoportable" in RED_FLAGS[DOMAIN_DOLOR]

    def test_fiebre_has_convulsiones(self):
        assert "convulsiones" in RED_FLAGS[DOMAIN_FIEBRE]

    def test_herida_has_pus(self):
        assert "pus" in RED_FLAGS[DOMAIN_HERIDA]

    def test_herida_has_se_abrio(self):
        assert "se abrió" in RED_FLAGS[DOMAIN_HERIDA]

    def test_apetito_has_vomito_todo(self):
        assert "vómito todo" in RED_FLAGS[DOMAIN_APETITO]

    def test_movilidad_has_caida(self):
        assert any("caída" in kw or "caida" in kw for kw in RED_FLAGS[DOMAIN_MOVILIDAD])

    def test_movilidad_has_no_puedo_respirar(self):
        assert "no puedo respirar" in RED_FLAGS[DOMAIN_MOVILIDAD]


class TestYellowTriggers:
    """Every domain has yellow trigger keywords."""

    def test_each_domain_has_yellow(self):
        for d in ALL_DOMAINS:
            assert len(YELLOW_TRIGGERS[d]) > 0, f"No yellow triggers for {d}"

    def test_yellow_lowercase(self):
        for d in ALL_DOMAINS:
            for kw in YELLOW_TRIGGERS[d]:
                assert kw == kw.lower(), f"Yellow not lowercase: {kw!r}"


class TestGreenIndicators:
    """Every domain has green indicator keywords."""

    def test_each_domain_has_green(self):
        for d in ALL_DOMAINS:
            assert len(GREEN_INDICATORS[d]) > 0, f"No green indicators for {d}"

    def test_green_lowercase(self):
        for d in ALL_DOMAINS:
            for kw in GREEN_INDICATORS[d]:
                assert kw == kw.lower(), f"Green not lowercase: {kw!r}"


class TestNegationMarkers:
    """Negation markers are lowercase and non-empty."""

    def test_not_empty(self):
        assert len(NEGATION_MARKERS) > 0

    def test_lowercase(self):
        for m in NEGATION_MARKERS:
            assert m == m.lower(), f"Negation marker not lowercase: {m!r}"

    def test_core_markers_present(self):
        assert "no" in NEGATION_MARKERS
        assert "sin" in NEGATION_MARKERS


class TestNumericThresholds:
    """Thresholds are in valid clinical ranges."""

    def test_pain_thresholds(self):
        assert 0 <= PAIN_YELLOW_THRESHOLD < PAIN_RED_THRESHOLD <= 10
        assert PAIN_YELLOW_THRESHOLD == 5
        assert PAIN_RED_THRESHOLD == 8

    def test_temp_thresholds(self):
        assert 35.0 <= TEMP_YELLOW_THRESHOLD < TEMP_RED_THRESHOLD <= 43.0
        assert TEMP_YELLOW_THRESHOLD == 37.5
        assert TEMP_RED_THRESHOLD == 38.5


class TestAmbiguityPhrases:
    """Ambiguity phrases are lowercase and include common uncertainty markers."""

    def test_not_empty(self):
        assert len(AMBIGUITY_PHRASES) > 0

    def test_lowercase(self):
        for p in AMBIGUITY_PHRASES:
            # Allow accented chars, but not uppercase
            assert p == p.lower(), f"Ambiguity phrase not lowercase: {p!r}"

    def test_mas_o_menos_present(self):
        assert "más o menos" in AMBIGUITY_PHRASES

    def test_no_se_present(self):
        assert "no sé" in AMBIGUITY_PHRASES


class TestCrossCuttingRedFlags:
    """Cross-cutting red flags include critical systemic symptoms."""

    def test_not_empty(self):
        assert len(CROSS_CUTTING_RED_FLAGS) > 0

    def test_lowercase(self):
        for f in CROSS_CUTTING_RED_FLAGS:
            assert f == f.lower(), f"Cross-cutting flag not lowercase: {f!r}"

    def test_chest_pain_present(self):
        assert "dolor en el pecho" in CROSS_CUTTING_RED_FLAGS

    def test_hemorrhage_present(self):
        assert "hemorragia" in CROSS_CUTTING_RED_FLAGS
