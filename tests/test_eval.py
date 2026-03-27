"""Unit tests for src/eval.py — metrics and evaluation helpers."""
import json
import pytest
from src.eval import (
    _field_similarity,
    _normalize_value,
    flesch_kincaid_grade,
    safe_json_loads,
)


class TestFleschKincaidGrade:
    def test_simple_text_lower_grade(self):
        simple = "The cat sat on the mat. Cats are nice pets."
        grade = flesch_kincaid_grade(simple)
        assert grade < 8.0

    def test_complex_text_higher_grade(self):
        complex_text = (
            "Hypertension, characterized by persistently elevated systolic and diastolic arterial pressure, "
            "is a prevalent cardiovascular pathophysiological condition associated with increased morbidity. "
            "Pharmacological interventions, including antihypertensive therapeutics, are frequently prescribed."
        )
        grade = flesch_kincaid_grade(complex_text)
        assert grade > 10.0

    def test_empty_text_no_crash(self):
        grade = flesch_kincaid_grade("")
        assert isinstance(grade, float)

    def test_single_word(self):
        grade = flesch_kincaid_grade("Hello.")
        assert isinstance(grade, float)


class TestSafeJsonLoads:
    def test_valid_json(self):
        text = '{"key": "value", "num": 42}'
        result = safe_json_loads(text)
        assert result == {"key": "value", "num": 42}

    def test_json_embedded_in_text(self):
        text = 'Here is the result: {"chief_complaint": "headache"} end.'
        result = safe_json_loads(text)
        assert result == {"chief_complaint": "headache"}

    def test_no_json_returns_none(self):
        assert safe_json_loads("No JSON here at all.") is None

    def test_malformed_json_returns_none(self):
        assert safe_json_loads("{malformed: json}") is None

    def test_empty_string_returns_none(self):
        assert safe_json_loads("") is None


class TestNormalizeValue:
    def test_string(self):
        assert _normalize_value("Headache") == "headache"

    def test_list_sorted(self):
        result = _normalize_value(["fever", "chills", "fatigue"])
        assert result == "chills fatigue fever"

    def test_empty_list(self):
        assert _normalize_value([]) == ""

    def test_dict(self):
        result = _normalize_value({"a": 1, "b": 2})
        assert "a:1" in result and "b:2" in result

    def test_none(self):
        assert _normalize_value(None) == "none"


class TestFieldSimilarity:
    def test_exact_string_match(self):
        assert _field_similarity("headache", "headache") == 1.0

    def test_case_insensitive(self):
        assert _field_similarity("Headache", "headache") == 1.0

    def test_partial_match_between_zero_and_one(self):
        sim = _field_similarity("headache and fever", "headache")
        assert 0.0 < sim < 1.0

    def test_empty_gold_with_pred_penalized(self):
        # Empty gold but non-empty pred gets 0.5 (penalty for hallucinating)
        assert _field_similarity("anything", "") == 0.5

    def test_empty_gold_empty_pred_gives_one(self):
        # Both empty is a perfect match
        assert _field_similarity("", "") == 1.0

    def test_list_similarity(self):
        sim = _field_similarity(["fever", "chills"], ["fever", "chills"])
        assert sim == 1.0

    def test_list_partial_similarity(self):
        sim = _field_similarity(["fever"], ["fever", "chills"])
        assert 0.0 < sim < 1.0

    def test_completely_different_strings(self):
        sim = _field_similarity("xyz", "abc")
        assert sim < 0.5
