"""Unit tests for src/data_prep.py — data pipeline helpers."""
import json
import pytest
from src.data_prep import (
    Record,
    bulletize,
    build_extraction_records,
    clean_text,
    make_synthetic_note,
    parse_pubmedqa_record,
    split_records,
)

REQUIRED_EXTRACTION_KEYS = [
    "chief_complaint", "symptoms", "duration",
    "vitals", "meds", "allergies", "past_history", "red_flags",
]


class TestCleanText:
    def test_collapses_whitespace(self):
        assert clean_text("hello   world") == "hello world"

    def test_strips_leading_trailing(self):
        assert clean_text("  hello  ") == "hello"

    def test_none_input(self):
        assert clean_text(None) == ""  # type: ignore[arg-type]

    def test_empty_string(self):
        assert clean_text("") == ""


class TestBulletize:
    def test_returns_exactly_n_bullets(self):
        text = "Sentence one. Sentence two is here. Sentence three follows. Sentence four continues. Sentence five ends."
        bullets = bulletize(text, n=5)
        assert len(bullets) == 5

    def test_pads_short_text(self):
        bullets = bulletize("Short.", n=5)
        assert len(bullets) == 5
        assert any("clinician" in b.lower() for b in bullets)

    def test_no_empty_bullets(self):
        bullets = bulletize("Hello there. How are you today.", n=5)
        assert all(b.strip() for b in bullets)

    def test_custom_n(self):
        text = "One. Two. Three. Four. Five. Six. Seven."
        assert len(bulletize(text, n=3)) == 3


class TestMakeSyntheticNote:
    def test_returns_note_and_dict(self):
        note, target = make_synthetic_note()
        assert isinstance(note, str)
        assert isinstance(target, dict)

    def test_all_required_keys_present(self):
        _, target = make_synthetic_note()
        for key in REQUIRED_EXTRACTION_KEYS:
            assert key in target, f"Missing key: {key}"

    def test_note_contains_chief_complaint(self):
        note, target = make_synthetic_note()
        assert target["chief_complaint"].lower() in note.lower()

    def test_vitals_has_all_fields(self):
        _, target = make_synthetic_note()
        vitals = target["vitals"]
        assert "temp_f" in vitals
        assert "heart_rate" in vitals
        assert "bp" in vitals
        assert "spo2" in vitals

    def test_vitals_in_realistic_range(self):
        _, target = make_synthetic_note()
        v = target["vitals"]
        assert 95 <= v["temp_f"] <= 105
        assert 50 <= v["heart_rate"] <= 130
        assert 88 <= v["spo2"] <= 100

    def test_reproducibility_with_same_seed(self):
        import random
        import src.data_prep as dp
        original_rng = dp.RNG
        dp.RNG = random.Random(123)
        note1, target1 = make_synthetic_note()
        dp.RNG = random.Random(123)
        note2, target2 = make_synthetic_note()
        dp.RNG = original_rng
        assert note1 == note2
        assert target1 == target2


class TestBuildExtractionRecords:
    def test_returns_correct_count(self):
        records = build_extraction_records(10)
        assert len(records) == 10

    def test_all_records_are_extract_task(self):
        records = build_extraction_records(5)
        assert all(r.task == "extract" for r in records)

    def test_output_is_valid_json(self):
        records = build_extraction_records(5)
        for r in records:
            obj = json.loads(r.output)
            for key in REQUIRED_EXTRACTION_KEYS:
                assert key in obj

    def test_record_has_required_fields(self):
        records = build_extraction_records(1)
        r = records[0]
        assert r.id
        assert r.task == "extract"
        assert r.instruction
        assert r.input
        assert r.output


class TestSplitRecords:
    def _make_records(self, n: int) -> list[Record]:
        return [
            Record(id=str(i), task="summarize", instruction="inst", input="in", output="out")
            for i in range(n)
        ]

    def test_splits_sum_to_total(self):
        records = self._make_records(100)
        splits = split_records(records)
        total = sum(len(v) for v in splits.values())
        assert total == 100

    def test_train_is_largest(self):
        records = self._make_records(100)
        splits = split_records(records)
        assert len(splits["train"]) > len(splits["val"])
        assert len(splits["train"]) > len(splits["test"])

    def test_no_overlap_between_splits(self):
        records = self._make_records(50)
        splits = split_records(records)
        train_ids = {r.id for r in splits["train"]}
        val_ids = {r.id for r in splits["val"]}
        test_ids = {r.id for r in splits["test"]}
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_tiny_dataset_no_crash(self):
        records = self._make_records(3)
        splits = split_records(records)
        assert sum(len(v) for v in splits.values()) == 3


class TestParsePubmedQARecord:
    def _make_example(self, question="What is hypertension?", long_answer="It is high blood pressure.", answer="yes"):
        return {
            "question": question,
            "context": {"contexts": ["Hypertension is elevated blood pressure."], "labels": [], "meshes": []},
            "long_answer": long_answer,
            "final_decision": answer,
        }

    def test_returns_two_records(self):
        ex = self._make_example()
        result = parse_pubmedqa_record(ex, idx=0)
        assert result is not None
        assert len(result) == 2

    def test_first_record_is_summarize(self):
        ex = self._make_example()
        summarize, term = parse_pubmedqa_record(ex, idx=0)
        assert summarize.task == "summarize"

    def test_second_record_is_term(self):
        ex = self._make_example()
        summarize, term = parse_pubmedqa_record(ex, idx=0)
        assert term.task == "term"

    def test_summarize_output_has_bullets(self):
        ex = self._make_example()
        summarize, _ = parse_pubmedqa_record(ex, idx=0)
        assert "- " in summarize.output

    def test_summarize_output_has_disclaimer(self):
        ex = self._make_example()
        summarize, _ = parse_pubmedqa_record(ex, idx=0)
        assert "safety disclaimer" in summarize.output.lower()

    def test_term_output_has_citation(self):
        ex = self._make_example()
        _, term = parse_pubmedqa_record(ex, idx=0)
        assert "[context-1]" in term.output

    def test_empty_context_and_answer_returns_none(self):
        ex = {
            "question": "What?",
            "context": {},
            "long_answer": "",
            "final_decision": "",
        }
        result = parse_pubmedqa_record(ex, idx=0)
        assert result is None
