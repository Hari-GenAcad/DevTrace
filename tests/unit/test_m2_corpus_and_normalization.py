"""
M2 Tests — Corpus, Ingestion & Evaluation Dataset

Covers:
  - Corpus loading (count, required fields, stable IDs)
  - Malformed corpus entries (strict and non-strict modes)
  - EvidenceChunk building and ID convention
  - Signal extraction (error codes, versions, technical terms)
  - Version normalization (upgrade pattern, historical, ambiguous)
  - Incident normalization (structured precedence, NL input)
  - Evaluation dataset loading and validation
  - Corpus integrity check (eval references valid doc IDs)

All tests are offline and deterministic.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.evaluation.dataset import (
    EvalLoadError,
    EvaluationCase,
    EvaluationDataset,
    load_eval_dataset,
)
from src.ingestion.loader import (
    CorpusLoadError,
    DocumentValidationError,
    build_chunks,
    load_corpus,
    load_corpus_and_chunks,
)
from src.models.enums import SystemOutcome
from src.normalization.normalizer import normalize_incident, normalize_incident_from_dict
from src.normalization.signals import extract_signals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_temp_json(data, suffix=".json") -> Path:
    """Write data to a temp file and return its path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    )
    json.dump(data, f, ensure_ascii=False)
    f.flush()
    return Path(f.name)


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

class TestCorpusLoading:

    def test_corpus_loads_successfully(self):
        """The default DevCore corpus loads without error."""
        docs = load_corpus()
        assert len(docs) > 0

    def test_corpus_has_expected_count(self):
        """Corpus contains approximately 25-35 documents (target range)."""
        docs = load_corpus()
        assert 25 <= len(docs) <= 35, f"Expected 25-35 docs, got {len(docs)}"

    def test_every_document_has_required_fields(self):
        """Every document has non-empty doc_id, title, content, and applies_to."""
        docs = load_corpus()
        for doc in docs:
            assert doc.doc_id, f"Empty doc_id on document: {doc}"
            assert doc.title, f"Empty title on: {doc.doc_id}"
            assert doc.content, f"Empty content on: {doc.doc_id}"
            assert doc.applies_to, f"Empty applies_to on: {doc.doc_id}"

    def test_every_document_has_topic(self):
        """Every corpus document has a topic field set."""
        docs = load_corpus()
        for doc in docs:
            assert doc.topic is not None, f"Missing topic on: {doc.doc_id}"

    def test_doc_ids_are_unique(self):
        """All document IDs in the corpus are unique."""
        docs = load_corpus()
        ids = [doc.doc_id for doc in docs]
        assert len(ids) == len(set(ids)), "Duplicate doc_ids found in corpus."

    def test_corpus_contains_versioned_and_agnostic_documents(self):
        """Corpus has documents with version specifiers AND version-agnostic '*' docs."""
        docs = load_corpus()
        applies_to_values = {doc.applies_to for doc in docs}
        # At least some version-specific docs.
        versioned = [v for v in applies_to_values if v != "*"]
        assert len(versioned) > 0, "No versioned documents found."
        # At least some version-agnostic docs.
        assert "*" in applies_to_values, "No version-agnostic (*) documents found."

    def test_corpus_contains_2x_and_3x_docs(self):
        """Corpus includes both SDK 2.x and SDK 3.x specific documents (for traps)."""
        docs = load_corpus()
        has_2x = any(">=2.0" in doc.applies_to for doc in docs)
        has_3x = any(">=3.0" in doc.applies_to for doc in docs)
        assert has_2x, "No SDK 2.x documents found."
        assert has_3x, "No SDK 3.x documents found."


# ---------------------------------------------------------------------------
# Stable IDs
# ---------------------------------------------------------------------------

class TestStableIds:

    def test_loading_twice_produces_same_ids(self):
        """Loading the corpus twice produces identical doc_ids in the same order."""
        docs1 = load_corpus()
        docs2 = load_corpus()
        ids1 = [d.doc_id for d in docs1]
        ids2 = [d.doc_id for d in docs2]
        assert ids1 == ids2

    def test_chunk_ids_derived_from_doc_ids(self):
        """EvidenceChunk IDs follow the '{doc_id}-C01' convention."""
        docs = load_corpus()
        chunks = build_chunks(docs)
        for doc, chunk in zip(docs, chunks):
            assert chunk.chunk_id == f"{doc.doc_id}-C01"
            assert chunk.doc_id == doc.doc_id

    def test_chunk_ids_are_unique(self):
        """All chunk IDs are unique."""
        docs = load_corpus()
        chunks = build_chunks(docs)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_count_matches_doc_count(self):
        """One chunk is produced per document."""
        docs = load_corpus()
        chunks = build_chunks(docs)
        assert len(chunks) == len(docs)

    def test_chunk_inherits_applies_to(self):
        """Each chunk's applies_to matches its parent document."""
        docs = load_corpus()
        chunks = build_chunks(docs)
        doc_map = {d.doc_id: d for d in docs}
        for chunk in chunks:
            assert chunk.applies_to == doc_map[chunk.doc_id].applies_to

    def test_load_corpus_and_chunks_convenience(self):
        """load_corpus_and_chunks() returns consistent docs and chunks."""
        docs, chunks = load_corpus_and_chunks()
        assert len(docs) == len(chunks)
        for doc, chunk in zip(docs, chunks):
            assert chunk.doc_id == doc.doc_id


# ---------------------------------------------------------------------------
# Malformed corpus entries
# ---------------------------------------------------------------------------

class TestCorpusValidation:

    def test_missing_doc_id_rejected(self):
        """A corpus entry without doc_id raises DocumentValidationError."""
        bad_corpus = [{"title": "Test", "content": "Content"}]
        path = _write_temp_json(bad_corpus)
        with pytest.raises(DocumentValidationError, match="doc_id"):
            load_corpus(path)

    def test_missing_title_rejected(self):
        """A corpus entry without title raises DocumentValidationError."""
        bad_corpus = [{"doc_id": "X-001", "content": "Content"}]
        path = _write_temp_json(bad_corpus)
        with pytest.raises(DocumentValidationError, match="title"):
            load_corpus(path)

    def test_missing_content_rejected(self):
        """A corpus entry without content raises DocumentValidationError."""
        bad_corpus = [{"doc_id": "X-001", "title": "Title"}]
        path = _write_temp_json(bad_corpus)
        with pytest.raises(DocumentValidationError, match="content"):
            load_corpus(path)

    def test_empty_content_rejected(self):
        """A corpus entry with empty/whitespace content is rejected."""
        bad_corpus = [{"doc_id": "X-001", "title": "Title", "content": "   "}]
        path = _write_temp_json(bad_corpus)
        with pytest.raises(DocumentValidationError):
            load_corpus(path)

    def test_duplicate_doc_id_rejected(self):
        """Two corpus entries with the same doc_id are rejected."""
        dup_corpus = [
            {"doc_id": "X-001", "title": "A", "content": "Content A"},
            {"doc_id": "X-001", "title": "B", "content": "Content B"},
        ]
        path = _write_temp_json(dup_corpus)
        with pytest.raises(DocumentValidationError, match="Duplicate"):
            load_corpus(path)

    def test_non_strict_mode_skips_bad_entries(self):
        """In non-strict mode, malformed entries are skipped and valid ones loaded."""
        mixed_corpus = [
            {"doc_id": "X-001", "title": "Good Doc", "content": "Valid content."},
            {"title": "No ID", "content": "No doc_id here."},  # malformed
        ]
        path = _write_temp_json(mixed_corpus)
        import warnings
        with warnings.catch_warnings(record=True):
            docs = load_corpus(path, strict=False)
        assert len(docs) == 1
        assert docs[0].doc_id == "X-001"

    def test_file_not_found_raises_corpus_load_error(self):
        """Loading a non-existent file raises CorpusLoadError."""
        with pytest.raises(CorpusLoadError, match="not found"):
            load_corpus(Path("/nonexistent/path/corpus.json"))

    def test_invalid_json_raises_corpus_load_error(self):
        """Loading a file with invalid JSON raises CorpusLoadError."""
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        f.write("{not valid json{{")
        f.flush()
        with pytest.raises(CorpusLoadError, match="valid JSON"):
            load_corpus(Path(f.name))

    def test_non_array_json_raises_corpus_load_error(self):
        """A JSON object (not array) at the top level raises CorpusLoadError."""
        path = _write_temp_json({"doc_id": "X-001", "title": "T", "content": "C"})
        with pytest.raises(CorpusLoadError, match="array"):
            load_corpus(path)


# ---------------------------------------------------------------------------
# Signal extraction — error codes
# ---------------------------------------------------------------------------

class TestErrorCodeExtraction:

    def test_single_error_code(self):
        signals = extract_signals("Getting AUTH_401 on every request.")
        assert "AUTH_401" in signals.error_codes

    def test_multiple_error_codes(self):
        signals = extract_signals("AUTH_401 followed by HTTP_429 during retry.")
        assert "AUTH_401" in signals.error_codes
        assert "HTTP_429" in signals.error_codes

    def test_error_codes_deduplicated(self):
        signals = extract_signals("AUTH_401 and AUTH_401 again.")
        assert signals.error_codes.count("AUTH_401") == 1

    def test_error_code_err_timeout(self):
        signals = extract_signals("Requests fail with ERR_TIMEOUT after 30 seconds.")
        assert "ERR_TIMEOUT" in signals.error_codes

    def test_error_code_not_extracted_from_lowercase(self):
        """Lowercase words should not be treated as error codes."""
        signals = extract_signals("the auth error is caused by a config issue")
        # No uppercase error codes present.
        assert "AUTH" not in signals.error_codes

    def test_structured_error_codes_preserved(self):
        """Structured error_codes are always included even if not in description."""
        signals = extract_signals(
            "Something is broken.",
            structured_error_codes=["AUTH_401"],
        )
        assert "AUTH_401" in signals.error_codes

    def test_structured_error_codes_merged_with_text(self):
        """Structured codes are merged with text-extracted codes."""
        signals = extract_signals(
            "Also seeing RATE_429 in logs.",
            structured_error_codes=["AUTH_401"],
        )
        assert "AUTH_401" in signals.error_codes
        assert "RATE_429" in signals.error_codes
        # Structured codes appear first.
        assert signals.error_codes.index("AUTH_401") < signals.error_codes.index("RATE_429")


# ---------------------------------------------------------------------------
# Signal extraction — versions
# ---------------------------------------------------------------------------

class TestVersionExtraction:

    def test_upgrade_pattern_from_to(self):
        """'upgraded from 2.8 to 3.1' → previous=2.8, current=3.1"""
        signals = extract_signals("AUTH_401 after upgrading from 2.8 to 3.1.")
        assert signals.current_version == "3.1"
        assert signals.previous_version == "2.8"

    def test_single_version_becomes_current(self):
        """A single version mention without history becomes current_version."""
        signals = extract_signals("We're on SDK 3.1 and getting AUTH_401.")
        assert signals.current_version == "3.1"
        assert signals.previous_version is None

    def test_historical_version_not_current(self):
        """'used to run 2.8' does NOT make 2.8 the current version."""
        signals = extract_signals("We used to run 2.8 but that was ages ago.")
        assert signals.current_version is None
        assert "2.8" in signals.raw_versions_found

    def test_explicit_sdk_prefix_parsed(self):
        """'SDK 3.1' parses correctly."""
        signals = extract_signals("Running SDK 3.1 in production.")
        assert signals.current_version == "3.1"

    def test_v_prefix_parsed(self):
        """'v2.8' parses to '2.8'."""
        signals = extract_signals("We upgraded from v2.8 to v3.1 last week.")
        assert signals.current_version == "3.1"
        assert signals.previous_version == "2.8"

    def test_ambiguous_two_versions_no_upgrade_context(self):
        """Two non-historical versions without clear upgrade context → current=None."""
        signals = extract_signals("We tested on 2.8 and 3.1 both fail.")
        # Cannot reliably determine which is current.
        assert signals.current_version is None

    def test_structured_version_overrides_text(self):
        """Structured current_version wins over what could be extracted from text."""
        signals = extract_signals(
            "Issue started at 2.8.",
            structured_current_version="3.1",
        )
        assert signals.current_version == "3.1"

    def test_structured_previous_overrides_text(self):
        """Structured previous_version wins."""
        signals = extract_signals(
            "We upgraded from 2.5 to 3.1.",
            structured_previous_version="2.8",
        )
        assert signals.previous_version == "2.8"
        assert signals.current_version == "3.1"

    def test_no_version_in_text(self):
        """No version mentions → both versions remain None."""
        signals = extract_signals("Getting AUTH_401 on all requests.")
        assert signals.current_version is None
        assert signals.previous_version is None


# ---------------------------------------------------------------------------
# Signal extraction — technical terms
# ---------------------------------------------------------------------------

class TestTechnicalTermExtraction:

    def test_bearer_term_found(self):
        signals = extract_signals("The Authorization header uses Bearer token format.")
        assert "bearer" in signals.technical_terms or "token" in signals.technical_terms

    def test_webhook_term_found(self):
        signals = extract_signals("Our webhook endpoint keeps failing.")
        assert "webhook" in signals.technical_terms

    def test_oauth_term_found(self):
        signals = extract_signals("We use OAuth for authentication.")
        assert "oauth" in signals.technical_terms


# ---------------------------------------------------------------------------
# Incident normalization
# ---------------------------------------------------------------------------

class TestIncidentNormalization:

    def test_natural_language_incident_normalized(self):
        """A realistic NL incident produces a valid structured TroubleshootingIncident."""
        result = normalize_incident(
            "AUTH_401 started after upgrading the SDK from 2.8 to 3.1. "
            "We were using the ApiKey header format."
        )
        assert result.incident.current_version == "3.1"
        assert result.incident.previous_version == "2.8"
        assert "AUTH_401" in result.incident.error_codes

    def test_structured_fields_take_precedence(self):
        """Structured current_version overrides what could be extracted from text."""
        result = normalize_incident(
            "Issues started at version 2.8.",
            current_version="3.1",
        )
        assert result.incident.current_version == "3.1"

    def test_structured_error_codes_merged(self):
        """Structured error codes are included even if the description says something different."""
        result = normalize_incident(
            "Something broke with the webhook delivery.",
            error_codes=["RATE_429"],
        )
        assert "RATE_429" in result.incident.error_codes

    def test_normalize_from_dict_structured(self):
        """normalize_incident_from_dict works with a fully structured dict."""
        raw = {
            "description": "AUTH_401 after SDK upgrade.",
            "current_version": "3.1",
            "previous_version": "2.8",
            "error_codes": ["AUTH_401"],
            "product": "DevCore SDK",
        }
        result = normalize_incident_from_dict(raw)
        assert result.incident.current_version == "3.1"
        assert result.incident.product == "DevCore SDK"
        assert "AUTH_401" in result.incident.error_codes

    def test_normalize_minimal_description(self):
        """A minimal description-only incident normalizes without error."""
        result = normalize_incident("API calls are failing.")
        assert result.incident.current_version is None
        assert result.incident.error_codes == []

    def test_signals_bundled_with_incident(self):
        """NormalizedIncident bundles both incident and extracted signals."""
        result = normalize_incident("AUTH_401 after upgrading from 2.8 to 3.1.")
        assert result.signals.current_version == "3.1"
        assert result.signals.error_codes == result.incident.error_codes

    def test_signals_to_dict(self):
        """ExtractedSignals.to_dict() returns a plain serializable dict."""
        result = normalize_incident("AUTH_401 after upgrading from 2.8 to 3.1.")
        d = result.signals.to_dict()
        assert isinstance(d, dict)
        assert "error_codes" in d
        assert "current_version" in d


# ---------------------------------------------------------------------------
# Evaluation dataset
# ---------------------------------------------------------------------------

class TestEvaluationDataset:

    def test_eval_dataset_loads(self):
        """The default evaluation dataset loads without error."""
        dataset = load_eval_dataset()
        assert len(dataset.cases) > 0

    def test_eval_dataset_has_target_count(self):
        """Dataset contains approximately 30-40 cases."""
        dataset = load_eval_dataset()
        assert 30 <= len(dataset.cases) <= 40, f"Expected 30-40 cases, got {len(dataset.cases)}"

    def test_all_case_ids_unique(self):
        """All evaluation case IDs are unique."""
        dataset = load_eval_dataset()
        ids = [c.case_id for c in dataset.cases]
        assert len(ids) == len(set(ids))

    def test_all_case_classes_valid(self):
        """All case_class values are from the defined set."""
        from src.evaluation.dataset import VALID_CASE_CLASSES
        dataset = load_eval_dataset()
        for case in dataset.cases:
            assert case.case_class in VALID_CASE_CLASSES

    def test_expected_outcomes_are_valid(self):
        """All expected_outcome values are valid SystemOutcome enum values."""
        dataset = load_eval_dataset()
        valid_outcomes = {o.value for o in SystemOutcome}
        for case in dataset.cases:
            assert case.expected_outcome.value in valid_outcomes

    def test_answered_cases_have_completeness(self):
        """All ANSWERED cases have expected_completeness set."""
        from src.models.enums import AnswerCompleteness
        dataset = load_eval_dataset()
        for case in dataset.cases:
            if case.expected_outcome == SystemOutcome.ANSWERED:
                assert case.expected_completeness is not None, (
                    f"Case {case.case_id} is ANSWERED but has no completeness."
                )

    def test_non_answered_cases_have_no_completeness(self):
        """Non-ANSWERED cases do not have completeness set."""
        dataset = load_eval_dataset()
        for case in dataset.cases:
            if case.expected_outcome != SystemOutcome.ANSWERED:
                assert case.expected_completeness is None, (
                    f"Case {case.case_id} is {case.expected_outcome} but has completeness set."
                )

    def test_case_class_distribution(self):
        """Dataset contains cases from all 8 expected class categories."""
        from src.evaluation.dataset import VALID_CASE_CLASSES
        dataset = load_eval_dataset()
        found_classes = {c.case_class for c in dataset.cases}
        for cls in VALID_CASE_CLASSES:
            assert cls in found_classes, f"No cases found for class: {cls}"

    def test_corpus_integrity_check_passes(self):
        """All gold/forbidden doc IDs in the eval dataset reference real corpus docs."""
        docs = load_corpus()
        valid_ids = {d.doc_id for d in docs}
        # Should not raise.
        dataset = load_eval_dataset(valid_doc_ids=valid_ids)
        assert len(dataset.cases) > 0

    def test_corpus_integrity_check_catches_bad_reference(self):
        """Corpus integrity check raises if a case references a nonexistent doc."""
        docs = load_corpus()
        valid_ids = {d.doc_id for d in docs}
        # Inject a fake case with a bad gold doc reference.
        bad_case = {
            "case_id": "FAKE-001",
            "case_class": "straightforward",
            "description": "Fake case.",
            "incident": {"description": "Fake incident."},
            "expected_outcome": "ANSWERED",
            "expected_completeness": "FULL",
            "gold_doc_ids": ["NONEXISTENT-DOC-999"],
            "forbidden_doc_ids": [],
        }
        path = _write_temp_json([bad_case])
        with pytest.raises(EvalLoadError, match="NONEXISTENT-DOC-999"):
            load_eval_dataset(path, valid_doc_ids=valid_ids)

    def test_duplicate_case_id_rejected(self):
        """Evaluation dataset with duplicate case_ids is rejected."""
        duplicate_cases = [
            {
                "case_id": "DUP-001",
                "case_class": "straightforward",
                "description": "Case 1.",
                "incident": {"description": "Incident 1."},
                "expected_outcome": "ANSWERED",
                "expected_completeness": "FULL",
                "gold_doc_ids": [],
                "forbidden_doc_ids": [],
            },
            {
                "case_id": "DUP-001",  # duplicate
                "case_class": "straightforward",
                "description": "Case 2.",
                "incident": {"description": "Incident 2."},
                "expected_outcome": "INSUFFICIENT_EVIDENCE",
                "expected_completeness": None,
                "gold_doc_ids": [],
                "forbidden_doc_ids": [],
            },
        ]
        path = _write_temp_json(duplicate_cases)
        with pytest.raises((EvalLoadError, ValidationError)):
            load_eval_dataset(path)

    def test_invalid_case_class_rejected(self):
        """A case with an invalid case_class is rejected."""
        bad_case = [
            {
                "case_id": "BAD-001",
                "case_class": "made_up_class",
                "description": "Bad case.",
                "incident": {"description": "Incident."},
                "expected_outcome": "ANSWERED",
                "expected_completeness": "FULL",
                "gold_doc_ids": [],
                "forbidden_doc_ids": [],
            }
        ]
        path = _write_temp_json(bad_case)
        with pytest.raises((EvalLoadError, Exception)):
            load_eval_dataset(path)

    def test_version_conflict_cases_have_forbidden_docs(self):
        """Version-conflict cases should have at least one forbidden_doc_id (wrong-version trap)."""
        dataset = load_eval_dataset()
        version_conflict_cases = [c for c in dataset.cases if c.case_class == "version_conflict"]
        assert all(
            len(c.forbidden_doc_ids) > 0 for c in version_conflict_cases
        ), "All version_conflict cases should have forbidden (wrong-version) doc IDs."

    def test_needs_info_cases_have_no_gold_docs(self):
        """NEEDS_INFO cases should not have gold supporting documents."""
        dataset = load_eval_dataset()
        needs_info_cases = [c for c in dataset.cases if c.expected_outcome == SystemOutcome.NEEDS_INFO]
        for case in needs_info_cases:
            assert len(case.gold_doc_ids) == 0, (
                f"NEEDS_INFO case {case.case_id} should have no gold docs."
            )
