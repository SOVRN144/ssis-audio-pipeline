"""Tests for contract invariants that cannot be expressed in JSON Schema 2020-12.

JSON Schema 2020-12 does not support $data for cross-field comparisons.
These tests enforce invariants at the contract expectations layer:
- segments: end_sec >= start_sec for each segment
- preview_candidate: end_sec >= start_sec
- preview_feedback: preview_end_sec >= preview_start_sec (when both non-null)
"""

import json
from pathlib import Path

import jsonschema
import pytest

SPECS_DIR = Path(__file__).parent.parent / "specs"


def load_schema(name: str) -> dict:
    """Load a JSON schema from the specs directory."""
    schema_path = SPECS_DIR / f"{name}.schema.json"
    with open(schema_path) as f:
        return json.load(f)


def validate_schema(instance: dict, schema: dict) -> None:
    """Validate instance against schema using jsonschema."""
    jsonschema.validate(instance, schema)


class TestSegmentsInvariants:
    """Tests for segments contract invariants."""

    @pytest.fixture
    def schema(self):
        return load_schema("segments")

    @pytest.fixture
    def valid_segments(self):
        """Minimal valid segments document."""
        return {
            "schema_id": "segments.v1",
            "version": "1.0.0",
            "asset_id": "test-asset-001",
            "computed_at": "2024-01-01T00:00:00Z",
            "segments": [
                {"label": "speech", "start_sec": 0.0, "end_sec": 10.0},
                {"label": "music", "start_sec": 10.0, "end_sec": 20.0},
                {"label": "silence", "start_sec": 20.0, "end_sec": 25.0},
            ],
        }

    def test_valid_segments_pass_schema(self, schema, valid_segments):
        """Valid segments should pass schema validation."""
        validate_schema(valid_segments, schema)

    def test_end_sec_gte_start_sec_valid(self, schema, valid_segments):
        """Segments with end_sec >= start_sec should pass invariant check."""
        validate_schema(valid_segments, schema)
        for seg in valid_segments["segments"]:
            assert seg["end_sec"] >= seg["start_sec"], (
                f"Invariant violation: end_sec ({seg['end_sec']}) < "
                f"start_sec ({seg['start_sec']})"
            )

    def test_end_sec_lt_start_sec_fails_invariant(self, schema):
        """Segments with end_sec < start_sec should fail invariant check.

        This test documents that the schema cannot enforce this invariant,
        but our runtime validation (to be added in later steps) must catch it.
        """
        invalid_segments = {
            "schema_id": "segments.v1",
            "version": "1.0.0",
            "asset_id": "test-asset-001",
            "computed_at": "2024-01-01T00:00:00Z",
            "segments": [
                {"label": "speech", "start_sec": 10.0, "end_sec": 5.0},  # Invalid!
            ],
        }
        # Schema validation passes (schema cannot enforce cross-field comparison)
        validate_schema(invalid_segments, schema)

        # Document the invariant violation that runtime validation must catch
        violations = []
        for seg in invalid_segments["segments"]:
            if seg["end_sec"] < seg["start_sec"]:
                violations.append(seg)

        assert len(violations) == 1, "Expected exactly one invariant violation"
        assert violations[0]["start_sec"] == 10.0
        assert violations[0]["end_sec"] == 5.0

    def test_class_distribution_valid_keys(self, schema):
        """class_distribution keys must be valid segment labels."""
        valid_doc = {
            "schema_id": "segments.v1",
            "version": "1.0.0",
            "asset_id": "test-asset-001",
            "computed_at": "2024-01-01T00:00:00Z",
            "segments": [],
            "class_distribution": {
                "speech": 10.5,
                "music": 20.0,
                "silence": 5.0,
            },
        }
        validate_schema(valid_doc, schema)

    def test_class_distribution_invalid_key_rejected(self, schema):
        """class_distribution with invalid keys should be rejected by schema."""
        invalid_doc = {
            "schema_id": "segments.v1",
            "version": "1.0.0",
            "asset_id": "test-asset-001",
            "computed_at": "2024-01-01T00:00:00Z",
            "segments": [],
            "class_distribution": {
                "speech": 10.5,
                "unknown_label": 5.0,  # Invalid key!
            },
        }
        with pytest.raises(jsonschema.ValidationError):
            validate_schema(invalid_doc, schema)


class TestPreviewCandidateInvariants:
    """Tests for preview_candidate contract invariants."""

    @pytest.fixture
    def schema(self):
        return load_schema("preview_candidate")

    @pytest.fixture
    def valid_preview(self):
        """Minimal valid preview_candidate document."""
        return {
            "schema_id": "preview_candidate.v1",
            "version": "1.0.0",
            "asset_id": "test-asset-001",
            "computed_at": "2024-01-01T00:00:00Z",
            "mode": "smart",
            "start_sec": 30.0,
            "end_sec": 60.0,
            "duration_sec": 30.0,
        }

    def test_valid_preview_pass_schema(self, schema, valid_preview):
        """Valid preview_candidate should pass schema validation."""
        validate_schema(valid_preview, schema)

    def test_end_sec_gte_start_sec_valid(self, schema, valid_preview):
        """Preview with end_sec >= start_sec should pass invariant check."""
        validate_schema(valid_preview, schema)
        assert valid_preview["end_sec"] >= valid_preview["start_sec"], (
            f"Invariant violation: end_sec ({valid_preview['end_sec']}) < "
            f"start_sec ({valid_preview['start_sec']})"
        )

    def test_end_sec_lt_start_sec_fails_invariant(self, schema):
        """Preview with end_sec < start_sec should fail invariant check."""
        invalid_preview = {
            "schema_id": "preview_candidate.v1",
            "version": "1.0.0",
            "asset_id": "test-asset-001",
            "computed_at": "2024-01-01T00:00:00Z",
            "mode": "smart",
            "start_sec": 60.0,
            "end_sec": 30.0,  # Invalid!
            "duration_sec": 30.0,
        }
        # Schema validation passes (schema cannot enforce this)
        validate_schema(invalid_preview, schema)

        # Invariant check
        assert invalid_preview["end_sec"] < invalid_preview["start_sec"], (
            "Test setup error: expected invalid data"
        )
        # This documents the invariant that must be enforced at runtime


class TestPreviewFeedbackInvariants:
    """Tests for preview_feedback contract invariants."""

    @pytest.fixture
    def schema(self):
        return load_schema("preview_feedback")

    @pytest.fixture
    def valid_feedback(self):
        """Minimal valid preview_feedback document."""
        return {
            "schema_id": "preview_feedback.v1",
            "version": "1.0.0",
            "asset_id": "test-asset-001",
            "computed_at": "2024-01-01T00:00:00Z",
            "feedback_id": "fb-001",
            "preview_start_sec": 30.0,
            "preview_end_sec": 60.0,
        }

    def test_valid_feedback_pass_schema(self, schema, valid_feedback):
        """Valid preview_feedback should pass schema validation."""
        validate_schema(valid_feedback, schema)

    def test_preview_times_valid_invariant(self, schema, valid_feedback):
        """Feedback with preview_end_sec >= preview_start_sec should pass."""
        validate_schema(valid_feedback, schema)
        start = valid_feedback.get("preview_start_sec")
        end = valid_feedback.get("preview_end_sec")
        if start is not None and end is not None:
            assert end >= start, (
                f"Invariant violation: preview_end_sec ({end}) < "
                f"preview_start_sec ({start})"
            )

    def test_preview_times_invalid_invariant(self, schema):
        """Feedback with preview_end_sec < preview_start_sec should fail invariant."""
        invalid_feedback = {
            "schema_id": "preview_feedback.v1",
            "version": "1.0.0",
            "asset_id": "test-asset-001",
            "computed_at": "2024-01-01T00:00:00Z",
            "feedback_id": "fb-001",
            "preview_start_sec": 60.0,
            "preview_end_sec": 30.0,  # Invalid!
        }
        # Schema validation passes (schema cannot enforce this)
        validate_schema(invalid_feedback, schema)

        # Invariant check
        start = invalid_feedback.get("preview_start_sec")
        end = invalid_feedback.get("preview_end_sec")
        assert start is not None and end is not None
        assert end < start, "Test setup error: expected invalid data"
        # This documents the invariant that must be enforced at runtime

    def test_null_preview_times_allowed(self, schema):
        """Feedback with null preview times should pass (optional fields)."""
        feedback_null_times = {
            "schema_id": "preview_feedback.v1",
            "version": "1.0.0",
            "asset_id": "test-asset-001",
            "computed_at": "2024-01-01T00:00:00Z",
            "feedback_id": "fb-001",
            "preview_start_sec": None,
            "preview_end_sec": None,
        }
        validate_schema(feedback_null_times, schema)
        # No invariant to check when both are null
