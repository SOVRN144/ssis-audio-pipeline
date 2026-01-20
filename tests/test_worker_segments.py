"""Tests for the segments worker (services/worker_segments/run.py).

All tests mock inaSpeechSegmenter to run without the actual model in CI.
"""

from __future__ import annotations

import json
import sys
import tempfile
import wave
from pathlib import Path
from unittest import mock

import pytest

# Mock inaSpeechSegmenter before importing worker
mock_ina = mock.MagicMock()
sys.modules["inaSpeechSegmenter"] = mock_ina

from app.orchestrator import SEGMENTS_SCHEMA_ID, SEGMENTS_VERSION  # noqa: E402
from services.worker_segments.run import (  # noqa: E402
    CANONICAL_LABELS,
    CONFIDENCE_BASE,
    CONFIDENCE_PENALTY_SHORT_DURATION,
    CONFIDENCE_TYPE,
    MERGE_GAP_SEC,
    MIN_MUSIC_SEC,
    MIN_SILENCE_SEC,
    MIN_SPEECH_SEC,
    SEGMENT_SOURCE,
    SEGMENT_SOURCE_DERIVED,
    SegmentData,
    SegmentsErrorCode,
    _adjust_confidence,
    _clamp_segments,
    _compute_metrics,
    _fill_silence_gaps,
    _filter_by_min_duration,
    _map_label,
    _merge_adjacent_same_label,
    _postprocess_segments,
    _validate_invariants,
    _validate_schema,
    run_segments_worker,
)

# --- Test Constants ---

SAMPLE_RATE = 22050
CHANNELS = 1
SAMPWIDTH = 2


# --- Fixtures ---


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        audio_dir = data_dir / "audio"
        segments_dir = data_dir / "segments"
        audio_dir.mkdir(parents=True, exist_ok=True)
        segments_dir.mkdir(parents=True, exist_ok=True)
        yield tmpdir, data_dir, audio_dir, segments_dir


def _create_test_wav(path: Path, duration_sec: float = 10.0) -> None:
    """Create a test WAV file with silence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(duration_sec * SAMPLE_RATE)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPWIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"\x00" * num_samples * SAMPWIDTH)


def _make_segment(label: str, start: float, end: float) -> SegmentData:
    """Helper to create SegmentData."""
    return SegmentData(
        label=label,
        start_sec=start,
        end_sec=end,
        confidence=CONFIDENCE_BASE.get(label, 0.70),
        source=SEGMENT_SOURCE,
    )


# --- Test A: Contract Fields Present ---


class TestContractFieldsPresent:
    """Tests that output contains all required contract fields."""

    def test_output_has_required_fields(self, temp_data_dir, monkeypatch):
        """Test that output JSON has all required fields per schema."""
        tmpdir, data_dir, audio_dir, segments_dir = temp_data_dir
        asset_id = "test-contract-fields"
        asset_dir = audio_dir / asset_id
        _create_test_wav(asset_dir / "normalized.wav", duration_sec=10.0)

        # Patch paths
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.SEGMENTS_DIR", segments_dir)
        monkeypatch.setattr("app.config.SEGMENTS_DIR", segments_dir)

        # Mock segmenter to return simple result
        mock_segments = [
            ("speech", 0.0, 5.0),
            ("music", 5.5, 10.0),
        ]
        with mock.patch(
            "services.worker_segments.run._run_segmenter",
            return_value=mock_segments,
        ):
            result = run_segments_worker(asset_id)

        assert result.ok
        assert result.artifact_path is not None

        # Read and verify output
        output_path = Path(result.artifact_path)
        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        # Check required contract fields
        assert data["schema_id"] == SEGMENTS_SCHEMA_ID
        assert data["version"] == SEGMENTS_VERSION
        assert data["asset_id"] == asset_id
        assert "computed_at" in data
        assert data["confidence_type"] == CONFIDENCE_TYPE
        assert "segments" in data
        assert isinstance(data["segments"], list)

        # Check segment structure
        for seg in data["segments"]:
            assert "label" in seg
            assert "start_sec" in seg
            assert "end_sec" in seg
            assert "confidence" in seg
            assert "source" in seg


# --- Test B: Label Enum Enforcement ---


class TestLabelEnumEnforcement:
    """Tests that labels are restricted to canonical set."""

    def test_canonical_labels_only(self):
        """Test that CANONICAL_LABELS is exactly {speech, music, noise, silence}."""
        assert CANONICAL_LABELS == frozenset({"speech", "music", "noise", "silence"})

    def test_label_mapping_known_labels(self):
        """Test label mapping for known inaSpeechSegmenter labels."""
        assert _map_label("speech") == "speech"
        assert _map_label("music") == "music"
        assert _map_label("noEnergy") == "silence"
        assert _map_label("noise") == "noise"
        assert _map_label("male") == "speech"
        assert _map_label("female") == "speech"

    def test_label_mapping_unknown_defaults_to_noise(self):
        """Test that unknown labels default to noise."""
        assert _map_label("unknown") == "noise"
        assert _map_label("something_else") == "noise"

    def test_invariant_rejects_invalid_label(self):
        """Test that invariant validation rejects invalid labels."""
        segments = [
            SegmentData(
                label="invalid_label",
                start_sec=0.0,
                end_sec=5.0,
                confidence=0.8,
                source=SEGMENT_SOURCE,
            )
        ]
        valid, error_msg = _validate_invariants(segments)
        assert not valid
        assert "invalid label" in error_msg.lower()


# --- Test C: Schema Validation Gate ---


class TestSchemaValidationGate:
    """Tests that invalid segments fail schema validation and produce no artifact."""

    def test_missing_schema_id_fails(self):
        """Test that missing schema_id fails validation."""
        data = {
            "version": "1.0.0",
            "asset_id": "test",
            "computed_at": "2024-01-01T00:00:00Z",
            "confidence_type": CONFIDENCE_TYPE,
            "segments": [],
        }
        valid, error_msg = _validate_schema(data)
        assert not valid
        assert "schema_id" in error_msg.lower()

    def test_missing_confidence_type_fails(self):
        """Test that missing confidence_type fails validation."""
        data = {
            "schema_id": SEGMENTS_SCHEMA_ID,
            "version": "1.0.0",
            "asset_id": "test",
            "computed_at": "2024-01-01T00:00:00Z",
            "segments": [],
        }
        valid, error_msg = _validate_schema(data)
        assert not valid
        assert "confidence_type" in error_msg.lower()

    def test_invalid_label_in_segment_fails(self):
        """Test that invalid label in segment fails validation."""
        data = {
            "schema_id": SEGMENTS_SCHEMA_ID,
            "version": "1.0.0",
            "asset_id": "test",
            "computed_at": "2024-01-01T00:00:00Z",
            "confidence_type": CONFIDENCE_TYPE,
            "segments": [{"label": "invalid", "start_sec": 0.0, "end_sec": 5.0}],
        }
        valid, error_msg = _validate_schema(data)
        assert not valid

    def test_invalid_model_segments_handled_gracefully(self, temp_data_dir, monkeypatch):
        """Test that invalid model segments are dropped gracefully.

        With clamping, segments where start >= end are dropped rather than
        causing validation failure. This produces a valid result with derived
        silence filling the gaps.
        """
        tmpdir, data_dir, audio_dir, segments_dir = temp_data_dir
        asset_id = "test-invalid-segment"
        asset_dir = audio_dir / asset_id
        _create_test_wav(asset_dir / "normalized.wav", duration_sec=10.0)

        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.SEGMENTS_DIR", segments_dir)
        monkeypatch.setattr("app.config.SEGMENTS_DIR", segments_dir)

        # Mock segmenter to return segment with start >= end (will be dropped by clamping)
        mock_segments = [
            ("speech", 5.0, 3.0),  # Invalid: start > end, will be dropped
        ]

        with mock.patch(
            "services.worker_segments.run._run_segmenter",
            return_value=mock_segments,
        ):
            result = run_segments_worker(asset_id)

        # With clamping, invalid segments are dropped gracefully
        # The entire duration becomes derived silence
        assert result.ok

        # Verify artifact was written
        output_path = segments_dir / f"{asset_id}.segments.v1.json"
        assert output_path.exists()

        # Verify the output contains only derived silence
        with open(output_path) as f:
            data = json.load(f)
        assert len(data["segments"]) == 1
        assert data["segments"][0]["label"] == "silence"
        assert data["segments"][0]["source"] == "derived"


# --- Test D: Threshold Post-Processing ---


class TestThresholdPostProcessing:
    """Tests that post-processing thresholds are applied correctly."""

    def test_threshold_constants_match_spec(self):
        """Test that threshold constants match Research Pack spec."""
        assert MIN_SPEECH_SEC == 0.8
        assert MIN_MUSIC_SEC == 3.4
        assert MIN_SILENCE_SEC == 0.5
        assert MERGE_GAP_SEC == 0.3

    def test_short_speech_filtered(self):
        """Test that speech < 0.8s is filtered out."""
        segments = [
            _make_segment("speech", 0.0, 0.5),  # Too short
            _make_segment("speech", 1.0, 2.0),  # OK (1.0s)
        ]
        filtered = _filter_by_min_duration(segments)
        assert len(filtered) == 1
        assert filtered[0].start_sec == 1.0

    def test_short_music_filtered(self):
        """Test that music < 3.4s is filtered out."""
        segments = [
            _make_segment("music", 0.0, 3.0),  # Too short
            _make_segment("music", 4.0, 8.0),  # OK (4.0s)
        ]
        filtered = _filter_by_min_duration(segments)
        assert len(filtered) == 1
        assert filtered[0].start_sec == 4.0

    def test_short_silence_filtered(self):
        """Test that silence < 0.5s is filtered out."""
        segments = [
            _make_segment("silence", 0.0, 0.3),  # Too short
            _make_segment("silence", 1.0, 2.0),  # OK
        ]
        filtered = _filter_by_min_duration(segments)
        assert len(filtered) == 1
        assert filtered[0].start_sec == 1.0

    def test_noise_not_filtered(self):
        """Test that noise is never filtered by duration."""
        segments = [
            _make_segment("noise", 0.0, 0.1),  # Very short but kept
        ]
        filtered = _filter_by_min_duration(segments)
        assert len(filtered) == 1

    def test_merge_adjacent_same_label(self):
        """Test merging adjacent segments with same label within MERGE_GAP_SEC."""
        segments = [
            _make_segment("speech", 0.0, 1.0),
            _make_segment("speech", 1.2, 2.0),  # Gap = 0.2s < 0.3s, should merge
        ]
        merged = _merge_adjacent_same_label(segments)
        assert len(merged) == 1
        assert merged[0].start_sec == 0.0
        assert merged[0].end_sec == 2.0

    def test_no_merge_different_labels(self):
        """Test that different labels are not merged."""
        segments = [
            _make_segment("speech", 0.0, 1.0),
            _make_segment("music", 1.1, 2.0),  # Different label
        ]
        merged = _merge_adjacent_same_label(segments)
        assert len(merged) == 2

    def test_no_merge_large_gap(self):
        """Test that large gaps prevent merging."""
        segments = [
            _make_segment("speech", 0.0, 1.0),
            _make_segment("speech", 2.0, 3.0),  # Gap = 1.0s > 0.3s
        ]
        merged = _merge_adjacent_same_label(segments)
        assert len(merged) == 2

    def test_merge_propagates_derived_source(self):
        """Test that merging mixed-source segments propagates source='derived'.

        When merging same-label segments, if either side has source='derived',
        the merged segment should have source='derived' to avoid mislabeling
        derived material as model-emitted.
        """
        # First segment is model-emitted, second is derived
        seg1 = SegmentData(
            label="speech",
            start_sec=0.0,
            end_sec=1.0,
            confidence=CONFIDENCE_BASE["speech"],
            source=SEGMENT_SOURCE,  # "inaspeechsegmenter"
        )
        seg2 = SegmentData(
            label="speech",
            start_sec=1.2,  # Gap = 0.2s <= 0.3s, will merge
            end_sec=2.0,
            confidence=CONFIDENCE_BASE["speech"],
            source=SEGMENT_SOURCE_DERIVED,  # "derived"
        )
        merged = _merge_adjacent_same_label([seg1, seg2])
        assert len(merged) == 1
        assert merged[0].source == "derived"

        # Reverse order: first derived, second model-emitted
        seg3 = SegmentData(
            label="music",
            start_sec=0.0,
            end_sec=1.0,
            confidence=CONFIDENCE_BASE["music"],
            source=SEGMENT_SOURCE_DERIVED,
        )
        seg4 = SegmentData(
            label="music",
            start_sec=1.1,  # Gap = 0.1s <= 0.3s, will merge
            end_sec=2.0,
            confidence=CONFIDENCE_BASE["music"],
            source=SEGMENT_SOURCE,
        )
        merged2 = _merge_adjacent_same_label([seg3, seg4])
        assert len(merged2) == 1
        assert merged2[0].source == "derived"

        # Both model-emitted: should keep model source
        seg5 = SegmentData(
            label="silence",
            start_sec=0.0,
            end_sec=1.0,
            confidence=CONFIDENCE_BASE["silence"],
            source=SEGMENT_SOURCE,
        )
        seg6 = SegmentData(
            label="silence",
            start_sec=1.1,
            end_sec=2.0,
            confidence=CONFIDENCE_BASE["silence"],
            source=SEGMENT_SOURCE,
        )
        merged3 = _merge_adjacent_same_label([seg5, seg6])
        assert len(merged3) == 1
        assert merged3[0].source == SEGMENT_SOURCE


# --- Test E: flip_rate Computation ---


class TestFlipRateComputation:
    """Tests for flip_rate metric correctness."""

    def test_flip_rate_no_transitions(self):
        """Test flip_rate with no label transitions."""
        segments = [
            _make_segment("speech", 0.0, 5.0),
            _make_segment("speech", 5.0, 10.0),
        ]
        metrics = _compute_metrics(segments, total_duration=10.0, segmentation_time_ms=100)
        assert metrics["flip_rate"] == 0.0

    def test_flip_rate_one_transition(self):
        """Test flip_rate with one transition."""
        segments = [
            _make_segment("speech", 0.0, 5.0),
            _make_segment("music", 5.0, 10.0),
        ]
        metrics = _compute_metrics(segments, total_duration=10.0, segmentation_time_ms=100)
        # 1 transition / 10.0 seconds = 0.1
        assert metrics["flip_rate"] == 0.1

    def test_flip_rate_multiple_transitions(self):
        """Test flip_rate with multiple transitions."""
        segments = [
            _make_segment("speech", 0.0, 2.5),
            _make_segment("music", 2.5, 5.0),
            _make_segment("silence", 5.0, 7.5),
            _make_segment("noise", 7.5, 10.0),
        ]
        metrics = _compute_metrics(segments, total_duration=10.0, segmentation_time_ms=100)
        # 3 transitions / 10.0 seconds = 0.3
        assert metrics["flip_rate"] == 0.3

    def test_flip_rate_zero_duration(self):
        """Test flip_rate with zero duration (edge case)."""
        segments = []
        metrics = _compute_metrics(segments, total_duration=0.0, segmentation_time_ms=100)
        assert metrics["flip_rate"] == 0.0

    def test_flip_rate_definition(self):
        """Test flip_rate = transitions / duration_sec."""
        # More explicit test
        segments = [
            _make_segment("speech", 0.0, 10.0),
            _make_segment("music", 10.0, 30.0),
            _make_segment("speech", 30.0, 40.0),
            _make_segment("silence", 40.0, 50.0),
        ]
        metrics = _compute_metrics(segments, total_duration=50.0, segmentation_time_ms=100)
        # 3 transitions / 50.0 seconds = 0.06
        assert metrics["flip_rate"] == 3 / 50.0


# --- Test F: Atomic Publish Boundary ---


class TestAtomicPublishBoundary:
    """Tests that .tmp files are not treated as final artifacts."""

    def test_tmp_file_not_final(self, temp_data_dir, monkeypatch):
        """Test that pre-existing .tmp file does not prevent processing."""
        tmpdir, data_dir, audio_dir, segments_dir = temp_data_dir
        asset_id = "test-atomic-tmp"
        asset_dir = audio_dir / asset_id
        _create_test_wav(asset_dir / "normalized.wav", duration_sec=10.0)

        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.SEGMENTS_DIR", segments_dir)
        monkeypatch.setattr("app.config.SEGMENTS_DIR", segments_dir)

        # Create orphan .tmp file
        tmp_path = segments_dir / f"{asset_id}.segments.v1.json.tmp"
        tmp_path.write_text("orphan temp data")

        # Mock segmenter
        mock_segments = [("speech", 0.0, 10.0)]

        with mock.patch(
            "services.worker_segments.run._run_segmenter",
            return_value=mock_segments,
        ):
            result = run_segments_worker(asset_id)

        assert result.ok

        # Final should exist
        final_path = segments_dir / f"{asset_id}.segments.v1.json"
        assert final_path.exists()

        # Tmp should be gone (overwritten by atomic_write)
        # Note: atomic_io overwrites tmp during write
        # The .tmp file is replaced during atomic rename

    def test_existing_final_returns_success(self, temp_data_dir, monkeypatch):
        """Test that existing final artifact returns success without re-processing."""
        tmpdir, data_dir, audio_dir, segments_dir = temp_data_dir
        asset_id = "test-idempotent"
        asset_dir = audio_dir / asset_id
        _create_test_wav(asset_dir / "normalized.wav", duration_sec=10.0)

        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.SEGMENTS_DIR", segments_dir)
        monkeypatch.setattr("app.config.SEGMENTS_DIR", segments_dir)

        # Pre-create the final output
        output_path = segments_dir / f"{asset_id}.segments.v1.json"
        output_path.write_text(
            json.dumps(
                {
                    "schema_id": SEGMENTS_SCHEMA_ID,
                    "version": SEGMENTS_VERSION,
                    "asset_id": asset_id,
                    "computed_at": "2024-01-01T00:00:00Z",
                    "confidence_type": CONFIDENCE_TYPE,
                    "segments": [],
                }
            )
        )

        segmenter_called = [False]

        def track_segmenter(*args, **kwargs):
            segmenter_called[0] = True
            return []

        with mock.patch(
            "services.worker_segments.run._run_segmenter",
            side_effect=track_segmenter,
        ):
            result = run_segments_worker(asset_id)

        assert result.ok
        assert not segmenter_called[0]
        assert "already exists" in result.message.lower()


# --- Test G: Error Mapping ---


class TestErrorMapping:
    """Tests for error code mapping."""

    def test_memory_error_maps_to_model_oom(self, temp_data_dir, monkeypatch):
        """Test that MemoryError maps to MODEL_OOM."""
        tmpdir, data_dir, audio_dir, segments_dir = temp_data_dir
        asset_id = "test-oom"
        asset_dir = audio_dir / asset_id
        _create_test_wav(asset_dir / "normalized.wav", duration_sec=10.0)

        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.SEGMENTS_DIR", segments_dir)
        monkeypatch.setattr("app.config.SEGMENTS_DIR", segments_dir)

        def raise_memory_error(*args, **kwargs):
            raise MemoryError("Out of memory")

        with mock.patch(
            "services.worker_segments.run._run_segmenter",
            side_effect=raise_memory_error,
        ):
            result = run_segments_worker(asset_id)

        assert not result.ok
        assert result.error_code == SegmentsErrorCode.MODEL_OOM.value

    def test_generic_exception_maps_to_segmentation_failed(self, temp_data_dir, monkeypatch):
        """Test that generic exception maps to SEGMENTATION_FAILED."""
        tmpdir, data_dir, audio_dir, segments_dir = temp_data_dir
        asset_id = "test-generic-error"
        asset_dir = audio_dir / asset_id
        _create_test_wav(asset_dir / "normalized.wav", duration_sec=10.0)

        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.SEGMENTS_DIR", segments_dir)
        monkeypatch.setattr("app.config.SEGMENTS_DIR", segments_dir)

        def raise_error(*args, **kwargs):
            raise RuntimeError("Something went wrong")

        with mock.patch(
            "services.worker_segments.run._run_segmenter",
            side_effect=raise_error,
        ):
            result = run_segments_worker(asset_id)

        assert not result.ok
        assert result.error_code == SegmentsErrorCode.SEGMENTATION_FAILED.value

    def test_missing_input_returns_input_not_found(self, temp_data_dir, monkeypatch):
        """Test that missing input file returns INPUT_NOT_FOUND."""
        tmpdir, data_dir, audio_dir, segments_dir = temp_data_dir
        asset_id = "test-missing-input"
        # Do NOT create normalized.wav

        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.SEGMENTS_DIR", segments_dir)
        monkeypatch.setattr("app.config.SEGMENTS_DIR", segments_dir)

        result = run_segments_worker(asset_id)

        assert not result.ok
        assert result.error_code == SegmentsErrorCode.INPUT_NOT_FOUND.value

    def test_oom_like_error_string_maps_to_model_oom(self, temp_data_dir, monkeypatch):
        """Test that OOM-like error strings map to MODEL_OOM."""
        tmpdir, data_dir, audio_dir, segments_dir = temp_data_dir
        asset_id = "test-oom-string"
        asset_dir = audio_dir / asset_id
        _create_test_wav(asset_dir / "normalized.wav", duration_sec=10.0)

        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.SEGMENTS_DIR", segments_dir)
        monkeypatch.setattr("app.config.SEGMENTS_DIR", segments_dir)

        def raise_oom_like(*args, **kwargs):
            raise RuntimeError("Failed to allocate memory for tensor")

        with mock.patch(
            "services.worker_segments.run._run_segmenter",
            side_effect=raise_oom_like,
        ):
            result = run_segments_worker(asset_id)

        assert not result.ok
        assert result.error_code == SegmentsErrorCode.MODEL_OOM.value


# --- Test: Silence Gap-Fill ---


class TestSilenceGapFill:
    """Tests for silence gap-fill functionality."""

    def test_gap_fill_leading_silence(self):
        """Test that leading gap is filled with silence."""
        segments = [
            _make_segment("speech", 2.0, 5.0),  # Starts at 2.0, leaving 2.0s gap
        ]
        filled = _fill_silence_gaps(segments, total_duration=5.0)
        assert len(filled) == 2
        assert filled[0].label == "silence"
        assert filled[0].start_sec == 0.0
        assert filled[0].end_sec == 2.0

    def test_gap_fill_trailing_silence(self):
        """Test that trailing gap is filled with silence."""
        segments = [
            _make_segment("speech", 0.0, 5.0),
        ]
        filled = _fill_silence_gaps(segments, total_duration=10.0)
        assert len(filled) == 2
        assert filled[1].label == "silence"
        assert filled[1].start_sec == 5.0
        assert filled[1].end_sec == 10.0

    def test_gap_fill_between_segments(self):
        """Test that gaps between segments are filled."""
        segments = [
            _make_segment("speech", 0.0, 3.0),
            _make_segment("music", 5.0, 10.0),  # Gap of 2.0s
        ]
        filled = _fill_silence_gaps(segments, total_duration=10.0)
        assert len(filled) == 3
        assert filled[1].label == "silence"
        assert filled[1].start_sec == 3.0
        assert filled[1].end_sec == 5.0

    def test_small_gap_not_filled(self):
        """Test that gaps < MIN_SILENCE_SEC are not filled."""
        segments = [
            _make_segment("speech", 0.0, 3.0),
            _make_segment("music", 3.3, 10.0),  # Gap of 0.3s < 0.5s
        ]
        filled = _fill_silence_gaps(segments, total_duration=10.0)
        assert len(filled) == 2  # No silence inserted


# --- Test: Metrics ---


class TestMetrics:
    """Tests for metrics computation."""

    def test_metrics_contain_required_keys(self):
        """Test that metrics contain all required keys."""
        segments = [
            _make_segment("speech", 0.0, 5.0),
            _make_segment("music", 5.0, 10.0),
        ]
        metrics = _compute_metrics(segments, total_duration=10.0, segmentation_time_ms=500)

        required_keys = [
            "segment_count",
            "speech_sec",
            "music_sec",
            "silence_sec",
            "noise_sec",
            "class_distribution",
            "flip_rate",
            "segmentation_time_ms",
            "source",
            "schema_id",
            "version",
        ]
        for key in required_keys:
            assert key in metrics

    def test_class_distribution_sums_to_one(self):
        """Test that class_distribution values sum to approximately 1."""
        segments = [
            _make_segment("speech", 0.0, 5.0),
            _make_segment("music", 5.0, 7.0),
            _make_segment("silence", 7.0, 9.0),
            _make_segment("noise", 9.0, 10.0),
        ]
        metrics = _compute_metrics(segments, total_duration=10.0, segmentation_time_ms=100)
        total = sum(metrics["class_distribution"].values())
        assert abs(total - 1.0) < 0.001

    def test_per_class_duration(self):
        """Test per-class duration computation."""
        segments = [
            _make_segment("speech", 0.0, 5.0),
            _make_segment("music", 5.0, 8.0),
        ]
        metrics = _compute_metrics(segments, total_duration=10.0, segmentation_time_ms=100)
        assert metrics["speech_sec"] == 5.0
        assert metrics["music_sec"] == 3.0
        assert metrics["silence_sec"] == 0.0
        assert metrics["noise_sec"] == 0.0


# --- Test: Invariants Validation ---


class TestInvariantsValidation:
    """Tests for invariants validation."""

    def test_nan_start_sec_fails(self):
        """Test that NaN start_sec fails validation."""
        segments = [
            SegmentData(
                label="speech",
                start_sec=float("nan"),
                end_sec=5.0,
                confidence=0.85,
                source=SEGMENT_SOURCE,
            )
        ]
        valid, error_msg = _validate_invariants(segments)
        assert not valid
        assert "nan" in error_msg.lower()

    def test_inf_end_sec_fails(self):
        """Test that Inf end_sec fails validation."""
        segments = [
            SegmentData(
                label="speech",
                start_sec=0.0,
                end_sec=float("inf"),
                confidence=0.85,
                source=SEGMENT_SOURCE,
            )
        ]
        valid, error_msg = _validate_invariants(segments)
        assert not valid
        assert "nan/inf" in error_msg.lower()

    def test_start_greater_than_end_fails(self):
        """Test that start_sec >= end_sec fails."""
        segments = [
            _make_segment("speech", 5.0, 3.0),  # Invalid
        ]
        valid, error_msg = _validate_invariants(segments)
        assert not valid
        assert "start_sec >= end_sec" in error_msg

    def test_overlapping_segments_fails(self):
        """Test that overlapping segments fail validation."""
        segments = [
            _make_segment("speech", 0.0, 5.0),
            _make_segment("music", 4.0, 8.0),  # Overlaps with first
        ]
        valid, error_msg = _validate_invariants(segments)
        assert not valid
        assert "overlaps" in error_msg.lower()

    def test_valid_segments_pass(self):
        """Test that valid segments pass validation."""
        segments = [
            _make_segment("speech", 0.0, 5.0),
            _make_segment("music", 5.0, 10.0),
        ]
        valid, error_msg = _validate_invariants(segments)
        assert valid
        assert error_msg is None


# --- Test: Full Post-Processing Pipeline ---


class TestFullPostProcessing:
    """Tests for complete post-processing pipeline."""

    def test_postprocess_sorts_segments(self):
        """Test that post-processing sorts segments by start_sec."""
        segments = [
            _make_segment("music", 5.0, 10.0),
            _make_segment("speech", 0.0, 4.0),  # Out of order
        ]
        processed = _postprocess_segments(segments, total_duration=10.0)
        assert processed[0].start_sec < processed[1].start_sec

    def test_postprocess_fills_and_filters_and_merges(self):
        """Test that post-processing applies all transformations."""
        # Use values that avoid floating point precision issues
        segments = [
            _make_segment("speech", 0.0, 0.5),  # Too short (0.5s < 0.8s), will be filtered
            _make_segment("speech", 1.0, 2.0),  # OK (1.0s >= 0.8s)
            _make_segment("speech", 2.25, 3.25),  # Gap 0.25s < 0.3s, will merge with previous
        ]
        processed = _postprocess_segments(segments, total_duration=10.0)

        # After fill silence gaps:
        # - 0.0-0.5 speech, 0.5-1.0 silence (gap=0.5s), 1.0-2.0 speech,
        #   2.25-3.25 speech, 3.25-10.0 silence
        # After filter:
        # - 0.0-0.5 speech filtered (0.5s < 0.8s)
        # - 0.5-1.0 silence kept (0.5s = 0.5s)
        # - 1.0-2.0 speech kept (1.0s >= 0.8s)
        # - 2.25-3.25 speech kept (1.0s >= 0.8s)
        # - 3.25-10.0 silence kept
        # After merge: speech 1.0-2.0 and 2.25-3.25 merge (gap=0.25s <= 0.3s)
        # Result: silence 0.5-1.0, speech 1.0-3.25, silence 3.25-10.0

        speech_segs = [s for s in processed if s.label == "speech"]
        silence_segs = [s for s in processed if s.label == "silence"]

        assert len(speech_segs) == 1
        assert speech_segs[0].start_sec == 1.0
        assert speech_segs[0].end_sec == 3.25

        assert len(silence_segs) >= 1  # At least trailing silence


# --- Test: Segment Source Labels ---


class TestSegmentSourceLabels:
    """Tests for correct source labeling (Gap 2)."""

    def test_derived_silence_has_source_derived(self):
        """Test that gap-filled silence segments have source='derived'."""
        segments = [
            _make_segment("speech", 2.0, 5.0),  # Leaves gap at start
        ]
        filled = _fill_silence_gaps(segments, total_duration=10.0)

        # Should have leading silence with source="derived"
        leading_silence = [s for s in filled if s.label == "silence" and s.start_sec == 0.0]
        assert len(leading_silence) == 1
        assert leading_silence[0].source == SEGMENT_SOURCE_DERIVED

        # Trailing silence should also have source="derived"
        trailing_silence = [s for s in filled if s.label == "silence" and s.end_sec == 10.0]
        assert len(trailing_silence) == 1
        assert trailing_silence[0].source == SEGMENT_SOURCE_DERIVED

    def test_model_emitted_segments_keep_source(self):
        """Test that model-emitted segments retain source='inaspeechsegmenter'."""
        segments = [
            _make_segment("speech", 2.0, 5.0),
        ]
        filled = _fill_silence_gaps(segments, total_duration=10.0)

        # The original speech segment should keep its source
        speech_segs = [s for s in filled if s.label == "speech"]
        assert len(speech_segs) == 1
        assert speech_segs[0].source == SEGMENT_SOURCE

    def test_empty_segments_derive_silence(self):
        """Test that empty input derives full silence with correct source."""
        filled = _fill_silence_gaps([], total_duration=5.0)
        assert len(filled) == 1
        assert filled[0].label == "silence"
        assert filled[0].source == SEGMENT_SOURCE_DERIVED

    def test_very_short_audio_returns_empty_segments(self):
        """Audio < MIN_SILENCE_SEC should return empty segments."""
        segments = _fill_silence_gaps([], total_duration=0.3)
        assert segments == []


# --- Test: Schema Allows Source Field ---


class TestSchemaAllowsSource:
    """Tests for schema allowing segment.source (Gap 3)."""

    def test_schema_validation_passes_with_source(self):
        """Test that schema validation passes when source field is present."""
        data = {
            "schema_id": SEGMENTS_SCHEMA_ID,
            "version": "1.0.0",
            "asset_id": "test",
            "computed_at": "2024-01-01T00:00:00Z",
            "confidence_type": CONFIDENCE_TYPE,
            "segments": [
                {
                    "label": "speech",
                    "start_sec": 0.0,
                    "end_sec": 5.0,
                    "confidence": 0.85,
                    "source": "inaspeechsegmenter",
                },
                {
                    "label": "silence",
                    "start_sec": 5.0,
                    "end_sec": 10.0,
                    "confidence": 0.95,
                    "source": "derived",
                },
            ],
        }
        valid, error_msg = _validate_schema(data)
        assert valid, f"Schema validation failed: {error_msg}"


# --- Test: Clamping and Bounds ---


class TestClampingAndBounds:
    """Tests for boundary clamping (Gap 4)."""

    def test_clamp_negative_start(self):
        """Test that negative start_sec is clamped to 0."""
        segments = [
            SegmentData(
                label="speech",
                start_sec=-1.0,
                end_sec=5.0,
                confidence=0.85,
                source=SEGMENT_SOURCE,
            )
        ]
        clamped = _clamp_segments(segments, total_duration=10.0)
        assert len(clamped) == 1
        assert clamped[0].start_sec == 0.0
        assert clamped[0].end_sec == 5.0

    def test_clamp_end_beyond_duration(self):
        """Test that end_sec beyond duration is clamped."""
        segments = [
            SegmentData(
                label="speech",
                start_sec=5.0,
                end_sec=15.0,  # Beyond 10.0s duration
                confidence=0.85,
                source=SEGMENT_SOURCE,
            )
        ]
        clamped = _clamp_segments(segments, total_duration=10.0)
        assert len(clamped) == 1
        assert clamped[0].start_sec == 5.0
        assert clamped[0].end_sec == 10.0

    def test_clamp_drops_invalid_after_clamping(self):
        """Test that segments with end <= start after clamping are dropped."""
        segments = [
            SegmentData(
                label="speech",
                start_sec=12.0,  # Beyond duration
                end_sec=15.0,  # Also beyond duration
                confidence=0.85,
                source=SEGMENT_SOURCE,
            )
        ]
        clamped = _clamp_segments(segments, total_duration=10.0)
        # After clamping: start=10, end=10 -> invalid, dropped
        assert len(clamped) == 0

    def test_invariants_check_bounds(self):
        """Test that invariants validation checks bounds when duration provided."""
        segments = [
            SegmentData(
                label="speech",
                start_sec=0.0,
                end_sec=15.0,  # Beyond 10s
                confidence=0.85,
                source=SEGMENT_SOURCE,
            )
        ]
        valid, error_msg = _validate_invariants(segments, total_duration=10.0)
        assert not valid
        assert "end_sec > total_duration" in error_msg

    def test_full_pipeline_clamps_out_of_range(self, temp_data_dir, monkeypatch):
        """Test that full pipeline clamps out-of-range model output."""
        tmpdir, data_dir, audio_dir, segments_dir = temp_data_dir
        asset_id = "test-clamp-pipeline"
        asset_dir = audio_dir / asset_id
        _create_test_wav(asset_dir / "normalized.wav", duration_sec=10.0)

        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.SEGMENTS_DIR", segments_dir)
        monkeypatch.setattr("app.config.SEGMENTS_DIR", segments_dir)

        # Mock segmenter returning out-of-range segments
        mock_segments = [
            ("speech", -2.0, 5.0),  # Negative start
            ("music", 8.0, 15.0),  # End beyond duration
        ]

        with mock.patch(
            "services.worker_segments.run._run_segmenter",
            return_value=mock_segments,
        ):
            result = run_segments_worker(asset_id)

        assert result.ok

        # Read and verify output
        output_path = Path(result.artifact_path)
        with open(output_path) as f:
            data = json.load(f)

        # All segments should be in bounds [0, 10]
        for seg in data["segments"]:
            assert seg["start_sec"] >= 0.0
            assert seg["end_sec"] <= 10.0
            assert seg["start_sec"] < seg["end_sec"]


# --- Test: Confidence Adjustments ---


class TestConfidenceAdjustments:
    """Tests for confidence adjustments (Gap 5)."""

    def test_short_duration_penalty(self):
        """Test that segments near min duration threshold get reduced confidence."""
        # Speech just at threshold (1.0s is exactly MIN_SPEECH_SEC * 1.2 = 0.96s)
        # Actually 0.8 * 1.2 = 0.96, so 0.9s < 0.96 triggers penalty
        segments = [
            SegmentData(
                label="speech",
                start_sec=0.0,
                end_sec=0.9,  # Just above 0.8s min, but < 0.8*1.2=0.96s
                confidence=CONFIDENCE_BASE["speech"],
                source=SEGMENT_SOURCE,
            )
        ]
        adjusted = _adjust_confidence(segments)
        assert adjusted[0].confidence < CONFIDENCE_BASE["speech"]
        expected = CONFIDENCE_BASE["speech"] - CONFIDENCE_PENALTY_SHORT_DURATION
        assert abs(adjusted[0].confidence - expected) < 0.001

    def test_derived_segment_penalty(self):
        """Test that derived segments get reduced confidence."""
        segments = [
            SegmentData(
                label="silence",
                start_sec=0.0,
                end_sec=5.0,  # Well above min duration
                confidence=CONFIDENCE_BASE["silence"],
                source=SEGMENT_SOURCE_DERIVED,
            )
        ]
        adjusted = _adjust_confidence(segments)
        # Derived penalty applied
        assert adjusted[0].confidence < CONFIDENCE_BASE["silence"]
        expected = CONFIDENCE_BASE["silence"] - CONFIDENCE_PENALTY_SHORT_DURATION
        assert abs(adjusted[0].confidence - expected) < 0.001

    def test_confidence_clamped_to_bounds(self):
        """Test that confidence is clamped to [0, 1]."""
        # Very low base confidence + penalties should not go below 0
        segments = [
            SegmentData(
                label="noise",
                start_sec=0.0,
                end_sec=0.1,  # Very short
                confidence=0.05,  # Already low
                source=SEGMENT_SOURCE_DERIVED,
            )
        ]
        adjusted = _adjust_confidence(segments)
        assert adjusted[0].confidence >= 0.0
        assert adjusted[0].confidence <= 1.0

    def test_no_penalty_for_long_model_segments(self):
        """Test that long model-emitted segments keep full confidence."""
        segments = [
            SegmentData(
                label="speech",
                start_sec=0.0,
                end_sec=5.0,  # Well above 0.8*1.2=0.96s threshold
                confidence=CONFIDENCE_BASE["speech"],
                source=SEGMENT_SOURCE,
            )
        ]
        adjusted = _adjust_confidence(segments)
        # No penalty applied
        assert adjusted[0].confidence == CONFIDENCE_BASE["speech"]

    def test_confidence_within_bounds_in_full_output(self, temp_data_dir, monkeypatch):
        """Test that all confidence values in final output are in [0, 1]."""
        tmpdir, data_dir, audio_dir, segments_dir = temp_data_dir
        asset_id = "test-conf-bounds"
        asset_dir = audio_dir / asset_id
        _create_test_wav(asset_dir / "normalized.wav", duration_sec=10.0)

        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.SEGMENTS_DIR", segments_dir)
        monkeypatch.setattr("app.config.SEGMENTS_DIR", segments_dir)

        mock_segments = [
            ("speech", 0.0, 0.9),  # Near threshold
            ("music", 4.0, 10.0),  # Long
        ]

        with mock.patch(
            "services.worker_segments.run._run_segmenter",
            return_value=mock_segments,
        ):
            result = run_segments_worker(asset_id)

        assert result.ok

        output_path = Path(result.artifact_path)
        with open(output_path) as f:
            data = json.load(f)

        for seg in data["segments"]:
            assert 0.0 <= seg["confidence"] <= 1.0
