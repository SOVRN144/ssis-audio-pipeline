"""SSIS Audio Pipeline - Segments Worker.

Computes speech/music/noise/silence segments from normalized WAV using inaSpeechSegmenter.

Stage: segments (STAGE_SEGMENTS)
Input: data/audio/{asset_id}/normalized.wav
Output: data/segments/{asset_id}.segments.v1.json

Backend: inaSpeechSegmenter (CPU)

Resilience features:
- Label mapping to canonical set: {"speech", "music", "noise", "silence"}
- Silence derivation via gap-fill (gaps >= MIN_SILENCE_SEC become silence)
- Heuristic confidence calculation
- Post-processing with Research Pack thresholds
- Schema validation before writing
- Invariants validation (sorted, non-overlapping, no NaN/Inf)
- Atomic publish for final JSON

Error codes (Blueprint section 8):
- INPUT_NOT_FOUND: normalized.wav does not exist
- MODEL_OOM: Out of memory during inference
- SEGMENTATION_FAILED: General segmentation failure
- SEGMENTS_INVALID: Schema or invariants validation failed
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from app.orchestrator import (
    ARTIFACT_TYPE_SEGMENTS_V1,
    SEGMENTS_SCHEMA_ID,
    SEGMENTS_VERSION,
    STAGE_SEGMENTS,
)
from app.utils.atomic_io import atomic_write_text
from app.utils.paths import audio_normalized_path, segments_json_path

logger = logging.getLogger(__name__)

# Stage identifier for this worker (imported from orchestrator for consistency)
WORKER_STAGE = STAGE_SEGMENTS


# --- Error Codes ---


class SegmentsErrorCode(str, Enum):
    """Error codes for segments stage per Blueprint section 8."""

    INPUT_NOT_FOUND = "INPUT_NOT_FOUND"
    MODEL_OOM = "MODEL_OOM"
    SEGMENTATION_FAILED = "SEGMENTATION_FAILED"
    SEGMENTS_INVALID = "SEGMENTS_INVALID"


# --- Result Types ---


@dataclass
class SegmentData:
    """A single segment region."""

    label: str
    start_sec: float
    end_sec: float
    confidence: float
    source: str


@dataclass
class SegmentsResult:
    """Result of segments worker execution."""

    ok: bool
    error_code: str | None = None
    message: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_path: str | None = None
    artifact_type: str | None = None
    schema_version: str | None = None
    segments: list[SegmentData] = field(default_factory=list)


# --- Research Pack Thresholds (Blueprint section 11) ---

# Minimum durations in seconds for post-processing
MIN_SPEECH_SEC = 0.8
MIN_MUSIC_SEC = 3.4
MIN_SILENCE_SEC = 0.5

# Merge gap threshold in seconds
MERGE_GAP_SEC = 0.3

# Confidence adjustment penalties (deterministic, small)
CONFIDENCE_PENALTY_SHORT_DURATION = 0.05  # Penalty for segments near min duration threshold
CONFIDENCE_PENALTY_ENERGY_MISMATCH = 0.10  # Penalty for energy/label contradiction

# --- Heuristic Confidence Base Values ---

CONFIDENCE_BASE = {
    "speech": 0.85,
    "music": 0.80,
    "noise": 0.70,
    "silence": 0.95,
}

# Label for confidence type at top-level
CONFIDENCE_TYPE = "heuristic_v1"

# Source identifier for model-emitted segments
SEGMENT_SOURCE = "inaspeechsegmenter"

# Source identifier for derived/gap-filled segments
SEGMENT_SOURCE_DERIVED = "derived"

# Canonical label set
CANONICAL_LABELS = frozenset({"speech", "music", "noise", "silence"})

# inaSpeechSegmenter label mapping to canonical labels
# Note: inaSpeechSegmenter outputs 'speech', 'music', 'noEnergy' (silence), and
# sometimes 'noise' for ambiguous regions. Keys must be lowercase since we
# normalize input with .lower() before lookup.
INA_LABEL_MAP = {
    "speech": "speech",
    "music": "music",
    "noenergy": "silence",  # inaSpeechSegmenter's label for silence
    "noise": "noise",
    # Some models may output these variants
    "male": "speech",
    "female": "speech",
}


# --- Schema Loading ---


def _load_schema() -> dict:
    """Load the segments JSON schema.

    Returns:
        Parsed JSON schema dict.

    Raises:
        FileNotFoundError: If schema file not found.
    """
    schema_path = Path(__file__).parent.parent.parent / "specs" / "segments.schema.json"
    with open(schema_path) as f:
        return json.load(f)


# --- Label Mapping ---


def _map_label(raw_label: str) -> str:
    """Map inaSpeechSegmenter label to canonical label.

    Args:
        raw_label: Raw label from segmenter.

    Returns:
        Canonical label (speech/music/noise/silence).

    Raises:
        ValueError: If label cannot be mapped.
    """
    mapped = INA_LABEL_MAP.get(raw_label.lower())
    if mapped is None:
        # Unknown label - treat as noise
        logger.warning("Unknown segment label '%s', treating as noise", raw_label)
        return "noise"
    return mapped


# --- Silence Gap-Fill ---


def _fill_silence_gaps(
    segments: list[SegmentData],
    total_duration: float,
) -> list[SegmentData]:
    """Fill gaps between segments with silence.

    Args:
        segments: List of segments sorted by start_sec.
        total_duration: Total audio duration in seconds.

    Returns:
        New list with silence segments filling gaps >= MIN_SILENCE_SEC.
        Derived silence segments have source="derived".
    """
    if not segments:
        # Empty input - entire duration is silence if long enough
        if total_duration >= MIN_SILENCE_SEC:
            return [
                SegmentData(
                    label="silence",
                    start_sec=0.0,
                    end_sec=total_duration,
                    confidence=CONFIDENCE_BASE["silence"],
                    source=SEGMENT_SOURCE_DERIVED,
                )
            ]
        return []

    result = []
    prev_end = 0.0

    for seg in segments:
        gap = seg.start_sec - prev_end
        if gap >= MIN_SILENCE_SEC:
            result.append(
                SegmentData(
                    label="silence",
                    start_sec=prev_end,
                    end_sec=seg.start_sec,
                    confidence=CONFIDENCE_BASE["silence"],
                    source=SEGMENT_SOURCE_DERIVED,
                )
            )
        result.append(seg)
        prev_end = seg.end_sec

    # Check trailing gap
    trailing_gap = total_duration - prev_end
    if trailing_gap >= MIN_SILENCE_SEC:
        result.append(
            SegmentData(
                label="silence",
                start_sec=prev_end,
                end_sec=total_duration,
                confidence=CONFIDENCE_BASE["silence"],
                source=SEGMENT_SOURCE_DERIVED,
            )
        )

    return result


# --- Post-Processing ---


def _filter_by_min_duration(segments: list[SegmentData]) -> list[SegmentData]:
    """Filter segments by minimum duration thresholds.

    Research Pack thresholds:
    - Speech: >= 0.8s
    - Music: >= 3.4s
    - Silence: >= 0.5s (already handled in gap-fill)
    - Noise: no minimum (kept as-is)

    Args:
        segments: List of segments.

    Returns:
        Filtered list.
    """
    result = []
    for seg in segments:
        duration = seg.end_sec - seg.start_sec
        if seg.label == "speech" and duration < MIN_SPEECH_SEC:
            continue
        if seg.label == "music" and duration < MIN_MUSIC_SEC:
            continue
        if seg.label == "silence" and duration < MIN_SILENCE_SEC:
            continue
        result.append(seg)
    return result


def _merge_adjacent_same_label(segments: list[SegmentData]) -> list[SegmentData]:
    """Merge adjacent segments with same label if gap <= MERGE_GAP_SEC.

    Args:
        segments: List of segments sorted by start_sec.

    Returns:
        Merged list.
    """
    if not segments:
        return []

    result = []
    current = segments[0]

    for seg in segments[1:]:
        gap = seg.start_sec - current.end_sec
        if seg.label == current.label and gap <= MERGE_GAP_SEC:
            # Merge: extend current segment, average confidence
            # Propagate "derived" if either side is derived to avoid
            # mislabeling derived material as model-emitted
            merged_source = (
                "derived"
                if (current.source == "derived" or seg.source == "derived")
                else current.source
            )
            avg_conf = (current.confidence + seg.confidence) / 2
            current = SegmentData(
                label=current.label,
                start_sec=current.start_sec,
                end_sec=seg.end_sec,
                confidence=avg_conf,
                source=merged_source,
            )
        else:
            result.append(current)
            current = seg

    result.append(current)
    return result


def _clamp_segments(
    segments: list[SegmentData],
    total_duration: float,
) -> list[SegmentData]:
    """Clamp segment boundaries to [0, total_duration] and drop invalid segments.

    Args:
        segments: List of segments.
        total_duration: Total audio duration in seconds.

    Returns:
        List of segments with clamped boundaries.
        Segments where end_sec <= start_sec after clamping are dropped.
    """
    result = []
    for seg in segments:
        # Clamp start and end to [0, total_duration]
        clamped_start = max(0.0, min(seg.start_sec, total_duration))
        clamped_end = max(0.0, min(seg.end_sec, total_duration))

        # Drop if invalid after clamping
        if clamped_end <= clamped_start:
            continue

        result.append(
            SegmentData(
                label=seg.label,
                start_sec=clamped_start,
                end_sec=clamped_end,
                confidence=seg.confidence,
                source=seg.source,
            )
        )
    return result


def _adjust_confidence(segments: list[SegmentData]) -> list[SegmentData]:
    """Apply deterministic confidence adjustments.

    Adjustments:
    A) Reduce confidence for segments near minimum duration threshold.
    B) Reduce confidence for derived silence (already lower base confidence).

    All confidence values are clamped to [0, 1].

    Args:
        segments: List of segments.

    Returns:
        List of segments with adjusted confidence values.
    """
    # Map label to its minimum duration threshold
    min_duration_map = {
        "speech": MIN_SPEECH_SEC,
        "music": MIN_MUSIC_SEC,
        "silence": MIN_SILENCE_SEC,
        "noise": 0.0,  # No minimum for noise
    }

    result = []
    for seg in segments:
        adjusted_conf = seg.confidence
        duration = seg.end_sec - seg.start_sec
        min_dur = min_duration_map.get(seg.label, 0.0)

        # A) Penalty for segments near the minimum duration threshold
        # If duration is within 20% of the minimum, apply penalty
        if min_dur > 0 and duration < min_dur * 1.2:
            adjusted_conf -= CONFIDENCE_PENALTY_SHORT_DURATION

        # B) Additional penalty for derived segments (gap-filled)
        # These are less certain than model-emitted segments
        if seg.source == SEGMENT_SOURCE_DERIVED:
            adjusted_conf -= CONFIDENCE_PENALTY_SHORT_DURATION

        # Clamp to [0, 1]
        adjusted_conf = max(0.0, min(1.0, adjusted_conf))

        result.append(
            SegmentData(
                label=seg.label,
                start_sec=seg.start_sec,
                end_sec=seg.end_sec,
                confidence=adjusted_conf,
                source=seg.source,
            )
        )

    return result


def _postprocess_segments(
    segments: list[SegmentData],
    total_duration: float,
) -> list[SegmentData]:
    """Apply full post-processing pipeline.

    1. Clamp boundaries to [0, total_duration]
    2. Sort by start_sec
    3. Fill silence gaps
    4. Filter by minimum durations
    5. Merge adjacent same-label segments
    6. Apply confidence adjustments

    Args:
        segments: Raw segments from segmenter.
        total_duration: Total audio duration in seconds.

    Returns:
        Post-processed segments.
    """
    # Clamp boundaries first (drops invalid segments)
    segments = _clamp_segments(segments, total_duration)

    # Sort by start time
    segments = sorted(segments, key=lambda s: s.start_sec)

    # Fill silence gaps
    segments = _fill_silence_gaps(segments, total_duration)

    # Filter by minimum duration
    segments = _filter_by_min_duration(segments)

    # Re-sort after filtering
    segments = sorted(segments, key=lambda s: s.start_sec)

    # Merge adjacent same-label
    segments = _merge_adjacent_same_label(segments)

    # Apply confidence adjustments
    segments = _adjust_confidence(segments)

    return segments


# --- Validation ---


def _validate_invariants(
    segments: list[SegmentData],
    total_duration: float | None = None,
) -> tuple[bool, str | None]:
    """Validate segment invariants.

    Invariants:
    - Sorted by start_sec
    - Non-overlapping
    - No NaN/Inf values
    - start_sec < end_sec for each segment
    - Labels are in canonical set
    - All segments are in bounds [0, total_duration] (if provided)

    Args:
        segments: List of segments.
        total_duration: Optional audio duration for bounds checking.

    Returns:
        Tuple of (ok, error_message).
    """
    prev_end = -1.0

    for i, seg in enumerate(segments):
        # Check for NaN/Inf
        if math.isnan(seg.start_sec) or math.isinf(seg.start_sec):
            return False, f"Segment {i} has NaN/Inf start_sec"
        if math.isnan(seg.end_sec) or math.isinf(seg.end_sec):
            return False, f"Segment {i} has NaN/Inf end_sec"
        if math.isnan(seg.confidence) or math.isinf(seg.confidence):
            return False, f"Segment {i} has NaN/Inf confidence"

        # Check start < end
        if seg.start_sec >= seg.end_sec:
            return False, f"Segment {i} has start_sec >= end_sec"

        # Check sorted and non-overlapping
        if seg.start_sec < prev_end:
            return False, f"Segment {i} overlaps with previous segment"

        # Check label is canonical
        if seg.label not in CANONICAL_LABELS:
            return False, f"Segment {i} has invalid label: {seg.label}"

        # Check bounds if total_duration provided
        if total_duration is not None:
            if seg.start_sec < 0:
                return False, f"Segment {i} has start_sec < 0"
            if seg.end_sec > total_duration:
                return False, f"Segment {i} has end_sec > total_duration"

        prev_end = seg.end_sec

    return True, None


def _validate_schema(data: dict) -> tuple[bool, str | None]:
    """Validate segment data against JSON schema.

    Args:
        data: Segment data dict.

    Returns:
        Tuple of (ok, error_message).
    """
    try:
        import jsonschema
    except ImportError:
        # If jsonschema not installed, do basic validation
        logger.warning("jsonschema not installed, using basic validation")
        return _basic_schema_validation(data)

    try:
        schema = _load_schema()
        jsonschema.validate(data, schema)
        return True, None
    except jsonschema.ValidationError as e:
        return False, str(e.message)
    except Exception as e:
        return False, f"Schema validation error: {e}"


def _basic_schema_validation(data: dict) -> tuple[bool, str | None]:
    """Basic schema validation without jsonschema library.

    Args:
        data: Segment data dict.

    Returns:
        Tuple of (ok, error_message).
    """
    required = ["schema_id", "version", "asset_id", "computed_at", "confidence_type", "segments"]
    for field_name in required:
        if field_name not in data:
            return False, f"Missing required field: {field_name}"

    if data["schema_id"] != SEGMENTS_SCHEMA_ID:
        return False, f"Invalid schema_id: {data['schema_id']}"

    if data["confidence_type"] != CONFIDENCE_TYPE:
        return False, f"Invalid confidence_type: {data['confidence_type']}"

    if not isinstance(data["segments"], list):
        return False, "segments must be a list"

    for i, seg in enumerate(data["segments"]):
        for key in ["label", "start_sec", "end_sec"]:
            if key not in seg:
                return False, f"Segment {i} missing required field: {key}"
        if seg["label"] not in CANONICAL_LABELS:
            return False, f"Segment {i} has invalid label: {seg['label']}"

    return True, None


# --- Metrics Computation ---


def _compute_metrics(
    segments: list[SegmentData],
    total_duration: float,
    segmentation_time_ms: int,
) -> dict[str, Any]:
    """Compute segment metrics.

    Args:
        segments: List of segments.
        total_duration: Total audio duration in seconds.
        segmentation_time_ms: Time taken for segmentation in milliseconds.

    Returns:
        Metrics dictionary.
    """
    # Compute per-class durations
    class_duration = {"speech": 0.0, "music": 0.0, "noise": 0.0, "silence": 0.0}
    for seg in segments:
        duration = seg.end_sec - seg.start_sec
        class_duration[seg.label] += duration

    # Compute class distribution (fraction of total)
    total_seg_duration = sum(class_duration.values())
    if total_seg_duration > 0:
        class_distribution = {k: v / total_seg_duration for k, v in class_duration.items()}
    else:
        class_distribution = dict.fromkeys(class_duration, 0.0)

    # Compute flip_rate = (# label transitions) / audio_duration_sec
    transitions = 0
    for i in range(1, len(segments)):
        if segments[i].label != segments[i - 1].label:
            transitions += 1

    flip_rate = transitions / total_duration if total_duration > 0 else 0.0

    return {
        "segment_count": len(segments),
        "speech_sec": class_duration["speech"],
        "music_sec": class_duration["music"],
        "silence_sec": class_duration["silence"],
        "noise_sec": class_duration["noise"],
        "class_distribution": class_distribution,
        "flip_rate": flip_rate,
        "segmentation_time_ms": segmentation_time_ms,
        "source": SEGMENT_SOURCE,
        "schema_id": SEGMENTS_SCHEMA_ID,
        "version": SEGMENTS_VERSION,
    }


# --- Audio Duration Helper ---


def _get_audio_duration(wav_path: Path) -> float:
    """Get audio duration from WAV file.

    Args:
        wav_path: Path to WAV file.

    Returns:
        Duration in seconds.
    """
    import wave

    with wave.open(str(wav_path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / rate


# --- Main Segmentation ---


def _run_segmenter(wav_path: Path) -> list[tuple[str, float, float]]:
    """Run inaSpeechSegmenter on audio file.

    Args:
        wav_path: Path to WAV file.

    Returns:
        List of (label, start_sec, end_sec) tuples.

    Raises:
        MemoryError: If OOM during inference.
        Exception: Other segmentation errors.
    """
    from inaSpeechSegmenter import Segmenter

    # Create segmenter (CPU-only)
    seg = Segmenter()

    # Run segmentation
    result = seg(str(wav_path))

    # result is list of (label, start_sec, end_sec) tuples
    return result


def run_segments_worker(
    asset_id: str,
    data_dir: Path | str,
) -> SegmentsResult:
    """Run the segments worker for an asset.

    This is the main entry point for the segments worker.

    Args:
        asset_id: The asset ID to process.
        data_dir: Base data directory (contains audio/, segments/).

    Returns:
        SegmentsResult with success/failure status and metrics.
    """
    start_time = time.monotonic()
    data_dir = Path(data_dir)

    # Get paths
    input_path = audio_normalized_path(asset_id)
    output_path = segments_json_path(asset_id)

    # --- Idempotency: Check if output already exists ---
    if output_path.exists():
        logger.info("Segments already exist for asset_id=%s", asset_id)
        return SegmentsResult(
            ok=True,
            message="Artifact already exists",
            artifact_path=str(output_path),
            artifact_type=ARTIFACT_TYPE_SEGMENTS_V1,
            schema_version=SEGMENTS_VERSION,
            metrics={
                "schema_id": SEGMENTS_SCHEMA_ID,
                "version": SEGMENTS_VERSION,
            },
        )

    # --- Input validation ---
    if not input_path.exists():
        logger.error("Normalized WAV not found for asset_id=%s: %s", asset_id, input_path)
        return SegmentsResult(
            ok=False,
            error_code=SegmentsErrorCode.INPUT_NOT_FOUND.value,
            message=f"Normalized WAV not found: {input_path}",
        )

    # --- Get audio duration ---
    try:
        total_duration = _get_audio_duration(input_path)
    except Exception as e:
        logger.error("Failed to read audio duration for asset_id=%s: %s", asset_id, e)
        return SegmentsResult(
            ok=False,
            error_code=SegmentsErrorCode.SEGMENTATION_FAILED.value,
            message=f"Failed to read audio duration: {e}",
        )

    # --- Run segmentation ---
    try:
        raw_segments = _run_segmenter(input_path)
    except MemoryError:
        logger.error("OOM during segmentation for asset_id=%s", asset_id)
        return SegmentsResult(
            ok=False,
            error_code=SegmentsErrorCode.MODEL_OOM.value,
            message="Out of memory during segmentation",
        )
    except Exception as e:
        error_str = str(e).lower()
        # Check for memory-related errors
        if "memory" in error_str or "oom" in error_str or "alloc" in error_str:
            logger.error("OOM-like error during segmentation for asset_id=%s: %s", asset_id, e)
            return SegmentsResult(
                ok=False,
                error_code=SegmentsErrorCode.MODEL_OOM.value,
                message=f"Out of memory during segmentation: {e}",
            )
        logger.error("Segmentation failed for asset_id=%s: %s", asset_id, e)
        return SegmentsResult(
            ok=False,
            error_code=SegmentsErrorCode.SEGMENTATION_FAILED.value,
            message=f"Segmentation failed: {e}",
        )

    # --- Map labels and create SegmentData ---
    segments = []
    for label, start, end in raw_segments:
        mapped_label = _map_label(label)
        confidence = CONFIDENCE_BASE.get(mapped_label, 0.70)
        segments.append(
            SegmentData(
                label=mapped_label,
                start_sec=start,
                end_sec=end,
                confidence=confidence,
                source=SEGMENT_SOURCE,
            )
        )

    # --- Post-process segments ---
    segments = _postprocess_segments(segments, total_duration)

    # --- Validate invariants (including bounds check) ---
    valid, error_msg = _validate_invariants(segments, total_duration)
    if not valid:
        logger.error(
            "Segment invariants validation failed for asset_id=%s: %s", asset_id, error_msg
        )
        return SegmentsResult(
            ok=False,
            error_code=SegmentsErrorCode.SEGMENTS_INVALID.value,
            message=f"Segment invariants validation failed: {error_msg}",
        )

    # --- Build output data ---
    computed_at = datetime.now(UTC).isoformat()

    output_data = {
        "schema_id": SEGMENTS_SCHEMA_ID,
        "version": SEGMENTS_VERSION,
        "asset_id": asset_id,
        "computed_at": computed_at,
        "confidence_type": CONFIDENCE_TYPE,
        "segments": [
            {
                "label": seg.label,
                "start_sec": seg.start_sec,
                "end_sec": seg.end_sec,
                "confidence": seg.confidence,
                "source": seg.source,
            }
            for seg in segments
        ],
    }

    # --- Validate schema ---
    valid, error_msg = _validate_schema(output_data)
    if not valid:
        logger.error("Schema validation failed for asset_id=%s: %s", asset_id, error_msg)
        return SegmentsResult(
            ok=False,
            error_code=SegmentsErrorCode.SEGMENTS_INVALID.value,
            message=f"Schema validation failed: {error_msg}",
        )

    # --- Atomic publish ---
    try:
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write atomically
        json_text = json.dumps(output_data, indent=2)
        atomic_write_text(output_path, json_text)
    except Exception as e:
        logger.error("Failed to write segments JSON for asset_id=%s: %s", asset_id, e)
        return SegmentsResult(
            ok=False,
            error_code=SegmentsErrorCode.SEGMENTATION_FAILED.value,
            message=f"Failed to write segments JSON: {e}",
        )

    # --- Success ---
    segmentation_time_ms = int((time.monotonic() - start_time) * 1000)
    metrics = _compute_metrics(segments, total_duration, segmentation_time_ms)

    logger.info(
        "Segments computed for asset_id=%s: %d segments, %.2fs total, %dms",
        asset_id,
        len(segments),
        total_duration,
        segmentation_time_ms,
    )

    return SegmentsResult(
        ok=True,
        message="Segmentation completed successfully",
        metrics=metrics,
        artifact_path=str(output_path),
        artifact_type=ARTIFACT_TYPE_SEGMENTS_V1,
        schema_version=SEGMENTS_VERSION,
        segments=segments,
    )


# --- Standalone Execution ---


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <asset_id>")
        sys.exit(1)

    from app.config import DATA_DIR

    result = run_segments_worker(sys.argv[1], DATA_DIR)
    if result.ok:
        print(f"Success: {result.artifact_path}")
        print(f"Metrics: {result.metrics}")
        sys.exit(0)
    else:
        print(f"Error: {result.error_code} - {result.message}")
        sys.exit(1)
