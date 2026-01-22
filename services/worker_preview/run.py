"""SSIS Audio Pipeline - Preview Worker.

Computes preview window selection from FeaturePack embeddings and segments data.

Stage: preview (STAGE_PREVIEW)
Input:
  - data/features/{asset_id}.{feature_spec_alias}.h5
  - data/segments/{asset_id}.segments.v1.json
Output: data/preview/{asset_id}.preview.v1.json

Candidate generation:
- Pause boundaries: low-energy regions > PAUSE_THRESHOLD_MS
- Segment boundaries: from segments.v1.json
- Window size: WINDOW_SEC (60s), minimum: MIN_WINDOW_FRACTION * WINDOW_SEC

Scoring (LOCKED weights):
- score = 0.6 * norm_energy_var + 0.4 * norm_emb_var
- Min-max normalize across candidates
- SCORE_THRESHOLD = 0.5 for selection, else fallback

Fallback strategy:
- Find first sustained energy > 300ms, use as intro_start
- mode = "smart" | "intro" | "fallback"

Error codes (Blueprint section 8):
- INPUT_NOT_FOUND: segments or features file does not exist
- FEATUREPACK_MISSING: FeaturePack for spec alias not found
- PREVIEW_FAILED: General preview computation failure
- PREVIEW_INVALID: Schema or invariants validation failed
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import wave
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from app.config import DEFAULT_FEATURE_SPEC_ID
from app.utils.atomic_io import atomic_write_text
from app.utils.hashing import feature_spec_alias as compute_feature_spec_alias
from app.utils.paths import (
    audio_normalized_path,
    features_h5_path,
    preview_json_path,
    segments_json_path,
)

logger = logging.getLogger(__name__)

# --- Constants (LOCKED) ---

# Stage identifier for this worker
STAGE_PREVIEW = "preview"

# Artifact type for preview stage output
ARTIFACT_TYPE_PREVIEW_V1 = "preview_v1"

# Schema identifiers
PREVIEW_SCHEMA_ID = "preview_candidate.v1"
PREVIEW_VERSION = "1.0.0"

# Preview window parameters
WINDOW_SEC = 60.0  # Target window duration
PAUSE_THRESHOLD_MS = 200  # Low-energy pause threshold in milliseconds
MIN_WINDOW_FRACTION = 0.75  # Minimum acceptable window as fraction of WINDOW_SEC
MIN_WINDOW_SEC = WINDOW_SEC * MIN_WINDOW_FRACTION  # 45s minimum

# Scoring weights (LOCKED)
ENERGY_WEIGHT = 0.6
EMBEDDING_WEIGHT = 0.4
SCORE_THRESHOLD = 0.5

# Fallback parameters
SUSTAINED_ENERGY_MS = 300  # Minimum sustained energy duration for intro detection

# Environment variable for feature spec alias override
FEATURE_SPEC_ALIAS_ENV = "SSIS_ACTIVE_FEATURE_SPEC_ALIAS"


# --- Error Codes ---


class PreviewErrorCode(str, Enum):
    """Error codes for preview stage per Blueprint section 8."""

    INPUT_NOT_FOUND = "INPUT_NOT_FOUND"
    FEATUREPACK_MISSING = "FEATUREPACK_MISSING"
    PREVIEW_FAILED = "PREVIEW_FAILED"
    PREVIEW_INVALID = "PREVIEW_INVALID"


# --- Result Types ---


@dataclass
class PreviewCandidate:
    """A candidate preview window."""

    start_sec: float
    end_sec: float
    energy_var: float
    emb_var: float
    score: float = 0.0


@dataclass
class PreviewResult:
    """Result of preview worker execution."""

    ok: bool
    error_code: str | None = None
    message: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_path: str | None = None
    artifact_type: str | None = None
    schema_version: str | None = None


# --- FeatureSpec Selection ---


def _get_feature_spec_alias() -> str:
    """Get the active feature spec alias.

    Selection rule (LOCKED):
    1. Check SSIS_ACTIVE_FEATURE_SPEC_ALIAS env var first (must be valid 12-char hex)
    2. If not set or invalid, derive from DEFAULT_FEATURE_SPEC_ID

    Returns:
        12-character hex feature spec alias.
    """
    env_alias = os.environ.get(FEATURE_SPEC_ALIAS_ENV)
    if env_alias:
        alias = env_alias.strip().lower()
        # Validate: must be exactly 12 hex characters
        if len(alias) != 12 or any(c not in "0123456789abcdef" for c in alias):
            logger.warning(
                "Invalid %s value '%s', falling back to default",
                FEATURE_SPEC_ALIAS_ENV,
                env_alias,
            )
            return compute_feature_spec_alias(DEFAULT_FEATURE_SPEC_ID)
        return alias
    return compute_feature_spec_alias(DEFAULT_FEATURE_SPEC_ID)


# --- Audio Duration Helper ---


def _get_audio_duration(wav_path: Path) -> float:
    """Get audio duration from WAV file.

    Args:
        wav_path: Path to WAV file.

    Returns:
        Duration in seconds.
    """
    with wave.open(str(wav_path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / rate


# --- Data Loading ---


def _load_segments(segments_path: Path) -> list[dict]:
    """Load segments from JSON file.

    Args:
        segments_path: Path to segments JSON file.

    Returns:
        List of segment dictionaries (empty if missing/invalid).
    """
    with open(segments_path) as f:
        data = json.load(f)
    segments = data.get("segments")
    if segments is None:
        logger.warning("No 'segments' key in %s, using empty list", segments_path)
        return []
    if not isinstance(segments, list):
        logger.warning("Invalid 'segments' type in %s, using empty list", segments_path)
        return []
    # Filter to dict entries only
    valid = [s for s in segments if isinstance(s, dict)]
    if len(valid) != len(segments):
        logger.warning(
            "Dropping %d non-dict segments in %s", len(segments) - len(valid), segments_path
        )
    return valid


def _load_features(h5_path: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Load features from HDF5 file.

    Args:
        h5_path: Path to HDF5 feature file.

    Returns:
        Tuple of (melspec, embeddings, mel_hop_sec, embed_hop_sec).
    """
    with h5py.File(h5_path, "r") as f:
        melspec = f["melspec"][:]
        embeddings = f["embeddings"][:]
        hop_length = f.attrs.get("hop_length", 220)
        sample_rate = f.attrs.get("sample_rate", 22050)
        embed_hop_sec = f.attrs.get("embed_hop_sec", 0.5)

    # Compute mel hop in seconds
    mel_hop_sec = hop_length / sample_rate

    return melspec, embeddings, mel_hop_sec, embed_hop_sec


# --- Boundary Detection ---


def _find_pause_boundaries(
    melspec: np.ndarray,
    mel_hop_sec: float,
    total_duration: float,
) -> list[float]:
    """Find pause boundaries based on low-energy regions.

    A pause boundary is the start of a low-energy region that exceeds
    PAUSE_THRESHOLD_MS in duration.

    Args:
        melspec: Log-mel spectrogram, shape (n_frames, n_mels).
        mel_hop_sec: Time step per mel frame in seconds.
        total_duration: Total audio duration in seconds.

    Returns:
        List of pause boundary times in seconds (deduplicated).
    """
    # Handle empty melspec gracefully
    if melspec.size == 0 or melspec.shape[0] == 0:
        return []

    # Compute frame-level energy (mean across mel bins)
    frame_energy = np.mean(melspec, axis=1)

    # Threshold for "low energy" - use lower quartile
    energy_threshold = np.percentile(frame_energy, 25)

    # Find low-energy frames
    low_energy_mask = frame_energy < energy_threshold

    # Find contiguous low-energy regions
    boundaries = []
    pause_threshold_frames = int((PAUSE_THRESHOLD_MS / 1000.0) / mel_hop_sec)

    in_pause = False
    pause_start = 0

    for i, is_low in enumerate(low_energy_mask):
        if is_low and not in_pause:
            # Start of potential pause
            in_pause = True
            pause_start = i
        elif not is_low and in_pause:
            # End of potential pause
            pause_length = i - pause_start
            if pause_length >= pause_threshold_frames:
                # Valid pause - record the boundary (start time)
                boundary_sec = pause_start * mel_hop_sec
                if 0 < boundary_sec < total_duration:
                    boundaries.append(boundary_sec)
            in_pause = False

    # Check final pause if still in one
    if in_pause:
        pause_length = len(low_energy_mask) - pause_start
        if pause_length >= pause_threshold_frames:
            boundary_sec = pause_start * mel_hop_sec
            if 0 < boundary_sec < total_duration:
                boundaries.append(boundary_sec)

    return boundaries


def _find_segment_boundaries(segments: list[dict]) -> list[float]:
    """Extract segment boundaries from segments data.

    Args:
        segments: List of segment dictionaries with start_sec and end_sec.

    Returns:
        List of boundary times in seconds (deduplicated).
    """
    boundaries: set[float] = set()
    for seg in segments:
        for field_name in ("start_sec", "end_sec"):
            value = seg.get(field_name)
            if value is None:
                continue
            if not isinstance(value, (int, float)):
                logger.warning("Invalid %s in segment: %r", field_name, value)
                continue
            if value > 0:
                boundaries.add(float(value))
    return sorted(boundaries)


def _merge_boundaries(
    pause_boundaries: list[float],
    segment_boundaries: list[float],
    total_duration: float,
) -> list[float]:
    """Merge and deduplicate boundaries.

    Args:
        pause_boundaries: Boundaries from pause detection.
        segment_boundaries: Boundaries from segments.
        total_duration: Total audio duration.

    Returns:
        Sorted, deduplicated list of boundary times.
    """
    # Combine all boundaries
    all_boundaries = set(pause_boundaries) | set(segment_boundaries)

    # Add 0 and total_duration as implicit boundaries
    all_boundaries.add(0.0)
    all_boundaries.add(total_duration)

    # Filter to valid range and sort
    valid = [b for b in all_boundaries if 0 <= b <= total_duration]
    return sorted(set(valid))


# --- Candidate Generation ---


def _generate_candidates(
    boundaries: list[float],
    melspec: np.ndarray,
    embeddings: np.ndarray,
    mel_hop_sec: float,
    embed_hop_sec: float,
    total_duration: float,
) -> list[PreviewCandidate]:
    """Generate preview window candidates from boundaries.

    Each candidate is a window of approximately WINDOW_SEC starting at a boundary.
    Windows are clamped to [0, total_duration] and dropped if < MIN_WINDOW_SEC.

    Args:
        boundaries: Sorted list of boundary times.
        melspec: Log-mel spectrogram.
        embeddings: YAMNet embeddings.
        mel_hop_sec: Time step per mel frame.
        embed_hop_sec: Time step per embedding frame.
        total_duration: Total audio duration.

    Returns:
        List of PreviewCandidate objects.
    """
    candidates = []
    seen_windows = set()

    for start in boundaries:
        # Clamp start to [0, total_duration] first
        start = max(0.0, min(start, total_duration))

        # Compute window end based on clamped start
        end = min(start + WINDOW_SEC, total_duration)

        # Skip if window too short
        duration = end - start
        if duration < MIN_WINDOW_SEC:
            continue

        # Deduplicate windows (round to 0.1s for dedup)
        window_key = (round(start, 1), round(end, 1))
        if window_key in seen_windows:
            continue
        seen_windows.add(window_key)

        # Compute energy variance for this window (from melspec)
        mel_start_idx = int(start / mel_hop_sec)
        mel_end_idx = int(end / mel_hop_sec)
        mel_start_idx = max(0, min(mel_start_idx, len(melspec) - 1))
        mel_end_idx = max(mel_start_idx + 1, min(mel_end_idx, len(melspec)))

        window_mel = melspec[mel_start_idx:mel_end_idx]
        if len(window_mel) > 0:
            frame_energy = np.mean(window_mel, axis=1)
            energy_var = float(np.var(frame_energy))
        else:
            energy_var = 0.0

        # Compute embedding variance for this window
        emb_start_idx = int(start / embed_hop_sec)
        emb_end_idx = int(end / embed_hop_sec)
        emb_start_idx = max(0, min(emb_start_idx, len(embeddings) - 1))
        emb_end_idx = max(emb_start_idx + 1, min(emb_end_idx, len(embeddings)))

        window_emb = embeddings[emb_start_idx:emb_end_idx]
        if len(window_emb) > 1:
            # Variance across time dimension (mean of per-dimension variances)
            emb_var = float(np.mean(np.var(window_emb, axis=0)))
        else:
            emb_var = 0.0

        candidates.append(
            PreviewCandidate(
                start_sec=start,
                end_sec=end,
                energy_var=energy_var,
                emb_var=emb_var,
            )
        )

    return candidates


# --- Scoring ---


def _normalize_minmax(values: list[float]) -> list[float]:
    """Min-max normalize a list of values to [0, 1].

    Args:
        values: List of float values.

    Returns:
        Normalized values. If all values are equal, returns 0.5 for all.
    """
    if not values:
        return []

    min_val = min(values)
    max_val = max(values)

    if max_val - min_val < 1e-10:
        # All values are (nearly) equal
        return [0.5] * len(values)

    return [(v - min_val) / (max_val - min_val) for v in values]


def _score_candidates(candidates: list[PreviewCandidate]) -> list[PreviewCandidate]:
    """Score candidates using LOCKED weights.

    score = ENERGY_WEIGHT * norm_energy_var + EMBEDDING_WEIGHT * norm_emb_var

    Args:
        candidates: List of candidates with energy_var and emb_var computed.

    Returns:
        Same candidates with score field populated.
    """
    if not candidates:
        return []

    # Extract variances
    energy_vars = [c.energy_var for c in candidates]
    emb_vars = [c.emb_var for c in candidates]

    # Normalize
    norm_energy = _normalize_minmax(energy_vars)
    norm_emb = _normalize_minmax(emb_vars)

    # Compute scores
    for i, candidate in enumerate(candidates):
        candidate.score = ENERGY_WEIGHT * norm_energy[i] + EMBEDDING_WEIGHT * norm_emb[i]

    return candidates


# --- Fallback Selection ---


def _find_intro_start(
    melspec: np.ndarray,
    mel_hop_sec: float,
    total_duration: float,
) -> float:
    """Find the first sustained energy region for intro fallback.

    Looks for the first region with energy above median that lasts at least
    SUSTAINED_ENERGY_MS.

    Args:
        melspec: Log-mel spectrogram.
        mel_hop_sec: Time step per mel frame.
        total_duration: Total audio duration.

    Returns:
        Start time in seconds for intro window.
    """
    if len(melspec) == 0:
        return 0.0

    # Compute frame-level energy
    frame_energy = np.mean(melspec, axis=1)

    # Threshold: median energy
    energy_threshold = np.median(frame_energy)

    # Find first sustained high-energy region
    sustained_frames = int((SUSTAINED_ENERGY_MS / 1000.0) / mel_hop_sec)

    high_energy_count = 0
    for i, energy in enumerate(frame_energy):
        if energy >= energy_threshold:
            high_energy_count += 1
            if high_energy_count >= sustained_frames:
                # Found sustained energy, return start of this region
                start_frame = i - high_energy_count + 1
                return max(0.0, start_frame * mel_hop_sec)
        else:
            high_energy_count = 0

    # Default to start
    return 0.0


# --- Schema Validation ---


def _load_schema() -> dict:
    """Load the preview candidate JSON schema.

    Returns:
        Parsed JSON schema dict.

    Raises:
        FileNotFoundError: If schema file does not exist.
    """
    schema_path = Path(__file__).parent.parent.parent / "specs" / "preview_candidate.schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Preview schema not found at {schema_path}")
    with open(schema_path) as f:
        return json.load(f)


def _validate_schema(data: dict) -> tuple[bool, str | None]:
    """Validate preview data against JSON schema.

    Args:
        data: Preview data dict.

    Returns:
        Tuple of (ok, error_message).
    """
    try:
        import jsonschema
    except ImportError:
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
        data: Preview data dict.

    Returns:
        Tuple of (ok, error_message).
    """
    required = [
        "schema_id",
        "version",
        "asset_id",
        "computed_at",
        "mode",
        "start_sec",
        "end_sec",
        "duration_sec",
    ]
    for field_name in required:
        if field_name not in data:
            return False, f"Missing required field: {field_name}"

    if data["schema_id"] != PREVIEW_SCHEMA_ID:
        return False, f"Invalid schema_id: {data['schema_id']}"

    valid_modes = {"smart", "intro", "fallback"}
    if data["mode"] not in valid_modes:
        return False, f"Invalid mode: {data['mode']}"

    if not isinstance(data["start_sec"], (int, float)):
        return False, "start_sec must be a number"
    if not isinstance(data["end_sec"], (int, float)):
        return False, "end_sec must be a number"
    if not isinstance(data["duration_sec"], (int, float)):
        return False, "duration_sec must be a number"

    return True, None


def _validate_invariants(data: dict) -> tuple[bool, str | None]:
    """Validate preview invariants.

    Invariants:
    - end_sec >= start_sec
    - duration_sec == end_sec - start_sec (approximately)
    - No NaN/Inf values in numeric fields (start_sec, end_sec, duration_sec, confidence, best_score)

    Args:
        data: Preview data dict.

    Returns:
        Tuple of (ok, error_message).
    """
    start = data.get("start_sec", 0)
    end = data.get("end_sec", 0)
    duration = data.get("duration_sec", 0)

    # Check for NaN/Inf in all numeric fields (including optional ones)
    numeric_fields = {
        "start_sec": start,
        "end_sec": end,
        "duration_sec": duration,
        "confidence": data.get("confidence"),
        "best_score": data.get("best_score"),
    }
    for field_name, value in numeric_fields.items():
        if value is None:
            continue
        if math.isnan(value) or math.isinf(value):
            return False, f"{field_name} contains NaN/Inf"

    # Check end >= start
    if end < start:
        return False, f"end_sec ({end}) < start_sec ({start})"

    # Check duration consistency (with small tolerance)
    expected_duration = end - start
    if abs(duration - expected_duration) > 0.01:
        return False, f"duration_sec ({duration}) != end_sec - start_sec ({expected_duration})"

    return True, None


# --- Main Worker ---


def run_preview_worker(asset_id: str) -> PreviewResult:
    """Run the preview worker for an asset.

    This is the main entry point for the preview worker.

    Args:
        asset_id: The asset ID to process.

    Returns:
        PreviewResult with success/failure status and metrics.
    """
    start_time = time.monotonic()

    # Get feature spec alias (LOCKED selection rule)
    spec_alias = _get_feature_spec_alias()

    # Get paths
    output_path = preview_json_path(asset_id)
    segments_path = segments_json_path(asset_id)
    features_path = features_h5_path(asset_id, spec_alias)
    wav_path = audio_normalized_path(asset_id)

    # --- Idempotency: Check if output already exists ---
    if output_path.exists():
        # Validate integrity before returning cached result
        try:
            with open(output_path) as f:
                existing = json.load(f)
            # Check asset_id and schema_id match
            if (
                existing.get("asset_id") != asset_id
                or existing.get("schema_id") != PREVIEW_SCHEMA_ID
            ):
                logger.warning(
                    "Existing preview for asset_id=%s has mismatched metadata, regenerating",
                    asset_id,
                )
            else:
                logger.info("Preview already exists for asset_id=%s", asset_id)
                return PreviewResult(
                    ok=True,
                    message="Artifact already exists",
                    artifact_path=str(output_path),
                    artifact_type=ARTIFACT_TYPE_PREVIEW_V1,
                    schema_version=PREVIEW_VERSION,
                    metrics={
                        "schema_id": PREVIEW_SCHEMA_ID,
                        "version": PREVIEW_VERSION,
                        "spec_alias_used": spec_alias,
                    },
                )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "Existing preview for asset_id=%s is corrupt (%s), regenerating",
                asset_id,
                e,
            )

    # --- Input validation: segments ---
    if not segments_path.exists():
        logger.error("Segments file not found for asset_id=%s: %s", asset_id, segments_path)
        return PreviewResult(
            ok=False,
            error_code=PreviewErrorCode.INPUT_NOT_FOUND.value,
            message=f"Segments file not found: {segments_path}",
            metrics={"spec_alias_used": spec_alias},
        )

    # --- Input validation: FeaturePack ---
    if not features_path.exists():
        logger.error(
            "FeaturePack not found for asset_id=%s, alias=%s: %s",
            asset_id,
            spec_alias,
            features_path,
        )
        return PreviewResult(
            ok=False,
            error_code=PreviewErrorCode.FEATUREPACK_MISSING.value,
            message=f"FeaturePack not found for alias {spec_alias}: {features_path}",
            metrics={"spec_alias_used": spec_alias},
        )

    # --- Get audio duration ---
    try:
        if wav_path.exists():
            total_duration = _get_audio_duration(wav_path)
        else:
            # Fallback: estimate from features
            logger.warning("WAV not found, estimating duration from features")
            with h5py.File(features_path, "r") as f:
                melspec = f["melspec"][:]
                hop_length = f.attrs.get("hop_length", 220)
                sample_rate = f.attrs.get("sample_rate", 22050)
            mel_hop_sec = hop_length / sample_rate
            total_duration = len(melspec) * mel_hop_sec
    except Exception as e:
        logger.error("Failed to get audio duration for asset_id=%s: %s", asset_id, e)
        return PreviewResult(
            ok=False,
            error_code=PreviewErrorCode.PREVIEW_FAILED.value,
            message=f"Failed to get audio duration: {e}",
            metrics={"spec_alias_used": spec_alias},
        )

    # --- Load data ---
    try:
        segments = _load_segments(segments_path)
        melspec, embeddings, mel_hop_sec, embed_hop_sec = _load_features(features_path)
    except Exception as e:
        logger.error("Failed to load data for asset_id=%s: %s", asset_id, e)
        return PreviewResult(
            ok=False,
            error_code=PreviewErrorCode.PREVIEW_FAILED.value,
            message=f"Failed to load data: {e}",
            metrics={"spec_alias_used": spec_alias},
        )

    # --- Find boundaries ---
    try:
        pause_boundaries = _find_pause_boundaries(melspec, mel_hop_sec, total_duration)
        segment_boundaries = _find_segment_boundaries(segments)
        all_boundaries = _merge_boundaries(pause_boundaries, segment_boundaries, total_duration)
    except Exception as e:
        logger.error("Failed to find boundaries for asset_id=%s: %s", asset_id, e)
        return PreviewResult(
            ok=False,
            error_code=PreviewErrorCode.PREVIEW_FAILED.value,
            message=f"Failed to find boundaries: {e}",
            metrics={"spec_alias_used": spec_alias},
        )

    # --- Generate candidates ---
    try:
        candidates = _generate_candidates(
            all_boundaries,
            melspec,
            embeddings,
            mel_hop_sec,
            embed_hop_sec,
            total_duration,
        )
    except Exception as e:
        logger.error("Failed to generate candidates for asset_id=%s: %s", asset_id, e)
        return PreviewResult(
            ok=False,
            error_code=PreviewErrorCode.PREVIEW_FAILED.value,
            message=f"Failed to generate candidates: {e}",
            metrics={"spec_alias_used": spec_alias},
        )

    # --- Score candidates ---
    candidates = _score_candidates(candidates)

    # --- Select best candidate or fallback ---
    mode = "fallback"
    fallback_used = True
    best_score = 0.0
    confidence = None
    reason = None

    if candidates:
        # Sort by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)
        best = candidates[0]
        best_score = best.score

        if best_score >= SCORE_THRESHOLD:
            # Smart selection
            mode = "smart"
            fallback_used = False
            start_sec = best.start_sec
            end_sec = best.end_sec
            confidence = best_score
            reason = f"Selected candidate with score {best_score:.3f}"
        else:
            # Score too low, use intro fallback
            mode = "intro"
            fallback_used = True
            intro_start = _find_intro_start(melspec, mel_hop_sec, total_duration)
            start_sec = intro_start
            end_sec = min(intro_start + WINDOW_SEC, total_duration)
            confidence = best_score  # Report best score even if not used
            reason = f"Best score {best_score:.3f} below threshold {SCORE_THRESHOLD}"
    else:
        # No valid candidates, full fallback
        mode = "fallback"
        fallback_used = True
        intro_start = _find_intro_start(melspec, mel_hop_sec, total_duration)
        start_sec = intro_start
        end_sec = min(intro_start + WINDOW_SEC, total_duration)
        reason = "No valid candidates generated"

    # Ensure minimum window duration even in fallback
    if end_sec - start_sec < MIN_WINDOW_SEC:
        # Try to extend window while preserving intro start if possible
        if total_duration >= MIN_WINDOW_SEC:
            if start_sec + MIN_WINDOW_SEC <= total_duration:
                # Can extend from current start_sec
                end_sec = min(start_sec + WINDOW_SEC, total_duration)
            else:
                # Must reset to beginning to fit minimum window
                start_sec = 0.0
                end_sec = min(WINDOW_SEC, total_duration)
        else:
            # Audio is shorter than minimum window, use full audio
            start_sec = 0.0
            end_sec = total_duration

    duration_sec = end_sec - start_sec

    # --- Build output data ---
    computed_at = datetime.now(UTC).isoformat()

    output_data = {
        "schema_id": PREVIEW_SCHEMA_ID,
        "version": PREVIEW_VERSION,
        "asset_id": asset_id,
        "computed_at": computed_at,
        "mode": mode,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": duration_sec,
        "confidence": confidence,
        "fallback_used": fallback_used,
        "reason": reason,
        "feature_spec_alias": spec_alias,
        "candidate_count": len(candidates),
        "best_score": best_score if candidates else None,
    }

    # --- Validate invariants ---
    valid, error_msg = _validate_invariants(output_data)
    if not valid:
        logger.error(
            "Preview invariants validation failed for asset_id=%s: %s", asset_id, error_msg
        )
        return PreviewResult(
            ok=False,
            error_code=PreviewErrorCode.PREVIEW_INVALID.value,
            message=f"Preview invariants validation failed: {error_msg}",
            metrics={
                "spec_alias_used": spec_alias,
                "candidate_count": len(candidates),
                "best_score": best_score,
                "fallback_used": fallback_used,
            },
        )

    # --- Validate schema ---
    valid, error_msg = _validate_schema(output_data)
    if not valid:
        logger.error("Preview schema validation failed for asset_id=%s: %s", asset_id, error_msg)
        return PreviewResult(
            ok=False,
            error_code=PreviewErrorCode.PREVIEW_INVALID.value,
            message=f"Preview schema validation failed: {error_msg}",
            metrics={
                "spec_alias_used": spec_alias,
                "candidate_count": len(candidates),
                "best_score": best_score,
                "fallback_used": fallback_used,
            },
        )

    # --- Atomic publish ---
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        json_text = json.dumps(output_data, indent=2)
        atomic_write_text(output_path, json_text)
    except Exception as e:
        logger.error("Failed to write preview JSON for asset_id=%s: %s", asset_id, e)
        return PreviewResult(
            ok=False,
            error_code=PreviewErrorCode.PREVIEW_FAILED.value,
            message=f"Failed to write preview JSON: {e}",
            metrics={
                "spec_alias_used": spec_alias,
                "candidate_count": len(candidates),
                "best_score": best_score,
                "fallback_used": fallback_used,
            },
        )

    # --- Success ---
    preview_time_ms = int((time.monotonic() - start_time) * 1000)

    metrics = {
        "spec_alias_used": spec_alias,
        "candidate_count": len(candidates),
        "best_score": best_score,
        "fallback_used": fallback_used,
        "score_threshold": SCORE_THRESHOLD,
        "window_sec": WINDOW_SEC,
        "preview_time_ms": preview_time_ms,
        "mode": mode,
        "schema_id": PREVIEW_SCHEMA_ID,
        "version": PREVIEW_VERSION,
    }

    logger.info(
        "Preview computed for asset_id=%s: mode=%s, score=%.3f, %dms",
        asset_id,
        mode,
        best_score,
        preview_time_ms,
    )

    return PreviewResult(
        ok=True,
        message="Preview computed successfully",
        metrics=metrics,
        artifact_path=str(output_path),
        artifact_type=ARTIFACT_TYPE_PREVIEW_V1,
        schema_version=PREVIEW_VERSION,
    )


# --- Standalone Execution ---


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <asset_id>")
        sys.exit(1)

    result = run_preview_worker(sys.argv[1])
    if result.ok:
        print(f"Success: {result.artifact_path}")
        print(f"Metrics: {result.metrics}")
        sys.exit(0)
    else:
        print(f"Error: {result.error_code} - {result.message}")
        sys.exit(1)
