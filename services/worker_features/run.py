"""SSIS Audio Pipeline - Features Worker.

Computes log-mel spectrograms and YAMNet embeddings from the canonical normalized WAV.

Stage: features (STAGE_FEATURES)
Input: data/audio/{asset_id}/normalized.wav
Output: data/features/{asset_id}.{feature_spec_alias}.h5

Backend: ONNX via onnxruntime (CPU)

Resilience features:
- Model SHA256 verification before inference
- NaN/Inf validation before writing HDF5
- Shape validation for mel and embeddings
- Atomic publish for final .h5

Dependencies:
- Requires onnxruntime installed
- Requires h5py installed
- Requires numpy installed
- Requires librosa installed (for mel spectrogram computation)

Error codes (Blueprint section 8):
- FEATURE_NAN: NaN or Inf detected in computed features
- MODEL_OOM: Out of memory during inference
- FEATURE_EXTRACTION_FAILED: General feature extraction failure
- FEATURE_SPEC_ALIAS_COLLISION: Alias exists with different spec_id
- INPUT_NOT_FOUND: normalized.wav does not exist

Failpoints (Step 8 resilience harness):
- FEATURES_AFTER_H5_TMP_WRITE: After writing HDF5 temp file, before fsync/rename
- FEATURES_BEFORE_H5_RENAME: Before atomic rename of final HDF5
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from app.config import CANONICAL_SAMPLE_RATE, DEFAULT_FEATURE_SPEC_ID, FEATURES_DIR
from app.db import FeatureSpecAliasCollision, init_db, register_feature_spec
from app.orchestrator import ARTIFACT_TYPE_FEATURES_H5
from app.utils.failpoints import maybe_fail
from app.utils.hashing import feature_spec_alias, sha256_file
from app.utils.paths import audio_normalized_path, features_h5_path

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# --- Locked v1.4 Config (Blueprint section 8) ---

# Mel-spectrogram parameters
SAMPLE_RATE = CANONICAL_SAMPLE_RATE  # 22050
N_MELS = 64
HOP_LENGTH = 220  # 10ms at 22050 Hz
WIN_LENGTH = 551  # 25ms at 22050 Hz
N_FFT = 1024

# YAMNet embedding parameters
EMBEDDING_DIM = 1024
EMBED_HOP_SEC = 0.5  # One embedding per 0.5s

# ONNX backend
BACKEND = "onnxruntime"

# Model identification
MODEL_ID = "yamnet"

# Schema version for HDF5 artifact
ARTIFACT_SCHEMA_VERSION = "1.0.0"

# --- Model Paths ---

# Get the directory containing this file
_WORKER_DIR = Path(__file__).parent
YAMNET_ONNX_DIR = _WORKER_DIR / "yamnet_onnx"
YAMNET_ONNX_PATH = YAMNET_ONNX_DIR / "yamnet.onnx"
YAMNET_SHA256_PATH = YAMNET_ONNX_DIR / "yamnet.onnx.sha256"

# Temporary file suffix
H5_TEMP_SUFFIX = ".tmp"


# --- Error Codes ---


class FeaturesErrorCode:
    """Error codes for features stage per Blueprint section 8."""

    FEATURE_NAN = "FEATURE_NAN"
    MODEL_OOM = "MODEL_OOM"
    FEATURE_EXTRACTION_FAILED = "FEATURE_EXTRACTION_FAILED"
    FEATURE_SPEC_ALIAS_COLLISION = "FEATURE_SPEC_ALIAS_COLLISION"
    INPUT_NOT_FOUND = "INPUT_NOT_FOUND"


# --- Result Types ---


@dataclass
class FeaturesResult:
    """Result of features worker execution."""

    ok: bool
    error_code: str | None = None
    message: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_path: str | None = None
    artifact_type: str | None = None
    schema_version: str | None = None
    feature_spec_alias: str | None = None


# --- Model Verification ---


def _is_placeholder_hash(hash_value: str) -> bool:
    """Check if a hash value appears to be a placeholder.

    Args:
        hash_value: The hash string to check.

    Returns:
        True if the hash appears to be a placeholder, False otherwise.
    """
    # Check for common placeholder indicators
    lower_hash = hash_value.lower()

    # Must be valid hex and exactly 64 characters for SHA256
    if len(hash_value) != 64:
        return True
    if not all(c in "0123456789abcdef" for c in lower_hash):
        return True

    # Check for placeholder keywords
    if "placeholder" in lower_hash or "todo" in lower_hash:
        return True

    return False


def _verify_model_integrity() -> tuple[bool, str | None, str | None]:
    """Verify ONNX model integrity via SHA256 hash.

    Returns:
        Tuple of (ok, expected_hash, actual_hash).
        On success: (True, hash, hash)
        On failure: (False, expected, actual) or (False, None, error_msg)
    """
    if not YAMNET_ONNX_PATH.exists():
        return False, None, f"Model file not found: {YAMNET_ONNX_PATH}"

    if not YAMNET_SHA256_PATH.exists():
        return False, None, f"Model hash file not found: {YAMNET_SHA256_PATH}"

    try:
        # Read expected hash (strip whitespace and handle common formats)
        expected_hash = YAMNET_SHA256_PATH.read_text().strip().lower()
        # Handle "hash  filename" format
        if " " in expected_hash:
            expected_hash = expected_hash.split()[0]

        # Guard: fail fast if placeholder hash detected
        if _is_placeholder_hash(expected_hash):
            return (
                False,
                None,
                "yamnet.onnx.sha256 placeholder detected - replace with real SHA256 digest",
            )

        # Compute actual hash
        actual_hash = sha256_file(YAMNET_ONNX_PATH)

        if expected_hash == actual_hash:
            return True, expected_hash, actual_hash
        else:
            return False, expected_hash, actual_hash

    except Exception as e:
        return False, None, str(e)


# --- Feature Computation ---


def _compute_mel_spectrogram(audio: np.ndarray, sr: int) -> np.ndarray:
    """Compute log-mel spectrogram.

    Args:
        audio: Audio samples (1D numpy array, float32).
        sr: Sample rate.

    Returns:
        Log-mel spectrogram, shape (n_frames, N_MELS).
    """
    import librosa

    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    # Convert to log scale (dB)
    # Add small epsilon to avoid log(0)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Transpose to (n_frames, n_mels)
    return log_mel.T.astype(np.float32)


def _compute_yamnet_embeddings(
    audio: np.ndarray,
    sr: int,
    session: Any,  # onnxruntime.InferenceSession
) -> np.ndarray:
    """Compute YAMNet embeddings using ONNX runtime.

    YAMNet expects input at 16kHz. We resample if necessary.

    Args:
        audio: Audio samples (1D numpy array, float32).
        sr: Sample rate of input audio.
        session: ONNX InferenceSession for YAMNet model.

    Returns:
        Embeddings, shape (n_embed_frames, EMBEDDING_DIM).
    """
    import librosa

    # YAMNet expects 16kHz input
    yamnet_sr = 16000
    if sr != yamnet_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=yamnet_sr)

    # YAMNet processes audio in windows
    # Each window is 0.96 seconds (15360 samples at 16kHz)
    # We use hop of EMBED_HOP_SEC (0.5s = 8000 samples)
    window_samples = int(0.96 * yamnet_sr)
    hop_samples = int(EMBED_HOP_SEC * yamnet_sr)

    embeddings = []
    for start in range(0, len(audio) - window_samples + 1, hop_samples):
        window = audio[start : start + window_samples]

        # YAMNet expects input shape (1, num_samples) or (num_samples,)
        # Actual shape depends on the specific ONNX model export
        input_name = session.get_inputs()[0].name
        input_data = window.astype(np.float32)

        # Check expected input shape
        input_shape = session.get_inputs()[0].shape
        if len(input_shape) == 2:
            input_data = input_data.reshape(1, -1)

        # Run inference
        outputs = session.run(None, {input_name: input_data})

        # Get embeddings output (typically the last dense layer before classification)
        # The exact output depends on the model export
        # Common: outputs[0] = scores, outputs[1] = embeddings
        if len(outputs) > 1:
            embedding = outputs[1]
        else:
            embedding = outputs[0]

        # Flatten if needed and ensure correct dimension
        embedding = np.array(embedding).flatten()
        if len(embedding) >= EMBEDDING_DIM:
            embedding = embedding[:EMBEDDING_DIM]
        else:
            # Pad if needed (shouldn't happen with proper model)
            logger.warning(
                "Embedding dimension %d < expected %d, padding with zeros",
                len(embedding),
                EMBEDDING_DIM,
            )
            embedding = np.pad(embedding, (0, EMBEDDING_DIM - len(embedding)))

        embeddings.append(embedding)

    if len(embeddings) == 0:
        # Audio too short for even one embedding - return single zero embedding
        return np.zeros((1, EMBEDDING_DIM), dtype=np.float32)

    return np.array(embeddings, dtype=np.float32)


# --- Validation ---


def _validate_features(
    mel: np.ndarray,
    embeddings: np.ndarray,
) -> tuple[bool, str | None, int]:
    """Validate computed features for NaN/Inf and correct shapes.

    Args:
        mel: Log-mel spectrogram, expected shape (n_frames, N_MELS).
        embeddings: YAMNet embeddings, expected shape (n_embed_frames, EMBEDDING_DIM).

    Returns:
        Tuple of (ok, error_code, nan_inf_count).
    """
    # Check for NaN/Inf in mel
    mel_nan_count = np.isnan(mel).sum() + np.isinf(mel).sum()

    # Check for NaN/Inf in embeddings
    embed_nan_count = np.isnan(embeddings).sum() + np.isinf(embeddings).sum()

    total_nan_inf = int(mel_nan_count + embed_nan_count)

    if total_nan_inf > 0:
        return False, FeaturesErrorCode.FEATURE_NAN, total_nan_inf

    # Shape validation
    if mel.ndim != 2 or mel.shape[1] != N_MELS:
        return False, FeaturesErrorCode.FEATURE_EXTRACTION_FAILED, 0

    if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
        return False, FeaturesErrorCode.FEATURE_EXTRACTION_FAILED, 0

    return True, None, 0


# --- HDF5 Writing ---


def _write_hdf5_atomic(
    final_path: Path,
    mel: np.ndarray,
    embeddings: np.ndarray,
    asset_id: str,
    spec_id: str,
    spec_alias: str,
    model_sha256: str,
) -> None:
    """Write HDF5 file with atomic publish semantics.

    Args:
        final_path: Final path for the HDF5 file.
        mel: Log-mel spectrogram array.
        embeddings: YAMNet embeddings array.
        asset_id: Asset identifier.
        spec_id: Feature spec ID.
        spec_alias: Feature spec alias.
        model_sha256: Verified model SHA256 hash.
    """
    temp_path = final_path.with_suffix(final_path.suffix + H5_TEMP_SUFFIX)

    # Ensure parent directory exists
    final_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean up any orphan temp file (best-effort, do not block extraction)
    if temp_path.exists():
        try:
            temp_path.unlink()
        except OSError:
            logger.warning("Failed to remove orphan temp file: %s", temp_path)

    computed_at = datetime.now(UTC).isoformat()

    # Write to temp file
    with h5py.File(temp_path, "w") as f:
        # Create datasets
        f.create_dataset("melspec", data=mel, dtype="float32")
        f.create_dataset("embeddings", data=embeddings, dtype="float32")

        # Write attributes (Blueprint section 6 versioned contract)
        f.attrs["schema_version"] = ARTIFACT_SCHEMA_VERSION
        f.attrs["asset_id"] = asset_id
        f.attrs["computed_at"] = computed_at
        f.attrs["feature_spec_id"] = spec_id
        f.attrs["feature_spec_alias"] = spec_alias
        f.attrs["model_id"] = MODEL_ID
        f.attrs["model_sha256"] = model_sha256
        f.attrs["sample_rate"] = SAMPLE_RATE
        f.attrs["hop_length"] = HOP_LENGTH
        f.attrs["win_length"] = WIN_LENGTH
        f.attrs["n_fft"] = N_FFT
        f.attrs["n_mels"] = N_MELS
        f.attrs["embedding_dim"] = EMBEDDING_DIM
        f.attrs["embed_hop_sec"] = EMBED_HOP_SEC
        f.attrs["backend"] = BACKEND

        # Flush HDF5 internal buffers
        f.flush()

        # fsync via HDF5 file handle
        # h5py File has an id attribute that can get the vfd handle
        try:
            vfd_handle = f.id.get_vfd_handle()
            if vfd_handle is not None:
                os.fsync(vfd_handle)
        except (AttributeError, OSError):
            # Fall back to closing and fsyncing the path
            pass

    # Failpoint: after temp file write, before final fsync/rename
    maybe_fail("FEATURES_AFTER_H5_TMP_WRITE")

    # fsync the temp file via os
    fd = os.open(temp_path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

    # Failpoint: before atomic rename
    maybe_fail("FEATURES_BEFORE_H5_RENAME")

    # Atomic rename
    os.replace(temp_path, final_path)

    # Best-effort fsync directory
    try:
        dir_fd = os.open(final_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except (OSError, AttributeError):
        pass


# --- Cleanup ---


def cleanup_orphan_h5_tmp(directory: Path | None = None) -> int:
    """Clean up orphan .h5.tmp files on startup.

    Args:
        directory: Directory to scan. Defaults to FEATURES_DIR.

    Returns:
        Number of files removed.
    """
    if directory is None:
        directory = FEATURES_DIR

    if not directory.exists():
        return 0

    removed = 0
    for tmp_file in directory.glob(f"*{H5_TEMP_SUFFIX}"):
        try:
            tmp_file.unlink()
            logger.debug("Cleaned up orphan temp file: %s", tmp_file)
            removed += 1
        except OSError:
            pass

    return removed


# --- Main Feature Extraction ---


def extract_features(
    session: Session,
    asset_id: str,
    feature_spec_id: str | None = None,
) -> FeaturesResult:
    """Extract features from a normalized audio asset.

    This is the main entry point for the features worker.

    Args:
        session: Database session.
        asset_id: The asset ID to process.
        feature_spec_id: Optional feature spec ID. Defaults to DEFAULT_FEATURE_SPEC_ID.

    Returns:
        FeaturesResult with success/failure status and metrics.
    """
    start_time = time.monotonic()

    # Use default feature spec if not specified
    if feature_spec_id is None:
        feature_spec_id = DEFAULT_FEATURE_SPEC_ID

    # Compute feature spec alias
    spec_alias = feature_spec_alias(feature_spec_id)

    # Get paths
    input_path = audio_normalized_path(asset_id)
    output_path = features_h5_path(asset_id, spec_alias)

    # --- Idempotency: Check if output already exists ---
    if output_path.exists():
        logger.info("Feature pack already exists for asset_id=%s, alias=%s", asset_id, spec_alias)
        return FeaturesResult(
            ok=True,
            message="Artifact already exists",
            artifact_path=str(output_path),
            artifact_type=ARTIFACT_TYPE_FEATURES_H5,
            schema_version=ARTIFACT_SCHEMA_VERSION,
            feature_spec_alias=spec_alias,
            metrics={
                "feature_spec_id": feature_spec_id,
                "feature_spec_alias": spec_alias,
            },
        )

    # --- Input validation ---
    if not input_path.exists():
        logger.error("Normalized WAV not found for asset_id=%s: %s", asset_id, input_path)
        return FeaturesResult(
            ok=False,
            error_code=FeaturesErrorCode.INPUT_NOT_FOUND,
            message=f"Normalized WAV not found: {input_path}",
        )

    # --- FeatureSpec registration with immutability enforcement ---
    try:
        register_feature_spec(session, feature_spec_id)
        session.flush()
    except FeatureSpecAliasCollision as e:
        logger.error("FeatureSpec alias collision: %s", e)
        return FeaturesResult(
            ok=False,
            error_code=FeaturesErrorCode.FEATURE_SPEC_ALIAS_COLLISION,
            message=str(e),
        )

    # --- Model integrity verification ---
    model_ok, expected_hash, actual_or_error = _verify_model_integrity()
    if not model_ok:
        logger.error(
            "Model integrity check failed: expected=%s, actual=%s", expected_hash, actual_or_error
        )
        return FeaturesResult(
            ok=False,
            error_code=FeaturesErrorCode.FEATURE_EXTRACTION_FAILED,
            message=f"Model integrity verification failed: {actual_or_error}",
        )

    model_sha256 = actual_or_error  # actual_or_error is the hash on success

    # --- Load audio ---
    try:
        import librosa

        audio, sr = librosa.load(str(input_path), sr=SAMPLE_RATE, mono=True)
        audio = audio.astype(np.float32)
    except MemoryError:
        logger.error("OOM during audio load for asset_id=%s", asset_id)
        return FeaturesResult(
            ok=False,
            error_code=FeaturesErrorCode.MODEL_OOM,
            message="Out of memory loading audio file",
        )
    except Exception as e:
        error_str = str(e).lower()
        # Check for memory-related errors that may not be MemoryError subclass
        if "memory" in error_str or "alloc" in error_str:
            logger.error("OOM-like error during audio load for asset_id=%s: %s", asset_id, e)
            return FeaturesResult(
                ok=False,
                error_code=FeaturesErrorCode.MODEL_OOM,
                message=f"Out of memory loading audio: {e}",
            )
        logger.error("Failed to load audio for asset_id=%s: %s", asset_id, e)
        return FeaturesResult(
            ok=False,
            error_code=FeaturesErrorCode.FEATURE_EXTRACTION_FAILED,
            message=f"Failed to load audio: {e}",
        )

    # --- Compute mel spectrogram ---
    try:
        mel = _compute_mel_spectrogram(audio, sr)
    except MemoryError:
        logger.error("OOM during mel computation for asset_id=%s", asset_id)
        return FeaturesResult(
            ok=False,
            error_code=FeaturesErrorCode.MODEL_OOM,
            message="Out of memory during mel spectrogram computation",
        )
    except Exception as e:
        logger.error("Mel computation failed for asset_id=%s: %s", asset_id, e)
        return FeaturesResult(
            ok=False,
            error_code=FeaturesErrorCode.FEATURE_EXTRACTION_FAILED,
            message=f"Mel spectrogram computation failed: {e}",
        )

    # --- Compute YAMNet embeddings ---
    try:
        import onnxruntime as ort

        # Create inference session (CPU only)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1  # Limit threads for determinism
        onnx_session = ort.InferenceSession(
            str(YAMNET_ONNX_PATH),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        embeddings = _compute_yamnet_embeddings(audio, sr, onnx_session)

    except MemoryError:
        logger.error("OOM during ONNX inference for asset_id=%s", asset_id)
        return FeaturesResult(
            ok=False,
            error_code=FeaturesErrorCode.MODEL_OOM,
            message="Out of memory during ONNX inference",
        )
    except Exception as e:
        error_str = str(e).lower()
        # Check for OOM-like errors from onnxruntime
        if "memory" in error_str or "oom" in error_str or "alloc" in error_str:
            logger.error("ONNX OOM for asset_id=%s: %s", asset_id, e)
            return FeaturesResult(
                ok=False,
                error_code=FeaturesErrorCode.MODEL_OOM,
                message=f"ONNX out of memory: {e}",
            )
        logger.error("ONNX inference failed for asset_id=%s: %s", asset_id, e)
        return FeaturesResult(
            ok=False,
            error_code=FeaturesErrorCode.FEATURE_EXTRACTION_FAILED,
            message=f"ONNX inference failed: {e}",
        )

    # --- Validate features ---
    valid, error_code, nan_inf_count = _validate_features(mel, embeddings)
    if not valid:
        logger.error(
            "Feature validation failed for asset_id=%s: %s (nan_inf_count=%d)",
            asset_id,
            error_code,
            nan_inf_count,
        )
        return FeaturesResult(
            ok=False,
            error_code=error_code,
            message=f"Feature validation failed: {error_code}",
            metrics={"nan_inf_count": nan_inf_count},
        )

    # --- Write HDF5 with atomic publish ---
    try:
        _write_hdf5_atomic(
            output_path,
            mel,
            embeddings,
            asset_id,
            feature_spec_id,
            spec_alias,
            model_sha256,
        )
    except Exception as e:
        logger.error("Failed to write HDF5 for asset_id=%s: %s", asset_id, e)
        return FeaturesResult(
            ok=False,
            error_code=FeaturesErrorCode.FEATURE_EXTRACTION_FAILED,
            message=f"Failed to write HDF5: {e}",
        )

    # --- Success ---
    feature_time_ms = int((time.monotonic() - start_time) * 1000)

    metrics = {
        "feature_spec_id": feature_spec_id,
        "feature_spec_alias": spec_alias,
        "mel_shape": list(mel.shape),
        "embedding_shape": list(embeddings.shape),
        "nan_inf_count": 0,
        "feature_time_ms": feature_time_ms,
    }

    logger.info(
        "Features extracted for asset_id=%s: mel=%s, embeddings=%s, time=%dms",
        asset_id,
        mel.shape,
        embeddings.shape,
        feature_time_ms,
    )

    return FeaturesResult(
        ok=True,
        message="Feature extraction completed successfully",
        metrics=metrics,
        artifact_path=str(output_path),
        artifact_type=ARTIFACT_TYPE_FEATURES_H5,
        schema_version=ARTIFACT_SCHEMA_VERSION,
        feature_spec_alias=spec_alias,
    )


# --- Standalone Execution ---


def run_features_worker(asset_id: str) -> FeaturesResult:
    """Run the features worker for an asset.

    This is the top-level function that initializes the database
    and calls extract_features.

    Args:
        asset_id: The asset ID to process.

    Returns:
        FeaturesResult with success/failure status and metrics.
    """
    # Best-effort cleanup of orphan temp files at startup
    try:
        removed = cleanup_orphan_h5_tmp()
        if removed > 0:
            logger.info("Cleaned up %d orphan temp file(s) at startup", removed)
    except Exception as e:
        logger.warning("Failed to cleanup orphan temp files at startup: %s", e)

    _, SessionFactory = init_db()
    session = SessionFactory()

    try:
        result = extract_features(session, asset_id)
        session.commit()
        return result
    except Exception as e:
        session.rollback()
        logger.exception("Features worker failed for asset_id=%s", asset_id)
        return FeaturesResult(
            ok=False,
            error_code=FeaturesErrorCode.FEATURE_EXTRACTION_FAILED,
            message=str(e),
        )
    finally:
        session.close()


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <asset_id>")
        sys.exit(1)

    result = run_features_worker(sys.argv[1])
    if result.ok:
        print(f"Success: {result.artifact_path}")
        print(f"Metrics: {result.metrics}")
        sys.exit(0)
    else:
        print(f"Error: {result.error_code} - {result.message}")
        sys.exit(1)
