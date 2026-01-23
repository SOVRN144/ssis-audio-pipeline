"""SSIS Audio Pipeline - Decode Worker.

Produces the canonical normalized WAV for an ingested asset.

Stage: decode (STAGE_DECODE)
Input: AudioAsset.source_uri
Output: data/audio/{asset_id}/normalized.wav

Canonical format: 22050 Hz, mono, 16-bit PCM WAV

Resilience features:
- Chunked processing with checkpointing (~60s chunks)
- Safe resume after power loss
- Atomic publish for final WAV only

Dependencies:
- Requires ffmpeg installed and in PATH

Error codes (Blueprint section 8):
- CODEC_UNSUPPORTED: ffmpeg cannot decode the format
- FILE_CORRUPT: source file is corrupt or unreadable
- FILE_TOO_SHORT: decoded audio is below minimum duration
- INPUT_NOT_FOUND: source file does not exist

Failpoints (Step 8 resilience harness):
- DECODE_AFTER_CHUNK_WRITE: After writing a PCM chunk, before checkpoint
- DECODE_AFTER_CHECKPOINT: After saving checkpoint
- DECODE_BEFORE_FINAL_RENAME: Before atomic rename of final WAV
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import select

from app.config import CANONICAL_CHANNELS, CANONICAL_SAMPLE_RATE
from app.db import init_db
from app.models import AudioAsset
from app.orchestrator import ARTIFACT_TYPE_NORMALIZED_WAV
from app.utils.checkpoints import (
    delete_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from app.utils.failpoints import maybe_fail
from app.utils.paths import audio_normalized_path

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# --- Constants ---

# Chunk duration in seconds for chunked processing
CHUNK_SECONDS = 60

# Minimum duration threshold in seconds
# Audio shorter than this is considered an error
MIN_DURATION_SEC = 1.7

# Canonical audio parameters
SAMPLE_RATE = CANONICAL_SAMPLE_RATE  # 22050
CHANNELS = CANONICAL_CHANNELS  # 1
SAMPWIDTH = 2  # 16-bit = 2 bytes

# Schema version for normalized WAV artifact
ARTIFACT_SCHEMA_VERSION = "1.0.0"

# ffmpeg subprocess timeout in seconds
# Must be < 600s lock TTL to allow orchestrator retry on timeout
FFMPEG_TIMEOUT_SECONDS = 300  # 5 minutes per chunk

# Fraction of chunk size to accept as valid on final chunk
EOF_CHUNK_TOLERANCE = 0.9

# File naming conventions
PCM_TEMP_SUFFIX = ".pcm.tmp"
CHECKPOINT_SUFFIX = ".ckpt.json"
WAV_TEMP_SUFFIX = ".wav.tmp"


# --- Error Codes ---


class DecodeErrorCode:
    """Error codes for decode stage per Blueprint section 8."""

    CODEC_UNSUPPORTED = "CODEC_UNSUPPORTED"
    FILE_CORRUPT = "FILE_CORRUPT"
    FILE_TOO_SHORT = "FILE_TOO_SHORT"
    INPUT_NOT_FOUND = "INPUT_NOT_FOUND"
    WORKER_ERROR = "WORKER_ERROR"


# --- Result Types ---


@dataclass
class DecodeMetrics:
    """Metrics collected during decode processing."""

    output_duration_sec: float = 0.0
    chunk_count: int = 0
    decode_time_ms: int = 0


@dataclass
class DecodeResult:
    """Result of decode worker execution."""

    ok: bool
    error_code: str | None = None
    message: str | None = None
    metrics: DecodeMetrics = field(default_factory=DecodeMetrics)
    artifact_path: str | None = None
    artifact_type: str | None = None
    schema_version: str | None = None


# --- Path Helpers ---


def _get_decode_paths(asset_id: str) -> dict[str, Path]:
    """Get all paths used during decode processing.

    Args:
        asset_id: The asset ID.

    Returns:
        Dictionary with keys: normalized_wav, pcm_tmp, checkpoint, wav_tmp
    """
    normalized_wav = audio_normalized_path(asset_id)
    parent = normalized_wav.parent

    return {
        "normalized_wav": normalized_wav,
        "pcm_tmp": parent / f"normalized{PCM_TEMP_SUFFIX}",
        "checkpoint": parent / f"normalized{CHECKPOINT_SUFFIX}",
        "wav_tmp": parent / f"normalized{WAV_TEMP_SUFFIX}",
    }


# --- Cleanup Helpers ---


def _cleanup_temp_files(paths: dict[str, Path]) -> None:
    """Clean up all temporary files.

    Args:
        paths: Dictionary of paths from _get_decode_paths.
    """
    for key in ["pcm_tmp", "checkpoint", "wav_tmp"]:
        path = paths.get(key)
        if path and path.exists():
            try:
                path.unlink()
                logger.debug("Cleaned up %s", path)
            except OSError as e:
                logger.warning("Failed to clean up %s: %s", path, e)


def _cleanup_orphan_wav_tmp(paths: dict[str, Path]) -> None:
    """Clean up orphan .wav.tmp file if present.

    Args:
        paths: Dictionary of paths from _get_decode_paths.
    """
    wav_tmp = paths.get("wav_tmp")
    if wav_tmp and wav_tmp.exists():
        try:
            wav_tmp.unlink()
            logger.debug("Cleaned up orphan wav.tmp: %s", wav_tmp)
        except OSError:
            pass


# --- ffmpeg Subprocess ---


def _decode_chunk(
    input_path: Path,
    start_seconds: float,
    chunk_seconds: float,
) -> tuple[bytes, str | None]:
    """Decode a chunk of audio using ffmpeg.

    Args:
        input_path: Path to the input audio file.
        start_seconds: Start time in seconds.
        chunk_seconds: Duration to decode in seconds.

    Returns:
        Tuple of (pcm_bytes, error_code).
        On success: (bytes, None)
        On error: (b"", error_code)
    """
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        str(start_seconds),
        "-t",
        str(chunk_seconds),
        "-i",
        str(input_path),
        "-ac",
        str(CHANNELS),
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "s16le",
        "-",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            check=False,
            timeout=FFMPEG_TIMEOUT_SECONDS,
        )

        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").lower()

            # Check for codec/format issues
            # Note: "no such file" refers to ffmpeg's internal codec/format lookup
            # errors (e.g., missing decoder library), not missing input files.
            # Input file existence is validated before _decode_chunk is called.
            if any(
                x in stderr
                for x in [
                    "decoder",
                    "codec",
                    "unsupported",
                    "invalid data",
                    "no such file",
                    "unknown format",
                ]
            ):
                return b"", DecodeErrorCode.CODEC_UNSUPPORTED

            # Other errors are file corruption
            return b"", DecodeErrorCode.FILE_CORRUPT

        return result.stdout, None

    except subprocess.TimeoutExpired:
        # Deterministic worker failure. Orchestrator will retry via Step 3 retry policy.
        logger.error(
            "ffmpeg timed out after %d seconds at position %.2fs",
            FFMPEG_TIMEOUT_SECONDS,
            start_seconds,
        )
        return b"", DecodeErrorCode.WORKER_ERROR
    except FileNotFoundError:
        # ffmpeg not installed
        logger.error("ffmpeg not found in PATH")
        return b"", DecodeErrorCode.WORKER_ERROR
    except OSError as e:
        logger.error("ffmpeg execution failed: %s", e)
        return b"", DecodeErrorCode.WORKER_ERROR


# --- Checkpoint Management ---


def _save_decode_checkpoint(
    checkpoint_path: Path,
    seconds_processed: float,
    pcm_tmp_name: str,
) -> None:
    """Save decode checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        seconds_processed: Seconds of audio processed so far.
        pcm_tmp_name: Basename of the PCM temp file.
    """
    data = {
        "seconds_processed": seconds_processed,
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "sampwidth": SAMPWIDTH,
        "pcm_tmp_name": pcm_tmp_name,
    }
    save_checkpoint(checkpoint_path, data)


def _load_decode_checkpoint(checkpoint_path: Path) -> dict | None:
    """Load and validate decode checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.

    Returns:
        Checkpoint data if valid, None otherwise.
    """
    data = load_checkpoint(checkpoint_path)
    if data is None:
        return None

    # Validate required fields
    required = ["seconds_processed", "sample_rate", "channels", "sampwidth", "pcm_tmp_name"]
    for field_name in required:
        if field_name not in data:
            logger.warning("Checkpoint missing field: %s", field_name)
            return None

    # Validate canonical params match
    if data["sample_rate"] != SAMPLE_RATE:
        logger.warning("Checkpoint sample_rate mismatch")
        return None
    if data["channels"] != CHANNELS:
        logger.warning("Checkpoint channels mismatch")
        return None
    if data["sampwidth"] != SAMPWIDTH:
        logger.warning("Checkpoint sampwidth mismatch")
        return None

    return data


# --- WAV Wrapping ---


def _wrap_pcm_to_wav(pcm_path: Path, wav_path: Path) -> float:
    """Wrap raw PCM data into a WAV file.

    Note: Current implementation reads full PCM file into memory.
    Acceptable for Step 4 baseline; streaming wrap can be added
    later if needed for multi-hour assets.

    Args:
        pcm_path: Path to raw PCM file.
        wav_path: Path for output WAV file.

    Returns:
        Duration of the WAV file in seconds.
    """
    # Read all PCM data
    pcm_data = pcm_path.read_bytes()

    # Calculate duration
    bytes_per_sample = SAMPWIDTH * CHANNELS
    num_samples = len(pcm_data) // bytes_per_sample
    duration_sec = num_samples / SAMPLE_RATE

    # Write WAV file
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPWIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data)

    return duration_sec


def _atomic_publish_wav(wav_tmp_path: Path, final_path: Path) -> None:
    """Atomically publish WAV file.

    Performs fsync on the temp file, then atomic rename.

    Args:
        wav_tmp_path: Path to temporary WAV file.
        final_path: Path for final WAV file.
    """
    # Ensure parent directory exists
    final_path.parent.mkdir(parents=True, exist_ok=True)

    # fsync the temp file
    fd = os.open(wav_tmp_path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

    # Failpoint: before final rename
    maybe_fail("DECODE_BEFORE_FINAL_RENAME")

    # Atomic rename
    os.replace(wav_tmp_path, final_path)

    # Best-effort fsync on directory
    try:
        dir_fd = os.open(final_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except (OSError, AttributeError):
        pass  # Best-effort


# --- Main Decode Logic ---


def decode_asset(
    session: Session,
    asset_id: str,
) -> DecodeResult:
    """Decode an audio asset to canonical normalized WAV.

    This is the main entry point for the decode worker.
    Implements chunked processing with checkpointing for resilience.

    Args:
        session: Database session for querying AudioAsset.
        asset_id: The asset ID to decode.

    Returns:
        DecodeResult with success/failure status and metrics.
    """
    paths = _get_decode_paths(asset_id)
    metrics = DecodeMetrics()

    # --- Resume Logic ---

    # 1. If normalized.wav already exists, return success (idempotent)
    if paths["normalized_wav"].exists():
        logger.info("Normalized WAV already exists for asset_id=%s", asset_id)
        # Read existing WAV to get duration for metrics
        try:
            with wave.open(str(paths["normalized_wav"]), "rb") as wf:
                duration = wf.getnframes() / wf.getframerate()
                metrics.output_duration_sec = duration
        except (OSError, wave.Error):
            pass
        return DecodeResult(
            ok=True,
            message="Artifact already exists",
            metrics=metrics,
            artifact_path=str(paths["normalized_wav"]),
            artifact_type=ARTIFACT_TYPE_NORMALIZED_WAV,
            schema_version=ARTIFACT_SCHEMA_VERSION,
        )

    # 2. Clean up orphan .wav.tmp
    _cleanup_orphan_wav_tmp(paths)

    # 3. Query AudioAsset to get source_uri
    stmt = select(AudioAsset).where(AudioAsset.asset_id == asset_id)
    asset = session.execute(stmt).scalar_one_or_none()

    if asset is None:
        logger.error("AudioAsset not found for asset_id=%s", asset_id)
        return DecodeResult(
            ok=False,
            error_code=DecodeErrorCode.INPUT_NOT_FOUND,
            message=f"AudioAsset not found: {asset_id}",
        )

    input_path = Path(asset.source_uri)
    if not input_path.exists():
        logger.error("Source file not found: %s", input_path)
        return DecodeResult(
            ok=False,
            error_code=DecodeErrorCode.INPUT_NOT_FOUND,
            message=f"Source file not found: {input_path}",
        )

    # 4. Determine resume state
    start_seconds = 0.0
    checkpoint_data = None

    pcm_tmp_exists = paths["pcm_tmp"].exists()
    checkpoint_exists = paths["checkpoint"].exists()

    if pcm_tmp_exists and checkpoint_exists:
        # Try to resume from checkpoint
        checkpoint_data = _load_decode_checkpoint(paths["checkpoint"])
        if checkpoint_data is not None:
            start_seconds = checkpoint_data["seconds_processed"]
            logger.info("Resuming decode from checkpoint: %.2f seconds processed", start_seconds)
        else:
            # Checkpoint invalid - restart from scratch
            logger.warning("Invalid checkpoint, restarting decode from beginning")
            paths["pcm_tmp"].unlink()
    elif pcm_tmp_exists and not checkpoint_exists:
        # PCM exists but no checkpoint - corrupted state, restart
        logger.warning("PCM temp exists without checkpoint, restarting decode")
        paths["pcm_tmp"].unlink()

    # Ensure parent directory exists
    paths["pcm_tmp"].parent.mkdir(parents=True, exist_ok=True)

    # --- Chunked Decode Loop ---

    chunk_count = 0
    total_decode_time_ms = 0
    is_first_chunk = start_seconds == 0.0

    # Open PCM temp file in append mode if resuming, write mode if starting fresh
    mode = "ab" if start_seconds > 0 else "wb"

    try:
        with open(paths["pcm_tmp"], mode) as pcm_file:
            current_pos = start_seconds

            while True:
                chunk_start_time = time.monotonic()

                # Decode chunk
                pcm_bytes, error_code = _decode_chunk(input_path, current_pos, CHUNK_SECONDS)

                chunk_time_ms = int((time.monotonic() - chunk_start_time) * 1000)
                total_decode_time_ms += chunk_time_ms

                # Handle errors
                if error_code is not None:
                    logger.error("Decode error at %.2f seconds: %s", current_pos, error_code)
                    _cleanup_temp_files(paths)
                    return DecodeResult(
                        ok=False,
                        error_code=error_code,
                        message=f"Decode failed at {current_pos:.2f}s",
                        metrics=DecodeMetrics(
                            chunk_count=chunk_count,
                            decode_time_ms=total_decode_time_ms,
                        ),
                    )

                # Check for empty output
                if len(pcm_bytes) == 0:
                    if is_first_chunk and chunk_count == 0:
                        # Empty on first chunk = corrupt file
                        logger.error("Empty output on first chunk - file corrupt")
                        _cleanup_temp_files(paths)
                        return DecodeResult(
                            ok=False,
                            error_code=DecodeErrorCode.FILE_CORRUPT,
                            message="Empty output on first chunk",
                            metrics=DecodeMetrics(
                                chunk_count=chunk_count,
                                decode_time_ms=total_decode_time_ms,
                            ),
                        )
                    else:
                        # Empty on later chunk = EOF, we're done
                        logger.debug("EOF reached after %d chunks", chunk_count)
                        break

                # Write PCM bytes
                pcm_file.write(pcm_bytes)
                pcm_file.flush()

                # Failpoint: after chunk write, before checkpoint
                maybe_fail("DECODE_AFTER_CHUNK_WRITE")

                chunk_count += 1
                is_first_chunk = False

                # Update position
                bytes_per_sample = SAMPWIDTH * CHANNELS
                chunk_samples = len(pcm_bytes) // bytes_per_sample
                chunk_duration = chunk_samples / SAMPLE_RATE
                current_pos += chunk_duration

                # Save checkpoint after each chunk
                _save_decode_checkpoint(
                    paths["checkpoint"],
                    current_pos,
                    paths["pcm_tmp"].name,
                )

                # Failpoint: after checkpoint save
                maybe_fail("DECODE_AFTER_CHECKPOINT")

                logger.debug("Chunk %d complete: %.2f seconds processed", chunk_count, current_pos)

                # If we got less than a full chunk, we've reached EOF
                if chunk_duration < CHUNK_SECONDS * EOF_CHUNK_TOLERANCE:  # Allow 10% tolerance
                    logger.debug("Short chunk detected, EOF reached")
                    break

    except OSError as e:
        logger.error("Failed to write PCM temp file: %s", e)
        _cleanup_temp_files(paths)
        return DecodeResult(
            ok=False,
            error_code=DecodeErrorCode.WORKER_ERROR,
            message=f"Failed to write temp file: {e}",
            metrics=DecodeMetrics(
                chunk_count=chunk_count,
                decode_time_ms=total_decode_time_ms,
            ),
        )

    # --- Finalize WAV ---

    # Check minimum duration
    if not paths["pcm_tmp"].exists():
        logger.error("PCM temp file missing after decode loop")
        return DecodeResult(
            ok=False,
            error_code=DecodeErrorCode.WORKER_ERROR,
            message="PCM temp file missing",
        )

    pcm_size = paths["pcm_tmp"].stat().st_size
    bytes_per_sample = SAMPWIDTH * CHANNELS
    total_samples = pcm_size // bytes_per_sample
    output_duration_sec = total_samples / SAMPLE_RATE

    if output_duration_sec < MIN_DURATION_SEC:
        logger.error(
            "Output duration %.2fs below minimum %.2fs",
            output_duration_sec,
            MIN_DURATION_SEC,
        )
        _cleanup_temp_files(paths)
        return DecodeResult(
            ok=False,
            error_code=DecodeErrorCode.FILE_TOO_SHORT,
            message=f"Output duration {output_duration_sec:.2f}s below minimum {MIN_DURATION_SEC}s",
            metrics=DecodeMetrics(
                output_duration_sec=output_duration_sec,
                chunk_count=chunk_count,
                decode_time_ms=total_decode_time_ms,
            ),
        )

    # Wrap PCM to WAV
    try:
        _wrap_pcm_to_wav(paths["pcm_tmp"], paths["wav_tmp"])
    except Exception as e:
        logger.error("Failed to wrap PCM to WAV: %s", e)
        _cleanup_temp_files(paths)
        return DecodeResult(
            ok=False,
            error_code=DecodeErrorCode.WORKER_ERROR,
            message=f"Failed to create WAV: {e}",
            metrics=DecodeMetrics(
                output_duration_sec=output_duration_sec,
                chunk_count=chunk_count,
                decode_time_ms=total_decode_time_ms,
            ),
        )

    # Atomic publish
    try:
        _atomic_publish_wav(paths["wav_tmp"], paths["normalized_wav"])
    except Exception as e:
        logger.error("Failed to publish WAV: %s", e)
        _cleanup_temp_files(paths)
        return DecodeResult(
            ok=False,
            error_code=DecodeErrorCode.WORKER_ERROR,
            message=f"Failed to publish WAV: {e}",
            metrics=DecodeMetrics(
                output_duration_sec=output_duration_sec,
                chunk_count=chunk_count,
                decode_time_ms=total_decode_time_ms,
            ),
        )

    # Clean up temp files on success
    delete_checkpoint(paths["checkpoint"])
    try:
        paths["pcm_tmp"].unlink()
    except OSError:
        pass

    metrics.output_duration_sec = output_duration_sec
    metrics.chunk_count = chunk_count
    metrics.decode_time_ms = total_decode_time_ms

    logger.info(
        "Decode complete for asset_id=%s: %.2fs, %d chunks, %dms decode time",
        asset_id,
        output_duration_sec,
        chunk_count,
        total_decode_time_ms,
    )

    return DecodeResult(
        ok=True,
        message="Decode completed successfully",
        metrics=metrics,
        artifact_path=str(paths["normalized_wav"]),
        artifact_type=ARTIFACT_TYPE_NORMALIZED_WAV,
        schema_version=ARTIFACT_SCHEMA_VERSION,
    )


# --- Standalone Execution ---


def run_decode_worker(asset_id: str) -> DecodeResult:
    """Run the decode worker for an asset.

    This is the top-level function that initializes the database
    and calls decode_asset.

    Args:
        asset_id: The asset ID to decode.

    Returns:
        DecodeResult with success/failure status and metrics.
    """
    _, SessionFactory = init_db()
    session = SessionFactory()

    try:
        return decode_asset(session, asset_id)
    finally:
        session.close()


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <asset_id>")
        sys.exit(1)

    result = run_decode_worker(sys.argv[1])
    if result.ok:
        print(f"Success: {result.artifact_path}")
        print(f"Duration: {result.metrics.output_duration_sec:.2f}s")
        print(f"Chunks: {result.metrics.chunk_count}")
        sys.exit(0)
    else:
        print(f"Error: {result.error_code} - {result.message}")
        sys.exit(1)
