"""SSIS Audio Pipeline - Audio metadata extraction utilities.

Best-effort metadata extraction using only stdlib (wave module for WAV files).
No heavy audio/ML dependencies. Extraction failures never block ingest.
"""

import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AudioMetadata:
    """Audio metadata extracted from file (best-effort).

    All fields may be None if extraction fails or is not supported.
    """

    duration_sec: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    format_guess: str | None = None


def extract_audio_metadata(path: str | Path) -> AudioMetadata:
    """Extract metadata from an audio file using stdlib only.

    Best-effort extraction:
    - For WAV files: uses stdlib wave module
    - For non-WAV files: returns format_guess from extension only

    This function NEVER raises exceptions. Failures are silently
    handled and return partial/empty metadata.

    Args:
        path: Path to the audio file.

    Returns:
        AudioMetadata with available fields filled in.
    """
    path = Path(path)
    ext = path.suffix.lower().lstrip(".")

    # Initialize with format guess from extension
    metadata = AudioMetadata(format_guess=ext if ext else None)

    # Try WAV extraction if it looks like a WAV file
    if ext == "wav":
        try:
            metadata = _extract_wav_metadata(path)
        except Exception:
            # Best-effort: keep format guess on failure
            pass

    return metadata


def _extract_wav_metadata(path: Path) -> AudioMetadata:
    """Extract metadata from a WAV file using stdlib wave module.

    Args:
        path: Path to the WAV file.

    Returns:
        AudioMetadata with WAV-specific fields.

    Raises:
        Exception: If wave module cannot read the file.
    """
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        n_frames = wf.getnframes()

        # Calculate duration
        duration_sec = n_frames / sample_rate if sample_rate > 0 else None

        return AudioMetadata(
            duration_sec=duration_sec,
            sample_rate=sample_rate,
            channels=channels,
            format_guess="wav",
        )


def guess_format_from_extension(filename: str) -> str | None:
    """Guess audio format from filename extension.

    Args:
        filename: Filename or path string.

    Returns:
        Lowercase extension without dot, or None if no extension.
    """
    path = Path(filename)
    ext = path.suffix.lower().lstrip(".")
    return ext if ext else None


__all__ = [
    "AudioMetadata",
    "extract_audio_metadata",
    "guess_format_from_extension",
]
