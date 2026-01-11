"""SSIS Audio Pipeline - Audio metadata extraction utilities.

STUB: Implementation deferred to Step 2.
This module will provide best-effort metadata extraction from audio files.
"""

# TODO (Step 2): Implement best-effort metadata extraction
#
# Blueprint section 8 (Stage A - Ingest) specifies:
# - Extract best-effort metadata: duration, channels, sample_rate
# - This is "best-effort" - failures should not block ingest
#
# Planned interface:
#
# @dataclass
# class AudioMetadata:
#     duration_sec: float | None
#     sample_rate: int | None
#     channels: int | None
#     format_guess: str | None
#
# def extract_audio_metadata(path: Path) -> AudioMetadata:
#     """Extract metadata from audio file. Returns None fields on failure."""
#     pass
#
# Implementation will likely use:
# - mutagen for tag/format detection
# - ffprobe via subprocess as fallback
# - No heavy audio loading (librosa/soundfile) at ingest time
