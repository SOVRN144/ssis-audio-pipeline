"""SSIS Audio Pipeline - Canonical path utilities.

Returns canonical Paths per Blueprint section 4. Does NOT create directories.
Directory creation is the responsibility of the calling code.
"""

from pathlib import Path

from app.config import AUDIO_DIR, FEATURES_DIR, PREVIEW_DIR, SEGMENTS_DIR


def audio_original_path(asset_id: str, ext: str) -> Path:
    """Get canonical path for original audio file.

    Args:
        asset_id: Unique asset identifier.
        ext: File extension (without leading dot, e.g., "mp3", "wav").

    Returns:
        Path: data/audio/{asset_id}/original.{ext}
    """
    # Normalize extension (remove leading dot if present)
    ext = ext.lstrip(".")
    return AUDIO_DIR / asset_id / f"original.{ext}"


def audio_normalized_path(asset_id: str) -> Path:
    """Get canonical path for normalized WAV file.

    Args:
        asset_id: Unique asset identifier.

    Returns:
        Path: data/audio/{asset_id}/normalized.wav
    """
    return AUDIO_DIR / asset_id / "normalized.wav"


def features_h5_path(asset_id: str, feature_spec_alias: str) -> Path:
    """Get canonical path for feature pack HDF5 file.

    Args:
        asset_id: Unique asset identifier.
        feature_spec_alias: 12-char hex alias from feature_spec_alias().

    Returns:
        Path: data/features/{asset_id}.{feature_spec_alias}.h5
    """
    return FEATURES_DIR / f"{asset_id}.{feature_spec_alias}.h5"


def segments_json_path(asset_id: str) -> Path:
    """Get canonical path for segments JSON file.

    Args:
        asset_id: Unique asset identifier.

    Returns:
        Path: data/segments/{asset_id}.segments.v1.json
    """
    return SEGMENTS_DIR / f"{asset_id}.segments.v1.json"


def preview_json_path(asset_id: str) -> Path:
    """Get canonical path for preview JSON file.

    Args:
        asset_id: Unique asset identifier.

    Returns:
        Path: data/preview/{asset_id}.preview.v1.json
    """
    return PREVIEW_DIR / f"{asset_id}.preview.v1.json"
