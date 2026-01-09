"""SSIS Audio Pipeline - Utility modules."""

from app.utils.atomic_io import atomic_write_bytes, atomic_write_text
from app.utils.hashing import feature_spec_alias, sha256_bytes, sha256_file
from app.utils.paths import (
    audio_normalized_path,
    audio_original_path,
    features_h5_path,
    preview_json_path,
    segments_json_path,
)

__all__ = [
    # atomic_io
    "atomic_write_bytes",
    "atomic_write_text",
    # hashing
    "sha256_file",
    "sha256_bytes",
    "feature_spec_alias",
    # paths
    "audio_original_path",
    "audio_normalized_path",
    "features_h5_path",
    "segments_json_path",
    "preview_json_path",
]
