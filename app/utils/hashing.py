"""SSIS Audio Pipeline - Hashing utilities.

All hash functions return HEX DIGEST ONLY (no prefix).
Per task spec: if a prefix is needed later, add it at storage/serialization layer.
"""

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    """Compute SHA256 hash of a file.

    Args:
        path: Path to the file to hash.

    Returns:
        SHA256 hex digest (64 lowercase hex characters, no prefix).

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file cannot be read.
    """
    path = Path(path)
    hasher = hashlib.sha256()

    # Read in chunks for memory efficiency with large files
    with open(path, "rb") as f:
        while chunk := f.read(65536):  # 64KB chunks
            hasher.update(chunk)

    return hasher.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute SHA256 hash of bytes.

    Args:
        data: Bytes to hash.

    Returns:
        SHA256 hex digest (64 lowercase hex characters, no prefix).
    """
    return hashlib.sha256(data).hexdigest()


def feature_spec_alias(feature_spec_id: str) -> str:
    """Compute the short alias for a feature spec ID.

    Per Blueprint section 5:
    - feature_spec_alias = first 12 chars of sha256(feature_spec_id)

    Args:
        feature_spec_id: Human-readable canonical feature spec identifier.

    Returns:
        12-character hex string alias.
    """
    full_hash = sha256_bytes(feature_spec_id.encode("utf-8"))
    return full_hash[:12]
