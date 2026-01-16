"""SSIS Audio Pipeline - Checkpoint utilities.

Provides worker checkpointing for resumable processing per Blueprint section 8.
Checkpoints enable decode/features workers to resume after power loss without
starting from scratch.

Checkpoint files are JSON documents with a schema_version for forward compatibility.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Schema version for checkpoint files (string per Blueprint versioned contracts)
CHECKPOINT_SCHEMA_VERSION = "1.0.0"

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "save_checkpoint",
    "load_checkpoint",
    "delete_checkpoint",
]


def save_checkpoint(
    checkpoint_path: str | Path,
    data: dict[str, Any],
) -> None:
    """Save checkpoint data to a JSON file.

    Overwrites any existing checkpoint at the same path.
    The schema_version is automatically added to the checkpoint data.

    Note: Checkpoints are NOT atomically published because they are
    intermediate state, not final artifacts. A partial checkpoint is
    recoverable (we restart from zero if checkpoint is corrupt).

    Args:
        checkpoint_path: Path to the checkpoint file.
        data: Checkpoint data dictionary. Must be JSON-serializable.
              Should NOT include 'schema_version' key (added automatically).
    """
    checkpoint_path = Path(checkpoint_path)

    # Ensure parent directory exists
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Add schema version
    checkpoint_data = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        **data,
    }

    # Write checkpoint (not atomic - intermediate state)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)
        f.flush()

    logger.debug("Saved checkpoint to %s", checkpoint_path)


def load_checkpoint(
    checkpoint_path: str | Path,
) -> dict[str, Any] | None:
    """Load checkpoint data from a JSON file.

    Returns None if the checkpoint does not exist or is corrupt/unparseable.
    Corrupt checkpoints are treated as "no checkpoint" - caller should
    restart from zero.

    Args:
        checkpoint_path: Path to the checkpoint file.

    Returns:
        Checkpoint data dictionary if valid, None if missing or corrupt.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        logger.debug("No checkpoint found at %s", checkpoint_path)
        return None

    try:
        with open(checkpoint_path, encoding="utf-8") as f:
            data = json.load(f)

        # Validate basic structure
        if not isinstance(data, dict):
            logger.warning("Checkpoint at %s is not a dict, treating as corrupt", checkpoint_path)
            return None

        # Check schema version for compatibility
        schema_version = data.get("schema_version")
        if schema_version != CHECKPOINT_SCHEMA_VERSION:
            logger.warning(
                "Checkpoint at %s has schema_version %s (expected %s), treating as incompatible",
                checkpoint_path,
                schema_version,
                CHECKPOINT_SCHEMA_VERSION,
            )
            return None

        logger.debug("Loaded checkpoint from %s", checkpoint_path)
        return data

    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load checkpoint from %s: %s", checkpoint_path, e)
        return None


def delete_checkpoint(
    checkpoint_path: str | Path,
) -> bool:
    """Delete a checkpoint file if it exists.

    Safe to call even if checkpoint does not exist.

    Args:
        checkpoint_path: Path to the checkpoint file.

    Returns:
        True if checkpoint was deleted, False if it did not exist.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        logger.debug("No checkpoint to delete at %s", checkpoint_path)
        return False

    try:
        checkpoint_path.unlink()
        logger.debug("Deleted checkpoint at %s", checkpoint_path)
        return True
    except OSError as e:
        logger.warning("Failed to delete checkpoint at %s: %s", checkpoint_path, e)
        return False
