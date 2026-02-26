"""SSIS Audio Pipeline - Configuration constants.

Minimal configuration for Step 1. No external config libraries.
All paths are relative to the repository root by default.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Repository root (parent of app/)
REPO_ROOT = Path(__file__).parent.parent.resolve()

# Data directories (canonical paths per Blueprint section 4)
DATA_DIR = REPO_ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
FEATURES_DIR = DATA_DIR / "features"
SEGMENTS_DIR = DATA_DIR / "segments"
PREVIEW_DIR = DATA_DIR / "preview"

# Logs directory
LOGS_DIR = REPO_ROOT / "logs"

# Database path
DB_PATH = DATA_DIR / "ssis.db"

# Queue directory and Huey database path (Blueprint section 9)
QUEUE_DIR = DATA_DIR / "queue"
HUEY_DB_PATH = QUEUE_DIR / "huey.db"


def _parse_int_env(name: str, default: int, *, min_value: int = 1) -> int:
    """Parse positive integer environment variable with warn+fallback semantics."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; expected integer >= %d, using default=%d",
            name,
            raw,
            min_value,
            default,
        )
        return default
    if value < min_value:
        logger.warning(
            "Invalid %s=%r; expected integer >= %d, using default=%d",
            name,
            raw,
            min_value,
            default,
        )
        return default
    return value


def _get_lock_ttl() -> int:
    """Get lock TTL from environment or use default.

    Environment variable SSIS_LOCK_TTL_SEC allows override for testing.
    Default is 600 seconds (~10 minutes) per Blueprint section 7.

    Returns:
        Lock TTL in seconds.
    """
    return _parse_int_env("SSIS_LOCK_TTL_SEC", 600, min_value=1)


# Stage lock TTL in seconds (Blueprint section 7: ~10 minutes)
# Override with SSIS_LOCK_TTL_SEC environment variable for testing
STAGE_LOCK_TTL_SECONDS = _get_lock_ttl()


def validate_config() -> None:
    """Run non-throwing config validations for warn+fallback settings."""
    _parse_int_env("SSIS_LOCK_TTL_SEC", 600, min_value=1)

# Canonical audio format (Blueprint section 1)
CANONICAL_SAMPLE_RATE = 22050
CANONICAL_CHANNELS = 1

# Default feature spec ID (Blueprint v1.4 locked default)
DEFAULT_FEATURE_SPEC_ID = "mel64_h10ms_w25ms_sr22050__yamnet1024_h0.5s_onnx"

# Retry policy (Blueprint section 9 - LOCKED)
# Retry delays in seconds: 60s, 300s (5min), 900s (15min)
RETRY_DELAYS_SECONDS = (60, 300, 900)
# Total attempts = initial attempt + 3 retries = 4
# After all retries exhausted, job is dead-lettered
MAX_ATTEMPTS_TOTAL = 1 + len(RETRY_DELAYS_SECONDS)
assert MAX_ATTEMPTS_TOTAL == 4, "Blueprint requires exactly 4 total attempts"
