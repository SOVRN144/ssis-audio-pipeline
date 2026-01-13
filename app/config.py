"""SSIS Audio Pipeline - Configuration constants.

Minimal configuration for Step 1. No external config libraries.
All paths are relative to the repository root by default.
"""

from pathlib import Path

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

# Stage lock TTL in seconds (Blueprint section 7: ~10 minutes)
STAGE_LOCK_TTL_SECONDS = 600

# Canonical audio format (Blueprint section 1)
CANONICAL_SAMPLE_RATE = 22050
CANONICAL_CHANNELS = 1

# Default feature spec ID (Blueprint v1.4 locked default)
DEFAULT_FEATURE_SPEC_ID = "mel64_h10ms_w25ms_sr22050__yamnet1024_h0.5s_onnx"

# Retry policy (Blueprint section 9 - LOCKED)
# Maximum attempts before dead-letter
MAX_RETRY_ATTEMPTS = 3
# Retry delays in seconds: 60s, 300s (5min), 900s (15min)
RETRY_DELAYS_SECONDS = (60, 300, 900)
