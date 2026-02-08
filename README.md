# SSIS Audio Pipeline

## Overview

SSIS Audio Pipeline (Blueprint #1) is a multi-stage, resilient audio processing system designed for streaming segmentation and analysis. The pipeline accepts raw audio files, normalizes them, extracts acoustic features, identifies speech/music segments, generates compact preview representations, and maintains telemetry for each processing job. Built for local-first operation without cloud dependencies, it emphasizes atomic file operations, contract-driven development, and comprehensive testing across contract, end-to-end, and resilience dimensions.

## Runtime Artifacts (Blueprint #1 v1.4)

The pipeline currently produces and indexes the following artifacts:

- **AudioAsset** metadata rows in `audio_assets`
- **normalized.wav** at `data/audio/{asset_id}/normalized.wav`
- **FeaturePack HDF5** at `data/features/{asset_id}.{feature_spec_alias}.h5`
- **FeaturePack metadata JSON** at `data/features/{asset_id}.{feature_spec_alias}.feature_pack.v1.json`
- **segments JSON** at `data/segments/{asset_id}.segments.v1.json`
- **preview JSON** at `data/preview/{asset_id}.preview.v1.json`
- **pipeline_jobs** stage telemetry rows
- **artifact_index** rows for published artifacts
- **feature_specs** registry rows for alias/spec immutability

## Implementation Status

- Ingest API, orchestrator, and all stage workers (`decode/features/segments/preview`) are implemented.
- Atomic publish is used for final artifact writes.
- Periodic Huey sweep enqueues `orchestrator_tick_task(asset_id)` for assets needing work.

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Git
- ffmpeg (required for decode worker)

#### Installing ffmpeg

The decode worker requires ffmpeg to be installed and available in PATH.

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html and add to PATH.

> **Note:** Tests mock ffmpeg calls and do not require ffmpeg to be installed.

### Installation

```bash
# Clone the repository
git clone https://github.com/SOVRN144/ssis-audio-pipeline.git
cd ssis-audio-pipeline

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

#### TensorFlow/Keras Verification

TensorFlow and Keras install with the base dependencies, so after running `pip install -e ".[dev]"` inside the virtual environment, verify the pinned versions with:

```bash
.venv/bin/python -c "import tensorflow as tf; import keras; print(tf.__version__); print(keras.__version__)"
```

Expected output should report `2.15.1` for TensorFlow and `2.15.0` for Keras. This guards against accidental upgrades to TensorFlow 2.20+ / Keras 3.x.

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test directory
pytest tests/contract/
```

### Huey Orchestrator Sweep

Run the Huey consumer with periodic tasks enabled:

```bash
huey_consumer.py app.huey_app.huey
```

The consumer executes `orchestrator_sweep_task` once per minute. The sweep looks
for assets that have a completed ingest job, have not been dead-lettered, and
still have pending pipeline stages. Up to 50 assets per sweep are enqueued via
`orchestrator_tick_task(asset_id)`, which prevents Huey from attempting to call
the tick task with missing arguments while still driving stage progression.

`huey_consumer.py app.huey_app.huey -S` is also valid; `-S` enables simple logging and does not disable periodic tasks. Use `-n` only if you want to disable periodic scheduling.

### End-to-End Local Runbook (3 Terminals)

Terminal A (API):

```bash
source .venv/bin/activate
export PYTHONUNBUFFERED=1
uvicorn services.ingest_api.main:app --host 127.0.0.1 --port 8001 --reload
```

Terminal B (Huey consumer + periodic sweep):

```bash
source .venv/bin/activate
export PYTHONUNBUFFERED=1
huey_consumer.py app.huey_app.huey -S
```

Terminal C (ingest + verification):

```bash
source .venv/bin/activate
ffmpeg -y -i tmp/demo_input_unique.wav -filter:a "apad=pad_dur=0.37,volume=0.98" -t 8.62 tmp/restart_unique2.wav
ABS_WAV="$(pwd)/tmp/restart_unique2.wav"
curl -s -X POST http://127.0.0.1:8001/v1/ingest/local \
  -H "Content-Type: application/json" \
  -d "{\"source_path\":\"$ABS_WAV\",\"owner_entity_id\":\"demo-local\",\"original_filename\":\"restart_unique2.wav\",\"metadata\":{\"test\":\"blueprint1.4_restart\"}}" \
| tee /tmp/ingest_resp_restart2.json

ASSET_ID="$(python - <<'PY'
import json
print(json.load(open("/tmp/ingest_resp_restart2.json"))["asset_id"])
PY
)"
echo "ASSET_ID=$ASSET_ID"

sqlite3 data/ssis.db "
select stage,status,attempt,created_at,job_id,error_code,substr(error_message,1,120)
from pipeline_jobs
where asset_id='$ASSET_ID'
order by id;"

ls -lh data/audio/$ASSET_ID/normalized.wav
ls -lh data/features/$ASSET_ID*.h5
ls -lh data/segments/$ASSET_ID.segments.v1.json
ls -lh data/preview/$ASSET_ID.preview.v1.json

sqlite3 data/ssis.db "
select artifact_type, feature_spec_alias, schema_version, created_at, artifact_path
from artifact_index
where asset_id='$ASSET_ID'
order by created_at;"
```

Feature alias note: if `SSIS_ACTIVE_FEATURE_SPEC_ALIAS` is set but the asset does not have a pack for that alias, preview falls back to the default alias when that default pack exists.

### Code Quality

```bash
# Run linter
ruff check .

# Auto-fix linting issues
ruff check --fix .

# Format code
ruff format .

# Type checking (optional)
mypy app/ services/
```

### Optional: Global git ignore for macOS/editor artifacts

1. Copy the repo file to a global ignore location:
   `cp .gitignore_global ~/.gitignore_global`
2. Tell git to use it:
   `git config --global core.excludesFile ~/.gitignore_global`

## Project Structure

```
ssis-audio-pipeline/
├── .claude/agents/         # Claude Code project-level subagents
├── app/                    # Core application modules
│   └── utils/              # Shared utilities
├── docs/                   # Human-readable documentation
│   └── blueprints/         # Blueprint PDFs and research docs
├── services/               # Service components
│   ├── ingest_api/         # Audio ingestion endpoint
│   ├── worker_decode/      # Audio normalization worker
│   ├── worker_features/    # Feature extraction worker
│   ├── worker_segments/    # Segmentation worker
│   └── worker_preview/     # Preview generation worker
├── tests/                  # Test suites
│   ├── contract/           # Contract tests
│   ├── e2e/                # End-to-end tests
│   └── resilience/         # Resilience tests
├── specs/                  # JSON Schemas
├── data/                   # Local data storage (gitignored)
├── logs/                   # Application logs (gitignored)
└── CLAUDE.md               # Repo operating rules for AI agents
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Documentation

Blueprint specification documents are available in `docs/blueprints/`:
- SSIS Blueprint #1 v1.4.pdf
- SSIS Research Pack v1.0.pdf
- SSIS Blueprint #1 v1.4 Checklist.pdf

`specs/` contains machine-readable JSON Schemas used by tests and runtime validation gates.
