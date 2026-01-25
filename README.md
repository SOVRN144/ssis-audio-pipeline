# SSIS Audio Pipeline

## Overview

SSIS Audio Pipeline (Blueprint #1) is a multi-stage, resilient audio processing system designed for streaming segmentation and analysis. The pipeline accepts raw audio files, normalizes them, extracts acoustic features, identifies speech/music segments, generates compact preview representations, and maintains telemetry for each processing job. Built for local-first operation without cloud dependencies, it emphasizes atomic file operations, contract-driven development, and comprehensive testing across contract, end-to-end, and resilience dimensions.

## Planned Artifacts (Blueprint #1 v1.4)

The following artifacts are defined by SSIS Blueprint #1 v1.4 and will be implemented in later steps:

- **AudioAsset**: Metadata record tracking the original audio file and processing state
- **normalized.wav**: 22050 Hz mono, 16-bit PCM WAV (canonical derivative)
- **FeaturePack.h5**: HDF5-stored acoustic feature vectors (MFCC, spectral features)
- **segments.json**: Timestamped speech/music classification segments with confidence scores
- **preview.json**: Compact audio summary with statistical features and representative snippets
- **pipeline_jobs**: Telemetry database tracking job state, timing, and error conditions
- **feature_specs**: Registry of feature extraction configurations and versioning

> **Note**: These artifacts are not yet implemented. Step 0 provides repo scaffolding and CI only.

## Development Roadmap

### Step 0: Repository Setup and CI Scaffolding (Current)

- [x] GitHub repository creation
- [x] Directory structure and placeholder files
- [x] Python tooling configuration (pyproject.toml, ruff, mypy)
- [x] CI pipeline (GitHub Actions)
- [x] Blueprint specification documents

### Step 1: Contracts + DB Primitives + Atomic I/O (Next)

- [ ] Define data contracts for all artifacts
- [ ] Implement database primitives for AudioAsset and pipeline_jobs
- [ ] Build atomic file I/O layer with crash resilience
- [ ] Contract tests for all interfaces
- [ ] End-to-end test scaffolding

### Step 2+: Worker Implementation

- [ ] Ingest API service
- [ ] Worker services (decode, features, segments, preview)
- [ ] Feature extraction pipeline
- [ ] Segment classification
- [ ] Preview generation

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
├── app/                    # Core application modules (placeholder)
│   └── utils/              # Shared utilities (placeholder)
├── docs/                   # Human-readable documentation
│   └── blueprints/         # Blueprint PDFs and research docs
├── services/               # Service components (placeholders)
│   ├── ingest_api/         # Audio ingestion endpoint
│   ├── worker_decode/      # Audio normalization worker
│   ├── worker_features/    # Feature extraction worker
│   ├── worker_segments/    # Segmentation worker
│   └── worker_preview/     # Preview generation worker
├── tests/                  # Test suites
│   ├── contract/           # Contract tests (placeholder)
│   ├── e2e/                # End-to-end tests (placeholder)
│   └── resilience/         # Resilience tests (placeholder)
├── specs/                  # Reserved for JSON Schemas (Step 1)
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

> **Note**: The `specs/` directory is reserved for machine-readable JSON Schemas, to be added in Step 1.
