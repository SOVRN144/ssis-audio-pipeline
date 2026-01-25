# SSIS Audio Pipeline Operator Runbook

This runbook describes how to install, operate, and troubleshoot the SSIS Audio Pipeline locally and in CI. Paths are relative to the repo root (`ssis-audio-pipeline/`).

## 1. Overview
- Purpose: Offline-first audio pipeline that ingests raw files, normalizes them, extracts mel/YamNet features, classifies speech/music/noise/silence segments, and computes preview windows. Orchestration lives in `app/orchestrator.py` and dispatches work via Huey (`app/huey_app.py`).
- Stages (in order): `ingest -> decode -> features -> segments -> preview`. Each worker resides in `services/worker_<stage>/run.py` and records metrics/artifacts in SQLite (`data/ssis.db`).
- Artifacts:
  - `data/audio/<asset_id>/normalized.wav` (decode, artifact type `normalized_wav`).
  - `data/features/<asset_id>.<feature_spec_alias>.h5` (features, artifact type `feature_pack`).
  - `data/segments/<asset_id>.segments.v1.json` (segments, schema `segments.v1`).
  - `data/preview/<asset_id>.preview.v1.json` (preview, schema `preview_candidate.v1`).
  - Metadata is stored in `audio_assets`, `pipeline_jobs`, `stage_locks`, `artifact_index`, and `feature_specs` tables inside `data/ssis.db`.

## 2. Quickstart (Fast Path)
Run the full pipeline for one asset on macOS/Linux.

### 2.1 Environment prep
```bash
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev,ml]"
brew install ffmpeg        # or sudo apt-get install -y ffmpeg
python scripts/fetch_yamnet_onnx.py
```
`.[ml]` installs the `features-ml` and `segments-ml` extras (onnxruntime, librosa, h5py, numpy, inaSpeechSegmenter).

### 2.2 Create and ingest a demo WAV
```bash
python - <<'PY'
from pathlib import Path
import math, wave, array
sr = 22050
seconds = 8
path = Path("tmp/demo_input.wav")
path.parent.mkdir(parents=True, exist_ok=True)
samples = array.array("h", (int(12000 * math.sin(2*math.pi*440*i/sr)) for i in range(sr*seconds)))
with wave.open(str(path), "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(samples.tobytes())
print(path)
PY
export ASSET_ID=$(python - <<'PY'
from app.db import init_db
from services.ingest_api.service import ingest_local_file
engine, SessionFactory = init_db()
with SessionFactory() as session:
    result = ingest_local_file(session, "tmp/demo_input.wav", owner_entity_id="demo-local")
    print(result.asset_id, end="")
PY
)
echo "Created asset $ASSET_ID"
```

### 2.3 Run workers sequentially
```bash
python services/worker_decode/run.py "$ASSET_ID"
python services/worker_features/run.py "$ASSET_ID"
python services/worker_segments/run.py "$ASSET_ID"
python services/worker_preview/run.py "$ASSET_ID"
```
Each CLI prints its artifact path; exit code 0 means success.

### 2.4 Verify artifacts
```bash
python - <<'PY'
import json, h5py, wave, os
from pathlib import Path
asset = os.environ["ASSET_ID"]
normalized = Path(f"data/audio/{asset}/normalized.wav")
with wave.open(str(normalized), "rb") as wf:
    print("normalized.wav:", wf.getnframes()/wf.getframerate(), "sec @", wf.getframerate(), "Hz")
feature_files = list(Path("data/features").glob(f"{asset}.*.h5"))
print("feature packs:", feature_files)
if feature_files:
    with h5py.File(feature_files[0], "r") as f:
        print("melspec shape", f["melspec"].shape, "embeddings shape", f["embeddings"].shape)
segments_path = Path(f"data/segments/{asset}.segments.v1.json")
if segments_path.exists():
    payload = json.loads(segments_path.read_text())
    print("segment count", len(payload.get("segments", [])))
preview_path = Path(f"data/preview/{asset}.preview.v1.json")
if preview_path.exists():
    payload = json.loads(preview_path.read_text())
    print("preview mode", payload.get("mode"), "window", payload.get("preview", {}))
PY
sqlite3 data/ssis.db "select stage,status,error_code,artifact_path from pipeline_jobs where asset_id='$ASSET_ID' order by id;"
```

## 3. Installation and Setup (Mac/Linux)
1. Python >= 3.11 (per `pyproject.toml`).
2. Virtualenv: `python3.11 -m venv .venv && source .venv/bin/activate`.
3. Python deps:
   - Base dev tools `pip install -e ".[dev]"` (pytest, ruff, mypy, jsonschema, numpy, h5py).
   - Extras `pip install -e ".[features-ml]"` for ONNX/librosa, `pip install -e ".[segments-ml]"` for inaSpeechSegmenter, or `pip install -e ".[ml]"` for both.
4. System deps: `ffmpeg` must be in PATH.
5. Models: run `python scripts/fetch_yamnet_onnx.py` (honors `YAMNET_ONNX_DOWNLOAD_URL`).
6. Data dirs: workers create directories automatically, but you can pre-create `data/`, `logs/`, and `data/queue`. SQLite lives in `data/ssis.db`, Huey queue in `data/queue/huey.db`.
7. Smoke test:
   ```bash
   ffmpeg -version
   python - <<'PY'
   import onnxruntime, librosa, h5py, numpy
   from inaSpeechSegmenter import Segmenter
   Segmenter()
   PY
   ```

## 3. Installation and Setup (Mac/Linux)
1. Python >= 3.11 (per `pyproject.toml`).
2. Virtualenv: `python3.11 -m venv .venv && source .venv/bin/activate`.
3. Python deps:
   - Base dev tools `pip install -e ".[dev]"` (pytest, ruff, mypy, jsonschema, numpy, h5py).
   - Extras `pip install -e ".[features-ml]"` for ONNX/librosa, `pip install -e ".[segments-ml]"` for inaSpeechSegmenter, or `pip install -e ".[ml]"` for both.
4. System deps: `ffmpeg` must be in PATH.
5. Models: run `python scripts/fetch_yamnet_onnx.py` (honors `YAMNET_ONNX_DOWNLOAD_URL`).
6. Data dirs: workers create directories automatically, but you can pre-create `data/`, `logs/`, and `data/queue`. SQLite lives in `data/ssis.db`, Huey queue in `data/queue/huey.db`.
7. Smoke test:
   ```bash
   ffmpeg -version
   python - <<'PY'
   import onnxruntime, librosa, h5py, numpy
   from inaSpeechSegmenter import Segmenter
   Segmenter()
   PY
   ```

## 4. Running the Pipeline End-to-End
### 4.1 Stage reference
- Ingest: FastAPI (`uvicorn services.ingest_api.main:app`) or `ingest_local_file`. Input is a file path plus optional metadata. Output is the copied original file and DB records; orchestrator ticks are enqueued.
- Decode: `python services/worker_decode/run.py <asset_id>` or orchestrator dispatch. Inputs come from `audio_assets`. Output is `data/audio/<asset_id>/normalized.wav`. Uses ffmpeg chunking, checkpoints, and failpoints `DECODE_AFTER_CHUNK_WRITE`, `DECODE_AFTER_CHECKPOINT`, `DECODE_BEFORE_FINAL_RENAME`.
- Features: `python services/worker_features/run.py <asset_id>`. Inputs: normalized WAV + YamNet ONNX. Output: FeaturePack HDF5 with alias recorded via `feature_spec_alias`.
- Segments: `python services/worker_segments/run.py <asset_id>`. Inputs: normalized WAV. Output: segments JSON + metrics; exceptions map to `SegmentsErrorCode`.
- Preview: `python services/worker_preview/run.py <asset_id>`. Inputs: FeaturePack + segments + normalized WAV. Output: preview JSON; `SSIS_ACTIVE_FEATURE_SPEC_ALIAS` chooses alias if needed.

### 4.2 Manual orchestration tips
1. Ingest (or insert `AudioAsset`) before running decode.
2. Decode short-circuits if `normalized.wav` exists and returns `DecodeErrorCode` values (`INPUT_NOT_FOUND`, `FILE_TOO_SHORT`, `CODEC_UNSUPPORTED`, `FILE_CORRUPT`, `WORKER_ERROR`).
3. Compute the default feature alias when needed:
   ```bash
   python - <<'PY'
   from app.utils.hashing import feature_spec_alias
   from app.config import DEFAULT_FEATURE_SPEC_ID
   print(feature_spec_alias(DEFAULT_FEATURE_SPEC_ID))
   PY
   ```
4. The segments worker calls `_resolve_segmenter_callable()` so CI patches work. Exceptions never publish artifacts and map via `_map_exception_to_error_code`.
5. Preview worker requires FeaturePack + segments; set `SSIS_ACTIVE_FEATURE_SPEC_ALIAS` when using a non-default alias.

### 4.3 Prod-like run (Huey + FastAPI)
1. Start the API: `uvicorn services.ingest_api.main:app --host 127.0.0.1 --port 8000 --reload`.
2. Start Huey consumer: `huey_consumer.py app.huey_app.huey` (queue stored at `data/queue/huey.db`).
3. Submit work: `http POST :8000/v1/ingest/local source_path=/abs/path.wav owner_entity_id=demo`.
4. Check progress: `sqlite3 data/ssis.db 'select stage,status,error_code from pipeline_jobs where asset_id="...";'` and inspect `data/` directories.

### 4.4 Artifact verification snippets
- Normalized WAV duration: `python - <<'PY' ... wave.open ... PY "$ASSET_ID"`.
- FeaturePack overview: `python - <<'PY' ... h5py.File ... PY "$ASSET_ID"`.
- Segments count: `jq '.segments | length' data/segments/$ASSET_ID.segments.v1.json`.
- Preview summary: `jq '{mode,preview}' data/preview/$ASSET_ID.preview.v1.json`.

## 5. Running Individual Workers and Services
- **Ingest API**: run via uvicorn (`uvicorn services.ingest_api.main:app --reload`). Endpoints `POST /v1/ingest/local` and `/v1/ingest/upload` map to `IngestLocalRequest`. Errors map to `IngestErrorCode` (`FILE_NOT_FOUND`, `HASH_FAILED`, `INGEST_FAILED`).
- **Decode worker**: `python services/worker_decode/run.py <asset_id>`. Requires ffmpeg, uses checkpoints and atomic publish. Failpoints: `DECODE_AFTER_CHUNK_WRITE`, `DECODE_AFTER_CHECKPOINT`, `DECODE_BEFORE_FINAL_RENAME`.
- **Features worker**: `python services/worker_features/run.py <asset_id>`. Requires normalized WAV + YamNet ONNX. Error codes: `FEATURE_NAN`, `MODEL_OOM`, `FEATURE_EXTRACTION_FAILED`, `FEATURE_SPEC_ALIAS_COLLISION`, `INPUT_NOT_FOUND`.
- **Segments worker**: `python services/worker_segments/run.py <asset_id>`. Requires normalized WAV. `_resolve_segmenter_callable()` ensures patched `_run_segmenter` objects are used. Exceptions map via `_map_exception_to_error_code` and suppress artifact writes.
- **Preview worker**: `python services/worker_preview/run.py <asset_id>`. Requires FeaturePack + segments + normalized WAV. Optional `SSIS_ACTIVE_FEATURE_SPEC_ALIAS` selects the alias to read.

## 6. Idempotency and Re-runs
- Workers short-circuit when artifacts exist (decode, segments, preview return `ok=True` and keep the file untouched).
- To re-run a stage:
  1. Delete the artifact file (example: `rm -f data/segments/$ASSET_ID.segments.v1.json`).
  2. Remove the `artifact_index` row so orchestrator does not skip it: `sqlite3 data/ssis.db "delete from artifact_index where asset_id='$ASSET_ID' and artifact_type='segments_v1';"`.
  3. Re-run the worker or call `enqueue_orchestrator_tick(asset_id)` from `app.huey_app`.
- Full reset: delete `data/audio/$ASSET_ID` plus associated FeaturePack/segments/preview files and purge DB rows (`audio_assets`, `pipeline_jobs`, `artifact_index`, `stage_locks`).
- Failpoints (resilience testing): enable with `SSIS_ENABLE_FAILPOINTS=1` and `SSIS_FAILPOINT=<NAME>` such as `DECODE_AFTER_CHUNK_WRITE`, `DECODE_AFTER_CHECKPOINT`, `DECODE_BEFORE_FINAL_RENAME`, `FEATURES_AFTER_H5_TMP_WRITE`, `FEATURES_BEFORE_H5_RENAME`, `ATOMIC_WRITE_AFTER_TMP_WRITE`, `ATOMIC_WRITE_AFTER_FSYNC_BEFORE_RENAME`, `ATOMIC_WRITE_AFTER_RENAME`. Use `SSIS_FAILPOINT_EXIT_CODE` and `SSIS_FAILPOINT_ONCE` to control exits.

## 7. Testing, Linting, CI Parity
- Unit tests: `pytest` (default `-v --tb=short`). Targeted suites:
  - `pytest -q tests/test_worker_segments.py::TestErrorMapping`.
  - `PYTEST_ADDOPTS=-rs pytest tests/test_mvp_acceptance.py` (requires ffmpeg, ML extras, YamNet model).
- Lint/format:
  ```bash
  ruff check .
  ruff format --check .
  ```
- Type checking: `mypy app/ services/`.
- CI workflow (`.github/workflows/ci.yml`): checkout -> Python 3.11 -> `pip install -e ".[dev]"` -> install ffmpeg -> `ruff check .` -> `ruff format --check .` -> `mypy app/ services/` (continue-on-error) -> `pytest -v`.
- ML workflow (`.github/workflows/ci-ml.yml`): checkout -> Python 3.11 -> install ffmpeg -> `pip install -e ".[dev,ml]"` -> cache inaSpeechSegmenter -> `python scripts/fetch_yamnet_onnx.py --force` -> `pytest tests/test_mvp_acceptance.py`.

## 8. Troubleshooting
| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Decode worker `INPUT_NOT_FOUND` | `audio_assets` row missing or original file deleted | Re-ingest the source file or ensure `audio_assets.source_uri` exists. |
| Segments worker `MODEL_OOM` | `_run_segmenter` raised `MemoryError` or RuntimeError with OOM text | Reduce concurrency, confirm `inaSpeechSegmenter` is installed, rerun. `_map_exception_to_error_code` uses tokens in `OOM_ERROR_PATTERN`. |
| Segments worker `SEGMENTATION_FAILED` | Generic RuntimeError (model missing, dependency mismatch) | Ensure `pip install -e ".[segments-ml]"`, verify `data/audio/<asset>/normalized.wav`, rerun with logging. |
| Segments schema/invariant failure (`SEGMENTS_INVALID`) | `_validate_schema` or `_validate_invariants` detected overlapping/unsorted segments | Inspect `data/segments/<asset>.segments.v1.json`, rerun worker. |
| Preview `FEATUREPACK_MISSING` or `INPUT_NOT_FOUND` | FeaturePack alias mismatch or missing segments | Query `sqlite3 data/ssis.db 'select feature_spec_alias from pipeline_jobs where stage="features" order by id desc limit 1;'` and set `SSIS_ACTIVE_FEATURE_SPEC_ALIAS`. |
| Preview schema failure | `_basic_schema_validation` or `_validate_invariants` failed | Inspect JSON, rerun `python services/worker_preview/run.py <asset_id>` after fixing inputs. |
| Tests cannot patch `_run_segmenter` | Module imported twice | Patch `services.worker_segments.run._run_segmenter`; worker resolves callables via `_resolve_segmenter_callable()`. |
| Missing deps (ffmpeg, onnxruntime, inaSpeechSegmenter) | Extras not installed or system packages missing | Re-run `pip install -e ".[ml]"`, verify `ffmpeg -version`, rerun `python scripts/fetch_yamnet_onnx.py`. |
| Huey queue stuck | Huey consumer not running or queue DB missing | Start `huey_consumer.py app.huey_app.huey`, ensure `data/queue` exists, inspect `data/queue/huey.db`. |

## 9. Configuration Reference
### Environment variables
| Variable | Default | Description |
| --- | --- | --- |
| `SSIS_LOCK_TTL_SEC` | 600 | Overrides stage-lock TTL seconds (see `app/config.py`). |
| `SSIS_ACTIVE_FEATURE_SPEC_ALIAS` | derived from `DEFAULT_FEATURE_SPEC_ID` | Forces preview worker to read FeaturePacks with a specific 12-character alias. |
| `SSIS_ENABLE_FAILPOINTS` | unset | Enables failpoints. |
| `SSIS_FAILPOINT` | unset | Name of failpoint to trigger. |
| `SSIS_FAILPOINT_EXIT_CODE` | 42 | Exit code used when a failpoint fires. |
| `SSIS_FAILPOINT_ONCE` | unset | Triggers once then clears. |
| `YAMNET_ONNX_DOWNLOAD_URL` | packaged default | Override download URL for `scripts/fetch_yamnet_onnx.py`. |

### Notable constants and paths
- Paths (`app/config.py`): `DATA_DIR=data`, `AUDIO_DIR=data/audio`, `FEATURES_DIR=data/features`, `SEGMENTS_DIR=data/segments`, `PREVIEW_DIR=data/preview`, `DB_PATH=data/ssis.db`, `HUEY_DB_PATH=data/queue/huey.db`, `LOGS_DIR=logs`.
- Decode worker: `CHUNK_SECONDS=60`, `MIN_DURATION_SEC=1.7`, `FFMPEG_TIMEOUT_SECONDS=300`.
- Segments worker: `MIN_SPEECH_SEC=0.8`, `MIN_MUSIC_SEC=3.4`, `MIN_SILENCE_SEC=0.5`, `MERGE_GAP_SEC=0.3`, `CONFIDENCE_BASE` mapping, schema id `segments.v1`, regex `OOM_ERROR_PATTERN` with `\boom\b`.
- Preview worker: `WINDOW_SEC=60`, `MIN_WINDOW_FRACTION=0.75`, `ENERGY_WEIGHT=0.6`, `EMBEDDING_WEIGHT=0.4`, `SCORE_THRESHOLD=0.5`.
- Feature spec: `DEFAULT_FEATURE_SPEC_ID="mel64_h10ms_w25ms_sr22050__yamnet1024_h0.5s_onnx"`; alias computed via `app.utils.hashing.feature_spec_alias`.

## 10. Appendix: Repo Map and CI
- Directories:
  - `app/` – config, DB helpers, orchestrator, Huey setup, utils.
  - `services/` – ingest API and worker implementations.
  - `tests/` – contract, worker, orchestrator, resilience suites.
  - `docs/` – human-readable docs (blueprints + this runbook).
  - `specs/` – JSON Schemas referenced by workers.
  - `scripts/` – helper scripts (`fetch_yamnet_onnx.py`).
  - `data/` – inputs, outputs, SQLite DB, queue files, logs (gitignored).
- CI workflows:
  - `.github/workflows/ci.yml` – installs `.[dev]`, ffmpeg, runs `ruff check`, `ruff format --check`, `mypy` (soft-fail), `pytest -v`.
  - `.github/workflows/ci-ml.yml` – installs `.[dev,ml]`, caches inaSpeechSegmenter, fetches YamNet, runs `pytest tests/test_mvp_acceptance.py`.

Use this runbook as the source of truth for onboarding, local ops, and CI parity.
