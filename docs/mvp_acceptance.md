# MVP Acceptance - SSIS Audio Pipeline

Per Blueprint v1.4 Section 14 Step 9, this document describes the MVP acceptance criteria
and how they are validated.

## MVP Acceptance Commands

### Run All Workers via CLI

Each worker can be invoked directly via CLI for offline CPU execution:

```bash
# Decode worker (requires asset in DB with source_uri pointing to valid audio)
python -m services.worker_decode.run <asset_id>

# Features worker (requires normalized.wav from decode)
python -m services.worker_features.run <asset_id>

# Segments worker (requires normalized.wav from decode)
python -m services.worker_segments.run <asset_id>

# Preview worker (requires features HDF5 and segments JSON)
python -m services.worker_preview.run <asset_id>
```

### Run MVP Acceptance Tests

```bash
# Run all MVP acceptance tests
pytest tests/test_mvp_acceptance.py -v

# Run specific test class
pytest tests/test_mvp_acceptance.py::TestOfflineCPURun -v
pytest tests/test_mvp_acceptance.py::TestDeterministicArtifacts -v
pytest tests/test_mvp_acceptance.py::TestJobTelemetry -v
```

### ML Dependencies & Assets

ML-heavy stages (features, segments, preview) require optional dependencies and
the YamNet ONNX model. They are intentionally excluded from default CI runs, but
you can prepare a local environment as follows:

1. Install the optional extras:

   ```bash
   pip install -e ".[dev,ml]"
   ```

2. Fetch the YamNet ONNX artifact so the features worker can launch
   onnxruntime. A helper script is provided, but it requires the future
   GitHub Release asset hosted at
   `https://github.com/SOVRN144/ssis-audio-pipeline-assets/releases/download/yamnet-v1/yamnet.onnx`
   (or a custom URL via `YAMNET_ONNX_DOWNLOAD_URL`). It also verifies the
   sha256 recorded in
   `services/worker_features/yamnet_onnx/yamnet.onnx.sha256`.

   ```bash
   python scripts/fetch_yamnet_onnx.py --force
   ```

   > **Note:** the repository currently ships a placeholder SHA. Publish the
   > official yamnet.onnx binary, compute `sha256sum yamnet.onnx`, update the
   > `.sha256` file with that 64-character digest, and then rerun the script.

3. The segments worker uses `inaSpeechSegmenter`, which downloads its acoustic
   models on first run into `~/.cache/inaSpeechSegmenter`. A future `ci-ml`
   workflow can either pre-seed that cache or allow outbound network access
   for the initial download.

## Expected Artifact Locations

Per Blueprint Section 4, artifacts are stored at these canonical paths:

| Stage    | Artifact Path                                    |
|----------|--------------------------------------------------|
| Decode   | `data/audio/{asset_id}/normalized.wav`           |
| Features | `data/features/{asset_id}.{feature_spec_alias}.h5` |
| Segments | `data/segments/{asset_id}.segments.v1.json`      |
| Preview  | `data/preview/{asset_id}.preview.v1.json`        |

### Feature Spec Alias

The `feature_spec_alias` is computed as:
```text
feature_spec_alias = sha256(feature_spec_id)[:12]
```

For the default v1.4 spec:
- `feature_spec_id`: `mel64_h10ms_w25ms_sr22050__yamnet1024_h0.5s_onnx`
- `feature_spec_alias`: First 12 hex characters of SHA256 hash

## Determinism Approach

### JSON Artifacts (Segments, Preview)

Determinism is validated via semantic hash comparison:
1. Normalize JSON (sort keys, round floats to 6 decimal places)
2. Exclude timestamp fields (`computed_at`, `created_at`, etc.)
3. Compute SHA256 of normalized JSON string
4. Compare hashes across runs

### HDF5 Artifacts (Features)

Determinism is validated via:
1. Dataset shape comparison (`melspec`, `embeddings`)
2. Attribute comparison (excluding timestamps)
3. For strict determinism: byte-identical comparison with fixed random seeds

### WAV Artifacts (Decode)

Determinism is validated via:
1. Byte-identical comparison of WAV file contents
2. Header verification (sample rate, channels, sample width)

## Ingest Omission Rationale

The ingest stage is intentionally omitted from MVP acceptance tests:

1. **Test Focus**: MVP acceptance tests focus on the worker pipeline
   (decode -> features -> segments -> preview), which runs offline on CPU.

2. **Complexity**: Ingest requires FastAPI server startup, HTTP client interaction,
   and file upload handling, which adds significant test complexity.

3. **Existing Coverage**: Ingest is thoroughly tested in:
   - `tests/test_ingest_api.py` - API endpoint tests
   - `tests/test_ingest_idempotency.py` - Idempotency enforcement tests

4. **Blueprint Alignment**: Section 14 Step 9 specifies "offline CPU run" which
   aligns with worker-only testing without network dependencies.

## Ingest Metrics Deferred

Per Blueprint Section 10, ingest requires these metrics:
- `file_size`
- `hash_time`
- `format_guess`

These metrics are NOT validated in MVP acceptance tests because ingest is not invoked.
They are validated in `tests/test_ingest_api.py`.

## Safe Restart Reference

Safe restart after interruption is validated in Step 8 resilience harness:
- `tests/test_resilience_harness.py`

Key test classes:
- `TestAtomicWriteFailpoints` - Atomic write crash recovery
- `TestLockReclamation` - Stale lock reclamation after crash
- `TestDecodeFailpoints` - Decode stage crash recovery
- `TestFeaturesFailpoints` - Features stage crash recovery

## Required Metrics per Stage

Per Blueprint Section 10, each stage must include these metrics in `PipelineJob.metrics_json`:

### Decode Metrics
- `output_duration_sec` - Duration of output WAV
- `chunk_count` - Number of chunks processed
- `decode_time_ms` - Total decode/resample time

### Features Metrics
- `feature_time_ms` - Inference time (mel + embeddings)
- `mel_shape` - Shape of mel spectrogram [frames, mels]
- `embedding_shape` - Shape of embeddings [frames, dims]
- `nan_inf_count` - Count of NaN/Inf values detected
- `feature_spec_id` - Full feature spec identifier
- `feature_spec_alias` - 12-char hex alias

### Segments Metrics
- `segment_count` - Number of segments
- `class_distribution` - Distribution of labels {speech, music, noise, silence}
- `flip_rate` - Label transitions per second

### Preview Metrics
- `candidate_count` - Number of preview candidates generated
- `best_score` - Score of selected candidate
- `fallback_used` - Whether fallback mode was used
- `spec_alias_used` - Feature spec alias used for embeddings

## Validation Checklist

- [ ] All 4 workers have CLI entrypoints (`if __name__ == "__main__"`)
- [ ] Decode produces valid 22050 Hz mono 16-bit WAV
- [ ] Features produces valid HDF5 with melspec and embeddings datasets
- [ ] Segments produces valid JSON matching segments.schema.json
- [ ] Preview produces valid JSON matching preview_candidate.schema.json
- [ ] All artifacts are written atomically (temp -> rename)
- [ ] Metrics include all Section 10 required fields
- [ ] Resilience harness tests pass (Step 8)
- [ ] Determinism tests pass (same input -> same output)
