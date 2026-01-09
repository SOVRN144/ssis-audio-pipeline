# SSIS Audio Pipeline — Repo Operating Rules (Blueprint #1 v1.4)

This repository implements SSIS Blueprint #1 v1.4 as the source of truth.

## Non-Negotiable Constraints

- **Offline-first, CPU-only.** No cloud services.
- **Power instability is a first-class design parameter:** resumable stages, safe restarts.
- **Idempotency at every stage:** safe to re-run after crash/power loss.
- **Atomic publish boundary for artifacts:** write temp → best-effort fsync → rename to final.
- **Stage locks** keyed on `(asset_id, stage, feature_spec_alias|null)` with TTL reclamation (~10 minutes).
- **Versioned contracts:** every artifact schema includes `schema_id`, `version`, `asset_id`, `computed_at`.
- **FeatureSpec alias immutability:** once alias→spec is registered, it is frozen. Collisions are a hard error.
- **HDF5 single-writer rule:** only `worker_features` writes `.h5`; other stages write JSON only.
- **No artifact overwrite:** new config → new FeatureSpec → new artifact. Keep prior artifacts intact.

## Directory Layout Rules

- `specs/` is reserved for machine-readable JSON Schemas (added in Step 1).
- Human-readable documents (PDFs, checklists, notes) belong in `docs/`.

## Canonical Runtime Paths

Do not change these paths:

- `data/audio/{asset_id}/original.<ext>`
- `data/audio/{asset_id}/normalized.wav`
- `data/features/{asset_id}.{feature_spec_alias}.h5`
- `data/segments/{asset_id}.segments.v1.json`
- `data/preview/{asset_id}.preview.v1.json`

## Step Discipline

- **Step 0/0.5:** scaffolding + CI + alignment docs only (NO pipeline logic).
- **Subsequent steps (Step 1+):** contracts + DB primitives + atomic I/O layer.
