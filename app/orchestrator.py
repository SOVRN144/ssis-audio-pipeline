"""SSIS Audio Pipeline - Orchestrator logic.

Implements stage planning, dispatch, retries, and dead-letter handling
per Blueprint section 9.

Stage progression: ingest -> decode -> features -> segments -> preview

Step 3 scope:
- Planning based on artifact existence + artifact_index
- StageLock acquisition with TTL and stale reclamation
- Retry policy: 4 total attempts (initial + 3 retries), delays 60s/300s/900s
- Dead-letter after all retries exhausted

Step 3 does NOT implement actual worker processing (decode/normalize/features).

Retry Semantics (Blueprint section 9):
---------------------------------------
Blueprint specifies three retry delays: 60s, 300s, 900s.
This means:
  - Attempt 1: Initial attempt (no delay)
  - Attempt 2: First retry after 60s delay
  - Attempt 3: Second retry after 300s delay
  - Attempt 4: Third retry after 900s delay
  - After attempt 4 fails: job is dead-lettered

Thus "3 retries" = 4 total attempts. MAX_ATTEMPTS_TOTAL is set to 4.
The delay index is (current_attempt - 1), so attempt 1 failure triggers
RETRY_DELAYS_SECONDS[0]=60s, etc.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import and_, select

from app.config import (
    DEFAULT_FEATURE_SPEC_ID,
    MAX_ATTEMPTS_TOTAL,
    RETRY_DELAYS_SECONDS,
)
from app.db import create_stage_lock, init_db
from app.models import ArtifactIndex, PipelineJob, StageLock, utc_now
from app.utils.hashing import feature_spec_alias as compute_feature_spec_alias
from app.utils.paths import audio_normalized_path, features_h5_path, segments_json_path

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# --- Stage Definitions ---

# Stage that follows ingest (Step 3 scope)
STAGE_DECODE = "decode"

# Stage that follows decode (Step 5 scope)
STAGE_FEATURES = "features"

# Stage that follows features (Step 6 scope)
STAGE_SEGMENTS = "segments"

# Artifact type for decode stage output
ARTIFACT_TYPE_NORMALIZED_WAV = "normalized_wav"

# Artifact type for features stage output
ARTIFACT_TYPE_FEATURES_H5 = "feature_pack"

# Artifact type for segments stage output (Step 6)
ARTIFACT_TYPE_SEGMENTS_V1 = "segments_v1"

# Schema identifiers for segments (Step 6)
SEGMENTS_SCHEMA_ID = "segments.v1"
SEGMENTS_VERSION = "1.0.0"

# Maximum lock reclaim events to store in metrics_json (prevent unbounded growth)
MAX_LOCK_RECLAIM_EVENTS = 10


# --- Metrics Helpers ---


def _record_lock_reclaim_metric(
    job: PipelineJob,
    reclaimed_worker_id: str,
    expired_at_str: str,
) -> None:
    """Record a lock reclaim event in the job's metrics_json.

    Keeps a counter and a bounded list of recent reclaim events.
    Safe if metrics_json is null or missing expected structure.

    Args:
        job: The PipelineJob to update.
        reclaimed_worker_id: Worker ID of the reclaimed lock.
        expired_at_str: ISO timestamp when the lock expired.
    """
    # Parse existing metrics or start fresh
    try:
        metrics = json.loads(job.metrics_json) if job.metrics_json else {}
    except (json.JSONDecodeError, TypeError):
        metrics = {}

    # Ensure metrics is a dict
    if not isinstance(metrics, dict):
        metrics = {}

    # Initialize lock_reclaim section if needed
    if "lock_reclaim" not in metrics:
        metrics["lock_reclaim"] = {"count": 0, "events": []}

    lock_reclaim = metrics["lock_reclaim"]

    # Increment counter
    lock_reclaim["count"] = lock_reclaim.get("count", 0) + 1

    # Add event (bounded list)
    events = lock_reclaim.get("events", [])
    if not isinstance(events, list):
        events = []

    events.append(
        {
            "reclaimed_worker_id": reclaimed_worker_id,
            "expired_at": expired_at_str,
            "reclaimed_at": utc_now().isoformat(),
        }
    )

    # Keep only the last N events
    if len(events) > MAX_LOCK_RECLAIM_EVENTS:
        events = events[-MAX_LOCK_RECLAIM_EVENTS:]

    lock_reclaim["events"] = events
    metrics["lock_reclaim"] = lock_reclaim

    # Write back
    job.metrics_json = json.dumps(metrics)


# --- Orchestrator Tick ---


def orchestrator_tick(asset_id: str) -> dict:
    """Run orchestrator planning and dispatch for an asset.

    This is the main entry point called by Huey task.
    Idempotent: safe to call multiple times.

    Args:
        asset_id: The asset ID to process.

    Returns:
        Dict describing what happened (for logging/debugging).
    """
    # Get a fresh session for this tick
    _, SessionFactory = init_db()
    session = SessionFactory()

    try:
        result = _orchestrator_tick_impl(session, asset_id)
        session.commit()
        return result
    except Exception as e:
        session.rollback()
        logger.exception("Orchestrator tick failed for asset_id=%s", asset_id)
        return {"status": "error", "asset_id": asset_id, "error": str(e)}
    finally:
        session.close()


def _orchestrator_tick_impl(session: Session, asset_id: str) -> dict:
    """Implementation of orchestrator tick.

    Args:
        session: Database session.
        asset_id: The asset ID to process.

    Returns:
        Dict describing the result.
    """
    # 1. Check if ingest is completed for this asset
    ingest_job = _find_completed_ingest_job(session, asset_id)
    if ingest_job is None:
        logger.debug("No completed ingest job for asset_id=%s", asset_id)
        return {"status": "no_work", "asset_id": asset_id, "reason": "no_completed_ingest"}

    # 2. Check for dead-letter jobs - skip if any stage is dead-lettered
    dead_letter_job = _find_dead_letter_job(session, asset_id)
    if dead_letter_job is not None:
        logger.info(
            "Asset %s has dead-letter job (stage=%s), skipping", asset_id, dead_letter_job.stage
        )
        return {
            "status": "skipped",
            "asset_id": asset_id,
            "reason": "dead_letter_exists",
            "dead_letter_stage": dead_letter_job.stage,
        }

    # 3. Determine next stage (Step 3: only decode after ingest)
    next_stage = _determine_next_stage(session, asset_id)
    if next_stage is None:
        logger.debug("No pending stages for asset_id=%s", asset_id)
        return {"status": "no_work", "asset_id": asset_id, "reason": "all_stages_complete"}

    # 4. Compute feature_spec_alias for STAGE_FEATURES (used consistently for lock/job/release)
    # This MUST be computed BEFORE lock acquisition to prevent lock-alias mismatch leaks
    feature_spec_alias = None
    if next_stage == STAGE_FEATURES:
        feature_spec_alias = compute_feature_spec_alias(DEFAULT_FEATURE_SPEC_ID)

    # 5. Check if artifact already exists (skip if so)
    if _artifact_exists(session, asset_id, next_stage, feature_spec_alias):
        logger.info("Artifact for stage %s already exists for asset_id=%s", next_stage, asset_id)
        return {
            "status": "skipped",
            "asset_id": asset_id,
            "stage": next_stage,
            "reason": "artifact_exists",
        }

    # 6. Check for existing lock (using computed alias for STAGE_FEATURES)
    lock_result = _handle_stage_lock(session, asset_id, next_stage, feature_spec_alias)
    if lock_result["action"] == "skip":
        return {
            "status": "skipped",
            "asset_id": asset_id,
            "stage": next_stage,
            "reason": lock_result["reason"],
        }

    # 7. Create or update the PipelineJob for this stage (with alias for STAGE_FEATURES)
    job = _get_or_create_stage_job(session, asset_id, next_stage, feature_spec_alias)

    # 7a. Ensure job.feature_spec_alias is set consistently (may be None for newly created jobs)
    if next_stage == STAGE_FEATURES and job.feature_spec_alias != feature_spec_alias:
        job.feature_spec_alias = feature_spec_alias
        session.flush()

    # 8. Record lock reclaim metric if applicable
    if "reclaim_info" in lock_result:
        reclaim_info = lock_result["reclaim_info"]
        _record_lock_reclaim_metric(
            job,
            reclaim_info["reclaimed_worker_id"],
            reclaim_info["expired_at"],
        )
        session.flush()

    # 9. Check if this is a retry scenario
    if job.status == "failed" and job.attempt >= MAX_ATTEMPTS_TOTAL:
        # Should have been caught as dead_letter, but handle defensively
        _mark_dead_letter(session, job, "Max attempts exceeded")
        return {
            "status": "dead_letter",
            "asset_id": asset_id,
            "stage": next_stage,
            "job_id": job.job_id,
        }

    # 10. Update job status to running
    job.status = "running"
    job.started_at = utc_now()
    session.flush()

    # 11. Dispatch the stage worker task
    from app.huey_app import enqueue_stage_worker

    enqueue_stage_worker(job.job_id, asset_id, next_stage)

    return {
        "status": "dispatched",
        "asset_id": asset_id,
        "stage": next_stage,
        "job_id": job.job_id,
        "attempt": job.attempt,
    }


# --- Stage Execution ---


def execute_stage(job_id: str, asset_id: str, stage: str) -> dict:
    """Execute a pipeline stage.

    Called by Huey worker task. Handles success/failure and retry logic.

    Step 3: Stub implementation - marks as failed since no real workers exist.
    Real workers will be implemented in Step 4+.

    Args:
        job_id: The PipelineJob ID.
        asset_id: The asset ID.
        stage: The pipeline stage.

    Returns:
        Dict with execution result.
    """
    _, SessionFactory = init_db()
    session = SessionFactory()

    try:
        result = _execute_stage_impl(session, job_id, asset_id, stage)
        session.commit()
        return result
    except Exception as e:
        session.rollback()
        logger.exception("Stage execution failed: job_id=%s, stage=%s", job_id, stage)
        # Handle failure in a new session
        session2 = SessionFactory()
        try:
            _handle_stage_failure(session2, job_id, "WORKER_ERROR", str(e))
            session2.commit()
        except Exception:
            session2.rollback()
        finally:
            session2.close()
        return {"status": "error", "job_id": job_id, "stage": stage, "error": str(e)}
    finally:
        session.close()


def _execute_stage_impl(session: Session, job_id: str, asset_id: str, stage: str) -> dict:
    """Implementation of stage execution.

    Dispatches to the appropriate worker based on stage.
    Currently implements: decode (Step 4), features (Step 5).

    Args:
        session: Database session.
        job_id: The PipelineJob ID.
        asset_id: The asset ID.
        stage: The pipeline stage.

    Returns:
        Dict with execution result.
    """
    # Find the job
    stmt = select(PipelineJob).where(PipelineJob.job_id == job_id)
    job = session.execute(stmt).scalar_one_or_none()

    if job is None:
        logger.error("Job not found: job_id=%s", job_id)
        return {"status": "error", "job_id": job_id, "error": "job_not_found"}

    if stage == STAGE_DECODE:
        return _execute_decode_stage(session, job, asset_id)

    if stage == STAGE_FEATURES:
        return _execute_features_stage(session, job, asset_id)

    if stage == STAGE_SEGMENTS:
        return _execute_segments_stage(session, job, asset_id)

    # Unknown stage
    logger.error("Unknown stage: %s", stage)
    _handle_stage_failure(session, job.job_id, "UNKNOWN_STAGE", f"Unknown stage: {stage}")
    return {"status": "error", "job_id": job_id, "stage": stage, "error": "unknown_stage"}


def _execute_decode_stage(session: Session, job: PipelineJob, asset_id: str) -> dict:
    """Execute the decode stage using the decode worker.

    Args:
        session: Database session.
        job: The PipelineJob record.
        asset_id: The asset ID.

    Returns:
        Dict with execution result.
    """
    from services.worker_decode.run import decode_asset

    # Run the decode worker
    result = decode_asset(session, asset_id)

    if result.ok:
        # Success - record artifact and update job
        if result.artifact_path and result.artifact_type:
            _record_artifact(
                session,
                asset_id,
                result.artifact_type,
                result.artifact_path,
                schema_version=result.schema_version or "1.0.0",
            )

        # Update job metrics
        _update_job_metrics(
            session,
            job,
            "decode",
            {
                "output_duration_sec": result.metrics.output_duration_sec,
                "chunk_count": result.metrics.chunk_count,
                "decode_time_ms": result.metrics.decode_time_ms,
            },
        )

        _mark_job_completed(session, job)
        _release_stage_lock(session, asset_id, STAGE_DECODE)

        return {
            "status": "completed",
            "job_id": job.job_id,
            "stage": STAGE_DECODE,
            "artifact_path": result.artifact_path,
            "metrics": {
                "output_duration_sec": result.metrics.output_duration_sec,
                "chunk_count": result.metrics.chunk_count,
                "decode_time_ms": result.metrics.decode_time_ms,
            },
        }
    else:
        # Failure - trigger retry logic
        error_code = result.error_code or "WORKER_ERROR"
        error_message = result.message or "Decode failed"

        logger.warning(
            "Decode failed for asset_id=%s: %s - %s",
            asset_id,
            error_code,
            error_message,
        )

        _handle_stage_failure(session, job.job_id, error_code, error_message)

        return {
            "status": "failed",
            "job_id": job.job_id,
            "stage": STAGE_DECODE,
            "error_code": error_code,
            "error_message": error_message,
        }


def _execute_features_stage(session: Session, job: PipelineJob, asset_id: str) -> dict:
    """Execute the features stage using the features worker.

    Args:
        session: Database session.
        job: The PipelineJob record.
        asset_id: The asset ID.

    Returns:
        Dict with execution result.
    """
    from services.worker_features.run import extract_features

    # Get the feature_spec_alias for this job (default v1.4 spec)
    # This MUST match what was used for lock acquisition in _orchestrator_tick_impl
    spec_alias = job.feature_spec_alias or compute_feature_spec_alias(DEFAULT_FEATURE_SPEC_ID)

    # Ensure job.feature_spec_alias is set if it was None (for consistency in failure path)
    if job.feature_spec_alias is None:
        job.feature_spec_alias = spec_alias
        session.flush()

    # Run the features worker
    result = extract_features(session, asset_id)

    if result.ok:
        # Success - record artifact and update job
        if result.artifact_path and result.artifact_type:
            _record_artifact(
                session,
                asset_id,
                result.artifact_type,
                result.artifact_path,
                feature_spec_alias=result.feature_spec_alias,
                schema_version=result.schema_version or "1.0.0",
            )

        # Update job metrics
        _update_job_metrics(
            session,
            job,
            "features",
            result.metrics,
        )

        _mark_job_completed(session, job)
        # Release lock with the SAME alias used for acquisition
        _release_stage_lock(session, asset_id, STAGE_FEATURES, spec_alias)

        return {
            "status": "completed",
            "job_id": job.job_id,
            "stage": STAGE_FEATURES,
            "artifact_path": result.artifact_path,
            "metrics": result.metrics,
        }
    else:
        # Failure - trigger retry logic
        # Note: _handle_stage_failure will use job.feature_spec_alias (now guaranteed set)
        error_code = result.error_code or "WORKER_ERROR"
        error_message = result.message or "Feature extraction failed"

        logger.warning(
            "Features failed for asset_id=%s: %s - %s",
            asset_id,
            error_code,
            error_message,
        )

        _handle_stage_failure(session, job.job_id, error_code, error_message)

        return {
            "status": "failed",
            "job_id": job.job_id,
            "stage": STAGE_FEATURES,
            "error_code": error_code,
            "error_message": error_message,
        }


def _execute_segments_stage(session: Session, job: PipelineJob, asset_id: str) -> dict:
    """Execute the segments stage using the segments worker.

    Args:
        session: Database session.
        job: The PipelineJob record.
        asset_id: The asset ID.

    Returns:
        Dict with execution result.
    """
    from app.config import DATA_DIR
    from services.worker_segments.run import run_segments_worker

    # Run the segments worker
    result = run_segments_worker(asset_id, DATA_DIR)

    if result.ok:
        # Success - record artifact and update job
        if result.artifact_path and result.artifact_type:
            _record_artifact(
                session,
                asset_id,
                result.artifact_type,
                result.artifact_path,
                schema_version=result.schema_version or SEGMENTS_VERSION,
            )

        # Update job metrics
        _update_job_metrics(
            session,
            job,
            "segments",
            result.metrics,
        )

        _mark_job_completed(session, job)
        # Release lock with feature_spec_alias=None for segments stage
        _release_stage_lock(session, asset_id, STAGE_SEGMENTS, None)

        return {
            "status": "completed",
            "job_id": job.job_id,
            "stage": STAGE_SEGMENTS,
            "artifact_path": result.artifact_path,
            "metrics": result.metrics,
        }
    else:
        # Failure - trigger retry logic
        error_code = result.error_code or "WORKER_ERROR"
        error_message = result.message or "Segmentation failed"

        logger.warning(
            "Segments failed for asset_id=%s: %s - %s",
            asset_id,
            error_code,
            error_message,
        )

        _handle_stage_failure(session, job.job_id, error_code, error_message)

        return {
            "status": "failed",
            "job_id": job.job_id,
            "stage": STAGE_SEGMENTS,
            "error_code": error_code,
            "error_message": error_message,
        }


def _update_job_metrics(
    session: Session,
    job: PipelineJob,
    namespace: str,
    metrics: dict,
) -> None:
    """Update job metrics_json with new metrics under a namespace.

    Merges metrics into existing metrics_json without overwriting other keys.

    Args:
        session: Database session.
        job: The PipelineJob to update.
        namespace: Namespace key for the metrics (e.g., "decode").
        metrics: Dictionary of metrics to store.
    """
    # Parse existing metrics or start fresh
    try:
        existing = json.loads(job.metrics_json) if job.metrics_json else {}
    except (json.JSONDecodeError, TypeError):
        existing = {}

    if not isinstance(existing, dict):
        existing = {}

    # Merge under namespace
    existing[namespace] = metrics

    job.metrics_json = json.dumps(existing)
    session.flush()


# --- Failure and Retry Handling ---


def _handle_stage_failure(
    session: Session, job_id: str, error_code: str, error_message: str
) -> None:
    """Handle a stage failure with retry logic.

    Per Blueprint section 9:
    - 3 retries with delays 60s, 300s, 900s
    - 4 total attempts (initial + 3 retries)
    - After 4th attempt fails: dead-letter

    Attempt mapping:
    - attempt 1 fails -> schedule attempt 2 after 60s
    - attempt 2 fails -> schedule attempt 3 after 300s
    - attempt 3 fails -> schedule attempt 4 after 900s
    - attempt 4 fails -> dead-letter

    Args:
        session: Database session.
        job_id: The PipelineJob ID.
        error_code: Error code for taxonomy.
        error_message: Human-readable error message.
    """
    stmt = select(PipelineJob).where(PipelineJob.job_id == job_id)
    job = session.execute(stmt).scalar_one_or_none()

    if job is None:
        logger.error("Cannot handle failure - job not found: job_id=%s", job_id)
        return

    current_attempt = job.attempt
    logger.info(
        "Stage failure: job_id=%s, stage=%s, attempt=%d/%d, error=%s",
        job_id,
        job.stage,
        current_attempt,
        MAX_ATTEMPTS_TOTAL,
        error_code,
    )

    if current_attempt >= MAX_ATTEMPTS_TOTAL:
        # Dead-letter: all retries exhausted
        _mark_dead_letter(session, job, error_message, error_code)
        _release_stage_lock(session, job.asset_id, job.stage, job.feature_spec_alias)
        return

    # Schedule retry with incremented attempt
    next_attempt = current_attempt + 1
    # Delay index: attempt 1 -> delay[0]=60s, attempt 2 -> delay[1]=300s, attempt 3 -> delay[2]=900s
    delay_seconds = RETRY_DELAYS_SECONDS[current_attempt - 1]

    job.attempt = next_attempt
    job.status = "queued"
    job.error_code = error_code
    job.error_message = error_message
    job.finished_at = utc_now()
    session.flush()

    # Release the lock so retry can re-acquire
    _release_stage_lock(session, job.asset_id, job.stage, job.feature_spec_alias)

    logger.info(
        "Scheduling retry: job_id=%s, next_attempt=%d, delay=%ds",
        job_id,
        next_attempt,
        delay_seconds,
    )

    # Re-enqueue orchestrator tick with delay (will re-dispatch the stage)
    from app.huey_app import orchestrator_tick_task

    orchestrator_tick_task.schedule((job.asset_id,), delay=delay_seconds)


def _mark_dead_letter(
    session: Session,
    job: PipelineJob,
    error_message: str,
    error_code: str | None = None,
) -> None:
    """Mark a job as dead-letter.

    Args:
        session: Database session.
        job: The PipelineJob to mark.
        error_message: Summary error message.
        error_code: Optional error code.
    """
    logger.warning(
        "Dead-letter: job_id=%s, asset_id=%s, stage=%s, attempts=%d",
        job.job_id,
        job.asset_id,
        job.stage,
        job.attempt,
    )
    job.status = "dead_letter"
    job.error_code = error_code or job.error_code or "DEAD_LETTER"
    job.error_message = error_message
    job.finished_at = utc_now()
    session.flush()


def _mark_job_completed(session: Session, job: PipelineJob) -> None:
    """Mark a job as completed.

    Args:
        session: Database session.
        job: The PipelineJob to mark.
    """
    job.status = "completed"
    job.finished_at = utc_now()
    job.error_code = None
    job.error_message = None
    session.flush()


# --- Lock Management ---


def _handle_stage_lock(
    session: Session,
    asset_id: str,
    stage: str,
    feature_spec_alias: str | None = None,
) -> dict:
    """Handle stage lock acquisition with stale reclamation.

    Args:
        session: Database session.
        asset_id: The asset ID.
        stage: The pipeline stage.
        feature_spec_alias: Optional feature spec alias (for feature stages).

    Returns:
        Dict with action and reason.
    """
    # Check for existing lock
    existing_lock = _find_stage_lock(session, asset_id, stage, feature_spec_alias)

    reclaim_info = None  # Track if we reclaimed a stale lock

    if existing_lock is not None:
        now = utc_now()
        # Ensure expires_at is timezone-aware for comparison (SQLite may return naive)
        expires_at = existing_lock.expires_at
        if expires_at.tzinfo is None:
            from datetime import UTC

            expires_at = expires_at.replace(tzinfo=UTC)
        if expires_at > now:
            # Lock is still active - skip
            logger.debug("Active lock exists for asset_id=%s, stage=%s", asset_id, stage)
            return {"action": "skip", "reason": "lock_active"}
        else:
            # Lock is stale - reclaim it
            logger.info(
                "Reclaiming stale lock for asset_id=%s, stage=%s (expired at %s)",
                asset_id,
                stage,
                existing_lock.expires_at,
            )
            # Capture reclaim info for metrics before deleting
            reclaim_info = {
                "reclaimed_worker_id": existing_lock.worker_id,
                "expired_at": existing_lock.expires_at.isoformat(),
            }
            session.delete(existing_lock)
            session.flush()

    # Create new lock
    worker_id = f"orchestrator-{uuid.uuid4().hex[:8]}"
    create_stage_lock(
        session=session,
        asset_id=asset_id,
        stage=stage,
        worker_id=worker_id,
        feature_spec_alias=feature_spec_alias,
    )

    result = {"action": "acquired", "worker_id": worker_id}
    if reclaim_info is not None:
        result["reclaim_info"] = reclaim_info
    return result


def _find_stage_lock(
    session: Session,
    asset_id: str,
    stage: str,
    feature_spec_alias: str | None = None,
) -> StageLock | None:
    """Find an existing stage lock.

    Args:
        session: Database session.
        asset_id: The asset ID.
        stage: The pipeline stage.
        feature_spec_alias: Optional feature spec alias.

    Returns:
        StageLock if found, None otherwise.
    """
    if feature_spec_alias is None:
        stmt = select(StageLock).where(
            and_(
                StageLock.asset_id == asset_id,
                StageLock.stage == stage,
                StageLock.feature_spec_alias.is_(None),
            )
        )
    else:
        stmt = select(StageLock).where(
            and_(
                StageLock.asset_id == asset_id,
                StageLock.stage == stage,
                StageLock.feature_spec_alias == feature_spec_alias,
            )
        )
    return session.execute(stmt).scalar_one_or_none()


def _release_stage_lock(
    session: Session,
    asset_id: str,
    stage: str,
    feature_spec_alias: str | None = None,
) -> None:
    """Release a stage lock.

    Args:
        session: Database session.
        asset_id: The asset ID.
        stage: The pipeline stage.
        feature_spec_alias: Optional feature spec alias.
    """
    lock = _find_stage_lock(session, asset_id, stage, feature_spec_alias)
    if lock is not None:
        session.delete(lock)
        session.flush()


# --- Artifact Tracking ---


def _artifact_exists(
    session: Session,
    asset_id: str,
    stage: str,
    feature_spec_alias: str | None = None,
) -> bool:
    """Check if artifact for a stage already exists.

    Checks both artifact_index DB and filesystem.

    Args:
        session: Database session.
        asset_id: The asset ID.
        stage: The pipeline stage.
        feature_spec_alias: Optional feature spec alias (for feature stages).

    Returns:
        True if artifact exists, False otherwise.
    """
    if stage == STAGE_DECODE:
        artifact_type = ARTIFACT_TYPE_NORMALIZED_WAV
        artifact_path = audio_normalized_path(asset_id)

        # Check filesystem first (authoritative)
        if artifact_path.exists():
            return True

        # Check artifact_index
        stmt = select(ArtifactIndex).where(
            and_(
                ArtifactIndex.asset_id == asset_id,
                ArtifactIndex.artifact_type == artifact_type,
            )
        )
        return session.execute(stmt).scalar_one_or_none() is not None

    if stage == STAGE_FEATURES:
        # For features stage, use the provided alias or compute default
        alias = (
            feature_spec_alias
            if feature_spec_alias
            else compute_feature_spec_alias(DEFAULT_FEATURE_SPEC_ID)
        )
        artifact_path = features_h5_path(asset_id, alias)

        # Check filesystem first (authoritative)
        if artifact_path.exists():
            return True

        # Check artifact_index
        stmt = select(ArtifactIndex).where(
            and_(
                ArtifactIndex.asset_id == asset_id,
                ArtifactIndex.artifact_type == ARTIFACT_TYPE_FEATURES_H5,
                ArtifactIndex.feature_spec_alias == alias,
            )
        )
        return session.execute(stmt).scalar_one_or_none() is not None

    if stage == STAGE_SEGMENTS:
        # For segments stage, check for segments JSON file (not .tmp)
        artifact_path = segments_json_path(asset_id)

        # Check filesystem first (authoritative)
        if artifact_path.exists():
            return True

        # Check artifact_index
        stmt = select(ArtifactIndex).where(
            and_(
                ArtifactIndex.asset_id == asset_id,
                ArtifactIndex.artifact_type == ARTIFACT_TYPE_SEGMENTS_V1,
            )
        )
        return session.execute(stmt).scalar_one_or_none() is not None

    # Unknown stage - assume no artifact
    return False


def _record_artifact(
    session: Session,
    asset_id: str,
    artifact_type: str,
    artifact_path: str,
    feature_spec_alias: str | None = None,
    schema_version: str = "1.0.0",
) -> None:
    """Record an artifact in the artifact_index.

    Idempotent: does nothing if artifact already recorded.

    Args:
        session: Database session.
        asset_id: The asset ID.
        artifact_type: Type of artifact.
        artifact_path: Path to the artifact.
        feature_spec_alias: Optional feature spec alias.
        schema_version: Schema version of the artifact.
    """
    # Check if already recorded
    if feature_spec_alias is None:
        stmt = select(ArtifactIndex).where(
            and_(
                ArtifactIndex.asset_id == asset_id,
                ArtifactIndex.artifact_type == artifact_type,
                ArtifactIndex.feature_spec_alias.is_(None),
            )
        )
    else:
        stmt = select(ArtifactIndex).where(
            and_(
                ArtifactIndex.asset_id == asset_id,
                ArtifactIndex.artifact_type == artifact_type,
                ArtifactIndex.feature_spec_alias == feature_spec_alias,
            )
        )

    existing = session.execute(stmt).scalar_one_or_none()
    if existing is not None:
        return  # Already recorded

    artifact = ArtifactIndex(
        asset_id=asset_id,
        artifact_type=artifact_type,
        artifact_path=artifact_path,
        feature_spec_alias=feature_spec_alias,
        schema_version=schema_version,
    )
    session.add(artifact)
    session.flush()


# --- Job Management ---


def _find_completed_ingest_job(session: Session, asset_id: str) -> PipelineJob | None:
    """Find a completed ingest job for the asset.

    Args:
        session: Database session.
        asset_id: The asset ID.

    Returns:
        PipelineJob if found, None otherwise.
    """
    stmt = select(PipelineJob).where(
        and_(
            PipelineJob.asset_id == asset_id,
            PipelineJob.stage == "ingest",
            PipelineJob.status == "completed",
        )
    )
    return session.execute(stmt).scalar_one_or_none()


def _find_dead_letter_job(session: Session, asset_id: str) -> PipelineJob | None:
    """Find any dead-letter job for the asset.

    Args:
        session: Database session.
        asset_id: The asset ID.

    Returns:
        PipelineJob if found, None otherwise.
    """
    stmt = select(PipelineJob).where(
        and_(
            PipelineJob.asset_id == asset_id,
            PipelineJob.status == "dead_letter",
        )
    )
    return session.execute(stmt).scalar_one_or_none()


def _get_or_create_stage_job(
    session: Session,
    asset_id: str,
    stage: str,
    feature_spec_alias: str | None = None,
) -> PipelineJob:
    """Get or create a PipelineJob for a stage.

    If a job exists and is in pending/queued/failed/running state, returns it.
    If no job exists, creates one with status=pending.

    Args:
        session: Database session.
        asset_id: The asset ID.
        stage: The pipeline stage.
        feature_spec_alias: Optional feature spec alias.

    Returns:
        The PipelineJob (existing or new).
    """
    # Look for existing job that can be resumed
    stmt = select(PipelineJob).where(
        and_(
            PipelineJob.asset_id == asset_id,
            PipelineJob.stage == stage,
            PipelineJob.status.in_(["pending", "queued", "failed", "running"]),
        )
    )
    existing = session.execute(stmt).scalar_one_or_none()

    if existing is not None:
        return existing

    # Create new job
    job_id = uuid.uuid4().hex
    job = PipelineJob(
        job_id=job_id,
        asset_id=asset_id,
        stage=stage,
        status="pending",
        attempt=1,
        feature_spec_alias=feature_spec_alias,
    )
    session.add(job)
    session.flush()

    logger.info("Created new job: job_id=%s, asset_id=%s, stage=%s", job_id, asset_id, stage)
    return job


def _determine_next_stage(session: Session, asset_id: str) -> str | None:
    """Determine the next stage to process for an asset.

    Stage progression: ingest -> decode -> features -> segments

    Args:
        session: Database session.
        asset_id: The asset ID.

    Returns:
        Stage name or None if all stages complete.
    """
    # Check if decode is needed
    if not _artifact_exists(session, asset_id, STAGE_DECODE):
        # Check for running/pending job that hasn't failed too many times
        stmt = select(PipelineJob).where(
            and_(
                PipelineJob.asset_id == asset_id,
                PipelineJob.stage == STAGE_DECODE,
            )
        )
        existing_job = session.execute(stmt).scalar_one_or_none()

        if existing_job is None:
            return STAGE_DECODE

        # If job is completed but no artifact, something is wrong - still try decode
        if existing_job.status == "completed":
            # Artifact check already done above, shouldn't reach here normally
            return None

        if existing_job.status == "dead_letter":
            return None  # Don't retry dead-letter

        # pending/failed/running - orchestrator will handle
        return STAGE_DECODE

    # Decode complete - check if features is needed (default v1.4 spec)
    default_alias = compute_feature_spec_alias(DEFAULT_FEATURE_SPEC_ID)
    if not _artifact_exists(session, asset_id, STAGE_FEATURES, default_alias):
        # Check for running/pending job
        stmt = select(PipelineJob).where(
            and_(
                PipelineJob.asset_id == asset_id,
                PipelineJob.stage == STAGE_FEATURES,
            )
        )
        existing_job = session.execute(stmt).scalar_one_or_none()

        if existing_job is None:
            return STAGE_FEATURES

        if existing_job.status == "completed":
            return None

        if existing_job.status == "dead_letter":
            return None  # Don't retry dead-letter

        # pending/failed/running - orchestrator will handle
        return STAGE_FEATURES

    # Features complete - check if segments is needed (Step 6)
    if not _artifact_exists(session, asset_id, STAGE_SEGMENTS):
        # Check for running/pending job
        stmt = select(PipelineJob).where(
            and_(
                PipelineJob.asset_id == asset_id,
                PipelineJob.stage == STAGE_SEGMENTS,
            )
        )
        existing_job = session.execute(stmt).scalar_one_or_none()

        if existing_job is None:
            return STAGE_SEGMENTS

        if existing_job.status == "completed":
            return None

        if existing_job.status == "dead_letter":
            return None  # Don't retry dead-letter

        # pending/failed/running - orchestrator will handle
        return STAGE_SEGMENTS

    # All stages complete
    return None


# --- Cleanup Utilities ---


def cleanup_stale_locks(session: Session, max_age_seconds: int | None = None) -> int:
    """Clean up stale locks.

    Args:
        session: Database session.
        max_age_seconds: Optional override for TTL. Defaults to STAGE_LOCK_TTL_SECONDS.

    Returns:
        Number of locks cleaned up.
    """
    # Note: max_age_seconds param reserved for future use; currently we use expires_at directly
    _ = max_age_seconds  # Silence unused warning; param kept for API compatibility

    stmt = select(StageLock).where(StageLock.expires_at < utc_now())
    stale_locks = session.execute(stmt).scalars().all()

    count = 0
    for lock in stale_locks:
        logger.info(
            "Cleaning up stale lock: asset_id=%s, stage=%s, expired_at=%s",
            lock.asset_id,
            lock.stage,
            lock.expires_at,
        )
        session.delete(lock)
        count += 1

    session.flush()
    return count
