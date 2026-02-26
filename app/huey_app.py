"""SSIS Audio Pipeline - Huey task queue configuration.

Huey setup with SQLite backend for offline-first operation (Blueprint section 9).

How to run:
1. Start the ingest API:
   uvicorn services.ingest_api.main:app --reload

2. Start the Huey consumer (processes queued tasks):
   huey_consumer.py app.huey_app.huey

The consumer will pick up orchestrator tick tasks and dispatch stage workers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from huey import SqliteHuey, crontab

from app.config import HUEY_DB_PATH, QUEUE_DIR

logger = logging.getLogger(__name__)


def _log_event(
    level: int,
    event: str,
    *,
    job_id: str | None = None,
    asset_id: str | None = None,
    stage: str | None = None,
    attempt: int | None = None,
    error_code: str | None = None,
    **extra: object,
) -> None:
    """Emit structured log event with stable keys for pipeline correlation."""
    payload: dict[str, object] = {"event": event, "component": "huey"}

    if job_id is not None:
        payload["job_id"] = job_id
    if asset_id is not None:
        payload["asset_id"] = asset_id
    if stage is not None:
        payload["stage"] = stage
    if attempt is not None:
        payload["attempt"] = attempt
    if error_code is not None:
        payload["error_code"] = error_code

    for key, value in extra.items():
        if value is not None:
            payload[key] = value

    logger.log(level, json.dumps(payload, sort_keys=True, default=str))


# Maximum number of assets to enqueue per periodic sweep to avoid hammering.
SWEEP_BATCH_LIMIT = 50


def _ensure_queue_dir() -> None:
    """Ensure the queue directory exists."""
    Path(QUEUE_DIR).mkdir(parents=True, exist_ok=True)


# Ensure queue directory exists before creating Huey instance
_ensure_queue_dir()

# SQLite-backed Huey instance (offline-friendly)
# Using immediate mode to avoid blocking
huey = SqliteHuey(
    name="ssis_pipeline",
    filename=str(HUEY_DB_PATH),
    immediate=False,  # Tasks queued for consumer processing
)


@huey.task()
def orchestrator_tick_task(asset_id: str) -> dict:
    """Huey task to run orchestrator tick for a specific asset.

    This is the entry point called by Huey consumer to process pipeline stages.

    Args:
        asset_id: The asset ID to process.

    Returns:
        Dict with tick result (for logging/debugging).
    """
    # Import here to avoid circular imports
    from app.orchestrator import orchestrator_tick

    _log_event(
        logging.INFO,
        "orchestrator_tick_task_started",
        asset_id=asset_id,
        stage="orchestrator_tick",
    )
    result = orchestrator_tick(asset_id)
    _log_event(
        logging.INFO,
        "orchestrator_tick_task_completed",
        asset_id=asset_id,
        stage="orchestrator_tick",
        result=result,
    )
    return result


@huey.task()
def stage_worker_task(job_id: str, asset_id: str, stage: str) -> dict:
    """Huey task to execute a pipeline stage.

    This task is dispatched by the orchestrator after acquiring a lock.
    The actual worker logic is a stub in Step 3 - real workers come in Step 4+.

    Args:
        job_id: The PipelineJob ID for this execution.
        asset_id: The asset ID being processed.
        stage: The pipeline stage to execute.

    Returns:
        Dict with execution result.
    """
    # Import here to avoid circular imports
    from app.orchestrator import execute_stage

    _log_event(
        logging.INFO,
        "stage_worker_task_started",
        job_id=job_id,
        asset_id=asset_id,
        stage=stage,
    )
    result = execute_stage(job_id, asset_id, stage)
    _log_event(
        logging.INFO,
        "stage_worker_task_completed",
        job_id=job_id,
        asset_id=asset_id,
        stage=stage,
        result=result,
    )
    return result


def enqueue_orchestrator_tick(asset_id: str) -> None:
    """Enqueue an orchestrator tick for the given asset.

    Non-blocking: returns immediately even if Huey consumer is not running.
    The task will be persisted in SQLite and processed when consumer starts.

    Args:
        asset_id: The asset ID to process.
    """
    _log_event(
        logging.INFO,
        "enqueue_orchestrator_tick",
        asset_id=asset_id,
        stage="orchestrator_tick",
    )
    orchestrator_tick_task(asset_id)


def enqueue_stage_worker(job_id: str, asset_id: str, stage: str, delay_seconds: int = 0) -> None:
    """Enqueue a stage worker task.

    Args:
        job_id: The PipelineJob ID.
        asset_id: The asset ID.
        stage: The pipeline stage.
        delay_seconds: Optional delay before execution (for retries).
    """
    _log_event(
        logging.INFO,
        "enqueue_stage_worker",
        job_id=job_id,
        asset_id=asset_id,
        stage=stage,
        delay_seconds=delay_seconds,
    )
    if delay_seconds > 0:
        stage_worker_task.schedule((job_id, asset_id, stage), delay=delay_seconds)
    else:
        stage_worker_task(job_id, asset_id, stage)


@huey.periodic_task(crontab(minute="*"))
def orchestrator_sweep_task() -> dict:
    """Periodic sweep that enqueues orchestrator ticks for assets needing work.

    Huey invokes this task with no arguments. The sweep inspects the database to
    find assets that have completed ingest, have not been dead-lettered, and
    still have stages remaining, then enqueues orchestrator ticks for a bounded
    batch of those assets.
    """
    from app.db import init_db
    from app.orchestrator import find_assets_needing_tick

    _, SessionFactory = init_db()
    session = SessionFactory()

    try:
        asset_ids = find_assets_needing_tick(session, limit=SWEEP_BATCH_LIMIT)

        if not asset_ids:
            _log_event(
                logging.DEBUG,
                "orchestrator_sweep_no_assets",
                stage="orchestrator_sweep",
                enqueued_count=0,
            )
            return {"enqueued": 0, "asset_ids": []}

        for asset_id in asset_ids:
            orchestrator_tick_task(asset_id)

        _log_event(
            logging.INFO,
            "orchestrator_sweep_enqueued",
            stage="orchestrator_sweep",
            enqueued_count=len(asset_ids),
            asset_ids=asset_ids,
        )
        return {"enqueued": len(asset_ids), "asset_ids": asset_ids}
    except Exception as exc:
        _log_event(
            logging.ERROR,
            "orchestrator_sweep_failed",
            stage="orchestrator_sweep",
            error_code="SWEEP_EXCEPTION",
            exc_type=type(exc).__name__,
            exc_msg=str(exc),
        )
        logger.exception("Orchestrator sweep failed")
        raise
    finally:
        session.close()
