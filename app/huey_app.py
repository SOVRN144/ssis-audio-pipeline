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

import logging
from pathlib import Path

from huey import SqliteHuey

from app.config import HUEY_DB_PATH, QUEUE_DIR

logger = logging.getLogger(__name__)


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

    logger.info("Orchestrator tick task started for asset_id=%s", asset_id)
    result = orchestrator_tick(asset_id)
    logger.info("Orchestrator tick task completed for asset_id=%s: %s", asset_id, result)
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

    logger.info(
        "Stage worker task started: job_id=%s, asset_id=%s, stage=%s",
        job_id, asset_id, stage
    )
    result = execute_stage(job_id, asset_id, stage)
    logger.info("Stage worker task completed: job_id=%s, result=%s", job_id, result)
    return result


def enqueue_orchestrator_tick(asset_id: str) -> None:
    """Enqueue an orchestrator tick for the given asset.

    Non-blocking: returns immediately even if Huey consumer is not running.
    The task will be persisted in SQLite and processed when consumer starts.

    Args:
        asset_id: The asset ID to process.
    """
    logger.info("Enqueueing orchestrator tick for asset_id=%s", asset_id)
    orchestrator_tick_task(asset_id)


def enqueue_stage_worker(job_id: str, asset_id: str, stage: str, delay_seconds: int = 0) -> None:
    """Enqueue a stage worker task.

    Args:
        job_id: The PipelineJob ID.
        asset_id: The asset ID.
        stage: The pipeline stage.
        delay_seconds: Optional delay before execution (for retries).
    """
    logger.info(
        "Enqueueing stage worker: job_id=%s, asset_id=%s, stage=%s, delay=%ds",
        job_id, asset_id, stage, delay_seconds
    )
    if delay_seconds > 0:
        stage_worker_task.schedule((job_id, asset_id, stage), delay=delay_seconds)
    else:
        stage_worker_task(job_id, asset_id, stage)
