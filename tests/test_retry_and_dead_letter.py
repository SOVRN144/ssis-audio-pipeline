"""Tests for retry policy and dead-letter logic.

Step 3: Tests Blueprint section 9 retry semantics:
- 3 attempts max
- Delays: 60s, 300s, 900s
- Dead-letter after 3 failures
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import select

from app.config import MAX_RETRY_ATTEMPTS, RETRY_DELAYS_SECONDS
from app.db import init_db
from app.models import AudioAsset, PipelineJob, StageLock
from app.orchestrator import (
    STAGE_DECODE,
    _handle_stage_failure,
    _mark_dead_letter,
    _orchestrator_tick_impl,
)


@pytest.fixture
def test_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        engine, SessionFactory = init_db(db_path)
        yield SessionFactory
        engine.dispose()


@pytest.fixture
def asset_with_decode_job(test_db):
    """Create an asset with ingest complete and decode job in progress."""
    session = test_db()
    try:
        asset_id = "test-asset-retry"

        asset = AudioAsset(
            asset_id=asset_id,
            content_hash="retry123",
            source_uri="/data/audio/test-asset-retry/original.wav",
            original_filename="test.wav",
        )
        session.add(asset)

        ingest_job = PipelineJob(
            job_id="ingest-retry",
            asset_id=asset_id,
            stage="ingest",
            status="completed",
            attempt=1,
        )
        session.add(ingest_job)

        decode_job = PipelineJob(
            job_id="decode-retry",
            asset_id=asset_id,
            stage=STAGE_DECODE,
            status="running",
            attempt=1,
        )
        session.add(decode_job)
        session.commit()

        yield asset_id, "decode-retry", test_db
    finally:
        session.close()


class TestRetryPolicy:
    """Tests for retry policy configuration."""

    def test_retry_policy_constants(self):
        """Verify retry policy constants match Blueprint section 9."""
        assert MAX_RETRY_ATTEMPTS == 3
        assert RETRY_DELAYS_SECONDS == (60, 300, 900)

    def test_attempt_starts_at_one(self, test_db):
        """Job attempt should start at 1 (not 0)."""
        session = test_db()
        try:
            asset_id = "test-attempt-start"

            asset = AudioAsset(
                asset_id=asset_id,
                content_hash="attemptstart123",
                source_uri="/test/path",
                original_filename="test.wav",
            )
            session.add(asset)

            job = PipelineJob(
                job_id="attempt-start-job",
                asset_id=asset_id,
                stage=STAGE_DECODE,
                status="pending",
            )
            session.add(job)
            session.commit()

            # Verify default attempt is 1
            assert job.attempt == 1
        finally:
            session.close()


class TestRetryBehavior:
    """Tests for retry behavior on failure."""

    def test_first_failure_increments_attempt(self, asset_with_decode_job):
        """First failure should increment attempt to 2."""
        asset_id, job_id, SessionFactory = asset_with_decode_job

        session = SessionFactory()
        try:
            with patch("app.huey_app.orchestrator_tick_task") as mock_tick:
                _handle_stage_failure(session, job_id, "TEST_ERROR", "Test failure")
                session.commit()

            # Verify attempt incremented
            stmt = select(PipelineJob).where(PipelineJob.job_id == job_id)
            job = session.execute(stmt).scalar_one()
            assert job.attempt == 2
            assert job.status == "failed"
            assert job.error_code == "TEST_ERROR"

            # Verify retry was scheduled with correct delay (60s for first retry)
            mock_tick.schedule.assert_called_once()
            call_kwargs = mock_tick.schedule.call_args
            assert call_kwargs[1]["delay"] == RETRY_DELAYS_SECONDS[0]  # 60s
        finally:
            session.close()

    def test_second_failure_uses_second_delay(self, test_db):
        """Second failure should use 300s delay."""
        session = test_db()
        try:
            asset_id = "test-second-fail"

            asset = AudioAsset(
                asset_id=asset_id,
                content_hash="secondfail123",
                source_uri="/test/path",
                original_filename="test.wav",
            )
            session.add(asset)

            job = PipelineJob(
                job_id="second-fail-job",
                asset_id=asset_id,
                stage=STAGE_DECODE,
                status="running",
                attempt=2,  # Already on second attempt
            )
            session.add(job)
            session.commit()

            with patch("app.huey_app.orchestrator_tick_task") as mock_tick:
                _handle_stage_failure(session, "second-fail-job", "TEST_ERROR", "Test failure")
                session.commit()

            # Verify attempt incremented to 3
            stmt = select(PipelineJob).where(PipelineJob.job_id == "second-fail-job")
            job = session.execute(stmt).scalar_one()
            assert job.attempt == 3

            # Verify delay is 300s (second delay value)
            mock_tick.schedule.assert_called_once()
            call_kwargs = mock_tick.schedule.call_args
            assert call_kwargs[1]["delay"] == RETRY_DELAYS_SECONDS[1]  # 300s
        finally:
            session.close()

    def test_third_failure_dead_letters(self, test_db):
        """Third failure should dead-letter (no more retries)."""
        session = test_db()
        try:
            asset_id = "test-third-fail"

            asset = AudioAsset(
                asset_id=asset_id,
                content_hash="thirdfail123",
                source_uri="/test/path",
                original_filename="test.wav",
            )
            session.add(asset)

            job = PipelineJob(
                job_id="third-fail-job",
                asset_id=asset_id,
                stage=STAGE_DECODE,
                status="running",
                attempt=3,  # On third attempt
            )
            session.add(job)
            session.commit()

            with patch("app.huey_app.orchestrator_tick_task") as mock_tick:
                _handle_stage_failure(session, "third-fail-job", "TEST_ERROR", "Test failure")
                session.commit()

            # Verify job is dead-lettered
            stmt = select(PipelineJob).where(PipelineJob.job_id == "third-fail-job")
            job = session.execute(stmt).scalar_one()
            assert job.status == "dead_letter"
            assert job.error_code == "TEST_ERROR"
            assert job.attempt == 3  # Should NOT increment past 3

            # Verify no retry was scheduled
            mock_tick.schedule.assert_not_called()
        finally:
            session.close()


class TestDeadLetterBehavior:
    """Tests for dead-letter skip logic."""

    def test_skip_dead_letter_jobs(self, test_db):
        """Orchestrator should skip assets with dead-letter jobs."""
        session = test_db()
        try:
            asset_id = "test-dead-letter"

            asset = AudioAsset(
                asset_id=asset_id,
                content_hash="deadletter123",
                source_uri="/test/path",
                original_filename="test.wav",
            )
            session.add(asset)

            ingest_job = PipelineJob(
                job_id="ingest-dl",
                asset_id=asset_id,
                stage="ingest",
                status="completed",
                attempt=1,
            )
            session.add(ingest_job)

            dead_job = PipelineJob(
                job_id="dead-decode",
                asset_id=asset_id,
                stage=STAGE_DECODE,
                status="dead_letter",
                attempt=3,
                error_code="WORKER_ERROR",
                error_message="Failed after 3 attempts",
            )
            session.add(dead_job)
            session.commit()

            result = _orchestrator_tick_impl(session, asset_id)

            assert result["status"] == "skipped"
            assert result["reason"] == "dead_letter_exists"
            assert result["dead_letter_stage"] == STAGE_DECODE
        finally:
            session.close()

    def test_dead_letter_not_rescheduled(self, test_db):
        """Dead-letter jobs should not be automatically rescheduled."""
        session = test_db()
        try:
            asset_id = "test-no-resched"

            asset = AudioAsset(
                asset_id=asset_id,
                content_hash="noresched123",
                source_uri="/test/path",
                original_filename="test.wav",
            )
            session.add(asset)

            ingest_job = PipelineJob(
                job_id="ingest-noresched",
                asset_id=asset_id,
                stage="ingest",
                status="completed",
                attempt=1,
            )
            session.add(ingest_job)

            dead_job = PipelineJob(
                job_id="dead-noresched",
                asset_id=asset_id,
                stage=STAGE_DECODE,
                status="dead_letter",
                attempt=3,
            )
            session.add(dead_job)
            session.commit()

            with patch("app.huey_app.enqueue_stage_worker") as mock_enqueue:
                result = _orchestrator_tick_impl(session, asset_id)

            # Should skip, not dispatch
            assert result["status"] == "skipped"
            mock_enqueue.assert_not_called()
        finally:
            session.close()

    def test_dead_letter_stores_error_info(self, test_db):
        """Dead-letter should store error_code and error_message."""
        session = test_db()
        try:
            asset_id = "test-dl-error"

            asset = AudioAsset(
                asset_id=asset_id,
                content_hash="dlerror123",
                source_uri="/test/path",
                original_filename="test.wav",
            )
            session.add(asset)

            job = PipelineJob(
                job_id="dl-error-job",
                asset_id=asset_id,
                stage=STAGE_DECODE,
                status="running",
                attempt=3,
            )
            session.add(job)
            session.commit()

            _mark_dead_letter(session, job, "Worker crashed", "WORKER_CRASH")
            session.commit()

            stmt = select(PipelineJob).where(PipelineJob.job_id == "dl-error-job")
            job = session.execute(stmt).scalar_one()
            assert job.status == "dead_letter"
            assert job.error_code == "WORKER_CRASH"
            assert job.error_message == "Worker crashed"
            assert job.finished_at is not None
        finally:
            session.close()


class TestLockReleaseOnFailure:
    """Tests for lock release on failure/dead-letter."""

    def test_lock_released_on_failure(self, test_db):
        """Stage lock should be released when job fails."""
        session = test_db()
        try:
            asset_id = "test-lock-release"

            asset = AudioAsset(
                asset_id=asset_id,
                content_hash="lockrelease123",
                source_uri="/test/path",
                original_filename="test.wav",
            )
            session.add(asset)

            job = PipelineJob(
                job_id="lock-release-job",
                asset_id=asset_id,
                stage=STAGE_DECODE,
                status="running",
                attempt=1,
            )
            session.add(job)

            from datetime import timedelta

            from app.models import utc_now

            lock = StageLock(
                asset_id=asset_id,
                stage=STAGE_DECODE,
                worker_id="test-worker",
                acquired_at=utc_now(),
                expires_at=utc_now() + timedelta(minutes=10),
            )
            session.add(lock)
            session.commit()

            with patch("app.huey_app.orchestrator_tick_task"):
                _handle_stage_failure(session, "lock-release-job", "TEST_ERROR", "Test")
                session.commit()

            # Lock should be released
            stmt = select(StageLock).where(
                StageLock.asset_id == asset_id,
                StageLock.stage == STAGE_DECODE,
            )
            lock = session.execute(stmt).scalar_one_or_none()
            assert lock is None
        finally:
            session.close()

    def test_lock_released_on_dead_letter(self, test_db):
        """Stage lock should be released when job is dead-lettered."""
        session = test_db()
        try:
            asset_id = "test-lock-dl"

            asset = AudioAsset(
                asset_id=asset_id,
                content_hash="lockdl123",
                source_uri="/test/path",
                original_filename="test.wav",
            )
            session.add(asset)

            job = PipelineJob(
                job_id="lock-dl-job",
                asset_id=asset_id,
                stage=STAGE_DECODE,
                status="running",
                attempt=3,  # Will dead-letter
            )
            session.add(job)

            from datetime import timedelta

            from app.models import utc_now

            lock = StageLock(
                asset_id=asset_id,
                stage=STAGE_DECODE,
                worker_id="test-worker",
                acquired_at=utc_now(),
                expires_at=utc_now() + timedelta(minutes=10),
            )
            session.add(lock)
            session.commit()

            with patch("app.huey_app.orchestrator_tick_task"):
                _handle_stage_failure(session, "lock-dl-job", "TEST_ERROR", "Test")
                session.commit()

            # Lock should be released even on dead-letter
            stmt = select(StageLock).where(
                StageLock.asset_id == asset_id,
                StageLock.stage == STAGE_DECODE,
            )
            lock = session.execute(stmt).scalar_one_or_none()
            assert lock is None
        finally:
            session.close()
