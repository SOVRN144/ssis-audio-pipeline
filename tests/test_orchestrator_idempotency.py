"""Tests for orchestrator idempotency.

Step 3: Ensures running tick twice creates no duplicates.
Step 5: Tests for STAGE_FEATURES lock alias consistency.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import func, select

from app.config import DEFAULT_FEATURE_SPEC_ID
from app.db import init_db
from app.models import ArtifactIndex, AudioAsset, PipelineJob, StageLock
from app.orchestrator import (
    ARTIFACT_TYPE_FEATURES_H5,
    ARTIFACT_TYPE_NORMALIZED_WAV,
    ARTIFACT_TYPE_SEGMENTS_V1,
    STAGE_DECODE,
    STAGE_FEATURES,
    STAGE_SEGMENTS,
    _orchestrator_tick_impl,
)
from app.utils.hashing import feature_spec_alias as compute_feature_spec_alias


@pytest.fixture
def test_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        engine, SessionFactory = init_db(db_path)
        yield SessionFactory
        engine.dispose()


@pytest.fixture
def asset_with_ingest(test_db):
    """Create an asset with completed ingest job."""
    session = test_db()
    try:
        asset_id = "test-asset-idemp"

        asset = AudioAsset(
            asset_id=asset_id,
            content_hash="idemp123",
            source_uri="/data/audio/test-asset-idemp/original.wav",
            original_filename="test.wav",
        )
        session.add(asset)

        job = PipelineJob(
            job_id="ingest-job-idemp",
            asset_id=asset_id,
            stage="ingest",
            status="completed",
            attempt=1,
        )
        session.add(job)
        session.commit()

        yield asset_id, test_db
    finally:
        session.close()


class TestOrchestratorIdempotency:
    """Tests for idempotent orchestrator tick behavior."""

    def test_tick_twice_creates_one_lock(self, asset_with_ingest):
        """Running tick twice should create exactly one StageLock."""
        asset_id, SessionFactory = asset_with_ingest

        session = SessionFactory()
        try:
            with patch("app.huey_app.enqueue_stage_worker"):
                # First tick
                result1 = _orchestrator_tick_impl(session, asset_id)
                session.commit()

                # Second tick
                result2 = _orchestrator_tick_impl(session, asset_id)
                session.commit()

            # First tick should dispatch
            assert result1["status"] == "dispatched"

            # Second tick should skip (lock active)
            assert result2["status"] == "skipped"
            assert result2["reason"] == "lock_active"

            # Verify only one lock exists
            stmt = (
                select(func.count())
                .select_from(StageLock)
                .where(
                    StageLock.asset_id == asset_id,
                    StageLock.stage == STAGE_DECODE,
                )
            )
            count = session.execute(stmt).scalar()
            assert count == 1
        finally:
            session.close()

    def test_tick_twice_creates_one_job(self, asset_with_ingest):
        """Running tick twice should create exactly one decode PipelineJob."""
        asset_id, SessionFactory = asset_with_ingest

        session = SessionFactory()
        try:
            with patch("app.huey_app.enqueue_stage_worker"):
                _orchestrator_tick_impl(session, asset_id)
                session.commit()

                _orchestrator_tick_impl(session, asset_id)
                session.commit()

            # Verify only one decode job exists
            stmt = (
                select(func.count())
                .select_from(PipelineJob)
                .where(
                    PipelineJob.asset_id == asset_id,
                    PipelineJob.stage == STAGE_DECODE,
                )
            )
            count = session.execute(stmt).scalar()
            assert count == 1
        finally:
            session.close()

    def test_skip_decode_if_artifact_exists_in_index(self, asset_with_ingest):
        """Should skip decode if artifact is recorded in artifact_index.

        When decode artifact exists, the next stage is features.
        This test verifies decode is skipped but features is dispatched.
        """
        asset_id, SessionFactory = asset_with_ingest

        session = SessionFactory()
        try:
            # Pre-record decode artifact in index
            artifact = ArtifactIndex(
                asset_id=asset_id,
                artifact_type=ARTIFACT_TYPE_NORMALIZED_WAV,
                artifact_path=f"/data/audio/{asset_id}/normalized.wav",
                schema_version="1.0.0",
            )
            session.add(artifact)
            session.commit()

            result = _orchestrator_tick_impl(session, asset_id)

            # Decode artifact exists => decode is skipped, features is dispatched
            # (or features artifact check fails, but decode stage is still skipped)
            assert result["status"] == "dispatched"
            assert result["stage"] == "features"

            # Verify no lock or job created for decode
            lock_stmt = (
                select(func.count())
                .select_from(StageLock)
                .where(
                    StageLock.asset_id == asset_id,
                    StageLock.stage == STAGE_DECODE,
                )
            )
            lock_count = session.execute(lock_stmt).scalar()
            assert lock_count == 0

            job_stmt = (
                select(func.count())
                .select_from(PipelineJob)
                .where(
                    PipelineJob.asset_id == asset_id,
                    PipelineJob.stage == STAGE_DECODE,
                )
            )
            job_count = session.execute(job_stmt).scalar()
            assert job_count == 0
        finally:
            session.close()

    def test_skip_decode_if_artifact_exists_on_filesystem(self, asset_with_ingest):
        """Should skip decode if artifact file exists on filesystem.

        When decode artifact exists on filesystem, decode is skipped
        and features stage is dispatched next.
        """
        asset_id, SessionFactory = asset_with_ingest

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create the normalized.wav file
            asset_dir = tmpdir_path / asset_id
            asset_dir.mkdir(parents=True)
            normalized_path = asset_dir / "normalized.wav"
            normalized_path.write_bytes(b"fake wav data")

            # Patch at config source so paths module picks it up
            with patch("app.config.AUDIO_DIR", tmpdir_path):
                # Also patch the paths module directly since it imports at module load
                with patch("app.utils.paths.AUDIO_DIR", tmpdir_path):
                    session = SessionFactory()
                    try:
                        result = _orchestrator_tick_impl(session, asset_id)

                        # Decode artifact exists => decode skipped, features dispatched
                        assert result["status"] == "dispatched"
                        assert result["stage"] == "features"
                    finally:
                        session.close()

    def test_idempotent_artifact_recording(self, test_db):
        """Recording same artifact twice should not create duplicates."""
        session = test_db()
        try:
            asset_id = "test-artifact-idemp"

            # Create asset
            asset = AudioAsset(
                asset_id=asset_id,
                content_hash="artifactidemp123",
                source_uri="/test/path",
                original_filename="test.wav",
            )
            session.add(asset)
            session.commit()

            from app.orchestrator import _record_artifact

            # Record artifact twice
            _record_artifact(
                session, asset_id, ARTIFACT_TYPE_NORMALIZED_WAV, "/path/to/normalized.wav"
            )
            session.flush()

            _record_artifact(
                session, asset_id, ARTIFACT_TYPE_NORMALIZED_WAV, "/path/to/normalized.wav"
            )
            session.flush()

            # Verify only one record exists
            stmt = (
                select(func.count())
                .select_from(ArtifactIndex)
                .where(
                    ArtifactIndex.asset_id == asset_id,
                    ArtifactIndex.artifact_type == ARTIFACT_TYPE_NORMALIZED_WAV,
                )
            )
            count = session.execute(stmt).scalar()
            assert count == 1
        finally:
            session.close()


class TestStaleLockReclamation:
    """Tests for stale lock reclamation."""

    def test_reclaim_stale_lock(self, asset_with_ingest):
        """Should reclaim stale locks and dispatch."""
        asset_id, SessionFactory = asset_with_ingest

        from datetime import timedelta

        from app.models import utc_now

        session = SessionFactory()
        try:
            # Create a stale lock (expired in the past)
            stale_lock = StageLock(
                asset_id=asset_id,
                stage=STAGE_DECODE,
                worker_id="old-worker",
                acquired_at=utc_now() - timedelta(hours=1),
                expires_at=utc_now() - timedelta(minutes=30),  # Expired
            )
            session.add(stale_lock)
            session.commit()

            with patch("app.huey_app.enqueue_stage_worker"):
                result = _orchestrator_tick_impl(session, asset_id)
                session.commit()

            # Should have reclaimed and dispatched
            assert result["status"] == "dispatched"

            # Lock should have been reclaimed (worker_id changed from "old-worker")
            # Note: SQLAlchemy may reuse the same row id when deleting and recreating
            # within the same transaction, so we check worker_id instead of row id
            stmt = select(StageLock).where(
                StageLock.asset_id == asset_id,
                StageLock.stage == STAGE_DECODE,
            )
            new_lock = session.execute(stmt).scalar_one()
            assert new_lock.worker_id != "old-worker"
            # Check new lock is not expired (handle potential timezone-naive datetime)
            from datetime import UTC

            expires_at = new_lock.expires_at
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            assert expires_at > utc_now()  # New lock should not be expired
        finally:
            session.close()

    def test_do_not_reclaim_active_lock(self, asset_with_ingest):
        """Should not reclaim locks that are still active."""
        asset_id, SessionFactory = asset_with_ingest

        from datetime import timedelta

        from app.models import utc_now

        session = SessionFactory()
        try:
            # Create an active lock (expires in the future)
            active_lock = StageLock(
                asset_id=asset_id,
                stage=STAGE_DECODE,
                worker_id="active-worker",
                acquired_at=utc_now(),
                expires_at=utc_now() + timedelta(minutes=5),  # Still active
            )
            session.add(active_lock)
            session.commit()

            result = _orchestrator_tick_impl(session, asset_id)

            # Should skip due to active lock
            assert result["status"] == "skipped"
            assert result["reason"] == "lock_active"

            # Original lock should still exist
            stmt = select(StageLock).where(
                StageLock.asset_id == asset_id,
                StageLock.stage == STAGE_DECODE,
            )
            lock = session.execute(stmt).scalar_one()
            assert lock.worker_id == "active-worker"
        finally:
            session.close()

    def test_reclaim_stale_lock_with_naive_datetime(self, asset_with_ingest):
        """Should safely reclaim stale locks when DB returns naive datetime.

        SQLite commonly returns naive datetime values (no tzinfo).
        The orchestrator must handle this defensively and still reclaim expired locks.
        """
        asset_id, SessionFactory = asset_with_ingest

        from datetime import UTC, datetime, timedelta

        session = SessionFactory()
        try:
            # Create a stale lock with NAIVE datetime (no tzinfo) - simulates SQLite behavior
            # We need to bypass SQLAlchemy's default handling, so use raw datetime
            past_naive = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=1)
            expired_naive = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=30)

            stale_lock = StageLock(
                asset_id=asset_id,
                stage=STAGE_DECODE,
                worker_id="naive-worker",
                acquired_at=past_naive,  # Naive datetime
                expires_at=expired_naive,  # Naive datetime - expired
            )
            session.add(stale_lock)
            session.commit()

            # Verify the lock was stored - we're testing that orchestrator
            # handles naive datetimes correctly (SQLite behavior)
            stmt = select(StageLock).where(StageLock.asset_id == asset_id)
            _ = session.execute(stmt).scalar_one()  # Confirm lock exists

            with patch("app.huey_app.enqueue_stage_worker"):
                # This should NOT crash even with naive datetime
                result = _orchestrator_tick_impl(session, asset_id)
                session.commit()

            # Should have reclaimed the stale lock and dispatched
            assert result["status"] == "dispatched"

            # Lock should have been reclaimed (worker_id changed)
            stmt = select(StageLock).where(
                StageLock.asset_id == asset_id,
                StageLock.stage == STAGE_DECODE,
            )
            new_lock = session.execute(stmt).scalar_one()
            assert new_lock.worker_id != "naive-worker"
        finally:
            session.close()

    def test_lock_reclaim_records_metrics(self, asset_with_ingest):
        """Stale lock reclaim should record metrics in PipelineJob.metrics_json."""
        import json

        asset_id, SessionFactory = asset_with_ingest

        from datetime import timedelta

        from app.models import utc_now

        session = SessionFactory()
        try:
            # Create a stale lock
            stale_lock = StageLock(
                asset_id=asset_id,
                stage=STAGE_DECODE,
                worker_id="metrics-test-worker",
                acquired_at=utc_now() - timedelta(hours=1),
                expires_at=utc_now() - timedelta(minutes=30),
            )
            session.add(stale_lock)
            session.commit()

            with patch("app.huey_app.enqueue_stage_worker"):
                result = _orchestrator_tick_impl(session, asset_id)
                session.commit()

            assert result["status"] == "dispatched"

            # Fetch the job and check metrics_json
            stmt = select(PipelineJob).where(
                PipelineJob.asset_id == asset_id,
                PipelineJob.stage == STAGE_DECODE,
            )
            job = session.execute(stmt).scalar_one()

            # Verify metrics_json contains lock_reclaim data
            assert job.metrics_json is not None
            metrics = json.loads(job.metrics_json)
            assert "lock_reclaim" in metrics
            assert metrics["lock_reclaim"]["count"] == 1
            assert len(metrics["lock_reclaim"]["events"]) == 1

            event = metrics["lock_reclaim"]["events"][0]
            assert event["reclaimed_worker_id"] == "metrics-test-worker"
            assert "expired_at" in event
            assert "reclaimed_at" in event
        finally:
            session.close()


class TestFeaturesLockAliasConsistency:
    """Tests for STAGE_FEATURES lock alias consistency (prevents lock leaks)."""

    @pytest.fixture
    def asset_with_decode_complete(self, test_db):
        """Create an asset with completed ingest and decode jobs."""
        session = test_db()
        try:
            asset_id = "test-features-lock"

            asset = AudioAsset(
                asset_id=asset_id,
                content_hash="features-lock-test",
                source_uri=f"/data/audio/{asset_id}/original.wav",
                original_filename="test.wav",
            )
            session.add(asset)

            # Completed ingest job
            ingest_job = PipelineJob(
                job_id="ingest-features-lock",
                asset_id=asset_id,
                stage="ingest",
                status="completed",
                attempt=1,
            )
            session.add(ingest_job)

            # Completed decode job
            decode_job = PipelineJob(
                job_id="decode-features-lock",
                asset_id=asset_id,
                stage=STAGE_DECODE,
                status="completed",
                attempt=1,
            )
            session.add(decode_job)

            # Decode artifact exists
            artifact = ArtifactIndex(
                asset_id=asset_id,
                artifact_type=ARTIFACT_TYPE_NORMALIZED_WAV,
                artifact_path=f"/data/audio/{asset_id}/normalized.wav",
                schema_version="1.0.0",
            )
            session.add(artifact)

            session.commit()
            yield asset_id, test_db
        finally:
            session.close()

    def test_features_lock_acquired_with_alias(self, asset_with_decode_complete):
        """Lock for STAGE_FEATURES must be acquired with computed alias."""
        asset_id, SessionFactory = asset_with_decode_complete
        expected_alias = compute_feature_spec_alias(DEFAULT_FEATURE_SPEC_ID)

        session = SessionFactory()
        try:
            with patch("app.huey_app.enqueue_stage_worker"):
                result = _orchestrator_tick_impl(session, asset_id)
                session.commit()

            assert result["status"] == "dispatched"
            assert result["stage"] == STAGE_FEATURES

            # Verify lock was acquired with the correct alias
            stmt = select(StageLock).where(
                StageLock.asset_id == asset_id,
                StageLock.stage == STAGE_FEATURES,
            )
            lock = session.execute(stmt).scalar_one()
            assert lock.feature_spec_alias == expected_alias
        finally:
            session.close()

    def test_features_job_has_alias_set(self, asset_with_decode_complete):
        """PipelineJob for STAGE_FEATURES must have feature_spec_alias set."""
        asset_id, SessionFactory = asset_with_decode_complete
        expected_alias = compute_feature_spec_alias(DEFAULT_FEATURE_SPEC_ID)

        session = SessionFactory()
        try:
            with patch("app.huey_app.enqueue_stage_worker"):
                result = _orchestrator_tick_impl(session, asset_id)
                session.commit()

            assert result["status"] == "dispatched"

            # Verify job has feature_spec_alias set
            stmt = select(PipelineJob).where(
                PipelineJob.asset_id == asset_id,
                PipelineJob.stage == STAGE_FEATURES,
            )
            job = session.execute(stmt).scalar_one()
            assert job.feature_spec_alias == expected_alias
        finally:
            session.close()

    def test_features_tick_twice_respects_lock_with_alias(self, asset_with_decode_complete):
        """Second tick should skip due to active lock (with alias)."""
        asset_id, SessionFactory = asset_with_decode_complete

        session = SessionFactory()
        try:
            with patch("app.huey_app.enqueue_stage_worker"):
                # First tick
                result1 = _orchestrator_tick_impl(session, asset_id)
                session.commit()

                # Second tick
                result2 = _orchestrator_tick_impl(session, asset_id)
                session.commit()

            assert result1["status"] == "dispatched"
            assert result2["status"] == "skipped"
            assert result2["reason"] == "lock_active"

            # Verify only one lock exists
            stmt = (
                select(func.count())
                .select_from(StageLock)
                .where(
                    StageLock.asset_id == asset_id,
                    StageLock.stage == STAGE_FEATURES,
                )
            )
            count = session.execute(stmt).scalar()
            assert count == 1
        finally:
            session.close()

    def test_features_artifact_skip_with_alias(self, asset_with_decode_complete):
        """Should skip STAGE_FEATURES if artifact with matching alias exists.

        Note: With STAGE_SEGMENTS now in the pipeline, this test verifies that
        STAGE_FEATURES is skipped (no features lock created) and STAGE_SEGMENTS
        is dispatched next.
        """
        asset_id, SessionFactory = asset_with_decode_complete
        expected_alias = compute_feature_spec_alias(DEFAULT_FEATURE_SPEC_ID)

        session = SessionFactory()
        try:
            # Pre-create features artifact with the expected alias
            features_artifact = ArtifactIndex(
                asset_id=asset_id,
                artifact_type=ARTIFACT_TYPE_FEATURES_H5,
                artifact_path=f"/data/features/{asset_id}.{expected_alias}.h5",
                feature_spec_alias=expected_alias,
                schema_version="1.0.0",
            )
            session.add(features_artifact)
            session.commit()

            result = _orchestrator_tick_impl(session, asset_id)

            # STAGE_FEATURES is skipped, STAGE_SEGMENTS is dispatched
            assert result["status"] == "dispatched"
            assert result["stage"] == STAGE_SEGMENTS

            # Verify no features lock created (features was skipped)
            stmt = (
                select(func.count())
                .select_from(StageLock)
                .where(
                    StageLock.asset_id == asset_id,
                    StageLock.stage == STAGE_FEATURES,
                )
            )
            count = session.execute(stmt).scalar()
            assert count == 0

            # Verify segments lock was created
            stmt = (
                select(func.count())
                .select_from(StageLock)
                .where(
                    StageLock.asset_id == asset_id,
                    StageLock.stage == STAGE_SEGMENTS,
                )
            )
            count = session.execute(stmt).scalar()
            assert count == 1
        finally:
            session.close()

    def test_all_stages_complete_returns_no_work(self, asset_with_decode_complete):
        """Should return no_work when all stages including segments are complete."""
        asset_id, SessionFactory = asset_with_decode_complete
        expected_alias = compute_feature_spec_alias(DEFAULT_FEATURE_SPEC_ID)

        session = SessionFactory()
        try:
            # Pre-create features artifact
            features_artifact = ArtifactIndex(
                asset_id=asset_id,
                artifact_type=ARTIFACT_TYPE_FEATURES_H5,
                artifact_path=f"/data/features/{asset_id}.{expected_alias}.h5",
                feature_spec_alias=expected_alias,
                schema_version="1.0.0",
            )
            session.add(features_artifact)

            # Pre-create segments artifact
            segments_artifact = ArtifactIndex(
                asset_id=asset_id,
                artifact_type=ARTIFACT_TYPE_SEGMENTS_V1,
                artifact_path=f"/data/segments/{asset_id}.segments.v1.json",
                feature_spec_alias=None,
                schema_version="1.0.0",
            )
            session.add(segments_artifact)
            session.commit()

            result = _orchestrator_tick_impl(session, asset_id)

            # All stages complete - no work
            assert result["status"] == "no_work"
            assert result["reason"] == "all_stages_complete"
        finally:
            session.close()
