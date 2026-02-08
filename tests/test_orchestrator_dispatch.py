"""Tests for orchestrator dispatch logic.

Step 3: Tests planning + dispatch of decode stage after ingest completion.
"""

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import select

from app.db import init_db
from app.models import AudioAsset, PipelineJob, StageLock
from app.orchestrator import (
    ARTIFACT_TYPE_FEATURES_H5,
    ARTIFACT_TYPE_NORMALIZED_WAV,
    ARTIFACT_TYPE_SEGMENTS_V1,
    STAGE_DECODE,
    STAGE_FEATURES,
    STAGE_PREVIEW,
    _find_completed_ingest_job,
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
        asset_id = "test-asset-001"

        # Create AudioAsset
        asset = AudioAsset(
            asset_id=asset_id,
            content_hash="abc123def456",
            source_uri="/data/audio/test-asset-001/original.wav",
            original_filename="test.wav",
        )
        session.add(asset)

        # Create completed ingest job
        job = PipelineJob(
            job_id="ingest-job-001",
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


class TestOrchestratorDispatch:
    """Tests for orchestrator tick dispatch logic."""

    def test_dispatches_decode_after_ingest(self, asset_with_ingest):
        """Given completed ingest, orchestrator should dispatch decode stage."""
        asset_id, SessionFactory = asset_with_ingest

        session = SessionFactory()
        try:
            # Mock the Huey enqueue to prevent actual task dispatch
            with patch("app.huey_app.enqueue_stage_worker") as mock_enqueue:
                result = _orchestrator_tick_impl(session, asset_id)
                session.commit()

            # Verify dispatch result
            assert result["status"] == "dispatched"
            assert result["asset_id"] == asset_id
            assert result["stage"] == STAGE_DECODE
            assert "job_id" in result
            assert result["attempt"] == 1

            # Verify StageLock was created
            stmt = select(StageLock).where(
                StageLock.asset_id == asset_id,
                StageLock.stage == STAGE_DECODE,
            )
            lock = session.execute(stmt).scalar_one()
            assert lock is not None
            assert lock.feature_spec_alias is None  # decode has no feature_spec

            # Verify PipelineJob was created
            stmt = select(PipelineJob).where(
                PipelineJob.asset_id == asset_id,
                PipelineJob.stage == STAGE_DECODE,
            )
            job = session.execute(stmt).scalar_one()
            assert job is not None
            assert job.status == "running"
            assert job.attempt == 1

            # Verify Huey was called
            mock_enqueue.assert_called_once()
            call_args = mock_enqueue.call_args[0]
            assert call_args[1] == asset_id
            assert call_args[2] == STAGE_DECODE
        finally:
            session.close()

    def test_lock_key_includes_feature_spec_alias_null(self, asset_with_ingest):
        """StageLock for decode should have feature_spec_alias=NULL."""
        asset_id, SessionFactory = asset_with_ingest

        session = SessionFactory()
        try:
            with patch("app.huey_app.enqueue_stage_worker"):
                _orchestrator_tick_impl(session, asset_id)
                session.commit()

            # Verify lock key structure
            stmt = select(StageLock).where(
                StageLock.asset_id == asset_id,
                StageLock.stage == STAGE_DECODE,
                StageLock.feature_spec_alias.is_(None),
            )
            lock = session.execute(stmt).scalar_one_or_none()
            assert lock is not None
        finally:
            session.close()

    def test_job_status_pending_on_creation(self, test_db):
        """New jobs should have status='pending' before running."""
        session = test_db()
        try:
            asset_id = "test-asset-pending"

            # Create asset and ingest job
            asset = AudioAsset(
                asset_id=asset_id,
                content_hash="pending123",
                source_uri="/test/path",
                original_filename="test.wav",
            )
            session.add(asset)

            ingest_job = PipelineJob(
                job_id="ingest-pending",
                asset_id=asset_id,
                stage="ingest",
                status="completed",
                attempt=1,
            )
            session.add(ingest_job)
            session.commit()

            # Import here to test job creation
            from app.orchestrator import _get_or_create_stage_job

            job = _get_or_create_stage_job(session, asset_id, STAGE_DECODE)
            session.flush()

            # Job should be pending before dispatch
            assert job.status == "pending"
            assert job.attempt == 1
        finally:
            session.close()

    def test_no_dispatch_without_completed_ingest(self, test_db):
        """Should not dispatch if no completed ingest job exists."""
        session = test_db()
        try:
            asset_id = "test-asset-no-ingest"

            # Create asset but no ingest job
            asset = AudioAsset(
                asset_id=asset_id,
                content_hash="noingest123",
                source_uri="/test/path",
                original_filename="test.wav",
            )
            session.add(asset)
            session.commit()

            result = _orchestrator_tick_impl(session, asset_id)

            assert result["status"] == "no_work"
            assert result["reason"] == "no_completed_ingest"
        finally:
            session.close()

    def test_no_dispatch_with_pending_ingest(self, test_db):
        """Should not dispatch if ingest is still pending."""
        session = test_db()
        try:
            asset_id = "test-asset-pending-ingest"

            asset = AudioAsset(
                asset_id=asset_id,
                content_hash="pendingingest123",
                source_uri="/test/path",
                original_filename="test.wav",
            )
            session.add(asset)

            # Pending ingest job
            job = PipelineJob(
                job_id="pending-ingest",
                asset_id=asset_id,
                stage="ingest",
                status="pending",
                attempt=1,
            )
            session.add(job)
            session.commit()

            result = _orchestrator_tick_impl(session, asset_id)

            assert result["status"] == "no_work"
            assert result["reason"] == "no_completed_ingest"
        finally:
            session.close()

    def test_featurepack_missing_reroutes_to_features_with_same_alias(self, test_db):
        """Preview FEATUREPACK_MISSING should dispatch features using the missing alias."""
        session = test_db()
        try:
            asset_id = "test-featurepack-reroute"
            default_alias = compute_feature_spec_alias(
                "mel64_h10ms_w25ms_sr22050__yamnet1024_h0.5s_onnx"
            )
            missing_alias = "abcdef123456"

            session.add(
                AudioAsset(
                    asset_id=asset_id,
                    content_hash="reroute123",
                    source_uri=f"/data/audio/{asset_id}/original.wav",
                    original_filename="test.wav",
                )
            )
            session.add(
                PipelineJob(
                    job_id="ingest-reroute",
                    asset_id=asset_id,
                    stage="ingest",
                    status="completed",
                    attempt=1,
                )
            )

            # Decode + default features + segments already exist, so planner reaches preview.
            from app.models import ArtifactIndex

            session.add(
                ArtifactIndex(
                    asset_id=asset_id,
                    artifact_type=ARTIFACT_TYPE_NORMALIZED_WAV,
                    artifact_path=f"/data/audio/{asset_id}/normalized.wav",
                    schema_version="1.0.0",
                )
            )
            session.add(
                ArtifactIndex(
                    asset_id=asset_id,
                    artifact_type=ARTIFACT_TYPE_FEATURES_H5,
                    artifact_path=f"/data/features/{asset_id}.{default_alias}.h5",
                    feature_spec_alias=default_alias,
                    schema_version="1.0.0",
                )
            )
            session.add(
                ArtifactIndex(
                    asset_id=asset_id,
                    artifact_type=ARTIFACT_TYPE_SEGMENTS_V1,
                    artifact_path=f"/data/segments/{asset_id}.segments.v1.json",
                    schema_version="1.0.0",
                )
            )

            session.add(
                PipelineJob(
                    job_id="preview-missing-pack",
                    asset_id=asset_id,
                    stage=STAGE_PREVIEW,
                    status="failed",
                    attempt=1,
                    error_code="FEATUREPACK_MISSING",
                    error_message=f"FeaturePack not found for alias {missing_alias}",
                    metrics_json='{"preview":{"spec_alias_used":"abcdef123456"}}',
                )
            )
            session.commit()

            with patch("app.huey_app.enqueue_stage_worker") as mock_enqueue:
                result = _orchestrator_tick_impl(session, asset_id)
                session.commit()

            assert result["status"] == "dispatched"
            assert result["stage"] == STAGE_FEATURES

            stmt = select(PipelineJob).where(
                PipelineJob.asset_id == asset_id,
                PipelineJob.stage == STAGE_FEATURES,
                PipelineJob.status == "running",
            )
            features_job = session.execute(stmt).scalar_one()
            assert features_job.feature_spec_alias == missing_alias

            call_args = mock_enqueue.call_args[0]
            assert call_args[2] == STAGE_FEATURES
        finally:
            session.close()

    def test_featurepack_missing_without_alias_caps_reroute(self, test_db):
        """If alias cannot be recovered and features already completed, do not reroute again."""
        session = test_db()
        try:
            asset_id = "test-featurepack-reroute-cap"
            default_alias = compute_feature_spec_alias(
                "mel64_h10ms_w25ms_sr22050__yamnet1024_h0.5s_onnx"
            )

            session.add(
                AudioAsset(
                    asset_id=asset_id,
                    content_hash="reroutecap123",
                    source_uri=f"/data/audio/{asset_id}/original.wav",
                    original_filename="test.wav",
                )
            )
            session.add(
                PipelineJob(
                    job_id="ingest-reroute-cap",
                    asset_id=asset_id,
                    stage="ingest",
                    status="completed",
                    attempt=1,
                )
            )

            from app.models import ArtifactIndex

            session.add(
                ArtifactIndex(
                    asset_id=asset_id,
                    artifact_type=ARTIFACT_TYPE_NORMALIZED_WAV,
                    artifact_path=f"/data/audio/{asset_id}/normalized.wav",
                    schema_version="1.0.0",
                )
            )
            session.add(
                ArtifactIndex(
                    asset_id=asset_id,
                    artifact_type=ARTIFACT_TYPE_FEATURES_H5,
                    artifact_path=f"/data/features/{asset_id}.{default_alias}.h5",
                    feature_spec_alias=default_alias,
                    schema_version="1.0.0",
                )
            )
            session.add(
                ArtifactIndex(
                    asset_id=asset_id,
                    artifact_type=ARTIFACT_TYPE_SEGMENTS_V1,
                    artifact_path=f"/data/segments/{asset_id}.segments.v1.json",
                    schema_version="1.0.0",
                )
            )
            session.add(
                PipelineJob(
                    job_id="features-completed-cap",
                    asset_id=asset_id,
                    stage=STAGE_FEATURES,
                    status="completed",
                    attempt=1,
                    feature_spec_alias=default_alias,
                )
            )
            session.add(
                PipelineJob(
                    job_id="preview-missing-pack-cap",
                    asset_id=asset_id,
                    stage=STAGE_PREVIEW,
                    status="failed",
                    attempt=1,
                    error_code="FEATUREPACK_MISSING",
                    error_message="FeaturePack missing but alias unavailable",
                )
            )
            session.commit()

            with patch("app.huey_app.enqueue_stage_worker") as mock_enqueue:
                result = _orchestrator_tick_impl(session, asset_id)
                session.commit()

            assert result["status"] == "dispatched"
            assert result["stage"] == STAGE_PREVIEW
            call_args = mock_enqueue.call_args[0]
            assert call_args[2] == STAGE_PREVIEW
        finally:
            session.close()


def test_find_completed_ingest_job_returns_latest(test_db):
    """Ensure _find_completed_ingest_job returns the newest completed ingest."""
    session = test_db()
    try:
        asset_id = "asset-multi-ingest"
        asset = AudioAsset(
            asset_id=asset_id,
            content_hash="hash-xyz",
            source_uri="/tmp/original.wav",
            original_filename="original.wav",
        )
        session.add(asset)

        base_time = datetime(2024, 1, 1, tzinfo=UTC)
        jobs = []
        for idx, offset_minutes in enumerate((0, 2, 5), start=1):
            ts = base_time + timedelta(minutes=offset_minutes)
            jobs.append(
                PipelineJob(
                    job_id=f"ingest-job-{idx}",
                    asset_id=asset_id,
                    stage="ingest",
                    status="completed",
                    attempt=1,
                    created_at=ts,
                    started_at=ts,
                    finished_at=ts,
                )
            )

        session.add_all(jobs)
        session.commit()

        latest = _find_completed_ingest_job(session, asset_id)
        assert latest is not None
        assert latest.job_id == "ingest-job-3"
        assert latest.created_at == base_time + timedelta(minutes=5)
    finally:
        session.close()
