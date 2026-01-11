"""Tests for the Ingest API endpoints."""

import tempfile
from pathlib import Path

import pytest
from sqlalchemy import select

from app.models import AudioAsset, PipelineJob


@pytest.fixture
def sample_mp3_file():
    """Create a sample non-WAV file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        # Write some arbitrary binary data (not a real MP3)
        f.write(b"fake mp3 content " * 100)
        yield Path(f.name)

    try:
        Path(f.name).unlink()
    except OSError:
        pass


class TestHealthCheck:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Health check should return ok."""
        test_client, _ = client
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestIngestLocal:
    """Tests for POST /v1/ingest/local endpoint."""

    def test_ingest_local_success(self, client, sample_audio_file):
        """Should successfully ingest a local file."""
        test_client, SessionFactory = client

        response = test_client.post(
            "/v1/ingest/local",
            json={"source_path": str(sample_audio_file)},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "asset_id" in data
        assert "job_id" in data
        assert data["is_duplicate"] is False

        # Verify DB records
        session = SessionFactory()
        try:
            # Check AudioAsset
            stmt = select(AudioAsset).where(AudioAsset.asset_id == data["asset_id"])
            asset = session.execute(stmt).scalar_one()
            assert asset is not None
            assert asset.original_filename == sample_audio_file.name
            assert asset.format_guess == "wav"
            assert asset.sample_rate == 22050
            assert asset.channels == 1

            # Check PipelineJob
            job_stmt = select(PipelineJob).where(PipelineJob.job_id == data["job_id"])
            job = session.execute(job_stmt).scalar_one()
            assert job is not None
            assert job.stage == "ingest"
            assert job.status == "completed"
            assert job.asset_id == data["asset_id"]
            assert job.error_code is None
        finally:
            session.close()

    def test_ingest_local_with_owner_entity_id(self, client, sample_audio_file):
        """Should store owner_entity_id when provided."""
        test_client, SessionFactory = client

        response = test_client.post(
            "/v1/ingest/local",
            json={
                "source_path": str(sample_audio_file),
                "owner_entity_id": "user-123",
            },
        )

        assert response.status_code == 200
        data = response.json()

        session = SessionFactory()
        try:
            stmt = select(AudioAsset).where(AudioAsset.asset_id == data["asset_id"])
            asset = session.execute(stmt).scalar_one()
            assert asset.owner_entity_id == "user-123"
        finally:
            session.close()

    def test_ingest_local_with_original_filename(self, client, sample_audio_file):
        """Should use original_filename override when provided."""
        test_client, SessionFactory = client

        response = test_client.post(
            "/v1/ingest/local",
            json={
                "source_path": str(sample_audio_file),
                "original_filename": "my-custom-name.wav",
            },
        )

        assert response.status_code == 200
        data = response.json()

        session = SessionFactory()
        try:
            stmt = select(AudioAsset).where(AudioAsset.asset_id == data["asset_id"])
            asset = session.execute(stmt).scalar_one()
            assert asset.original_filename == "my-custom-name.wav"
        finally:
            session.close()

    def test_ingest_local_file_not_found(self, client):
        """Should return 404 for non-existent file."""
        test_client, _ = client

        response = test_client.post(
            "/v1/ingest/local",
            json={"source_path": "/nonexistent/path/file.wav"},
        )

        assert response.status_code == 404
        data = response.json()
        assert data["status"] == "error"
        assert data["error_code"] == "FILE_NOT_FOUND"
        assert "error_message" in data

    def test_ingest_local_non_wav_file(self, client, sample_mp3_file):
        """Should handle non-WAV files (best-effort metadata)."""
        test_client, SessionFactory = client

        response = test_client.post(
            "/v1/ingest/local",
            json={"source_path": str(sample_mp3_file)},
        )

        assert response.status_code == 200
        data = response.json()

        session = SessionFactory()
        try:
            stmt = select(AudioAsset).where(AudioAsset.asset_id == data["asset_id"])
            asset = session.execute(stmt).scalar_one()
            assert asset.format_guess == "mp3"
            # Non-WAV files have null metadata
            assert asset.sample_rate is None
            assert asset.channels is None
        finally:
            session.close()


class TestIngestUpload:
    """Tests for POST /v1/ingest/upload endpoint."""

    def test_ingest_upload_success(self, client, sample_audio_file):
        """Should successfully ingest an uploaded file."""
        test_client, SessionFactory = client

        with open(sample_audio_file, "rb") as f:
            response = test_client.post(
                "/v1/ingest/upload",
                files={"file": ("test.wav", f, "audio/wav")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "asset_id" in data
        assert "job_id" in data
        assert data["is_duplicate"] is False

        # Verify DB records
        session = SessionFactory()
        try:
            stmt = select(AudioAsset).where(AudioAsset.asset_id == data["asset_id"])
            asset = session.execute(stmt).scalar_one()
            assert asset is not None
            assert asset.original_filename == "test.wav"

            job_stmt = select(PipelineJob).where(PipelineJob.job_id == data["job_id"])
            job = session.execute(job_stmt).scalar_one()
            assert job.stage == "ingest"
            assert job.status == "completed"
        finally:
            session.close()

    def test_ingest_upload_with_owner(self, client, sample_audio_file):
        """Should store owner_entity_id for uploaded file."""
        test_client, SessionFactory = client

        with open(sample_audio_file, "rb") as f:
            response = test_client.post(
                "/v1/ingest/upload",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"owner_entity_id": "user-456"},
            )

        assert response.status_code == 200
        data = response.json()

        session = SessionFactory()
        try:
            stmt = select(AudioAsset).where(AudioAsset.asset_id == data["asset_id"])
            asset = session.execute(stmt).scalar_one()
            assert asset.owner_entity_id == "user-456"
        finally:
            session.close()

    def test_ingest_upload_with_filename_override(self, client, sample_audio_file):
        """Should use original_filename override for uploads."""
        test_client, SessionFactory = client

        with open(sample_audio_file, "rb") as f:
            response = test_client.post(
                "/v1/ingest/upload",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"original_filename": "custom-upload.wav"},
            )

        assert response.status_code == 200
        data = response.json()

        session = SessionFactory()
        try:
            stmt = select(AudioAsset).where(AudioAsset.asset_id == data["asset_id"])
            asset = session.execute(stmt).scalar_one()
            assert asset.original_filename == "custom-upload.wav"
        finally:
            session.close()

    def test_ingest_upload_invalid_metadata_json(self, client, sample_audio_file):
        """Should reject invalid JSON in metadata field."""
        test_client, _ = client

        with open(sample_audio_file, "rb") as f:
            response = test_client.post(
                "/v1/ingest/upload",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"metadata": "not valid json"},
            )

        assert response.status_code == 500
        data = response.json()
        assert data["error_code"] == "INGEST_FAILED"
        assert "Invalid metadata JSON" in data["error_message"]


class TestRequestValidation:
    """Tests for request validation."""

    def test_missing_source_path(self, client):
        """Should require source_path."""
        test_client, _ = client

        response = test_client.post(
            "/v1/ingest/local",
            json={},
        )

        assert response.status_code == 422  # Validation error

    def test_empty_source_path(self, client):
        """Should reject empty source_path."""
        test_client, _ = client

        response = test_client.post(
            "/v1/ingest/local",
            json={"source_path": ""},
        )

        assert response.status_code == 422  # Validation error

    def test_extra_fields_rejected(self, client, sample_audio_file):
        """Should reject extra fields (extra='forbid')."""
        test_client, _ = client

        response = test_client.post(
            "/v1/ingest/local",
            json={
                "source_path": str(sample_audio_file),
                "unknown_field": "value",
            },
        )

        assert response.status_code == 422  # Validation error
