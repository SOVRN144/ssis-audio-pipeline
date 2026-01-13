"""Tests for ingest idempotency behavior.

Per Blueprint section 8:
- If owner_entity_id is provided: idempotency key = (owner_entity_id, content_hash)
  - If AudioAsset exists for that key, return existing asset_id (no duplication)
- If owner_entity_id is null: non-idempotent, always creates new asset
"""

import tempfile
import wave
from pathlib import Path

import pytest
from sqlalchemy import func, select

from app.models import AudioAsset, PipelineJob


@pytest.fixture
def different_audio_file():
    """Create a different audio file (different content hash)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        with wave.open(f.name, "wb") as wf:
            wf.setnchannels(2)  # Different: stereo
            wf.setsampwidth(2)
            wf.setframerate(44100)  # Different: sample rate
            wf.writeframes(b"\xff" * 44100 * 4)  # Different content

        yield Path(f.name)

    try:
        Path(f.name).unlink()
    except OSError:
        pass


class TestIdempotentIngest:
    """Tests for idempotent ingest (with owner_entity_id)."""

    def test_same_owner_same_content_returns_existing(self, client, sample_audio_file):
        """Same owner + same content should return existing asset (idempotent)."""
        test_client, SessionFactory = client
        owner_id = "user-idempotent-test"

        # First ingest
        response1 = test_client.post(
            "/v1/ingest/local",
            json={
                "source_path": str(sample_audio_file),
                "owner_entity_id": owner_id,
            },
        )
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["is_duplicate"] is False

        # Second ingest (same file, same owner)
        response2 = test_client.post(
            "/v1/ingest/local",
            json={
                "source_path": str(sample_audio_file),
                "owner_entity_id": owner_id,
            },
        )
        assert response2.status_code == 200
        data2 = response2.json()

        # Should return same asset_id
        assert data2["asset_id"] == data1["asset_id"]
        assert data2["is_duplicate"] is True
        # Should have different job_id (new job record each time)
        assert data2["job_id"] != data1["job_id"]

        # Verify only one AudioAsset exists
        session = SessionFactory()
        try:
            count_stmt = select(func.count()).select_from(AudioAsset)
            count = session.execute(count_stmt).scalar()
            assert count == 1

            # But two PipelineJob records
            job_count_stmt = select(func.count()).select_from(PipelineJob)
            job_count = session.execute(job_count_stmt).scalar()
            assert job_count == 2
        finally:
            session.close()

    def test_same_owner_different_content_creates_new(
        self, client, sample_audio_file, different_audio_file
    ):
        """Same owner + different content should create new asset."""
        test_client, SessionFactory = client
        owner_id = "user-diff-content"

        # First ingest
        response1 = test_client.post(
            "/v1/ingest/local",
            json={
                "source_path": str(sample_audio_file),
                "owner_entity_id": owner_id,
            },
        )
        assert response1.status_code == 200
        data1 = response1.json()

        # Second ingest (different file, same owner)
        response2 = test_client.post(
            "/v1/ingest/local",
            json={
                "source_path": str(different_audio_file),
                "owner_entity_id": owner_id,
            },
        )
        assert response2.status_code == 200
        data2 = response2.json()

        # Should create new asset
        assert data2["asset_id"] != data1["asset_id"]
        assert data2["is_duplicate"] is False

        # Verify two assets exist
        session = SessionFactory()
        try:
            count_stmt = select(func.count()).select_from(AudioAsset)
            count = session.execute(count_stmt).scalar()
            assert count == 2
        finally:
            session.close()

    def test_different_owner_same_content_creates_new(self, client, sample_audio_file):
        """Different owners with same content should create separate assets."""
        test_client, SessionFactory = client

        # First ingest (owner A)
        response1 = test_client.post(
            "/v1/ingest/local",
            json={
                "source_path": str(sample_audio_file),
                "owner_entity_id": "owner-A",
            },
        )
        assert response1.status_code == 200
        data1 = response1.json()

        # Second ingest (owner B, same file)
        response2 = test_client.post(
            "/v1/ingest/local",
            json={
                "source_path": str(sample_audio_file),
                "owner_entity_id": "owner-B",
            },
        )
        assert response2.status_code == 200
        data2 = response2.json()

        # Different owners = different assets (even with same content)
        assert data2["asset_id"] != data1["asset_id"]
        assert data2["is_duplicate"] is False

        # Verify two assets exist with same content_hash but different owners
        session = SessionFactory()
        try:
            stmt = select(AudioAsset).order_by(AudioAsset.created_at)
            assets = session.execute(stmt).scalars().all()
            assert len(assets) == 2
            assert assets[0].content_hash == assets[1].content_hash
            assert assets[0].owner_entity_id == "owner-A"
            assert assets[1].owner_entity_id == "owner-B"
        finally:
            session.close()


class TestNonIdempotentIngest:
    """Tests for non-idempotent ingest (without owner_entity_id)."""

    def test_null_owner_always_creates_new_asset(self, client, sample_audio_file):
        """Null owner_entity_id should always create new asset (non-idempotent)."""
        test_client, SessionFactory = client

        # First ingest (no owner)
        response1 = test_client.post(
            "/v1/ingest/local",
            json={"source_path": str(sample_audio_file)},
        )
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["is_duplicate"] is False

        # Second ingest (same file, still no owner)
        response2 = test_client.post(
            "/v1/ingest/local",
            json={"source_path": str(sample_audio_file)},
        )
        assert response2.status_code == 200
        data2 = response2.json()

        # Should create new asset each time
        assert data2["asset_id"] != data1["asset_id"]
        assert data2["is_duplicate"] is False

        # Third ingest
        response3 = test_client.post(
            "/v1/ingest/local",
            json={"source_path": str(sample_audio_file)},
        )
        assert response3.status_code == 200
        data3 = response3.json()
        assert data3["asset_id"] != data1["asset_id"]
        assert data3["asset_id"] != data2["asset_id"]

        # Verify three assets exist
        session = SessionFactory()
        try:
            count_stmt = select(func.count()).select_from(AudioAsset)
            count = session.execute(count_stmt).scalar()
            assert count == 3

            # All should have null owner_entity_id
            stmt = select(AudioAsset)
            assets = session.execute(stmt).scalars().all()
            for asset in assets:
                assert asset.owner_entity_id is None
        finally:
            session.close()

    def test_explicit_null_owner_is_non_idempotent(self, client, sample_audio_file):
        """Explicitly passing null owner should be non-idempotent."""
        test_client, SessionFactory = client

        response1 = test_client.post(
            "/v1/ingest/local",
            json={
                "source_path": str(sample_audio_file),
                "owner_entity_id": None,
            },
        )
        assert response1.status_code == 200
        data1 = response1.json()

        response2 = test_client.post(
            "/v1/ingest/local",
            json={
                "source_path": str(sample_audio_file),
                "owner_entity_id": None,
            },
        )
        assert response2.status_code == 200
        data2 = response2.json()

        # Different assets
        assert data2["asset_id"] != data1["asset_id"]


class TestIdempotencyWithUpload:
    """Tests for idempotency with upload endpoint."""

    def test_upload_idempotency_with_owner(self, client, sample_audio_file):
        """Upload endpoint should also respect idempotency."""
        test_client, SessionFactory = client
        owner_id = "upload-user-123"

        # First upload
        with open(sample_audio_file, "rb") as f:
            response1 = test_client.post(
                "/v1/ingest/upload",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"owner_entity_id": owner_id},
            )
        assert response1.status_code == 200
        data1 = response1.json()

        # Second upload (same content, same owner)
        with open(sample_audio_file, "rb") as f:
            response2 = test_client.post(
                "/v1/ingest/upload",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"owner_entity_id": owner_id},
            )
        assert response2.status_code == 200
        data2 = response2.json()

        # Should return same asset
        assert data2["asset_id"] == data1["asset_id"]
        assert data2["is_duplicate"] is True

    def test_upload_non_idempotent_without_owner(self, client, sample_audio_file):
        """Upload without owner should be non-idempotent."""
        test_client, _ = client

        # First upload
        with open(sample_audio_file, "rb") as f:
            response1 = test_client.post(
                "/v1/ingest/upload",
                files={"file": ("test.wav", f, "audio/wav")},
            )
        assert response1.status_code == 200
        data1 = response1.json()

        # Second upload (same content, no owner)
        with open(sample_audio_file, "rb") as f:
            response2 = test_client.post(
                "/v1/ingest/upload",
                files={"file": ("test.wav", f, "audio/wav")},
            )
        assert response2.status_code == 200
        data2 = response2.json()

        # Should create new asset
        assert data2["asset_id"] != data1["asset_id"]
        assert data2["is_duplicate"] is False


class TestDbIdempotencyConstraint:
    """Tests for DB-level idempotency enforcement."""

    def test_unique_constraint_exists(self, client, sample_audio_file):
        """Verify the (owner_entity_id, content_hash) uniqueness constraint."""
        test_client, SessionFactory = client
        owner_id = "constraint-test-user"

        # First ingest
        response = test_client.post(
            "/v1/ingest/local",
            json={
                "source_path": str(sample_audio_file),
                "owner_entity_id": owner_id,
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Get the content_hash
        session = SessionFactory()
        try:
            stmt = select(AudioAsset).where(AudioAsset.asset_id == data["asset_id"])
            _asset = session.execute(stmt).scalar_one()  # Verify asset exists

            # Verify uniqueness constraint name exists in table args
            from app.models import AudioAsset as AssetModel

            constraints = [c.name for c in AssetModel.__table__.constraints if hasattr(c, "name")]
            assert "uq_owner_content_hash" in constraints
        finally:
            session.close()
