"""Tests for app.db module and FeatureSpec immutability."""

import sqlite3
import tempfile
from datetime import timedelta
from pathlib import Path
from unittest import mock

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from app.config import STAGE_LOCK_TTL_SECONDS
from app.db import (
    FeatureSpecAliasCollision,
    create_stage_lock,
    init_db,
    register_feature_spec,
)
from app.models import FeatureSpec, StageLock
from app.utils.hashing import feature_spec_alias


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        engine, SessionFactory = init_db(db_path)
        yield db_path, engine, SessionFactory
        engine.dispose()


class TestInitDb:
    """Tests for database initialization."""

    def test_creates_database_file(self, temp_db):
        """init_db should create the database file."""
        db_path, engine, _ = temp_db
        assert db_path.exists()

    def test_creates_all_tables(self, temp_db):
        """init_db should create all defined tables."""
        _, engine, _ = temp_db

        from sqlalchemy import inspect

        inspector = inspect(engine)
        tables = inspector.get_table_names()

        assert "audio_assets" in tables
        assert "pipeline_jobs" in tables
        assert "stage_locks" in tables
        assert "feature_specs" in tables
        assert "artifact_index" in tables

    def test_idempotent(self, temp_db):
        """init_db should be safe to call multiple times."""
        db_path, engine, _ = temp_db

        # Call init_db again on same database
        engine2, _ = init_db(db_path)

        # Should not raise, tables should still exist
        from sqlalchemy import inspect

        inspector = inspect(engine2)
        tables = inspector.get_table_names()
        assert "audio_assets" in tables
        engine2.dispose()

    def test_stage_lock_scope_key_migration_dedupes_legacy_rows(self):
        """Legacy stage_locks rows should be backfilled/deduped with lock_scope_key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "legacy.db"

            # Simulate legacy SQLite table (no lock_scope_key, nullable alias uniqueness gap).
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                CREATE TABLE stage_locks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset_id VARCHAR(64) NOT NULL,
                    stage VARCHAR(32) NOT NULL,
                    feature_spec_alias VARCHAR(12),
                    worker_id VARCHAR(64) NOT NULL,
                    acquired_at DATETIME NOT NULL,
                    expires_at DATETIME NOT NULL,
                    CONSTRAINT uq_stage_lock_key UNIQUE (asset_id, stage, feature_spec_alias)
                )
                """
            )
            now = "2026-02-08T00:00:00+00:00"
            conn.execute(
                """
                INSERT INTO stage_locks
                (asset_id, stage, feature_spec_alias, worker_id, acquired_at, expires_at)
                VALUES
                ('asset-1', 'decode', NULL, 'worker-old-1', ?, ?),
                ('asset-1', 'decode', NULL, 'worker-old-2', ?, ?),
                ('asset-1', 'features', 'abc123def456', 'worker-feat', ?, ?)
                """,
                (now, now, now, now, now, now),
            )
            conn.commit()
            conn.close()

            engine, SessionFactory = init_db(db_path)
            session = SessionFactory()
            try:
                rows = session.execute(
                    select(StageLock).where(StageLock.asset_id == "asset-1").order_by(StageLock.id)
                ).scalars()
                rows = list(rows)

                # Duplicate NULL-alias rows should be collapsed to one, alias row should remain.
                assert len(rows) == 2
                assert sum(1 for row in rows if row.stage == "decode") == 1
                assert sum(1 for row in rows if row.stage == "features") == 1

                decode_row = next(row for row in rows if row.stage == "decode")
                features_row = next(row for row in rows if row.stage == "features")
                assert decode_row.lock_scope_key == "__none__"
                assert features_row.lock_scope_key == "abc123def456"

                indexes = session.execute(text("PRAGMA index_list(stage_locks)")).all()
                assert any(idx[1] == "uq_stage_lock_scope_key" and idx[2] == 1 for idx in indexes)
            finally:
                session.close()
                engine.dispose()


class TestRegisterFeatureSpec:
    """Tests for FeatureSpec immutability primitive."""

    def test_insert_new_spec(self, temp_db):
        """Should insert new feature spec when alias does not exist."""
        _, engine, SessionFactory = temp_db
        session = SessionFactory()

        try:
            spec_id = "test_spec_v1"
            expected_alias = feature_spec_alias(spec_id)

            alias = register_feature_spec(session, spec_id, notes="test notes")
            session.commit()

            assert alias == expected_alias
            assert len(alias) == 12

            # Verify in database (SQLAlchemy 2.0 style)
            stmt = select(FeatureSpec).where(FeatureSpec.alias == alias)
            spec = session.execute(stmt).scalar_one_or_none()
            assert spec is not None
            assert spec.feature_spec_id == spec_id
            assert spec.notes == "test notes"
        finally:
            session.close()

    def test_noop_same_spec(self, temp_db):
        """Should be no-op when registering same spec_id again."""
        _, engine, SessionFactory = temp_db
        session = SessionFactory()

        try:
            spec_id = "test_spec_v1"

            # Register first time
            alias1 = register_feature_spec(session, spec_id)
            session.commit()

            # Get created_at from first registration (SQLAlchemy 2.0 style)
            stmt = select(FeatureSpec).where(FeatureSpec.alias == alias1)
            spec1 = session.execute(stmt).scalar_one()
            created_at_1 = spec1.created_at

            # Register same spec again
            alias2 = register_feature_spec(session, spec_id)
            session.commit()

            # Should return same alias
            assert alias1 == alias2

            # Should not create duplicate (SQLAlchemy 2.0 style)
            from sqlalchemy import func

            count_stmt = (
                select(func.count()).select_from(FeatureSpec).where(FeatureSpec.alias == alias1)
            )
            count = session.execute(count_stmt).scalar()
            assert count == 1

            # created_at should be unchanged (no update)
            spec2 = session.execute(stmt).scalar_one()
            assert spec2.created_at == created_at_1
        finally:
            session.close()

    def test_collision_different_spec_via_mock(self, temp_db):
        """Should raise FeatureSpecAliasCollision when alias maps to different spec.

        Uses mock.patch to force a hash collision by making the second spec_id
        resolve to the same alias as the first.
        """
        _, engine, SessionFactory = temp_db
        session = SessionFactory()

        try:
            spec_id_1 = "original_spec_id"
            spec_id_2 = "different_spec_id"

            # Register first spec normally
            alias1 = register_feature_spec(session, spec_id_1)
            session.commit()

            # Patch feature_spec_alias at the source module to return the same alias
            # This simulates a hash collision
            with mock.patch("app.utils.hashing.feature_spec_alias", return_value=alias1):
                with pytest.raises(FeatureSpecAliasCollision) as exc_info:
                    register_feature_spec(session, spec_id_2)

            # Verify exception contains correct information
            exc = exc_info.value
            assert exc.alias == alias1
            assert exc.existing_spec_id == spec_id_1
            assert exc.new_spec_id == spec_id_2
            assert "FEATURE_SPEC_ALIAS_COLLISION" in str(exc)
        finally:
            session.close()

    def test_multiple_different_specs(self, temp_db):
        """Should allow registering multiple different feature specs."""
        _, engine, SessionFactory = temp_db
        session = SessionFactory()

        try:
            spec_ids = [
                "mel64_h10ms_w25ms_sr22050__yamnet1024_h0.5s_onnx",
                "mel128_h10ms_w25ms_sr44100__yamnet1024_h0.5s_onnx",
                "mel64_h5ms_w25ms_sr22050__clap_h0.5s_onnx",
            ]

            aliases = []
            for spec_id in spec_ids:
                alias = register_feature_spec(session, spec_id)
                session.commit()
                aliases.append(alias)

            # All aliases should be unique
            assert len(set(aliases)) == len(aliases)

            # All should be in database (SQLAlchemy 2.0 style)
            from sqlalchemy import func

            count_stmt = select(func.count()).select_from(FeatureSpec)
            count = session.execute(count_stmt).scalar()
            assert count == len(spec_ids)
        finally:
            session.close()


class TestFeatureSpecAliasCollisionException:
    """Tests for FeatureSpecAliasCollision exception."""

    def test_exception_attributes(self):
        """Exception should store all relevant information."""
        exc = FeatureSpecAliasCollision("abc123", "old_spec", "new_spec")

        assert exc.alias == "abc123"
        assert exc.existing_spec_id == "old_spec"
        assert exc.new_spec_id == "new_spec"

    def test_exception_message(self):
        """Exception message should include error code and details."""
        exc = FeatureSpecAliasCollision("abc123", "old_spec", "new_spec")
        msg = str(exc)

        assert "FEATURE_SPEC_ALIAS_COLLISION" in msg
        assert "abc123" in msg
        assert "old_spec" in msg
        assert "new_spec" in msg


class TestCreateStageLock:
    """Tests for StageLock creation primitive."""

    def test_creates_lock_with_expires_at(self, temp_db):
        """create_stage_lock should set expires_at based on TTL."""
        _, engine, SessionFactory = temp_db
        session = SessionFactory()

        try:
            lock = create_stage_lock(
                session,
                asset_id="test-asset-001",
                stage="normalize",
                worker_id="worker-001",
            )
            session.commit()

            # Verify lock was created
            assert lock.id is not None
            assert lock.asset_id == "test-asset-001"
            assert lock.stage == "normalize"
            assert lock.worker_id == "worker-001"
            assert lock.feature_spec_alias is None

            # Verify timestamps
            assert lock.acquired_at is not None
            assert lock.expires_at is not None

            # expires_at should be >= acquired_at + TTL (with small tolerance for timing)
            expected_expires = lock.acquired_at + timedelta(seconds=STAGE_LOCK_TTL_SECONDS)
            # Allow 1 second tolerance for test execution timing
            assert abs((lock.expires_at - expected_expires).total_seconds()) < 1
        finally:
            session.close()

    def test_creates_lock_with_feature_spec_alias(self, temp_db):
        """create_stage_lock should support feature_spec_alias."""
        _, engine, SessionFactory = temp_db
        session = SessionFactory()

        try:
            lock = create_stage_lock(
                session,
                asset_id="test-asset-002",
                stage="features",
                worker_id="worker-002",
                feature_spec_alias="abc123def456",
            )
            session.commit()

            assert lock.feature_spec_alias == "abc123def456"
            assert lock.expires_at > lock.acquired_at
        finally:
            session.close()

    def test_creates_lock_with_custom_ttl(self, temp_db):
        """create_stage_lock should respect custom TTL."""
        _, engine, SessionFactory = temp_db
        session = SessionFactory()

        try:
            custom_ttl = 120  # 2 minutes
            lock = create_stage_lock(
                session,
                asset_id="test-asset-003",
                stage="segment",
                worker_id="worker-003",
                ttl_seconds=custom_ttl,
            )
            session.commit()

            expected_expires = lock.acquired_at + timedelta(seconds=custom_ttl)
            # Allow 1 second tolerance
            assert abs((lock.expires_at - expected_expires).total_seconds()) < 1
        finally:
            session.close()

    def test_lock_persists_in_database(self, temp_db):
        """StageLock should be retrievable from database after commit."""
        _, engine, SessionFactory = temp_db
        session = SessionFactory()

        try:
            lock = create_stage_lock(
                session,
                asset_id="test-asset-004",
                stage="preview",
                worker_id="worker-004",
            )
            session.commit()
            lock_id = lock.id

            # Query back from database
            stmt = select(StageLock).where(StageLock.id == lock_id)
            retrieved = session.execute(stmt).scalar_one()

            assert retrieved.asset_id == "test-asset-004"
            assert retrieved.stage == "preview"
            assert retrieved.worker_id == "worker-004"
            assert retrieved.expires_at is not None
            assert retrieved.expires_at > retrieved.acquired_at
        finally:
            session.close()

    def test_null_alias_lock_uniqueness_enforced(self, temp_db):
        """Second (asset_id, stage, NULL) lock insert must fail under lock_scope_key index."""
        _, _, SessionFactory = temp_db
        session = SessionFactory()

        try:
            create_stage_lock(
                session,
                asset_id="asset-lock-null-1",
                stage="decode",
                worker_id="worker-1",
                feature_spec_alias=None,
            )
            session.commit()

            with pytest.raises(IntegrityError):
                create_stage_lock(
                    session,
                    asset_id="asset-lock-null-1",
                    stage="decode",
                    worker_id="worker-2",
                    feature_spec_alias=None,
                )

            session.rollback()

            rows = session.execute(
                select(StageLock).where(
                    StageLock.asset_id == "asset-lock-null-1",
                    StageLock.stage == "decode",
                )
            ).scalars()
            rows = list(rows)
            assert len(rows) == 1
            assert rows[0].lock_scope_key == "__none__"
        finally:
            session.close()
