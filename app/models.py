"""SSIS Audio Pipeline - SQLAlchemy ORM models.

Database tables per Blueprint section 7:
1. audio_assets
2. pipeline_jobs
3. stage_locks
4. artifact_index
5. feature_specs
"""

from datetime import UTC, datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


def utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(UTC)


class AudioAsset(Base):
    """Canonical identity and source metadata for an audio asset.

    Corresponds to audio_asset.schema.json.
    """

    __tablename__ = "audio_assets"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Unique asset identifier (used in file paths and references)
    asset_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)

    # Content hash for idempotency (SHA256 hex digest)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # Source file information
    source_uri: Mapped[str] = mapped_column(Text, nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)

    # Optional owner for idempotency enforcement
    owner_entity_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    # Best-effort metadata (nullable as extraction may fail)
    duration_sec: Mapped[float | None] = mapped_column(nullable=True)
    sample_rate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    channels: Mapped[int | None] = mapped_column(Integer, nullable=True)
    format_guess: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    # Idempotency constraint: same owner + same content = same asset
    __table_args__ = (
        UniqueConstraint("owner_entity_id", "content_hash", name="uq_owner_content_hash"),
    )


class PipelineJob(Base):
    """Per-stage job telemetry record.

    Corresponds to pipeline_job.schema.json.
    """

    __tablename__ = "pipeline_jobs"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Unique job identifier
    job_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)

    # Reference to audio asset
    asset_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("audio_assets.asset_id"), nullable=False, index=True
    )

    # Stage information
    stage: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", index=True)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    # Worker tracking
    worker_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Timing
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Error tracking (per Blueprint error taxonomy)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Stage metrics as JSON string (parsed by application)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Feature spec alias if applicable
    feature_spec_alias: Mapped[str | None] = mapped_column(String(12), nullable=True)

    __table_args__ = (Index("ix_jobs_asset_stage", "asset_id", "stage"),)


class StageLock(Base):
    """Stage lock for preventing duplicate work.

    Lock key: (asset_id, stage, feature_spec_alias|null)
    Per Blueprint section 7.

    TTL reclamation: locks with expires_at < now can be reclaimed by orchestrator.
    """

    __tablename__ = "stage_locks"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Lock key components
    asset_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    feature_spec_alias: Mapped[str | None] = mapped_column(String(12), nullable=True)

    # Lock holder
    worker_id: Mapped[str] = mapped_column(String(64), nullable=False)

    # Timestamps for TTL calculation
    acquired_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    # Explicit expiry timestamp (acquired_at + TTL). Locks with expires_at < now
    # are eligible for reclamation. Set at lock creation time.
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Unique constraint on lock key (with nullable feature_spec_alias)
    __table_args__ = (
        UniqueConstraint("asset_id", "stage", "feature_spec_alias", name="uq_stage_lock_key"),
        Index("ix_locks_acquired_at", "acquired_at"),
        Index("ix_locks_expires_at", "expires_at"),
    )


class FeatureSpec(Base):
    """Feature specification registry.

    Maps short aliases to full feature spec IDs.
    Per Blueprint section 5.
    """

    __tablename__ = "feature_specs"

    # Alias is the primary key (first 12 chars of sha256(feature_spec_id))
    alias: Mapped[str] = mapped_column(String(12), primary_key=True)

    # Full human-readable feature spec ID
    feature_spec_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    # Optional notes
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class ArtifactIndex(Base):
    """Index of artifacts produced for each asset.

    Tracks which artifacts exist per asset for orchestrator planning.
    Per Blueprint section 7 (recommended).
    """

    __tablename__ = "artifact_index"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Reference to audio asset
    asset_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("audio_assets.asset_id"), nullable=False, index=True
    )

    # Artifact type (e.g., "normalized_wav", "feature_pack", "segments", "preview")
    artifact_type: Mapped[str] = mapped_column(String(32), nullable=False)

    # Path to the artifact file
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)

    # Optional feature spec alias (for feature_pack artifacts)
    feature_spec_alias: Mapped[str | None] = mapped_column(String(12), nullable=True)

    # Schema version of the artifact
    schema_version: Mapped[str] = mapped_column(String(16), nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    # Content hash of the artifact (for integrity verification)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    __table_args__ = (
        UniqueConstraint("asset_id", "artifact_type", "feature_spec_alias", name="uq_artifact_key"),
        Index("ix_artifact_type", "artifact_type"),
    )
