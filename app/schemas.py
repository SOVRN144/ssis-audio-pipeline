"""SSIS Audio Pipeline - Pydantic models for API validation.

Pydantic models for request/response validation corresponding to
JSON schemas in /specs. Used by FastAPI for runtime validation.
"""

from datetime import datetime  # noqa: I001
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# --- Request Models ---


class IngestLocalRequest(BaseModel):
    """Request payload for local file ingest.

    Corresponds to specs/ingest_local_request.schema.json.
    """

    model_config = ConfigDict(extra="forbid")

    source_path: str = Field(
        ...,
        min_length=1,
        description="Absolute path to the local audio file to ingest",
    )
    owner_entity_id: str | None = Field(
        default=None,
        min_length=1,
        description="Optional owner entity ID for idempotency enforcement",
    )
    original_filename: str | None = Field(
        default=None,
        min_length=1,
        description="Override filename (defaults to basename of source_path)",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional additional metadata to store with the asset",
    )


class IngestUploadRequest(BaseModel):
    """Request payload for upload ingest (form fields only, file is separate).

    The file itself is handled via UploadFile in FastAPI.
    """

    model_config = ConfigDict(extra="forbid")

    owner_entity_id: str | None = Field(
        default=None,
        min_length=1,
        description="Optional owner entity ID for idempotency enforcement",
    )
    original_filename: str | None = Field(
        default=None,
        min_length=1,
        description="Override filename (defaults to uploaded filename)",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional additional metadata to store with the asset",
    )


# --- Response Models ---


class IngestSuccessResponse(BaseModel):
    """Response for successful ingest operations."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(default="success", description="Operation status")
    asset_id: str = Field(..., description="Unique identifier for the ingested asset")
    job_id: str = Field(..., description="Unique identifier for the ingest job")
    is_duplicate: bool = Field(
        default=False,
        description="True if asset already existed (idempotent ingest)",
    )


class IngestErrorResponse(BaseModel):
    """Response for failed ingest operations."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(default="error", description="Operation status")
    error_code: str = Field(..., description="Error taxonomy code from Blueprint section 8")
    error_message: str = Field(..., description="Human-readable error description")


# --- Internal Data Transfer Models ---


class AudioMetadata(BaseModel):
    """Audio metadata extracted from file (best-effort).

    All fields are optional as extraction may fail.
    """

    model_config = ConfigDict(extra="forbid")

    duration_sec: float | None = Field(default=None, ge=0, description="Duration in seconds")
    sample_rate: int | None = Field(default=None, ge=1, description="Sample rate in Hz")
    channels: int | None = Field(default=None, ge=1, description="Channel count")
    format_guess: str | None = Field(default=None, description="Guessed audio format")


class AudioAssetResponse(BaseModel):
    """Audio asset response model for API serialization.

    Corresponds to specs/audio_asset.schema.json.
    """

    model_config = ConfigDict(extra="forbid")

    schema_id: str = Field(default="audio_asset.v1", description="Schema identifier")
    version: str = Field(default="1.0.0", description="Schema version")
    asset_id: str = Field(..., description="Unique asset identifier")
    computed_at: datetime = Field(..., description="When this record was created")
    content_hash: str = Field(..., description="SHA256 hex digest of original content")
    source_uri: str = Field(..., description="Path to original in SSIS storage")
    original_filename: str = Field(..., description="Original filename")
    owner_entity_id: str | None = Field(default=None, description="Owner entity ID")
    duration_sec: float | None = Field(default=None, description="Duration in seconds")
    sample_rate: int | None = Field(default=None, description="Sample rate in Hz")
    channels: int | None = Field(default=None, description="Channel count")
    format_guess: str | None = Field(default=None, description="Audio format guess")


class PipelineJobResponse(BaseModel):
    """Pipeline job response model for API serialization.

    Corresponds to specs/pipeline_job.schema.json.
    """

    model_config = ConfigDict(extra="forbid")

    schema_id: str = Field(default="pipeline_job.v1", description="Schema identifier")
    version: str = Field(default="1.0.0", description="Schema version")
    job_id: str = Field(..., description="Unique job identifier")
    asset_id: str = Field(..., description="Reference to audio asset")
    computed_at: datetime = Field(..., description="When this job record was created")
    stage: str = Field(..., description="Pipeline stage name")
    status: str = Field(..., description="Current job status")
    attempt: int = Field(default=1, ge=1, description="Current attempt number")
    worker_id: str | None = Field(default=None, description="Worker processing this job")
    started_at: datetime | None = Field(default=None, description="When execution started")
    finished_at: datetime | None = Field(default=None, description="When execution finished")
    error_code: str | None = Field(default=None, description="Error code if failed")
    error_message: str | None = Field(default=None, description="Error message if failed")
    feature_spec_alias: str | None = Field(default=None, description="FeatureSpec alias")


__all__ = [
    "IngestLocalRequest",
    "IngestUploadRequest",
    "IngestSuccessResponse",
    "IngestErrorResponse",
    "AudioMetadata",
    "AudioAssetResponse",
    "PipelineJobResponse",
]
