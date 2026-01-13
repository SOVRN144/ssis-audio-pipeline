"""SSIS Audio Pipeline - Ingest service logic.

Core ingest business logic implementing:
- Idempotency via (owner_entity_id, content_hash)
- Atomic file persistence
- AudioAsset + PipelineJob DB record creation

NO orchestrator logic, NO workers, NO Huey. Step 2 scope only.
"""

from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AudioAsset, PipelineJob
from app.utils.atomic_io import atomic_copy_file
from app.utils.audio_meta import extract_audio_metadata, guess_format_from_extension
from app.utils.hashing import sha256_file
from app.utils.paths import audio_original_path

if TYPE_CHECKING:
    from typing import BinaryIO


# --- Error Codes (Blueprint section 8) ---


class IngestErrorCode(StrEnum):
    """Error codes for ingest stage per Blueprint section 8."""

    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    HASH_FAILED = "HASH_FAILED"
    INGEST_FAILED = "INGEST_FAILED"


class IngestError(Exception):
    """Base exception for ingest errors."""

    def __init__(self, error_code: str, message: str):
        self.error_code = error_code
        self.message = message
        super().__init__(f"{error_code}: {message}")


class FileNotFoundIngestError(IngestError):
    """Source file not found."""

    def __init__(self, path: str):
        super().__init__(IngestErrorCode.FILE_NOT_FOUND, f"Source file not found: {path}")


class HashFailedError(IngestError):
    """Unable to compute content hash."""

    def __init__(self, path: str, reason: str):
        super().__init__(IngestErrorCode.HASH_FAILED, f"Hash failed for {path}: {reason}")


class IngestFailedError(IngestError):
    """Generic ingest failure."""

    def __init__(self, reason: str):
        super().__init__(IngestErrorCode.INGEST_FAILED, f"Ingest failed: {reason}")


# --- Result Types ---


@dataclass
class IngestResult:
    """Result of a successful ingest operation."""

    asset_id: str
    job_id: str | None
    is_duplicate: bool


# --- Ingest Service ---


def generate_asset_id() -> str:
    """Generate a unique asset ID.

    Uses UUID4 for uniqueness. Format: uuid4 hex (32 chars).
    """
    return uuid.uuid4().hex


def generate_job_id() -> str:
    """Generate a unique job ID.

    Uses UUID4 for uniqueness. Format: uuid4 hex (32 chars).
    """
    return uuid.uuid4().hex


def ingest_local_file(
    session: Session,
    source_path: str | Path,
    owner_entity_id: str | None = None,
    original_filename: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> IngestResult:
    """Ingest a local audio file.

    Implements Blueprint section 8 (Stage A - Ingest):
    1. Compute content hash
    2. Check idempotency (if owner_entity_id provided)
    3. Atomically copy file to canonical location
    4. Insert AudioAsset record
    5. Insert PipelineJob record (stage="ingest", status="completed")

    Args:
        session: Active database session.
        source_path: Path to the local audio file.
        owner_entity_id: Optional owner for idempotency enforcement.
        original_filename: Override filename (defaults to basename of source_path).
        metadata: Optional user-provided metadata (stored as JSON).

    Returns:
        IngestResult with asset_id, job_id, and is_duplicate flag.

    Raises:
        FileNotFoundIngestError: If source file does not exist.
        HashFailedError: If content hash cannot be computed.
        IngestFailedError: If file copy or DB operation fails.

    Note:
        This function commits the session on success (and on duplicate short-circuit).
        Callers should not wrap it in a transaction expecting rollback.
    """
    source_path = Path(source_path)

    # 1. Verify source exists
    if not source_path.exists():
        raise FileNotFoundIngestError(str(source_path))

    # 2. Compute content hash
    try:
        content_hash = sha256_file(source_path)
    except Exception as e:
        raise HashFailedError(str(source_path), str(e)) from e

    # 3. Determine effective filename and extension
    effective_filename = original_filename or source_path.name
    ext = guess_format_from_extension(effective_filename) or "bin"

    # 4. Check idempotency (only if owner_entity_id is provided)
    if owner_entity_id is not None:
        existing = _find_existing_asset(session, owner_entity_id, content_hash)
        if existing is not None:
            # Asset already exists - create job record and return
            job_id = _create_ingest_job(session, existing.asset_id, status="completed")
            session.commit()
            return IngestResult(
                asset_id=existing.asset_id,
                job_id=job_id,
                is_duplicate=True,
            )

    # 5. Generate new asset_id
    asset_id = generate_asset_id()

    # 6. Compute canonical destination path
    dest_path = audio_original_path(asset_id, ext)

    # 7. Atomically copy file to destination
    try:
        atomic_copy_file(source_path, dest_path)
    except FileNotFoundError as e:
        raise IngestFailedError(f"File copy failed (source or destination issue): {e}") from e
    except OSError as e:
        raise IngestFailedError(f"File copy failed: {e}") from e

    # 8. Extract audio metadata (best-effort, never fails)
    audio_meta = extract_audio_metadata(dest_path)

    # 9. Insert AudioAsset record
    asset = AudioAsset(
        asset_id=asset_id,
        content_hash=content_hash,
        source_uri=str(dest_path),
        original_filename=effective_filename,
        owner_entity_id=owner_entity_id,
        duration_sec=audio_meta.duration_sec,
        sample_rate=audio_meta.sample_rate,
        channels=audio_meta.channels,
        format_guess=audio_meta.format_guess,
        user_metadata=metadata,
    )
    session.add(asset)

    # 10. Create PipelineJob record
    job_id = _create_ingest_job(session, asset_id, status="completed")

    # 11. Commit transaction (all or nothing)
    try:
        session.commit()
    except Exception as e:
        session.rollback()
        # Cleanup orphan file on DB failure
        try:
            dest_path.unlink()
        except OSError:
            pass
        raise IngestFailedError(f"Database commit failed: {e}") from e

    # Enqueue orchestrator tick (non-blocking)
    _enqueue_orchestrator_tick_safe(asset_id)

    return IngestResult(
        asset_id=asset_id,
        job_id=job_id,
        is_duplicate=False,
    )


def ingest_upload_stream(
    session: Session,
    stream: BinaryIO,
    filename: str,
    owner_entity_id: str | None = None,
    original_filename: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> IngestResult:
    """Ingest an uploaded file from a stream.

    Similar to ingest_local_file but reads from a stream.
    For uploads, we write to temp location first, compute hash,
    then check idempotency before finalizing.

    Args:
        session: Active database session.
        stream: File-like object with read() method.
        filename: Original filename from upload.
        owner_entity_id: Optional owner for idempotency enforcement.
        original_filename: Override filename (defaults to upload filename).
        metadata: Optional user-provided metadata (stored as JSON).

    Returns:
        IngestResult with asset_id, job_id, and is_duplicate flag.

    Raises:
        HashFailedError: If content hash cannot be computed.
        IngestFailedError: If write or DB operation fails.

    Note:
        This function commits the session on success (and on duplicate short-circuit).
        Callers should not wrap it in a transaction expecting rollback.
    """
    # 1. Determine effective filename and extension
    effective_filename = original_filename or filename
    ext = guess_format_from_extension(effective_filename) or "bin"

    # 2. Write stream to temporary file for hashing
    # (We need to compute hash before knowing if it is a duplicate)
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp_path = Path(tmp.name)
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                tmp.write(chunk)
            tmp.flush()
    except OSError as e:
        raise IngestFailedError(f"Failed to write temp file: {e}") from e

    try:
        # 3. Compute content hash
        try:
            content_hash = sha256_file(tmp_path)
        except Exception as e:
            raise HashFailedError(str(tmp_path), str(e)) from e

        # 4. Check idempotency (only if owner_entity_id is provided)
        if owner_entity_id is not None:
            existing = _find_existing_asset(session, owner_entity_id, content_hash)
            if existing is not None:
                # Asset already exists - create job record and return
                job_id = _create_ingest_job(session, existing.asset_id, status="completed")
                session.commit()
                return IngestResult(
                    asset_id=existing.asset_id,
                    job_id=job_id,
                    is_duplicate=True,
                )

        # 5. Generate new asset_id
        asset_id = generate_asset_id()

        # 6. Compute canonical destination path
        dest_path = audio_original_path(asset_id, ext)

        # 7. Atomically copy temp file to destination
        try:
            atomic_copy_file(tmp_path, dest_path)
        except OSError as e:
            raise IngestFailedError(f"File copy failed: {e}") from e

        # 8. Extract audio metadata (best-effort)
        audio_meta = extract_audio_metadata(dest_path)

        # 9. Insert AudioAsset record
        asset = AudioAsset(
            asset_id=asset_id,
            content_hash=content_hash,
            source_uri=str(dest_path),
            original_filename=effective_filename,
            owner_entity_id=owner_entity_id,
            duration_sec=audio_meta.duration_sec,
            sample_rate=audio_meta.sample_rate,
            channels=audio_meta.channels,
            format_guess=audio_meta.format_guess,
            user_metadata=metadata,
        )
        session.add(asset)

        # 10. Create PipelineJob record
        job_id = _create_ingest_job(session, asset_id, status="completed")

        # 11. Commit transaction
        try:
            session.commit()
        except Exception as e:
            session.rollback()
            # Cleanup orphan file on DB failure
            try:
                dest_path.unlink()
            except OSError:
                pass
            raise IngestFailedError(f"Database commit failed: {e}") from e

        # Enqueue orchestrator tick (non-blocking)
        _enqueue_orchestrator_tick_safe(asset_id)

        return IngestResult(
            asset_id=asset_id,
            job_id=job_id,
            is_duplicate=False,
        )

    finally:
        # Always cleanup temp file
        try:
            tmp_path.unlink()
        except OSError:
            pass


def create_failed_ingest_job(
    session: Session,
    error_code: str,
    error_message: str,
    asset_id: str | None = None,
) -> str | None:
    """Create a failed ingest job record.

    Used when ingest fails before an asset can be created.
    If asset_id is None, returns None (no job persisted).

    Args:
        session: Active database session.
        error_code: Error code from IngestErrorCode.
        error_message: Human-readable error message.
        asset_id: Optional asset ID if one was generated.

    Returns:
        The generated job_id if persisted, None if not persisted.
    """
    # If no asset_id, we cannot create a proper job (FK constraint)
    # Return None to indicate no job was persisted
    if asset_id is None:
        return None

    return _create_ingest_job(
        session,
        asset_id,
        status="failed",
        error_code=error_code,
        error_message=error_message,
    )


# --- Internal Helpers ---


def _find_existing_asset(
    session: Session,
    owner_entity_id: str,
    content_hash: str,
) -> AudioAsset | None:
    """Find existing asset by owner_entity_id and content_hash.

    This is the idempotency check per Blueprint section 8.

    Args:
        session: Active database session.
        owner_entity_id: Owner entity ID.
        content_hash: SHA256 content hash.

    Returns:
        AudioAsset if found, None otherwise.
    """
    stmt = select(AudioAsset).where(
        AudioAsset.owner_entity_id == owner_entity_id,
        AudioAsset.content_hash == content_hash,
    )
    return session.execute(stmt).scalar_one_or_none()


def _create_ingest_job(
    session: Session,
    asset_id: str,
    status: str,
    error_code: str | None = None,
    error_message: str | None = None,
) -> str:
    """Create a PipelineJob record for the ingest stage.

    Args:
        session: Active database session.
        asset_id: Reference to audio asset.
        status: Job status ("completed" or "failed").
        error_code: Error code if failed.
        error_message: Error message if failed.

    Returns:
        The generated job_id.
    """
    job_id = generate_job_id()
    now = datetime.now(UTC)

    job = PipelineJob(
        job_id=job_id,
        asset_id=asset_id,
        stage="ingest",
        status=status,
        attempt=1,
        started_at=now,
        finished_at=now,
        error_code=error_code,
        error_message=error_message,
    )
    session.add(job)
    session.flush()

    return job_id


def _enqueue_orchestrator_tick_safe(asset_id: str) -> None:
    """Enqueue orchestrator tick, silently handling errors.

    This is non-blocking and best-effort. If Huey is not available
    or enqueueing fails, we log the error but do not fail the ingest.

    Args:
        asset_id: The asset ID to enqueue for processing.
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        from app.huey_app import enqueue_orchestrator_tick

        enqueue_orchestrator_tick(asset_id)
        logger.debug("Enqueued orchestrator tick for asset_id=%s", asset_id)
    except Exception:
        # Best-effort: log but do not fail ingest
        logger.warning(
            "Failed to enqueue orchestrator tick for asset_id=%s (non-fatal)",
            asset_id,
            exc_info=True,
        )
