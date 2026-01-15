"""SSIS Audio Pipeline - Ingest API FastAPI application.

FastAPI service for audio file ingestion.
Step 2 business logic: local + upload ingest endpoints with persistence and idempotency.

On successful ingest, a best-effort non-blocking enqueue of the orchestrator tick
is triggered (Step 3). The orchestrator handles stage planning and dispatch; this
module does NOT contain orchestrator logic or DSP processing.

Run with:
    uvicorn services.ingest_api.main:app --reload  # dev server only
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.db import init_db
from app.schemas import IngestErrorResponse, IngestLocalRequest, IngestSuccessResponse
from services.ingest_api.service import (
    FileNotFoundIngestError,
    HashFailedError,
    IngestError,
    IngestErrorCode,
    IngestFailedError,
    ingest_local_file,
    ingest_upload_stream,
)

logger = logging.getLogger(__name__)

# --- Database Setup ---

# Module-level session factory (initialized on startup)
_session_factory = None


def get_session_factory():
    """Get the session factory.

    Raises:
        RuntimeError: If session factory not initialized (app lifespan not invoked).
    """
    global _session_factory
    if _session_factory is None:
        raise RuntimeError("Session factory not initialized. App lifespan not invoked?")
    return _session_factory


def get_db_session():
    """Dependency that provides a database session."""
    SessionFactory = get_session_factory()
    session = SessionFactory()
    try:
        yield session
    finally:
        session.close()


# --- Lifespan ---


def _cleanup_orphan_temp_files_safe() -> None:
    """Clean up orphan temp files on startup (best-effort).

    Per resilience plan: wire into startup, never crash on errors.
    Targets canonical audio directories under data/audio/.
    """
    from app.config import AUDIO_DIR
    from app.utils.atomic_io import cleanup_orphan_temp_files

    try:
        # Recursively clean up temp files under audio directory
        if AUDIO_DIR.exists():
            total_cleaned = 0
            # Clean top-level
            total_cleaned += cleanup_orphan_temp_files(AUDIO_DIR)
            # Clean asset subdirectories
            for subdir in AUDIO_DIR.iterdir():
                if subdir.is_dir():
                    total_cleaned += cleanup_orphan_temp_files(subdir)
            if total_cleaned > 0:
                logger.info("Startup cleanup: removed %d orphan temp files", total_cleaned)
    except Exception:
        # Best-effort: never crash startup
        logger.warning("Startup cleanup failed (non-fatal)", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler.

    Initializes database on startup and cleans up orphan temp files.
    """
    # Startup: initialize database
    global _session_factory
    _, _session_factory = init_db()

    # Startup: clean up orphan temp files (best-effort, never fails)
    _cleanup_orphan_temp_files_safe()

    yield
    # Shutdown: nothing special needed


# --- FastAPI App ---


app = FastAPI(
    title="SSIS Audio Pipeline - Ingest API",
    description="Audio file ingestion service (local + upload). Step 2.",
    version="0.1.0",
    lifespan=lifespan,
)


# --- Error Handling ---


def error_code_to_status(error_code: str) -> int:
    """Map error codes to HTTP status codes.

    Per Blueprint section 8:
    - FILE_NOT_FOUND -> 404
    - HASH_FAILED -> 500
    - INGEST_FAILED -> 500
    """
    if error_code == IngestErrorCode.FILE_NOT_FOUND:
        return 404
    return 500


def make_error_response(error_code: str, error_message: str) -> JSONResponse:
    """Create a JSON error response."""
    return JSONResponse(
        status_code=error_code_to_status(error_code),
        content=IngestErrorResponse(
            error_code=error_code,
            error_message=error_message,
        ).model_dump(),
    )


# --- Endpoints ---


@app.post(
    "/v1/ingest/local",
    response_model=IngestSuccessResponse,
    responses={
        404: {"model": IngestErrorResponse, "description": "Source file not found"},
        500: {"model": IngestErrorResponse, "description": "Ingest failed"},
    },
    summary="Ingest a local audio file",
    description="Ingest an audio file from a local filesystem path.",
)
def ingest_local(
    request: IngestLocalRequest,
    session: Annotated[Session, Depends(get_db_session)],
):
    """Ingest a local audio file.

    Idempotency rule:
    - If owner_entity_id is provided and (owner_entity_id, content_hash) exists,
      returns the existing asset_id (no duplication).
    - If owner_entity_id is null, always creates a new asset.
    """
    try:
        result = ingest_local_file(
            session=session,
            source_path=request.source_path,
            owner_entity_id=request.owner_entity_id,
            original_filename=request.original_filename,
            metadata=request.metadata,
        )
        return IngestSuccessResponse(
            asset_id=result.asset_id,
            job_id=result.job_id,
            is_duplicate=result.is_duplicate,
        )
    except FileNotFoundIngestError as e:
        return make_error_response(e.error_code, e.message)
    except HashFailedError as e:
        return make_error_response(e.error_code, e.message)
    except IngestFailedError as e:
        return make_error_response(e.error_code, e.message)
    except IngestError as e:
        return make_error_response(e.error_code, e.message)
    except Exception:
        # Log full exception server-side, return generic message to client
        logger.exception("Unexpected error during local ingest")
        return make_error_response(
            IngestErrorCode.INGEST_FAILED,
            "An unexpected error occurred during ingest",
        )


@app.post(
    "/v1/ingest/upload",
    response_model=IngestSuccessResponse,
    responses={
        500: {"model": IngestErrorResponse, "description": "Ingest failed"},
    },
    summary="Ingest an uploaded audio file",
    description="Ingest an audio file via multipart upload.",
)
async def ingest_upload(
    session: Annotated[Session, Depends(get_db_session)],
    file: Annotated[UploadFile, File(description="Audio file to ingest")],
    owner_entity_id: Annotated[
        str | None, Form(description="Optional owner entity ID for idempotency")
    ] = None,
    original_filename: Annotated[str | None, Form(description="Override filename")] = None,
    metadata: Annotated[str | None, Form(description="Optional JSON metadata object")] = None,
):
    """Ingest an uploaded audio file.

    Accepts multipart form data with:
    - file: The audio file (required)
    - owner_entity_id: Optional owner for idempotency
    - original_filename: Override filename
    - metadata: Optional JSON metadata string

    Idempotency rule same as /v1/ingest/local.
    """
    # Parse metadata if provided
    parsed_metadata = None
    if metadata is not None:
        try:
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return make_error_response(
                IngestErrorCode.INGEST_FAILED,
                "Invalid metadata JSON",
            )

    # Determine filename
    upload_filename = file.filename or "unknown"

    try:
        result = ingest_upload_stream(
            session=session,
            stream=file.file,
            filename=upload_filename,
            owner_entity_id=owner_entity_id,
            original_filename=original_filename,
            metadata=parsed_metadata,
        )
        return IngestSuccessResponse(
            asset_id=result.asset_id,
            job_id=result.job_id,
            is_duplicate=result.is_duplicate,
        )
    except HashFailedError as e:
        return make_error_response(e.error_code, e.message)
    except IngestFailedError as e:
        return make_error_response(e.error_code, e.message)
    except IngestError as e:
        return make_error_response(e.error_code, e.message)
    except Exception:
        # Log full exception server-side, return generic message to client
        logger.exception("Unexpected error during upload ingest")
        return make_error_response(
            IngestErrorCode.INGEST_FAILED,
            "An unexpected error occurred during ingest",
        )


@app.get("/health", summary="Health check")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}


# --- For testing: allow overriding session factory ---


def override_session_factory(factory):
    """Override the session factory for testing."""
    global _session_factory
    _session_factory = factory
