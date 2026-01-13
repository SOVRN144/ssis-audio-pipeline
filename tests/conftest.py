"""Shared pytest fixtures for SSIS Audio Pipeline tests.

This module contains common fixtures used across multiple test files,
reducing duplication and improving test maintainability.
"""

import tempfile
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.db import init_db
from services.ingest_api.main import app, get_db_session, override_session_factory


@pytest.fixture
def temp_db():
    """Create a temporary database for testing.

    Creates an isolated SQLite database in a temporary directory.
    The database is cleaned up after the test completes.

    Yields:
        tuple: (db_path, engine, SessionFactory)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        engine, SessionFactory = init_db(db_path)
        override_session_factory(SessionFactory)
        yield db_path, engine, SessionFactory
        engine.dispose()


@pytest.fixture
def client(temp_db):
    """Create a FastAPI test client with temp database.

    Overrides the database dependency to use the temporary test database.
    The dependency override is cleared after the test completes.

    Args:
        temp_db: Temporary database fixture.

    Yields:
        tuple: (test_client, SessionFactory)
    """
    db_path, engine, SessionFactory = temp_db

    # Override the dependency
    def get_test_session():
        session = SessionFactory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db_session] = get_test_session

    with TestClient(app) as client:
        yield client, SessionFactory

    # Clean up dependency overrides
    app.dependency_overrides.clear()


@pytest.fixture
def sample_audio_file():
    """Create a sample WAV audio file for testing.

    Creates a minimal valid WAV file (1 second of silence, mono, 22050 Hz).
    The file is automatically cleaned up after the test completes.

    Yields:
        Path: Path to the temporary WAV file.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # Create a minimal valid WAV file
        with wave.open(f.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            # Write 1 second of silence (22050 samples * 2 bytes)
            wf.writeframes(b"\x00" * 22050 * 2)

        yield Path(f.name)

    # Cleanup
    try:
        Path(f.name).unlink()
    except OSError:
        pass
