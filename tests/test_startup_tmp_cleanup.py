"""Tests for startup temp file cleanup.

Step 3: Tests resilience hook for cleaning up orphan .tmp files on startup.
"""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_audio_dir():
    """Create a temporary audio directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = Path(tmpdir) / "audio"
        audio_dir.mkdir()

        # Create some asset subdirectories
        asset1_dir = audio_dir / "asset-001"
        asset1_dir.mkdir()

        asset2_dir = audio_dir / "asset-002"
        asset2_dir.mkdir()

        yield audio_dir


class TestOrphanTempCleanup:
    """Tests for cleanup_orphan_temp_files function."""

    def test_cleanup_removes_tmp_files(self, temp_audio_dir):
        """Should remove .tmp files from directory."""
        from app.utils.atomic_io import cleanup_orphan_temp_files

        # Create some .tmp files
        (temp_audio_dir / "file1.wav.tmp").write_bytes(b"orphan1")
        (temp_audio_dir / "file2.wav.tmp").write_bytes(b"orphan2")
        (temp_audio_dir / "final.wav").write_bytes(b"real file")

        removed = cleanup_orphan_temp_files(temp_audio_dir)

        assert removed == 2
        assert not (temp_audio_dir / "file1.wav.tmp").exists()
        assert not (temp_audio_dir / "file2.wav.tmp").exists()
        assert (temp_audio_dir / "final.wav").exists()  # Real file preserved

    def test_cleanup_handles_empty_directory(self, temp_audio_dir):
        """Should handle empty directories gracefully."""
        from app.utils.atomic_io import cleanup_orphan_temp_files

        empty_dir = temp_audio_dir / "empty"
        empty_dir.mkdir()

        removed = cleanup_orphan_temp_files(empty_dir)

        assert removed == 0

    def test_cleanup_handles_nonexistent_directory(self):
        """Should handle nonexistent directories gracefully."""
        from app.utils.atomic_io import cleanup_orphan_temp_files

        removed = cleanup_orphan_temp_files("/nonexistent/path/12345")

        assert removed == 0

    def test_cleanup_in_subdirectories(self, temp_audio_dir):
        """Should clean up .tmp files in asset subdirectories."""
        from app.utils.atomic_io import cleanup_orphan_temp_files

        asset_dir = temp_audio_dir / "asset-001"

        # Create orphan temp files in asset subdirectory
        (asset_dir / "original.wav.tmp").write_bytes(b"orphan")
        (asset_dir / "normalized.wav.tmp").write_bytes(b"orphan")
        (asset_dir / "original.wav").write_bytes(b"real file")

        removed = cleanup_orphan_temp_files(asset_dir)

        assert removed == 2
        assert not (asset_dir / "original.wav.tmp").exists()
        assert not (asset_dir / "normalized.wav.tmp").exists()
        assert (asset_dir / "original.wav").exists()


class TestStartupCleanupHook:
    """Tests for startup cleanup hook in FastAPI lifespan."""

    def test_startup_cleanup_invoked(self, temp_audio_dir):
        """Startup should invoke cleanup for audio directories."""
        from unittest.mock import patch

        # Create orphan temp files
        asset_dir = temp_audio_dir / "asset-001"
        (asset_dir / "orphan.wav.tmp").write_bytes(b"orphan")

        # Patch AUDIO_DIR at the config module level (source of truth)
        with patch("app.config.AUDIO_DIR", temp_audio_dir):
            # Reload the main module so its module-level imports pick up the patched config.
            # This is necessary because the cleanup function imports AUDIO_DIR at call time.
            # Note: This pattern can be fragile under pytest-xdist parallel execution;
            # may need serial marker (pytest.mark.serial) if issues arise.
            import importlib

            import services.ingest_api.main as main_module

            importlib.reload(main_module)

            main_module._cleanup_orphan_temp_files_safe()

        # Verify orphan was cleaned
        assert not (asset_dir / "orphan.wav.tmp").exists()

    def test_startup_cleanup_never_crashes(self, temp_audio_dir):
        """Startup cleanup should never crash even on errors."""
        from unittest.mock import MagicMock, patch

        # Create a mock that raises on exists()
        mock_dir = MagicMock()
        mock_dir.exists.side_effect = PermissionError("Access denied")

        # Patch at config source; reload main module to pick up patched value
        with patch("app.config.AUDIO_DIR", mock_dir):
            import importlib

            import services.ingest_api.main as main_module

            importlib.reload(main_module)

            # Should not raise
            main_module._cleanup_orphan_temp_files_safe()

    def test_startup_cleanup_logs_count(self, temp_audio_dir, caplog):
        """Startup cleanup should log number of files removed."""
        import logging
        from unittest.mock import patch

        # Create orphan temp files
        asset_dir = temp_audio_dir / "asset-001"
        (asset_dir / "orphan1.tmp").write_bytes(b"orphan")
        (temp_audio_dir / "orphan2.tmp").write_bytes(b"orphan")

        # Patch at config source; reload main module to pick up patched value
        with patch("app.config.AUDIO_DIR", temp_audio_dir):
            import importlib

            import services.ingest_api.main as main_module

            importlib.reload(main_module)

            with caplog.at_level(logging.INFO):
                main_module._cleanup_orphan_temp_files_safe()

        # Verify log message
        assert any(
            "removed" in record.message and "orphan temp files" in record.message
            for record in caplog.records
        )


class TestStartupCleanupIntegration:
    """Integration tests for startup cleanup with FastAPI lifespan."""

    def test_lifespan_includes_cleanup(self, temp_audio_dir):
        """FastAPI lifespan should include cleanup hook."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from fastapi.testclient import TestClient

        # Create orphan temp file
        asset_dir = temp_audio_dir / "asset-test"
        asset_dir.mkdir()
        orphan_path = asset_dir / "test.wav.tmp"
        orphan_path.write_bytes(b"orphan data")

        with tempfile.TemporaryDirectory() as db_tmpdir:
            db_path = Path(db_tmpdir) / "test.db"

            # Patch at config source; reload main module to pick up patched values
            with patch("app.config.AUDIO_DIR", temp_audio_dir):
                with patch("app.config.DB_PATH", db_path):
                    import importlib

                    import services.ingest_api.main as main_module

                    importlib.reload(main_module)

                    # TestClient invokes lifespan
                    with TestClient(main_module.app):
                        pass  # Lifespan runs on enter

            # Verify orphan was cleaned during startup
            assert not orphan_path.exists()
