"""Tests for atomic copy and filesystem operations.

Tests atomic file copy semantics:
- No .tmp files left on success
- Orphan temp cleanup on simulated failure
- Exception-based failure injection
"""

import io
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from app.utils.atomic_io import (
    atomic_copy_file,
    atomic_stream_to_file,
    cleanup_orphan_temp_files,
)


class TestAtomicCopyFile:
    """Tests for atomic_copy_file function."""

    def test_copies_file_successfully(self):
        """Should copy file content correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.bin"
            dest = Path(tmpdir) / "dest" / "output.bin"

            # Create source file
            content = b"test binary content " * 100
            source.write_bytes(content)

            # Copy atomically
            atomic_copy_file(source, dest)

            # Verify
            assert dest.exists()
            assert dest.read_bytes() == content

    def test_creates_parent_directories(self):
        """Should create nested parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.bin"
            dest = Path(tmpdir) / "deep" / "nested" / "path" / "output.bin"

            source.write_bytes(b"data")
            atomic_copy_file(source, dest)

            assert dest.exists()
            assert dest.read_bytes() == b"data"

    def test_no_temp_file_on_success(self):
        """Temp file should not exist after successful copy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.bin"
            dest = Path(tmpdir) / "output.bin"
            temp_path = dest.with_suffix(".bin.tmp")

            source.write_bytes(b"data")
            atomic_copy_file(source, dest)

            assert dest.exists()
            assert not temp_path.exists()

    def test_raises_file_not_found(self):
        """Should raise FileNotFoundError for missing source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "nonexistent.bin"
            dest = Path(tmpdir) / "output.bin"

            with pytest.raises(FileNotFoundError):
                atomic_copy_file(source, dest)

    def test_overwrites_existing_temp(self):
        """Should succeed even if orphan temp file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.bin"
            dest = Path(tmpdir) / "output.bin"
            temp_path = dest.with_suffix(".bin.tmp")

            source.write_bytes(b"new data")
            temp_path.write_bytes(b"orphan temp")

            atomic_copy_file(source, dest)

            assert dest.exists()
            assert dest.read_bytes() == b"new data"
            assert not temp_path.exists()

    def test_overwrites_existing_dest(self):
        """Should atomically replace existing destination."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.bin"
            dest = Path(tmpdir) / "output.bin"

            source.write_bytes(b"new content")
            dest.write_bytes(b"old content")

            atomic_copy_file(source, dest)

            assert dest.read_bytes() == b"new content"

    def test_cleanup_on_write_failure(self):
        """Should cleanup temp file on write failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.bin"
            dest = Path(tmpdir) / "output.bin"
            temp_path = dest.with_suffix(".bin.tmp")

            source.write_bytes(b"x" * 1000)

            # Mock os.write to fail after partial write
            original_write = os.write

            def failing_write(fd, data):
                # Write some data then fail
                original_write(fd, data[:10])
                raise OSError("Simulated write failure")

            with mock.patch("os.write", side_effect=failing_write):
                with pytest.raises(OSError):
                    atomic_copy_file(source, dest)

            # Temp file should be cleaned up
            assert not temp_path.exists()
            # Dest should not exist (was never created)
            assert not dest.exists()


class TestAtomicStreamToFile:
    """Tests for atomic_stream_to_file function."""

    def test_writes_stream_successfully(self):
        """Should write stream content to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "output.bin"
            content = b"stream content " * 100

            stream = io.BytesIO(content)
            bytes_written = atomic_stream_to_file(stream, dest)

            assert dest.exists()
            assert dest.read_bytes() == content
            assert bytes_written == len(content)

    def test_no_temp_file_on_success(self):
        """Temp file should not exist after successful write."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "output.bin"
            temp_path = dest.with_suffix(".bin.tmp")

            stream = io.BytesIO(b"data")
            atomic_stream_to_file(stream, dest)

            assert dest.exists()
            assert not temp_path.exists()

    def test_handles_text_stream(self):
        """Should handle text streams by encoding to UTF-8."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "output.txt"

            # StringIO would return strings from read()
            class TextStream:
                def __init__(self, text):
                    self.text = text
                    self.pos = 0

                def read(self, size):
                    if self.pos >= len(self.text):
                        return ""
                    chunk = self.text[self.pos : self.pos + size]
                    self.pos += size
                    return chunk

            stream = TextStream("hello world")
            atomic_stream_to_file(stream, dest)

            assert dest.exists()
            assert dest.read_text() == "hello world"

    def test_cleanup_on_stream_failure(self):
        """Should cleanup temp file on stream read failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "output.bin"
            temp_path = dest.with_suffix(".bin.tmp")

            class FailingStream:
                def __init__(self):
                    self.reads = 0

                def read(self, size):
                    self.reads += 1
                    if self.reads > 2:
                        raise OSError("Simulated stream failure")
                    return b"x" * size

            stream = FailingStream()
            with pytest.raises(OSError):
                atomic_stream_to_file(stream, dest)

            # Temp file should be cleaned up
            assert not temp_path.exists()


class TestCleanupOrphanTempFiles:
    """Tests for cleanup_orphan_temp_files function."""

    def test_removes_temp_files(self):
        """Should remove .tmp files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create some temp files
            (tmpdir / "file1.wav.tmp").write_bytes(b"orphan 1")
            (tmpdir / "file2.mp3.tmp").write_bytes(b"orphan 2")
            (tmpdir / "file3.json.tmp").write_bytes(b"orphan 3")
            # Create a non-temp file
            (tmpdir / "keep.wav").write_bytes(b"keep this")

            removed = cleanup_orphan_temp_files(tmpdir)

            assert removed == 3
            assert not (tmpdir / "file1.wav.tmp").exists()
            assert not (tmpdir / "file2.mp3.tmp").exists()
            assert not (tmpdir / "file3.json.tmp").exists()
            # Non-temp file should remain
            assert (tmpdir / "keep.wav").exists()

    def test_handles_nonexistent_directory(self):
        """Should handle non-existent directory gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = Path(tmpdir) / "does_not_exist"
            removed = cleanup_orphan_temp_files(nonexistent)
            assert removed == 0

    def test_custom_suffix(self):
        """Should respect custom temp suffix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            (tmpdir / "file1.partial").write_bytes(b"orphan 1")
            (tmpdir / "file2.partial").write_bytes(b"orphan 2")
            (tmpdir / "file3.tmp").write_bytes(b"not this")

            removed = cleanup_orphan_temp_files(tmpdir, temp_suffix=".partial")

            assert removed == 2
            assert not (tmpdir / "file1.partial").exists()
            assert not (tmpdir / "file2.partial").exists()
            assert (tmpdir / "file3.tmp").exists()

    def test_handles_permission_errors(self):
        """Should continue on permission errors (best-effort)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            (tmpdir / "file1.tmp").write_bytes(b"orphan 1")
            (tmpdir / "file2.tmp").write_bytes(b"orphan 2")

            # Mock unlink to fail for one file
            original_unlink = Path.unlink
            call_count = [0]

            def partial_fail_unlink(self, missing_ok=False):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise OSError("Permission denied")
                return original_unlink(self, missing_ok=missing_ok)

            with mock.patch.object(Path, "unlink", partial_fail_unlink):
                removed = cleanup_orphan_temp_files(tmpdir)

            # Should still remove at least one file
            assert removed >= 1


class TestAtomicCopyIdempotency:
    """Tests for idempotent behavior of atomic operations."""

    def test_copy_is_idempotent(self):
        """Repeated atomic copy should produce same result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.bin"
            dest = Path(tmpdir) / "output.bin"

            content = b"idempotent content"
            source.write_bytes(content)

            # Copy multiple times
            atomic_copy_file(source, dest)
            atomic_copy_file(source, dest)
            atomic_copy_file(source, dest)

            # Result should be same
            assert dest.read_bytes() == content

    def test_stream_write_is_idempotent(self):
        """Repeated stream write should produce same result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "output.bin"
            content = b"idempotent stream content"

            # Write multiple times
            for _ in range(3):
                stream = io.BytesIO(content)
                atomic_stream_to_file(stream, dest)

            # Result should be same
            assert dest.read_bytes() == content
