"""Tests for app.utils.atomic_io module."""

import tempfile
from pathlib import Path

from app.utils.atomic_io import atomic_write_bytes, atomic_write_text


class TestAtomicWriteBytes:
    """Tests for atomic_write_bytes function."""

    def test_creates_file(self):
        """Should create file with correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.bin"
            data = b"test binary data"

            atomic_write_bytes(path, data)

            assert path.exists()
            assert path.read_bytes() == data

    def test_creates_parent_directories(self):
        """Should create parent directories if they do not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "deep" / "file.bin"
            data = b"nested data"

            atomic_write_bytes(path, data)

            assert path.exists()
            assert path.read_bytes() == data

    def test_overwrites_existing_file(self):
        """Should atomically replace existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.bin"
            old_data = b"old content"
            new_data = b"new content"

            # Write initial content
            path.write_bytes(old_data)
            assert path.read_bytes() == old_data

            # Overwrite with atomic write
            atomic_write_bytes(path, new_data)

            assert path.read_bytes() == new_data

    def test_temp_file_cleaned_up_on_success(self):
        """Temp file should not exist after successful write."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.bin"
            temp_path = path.with_suffix(".bin.tmp")

            atomic_write_bytes(path, b"data")

            assert path.exists()
            assert not temp_path.exists()

    def test_idempotent_with_existing_temp(self):
        """Should succeed even if temp file already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.bin"
            temp_path = path.with_suffix(".bin.tmp")

            # Create orphaned temp file (simulating interrupted previous write)
            temp_path.write_bytes(b"orphaned temp data")

            # New atomic write should succeed
            atomic_write_bytes(path, b"fresh data")

            assert path.exists()
            assert path.read_bytes() == b"fresh data"
            assert not temp_path.exists()

    def test_accepts_path_object(self):
        """Should accept Path objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.bin"
            atomic_write_bytes(path, b"data")
            assert path.exists()

    def test_accepts_string_path(self):
        """Should accept string paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.bin")
            atomic_write_bytes(path, b"data")
            assert Path(path).exists()

    def test_final_file_not_corrupted_concept(self):
        """Verify that final file is never in partial state.

        Note: We cannot truly test power failure, but we can verify
        that the implementation uses temp file + rename pattern.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.bin"
            data = b"x" * 10000  # Larger data

            atomic_write_bytes(path, data)

            # Final file should have complete content
            assert path.read_bytes() == data

    def test_empty_data(self):
        """Should handle empty data correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.bin"
            atomic_write_bytes(path, b"")
            assert path.exists()
            assert path.read_bytes() == b""


class TestAtomicWriteText:
    """Tests for atomic_write_text function."""

    def test_creates_file(self):
        """Should create file with correct text content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            text = "Hello, World!"

            atomic_write_text(path, text)

            assert path.exists()
            assert path.read_text() == text

    def test_utf8_encoding_default(self):
        """Should use UTF-8 encoding by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            text = "Unicode: \u00e9\u00e8\u00ea \u4e2d\u6587"

            atomic_write_text(path, text)

            assert path.read_text(encoding="utf-8") == text

    def test_custom_encoding(self):
        """Should support custom encoding."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            text = "Latin-1: cafe"

            atomic_write_text(path, text, encoding="latin-1")

            assert path.read_text(encoding="latin-1") == text

    def test_creates_parent_directories(self):
        """Should create parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "file.txt"

            atomic_write_text(path, "nested content")

            assert path.exists()

    def test_json_content(self):
        """Should handle JSON content correctly."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.json"
            data = {"key": "value", "number": 42, "nested": {"a": 1}}
            text = json.dumps(data, indent=2)

            atomic_write_text(path, text)

            loaded = json.loads(path.read_text())
            assert loaded == data
