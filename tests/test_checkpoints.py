"""Tests for app.utils.checkpoints module."""

import json
import tempfile
from pathlib import Path

from app.utils.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION,
    delete_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


class TestSaveCheckpoint:
    """Tests for save_checkpoint function."""

    def test_creates_checkpoint_file(self):
        """Should create checkpoint file with correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.ckpt.json"
            data = {"key": "value", "number": 42}

            save_checkpoint(path, data)

            assert path.exists()
            content = json.loads(path.read_text())
            assert content["key"] == "value"
            assert content["number"] == 42

    def test_adds_schema_version(self):
        """Should add schema_version to checkpoint data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.ckpt.json"
            data = {"key": "value"}

            save_checkpoint(path, data)

            content = json.loads(path.read_text())
            assert "schema_version" in content
            assert content["schema_version"] == CHECKPOINT_SCHEMA_VERSION

    def test_creates_parent_directories(self):
        """Should create parent directories if they do not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "deep" / "test.ckpt.json"
            data = {"key": "value"}

            save_checkpoint(path, data)

            assert path.exists()

    def test_overwrites_existing_checkpoint(self):
        """Should overwrite existing checkpoint file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.ckpt.json"

            save_checkpoint(path, {"old": "data"})
            save_checkpoint(path, {"new": "data"})

            content = json.loads(path.read_text())
            assert "new" in content
            assert "old" not in content

    def test_accepts_string_path(self):
        """Should accept string paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.ckpt.json")

            save_checkpoint(path, {"key": "value"})

            assert Path(path).exists()


class TestLoadCheckpoint:
    """Tests for load_checkpoint function."""

    def test_loads_valid_checkpoint(self):
        """Should load valid checkpoint data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.ckpt.json"
            save_checkpoint(path, {"key": "value", "number": 42})

            result = load_checkpoint(path)

            assert result is not None
            assert result["key"] == "value"
            assert result["number"] == 42
            assert result["schema_version"] == CHECKPOINT_SCHEMA_VERSION

    def test_returns_none_for_missing_file(self):
        """Should return None when checkpoint file does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.ckpt.json"

            result = load_checkpoint(path)

            assert result is None

    def test_returns_none_for_invalid_json(self):
        """Should return None when file contains invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.ckpt.json"
            path.write_text("not valid json {{{")

            result = load_checkpoint(path)

            assert result is None

    def test_returns_none_for_non_dict_json(self):
        """Should return None when JSON is not a dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "array.ckpt.json"
            path.write_text('["not", "a", "dict"]')

            result = load_checkpoint(path)

            assert result is None

    def test_returns_none_for_wrong_schema_version(self):
        """Should return None when schema_version does not match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "old.ckpt.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "0.0.1",
                        "key": "value",
                    }
                )
            )

            result = load_checkpoint(path)

            assert result is None

    def test_returns_none_for_missing_schema_version(self):
        """Should return None when schema_version is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "no_version.ckpt.json"
            path.write_text(json.dumps({"key": "value"}))

            result = load_checkpoint(path)

            assert result is None

    def test_accepts_string_path(self):
        """Should accept string paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.ckpt.json")
            save_checkpoint(path, {"key": "value"})

            result = load_checkpoint(path)

            assert result is not None


class TestDeleteCheckpoint:
    """Tests for delete_checkpoint function."""

    def test_deletes_existing_checkpoint(self):
        """Should delete existing checkpoint file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.ckpt.json"
            save_checkpoint(path, {"key": "value"})
            assert path.exists()

            result = delete_checkpoint(path)

            assert result is True
            assert not path.exists()

    def test_returns_false_for_missing_file(self):
        """Should return False when file does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.ckpt.json"

            result = delete_checkpoint(path)

            assert result is False

    def test_accepts_string_path(self):
        """Should accept string paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.ckpt.json")
            save_checkpoint(path, {"key": "value"})

            result = delete_checkpoint(path)

            assert result is True
            assert not Path(path).exists()


class TestCheckpointRoundTrip:
    """Integration tests for checkpoint save/load/delete cycle."""

    def test_save_load_delete_cycle(self):
        """Should handle complete save/load/delete cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cycle.ckpt.json"
            data = {
                "seconds_processed": 60.5,
                "sample_rate": 22050,
                "channels": 1,
            }

            # Save
            save_checkpoint(path, data)
            assert path.exists()

            # Load
            loaded = load_checkpoint(path)
            assert loaded is not None
            assert loaded["seconds_processed"] == 60.5
            assert loaded["sample_rate"] == 22050
            assert loaded["channels"] == 1
            assert loaded["schema_version"] == CHECKPOINT_SCHEMA_VERSION

            # Delete
            result = delete_checkpoint(path)
            assert result is True
            assert not path.exists()

            # Load after delete
            loaded_after = load_checkpoint(path)
            assert loaded_after is None
