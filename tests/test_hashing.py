"""Tests for app.utils.hashing module."""

import tempfile
from pathlib import Path

import pytest

from app.utils.hashing import feature_spec_alias, sha256_bytes, sha256_file


class TestSha256Bytes:
    """Tests for sha256_bytes function."""

    def test_empty_bytes(self):
        """Empty bytes should produce known SHA256 hash."""
        # SHA256 of empty input is a well-known constant
        expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert sha256_bytes(b"") == expected

    def test_known_input(self):
        """Known input should produce expected hash."""
        # SHA256("hello") is a known value
        expected = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        assert sha256_bytes(b"hello") == expected

    def test_returns_hex_only(self):
        """Hash should be hex digest only, no prefix."""
        result = sha256_bytes(b"test")
        assert not result.startswith("sha256:")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self):
        """Same input should always produce same output."""
        data = b"deterministic test data"
        assert sha256_bytes(data) == sha256_bytes(data)


class TestSha256File:
    """Tests for sha256_file function."""

    def test_file_hash_matches_bytes_hash(self):
        """File hash should match hash of file contents."""
        content = b"test file content for hashing"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            file_hash = sha256_file(temp_path)
            bytes_hash = sha256_bytes(content)
            assert file_hash == bytes_hash
        finally:
            Path(temp_path).unlink()

    def test_accepts_path_object(self):
        """Should accept Path objects."""
        content = b"path object test"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            result = sha256_file(temp_path)
            assert len(result) == 64
        finally:
            temp_path.unlink()

    def test_nonexistent_file_raises(self):
        """Nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            sha256_file("/nonexistent/path/file.txt")

    def test_returns_hex_only(self):
        """File hash should be hex digest only, no prefix."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"content")
            temp_path = f.name

        try:
            result = sha256_file(temp_path)
            assert not result.startswith("sha256:")
            assert len(result) == 64
        finally:
            Path(temp_path).unlink()


class TestFeatureSpecAlias:
    """Tests for feature_spec_alias function."""

    def test_alias_length_is_12(self):
        """Alias should always be exactly 12 characters."""
        spec_id = "mel64_h10ms_w25ms_sr22050__yamnet1024_h0.5s_onnx"
        alias = feature_spec_alias(spec_id)
        assert len(alias) == 12

    def test_alias_is_hex(self):
        """Alias should contain only hex characters."""
        spec_id = "any_feature_spec_id_string"
        alias = feature_spec_alias(spec_id)
        assert all(c in "0123456789abcdef" for c in alias)

    def test_alias_is_deterministic(self):
        """Same spec_id should always produce same alias."""
        spec_id = "test_spec_id"
        assert feature_spec_alias(spec_id) == feature_spec_alias(spec_id)

    def test_different_specs_different_aliases(self):
        """Different spec_ids should produce different aliases."""
        alias1 = feature_spec_alias("spec_v1")
        alias2 = feature_spec_alias("spec_v2")
        assert alias1 != alias2

    def test_alias_is_prefix_of_full_hash(self):
        """Alias should be first 12 chars of sha256(spec_id)."""
        spec_id = "test_spec"
        full_hash = sha256_bytes(spec_id.encode("utf-8"))
        alias = feature_spec_alias(spec_id)
        assert alias == full_hash[:12]

    def test_default_feature_spec(self):
        """Blueprint v1.4 default spec should produce valid alias."""
        from app.config import DEFAULT_FEATURE_SPEC_ID

        alias = feature_spec_alias(DEFAULT_FEATURE_SPEC_ID)
        assert len(alias) == 12
        assert all(c in "0123456789abcdef" for c in alias)
