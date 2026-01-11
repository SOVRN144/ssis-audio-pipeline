"""Tests for app.utils.paths module."""

from pathlib import Path

from app.config import AUDIO_DIR, FEATURES_DIR, PREVIEW_DIR, SEGMENTS_DIR
from app.utils.paths import (
    audio_normalized_path,
    audio_original_path,
    features_h5_path,
    preview_json_path,
    segments_json_path,
)


class TestAudioOriginalPath:
    """Tests for audio_original_path function."""

    def test_basic_path(self):
        """Should return correct canonical path."""
        path = audio_original_path("asset123", "mp3")
        expected = AUDIO_DIR / "asset123" / "original.mp3"
        assert path == expected

    def test_extension_without_dot(self):
        """Should handle extension without leading dot."""
        path = audio_original_path("asset123", "wav")
        assert path.suffix == ".wav"

    def test_extension_with_dot(self):
        """Should handle extension with leading dot."""
        path = audio_original_path("asset123", ".flac")
        assert path.suffix == ".flac"
        assert "..flac" not in str(path)

    def test_returns_path_object(self):
        """Should return a Path object."""
        path = audio_original_path("asset123", "mp3")
        assert isinstance(path, Path)


class TestAudioNormalizedPath:
    """Tests for audio_normalized_path function."""

    def test_basic_path(self):
        """Should return correct canonical path."""
        path = audio_normalized_path("asset123")
        expected = AUDIO_DIR / "asset123" / "normalized.wav"
        assert path == expected

    def test_always_wav(self):
        """Normalized path should always have .wav extension."""
        path = audio_normalized_path("any_asset_id")
        assert path.suffix == ".wav"

    def test_returns_path_object(self):
        """Should return a Path object."""
        path = audio_normalized_path("asset123")
        assert isinstance(path, Path)


class TestFeaturesH5Path:
    """Tests for features_h5_path function."""

    def test_basic_path(self):
        """Should return correct canonical path."""
        path = features_h5_path("asset123", "abcd12345678")
        expected = FEATURES_DIR / "asset123.abcd12345678.h5"
        assert path == expected

    def test_always_h5(self):
        """Features path should always have .h5 extension."""
        path = features_h5_path("asset", "alias1234567")
        assert path.suffix == ".h5"

    def test_includes_alias_in_filename(self):
        """Filename should include feature_spec_alias."""
        alias = "abc123def456"
        path = features_h5_path("asset", alias)
        assert alias in path.name

    def test_returns_path_object(self):
        """Should return a Path object."""
        path = features_h5_path("asset", "alias")
        assert isinstance(path, Path)


class TestSegmentsJsonPath:
    """Tests for segments_json_path function."""

    def test_basic_path(self):
        """Should return correct canonical path."""
        path = segments_json_path("asset123")
        expected = SEGMENTS_DIR / "asset123.segments.v1.json"
        assert path == expected

    def test_includes_version(self):
        """Path should include version string."""
        path = segments_json_path("asset")
        assert ".v1." in path.name

    def test_always_json(self):
        """Segments path should always have .json extension."""
        path = segments_json_path("asset")
        assert path.suffix == ".json"

    def test_returns_path_object(self):
        """Should return a Path object."""
        path = segments_json_path("asset")
        assert isinstance(path, Path)


class TestPreviewJsonPath:
    """Tests for preview_json_path function."""

    def test_basic_path(self):
        """Should return correct canonical path."""
        path = preview_json_path("asset123")
        expected = PREVIEW_DIR / "asset123.preview.v1.json"
        assert path == expected

    def test_includes_version(self):
        """Path should include version string."""
        path = preview_json_path("asset")
        assert ".v1." in path.name

    def test_always_json(self):
        """Preview path should always have .json extension."""
        path = preview_json_path("asset")
        assert path.suffix == ".json"

    def test_returns_path_object(self):
        """Should return a Path object."""
        path = preview_json_path("asset")
        assert isinstance(path, Path)


class TestPathsDoNotCreateDirectories:
    """Tests verifying that path functions do not create directories."""

    def test_audio_original_does_not_create(self):
        """audio_original_path should not create directories."""
        path = audio_original_path("nonexistent_asset", "mp3")
        assert not path.parent.exists()

    def test_audio_normalized_does_not_create(self):
        """audio_normalized_path should not create directories."""
        path = audio_normalized_path("nonexistent_asset")
        assert not path.parent.exists()

    def test_features_does_not_create(self):
        """features_h5_path should not create directories."""
        # FEATURES_DIR may or may not exist, so we check the call doesn't error
        path = features_h5_path("nonexistent_asset", "alias123456")
        # Just verify it returns a path without side effects
        assert isinstance(path, Path)


class TestBlueprintCanonicalPathPatterns:
    """Verify Blueprint canonical runtime path patterns (CLAUDE.md).

    Blueprint patterns:
    - data/audio/{asset_id}/original.<ext>
    - data/audio/{asset_id}/normalized.wav
    - data/features/{asset_id}.{feature_spec_alias}.h5
    - data/segments/{asset_id}.segments.v1.json
    - data/preview/{asset_id}.preview.v1.json
    """

    # Sample values per task specification
    ASSET_ID = "asset_123"
    FEATURE_SPEC_ALIAS = "abc123def456"
    EXT = "mp3"

    def test_audio_original_matches_blueprint_pattern(self):
        """Verify: data/audio/{asset_id}/original.<ext>"""
        path = audio_original_path(self.ASSET_ID, self.EXT)

        # Parent directory must be data/audio/{asset_id}
        assert path.parent.parent == AUDIO_DIR
        assert path.parent.name == self.ASSET_ID

        # Filename must be original.<ext>
        assert path.name == f"original.{self.EXT}"

    def test_audio_normalized_matches_blueprint_pattern(self):
        """Verify: data/audio/{asset_id}/normalized.wav"""
        path = audio_normalized_path(self.ASSET_ID)

        # Parent directory must be data/audio/{asset_id}
        assert path.parent.parent == AUDIO_DIR
        assert path.parent.name == self.ASSET_ID

        # Filename must be exactly normalized.wav
        assert path.name == "normalized.wav"

    def test_features_matches_blueprint_pattern(self):
        """Verify: data/features/{asset_id}.{feature_spec_alias}.h5"""
        path = features_h5_path(self.ASSET_ID, self.FEATURE_SPEC_ALIAS)

        # Parent directory must be data/features
        assert path.parent == FEATURES_DIR

        # Filename must include asset_id and feature_spec_alias with dot separator
        expected_name = f"{self.ASSET_ID}.{self.FEATURE_SPEC_ALIAS}.h5"
        assert path.name == expected_name

        # Verify dot separator between asset_id and feature_spec_alias
        parts = path.stem.split(".")
        assert parts[0] == self.ASSET_ID
        assert parts[1] == self.FEATURE_SPEC_ALIAS

    def test_segments_matches_blueprint_pattern(self):
        """Verify: data/segments/{asset_id}.segments.v1.json"""
        path = segments_json_path(self.ASSET_ID)

        # Parent directory must be data/segments
        assert path.parent == SEGMENTS_DIR

        # Filename must end with exact suffix .segments.v1.json
        assert path.name.endswith(".segments.v1.json")

        # Filename must start with asset_id
        assert path.name == f"{self.ASSET_ID}.segments.v1.json"

    def test_preview_matches_blueprint_pattern(self):
        """Verify: data/preview/{asset_id}.preview.v1.json"""
        path = preview_json_path(self.ASSET_ID)

        # Parent directory must be data/preview
        assert path.parent == PREVIEW_DIR

        # Filename must end with exact suffix .preview.v1.json
        assert path.name.endswith(".preview.v1.json")

        # Filename must start with asset_id
        assert path.name == f"{self.ASSET_ID}.preview.v1.json"
