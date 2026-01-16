"""Tests for the features worker (services/worker_features/run.py).

All tests mock onnxruntime to run without the actual ONNX model.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
import wave
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import h5py
import numpy as np
import pytest

from app.config import DEFAULT_FEATURE_SPEC_ID
from app.db import FeatureSpecAliasCollision, init_db, register_feature_spec
from app.models import FeatureSpec
from app.utils.hashing import feature_spec_alias
from services.worker_features.run import (
    ARTIFACT_SCHEMA_VERSION,
    EMBED_HOP_SEC,
    EMBEDDING_DIM,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    WIN_LENGTH,
    FeaturesErrorCode,
    _validate_features,
    extract_features,
)

if TYPE_CHECKING:
    pass


# --- Mock external modules ---
# Create mock modules for onnxruntime and librosa if not installed
# This allows tests to run in CI without these heavy dependencies

if "onnxruntime" not in sys.modules:
    mock_ort = mock.MagicMock()
    mock_ort.SessionOptions = mock.MagicMock()
    mock_ort.InferenceSession = mock.MagicMock()
    sys.modules["onnxruntime"] = mock_ort

if "librosa" not in sys.modules:
    mock_librosa = mock.MagicMock()
    mock_librosa.load = mock.MagicMock(return_value=(np.zeros(SAMPLE_RATE * 3), SAMPLE_RATE))
    mock_librosa.feature = mock.MagicMock()
    mock_librosa.feature.melspectrogram = mock.MagicMock()
    mock_librosa.power_to_db = mock.MagicMock()
    mock_librosa.resample = mock.MagicMock()
    sys.modules["librosa"] = mock_librosa
    sys.modules["librosa.feature"] = mock_librosa.feature


# --- Fixtures ---


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        engine, SessionFactory = init_db(db_path)
        yield tmpdir, engine, SessionFactory
        engine.dispose()


@pytest.fixture
def temp_dirs(temp_db):
    """Create temporary audio and features directories."""
    tmpdir, engine, SessionFactory = temp_db
    audio_dir = Path(tmpdir) / "data" / "audio"
    features_dir = Path(tmpdir) / "data" / "features"
    audio_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    return tmpdir, engine, SessionFactory, audio_dir, features_dir


def _create_test_wav(path: Path, duration_sec: float = 3.0) -> None:
    """Create a test WAV file with silence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(duration_sec * SAMPLE_RATE)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"\x00" * num_samples * 2)


def _generate_mock_mel(n_frames: int = 100) -> np.ndarray:
    """Generate mock mel spectrogram."""
    return np.random.randn(n_frames, N_MELS).astype(np.float32)


def _generate_mock_embeddings(n_embed_frames: int = 6) -> np.ndarray:
    """Generate mock YAMNet embeddings."""
    return np.random.randn(n_embed_frames, EMBEDDING_DIM).astype(np.float32)


# --- Test: FeatureSpec ID/Alias Derivation (A) ---


class TestFeatureSpecDerivation:
    """Tests for FeatureSpec ID and alias derivation."""

    def test_feature_spec_id_equals_locked_string(self):
        """Test that feature_spec_id equals the exact locked v1.4 string."""
        expected = "mel64_h10ms_w25ms_sr22050__yamnet1024_h0.5s_onnx"
        assert DEFAULT_FEATURE_SPEC_ID == expected

    def test_feature_spec_alias_equals_sha256_first_12(self):
        """Test that feature_spec_alias equals sha256(id)[:12]."""
        spec_id = DEFAULT_FEATURE_SPEC_ID
        expected_alias = hashlib.sha256(spec_id.encode()).hexdigest()[:12]
        actual_alias = feature_spec_alias(spec_id)
        assert actual_alias == expected_alias

    def test_alias_is_12_hex_chars(self):
        """Test that alias is exactly 12 lowercase hex characters."""
        alias = feature_spec_alias(DEFAULT_FEATURE_SPEC_ID)
        assert len(alias) == 12
        assert all(c in "0123456789abcdef" for c in alias)


# --- Test: FeatureSpec Immutability / Collision (B) ---


class TestFeatureSpecImmutability:
    """Tests for FeatureSpec immutability enforcement."""

    def test_insert_new_spec_succeeds(self, temp_db):
        """Test that inserting a new spec succeeds."""
        _, _, SessionFactory = temp_db
        session = SessionFactory()

        alias = register_feature_spec(session, DEFAULT_FEATURE_SPEC_ID)
        session.commit()

        # Verify it was inserted
        spec = session.query(FeatureSpec).filter_by(alias=alias).first()
        assert spec is not None
        assert spec.feature_spec_id == DEFAULT_FEATURE_SPEC_ID

        session.close()

    def test_insert_same_spec_is_idempotent(self, temp_db):
        """Test that inserting the same spec twice succeeds (idempotent)."""
        _, _, SessionFactory = temp_db
        session = SessionFactory()

        # Insert first time
        alias1 = register_feature_spec(session, DEFAULT_FEATURE_SPEC_ID)
        session.commit()

        # Insert second time - should succeed
        alias2 = register_feature_spec(session, DEFAULT_FEATURE_SPEC_ID)
        session.commit()

        assert alias1 == alias2

        session.close()

    def test_insert_different_spec_same_alias_fails(self, temp_db):
        """Test that inserting a different spec with same alias fails.

        This is a contrived test since realistic spec IDs won't collide.
        We manually insert a spec with a known alias, then try to register
        a different spec_id that would have the same alias (by mocking).
        """
        _, _, SessionFactory = temp_db
        session = SessionFactory()

        # Insert a spec
        register_feature_spec(session, DEFAULT_FEATURE_SPEC_ID)
        session.commit()

        # Now manually create a collision scenario
        existing_alias = feature_spec_alias(DEFAULT_FEATURE_SPEC_ID)

        # Create a FeatureSpec directly with same alias but different spec_id
        # This simulates what would happen if two different spec_ids hash to same alias
        different_spec_id = "completely_different_spec_id_that_should_collide"

        # Mock the alias computation to return the same alias
        # Need to patch at the hashing module since db.py imports it locally
        with mock.patch(
            "app.utils.hashing.feature_spec_alias",
            return_value=existing_alias,
        ):
            with pytest.raises(FeatureSpecAliasCollision) as exc_info:
                register_feature_spec(session, different_spec_id)

        assert exc_info.value.alias == existing_alias
        assert exc_info.value.existing_spec_id == DEFAULT_FEATURE_SPEC_ID
        assert exc_info.value.new_spec_id == different_spec_id

        session.close()


# --- Test: Model SHA Verification (C) ---


class TestModelShaVerification:
    """Tests for model SHA256 verification."""

    def test_placeholder_hash_fails_fast(self, temp_dirs, monkeypatch):
        """Test that placeholder SHA file fails with FEATURE_EXTRACTION_FAILED."""
        tmpdir, _, SessionFactory, audio_dir, features_dir = temp_dirs
        session = SessionFactory()

        asset_id = "test-placeholder-sha"
        asset_dir = audio_dir / asset_id
        _create_test_wav(asset_dir / "normalized.wav")

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.config.FEATURES_DIR", features_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.FEATURES_DIR", features_dir)

        # Create fake model with PLACEHOLDER hash (typical repo placeholder)
        yamnet_dir = Path(tmpdir) / "yamnet_onnx"
        yamnet_dir.mkdir(parents=True, exist_ok=True)
        (yamnet_dir / "yamnet.onnx").write_bytes(b"fake model content")
        # Use the actual placeholder from the repo
        (yamnet_dir / "yamnet.onnx.sha256").write_text(
            "placeholder_sha256_for_testing_must_be_replaced_with_real_model_hash"
        )

        monkeypatch.setattr(
            "services.worker_features.run.YAMNET_ONNX_PATH", yamnet_dir / "yamnet.onnx"
        )
        monkeypatch.setattr(
            "services.worker_features.run.YAMNET_SHA256_PATH", yamnet_dir / "yamnet.onnx.sha256"
        )

        result = extract_features(session, asset_id)

        assert not result.ok
        assert result.error_code == FeaturesErrorCode.FEATURE_EXTRACTION_FAILED
        assert "placeholder" in result.message.lower()

        # Verify no HDF5 was written
        spec_alias = feature_spec_alias(DEFAULT_FEATURE_SPEC_ID)
        output_path = features_dir / f"{asset_id}.{spec_alias}.h5"
        assert not output_path.exists()

        session.close()

    def test_short_hash_fails_as_placeholder(self, temp_dirs, monkeypatch):
        """Test that short/invalid SHA fails as placeholder."""
        tmpdir, _, SessionFactory, audio_dir, features_dir = temp_dirs
        session = SessionFactory()

        asset_id = "test-short-sha"
        asset_dir = audio_dir / asset_id
        _create_test_wav(asset_dir / "normalized.wav")

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.config.FEATURES_DIR", features_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.FEATURES_DIR", features_dir)

        yamnet_dir = Path(tmpdir) / "yamnet_onnx"
        yamnet_dir.mkdir(parents=True, exist_ok=True)
        (yamnet_dir / "yamnet.onnx").write_bytes(b"fake model")
        # Too short to be valid SHA256
        (yamnet_dir / "yamnet.onnx.sha256").write_text("abc123")

        monkeypatch.setattr(
            "services.worker_features.run.YAMNET_ONNX_PATH", yamnet_dir / "yamnet.onnx"
        )
        monkeypatch.setattr(
            "services.worker_features.run.YAMNET_SHA256_PATH", yamnet_dir / "yamnet.onnx.sha256"
        )

        result = extract_features(session, asset_id)

        assert not result.ok
        assert result.error_code == FeaturesErrorCode.FEATURE_EXTRACTION_FAILED
        assert "placeholder" in result.message.lower()

        session.close()

    def test_matching_hash_passes(self, temp_dirs, monkeypatch):
        """Test that matching model hash allows extraction to proceed."""
        tmpdir, _, SessionFactory, audio_dir, features_dir = temp_dirs
        session = SessionFactory()

        asset_id = "test-sha-match"
        asset_dir = audio_dir / asset_id
        _create_test_wav(asset_dir / "normalized.wav")

        # Patch paths
        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.config.FEATURES_DIR", features_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.FEATURES_DIR", features_dir)

        # Create fake model with known hash
        model_content = b"fake yamnet model content"
        expected_hash = hashlib.sha256(model_content).hexdigest()

        yamnet_dir = Path(tmpdir) / "yamnet_onnx"
        yamnet_dir.mkdir(parents=True, exist_ok=True)
        (yamnet_dir / "yamnet.onnx").write_bytes(model_content)
        (yamnet_dir / "yamnet.onnx.sha256").write_text(expected_hash)

        monkeypatch.setattr(
            "services.worker_features.run.YAMNET_ONNX_PATH", yamnet_dir / "yamnet.onnx"
        )
        monkeypatch.setattr(
            "services.worker_features.run.YAMNET_SHA256_PATH", yamnet_dir / "yamnet.onnx.sha256"
        )

        # Mock ONNX and librosa
        mock_mel = _generate_mock_mel()
        mock_embeddings = _generate_mock_embeddings()

        with mock.patch(
            "librosa.load", return_value=(np.zeros(SAMPLE_RATE * 3, dtype=np.float32), SAMPLE_RATE)
        ):
            with mock.patch(
                "services.worker_features.run._compute_mel_spectrogram", return_value=mock_mel
            ):
                with mock.patch("onnxruntime.SessionOptions"):
                    mock_session = mock.Mock()
                    mock_session.get_inputs.return_value = [
                        mock.Mock(name="input", shape=[1, 15360])
                    ]
                    mock_session.run.return_value = [np.zeros(521), mock_embeddings[0]]
                    with mock.patch("onnxruntime.InferenceSession", return_value=mock_session):
                        with mock.patch(
                            "services.worker_features.run._compute_yamnet_embeddings",
                            return_value=mock_embeddings,
                        ):
                            result = extract_features(session, asset_id)

        assert result.ok
        session.close()

    def test_mismatched_hash_fails(self, temp_dirs, monkeypatch):
        """Test that mismatched model hash fails with FEATURE_EXTRACTION_FAILED."""
        tmpdir, _, SessionFactory, audio_dir, features_dir = temp_dirs
        session = SessionFactory()

        asset_id = "test-sha-mismatch"
        asset_dir = audio_dir / asset_id
        _create_test_wav(asset_dir / "normalized.wav")

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.config.FEATURES_DIR", features_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.FEATURES_DIR", features_dir)

        # Create fake model with wrong hash
        yamnet_dir = Path(tmpdir) / "yamnet_onnx"
        yamnet_dir.mkdir(parents=True, exist_ok=True)
        (yamnet_dir / "yamnet.onnx").write_bytes(b"fake model content")
        (yamnet_dir / "yamnet.onnx.sha256").write_text(
            "0000000000000000000000000000000000000000000000000000000000000000"
        )

        monkeypatch.setattr(
            "services.worker_features.run.YAMNET_ONNX_PATH", yamnet_dir / "yamnet.onnx"
        )
        monkeypatch.setattr(
            "services.worker_features.run.YAMNET_SHA256_PATH", yamnet_dir / "yamnet.onnx.sha256"
        )

        result = extract_features(session, asset_id)

        assert not result.ok
        assert result.error_code == FeaturesErrorCode.FEATURE_EXTRACTION_FAILED
        assert "integrity" in result.message.lower() or "verification" in result.message.lower()

        # Verify no HDF5 was written
        spec_alias = feature_spec_alias(DEFAULT_FEATURE_SPEC_ID)
        output_path = features_dir / f"{asset_id}.{spec_alias}.h5"
        assert not output_path.exists()

        session.close()


# --- Test: Validation (D) ---


class TestValidation:
    """Tests for NaN/Inf and shape validation."""

    def test_nan_in_mel_fails(self):
        """Test that NaN in mel spectrogram fails with FEATURE_NAN."""
        mel = _generate_mock_mel()
        mel[10, 5] = np.nan
        embeddings = _generate_mock_embeddings()

        ok, error_code, nan_count = _validate_features(mel, embeddings)

        assert not ok
        assert error_code == FeaturesErrorCode.FEATURE_NAN
        assert nan_count >= 1

    def test_inf_in_mel_fails(self):
        """Test that Inf in mel spectrogram fails with FEATURE_NAN."""
        mel = _generate_mock_mel()
        mel[10, 5] = np.inf
        embeddings = _generate_mock_embeddings()

        ok, error_code, nan_count = _validate_features(mel, embeddings)

        assert not ok
        assert error_code == FeaturesErrorCode.FEATURE_NAN
        assert nan_count >= 1

    def test_nan_in_embeddings_fails(self):
        """Test that NaN in embeddings fails with FEATURE_NAN."""
        mel = _generate_mock_mel()
        embeddings = _generate_mock_embeddings()
        embeddings[2, 100] = np.nan

        ok, error_code, nan_count = _validate_features(mel, embeddings)

        assert not ok
        assert error_code == FeaturesErrorCode.FEATURE_NAN
        assert nan_count >= 1

    def test_wrong_mel_shape_fails(self):
        """Test that wrong mel shape fails with FEATURE_EXTRACTION_FAILED."""
        mel = np.random.randn(100, 32).astype(np.float32)  # Wrong n_mels
        embeddings = _generate_mock_embeddings()

        ok, error_code, _ = _validate_features(mel, embeddings)

        assert not ok
        assert error_code == FeaturesErrorCode.FEATURE_EXTRACTION_FAILED

    def test_wrong_embedding_shape_fails(self):
        """Test that wrong embedding shape fails with FEATURE_EXTRACTION_FAILED."""
        mel = _generate_mock_mel()
        embeddings = np.random.randn(6, 512).astype(np.float32)  # Wrong dim

        ok, error_code, _ = _validate_features(mel, embeddings)

        assert not ok
        assert error_code == FeaturesErrorCode.FEATURE_EXTRACTION_FAILED

    def test_valid_features_pass(self):
        """Test that valid features pass validation."""
        mel = _generate_mock_mel()
        embeddings = _generate_mock_embeddings()

        ok, error_code, nan_count = _validate_features(mel, embeddings)

        assert ok
        assert error_code is None
        assert nan_count == 0

    def test_nan_injection_no_h5_written(self, temp_dirs, monkeypatch):
        """Test that NaN injection prevents HDF5 from being written."""
        tmpdir, _, SessionFactory, audio_dir, features_dir = temp_dirs
        session = SessionFactory()

        asset_id = "test-nan-no-write"
        asset_dir = audio_dir / asset_id
        _create_test_wav(asset_dir / "normalized.wav")

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.config.FEATURES_DIR", features_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.FEATURES_DIR", features_dir)

        # Setup model verification to pass
        model_content = b"fake model"
        expected_hash = hashlib.sha256(model_content).hexdigest()
        yamnet_dir = Path(tmpdir) / "yamnet_onnx"
        yamnet_dir.mkdir(parents=True, exist_ok=True)
        (yamnet_dir / "yamnet.onnx").write_bytes(model_content)
        (yamnet_dir / "yamnet.onnx.sha256").write_text(expected_hash)
        monkeypatch.setattr(
            "services.worker_features.run.YAMNET_ONNX_PATH", yamnet_dir / "yamnet.onnx"
        )
        monkeypatch.setattr(
            "services.worker_features.run.YAMNET_SHA256_PATH", yamnet_dir / "yamnet.onnx.sha256"
        )

        # Generate mel with NaN
        mel_with_nan = _generate_mock_mel()
        mel_with_nan[5, 10] = np.nan
        mock_embeddings = _generate_mock_embeddings()

        with mock.patch(
            "librosa.load", return_value=(np.zeros(SAMPLE_RATE * 3, dtype=np.float32), SAMPLE_RATE)
        ):
            with mock.patch(
                "services.worker_features.run._compute_mel_spectrogram", return_value=mel_with_nan
            ):
                with mock.patch("onnxruntime.SessionOptions"):
                    mock_session = mock.Mock()
                    mock_session.get_inputs.return_value = [
                        mock.Mock(name="input", shape=[1, 15360])
                    ]
                    mock_session.run.return_value = [np.zeros(521), mock_embeddings[0]]
                    with mock.patch("onnxruntime.InferenceSession", return_value=mock_session):
                        with mock.patch(
                            "services.worker_features.run._compute_yamnet_embeddings",
                            return_value=mock_embeddings,
                        ):
                            result = extract_features(session, asset_id)

        assert not result.ok
        assert result.error_code == FeaturesErrorCode.FEATURE_NAN

        # Verify no HDF5 written
        spec_alias = feature_spec_alias(DEFAULT_FEATURE_SPEC_ID)
        output_path = features_dir / f"{asset_id}.{spec_alias}.h5"
        assert not output_path.exists()

        session.close()


# --- Test: Atomic Publish Boundary (E) ---


class TestAtomicPublish:
    """Tests for atomic publish behavior."""

    def test_tmp_not_treated_as_final(self, temp_dirs, monkeypatch):
        """Test that .tmp files are not treated as final artifacts."""
        tmpdir, _, SessionFactory, audio_dir, features_dir = temp_dirs
        session = SessionFactory()

        asset_id = "test-atomic-tmp"
        asset_dir = audio_dir / asset_id
        _create_test_wav(asset_dir / "normalized.wav")

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.config.FEATURES_DIR", features_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.FEATURES_DIR", features_dir)

        # Create an orphan .tmp file
        spec_alias = feature_spec_alias(DEFAULT_FEATURE_SPEC_ID)
        tmp_path = features_dir / f"{asset_id}.{spec_alias}.h5.tmp"
        tmp_path.write_bytes(b"orphan temp data")

        # Setup model verification
        model_content = b"fake model"
        expected_hash = hashlib.sha256(model_content).hexdigest()
        yamnet_dir = Path(tmpdir) / "yamnet_onnx"
        yamnet_dir.mkdir(parents=True, exist_ok=True)
        (yamnet_dir / "yamnet.onnx").write_bytes(model_content)
        (yamnet_dir / "yamnet.onnx.sha256").write_text(expected_hash)
        monkeypatch.setattr(
            "services.worker_features.run.YAMNET_ONNX_PATH", yamnet_dir / "yamnet.onnx"
        )
        monkeypatch.setattr(
            "services.worker_features.run.YAMNET_SHA256_PATH", yamnet_dir / "yamnet.onnx.sha256"
        )

        mock_mel = _generate_mock_mel()
        mock_embeddings = _generate_mock_embeddings()

        with mock.patch(
            "librosa.load", return_value=(np.zeros(SAMPLE_RATE * 3, dtype=np.float32), SAMPLE_RATE)
        ):
            with mock.patch(
                "services.worker_features.run._compute_mel_spectrogram", return_value=mock_mel
            ):
                with mock.patch("onnxruntime.SessionOptions"):
                    mock_session = mock.Mock()
                    mock_session.get_inputs.return_value = [
                        mock.Mock(name="input", shape=[1, 15360])
                    ]
                    mock_session.run.return_value = [np.zeros(521), mock_embeddings[0]]
                    with mock.patch("onnxruntime.InferenceSession", return_value=mock_session):
                        with mock.patch(
                            "services.worker_features.run._compute_yamnet_embeddings",
                            return_value=mock_embeddings,
                        ):
                            result = extract_features(session, asset_id)

        assert result.ok
        # Final should exist
        final_path = features_dir / f"{asset_id}.{spec_alias}.h5"
        assert final_path.exists()
        # Temp should not exist (cleaned up or replaced)
        assert not tmp_path.exists()

        session.close()

    def test_final_h5_appears_only_after_completion(self, temp_dirs, monkeypatch):
        """Test that final .h5 only appears after successful completion."""
        tmpdir, _, SessionFactory, audio_dir, features_dir = temp_dirs
        session = SessionFactory()

        asset_id = "test-atomic-final"
        asset_dir = audio_dir / asset_id
        _create_test_wav(asset_dir / "normalized.wav")

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.config.FEATURES_DIR", features_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.FEATURES_DIR", features_dir)

        # Setup model verification
        model_content = b"fake model"
        expected_hash = hashlib.sha256(model_content).hexdigest()
        yamnet_dir = Path(tmpdir) / "yamnet_onnx"
        yamnet_dir.mkdir(parents=True, exist_ok=True)
        (yamnet_dir / "yamnet.onnx").write_bytes(model_content)
        (yamnet_dir / "yamnet.onnx.sha256").write_text(expected_hash)
        monkeypatch.setattr(
            "services.worker_features.run.YAMNET_ONNX_PATH", yamnet_dir / "yamnet.onnx"
        )
        monkeypatch.setattr(
            "services.worker_features.run.YAMNET_SHA256_PATH", yamnet_dir / "yamnet.onnx.sha256"
        )

        spec_alias = feature_spec_alias(DEFAULT_FEATURE_SPEC_ID)
        final_path = features_dir / f"{asset_id}.{spec_alias}.h5"

        final_existed_during_extraction = [False]

        mock_mel = _generate_mock_mel()
        mock_embeddings = _generate_mock_embeddings()

        def check_final_during_compute(*args, **kwargs):
            if final_path.exists():
                final_existed_during_extraction[0] = True
            return mock_embeddings

        with mock.patch(
            "librosa.load", return_value=(np.zeros(SAMPLE_RATE * 3, dtype=np.float32), SAMPLE_RATE)
        ):
            with mock.patch(
                "services.worker_features.run._compute_mel_spectrogram", return_value=mock_mel
            ):
                with mock.patch("onnxruntime.SessionOptions"):
                    mock_session = mock.Mock()
                    mock_session.get_inputs.return_value = [
                        mock.Mock(name="input", shape=[1, 15360])
                    ]
                    mock_session.run.return_value = [np.zeros(521), mock_embeddings[0]]
                    with mock.patch("onnxruntime.InferenceSession", return_value=mock_session):
                        with mock.patch(
                            "services.worker_features.run._compute_yamnet_embeddings",
                            side_effect=check_final_during_compute,
                        ):
                            result = extract_features(session, asset_id)

        assert result.ok
        assert not final_existed_during_extraction[0]
        assert final_path.exists()

        session.close()

    def test_h5_contains_required_datasets_and_attrs(self, temp_dirs, monkeypatch):
        """Test that HDF5 contains required datasets and attributes."""
        tmpdir, _, SessionFactory, audio_dir, features_dir = temp_dirs
        session = SessionFactory()

        asset_id = "test-h5-contents"
        asset_dir = audio_dir / asset_id
        _create_test_wav(asset_dir / "normalized.wav")

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.config.FEATURES_DIR", features_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.FEATURES_DIR", features_dir)

        # Setup model verification
        model_content = b"fake model for h5 test"
        expected_hash = hashlib.sha256(model_content).hexdigest()
        yamnet_dir = Path(tmpdir) / "yamnet_onnx"
        yamnet_dir.mkdir(parents=True, exist_ok=True)
        (yamnet_dir / "yamnet.onnx").write_bytes(model_content)
        (yamnet_dir / "yamnet.onnx.sha256").write_text(expected_hash)
        monkeypatch.setattr(
            "services.worker_features.run.YAMNET_ONNX_PATH", yamnet_dir / "yamnet.onnx"
        )
        monkeypatch.setattr(
            "services.worker_features.run.YAMNET_SHA256_PATH", yamnet_dir / "yamnet.onnx.sha256"
        )

        mock_mel = _generate_mock_mel(n_frames=150)
        mock_embeddings = _generate_mock_embeddings(n_embed_frames=8)

        with mock.patch(
            "librosa.load", return_value=(np.zeros(SAMPLE_RATE * 3, dtype=np.float32), SAMPLE_RATE)
        ):
            with mock.patch(
                "services.worker_features.run._compute_mel_spectrogram", return_value=mock_mel
            ):
                with mock.patch("onnxruntime.SessionOptions"):
                    mock_session = mock.Mock()
                    mock_session.get_inputs.return_value = [
                        mock.Mock(name="input", shape=[1, 15360])
                    ]
                    mock_session.run.return_value = [np.zeros(521), mock_embeddings[0]]
                    with mock.patch("onnxruntime.InferenceSession", return_value=mock_session):
                        with mock.patch(
                            "services.worker_features.run._compute_yamnet_embeddings",
                            return_value=mock_embeddings,
                        ):
                            result = extract_features(session, asset_id)

        assert result.ok

        # Read HDF5 and verify contents
        spec_alias = feature_spec_alias(DEFAULT_FEATURE_SPEC_ID)
        h5_path = features_dir / f"{asset_id}.{spec_alias}.h5"

        with h5py.File(h5_path, "r") as f:
            # Check datasets
            assert "melspec" in f
            assert "embeddings" in f
            assert f["melspec"].shape == (150, N_MELS)
            assert f["embeddings"].shape == (8, EMBEDDING_DIM)

            # Check required attributes (Blueprint section 6)
            assert f.attrs["schema_version"] == ARTIFACT_SCHEMA_VERSION
            assert f.attrs["asset_id"] == asset_id
            assert "computed_at" in f.attrs
            assert f.attrs["feature_spec_id"] == DEFAULT_FEATURE_SPEC_ID
            assert f.attrs["feature_spec_alias"] == spec_alias
            assert f.attrs["model_id"] == "yamnet"
            assert f.attrs["model_sha256"] == expected_hash
            assert f.attrs["sample_rate"] == SAMPLE_RATE
            assert f.attrs["hop_length"] == HOP_LENGTH
            assert f.attrs["win_length"] == WIN_LENGTH
            assert f.attrs["n_fft"] == N_FFT
            assert f.attrs["n_mels"] == N_MELS
            assert f.attrs["embedding_dim"] == EMBEDDING_DIM
            assert f.attrs["embed_hop_sec"] == EMBED_HOP_SEC
            assert f.attrs["backend"] == "onnxruntime"

        session.close()


# --- Test: Metrics Merge (F) ---


class TestMetricsMerge:
    """Tests for metrics collection and merge."""

    def test_metrics_contain_required_keys(self, temp_dirs, monkeypatch):
        """Test that metrics contain all required keys."""
        tmpdir, _, SessionFactory, audio_dir, features_dir = temp_dirs
        session = SessionFactory()

        asset_id = "test-metrics"
        asset_dir = audio_dir / asset_id
        _create_test_wav(asset_dir / "normalized.wav")

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.config.FEATURES_DIR", features_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.FEATURES_DIR", features_dir)

        # Setup model verification
        model_content = b"fake model for metrics"
        expected_hash = hashlib.sha256(model_content).hexdigest()
        yamnet_dir = Path(tmpdir) / "yamnet_onnx"
        yamnet_dir.mkdir(parents=True, exist_ok=True)
        (yamnet_dir / "yamnet.onnx").write_bytes(model_content)
        (yamnet_dir / "yamnet.onnx.sha256").write_text(expected_hash)
        monkeypatch.setattr(
            "services.worker_features.run.YAMNET_ONNX_PATH", yamnet_dir / "yamnet.onnx"
        )
        monkeypatch.setattr(
            "services.worker_features.run.YAMNET_SHA256_PATH", yamnet_dir / "yamnet.onnx.sha256"
        )

        mock_mel = _generate_mock_mel(n_frames=100)
        mock_embeddings = _generate_mock_embeddings(n_embed_frames=5)

        with mock.patch(
            "librosa.load", return_value=(np.zeros(SAMPLE_RATE * 3, dtype=np.float32), SAMPLE_RATE)
        ):
            with mock.patch(
                "services.worker_features.run._compute_mel_spectrogram", return_value=mock_mel
            ):
                with mock.patch("onnxruntime.SessionOptions"):
                    mock_session = mock.Mock()
                    mock_session.get_inputs.return_value = [
                        mock.Mock(name="input", shape=[1, 15360])
                    ]
                    mock_session.run.return_value = [np.zeros(521), mock_embeddings[0]]
                    with mock.patch("onnxruntime.InferenceSession", return_value=mock_session):
                        with mock.patch(
                            "services.worker_features.run._compute_yamnet_embeddings",
                            return_value=mock_embeddings,
                        ):
                            result = extract_features(session, asset_id)

        assert result.ok
        metrics = result.metrics

        # Check required keys per Blueprint section 10
        assert "feature_spec_id" in metrics
        assert "feature_spec_alias" in metrics
        assert "mel_shape" in metrics
        assert "embedding_shape" in metrics
        assert "nan_inf_count" in metrics
        assert "feature_time_ms" in metrics

        # Check values
        assert metrics["feature_spec_id"] == DEFAULT_FEATURE_SPEC_ID
        assert metrics["feature_spec_alias"] == feature_spec_alias(DEFAULT_FEATURE_SPEC_ID)
        assert metrics["mel_shape"] == [100, N_MELS]
        assert metrics["embedding_shape"] == [5, EMBEDDING_DIM]
        assert metrics["nan_inf_count"] == 0
        assert metrics["feature_time_ms"] >= 0

        session.close()


# --- Test: OOM Mapping (G) ---


class TestOomMapping:
    """Tests for OOM error mapping."""

    def test_memory_error_maps_to_model_oom(self, temp_dirs, monkeypatch):
        """Test that MemoryError maps to MODEL_OOM."""
        tmpdir, _, SessionFactory, audio_dir, features_dir = temp_dirs
        session = SessionFactory()

        asset_id = "test-oom"
        asset_dir = audio_dir / asset_id
        _create_test_wav(asset_dir / "normalized.wav")

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.config.FEATURES_DIR", features_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.FEATURES_DIR", features_dir)

        # Setup model verification
        model_content = b"fake model for oom"
        expected_hash = hashlib.sha256(model_content).hexdigest()
        yamnet_dir = Path(tmpdir) / "yamnet_onnx"
        yamnet_dir.mkdir(parents=True, exist_ok=True)
        (yamnet_dir / "yamnet.onnx").write_bytes(model_content)
        (yamnet_dir / "yamnet.onnx.sha256").write_text(expected_hash)
        monkeypatch.setattr(
            "services.worker_features.run.YAMNET_ONNX_PATH", yamnet_dir / "yamnet.onnx"
        )
        monkeypatch.setattr(
            "services.worker_features.run.YAMNET_SHA256_PATH", yamnet_dir / "yamnet.onnx.sha256"
        )

        def raise_memory_error(*args, **kwargs):
            raise MemoryError("Out of memory during mel computation")

        with mock.patch(
            "librosa.load", return_value=(np.zeros(SAMPLE_RATE * 3, dtype=np.float32), SAMPLE_RATE)
        ):
            with mock.patch(
                "services.worker_features.run._compute_mel_spectrogram",
                side_effect=raise_memory_error,
            ):
                result = extract_features(session, asset_id)

        assert not result.ok
        assert result.error_code == FeaturesErrorCode.MODEL_OOM

        session.close()

    def test_onnx_oom_error_maps_to_model_oom(self, temp_dirs, monkeypatch):
        """Test that onnxruntime OOM-like error maps to MODEL_OOM."""
        tmpdir, _, SessionFactory, audio_dir, features_dir = temp_dirs
        session = SessionFactory()

        asset_id = "test-onnx-oom"
        asset_dir = audio_dir / asset_id
        _create_test_wav(asset_dir / "normalized.wav")

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.config.FEATURES_DIR", features_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.FEATURES_DIR", features_dir)

        # Setup model verification
        model_content = b"fake model for onnx oom"
        expected_hash = hashlib.sha256(model_content).hexdigest()
        yamnet_dir = Path(tmpdir) / "yamnet_onnx"
        yamnet_dir.mkdir(parents=True, exist_ok=True)
        (yamnet_dir / "yamnet.onnx").write_bytes(model_content)
        (yamnet_dir / "yamnet.onnx.sha256").write_text(expected_hash)
        monkeypatch.setattr(
            "services.worker_features.run.YAMNET_ONNX_PATH", yamnet_dir / "yamnet.onnx"
        )
        monkeypatch.setattr(
            "services.worker_features.run.YAMNET_SHA256_PATH", yamnet_dir / "yamnet.onnx.sha256"
        )

        mock_mel = _generate_mock_mel()

        def raise_onnx_oom(*args, **kwargs):
            raise RuntimeError("Failed to allocate memory for ONNX runtime")

        with mock.patch(
            "librosa.load", return_value=(np.zeros(SAMPLE_RATE * 3, dtype=np.float32), SAMPLE_RATE)
        ):
            with mock.patch(
                "services.worker_features.run._compute_mel_spectrogram", return_value=mock_mel
            ):
                with mock.patch("onnxruntime.SessionOptions"):
                    with mock.patch("onnxruntime.InferenceSession", side_effect=raise_onnx_oom):
                        result = extract_features(session, asset_id)

        assert not result.ok
        assert result.error_code == FeaturesErrorCode.MODEL_OOM

        session.close()

    def test_memory_error_during_audio_load_maps_to_model_oom(self, temp_dirs, monkeypatch):
        """Test that MemoryError during librosa.load maps to MODEL_OOM."""
        tmpdir, _, SessionFactory, audio_dir, features_dir = temp_dirs
        session = SessionFactory()

        asset_id = "test-audio-load-oom"
        asset_dir = audio_dir / asset_id
        _create_test_wav(asset_dir / "normalized.wav")

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.config.FEATURES_DIR", features_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.FEATURES_DIR", features_dir)

        # Setup model verification to pass
        model_content = b"fake model for audio load oom"
        expected_hash = hashlib.sha256(model_content).hexdigest()
        yamnet_dir = Path(tmpdir) / "yamnet_onnx"
        yamnet_dir.mkdir(parents=True, exist_ok=True)
        (yamnet_dir / "yamnet.onnx").write_bytes(model_content)
        (yamnet_dir / "yamnet.onnx.sha256").write_text(expected_hash)
        monkeypatch.setattr(
            "services.worker_features.run.YAMNET_ONNX_PATH", yamnet_dir / "yamnet.onnx"
        )
        monkeypatch.setattr(
            "services.worker_features.run.YAMNET_SHA256_PATH", yamnet_dir / "yamnet.onnx.sha256"
        )

        def raise_memory_error(*args, **kwargs):
            raise MemoryError("Out of memory loading audio file")

        with mock.patch("librosa.load", side_effect=raise_memory_error):
            result = extract_features(session, asset_id)

        assert not result.ok
        assert result.error_code == FeaturesErrorCode.MODEL_OOM

        # Verify no HDF5 artifact written
        spec_alias = feature_spec_alias(DEFAULT_FEATURE_SPEC_ID)
        output_path = features_dir / f"{asset_id}.{spec_alias}.h5"
        assert not output_path.exists()

        session.close()


# --- Test: Input Not Found ---


class TestInputNotFound:
    """Tests for INPUT_NOT_FOUND error code."""

    def test_missing_normalized_wav_fails(self, temp_dirs, monkeypatch):
        """Test that missing normalized.wav fails with INPUT_NOT_FOUND."""
        tmpdir, _, SessionFactory, audio_dir, features_dir = temp_dirs
        session = SessionFactory()

        asset_id = "test-missing-input"
        # Do NOT create the normalized.wav

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.config.FEATURES_DIR", features_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.FEATURES_DIR", features_dir)

        result = extract_features(session, asset_id)

        assert not result.ok
        assert result.error_code == FeaturesErrorCode.INPUT_NOT_FOUND

        session.close()


# --- Test: Idempotency ---


class TestIdempotency:
    """Tests for idempotent feature extraction."""

    def test_existing_h5_returns_success_without_extraction(self, temp_dirs, monkeypatch):
        """Test that existing .h5 returns success without re-extracting."""
        tmpdir, _, SessionFactory, audio_dir, features_dir = temp_dirs
        session = SessionFactory()

        asset_id = "test-idempotent"
        asset_dir = audio_dir / asset_id
        _create_test_wav(asset_dir / "normalized.wav")

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.config.FEATURES_DIR", features_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.FEATURES_DIR", features_dir)

        # Pre-create the HDF5 file
        spec_alias = feature_spec_alias(DEFAULT_FEATURE_SPEC_ID)
        h5_path = features_dir / f"{asset_id}.{spec_alias}.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("melspec", data=np.zeros((10, N_MELS)))
            f.create_dataset("embeddings", data=np.zeros((2, EMBEDDING_DIM)))

        extraction_called = [False]

        def track_extraction(*args, **kwargs):
            extraction_called[0] = True
            return _generate_mock_mel()

        with mock.patch(
            "services.worker_features.run._compute_mel_spectrogram",
            side_effect=track_extraction,
        ):
            result = extract_features(session, asset_id)

        assert result.ok
        assert not extraction_called[0]
        assert "already exists" in result.message.lower()

        session.close()


# --- Test: Constants Match Config ---


class TestConstantsMatchConfig:
    """Tests that worker constants match app/config.py."""

    def test_sample_rate_matches_canonical(self):
        """Test that SAMPLE_RATE matches CANONICAL_SAMPLE_RATE."""
        from app.config import CANONICAL_SAMPLE_RATE

        assert SAMPLE_RATE == CANONICAL_SAMPLE_RATE == 22050

    def test_locked_config_values(self):
        """Test locked v1.4 configuration values."""
        assert N_MELS == 64
        assert HOP_LENGTH == 220  # 10ms at 22050 Hz
        assert WIN_LENGTH == 551  # 25ms at 22050 Hz
        assert N_FFT == 1024
        assert EMBEDDING_DIM == 1024
        assert EMBED_HOP_SEC == 0.5
