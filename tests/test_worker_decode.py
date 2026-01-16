"""Tests for the decode worker (services/worker_decode/run.py).

All tests mock ffmpeg subprocess calls to run without ffmpeg installed.
"""

from __future__ import annotations

import json
import struct
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import pytest

from app.db import init_db
from app.models import AudioAsset
from app.orchestrator import ARTIFACT_TYPE_NORMALIZED_WAV
from app.utils.checkpoints import save_checkpoint
from services.worker_decode.run import (
    ARTIFACT_SCHEMA_VERSION,
    CHANNELS,
    FFMPEG_TIMEOUT_SECONDS,
    PCM_TEMP_SUFFIX,
    SAMPLE_RATE,
    SAMPWIDTH,
    DecodeErrorCode,
    decode_asset,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


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
def temp_audio_dir(temp_db):
    """Create a temporary audio directory structure."""
    tmpdir, engine, SessionFactory = temp_db
    audio_dir = Path(tmpdir) / "data" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    return tmpdir, engine, SessionFactory, audio_dir


def _generate_pcm_bytes(duration_sec: float, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Generate synthetic PCM bytes for testing.

    Creates a simple sine wave pattern.
    """
    num_samples = int(duration_sec * sample_rate)
    # Generate simple pattern (not a real sine wave, just predictable bytes)
    samples = []
    for i in range(num_samples):
        # Simple alternating pattern
        value = (i % 32768) - 16384
        samples.append(struct.pack("<h", value))
    return b"".join(samples)


def _create_test_asset(
    session: Session,
    asset_id: str,
    source_path: Path,
) -> AudioAsset:
    """Create a test AudioAsset record."""
    asset = AudioAsset(
        asset_id=asset_id,
        content_hash="test_hash_" + asset_id,
        source_uri=str(source_path),
        original_filename="test.wav",
    )
    session.add(asset)
    session.flush()
    return asset


def _create_test_source_file(path: Path) -> None:
    """Create a minimal test source file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Create a minimal valid WAV file
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(b"\x00" * 22050 * 2)  # 1 second of silence


# --- Test: Checkpoint Resume ---


class TestCheckpointResume:
    """Tests for checkpoint and resume functionality."""

    def test_resume_from_checkpoint(self, temp_audio_dir, monkeypatch):
        """Test that decode resumes from a valid checkpoint."""
        tmpdir, engine, SessionFactory, audio_dir = temp_audio_dir
        session = SessionFactory()

        # Patch config paths
        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)

        asset_id = "test-resume-001"
        asset_dir = audio_dir / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)

        # Create source file
        source_path = asset_dir / "original.wav"
        _create_test_source_file(source_path)

        # Create asset record
        _create_test_asset(session, asset_id, source_path)
        session.commit()

        # Patch paths to use our temp dir
        monkeypatch.setattr(
            "services.worker_decode.run._get_decode_paths",
            lambda aid: {
                "normalized_wav": asset_dir / "normalized.wav",
                "pcm_tmp": asset_dir / f"normalized{PCM_TEMP_SUFFIX}",
                "checkpoint": asset_dir / "normalized.ckpt.json",
                "wav_tmp": asset_dir / "normalized.wav.tmp",
            },
        )

        # Write some PCM data (simulating chunk 1)
        chunk1_duration = 30.0  # 30 seconds processed
        chunk1_pcm = _generate_pcm_bytes(chunk1_duration)
        (asset_dir / f"normalized{PCM_TEMP_SUFFIX}").write_bytes(chunk1_pcm)

        # Write checkpoint
        save_checkpoint(
            asset_dir / "normalized.ckpt.json",
            {
                "seconds_processed": chunk1_duration,
                "sample_rate": SAMPLE_RATE,
                "channels": CHANNELS,
                "sampwidth": SAMPWIDTH,
                "pcm_tmp_name": f"normalized{PCM_TEMP_SUFFIX}",
            },
        )

        # Mock ffmpeg to return chunk 2 data then empty (EOF)
        chunk2_duration = 2.5  # 2.5 more seconds
        chunk2_pcm = _generate_pcm_bytes(chunk2_duration)

        call_count = [0]

        def mock_run(cmd, capture_output=True, check=False, timeout=None):
            call_count[0] += 1
            result = mock.Mock()
            result.returncode = 0

            # Check that -ss is set to checkpoint position
            if "-ss" in cmd:
                ss_idx = cmd.index("-ss")
                ss_value = float(cmd[ss_idx + 1])
                if call_count[0] == 1:
                    # First call after resume should start at checkpoint
                    assert ss_value == pytest.approx(chunk1_duration, abs=1.0)
                    result.stdout = chunk2_pcm
                else:
                    # Subsequent calls return empty (EOF)
                    result.stdout = b""
            else:
                result.stdout = b""

            return result

        with mock.patch("subprocess.run", side_effect=mock_run):
            result = decode_asset(session, asset_id)

        assert result.ok
        assert call_count[0] >= 1  # At least one ffmpeg call
        # Verify final WAV exists
        assert (asset_dir / "normalized.wav").exists()
        # Verify temps cleaned up
        assert not (asset_dir / f"normalized{PCM_TEMP_SUFFIX}").exists()
        assert not (asset_dir / "normalized.ckpt.json").exists()

        session.close()

    def test_checkpoint_saved_after_each_chunk(self, temp_audio_dir, monkeypatch):
        """Test that checkpoint is saved after each chunk."""
        tmpdir, engine, SessionFactory, audio_dir = temp_audio_dir
        session = SessionFactory()

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)

        asset_id = "test-checkpoint-001"
        asset_dir = audio_dir / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)

        source_path = asset_dir / "original.wav"
        _create_test_source_file(source_path)
        _create_test_asset(session, asset_id, source_path)
        session.commit()

        monkeypatch.setattr(
            "services.worker_decode.run._get_decode_paths",
            lambda aid: {
                "normalized_wav": asset_dir / "normalized.wav",
                "pcm_tmp": asset_dir / f"normalized{PCM_TEMP_SUFFIX}",
                "checkpoint": asset_dir / "normalized.ckpt.json",
                "wav_tmp": asset_dir / "normalized.wav.tmp",
            },
        )

        # Track checkpoint saves
        checkpoint_saves = []

        original_save = save_checkpoint

        def track_save(path, data):
            checkpoint_saves.append(data.copy())
            return original_save(path, data)

        # Generate 2.5 seconds of audio (single chunk)
        chunk_pcm = _generate_pcm_bytes(2.5)

        def mock_run(cmd, capture_output=True, check=False, timeout=None):
            result = mock.Mock()
            result.returncode = 0

            if "-ss" in cmd:
                ss_idx = cmd.index("-ss")
                ss_value = float(cmd[ss_idx + 1])
                if ss_value == 0:
                    result.stdout = chunk_pcm
                else:
                    result.stdout = b""
            else:
                result.stdout = chunk_pcm

            return result

        with mock.patch("subprocess.run", side_effect=mock_run):
            with mock.patch("services.worker_decode.run.save_checkpoint", side_effect=track_save):
                result = decode_asset(session, asset_id)

        assert result.ok
        assert len(checkpoint_saves) >= 1  # At least one checkpoint save
        # First checkpoint should have seconds_processed > 0
        assert checkpoint_saves[0]["seconds_processed"] > 0

        session.close()


# --- Test: Atomic Publish Correctness ---


class TestAtomicPublish:
    """Tests for atomic publish behavior."""

    def test_tmp_not_treated_as_final(self, temp_audio_dir, monkeypatch):
        """Test that .tmp files are not treated as final artifacts."""
        tmpdir, engine, SessionFactory, audio_dir = temp_audio_dir
        session = SessionFactory()

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)

        asset_id = "test-atomic-001"
        asset_dir = audio_dir / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)

        source_path = asset_dir / "original.wav"
        _create_test_source_file(source_path)
        _create_test_asset(session, asset_id, source_path)
        session.commit()

        monkeypatch.setattr(
            "services.worker_decode.run._get_decode_paths",
            lambda aid: {
                "normalized_wav": asset_dir / "normalized.wav",
                "pcm_tmp": asset_dir / f"normalized{PCM_TEMP_SUFFIX}",
                "checkpoint": asset_dir / "normalized.ckpt.json",
                "wav_tmp": asset_dir / "normalized.wav.tmp",
            },
        )

        # Create orphan .wav.tmp (simulating interrupted previous run)
        (asset_dir / "normalized.wav.tmp").write_bytes(b"orphan data")

        chunk_pcm = _generate_pcm_bytes(2.5)

        def mock_run(cmd, capture_output=True, check=False, timeout=None):
            result = mock.Mock()
            result.returncode = 0
            if "-ss" in cmd:
                ss_idx = cmd.index("-ss")
                ss_value = float(cmd[ss_idx + 1])
                if ss_value == 0:
                    result.stdout = chunk_pcm
                else:
                    result.stdout = b""
            return result

        with mock.patch("subprocess.run", side_effect=mock_run):
            result = decode_asset(session, asset_id)

        assert result.ok
        # Final WAV should exist
        assert (asset_dir / "normalized.wav").exists()
        # Temp should not exist
        assert not (asset_dir / "normalized.wav.tmp").exists()

        session.close()

    def test_final_wav_has_correct_format(self, temp_audio_dir, monkeypatch):
        """Test that final WAV has correct format: 22050 Hz, mono, 16-bit."""
        tmpdir, engine, SessionFactory, audio_dir = temp_audio_dir
        session = SessionFactory()

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)

        asset_id = "test-format-001"
        asset_dir = audio_dir / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)

        source_path = asset_dir / "original.wav"
        _create_test_source_file(source_path)
        _create_test_asset(session, asset_id, source_path)
        session.commit()

        monkeypatch.setattr(
            "services.worker_decode.run._get_decode_paths",
            lambda aid: {
                "normalized_wav": asset_dir / "normalized.wav",
                "pcm_tmp": asset_dir / f"normalized{PCM_TEMP_SUFFIX}",
                "checkpoint": asset_dir / "normalized.ckpt.json",
                "wav_tmp": asset_dir / "normalized.wav.tmp",
            },
        )

        chunk_pcm = _generate_pcm_bytes(3.0)

        def mock_run(cmd, capture_output=True, check=False, timeout=None):
            result = mock.Mock()
            result.returncode = 0
            if "-ss" in cmd:
                ss_idx = cmd.index("-ss")
                ss_value = float(cmd[ss_idx + 1])
                if ss_value == 0:
                    result.stdout = chunk_pcm
                else:
                    result.stdout = b""
            return result

        with mock.patch("subprocess.run", side_effect=mock_run):
            result = decode_asset(session, asset_id)

        assert result.ok

        # Read final WAV and verify format
        wav_path = asset_dir / "normalized.wav"
        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getframerate() == 22050
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2

        session.close()

    def test_final_appears_only_after_completion(self, temp_audio_dir, monkeypatch):
        """Test that normalized.wav only appears after successful completion."""
        tmpdir, engine, SessionFactory, audio_dir = temp_audio_dir
        session = SessionFactory()

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)

        asset_id = "test-completion-001"
        asset_dir = audio_dir / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)

        source_path = asset_dir / "original.wav"
        _create_test_source_file(source_path)
        _create_test_asset(session, asset_id, source_path)
        session.commit()

        monkeypatch.setattr(
            "services.worker_decode.run._get_decode_paths",
            lambda aid: {
                "normalized_wav": asset_dir / "normalized.wav",
                "pcm_tmp": asset_dir / f"normalized{PCM_TEMP_SUFFIX}",
                "checkpoint": asset_dir / "normalized.ckpt.json",
                "wav_tmp": asset_dir / "normalized.wav.tmp",
            },
        )

        final_wav_existed_during_decode = [False]

        chunk_pcm = _generate_pcm_bytes(2.5)

        def mock_run(cmd, capture_output=True, check=False, timeout=None):
            # Check if final exists during decode
            if (asset_dir / "normalized.wav").exists():
                final_wav_existed_during_decode[0] = True

            result = mock.Mock()
            result.returncode = 0
            if "-ss" in cmd:
                ss_idx = cmd.index("-ss")
                ss_value = float(cmd[ss_idx + 1])
                if ss_value == 0:
                    result.stdout = chunk_pcm
                else:
                    result.stdout = b""
            return result

        with mock.patch("subprocess.run", side_effect=mock_run):
            result = decode_asset(session, asset_id)

        assert result.ok
        assert not final_wav_existed_during_decode[0]
        assert (asset_dir / "normalized.wav").exists()

        session.close()


# --- Test: Error Mapping ---


class TestErrorMapping:
    """Tests for error code mapping from ffmpeg."""

    def test_codec_unsupported_on_decoder_error(self, temp_audio_dir, monkeypatch):
        """Test CODEC_UNSUPPORTED error for decoder/format issues."""
        tmpdir, engine, SessionFactory, audio_dir = temp_audio_dir
        session = SessionFactory()

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)

        asset_id = "test-codec-001"
        asset_dir = audio_dir / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)

        source_path = asset_dir / "original.wav"
        _create_test_source_file(source_path)
        _create_test_asset(session, asset_id, source_path)
        session.commit()

        monkeypatch.setattr(
            "services.worker_decode.run._get_decode_paths",
            lambda aid: {
                "normalized_wav": asset_dir / "normalized.wav",
                "pcm_tmp": asset_dir / f"normalized{PCM_TEMP_SUFFIX}",
                "checkpoint": asset_dir / "normalized.ckpt.json",
                "wav_tmp": asset_dir / "normalized.wav.tmp",
            },
        )

        def mock_run(cmd, capture_output=True, check=False, timeout=None):
            result = mock.Mock()
            result.returncode = 1
            result.stdout = b""
            result.stderr = b"Decoder not found for codec xyz"
            return result

        with mock.patch("subprocess.run", side_effect=mock_run):
            result = decode_asset(session, asset_id)

        assert not result.ok
        assert result.error_code == DecodeErrorCode.CODEC_UNSUPPORTED

        session.close()

    def test_file_corrupt_on_other_nonzero(self, temp_audio_dir, monkeypatch):
        """Test FILE_CORRUPT error for other nonzero return codes."""
        tmpdir, engine, SessionFactory, audio_dir = temp_audio_dir
        session = SessionFactory()

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)

        asset_id = "test-corrupt-001"
        asset_dir = audio_dir / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)

        source_path = asset_dir / "original.wav"
        _create_test_source_file(source_path)
        _create_test_asset(session, asset_id, source_path)
        session.commit()

        monkeypatch.setattr(
            "services.worker_decode.run._get_decode_paths",
            lambda aid: {
                "normalized_wav": asset_dir / "normalized.wav",
                "pcm_tmp": asset_dir / f"normalized{PCM_TEMP_SUFFIX}",
                "checkpoint": asset_dir / "normalized.ckpt.json",
                "wav_tmp": asset_dir / "normalized.wav.tmp",
            },
        )

        def mock_run(cmd, capture_output=True, check=False, timeout=None):
            result = mock.Mock()
            result.returncode = 1
            result.stdout = b""
            result.stderr = b"Some random error"
            return result

        with mock.patch("subprocess.run", side_effect=mock_run):
            result = decode_asset(session, asset_id)

        assert not result.ok
        assert result.error_code == DecodeErrorCode.FILE_CORRUPT

        session.close()

    def test_file_corrupt_on_empty_first_chunk(self, temp_audio_dir, monkeypatch):
        """Test FILE_CORRUPT error when first chunk returns empty."""
        tmpdir, engine, SessionFactory, audio_dir = temp_audio_dir
        session = SessionFactory()

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)

        asset_id = "test-empty-001"
        asset_dir = audio_dir / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)

        source_path = asset_dir / "original.wav"
        _create_test_source_file(source_path)
        _create_test_asset(session, asset_id, source_path)
        session.commit()

        monkeypatch.setattr(
            "services.worker_decode.run._get_decode_paths",
            lambda aid: {
                "normalized_wav": asset_dir / "normalized.wav",
                "pcm_tmp": asset_dir / f"normalized{PCM_TEMP_SUFFIX}",
                "checkpoint": asset_dir / "normalized.ckpt.json",
                "wav_tmp": asset_dir / "normalized.wav.tmp",
            },
        )

        def mock_run(cmd, capture_output=True, check=False, timeout=None):
            result = mock.Mock()
            result.returncode = 0
            result.stdout = b""  # Empty output on first chunk
            return result

        with mock.patch("subprocess.run", side_effect=mock_run):
            result = decode_asset(session, asset_id)

        assert not result.ok
        assert result.error_code == DecodeErrorCode.FILE_CORRUPT

        session.close()

    def test_file_too_short_under_minimum_duration(self, temp_audio_dir, monkeypatch):
        """Test FILE_TOO_SHORT error when output is below MIN_DURATION_SEC."""
        tmpdir, engine, SessionFactory, audio_dir = temp_audio_dir
        session = SessionFactory()

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)

        asset_id = "test-short-001"
        asset_dir = audio_dir / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)

        source_path = asset_dir / "original.wav"
        _create_test_source_file(source_path)
        _create_test_asset(session, asset_id, source_path)
        session.commit()

        monkeypatch.setattr(
            "services.worker_decode.run._get_decode_paths",
            lambda aid: {
                "normalized_wav": asset_dir / "normalized.wav",
                "pcm_tmp": asset_dir / f"normalized{PCM_TEMP_SUFFIX}",
                "checkpoint": asset_dir / "normalized.ckpt.json",
                "wav_tmp": asset_dir / "normalized.wav.tmp",
            },
        )

        # Generate only 1 second of audio (below MIN_DURATION_SEC = 1.7)
        short_pcm = _generate_pcm_bytes(1.0)

        def mock_run(cmd, capture_output=True, check=False, timeout=None):
            result = mock.Mock()
            result.returncode = 0
            if "-ss" in cmd:
                ss_idx = cmd.index("-ss")
                ss_value = float(cmd[ss_idx + 1])
                if ss_value == 0:
                    result.stdout = short_pcm
                else:
                    result.stdout = b""
            return result

        with mock.patch("subprocess.run", side_effect=mock_run):
            result = decode_asset(session, asset_id)

        assert not result.ok
        assert result.error_code == DecodeErrorCode.FILE_TOO_SHORT

        session.close()

    def test_input_not_found_missing_asset(self, temp_audio_dir, monkeypatch):
        """Test INPUT_NOT_FOUND when asset record does not exist."""
        tmpdir, engine, SessionFactory, audio_dir = temp_audio_dir
        session = SessionFactory()

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)

        asset_id = "nonexistent-asset"
        asset_dir = audio_dir / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(
            "services.worker_decode.run._get_decode_paths",
            lambda aid: {
                "normalized_wav": asset_dir / "normalized.wav",
                "pcm_tmp": asset_dir / f"normalized{PCM_TEMP_SUFFIX}",
                "checkpoint": asset_dir / "normalized.ckpt.json",
                "wav_tmp": asset_dir / "normalized.wav.tmp",
            },
        )

        result = decode_asset(session, asset_id)

        assert not result.ok
        assert result.error_code == DecodeErrorCode.INPUT_NOT_FOUND

        session.close()

    def test_input_not_found_missing_source_file(self, temp_audio_dir, monkeypatch):
        """Test INPUT_NOT_FOUND when source file does not exist."""
        tmpdir, engine, SessionFactory, audio_dir = temp_audio_dir
        session = SessionFactory()

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)

        asset_id = "test-missing-source"
        asset_dir = audio_dir / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)

        # Create asset pointing to non-existent file
        source_path = asset_dir / "nonexistent.wav"
        _create_test_asset(session, asset_id, source_path)
        session.commit()

        monkeypatch.setattr(
            "services.worker_decode.run._get_decode_paths",
            lambda aid: {
                "normalized_wav": asset_dir / "normalized.wav",
                "pcm_tmp": asset_dir / f"normalized{PCM_TEMP_SUFFIX}",
                "checkpoint": asset_dir / "normalized.ckpt.json",
                "wav_tmp": asset_dir / "normalized.wav.tmp",
            },
        )

        result = decode_asset(session, asset_id)

        assert not result.ok
        assert result.error_code == DecodeErrorCode.INPUT_NOT_FOUND

        session.close()


# --- Test: Metrics Written ---


class TestMetricsWritten:
    """Tests for decode metrics collection."""

    def test_metrics_contain_required_keys(self, temp_audio_dir, monkeypatch):
        """Test that metrics contain all required keys."""
        tmpdir, engine, SessionFactory, audio_dir = temp_audio_dir
        session = SessionFactory()

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)

        asset_id = "test-metrics-001"
        asset_dir = audio_dir / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)

        source_path = asset_dir / "original.wav"
        _create_test_source_file(source_path)
        _create_test_asset(session, asset_id, source_path)
        session.commit()

        monkeypatch.setattr(
            "services.worker_decode.run._get_decode_paths",
            lambda aid: {
                "normalized_wav": asset_dir / "normalized.wav",
                "pcm_tmp": asset_dir / f"normalized{PCM_TEMP_SUFFIX}",
                "checkpoint": asset_dir / "normalized.ckpt.json",
                "wav_tmp": asset_dir / "normalized.wav.tmp",
            },
        )

        chunk_pcm = _generate_pcm_bytes(3.0)

        def mock_run(cmd, capture_output=True, check=False, timeout=None):
            result = mock.Mock()
            result.returncode = 0
            if "-ss" in cmd:
                ss_idx = cmd.index("-ss")
                ss_value = float(cmd[ss_idx + 1])
                if ss_value == 0:
                    result.stdout = chunk_pcm
                else:
                    result.stdout = b""
            return result

        with mock.patch("subprocess.run", side_effect=mock_run):
            result = decode_asset(session, asset_id)

        assert result.ok
        assert result.metrics.output_duration_sec > 0
        assert result.metrics.chunk_count >= 1
        assert result.metrics.decode_time_ms >= 0

        session.close()


# --- Test: Orchestrator Integration ---


class TestOrchestratorIntegration:
    """Tests for orchestrator integration."""

    def test_success_triggers_artifact_recording(self, temp_audio_dir, monkeypatch):
        """Test that success returns correct artifact info for recording."""
        tmpdir, engine, SessionFactory, audio_dir = temp_audio_dir
        session = SessionFactory()

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)

        asset_id = "test-artifact-001"
        asset_dir = audio_dir / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)

        source_path = asset_dir / "original.wav"
        _create_test_source_file(source_path)
        _create_test_asset(session, asset_id, source_path)
        session.commit()

        monkeypatch.setattr(
            "services.worker_decode.run._get_decode_paths",
            lambda aid: {
                "normalized_wav": asset_dir / "normalized.wav",
                "pcm_tmp": asset_dir / f"normalized{PCM_TEMP_SUFFIX}",
                "checkpoint": asset_dir / "normalized.ckpt.json",
                "wav_tmp": asset_dir / "normalized.wav.tmp",
            },
        )

        chunk_pcm = _generate_pcm_bytes(2.5)

        def mock_run(cmd, capture_output=True, check=False, timeout=None):
            result = mock.Mock()
            result.returncode = 0
            if "-ss" in cmd:
                ss_idx = cmd.index("-ss")
                ss_value = float(cmd[ss_idx + 1])
                if ss_value == 0:
                    result.stdout = chunk_pcm
                else:
                    result.stdout = b""
            return result

        with mock.patch("subprocess.run", side_effect=mock_run):
            result = decode_asset(session, asset_id)

        assert result.ok
        assert result.artifact_type == ARTIFACT_TYPE_NORMALIZED_WAV
        assert result.schema_version == ARTIFACT_SCHEMA_VERSION
        assert result.artifact_path is not None
        assert "normalized.wav" in result.artifact_path

        session.close()

    def test_failure_returns_correct_error_code(self, temp_audio_dir, monkeypatch):
        """Test that failure returns correct error_code for orchestrator."""
        tmpdir, engine, SessionFactory, audio_dir = temp_audio_dir
        session = SessionFactory()

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)

        asset_id = "test-error-001"
        asset_dir = audio_dir / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)

        source_path = asset_dir / "original.wav"
        _create_test_source_file(source_path)
        _create_test_asset(session, asset_id, source_path)
        session.commit()

        monkeypatch.setattr(
            "services.worker_decode.run._get_decode_paths",
            lambda aid: {
                "normalized_wav": asset_dir / "normalized.wav",
                "pcm_tmp": asset_dir / f"normalized{PCM_TEMP_SUFFIX}",
                "checkpoint": asset_dir / "normalized.ckpt.json",
                "wav_tmp": asset_dir / "normalized.wav.tmp",
            },
        )

        def mock_run(cmd, capture_output=True, check=False, timeout=None):
            result = mock.Mock()
            result.returncode = 1
            result.stdout = b""
            result.stderr = b"Unknown format specified"
            return result

        with mock.patch("subprocess.run", side_effect=mock_run):
            result = decode_asset(session, asset_id)

        assert not result.ok
        assert result.error_code == DecodeErrorCode.CODEC_UNSUPPORTED
        assert result.message is not None

        session.close()


# --- Test: Idempotency ---


class TestIdempotency:
    """Tests for idempotent decode behavior."""

    def test_existing_wav_returns_success_without_decode(self, temp_audio_dir, monkeypatch):
        """Test that existing normalized.wav returns success without decoding."""
        tmpdir, engine, SessionFactory, audio_dir = temp_audio_dir
        session = SessionFactory()

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)

        asset_id = "test-idempotent-001"
        asset_dir = audio_dir / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)

        source_path = asset_dir / "original.wav"
        _create_test_source_file(source_path)
        _create_test_asset(session, asset_id, source_path)
        session.commit()

        # Pre-create normalized.wav
        wav_path = asset_dir / "normalized.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(b"\x00" * 22050 * 4)  # 2 seconds

        monkeypatch.setattr(
            "services.worker_decode.run._get_decode_paths",
            lambda aid: {
                "normalized_wav": wav_path,
                "pcm_tmp": asset_dir / f"normalized{PCM_TEMP_SUFFIX}",
                "checkpoint": asset_dir / "normalized.ckpt.json",
                "wav_tmp": asset_dir / "normalized.wav.tmp",
            },
        )

        ffmpeg_called = [False]

        def mock_run(cmd, capture_output=True, check=False, timeout=None):
            ffmpeg_called[0] = True
            result = mock.Mock()
            result.returncode = 0
            result.stdout = b""
            return result

        with mock.patch("subprocess.run", side_effect=mock_run):
            result = decode_asset(session, asset_id)

        assert result.ok
        assert not ffmpeg_called[0]  # ffmpeg should not be called
        assert "already exists" in result.message.lower()

        session.close()


# --- Test: Checkpoint Validation ---


class TestCheckpointValidation:
    """Tests for checkpoint validation and handling."""

    def test_invalid_checkpoint_restarts_decode(self, temp_audio_dir, monkeypatch):
        """Test that invalid checkpoint causes restart from beginning."""
        tmpdir, engine, SessionFactory, audio_dir = temp_audio_dir
        session = SessionFactory()

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)

        asset_id = "test-invalid-ckpt-001"
        asset_dir = audio_dir / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)

        source_path = asset_dir / "original.wav"
        _create_test_source_file(source_path)
        _create_test_asset(session, asset_id, source_path)
        session.commit()

        monkeypatch.setattr(
            "services.worker_decode.run._get_decode_paths",
            lambda aid: {
                "normalized_wav": asset_dir / "normalized.wav",
                "pcm_tmp": asset_dir / f"normalized{PCM_TEMP_SUFFIX}",
                "checkpoint": asset_dir / "normalized.ckpt.json",
                "wav_tmp": asset_dir / "normalized.wav.tmp",
            },
        )

        # Create PCM tmp with some data
        (asset_dir / f"normalized{PCM_TEMP_SUFFIX}").write_bytes(b"old pcm data")

        # Create invalid checkpoint (wrong schema version)
        ckpt_path = asset_dir / "normalized.ckpt.json"
        with open(ckpt_path, "w") as f:
            json.dump({"schema_version": "0.0.1", "seconds_processed": 30.0}, f)

        chunk_pcm = _generate_pcm_bytes(2.5)
        call_positions = []

        def mock_run(cmd, capture_output=True, check=False, timeout=None):
            if "-ss" in cmd:
                ss_idx = cmd.index("-ss")
                ss_value = float(cmd[ss_idx + 1])
                call_positions.append(ss_value)

            result = mock.Mock()
            result.returncode = 0
            if "-ss" in cmd and float(cmd[cmd.index("-ss") + 1]) == 0:
                result.stdout = chunk_pcm
            else:
                result.stdout = b""
            return result

        with mock.patch("subprocess.run", side_effect=mock_run):
            result = decode_asset(session, asset_id)

        assert result.ok
        # Should have started from 0 (not from checkpoint)
        assert len(call_positions) >= 1
        assert call_positions[0] == 0

        session.close()

    def test_pcm_without_checkpoint_restarts(self, temp_audio_dir, monkeypatch):
        """Test that PCM temp without checkpoint causes restart."""
        tmpdir, engine, SessionFactory, audio_dir = temp_audio_dir
        session = SessionFactory()

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)

        asset_id = "test-pcm-no-ckpt-001"
        asset_dir = audio_dir / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)

        source_path = asset_dir / "original.wav"
        _create_test_source_file(source_path)
        _create_test_asset(session, asset_id, source_path)
        session.commit()

        monkeypatch.setattr(
            "services.worker_decode.run._get_decode_paths",
            lambda aid: {
                "normalized_wav": asset_dir / "normalized.wav",
                "pcm_tmp": asset_dir / f"normalized{PCM_TEMP_SUFFIX}",
                "checkpoint": asset_dir / "normalized.ckpt.json",
                "wav_tmp": asset_dir / "normalized.wav.tmp",
            },
        )

        # Create PCM tmp without checkpoint (simulating crash mid-chunk)
        (asset_dir / f"normalized{PCM_TEMP_SUFFIX}").write_bytes(b"orphan pcm data")

        chunk_pcm = _generate_pcm_bytes(2.5)
        call_positions = []

        def mock_run(cmd, capture_output=True, check=False, timeout=None):
            if "-ss" in cmd:
                ss_idx = cmd.index("-ss")
                ss_value = float(cmd[ss_idx + 1])
                call_positions.append(ss_value)

            result = mock.Mock()
            result.returncode = 0
            if "-ss" in cmd and float(cmd[cmd.index("-ss") + 1]) == 0:
                result.stdout = chunk_pcm
            else:
                result.stdout = b""
            return result

        with mock.patch("subprocess.run", side_effect=mock_run):
            result = decode_asset(session, asset_id)

        assert result.ok
        # Should have started from 0
        assert len(call_positions) >= 1
        assert call_positions[0] == 0

        session.close()


# --- Test: Timeout Handling ---


class TestTimeoutHandling:
    """Tests for ffmpeg subprocess timeout handling."""

    def test_timeout_returns_worker_error(self, temp_audio_dir, monkeypatch):
        """Test that TimeoutExpired raises WORKER_ERROR and no artifact is published."""
        tmpdir, engine, SessionFactory, audio_dir = temp_audio_dir
        session = SessionFactory()

        monkeypatch.setattr("app.config.AUDIO_DIR", audio_dir)
        monkeypatch.setattr("app.utils.paths.AUDIO_DIR", audio_dir)

        asset_id = "test-timeout-001"
        asset_dir = audio_dir / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)

        source_path = asset_dir / "original.wav"
        _create_test_source_file(source_path)
        _create_test_asset(session, asset_id, source_path)
        session.commit()

        monkeypatch.setattr(
            "services.worker_decode.run._get_decode_paths",
            lambda aid: {
                "normalized_wav": asset_dir / "normalized.wav",
                "pcm_tmp": asset_dir / f"normalized{PCM_TEMP_SUFFIX}",
                "checkpoint": asset_dir / "normalized.ckpt.json",
                "wav_tmp": asset_dir / "normalized.wav.tmp",
            },
        )

        def mock_run(cmd, capture_output=True, check=False, timeout=None):
            # Simulate ffmpeg hanging and timing out
            raise subprocess.TimeoutExpired(cmd, timeout or FFMPEG_TIMEOUT_SECONDS)

        with mock.patch("subprocess.run", side_effect=mock_run):
            result = decode_asset(session, asset_id)

        # Should return failure with WORKER_ERROR
        assert not result.ok
        assert result.error_code == DecodeErrorCode.WORKER_ERROR

        # No final artifact should exist
        assert not (asset_dir / "normalized.wav").exists()

        # Temp files should be cleaned up
        assert not (asset_dir / f"normalized{PCM_TEMP_SUFFIX}").exists()
        assert not (asset_dir / "normalized.ckpt.json").exists()

        session.close()

    def test_timeout_constant_below_lock_ttl(self):
        """Verify FFMPEG_TIMEOUT_SECONDS is below the 600s lock TTL."""
        # Lock TTL is ~600 seconds per Blueprint; timeout must be less
        # to allow orchestrator retry on timeout
        assert FFMPEG_TIMEOUT_SECONDS < 600
        assert FFMPEG_TIMEOUT_SECONDS == 300  # 5 minutes as specified
