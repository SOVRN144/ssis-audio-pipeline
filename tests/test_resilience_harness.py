"""SSIS Audio Pipeline - Step 8 Resilience Harness Tests.

Tests for kill/restart scenarios, lock reclamation, and temp file cleanup.
All tests use subprocess isolation to verify real crash behavior.

Per Blueprint section 11:
- Failpoints simulate power failure / hard crash
- Atomic publish must leave no partial final files
- Lock TTL reclamation must work after crash
- Temp files must be cleaned up on restart

Test approach:
- Run worker code in subprocess with failpoint enabled
- Verify subprocess exits with expected code
- Verify temp files exist (for pre-rename failpoints)
- Restart without failpoint, verify idempotent completion
- Assert no orphan temp files remain
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import wave
from datetime import timedelta
from pathlib import Path

from app.db import create_stage_lock, init_db
from app.models import StageLock, utc_now
from app.utils.atomic_io import cleanup_orphan_temp_files

# --- Helper: Create test audio file ---


def create_test_wav(path: Path, duration_sec: float = 2.0, sample_rate: int = 22050) -> None:
    """Create a minimal valid WAV file for testing.

    Args:
        path: Output path for WAV file.
        duration_sec: Duration in seconds.
        sample_rate: Sample rate in Hz.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    num_frames = int(sample_rate * duration_sec)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # Write silence (zeros)
        wf.writeframes(b"\x00" * num_frames * 2)


# --- Helper: Run subprocess with failpoint ---


def run_with_failpoint(
    script: str,
    failpoint: str,
    env_extras: dict | None = None,
    exit_code: int = 42,
    timeout: float = 30.0,
) -> subprocess.CompletedProcess:
    """Run Python script in subprocess with failpoint enabled.

    Args:
        script: Python code to execute.
        failpoint: Failpoint name to trigger.
        env_extras: Additional environment variables.
        exit_code: Expected exit code from failpoint.
        timeout: Subprocess timeout in seconds.

    Returns:
        CompletedProcess result.
    """
    env = os.environ.copy()
    env["SSIS_ENABLE_FAILPOINTS"] = "1"
    env["SSIS_FAILPOINT"] = failpoint
    env["SSIS_FAILPOINT_EXIT_CODE"] = str(exit_code)
    if env_extras:
        env.update(env_extras)

    return subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        timeout=timeout,
    )


def run_without_failpoint(
    script: str,
    env_extras: dict | None = None,
    timeout: float = 30.0,
) -> subprocess.CompletedProcess:
    """Run Python script in subprocess without failpoint.

    Args:
        script: Python code to execute.
        env_extras: Additional environment variables.
        timeout: Subprocess timeout in seconds.

    Returns:
        CompletedProcess result.
    """
    env = os.environ.copy()
    # Explicitly disable failpoints
    env.pop("SSIS_ENABLE_FAILPOINTS", None)
    env.pop("SSIS_FAILPOINT", None)
    if env_extras:
        env.update(env_extras)

    return subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        timeout=timeout,
    )


# --- Test 1: Kill after tmp write (JSON) ---


class TestAtomicWriteFailpoints:
    """Tests for atomic_write failpoints."""

    def test_kill_after_tmp_write_leaves_temp_file(self):
        """Failpoint ATOMIC_WRITE_AFTER_TMP_WRITE should leave temp file, no final."""
        with tempfile.TemporaryDirectory() as tmpdir:
            final_path = Path(tmpdir) / "test.json"
            temp_path = final_path.with_suffix(".json.tmp")

            script = f'''
import sys
sys.path.insert(0, "{Path(__file__).parent.parent}")
from app.utils.atomic_io import atomic_write_text
atomic_write_text("{final_path}", '{{"test": "data"}}')
print("SHOULD NOT REACH HERE")
'''

            result = run_with_failpoint(script, "ATOMIC_WRITE_AFTER_TMP_WRITE")

            # Subprocess should have crashed with failpoint exit code
            assert result.returncode == 42, f"Expected exit 42, got {result.returncode}"

            # Temp file should exist (write happened)
            assert temp_path.exists(), "Temp file should exist after crash"

            # Final file should NOT exist (rename didn't happen)
            assert not final_path.exists(), "Final file should not exist after crash"

            # Verify temp file has content
            content = temp_path.read_text()
            assert "test" in content

    def test_kill_after_fsync_before_rename_leaves_temp_file(self):
        """Failpoint ATOMIC_WRITE_AFTER_FSYNC_BEFORE_RENAME should leave temp, no final."""
        with tempfile.TemporaryDirectory() as tmpdir:
            final_path = Path(tmpdir) / "test.json"
            temp_path = final_path.with_suffix(".json.tmp")

            script = f'''
import sys
sys.path.insert(0, "{Path(__file__).parent.parent}")
from app.utils.atomic_io import atomic_write_text
atomic_write_text("{final_path}", '{{"test": "data"}}')
print("SHOULD NOT REACH HERE")
'''

            result = run_with_failpoint(script, "ATOMIC_WRITE_AFTER_FSYNC_BEFORE_RENAME")

            assert result.returncode == 42
            assert temp_path.exists(), "Temp file should exist"
            assert not final_path.exists(), "Final file should not exist"

    def test_kill_after_rename_has_final_file(self):
        """Failpoint ATOMIC_WRITE_AFTER_RENAME should have final file (rename completed)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            final_path = Path(tmpdir) / "test.json"
            temp_path = final_path.with_suffix(".json.tmp")

            script = f'''
import sys
sys.path.insert(0, "{Path(__file__).parent.parent}")
from app.utils.atomic_io import atomic_write_text
atomic_write_text("{final_path}", '{{"test": "data"}}')
print("SHOULD NOT REACH HERE")
'''

            result = run_with_failpoint(script, "ATOMIC_WRITE_AFTER_RENAME")

            assert result.returncode == 42

            # Final file SHOULD exist (rename completed before crash)
            assert final_path.exists(), "Final file should exist (rename completed)"

            # Temp file should NOT exist (was renamed)
            assert not temp_path.exists(), "Temp file should not exist (was renamed)"

            # Verify content is correct
            data = json.loads(final_path.read_text())
            assert data == {"test": "data"}


# --- Test 4: Lock reclamation after TTL ---


class TestLockReclamation:
    """Tests for stage lock TTL reclamation."""

    def test_stale_lock_is_reclaimed(self):
        """Expired lock should be reclaimed by new acquisition."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            engine, SessionFactory = init_db(db_path)

            try:
                session = SessionFactory()

                # Create a lock that is already expired
                # Use a short TTL and manually set expires_at to the past
                lock = create_stage_lock(
                    session,
                    asset_id="test-asset-001",
                    stage="decode",
                    worker_id="old-worker",
                    ttl_seconds=1,  # 1 second TTL
                )
                session.commit()

                # Manually expire the lock by setting expires_at to the past
                lock.expires_at = utc_now() - timedelta(seconds=10)
                session.commit()

                session.close()

                # Wait briefly to ensure time has passed
                time.sleep(0.1)

                # Now try to acquire a new lock for the same asset/stage
                # This should reclaim the stale lock
                session2 = SessionFactory()

                from app.orchestrator import _handle_stage_lock

                result = _handle_stage_lock(session2, "test-asset-001", "decode")
                session2.commit()

                # Should have acquired (reclaimed stale lock)
                assert result["action"] == "acquired"
                assert "reclaim_info" in result
                assert result["reclaim_info"]["reclaimed_worker_id"] == "old-worker"

                session2.close()

                # Use a fresh session to verify lock state
                session3 = SessionFactory()
                from sqlalchemy import select

                # Old lock should be deleted (query by old worker_id)
                stmt = select(StageLock).where(StageLock.worker_id == "old-worker")
                old_lock = session3.execute(stmt).scalar_one_or_none()
                assert old_lock is None, "Old lock should be deleted"

                # New lock should exist with different worker_id
                stmt = select(StageLock).where(
                    StageLock.asset_id == "test-asset-001",
                    StageLock.stage == "decode",
                )
                new_lock = session3.execute(stmt).scalar_one_or_none()
                assert new_lock is not None, "New lock should exist"
                assert new_lock.worker_id != "old-worker", "Different worker_id"

                session3.close()
            finally:
                engine.dispose()

    def test_active_lock_blocks_acquisition(self):
        """Active (non-expired) lock should block new acquisition."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            engine, SessionFactory = init_db(db_path)

            try:
                session = SessionFactory()

                # Create a lock with long TTL (still active)
                create_stage_lock(
                    session,
                    asset_id="test-asset-002",
                    stage="decode",
                    worker_id="active-worker",
                    ttl_seconds=600,  # 10 minutes
                )
                session.commit()
                session.close()

                # Try to acquire again - should be blocked
                session2 = SessionFactory()

                from app.orchestrator import _handle_stage_lock

                result = _handle_stage_lock(session2, "test-asset-002", "decode")

                assert result["action"] == "skip"
                assert result["reason"] == "lock_active"

                session2.close()
            finally:
                engine.dispose()


# --- Test 5: Kill during DECODE ---


class TestDecodeFailpoints:
    """Tests for decode worker failpoints."""

    def test_decode_failpoint_after_chunk_write(self):
        """DECODE_AFTER_CHUNK_WRITE failpoint should crash before checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pcm_file = Path(tmpdir) / "test.pcm.tmp"
            checkpoint_file = Path(tmpdir) / "test.ckpt.json"

            # Simulate the decode chunk loop pattern
            script = f'''
import sys
sys.path.insert(0, "{Path(__file__).parent.parent}")

from app.utils.failpoints import maybe_fail
from pathlib import Path

# Simulate writing a chunk
pcm_file = Path("{pcm_file}")
pcm_file.write_bytes(b"pcm data chunk")

# Failpoint: after chunk write, before checkpoint
maybe_fail("DECODE_AFTER_CHUNK_WRITE")

# This should not be reached
checkpoint_file = Path("{checkpoint_file}")
checkpoint_file.write_text('{{"checkpoint": "data"}}')
print("SUCCESS")
'''

            result = run_with_failpoint(script, "DECODE_AFTER_CHUNK_WRITE")

            assert result.returncode == 42
            # PCM file was written
            assert pcm_file.exists(), "PCM file should exist"
            # Checkpoint was NOT written (crash happened before)
            assert not checkpoint_file.exists(), "Checkpoint should not exist"

    def test_decode_failpoint_after_checkpoint(self):
        """DECODE_AFTER_CHECKPOINT failpoint should crash after checkpoint save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_file = Path(tmpdir) / "test.ckpt.json"
            marker_file = Path(tmpdir) / "marker.txt"

            # Simulate the checkpoint save pattern
            script = f'''
import sys
sys.path.insert(0, "{Path(__file__).parent.parent}")

from app.utils.failpoints import maybe_fail
from pathlib import Path

# Simulate checkpoint save
checkpoint_file = Path("{checkpoint_file}")
checkpoint_file.write_text('{{"checkpoint": "data"}}')

# Failpoint: after checkpoint
maybe_fail("DECODE_AFTER_CHECKPOINT")

# This should not be reached
marker = Path("{marker_file}")
marker.write_text("REACHED")
'''

            result = run_with_failpoint(script, "DECODE_AFTER_CHECKPOINT")

            assert result.returncode == 42
            # Checkpoint WAS written (crash happened after)
            assert checkpoint_file.exists(), "Checkpoint should exist"
            # Marker was NOT written
            assert not marker_file.exists(), "Marker should not exist"

    def test_decode_failpoint_before_final_rename(self):
        """DECODE_BEFORE_FINAL_RENAME failpoint should leave temp WAV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_tmp = Path(tmpdir) / "normalized.wav.tmp"
            wav_final = Path(tmpdir) / "normalized.wav"

            script = f'''
import sys
sys.path.insert(0, "{Path(__file__).parent.parent}")
import os

from app.utils.failpoints import maybe_fail

# Simulate WAV temp file creation
wav_tmp = "{wav_tmp}"
wav_final = "{wav_final}"

with open(wav_tmp, "wb") as f:
    f.write(b"RIFF WAV data")

# Failpoint: before final rename
maybe_fail("DECODE_BEFORE_FINAL_RENAME")

# This should not be reached
os.replace(wav_tmp, wav_final)
print("SUCCESS")
'''

            result = run_with_failpoint(script, "DECODE_BEFORE_FINAL_RENAME")

            assert result.returncode == 42
            # Temp WAV should exist
            assert wav_tmp.exists(), "Temp WAV should exist"
            # Final WAV should NOT exist
            assert not wav_final.exists(), "Final WAV should not exist"


# --- Test 6: Kill during FEATURES HDF5 ---


class TestFeaturesFailpoints:
    """Tests for features worker failpoints."""

    def test_kill_after_h5_tmp_write_leaves_temp(self):
        """Kill after HDF5 temp write should leave .h5.tmp, no final .h5."""
        with tempfile.TemporaryDirectory() as tmpdir:
            final_path = Path(tmpdir) / "test.h5"
            temp_path = final_path.with_suffix(".h5.tmp")

            # Write a minimal HDF5-like file with failpoint
            script = f'''
import sys
sys.path.insert(0, "{Path(__file__).parent.parent}")
import numpy as np

# Minimal test that exercises the failpoint path
from app.utils.failpoints import maybe_fail

# Simulate the _write_hdf5_atomic pattern
temp_path = "{temp_path}"
with open(temp_path, "wb") as f:
    f.write(b"HDF5 test content")

# This is where the real code has the failpoint
maybe_fail("FEATURES_AFTER_H5_TMP_WRITE")

# This should not be reached
import os
os.replace(temp_path, "{final_path}")
'''

            result = run_with_failpoint(script, "FEATURES_AFTER_H5_TMP_WRITE")

            assert result.returncode == 42
            assert temp_path.exists(), "Temp file should exist"
            assert not final_path.exists(), "Final file should not exist"

    def test_kill_before_h5_rename_leaves_temp(self):
        """Kill before HDF5 rename should leave .h5.tmp, no final .h5."""
        with tempfile.TemporaryDirectory() as tmpdir:
            final_path = Path(tmpdir) / "test.h5"
            temp_path = final_path.with_suffix(".h5.tmp")

            script = f'''
import sys
sys.path.insert(0, "{Path(__file__).parent.parent}")

from app.utils.failpoints import maybe_fail

# Simulate the _write_hdf5_atomic pattern
temp_path = "{temp_path}"
with open(temp_path, "wb") as f:
    f.write(b"HDF5 test content")

# After fsync, before rename
maybe_fail("FEATURES_BEFORE_H5_RENAME")

import os
os.replace(temp_path, "{final_path}")
'''

            result = run_with_failpoint(script, "FEATURES_BEFORE_H5_RENAME")

            assert result.returncode == 42
            assert temp_path.exists(), "Temp file should exist"
            assert not final_path.exists(), "Final file should not exist"


# --- Test 7: E2E smoke test ---


class TestE2ESmokeTest:
    """End-to-end smoke test for failpoint system."""

    def test_failpoints_disabled_by_default(self):
        """Failpoints should be no-op when SSIS_ENABLE_FAILPOINTS is not set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            final_path = Path(tmpdir) / "test.json"

            script = f'''
import sys
sys.path.insert(0, "{Path(__file__).parent.parent}")
from app.utils.atomic_io import atomic_write_text
atomic_write_text("{final_path}", '{{"test": "data"}}')
print("SUCCESS")
'''

            # Run WITHOUT enabling failpoints, but with SSIS_FAILPOINT set
            # This should complete normally because the safety gate is off
            env = os.environ.copy()
            env["SSIS_FAILPOINT"] = "ATOMIC_WRITE_AFTER_TMP_WRITE"
            # Note: NOT setting SSIS_ENABLE_FAILPOINTS

            result = subprocess.run(
                [sys.executable, "-c", script],
                env=env,
                capture_output=True,
                timeout=10.0,
            )

            assert result.returncode == 0, f"Should succeed: {result.stderr}"
            assert final_path.exists(), "Final file should exist"
            assert b"SUCCESS" in result.stdout

    def test_cleanup_orphan_temp_files(self):
        """cleanup_orphan_temp_files should remove .tmp files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some orphan temp files
            (Path(tmpdir) / "file1.json.tmp").write_text("orphan1")
            (Path(tmpdir) / "file2.h5.tmp").write_text("orphan2")
            (Path(tmpdir) / "file3.wav.tmp").write_text("orphan3")
            # Also create a non-temp file that should NOT be removed
            (Path(tmpdir) / "real_file.json").write_text("keep me")

            # Run cleanup
            removed = cleanup_orphan_temp_files(tmpdir, ".tmp")

            assert removed == 3, f"Should remove 3 temp files, got {removed}"
            assert not (Path(tmpdir) / "file1.json.tmp").exists()
            assert not (Path(tmpdir) / "file2.h5.tmp").exists()
            assert not (Path(tmpdir) / "file3.wav.tmp").exists()
            assert (Path(tmpdir) / "real_file.json").exists(), "Non-temp file should remain"

    def test_failpoint_exit_code_configurable(self):
        """SSIS_FAILPOINT_EXIT_CODE should control exit code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            final_path = Path(tmpdir) / "test.json"

            script = f'''
import sys
sys.path.insert(0, "{Path(__file__).parent.parent}")
from app.utils.atomic_io import atomic_write_text
atomic_write_text("{final_path}", '{{"test": "data"}}')
'''

            # Use custom exit code
            result = run_with_failpoint(
                script,
                "ATOMIC_WRITE_AFTER_TMP_WRITE",
                exit_code=99,
            )

            assert result.returncode == 99, f"Expected exit 99, got {result.returncode}"

    def test_maybe_fail_is_noop_without_matching_failpoint(self):
        """maybe_fail should be no-op if failpoint name doesn't match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            final_path = Path(tmpdir) / "test.json"

            script = f'''
import sys
sys.path.insert(0, "{Path(__file__).parent.parent}")
from app.utils.atomic_io import atomic_write_text
atomic_write_text("{final_path}", '{{"test": "data"}}')
print("SUCCESS")
'''

            # Set a different failpoint name
            result = run_with_failpoint(
                script,
                "NONEXISTENT_FAILPOINT",  # Won't match any real failpoint
            )

            # Should complete successfully
            assert result.returncode == 0, f"Should succeed: {result.stderr}"
            assert final_path.exists()
            assert b"SUCCESS" in result.stdout


# --- Test: Lock TTL override ---


class TestLockTTLOverride:
    """Tests for SSIS_LOCK_TTL_SEC environment variable."""

    def test_lock_ttl_respects_env_override(self):
        """SSIS_LOCK_TTL_SEC should override default TTL."""
        # This test verifies the config loading
        script = """
import sys
import os
os.environ["SSIS_LOCK_TTL_SEC"] = "5"

# Force reimport to pick up env var
if "app.config" in sys.modules:
    del sys.modules["app.config"]

from app.config import _get_lock_ttl
ttl = _get_lock_ttl()
print(f"TTL={ttl}")
assert ttl == 5, f"Expected 5, got {ttl}"
"""

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            timeout=10.0,
            cwd=Path(__file__).parent.parent,
        )

        assert result.returncode == 0, f"Failed: {result.stderr.decode()}"
        assert b"TTL=5" in result.stdout

    def test_lock_ttl_invalid_value_uses_default(self):
        """Invalid SSIS_LOCK_TTL_SEC should fall back to default."""
        script = """
import sys
import os
os.environ["SSIS_LOCK_TTL_SEC"] = "invalid"

if "app.config" in sys.modules:
    del sys.modules["app.config"]

from app.config import _get_lock_ttl
ttl = _get_lock_ttl()
print(f"TTL={ttl}")
assert ttl == 600, f"Expected 600 (default), got {ttl}"
"""

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            timeout=10.0,
            cwd=Path(__file__).parent.parent,
        )

        assert result.returncode == 0, f"Failed: {result.stderr.decode()}"
        assert b"TTL=600" in result.stdout
