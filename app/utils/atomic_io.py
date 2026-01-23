"""SSIS Audio Pipeline - Atomic I/O utilities.

Implements atomic publish rule per Blueprint section 4:
1. Write to temp path in same directory
2. Flush + best-effort fsync
3. Rename temp -> final (the publish boundary)

This ensures that the final path either contains complete valid data
or does not exist. Partial writes only affect the temp file.

Failpoints (Step 8 resilience harness):
- ATOMIC_WRITE_AFTER_TMP_WRITE: After writing to temp file, before fsync
- ATOMIC_WRITE_AFTER_FSYNC_BEFORE_RENAME: After fsync, before atomic rename
- ATOMIC_WRITE_AFTER_RENAME: After atomic rename completes
"""

import os
from pathlib import Path

from app.utils.failpoints import maybe_fail


def _write_all(fd: int, data: bytes) -> None:
    """Write all bytes to a file descriptor, handling partial writes.

    Loops until all bytes are written, handling short writes and EINTR.
    This prevents silent data truncation when os.write() doesn't write
    the full buffer in one call (which can happen on pipes, sockets,
    or under heavy I/O load).

    Args:
        fd: File descriptor to write to.
        data: Bytes to write.

    Raises:
        OSError: If write fails or returns 0 bytes unexpectedly.
    """
    total_written = 0
    data_len = len(data)

    while total_written < data_len:
        try:
            written = os.write(fd, data[total_written:])
            if written == 0:
                # os.write() should never return 0 for non-empty data
                raise OSError("os.write() returned 0 bytes unexpectedly")
            total_written += written
        except InterruptedError:
            # EINTR: interrupted system call, retry the write
            continue


def atomic_write_bytes(
    final_path: str | Path,
    data: bytes,
    temp_suffix: str = ".tmp",
) -> None:
    """Atomically write bytes to a file.

    Idempotent: safe to call even if temp file exists (overwrites temp).
    Never corrupts final path - atomic rename ensures all-or-nothing.

    Args:
        final_path: The target path for the final file.
        data: Bytes to write.
        temp_suffix: Suffix for the temporary file (default: ".tmp").

    Raises:
        OSError: If directory creation, write, or rename fails.
    """
    final_path = Path(final_path)
    temp_path = final_path.with_suffix(final_path.suffix + temp_suffix)

    # Ensure parent directory exists
    final_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file with cleanup on failure
    fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        _write_all(fd, data)

        # Failpoint: after temp write, before fsync
        maybe_fail("ATOMIC_WRITE_AFTER_TMP_WRITE")

        # Flush to OS buffers
        os.fsync(fd)
    except OSError:
        # Close fd and cleanup orphan temp file on write/fsync failure
        os.close(fd)
        try:
            os.remove(temp_path)
        except OSError:
            pass  # Best-effort cleanup
        raise
    else:
        os.close(fd)

    # Best-effort fsync on directory for rename durability
    _fsync_directory(final_path.parent)

    # Failpoint: after fsync, before rename
    maybe_fail("ATOMIC_WRITE_AFTER_FSYNC_BEFORE_RENAME")

    # Atomic rename (POSIX guarantees atomicity)
    os.replace(temp_path, final_path)

    # Failpoint: after rename (for verifying completed writes)
    maybe_fail("ATOMIC_WRITE_AFTER_RENAME")


def atomic_write_text(
    final_path: str | Path,
    text: str,
    encoding: str = "utf-8",
    temp_suffix: str = ".tmp",
) -> None:
    """Atomically write text to a file.

    Idempotent: safe to call even if temp file exists (overwrites temp).
    Never corrupts final path - atomic rename ensures all-or-nothing.

    Args:
        final_path: The target path for the final file.
        text: Text string to write.
        encoding: Text encoding (default: utf-8).
        temp_suffix: Suffix for the temporary file (default: ".tmp").

    Raises:
        OSError: If directory creation, write, or rename fails.
    """
    atomic_write_bytes(final_path, text.encode(encoding), temp_suffix)


def _fsync_directory(dir_path: Path) -> None:
    """Best-effort fsync on a directory.

    This helps ensure rename durability on some filesystems.
    Silently ignores errors as this is best-effort.

    Args:
        dir_path: Directory path to sync.
    """
    try:
        fd = os.open(dir_path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except (OSError, AttributeError):
        # O_DIRECTORY may not be available on all platforms
        # or fsync on directory may fail - this is best-effort
        pass


def atomic_copy_file(
    source_path: str | Path,
    final_path: str | Path,
    temp_suffix: str = ".tmp",
    chunk_size: int = 65536,
) -> None:
    """Atomically copy a file from source to destination.

    Implements atomic publish semantics per Blueprint section 4:
    1. Copy to temp file in same directory as final
    2. Flush + best-effort fsync
    3. Rename temp -> final (atomic publish boundary)
    4. Best-effort fsync directory

    On failure, orphan temp files are cleaned up.

    Args:
        source_path: Path to the source file.
        final_path: Target path for the copied file.
        temp_suffix: Suffix for the temporary file (default: ".tmp").
        chunk_size: Buffer size for copying (default: 64KB).

    Raises:
        FileNotFoundError: If source file does not exist.
        OSError: If copy or rename fails.
    """
    source_path = Path(source_path)
    final_path = Path(final_path)
    temp_path = final_path.with_suffix(final_path.suffix + temp_suffix)

    # Verify source exists (raises FileNotFoundError if not)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    # Ensure parent directory exists
    final_path.parent.mkdir(parents=True, exist_ok=True)

    # Open source for reading
    src_fd = os.open(source_path, os.O_RDONLY)
    try:
        # Open temp file for writing
        dst_fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            # Copy in chunks
            while True:
                chunk = os.read(src_fd, chunk_size)
                if not chunk:
                    break
                _write_all(dst_fd, chunk)

            # Flush to OS buffers
            os.fsync(dst_fd)
        except OSError:
            # Cleanup temp file on failure
            os.close(dst_fd)
            try:
                os.remove(temp_path)
            except OSError:
                pass  # Best-effort cleanup
            raise
        else:
            os.close(dst_fd)
    finally:
        os.close(src_fd)

    # Atomic rename (POSIX guarantees atomicity)
    os.replace(temp_path, final_path)

    # Best-effort fsync directory after rename
    _fsync_directory(final_path.parent)


def atomic_stream_to_file(
    stream,
    final_path: str | Path,
    temp_suffix: str = ".tmp",
    chunk_size: int = 65536,
) -> int:
    """Atomically write a stream to a file.

    Used for upload ingestion where data comes from a file-like object.
    Implements atomic publish semantics per Blueprint section 4.

    Args:
        stream: File-like object with read() method.
        final_path: Target path for the output file.
        temp_suffix: Suffix for the temporary file (default: ".tmp").
        chunk_size: Buffer size for reading (default: 64KB).

    Returns:
        Total bytes written.

    Raises:
        OSError: If write or rename fails.
    """
    final_path = Path(final_path)
    temp_path = final_path.with_suffix(final_path.suffix + temp_suffix)

    # Ensure parent directory exists
    final_path.parent.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")
            _write_all(fd, chunk)
            total_bytes += len(chunk)

        # Flush to OS buffers
        os.fsync(fd)
    except OSError:
        # Cleanup temp file on failure
        os.close(fd)
        try:
            os.remove(temp_path)
        except OSError:
            pass  # Best-effort cleanup
        raise
    else:
        os.close(fd)

    # Atomic rename
    os.replace(temp_path, final_path)

    # Best-effort fsync directory after rename
    _fsync_directory(final_path.parent)

    return total_bytes


def cleanup_orphan_temp_files(directory: str | Path, temp_suffix: str = ".tmp") -> int:
    """Clean up orphan temp files in a directory.

    Called during startup or error recovery to remove incomplete writes.

    Args:
        directory: Directory to scan for temp files.
        temp_suffix: Suffix pattern to match (default: ".tmp").

    Returns:
        Number of files removed.
    """
    directory = Path(directory)
    removed = 0

    if not directory.exists():
        return 0

    for temp_file in directory.glob(f"*{temp_suffix}"):
        try:
            temp_file.unlink()
            removed += 1
        except OSError:
            pass  # Best-effort cleanup

    return removed
