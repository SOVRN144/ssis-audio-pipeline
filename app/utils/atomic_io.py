"""SSIS Audio Pipeline - Atomic I/O utilities.

Implements atomic publish rule per Blueprint section 4:
1. Write to temp path in same directory
2. Flush + best-effort fsync
3. Rename temp -> final (the publish boundary)

This ensures that the final path either contains complete valid data
or does not exist. Partial writes only affect the temp file.
"""

import os
from pathlib import Path


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
        os.write(fd, data)
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

    # Atomic rename (POSIX guarantees atomicity)
    os.replace(temp_path, final_path)


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
