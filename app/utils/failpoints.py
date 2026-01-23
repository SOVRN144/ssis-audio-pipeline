"""SSIS Audio Pipeline - Failpoint injection for resilience testing.

Provides deterministic crash injection for testing power-failure scenarios.
Used to verify atomic publish, checkpoint resumption, and lock reclamation.

Safety gate: Failpoints are only active when SSIS_ENABLE_FAILPOINTS=1.
This ensures no production impact - the default is a complete no-op.

Environment variables:
- SSIS_ENABLE_FAILPOINTS: Set to "1" to enable failpoint system (default: disabled)
- SSIS_FAILPOINT: Name of the failpoint to trigger (e.g., "ATOMIC_WRITE_AFTER_TMP_WRITE")
- SSIS_FAILPOINT_EXIT_CODE: Exit code to use when crashing (default: 42)
- SSIS_FAILPOINT_ONCE: Set to "1" to only trigger once, then clear (default: trigger every time)

Usage:
    from app.utils.failpoints import maybe_fail

    # In critical code paths:
    maybe_fail("ATOMIC_WRITE_AFTER_TMP_WRITE")

Failpoint naming convention:
    FAILPOINT_{COMPONENT}_{LOCATION}
    e.g., FAILPOINT_ATOMIC_WRITE_AFTER_TMP_WRITE, FAILPOINT_DECODE_AFTER_CHECKPOINT
"""

from __future__ import annotations

import os


def maybe_fail(point: str) -> None:
    """Check if a failpoint should trigger and crash if so.

    This function is designed to be a complete no-op when failpoints are
    disabled (the default). When enabled, it checks if the specified failpoint
    matches SSIS_FAILPOINT and triggers os._exit() to simulate a crash.

    Uses os._exit() instead of sys.exit() or exceptions because:
    - os._exit() bypasses Python's cleanup (atexit, finally blocks, etc.)
    - This simulates a real crash/power failure more accurately
    - Exceptions could be caught, defeating the purpose of crash testing

    Args:
        point: The failpoint name to check (e.g., "ATOMIC_WRITE_AFTER_TMP_WRITE").
               By convention, should not include "FAILPOINT_" prefix.
    """
    # Safety gate: no-op unless explicitly enabled
    if os.environ.get("SSIS_ENABLE_FAILPOINTS") != "1":
        return

    # Check if this specific failpoint should trigger
    target = os.environ.get("SSIS_FAILPOINT", "")
    if not target:
        return

    # Support both with and without FAILPOINT_ prefix for flexibility
    normalized_point = point.upper()
    normalized_target = target.upper()

    # Strip FAILPOINT_ prefix if present for comparison
    if normalized_point.startswith("FAILPOINT_"):
        normalized_point = normalized_point[len("FAILPOINT_") :]
    if normalized_target.startswith("FAILPOINT_"):
        normalized_target = normalized_target[len("FAILPOINT_") :]

    if normalized_point != normalized_target:
        return

    # Failpoint matches - prepare to crash

    # Get exit code (default 42 - a distinctive code for test verification)
    try:
        exit_code = int(os.environ.get("SSIS_FAILPOINT_EXIT_CODE", "42"))
    except ValueError:
        exit_code = 42

    # Check if this is a one-shot failpoint
    if os.environ.get("SSIS_FAILPOINT_ONCE") == "1":
        # Clear the failpoint so it only triggers once
        # Note: This only affects the current process. For subprocess testing,
        # the parent should not set SSIS_FAILPOINT_ONCE when restarting.
        os.environ.pop("SSIS_FAILPOINT", None)
        os.environ.pop("SSIS_FAILPOINT_ONCE", None)

    # Crash immediately - this bypasses all cleanup
    # This is intentional for simulating power failure / hard crash
    os._exit(exit_code)


def is_failpoint_enabled() -> bool:
    """Check if the failpoint system is enabled.

    Returns:
        True if SSIS_ENABLE_FAILPOINTS=1, False otherwise.
    """
    return os.environ.get("SSIS_ENABLE_FAILPOINTS") == "1"


def get_active_failpoint() -> str | None:
    """Get the currently active failpoint name, if any.

    Returns:
        The failpoint name (without FAILPOINT_ prefix) or None.
    """
    if not is_failpoint_enabled():
        return None
    target = os.environ.get("SSIS_FAILPOINT", "")
    if not target:
        return None
    # Normalize: strip prefix if present
    if target.upper().startswith("FAILPOINT_"):
        return target[len("FAILPOINT_") :].upper()
    return target.upper()
