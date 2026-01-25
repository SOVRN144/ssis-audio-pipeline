from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest


@pytest.fixture
def restore_numpy_int():
    had_attr = hasattr(np, "int")
    original = getattr(np, "int", None)
    if had_attr:
        delattr(np, "int")
    try:
        yield had_attr, original
    finally:
        if had_attr and original is not None:
            np.int = original  # type: ignore[attr-defined]
        elif not had_attr and hasattr(np, "int"):
            delattr(np, "int")


def test_segments_module_restores_np_int(monkeypatch, restore_numpy_int):
    # Ensure the module is re-imported fresh
    sys.modules.pop("services.worker_segments.run", None)

    # Mock inaSpeechSegmenter import to avoid bringing in heavy deps
    dummy_module = type("Dummy", (), {})()
    monkeypatch.setitem(sys.modules, "inaSpeechSegmenter", dummy_module)

    importlib.import_module("services.worker_segments.run")
    assert hasattr(np, "int")
    assert np.int is int  # type: ignore[attr-defined]

    # Clean up to avoid bleeding into other tests
    sys.modules.pop("services.worker_segments.run", None)
