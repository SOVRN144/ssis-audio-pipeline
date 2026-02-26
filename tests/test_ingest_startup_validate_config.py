"""Tests for ingest API startup config validation hook."""

import importlib
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


def test_validate_config_called_once_on_startup():
    """validate_config should run exactly once during FastAPI lifespan startup."""
    import services.ingest_api.main as main_module

    importlib.reload(main_module)

    fake_engine = MagicMock()
    fake_session_factory = MagicMock()

    with patch.object(main_module, "validate_config") as mock_validate:
        with patch.object(main_module, "init_db", return_value=(fake_engine, fake_session_factory)):
            with patch.object(main_module, "_cleanup_orphan_temp_files_safe", return_value=None):
                with TestClient(main_module.app):
                    pass

    mock_validate.assert_called_once_with()
